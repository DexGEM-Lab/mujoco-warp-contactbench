# MuJoCo Warp ContactBench

Standalone MJX-Warp sim-to-Lance export repo.

This repo is intentionally decoupled from the larger `contactbench` workspace. The MANO hand asset and `lance_manager` are git submodules. The repo contains MuJoCo scene builders, MJX-Warp contact extraction, direct Lance export, and Docker/runtime helpers.

## Layout

```text
assets/mano_hand_s02/        submodule: MANO MJCF/URDF/STL assets used by simulation
3rd_party/lance_manager/     submodule: generated_data Lance writer/schema stack
sim/                         MJX-Warp simulation, scenarios, schema helpers, and export code
sim/benchmarks/ball_pit/     deterministic ball-pit scenario and camera helpers
sim/common/                  contact schema helpers and validators
scripts/                     build, sim-to-Lance, smoke test, and env helper scripts
outputs/                     generated outputs; ignored by git
Dockerfile                   Debian + Miniforge + CUDA + uv runtime image
```

## Clone / Submodules

After cloning this repo elsewhere, initialize submodules first:

```bash
git submodule update --init --recursive
```

Current submodules:

```text
assets/mano_hand_s02 -> git@192.168.10.116:ai/group-ai-public/group-sim-assets/mano_hand_s02.git
3rd_party/lance_manager -> git@192.168.10.116:ai/group-dexcanvas/lance_manager.git
```

## Pi Task Worktrees

From the primary worktree, create a feature task, linked worktree, and matching Pi session with:

```bash
scripts/start_pi_task.sh feat controller-sync
```

Use `feature` as an alias for `feat`, or `scripts/start_pi_task.sh case <context> <topic>` for case work. Add `--dry-run` to inspect without changing Git or `--no-launch` to create the branch and worktree without starting Pi. The enforced branch topology and naming rules are in [`.git-guard/contribution.md`](.git-guard/contribution.md).

After this feature is merged into `dev` and the primary worktree is switched to `dev`, run `.git-guard/enable.sh` there to enable hooks repository-locally. GitGuard is a local accidental-workflow guard, not a security boundary.

## Local uv Environment

```bash
scripts/setup_local_env.sh
JAX_PLATFORMS=cpu .venv/bin/python sim/smoke_test.py --strict --device cpu
```

The local uv environment is useful for CPU checks and development. GPU execution is supported through the Docker image by default.

## ManoRL reference replay (local, no Docker)

The first migration slice is isolated under `sim/manorl/`. It reads one settled
trajectory without `lance_manager`, builds a MuJoCo model from curated source
URDF/collision assets, and runs residual-off reference control. The current
ManoRL path implements observations, target rewards, native SKRL PPO training,
and native checkpoint round trips; conversion from Isaac rl-games checkpoints
remains deliberately unsupported.

Copying the PhysX drive values into an external MuJoCo torque law was falsified
in free space: the explicit damping kick drove the maximum DOF velocity to about
`1082 rad/s`. The replay therefore uses MuJoCo-native position actuators under
`implicitfast`, initially with wrist `kp=200`, `dampratio=1`; the five finger
blocks retain proportional gains `[6, 4, 3, 3]` with `dampratio=1`. The source
PhysX gains remain trace metadata but are not applied as MuJoCo torques. Hand
self-collision follows the source `disable_within_finger` mode: same-finger
collision pairs are excluded while palm/finger and cross-finger collisions
remain enabled.

```bash
conda activate manorl_mujoco
JAX_PLATFORMS=cpu python -m pytest -q tests/manorl
python -m sim.manorl.replay_reference \
  --backend mjx-warp \
  --device cpu \
  --wrist-kp 200 \
  --wrist-dampratio 1 \
  --output outputs/manorl/cube1_01_009_mjx_warp
```

A bounded free-space controller diagnostic disables only hand collision geoms;
object-floor contact remains active:

```bash
python -m sim.manorl.replay_reference \
  --backend mujoco-cpu \
  --max-steps 16 \
  --no-hand-contacts \
  --output outputs/manorl/native_servo_free_space_16
```

This diagnostic separates controller instability from contact geometry. It is
not a fallback data path and still requires the exact Lance dataset.

### View the environment test

From a desktop terminal with an X11/Wayland graphical session, open the
interactive MuJoCo viewer for the actual `MujocoManoEnvironment` test path. It
uses the fixed cube1 row `cube1_01_009`, source command schedule
`0,0,1,...,789`, and `residual_enabled=True` by default. The viewer supplies a
zero 26D action, so its controller target still follows the source mocap
reference. The MuJoCo right-side actuator pane shows the current `ctrl` target;
the terminal prints the complete command vector and source/reference indices.

```bash
conda activate manorl_mujoco
JAX_PLATFORMS=cpu python -m sim.manorl.view_environment \
  --device cpu --speed 0.25 --use_residual true --terminal true
```

When `--rerun-output` records the same environment run, its charts open on the
`step` timeline. Select `simulation` in Rerun's time panel to chart against
elapsed seconds. The `Collision geometry contact forces` tab records one
continuous world-frame net force series per compiled collision geom, including
inactive geoms and the floor; its static metadata table maps deterministic geom
IDs to labels, bodies, and mesh assets. The separate `ManoHand-object contact
forces` pane combines the 16 source-order hand-contact magnitude curves with an
`Object gravity` magnitude reference; each hand curve is the net world-frame
force exerted on the object by one mapped hand collision geom, with floor and
non-hand-object rows excluded. The all-geometry tab records cube gravity as
`body_subtreemass * gravity` in N. The stable `.rrd` changes only
after the delayed reset is applied; interrupting the run discards its active
partial episode.

Use `--device gpu` on a CUDA JAX environment and `--no-loop` to stop after the
single 791-call replay. Both runtime controls default to `true`: use
`--use_residual false` for source-reference diagnostics and `--terminal false`
for formal source-horizon termination only.

### ManoRL PPO Training

The Cube1 fast-training contract is documented in
[`docs/manorl_cube1_training_protocol.md`](docs/manorl_cube1_training_protocol.md).
PPO optimizes the raw environment reward at `1.0x`; it intentionally does not
reuse IsaacGym's `0.5x` reward shaper. Run the fixed 64-world budget with W&B
tracking disabled by default. This executes 64 updates of 48 rollout steps
(196,608 transitions); it has no wall-clock cutoff unless one is explicitly
requested:

```bash
JAX_PLATFORMS=cuda /home/jay/anaconda3/envs/manorl_mujoco/bin/python \
  -m tools.train_manorl_cube1 \
  --output outputs/manorl/cube1_03_scratch_run \
  --object cube1 --gesture 03 --num-envs 64 --updates 64
```

For an opt-in safety cap that may stop before all 64 updates complete, add
`--wall-clock-seconds <positive-seconds>`.

To create one W&B run using the authenticated default account, provision the
locked SDK in the documented training interpreter, then add explicit tracking
options. No API key or credentials belong in this repository:

```bash
/home/jay/anaconda3/envs/manorl_mujoco/bin/uv pip install \
  --python /home/jay/anaconda3/envs/manorl_mujoco/bin/python \
  "wandb==0.28.0"
```

```bash
  --wandb true --wandb-project one_policy --wandb-group s02 \
  --wandb-tags manorl,mujoco,skrl
```

An omitted W&B name is derived from the output prefix, object, and gesture. The
SDK cache is stored at `<output-parent>/wandb`, so the documented
`outputs/manorl/...` prefixes keep it under ignored outputs. The run logs PPO
updates by environment transitions, records zero/untrained/trained evaluation
summaries and final acceptance values, then uploads the checkpoint and sidecar,
metrics JSON, evaluation trace, and any completed Rerun recording.

To inspect the explicitly selected generated cube1 Lance row 507 under current
training termination semantics, use the same production environment with its
separate versioned selector:

```bash
JAX_PLATFORMS=cpu python -m sim.manorl.view_environment \
  --device cpu \
  --trajectory generated-cube1-row-507 \
  --terminal true \
  --loop \
  --speed 0.25
```

This selector binds generated Lance version 236, row 507, UUID
`00f45dd5-6699-5be1-8948-d6f7b623da48`, and source window `[17,735)`. It is a
generated cube1 test trajectory, not a claim that the dataset identifies source
gesture `01`.

To run ten independent accepted `cube1` gesture-`01` trajectories in parallel
and view all ten in one MuJoCo-rendered window, use the pinned batch contract:

```bash
JAX_PLATFORMS=cpu python -m sim.manorl.view_environment \
  --device cpu \
  --trajectory accepted-cube1-action-01-batch10 \
  --num-envs 10 \
  --tile-envs 10 \
  --terminal true \
  --loop \
  --speed 0.25
```

The batch contains ten distinct fully padded action-`01` trajectories. They
retain independent reference progress and delayed resets while sharing one
compiled cube1 model and batched MJX-Warp physics. `--tile-envs 1` preserves
the native actuator-pane viewer; larger values render the first N batch worlds
as tiles in one GLFW/MuJoCo window. In tiled mode, drag with the left mouse button to rotate, right mouse button to pan horizontally, middle mouse button to pan vertically, use the scroll wheel to zoom, press `R` to reset the view, and press `Esc` to close the window.

The accepted input is row 1 of:

```text
/mnt/nas-222-project/mocap_v2/lance_datasets/human_p1_remake/npy_s02_v3.lance
```

It must resolve to version `132`, UUID `d5bc2bc6-9458-52d0-bccc-66c9ec21bae3`,
file UUID `e6fe4732-72cd-5ab7-93e6-2e62dc0263a5`, identity `cube1_01_009`, and
source slice `[440,1232)`. The loader calls `dataset.take([1], columns=...)`;
there is no broad scan or NPY fallback. The `pylance` package imports as `lance`.

The source counter schedule is intentionally preserved from
`IsaacGymEnvs/isaacgymenvs/tasks/mano_hand.py::pre_physics_step`: 791 physics
calls command slice indices `0, 0, 1, ..., 789` and post-step comparisons use
`0, 1, ..., 790`. Slice index 791 is never consumed because the source checks
termination after progress reaches `L-1`. Each call executes two 0.0025 s
physics substeps; capture timestamps are validated for ordering but do not drive
simulation time.

The trace is a compressed `.npz`; its adjacent `.json` records the exact
configuration, trajectory identity, hand/object errors, raw Lance and
MuJoCo-support-shifted object references, actual actuator forces,
velocity and effort-saturation diagnostics, engine warnings, hand-object
penetration when observable, and contact-capacity flags. These are MuJoCo replay
errors. No Isaac parity claim is made without an Isaac trace.

The existing `3rd_party/lance_manager` pin may be broken or unavailable and is
irrelevant to this input-only replay: this slice depends only on the public
`pylance` reader. Curated files under `sim/manorl/runtime_assets/` come from
`all_assets` commit `ead79126589d1abf2362ea30b9d674d9e675a2f9`; the manifest
records provenance and SHA256 digests. The runtime uses the source cube URDF
and the geometrically equivalent compact `cube1_aligned.stl` collision mesh.

## Build Docker Image

```bash
scripts/build_docker.sh
```

This creates:

```text
mujoco-warp-contactbench:latest
```

The image does not copy repo source code. Runtime scripts mount the workspace at `/workspace/mujoco-warp-contactbench`, so code changes do not require image rebuilds unless dependencies or the Dockerfile change.

## Docker X11 Viewer

`docker-compose.yml` forwards the host X11 socket and `${HOME}/.Xauthority` into the container so GUI tools such as `mujoco.viewer` can open on the host display. From an active X11 desktop/session:

```bash
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose --profile gpu run --rm contactbench-gpu bash
```

Inside the container, run GUI tools directly, for example:

```bash
xclock
python -m mujoco.viewer path/to/model.xml
```

The compose service uses host networking so SSH X11 forwarding values such as `DISPLAY=localhost:16.0` continue to work inside the container.

## Sim-to-Lance Export

Run MJX-Warp simulation and write Lance directly, without contact JSON intermediates. GPU is the default device:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_sim_lance.sh
```

Use CPU execution for the same MJX-Warp backend with:

```bash
DEVICE=cpu scripts/run_sim_lance.sh
```

Default output:

```text
outputs/mjx_warp_contactbench_generated.lance
```

Extra simulation arguments are forwarded to `sim/sim_to_lance.py`, for example:

```bash
DEVICE=cpu OUTPUT=outputs/test.lance scripts/run_sim_lance.sh --duration-seconds 1 --ball-count 16
```

The exporter uses `3rd_party/lance_manager/schema/schemas/generated_data_schema.jsonc` and writes with the submodule's Lance writer.

## Smoke Test

```bash
scripts/run_smoke_test.sh
```

This validates MuJoCo, JAX/MJX availability for the requested device, and a short MJX-Warp sim-to-Lance export. Set `DEVICE=cpu` to smoke test CPU execution.

## Notes

- MJX-Warp contact points are raw `_impl.contact__pos`, not projected onto the ball or hand mesh.
- MJX-Warp internal `_impl` fields are useful but private/unstable compared with stable MuJoCo CPU APIs.
- `outputs/` is intentionally ignored and contains local generated artifacts.
