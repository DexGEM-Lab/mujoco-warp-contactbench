# MuJoCo Warp ContactBench

Standalone MJX-Warp sim-to-Lance export repo.

This repo is intentionally decoupled from the larger `contactbench` workspace. The
MANO hand asset, complete source asset collection, and `lance_manager` are git
submodules. The repo contains MuJoCo scene builders, MJX-Warp contact extraction,
direct Lance export, and Docker/runtime helpers.

## Layout

```text
assets/mano_hand_s02/        submodule: MANO MJCF/URDF/STL assets used by simulation
assets/all_assets/           submodule: complete upstream Gym-version asset collection
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
assets/all_assets -> git@192.168.10.116:jieqiangsun/all_assets.git @ e7910212e54367008ecb7484e5e9354e822de03e (fixed pin)
assets/all_assets/Assets/sim/mano_assets -> git@192.168.10.116:ai/group-dexcanvas/mano_assets.git @ 31596655f25281b0f4e20c47bf20ef0b19ff8f4a (nested fixed pin)
3rd_party/lance_manager -> git@192.168.10.116:ai/group-dexcanvas/lance_manager.git
```

The `assets/mano_hand_s02` and `assets/all_assets` submodules track the
upstream `main` branch for explicit remote updates. The commits shown above
remain the root repository's gitlink pins; changing an upstream branch does
not change this checkout until the updated gitlink is committed here. To
intentionally advance a tracked asset, run for example:

```bash
git submodule update --remote assets/all_assets
git add .gitmodules assets/all_assets
git commit -m "Update all_assets submodule"
```

`assets/all_assets` is the authoritative source checkout for ManoRL object
URDFs, CoACD collision pieces, grasp mappings, and visual meshes. Its nested
`Assets/sim/mano_assets` pin supplies the object visual/source meshes. Curated
files under `sim/manorl/runtime_assets/` remain compatibility inputs for the
original hand/cube path. When a local checkout of `all_assets` already has the
pinned objects, it can be used as a Git reference to avoid downloading the
object database again:

```bash
git submodule update --init \
  --reference /path/to/existing/all_assets \
  assets/all_assets
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
and native checkpoint round trips. Raw Isaac rl-games checkpoints are still
rejected by the native loader and are no longer convertible. Training, resume,
and visualization accept only current native 28-DoF-per-hand MuJoCo
checkpoints with their sidecars.

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
zero 28D action to the current MuJoCo hand, so its controller target still
follows the source mocap reference. The MuJoCo right-side actuator pane shows
the current `ctrl` target;
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
`body_subtreemass * gravity` in N. The stable `.rrd` changes immediately after
the terminal snapshot is recorded; interrupting the run discards its active
partial episode.

Use `--device gpu` on a CUDA JAX environment and `--no-loop` to stop after the
single 791-call replay. Both runtime controls default to `true`: use
`--use_residual false` for source-reference diagnostics and `--terminal false`
for formal source-horizon termination only.

To render a deterministic mean policy, keep its `.pt.json` sidecar beside the
checkpoint. The sidecar records the checkpoint's versioned runtime and resolved
hand/action layout; model key/shape compatibility and current reward, PPO, and
environment contracts are checked when the checkpoint loads.

```bash
JAX_PLATFORMS=cuda python -m sim.manorl.view_environment \
  --device gpu \
  --checkpoint outputs/manorl/banana_right_28dof.checkpoint.pt \
  --object cube1 --gesture 01 --num-envs 1 --render-env 0 --no-loop
```

### ManoRL PPO Training

The Cube1 production training contract is documented in
[`docs/manorl_cube1_training_protocol.md`](docs/manorl_cube1_training_protocol.md).
PPO uses the source-aligned `0.5x` reward shaper over the raw environment
reward. Observation contact direction and the `1.0x` pair-filtered contact
reward use the same strict `0.2 N` threshold; maximum contact quality remains
`0.4`. W&B tracking is
enabled by default; pass `--wandb false` for a local-only diagnostic. The
default run uses 2,048 worlds, 8,000 updates, 48 rollout steps, a 4,096-sample
minibatch, FiLM, dynamic point-cloud sampling, residual actions, terminal
deviation handling, and deterministic evaluation that covers every selected
object/action pair. A single-pair run uses one evaluation world; a multi-pair
run automatically uses at least one world per pair, bounded at 128. This is
786,432,000 transitions; it has no wall-clock cutoff unless one is explicitly
requested:

```bash
JAX_PLATFORMS=cuda /home/jay/anaconda3/envs/manorl_mujoco/bin/python \
  -m tools.train_manorl_cube1 \
  --output outputs/manorl/cube1_01_default \
  --object cube1 --gesture 01 --num-envs 2048 --updates 8000 \
  --checkpoint-interval-updates 200 --evaluation-num-envs 1
```

The two stable repository entrypoints cover routine training and checkpoint viewing without rewriting launch scripts:

```bash
# Train every eligible gesture for one object on physical GPU 0.
./train.sh cube1 2048 0

# Render 20 cube1/action-01 trajectories from one checkpoint on physical GPU 0.
CHECKPOINT=outputs/manorl/<run>/training/checkpoint-000900.pt \
  ./inference.sh cube1 01 20 0

# Render the fixed no-checkpoint reference test: cube1/action-01, N20,
# residual actions disabled, on physical GPU 0.
./test.sh 0
```

`train.sh` arguments are `object`, `num_envs`, and `physical_gpu`; use object `all` for all eligible object/action pairs. `inference.sh` arguments are `object`, `gesture`, `render_count`, and `physical_gpu`, with the checkpoint supplied through `CHECKPOINT` or `MANORL_CHECKPOINT`. `test.sh` has a fixed cube1/action-01, N20, no-checkpoint contract with residual actions disabled; its optional argument selects the physical GPU. All three scripts generate their remaining runtime contract from stable defaults. Dataset, update count, W&B, device, and playback overrides remain available through `MANORL_*` environment variables documented in each script.

The trainer also accepts exact multi-object/action selection. Use
`--pairs cube1:01,cube1:02,cube2:01` for only those pairs, or `--all-pairs` for
every eligible pair in the pinned Lance dataset. Mixed-object batches run
headless through one static MJX-Warp model per object; GUI and Rerun recording
remain single-object modes.

### Corrected v2 checkpoint rollout synthesis

`./synthesize.sh` runs a deterministic checkpoint mean policy on GPU and writes
one independent complete source-length trajectory per assigned environment:

```bash
CHECKPOINT=outputs/manorl/<run>/training/checkpoint-000500.pt \\
  ./synthesize.sh cube2 02 5 0
```

The output is a nested Lance dataset plus a sibling `.manifest.json`. The v2
contract is `synthetic_mano_28d_checkpoint_rollout_v2`: timestamps use the
actual `0.005 s` control interval (`data_fps=200`), `force_normal` contains the
solved normal component with scale `1.0`, and all force frames use a consistent
hand-to-object direction. `pos_joint` and `total_force_joint` use the live
collision-link transform rather than the historical wrist fallback. Each row
also stores 28D physical and controller targets, 21 keypoints, reference frame
indices, policy mean/processed actions, observations, rewards, termination
codes, checkpoint SHA256, runtime sidecar, action contract, and source identity.

`--num-envs` controls the simultaneous identities; set it to the number of
eligible `cube2:02` rows to export the whole pair, or use a smaller value and
advance `--pair-assignment-cycle` across bounded batches. The exporter refuses
to overwrite an existing dataset unless `--replace` is passed to the Python
CLI. It requires a native checkpoint sidecar and preserves the checkpoint's
serialized residual-action/CCD settings. On hosts where nested Lance row reads
are natively unstable, set `MANORL_PREDECODED_MANIFEST` to the validated
isolated-predecode manifest; every selected pickle is SHA256-checked before the
GPU rollout. The exporter seeds NumPy and Torch/CUDA with `42` by default so
the stochastic point-cloud observation source is fixed; override it with
`MANORL_SYNTH_SEED` or `--seed`. The seed is recorded in every row. MJX-Warp
GPU contact reductions can still vary at floating-point scale, and closed-loop
rollouts can amplify that numerical variation, so the contract does not claim
bitwise replay across separate processes.

Finger residual increments and their cumulative caps have independent explicit
multipliers, both defaulting to `2.0`. Wrist XYZ residuals default to a
`0.003 m` per-step scale and a symmetric `0.03 m` cumulative cap. Override them
with `--joint-scale-multiplier`, `--joint-max-offset-multiplier`,
`--position-scale`, and `--max-position-offset`. The values are recorded in
W&B, Rerun, and native checkpoint metadata; resume rejects a checkpoint whose
residual-action contract differs from the target runtime. The base
`thumbCmcTwist` scale/cap are `0.008`/`0.08`; each non-thumb
`fingerMcpFlex` scale/cap is `0.0025`/`0.025`, before the global `2.0`
multipliers. V4/v5 sidecars remain readable and restore their explicit
residual-action contracts.

For an opt-in safety cap that may stop before all 8,000 updates complete, add
`--wall-clock-seconds <positive-seconds>`. `--evaluation-num-envs` requests a
minimum diagnostic count within `1..min(--num-envs, 128)`. The trainer raises
that count when necessary to cover every resolved object/action pair once, so
evaluation cannot silently report only the first pair or recreate a second
full-size runtime.

Every 200 completed updates, the default cadence writes
`<output>/checkpoint-000200.pt` plus its `.pt.json` sidecar. Override the cadence
with `--checkpoint-interval-updates <count>`. `<output>/last.pt` atomically
follows the latest completed checkpoint; its fixed sidecar records compatibility
only. Exact progress remains in the immutable numbered and final checkpoint
sidecars. Sibling output prefixes have independent checkpoint namespaces.

Training stdout defaults to compact human summaries. Exact completed episode
returns are first flushed to `<output>.episodes.jsonl.partial`, then summarized
as count/mean/min/max; successful completion atomically publishes
`<output>.episodes.jsonl`. Use `--console-format json` for the original JSON
stdout event stream and final JSON result. Existing or active captured logs can
be summarized without restarting training:

```bash
/home/jay/anaconda3/envs/manorl_mujoco/bin/python \
  tools/watch_manorl_training.py outputs/manorl/server2.train.log --follow
```

For a local CUDA desktop session, the 8-world visual wrapper opens a tiled
MuJoCo view of the actual PPO rollout worlds. It uses the target interpreter,
8 training/evaluation worlds, one 384-sample minibatch, W&B off, a unique output
prefix, and accepts extra trainer arguments after its defaults:

```bash
JAX_PLATFORMS=cuda scripts/train_manorl_cube1_visual.sh --updates 64
```

`--headless true` remains the default for the trainer. With `--headless false`,
`--viewer-envs` must be within `--num-envs` and `--viewer-stride` is positive.
The viewer mirrors post-step states only; it never selects actions or advances
physics. It uses the same tiled controls as the standalone viewer: left-drag
rotate, right-drag horizontal pan, middle-drag vertical pan, wheel zoom, and R
reset. Close its window (or press Esc) to finish the active complete PPO
rollout/update, then publish the normal checkpoint, evaluation, and artifacts.
The graphical path requires `DISPLAY` or `WAYLAND_DISPLAY`; it still requires
CUDA Torch and MJX-Warp, and has no CPU fallback.

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
updates by environment transitions. One shared policy produces one W&B run;
global metrics remain at their existing keys, while Gym-aligned object keys
such as `reward_mean/object_cube1` and object/action keys such as
`reward_mean/cube1_01` keep pair behavior separate. Episode, success, reward
component, and zero/untrained/trained evaluation metrics use the same hierarchy.
The run then uploads the checkpoint and sidecar, metrics JSON, evaluation trace,
episode JSONL, and any completed Rerun recording.

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
retain independent reference progress and indexed episode resets while sharing one
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
`pylance` reader. Authoritative multi-object runtimes use `all_assets` commit
`e7910212e54367008ecb7484e5e9354e822de03e` with nested `mano_assets` commit
`31596655f25281b0f4e20c47bf20ef0b19ff8f4a`; runtime entries pin every required
URDF and CoACD piece by SHA256. The original curated hand/cube compatibility
manifest retains its own source provenance and digests.

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
