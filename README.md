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
assets/all_assets -> git@192.168.10.116:jieqiangsun/all_assets.git @ 7228b5cfce8d9a072ed4bded7a489cf73d521b68 (fixed pin)
assets/all_assets/Assets/sim/mano_assets -> git@192.168.10.116:ai/group-dexcanvas/mano_assets.git @ cde03ef94816b589f574ca6f358695005d3d1a3f (nested fixed pin)
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

The primary worktree coordinates two protected product lines: `dev` for ManoRL
and `dexhand` for DexHandRL. Neither line accepts direct commits. Create tasks
from the primary worktree with the matching product prefix:

```bash
# ManoRL task from dev
scripts/start_pi_task.sh feat controller-sync
scripts/start_pi_task.sh case cube1 contact-tuning

# DexHandRL task from dexhand
scripts/start_pi_task.sh dexfeat controller-sync
scripts/start_pi_task.sh dexcase cube1 contact-tuning
```

`feature` aliases `feat`, and `dexfeature` aliases `dexfeat`. Add `--dry-run` to
inspect without changing Git or `--no-launch` to create the branch and worktree
without starting Pi. The enforced branch topology and naming rules are in
[`.git-guard/contribution.md`](.git-guard/contribution.md).

After the workflow is installed in the primary worktree, run
`.git-guard/enable.sh` there to enable hooks repository-locally. GitGuard is a
local accidental-workflow guard, not a security boundary.

## Documentation Map

| Topic | Document |
|---|---|
| Cube1 training protocol (budget, launch, W&B) | [`docs/manorl_cube1_training_protocol.md`](docs/manorl_cube1_training_protocol.md) |
| Synthetic Lance production (approach prefix, offline retreat, save gate) | [`docs/manorl_synthesis_approach_prefix.md`](docs/manorl_synthesis_approach_prefix.md) |
| Lance isolation runbook (Source → Compile → Run, package, publish) | [`docs/manorl_lance_isolation_runbook.md`](docs/manorl_lance_isolation_runbook.md) |
| Phase 5A ABI inventory (observation/residual/reward/checkpoint) | [`docs/manorl_phase5_abi_inventory.md`](docs/manorl_phase5_abi_inventory.md) |

## Local uv Environment

```bash
scripts/setup_local_env.sh
JAX_PLATFORMS=cpu .venv/bin/python sim/smoke_test.py --strict --device cpu
```

The local uv environment is useful for CPU checks and development. GPU execution is supported through the Docker image by default.

## ManoRL reference replay (local, no Docker)

The ManoRL path under `sim/manorl/` reads settled trajectories without
`lance_manager`, builds a MuJoCo model from curated source URDF/collision
assets, and runs residual-off reference control. It implements observations,
target rewards, native SKRL PPO training, and native checkpoint round trips.
Only current native 28-DoF-per-hand MuJoCo checkpoints with their sidecars are
accepted for training, resume, and visualization.

The replay uses MuJoCo-native position actuators under `implicitfast`,
initially with wrist `kp=200`, `dampratio=1`; the five finger blocks retain
proportional gains `[6, 4, 3, 3]` with `dampratio=1`. Copying the source PhysX
drive values into an external MuJoCo torque law was falsified in free space
(explicit damping kick drove DOF velocity to ~1082 rad/s); the source PhysX
gains remain trace metadata only. Hand self-collision follows the source
`disable_within_finger` mode.

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
`0,0,1,...,789`, and `residual_enabled=True` by default.

```bash
conda activate manorl_mujoco
JAX_PLATFORMS=cpu python -m sim.manorl.view_environment \
  --device cpu --speed 0.25 --use_residual true --terminal true
```

Use `--device gpu` on a CUDA JAX environment and `--no-loop` to stop after the
single 791-call replay. Both runtime controls default to `true`: use
`--use_residual false` for source-reference diagnostics and `--terminal false`
for formal source-horizon termination only. When `--rerun-output` records the
same environment run, its charts open on the `step` timeline; select
`simulation` in Rerun's time panel to chart against elapsed seconds. The
`Collision geometry contact forces` tab records one continuous world-frame net
force series per compiled collision geom (including inactive geoms and the
floor); `ManoHand-object contact forces` combines the 16 source-order
hand-contact magnitude curves with an `Object gravity` magnitude reference.

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

### Direct Lance target-DOF replay

`tools/replay_manorl_target_dof.py` reads one explicit synthetic Lance
`dataset/version/row` and replays its recorded post-`command_target` 28D
controller targets. It requests only `index`, `trajectory_metadata`,
`timestamp`, `hands`, `objects`, and `provenance`; it does not load a policy
checkpoint or run training. The decoder keeps the generated row identity and
the original source lineage in separate report fields.

Run a bounded headless replay on a healthy Lance host:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH=. python tools/replay_manorl_target_dof.py \
  --dataset /path/to/synthetic.lance --dataset-version 1 --row-index 2239 \
  --device gpu --headless --output outputs/manorl/replay/report.json
```

For one local X11 display, export its display explicitly before opening the
MuJoCo viewer:

```bash
export DISPLAY=:1
export XAUTHORITY=/run/user/1000/gdm/Xauthority
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH=. python tools/replay_manorl_target_dof.py \
  --dataset /path/to/synthetic.lance --dataset-version 1 --row-index 2239 \
  --device gpu --speed 0.25 --loop
```

The viewer mirrors MJX-Warp state into a native visual model; it does not run a
second native simulation.

**Legacy fixed selectors** (debug/test assets, not dataset claims):
`generated-cube1-row-507` binds generated Lance version 236, row 507, UUID
`00f45dd5-6699-5be1-8948-d6f7b623da48`, source window `[17,735)`.
`accepted-cube1-action-01-batch10` runs ten independent accepted cube1/action-01
trajectories in one tiled window (row 1 of
`human_p1_remake/npy_s02_v3.lance` version 132, UUID
`d5bc2bc6-9458-52d0-bccc-66c9ec21bae3`, identity `cube1_01_009`, source slice
`[440,1232)`); in tiled mode left-drag rotates, right-drag pans horizontal,
middle-drag pans vertical, wheel zooms, `R` resets, `Esc` closes. The source
counter schedule is preserved from
`IsaacGymEnvs/isaacgymenvs/tasks/mano_hand.py::pre_physics_step`: 791 physics
calls command slice indices `0, 0, 1, ..., 789`; slice index 791 is never
consumed. The original curated hand/cube compatibility manifest retains its own
source provenance and digests. A row containing explicit Warp CCD scratch settings
requires `--device gpu`; `--device cpu --allow-physics-override` is only an
explicitly non-identical diagnostic. Direct Lance/PyArrow replay must not be
run on the currently disqualified Server2 until its physical
DIMM/channel/CPU-IMC isolation is complete.

## ManoRL PPO Training

The Cube1 production training contract (budget, launch, W&B, evaluation,
visual test, artifacts) is documented in
[`docs/manorl_cube1_training_protocol.md`](docs/manorl_cube1_training_protocol.md).
Key facts:

- PPO uses the source-aligned `0.5x` reward shaper; observation contact
  direction and the `1.0x` pair-filtered contact reward use the same strict
  `0.2 N` threshold; maximum contact quality is `0.4`.
- Default run: 2,048 worlds, 8,000 updates, 48 rollout steps, 4,096-sample
  minibatch, FiLM, dynamic point-cloud sampling, residual actions, terminal
  deviation handling. Default 120 Hz; `--reference-fps {100,120}` selects a
  coupled source/reference and policy/control clock (100 Hz -> 400 Hz physics,
  120 Hz -> 480 Hz physics, always exactly four equal substeps per inference).
- New selections use exactly 180 source/policy steps of pre-padding and 250
  post-padding steps. Hand angular coordinates are unwrapped before
  interpolation, object position is linearly interpolated, object orientation
  uses quaternion SLERP.
- Production training uses a **Source → Compile → Run** boundary: Lance is the
  versioned archival source opened only by short-lived compiler workers; the
  long-lived MuJoCo/MJX/PPO process consumes the content-addressed
  `manorl.trajectory_package.v1` without importing Lance/PyArrow. See
  [`docs/manorl_lance_isolation_runbook.md`](docs/manorl_lance_isolation_runbook.md).

```bash
# Train every eligible gesture at the default coupled 120 Hz source/policy clock.
./train.sh cube1 2048 0

# Select coupled 100 Hz source/policy control with 400 Hz physics.
MANORL_REFERENCE_FPS=100 ./train.sh cube1 2048 0

# Render 20 cube1/action-01 trajectories.
CHECKPOINT=outputs/manorl/<run>/training/checkpoint-000900.pt \
  ./inference.sh cube1 01 20 0

# Fixed no-checkpoint reference test: cube1/action-01, N20, no residual,
# physical GPU 0. Padding defaults 180/250; optional overrides follow the GPU.
./test.sh 0
./test.sh 0 73 91
```

`train.sh` arguments are `object`, `num_envs`, and `physical_gpu`; use object
`all` for all eligible object/action pairs. `MANORL_REFERENCE_FPS=100|120`
selects the coupled source/policy clock, defaulting to 120 for new training.
`MANORL_WARM_START_CHECKPOINT` and `MANORL_WARM_START_PRIOR_UPDATES` must be
supplied together to transfer policy/value/normalizers while resetting
optimizer, scheduler, memory, and run progress. `inference.sh` arguments are
`object`, `gesture`, `render_count`, and `physical_gpu`, with the checkpoint
supplied through `CHECKPOINT` or `MANORL_CHECKPOINT`. `test.sh` has a fixed
cube1/action-01, N20, no-checkpoint contract with residual actions disabled;
its positional arguments are `physical_gpu`, `pre_padding`, and `post_padding`.
`MANORL_PRE_PADDING` and `MANORL_POST_PADDING` provide equivalent environment
overrides. Dataset, update count, W&B, device, and playback overrides remain
available through `MANORL_*` environment variables documented in each script.

The trainer also accepts exact multi-object/action selection via
`--pairs cube1:01,cube1:02,cube2:01` or `--all-pairs`. Mixed-object batches run
headless through one static MJX-Warp model per object; GUI and Rerun recording
remain single-object modes. Strict resume binds the package schema, package
manifest SHA256, and catalog digest in the checkpoint environment ABI; moving a
verified package does not change its identity, changing any catalog byte does.

### Training output and monitoring

Every 200 completed updates, the default cadence writes
`<output>/checkpoint-000200.pt` plus its `.pt.json` sidecar. Override the
cadence with `--checkpoint-interval-updates <count>`. `<output>/last.pt`
atomically follows the latest completed checkpoint; its fixed sidecar records
compatibility only. Exact progress remains in the immutable numbered and final
checkpoint sidecars. Sibling output prefixes have independent checkpoint
namespaces.

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
MuJoCo view of the actual PPO rollout worlds:

```bash
JAX_PLATFORMS=cuda scripts/train_manorl_cube1_visual.sh --updates 64
```

`--headless true` remains the default for the trainer. With `--headless false`,
`--viewer-envs` must be within `--num-envs` and `--viewer-stride` is positive.
The viewer mirrors post-step states only; it never selects actions or advances
physics. The graphical path requires `DISPLAY` or `WAYLAND_DISPLAY`; it still
requires CUDA Torch and MJX-Warp, and has no CPU fallback.

For an opt-in safety cap that may stop before all 8,000 updates complete, add
`--wall-clock-seconds <positive-seconds>`. `--evaluation-num-envs` requests a
minimum diagnostic count within `1..min(--num-envs, 128)`; the trainer raises
that count when necessary to cover every resolved object/action pair once.
Residual-action defaults: wrist XYZ `0.003 m` per-step scale and `0.03 m`
cumulative cap; base `thumbCmcTwist` scale/cap `0.008`/`0.08`, each non-thumb
`fingerMcpFlex` scale/cap `0.0025`/`0.025`, before the global `2.0`
`--joint-scale-multiplier` / `--joint-max-offset-multiplier`. Resume rejects a
checkpoint whose residual-action contract differs from the target runtime.

W&B tracking is enabled by default; pass `--wandb false` for a local-only
diagnostic. An omitted W&B name is derived from the output prefix, object, and
gesture. The run logs PPO updates by environment transitions; episode,
success, reward component, and zero/untrained/trained evaluation metrics use
the same hierarchy, with object keys such as `reward_mean/object_cube1` and
object/action keys such as `reward_mean/cube1_01`. The run uploads the
checkpoint and sidecar, metrics JSON, evaluation trace, episode JSONL, and any
completed Rerun recording. Full W&B section:
[`docs/manorl_cube1_training_protocol.md`](docs/manorl_cube1_training_protocol.md).

## Synthetic Lance synthesis

The default synthesis operation and the save gate are documented in
[`docs/manorl_synthesis_approach_prefix.md`](docs/manorl_synthesis_approach_prefix.md).

### Two-stage production (standard since 2026-08-27)

Every published synthesis dataset is a two-stage product:

```text
stage 1 (simulation, MJX + PPO):
  canonical pre60 reference
    -> prepend a seeded 4 cm Far/Near approach prefix (policy-free)
    -> checkpoint policy runs the complete original reference
    -> three-rule save gate: reason 1 + final XYZ mean <=35 deg
       + >=101 right-hand/object contact frames >0.2 N (plus prefix gate)
    -> compact synthetic_mano_target_replay_visual_v2_contact rows
       with real solver contact

stage 2 (offline, kinematic, no solver):
  anchor = true last solved hand-object contact frame + 15
  retreat = replace the entire tail after the anchor (155-251 frames,
    median 209) in right-wrist XYZ only, via smoothstep tail deformation
    (direction = anchor -> original-final-wrist + extra 3-15 cm, +/-30 deg,
    Z +4-10 cm); length, timestamps, fingers, object, contact and reference
    preserved from the base row
  tool: tools/build_manorl_full_retreat.py (deterministic)
  -> published dataset carries the _with_retreat suffix
```

Rationale: stage 1 owns "how to go" (approach/grasp/place dynamics and real
solver contact); stage 2 owns "how to return" (kinematic retreat), and can use
the persisted contact array to know the true contact-end frame, which runtime
synthesis cannot. Do not reintroduce in-simulation retreat and do not use the
short 29-frame retreat variant.

Example published dataset:
`for_vla_manorl_prefix_near_far_4pairs_20260826_with_retreat.lance` (1035 rows,
all with full-length retreat; sibling manifest/validation/NAS_PUBLICATION
sidecars with SHA256).

### Compact output

`./synthesize.sh` defaults to the compact replay/visual contract
`synthetic_mano_target_replay_visual_v2_contact`. It retains target/recorded
DOF, object poses, MANO global pose, 48D hand pose, 21-joint visual frames,
contact/reference/command mapping, and minimal lineage. Full checkpoint runtime
metadata is recorded once in the sibling manifest or catalog; each row keeps
its canonical metadata SHA256, checkpoint identity, and explicit Warp CCD
settings. Compact rows carry the source v2.2/v2.3 clock contract and are
directly consumable by target replay.

```bash
CHECKPOINT=outputs/manorl/<run>/training/checkpoint-000500.pt \
  ./synthesize.sh cube2 02 5 0

# Explicit full/audit output with contact, reference, rollout, and observations:
MANORL_SYNTH_OUTPUT_FORMAT=full \
CHECKPOINT=outputs/manorl/<run>/training/checkpoint-000500.pt \
  ./synthesize.sh cube2 02 5 0
```

To augment one row that already succeeded in a scalable synthetic publication,
first create an accepted-parent descriptor, then synthesize with one active env:

```bash
python tools/select_manorl_synthetic_parent.py \
  --input /path/prior-successes.lance \
  --row-uuid <accepted-row-uuid> \
  --output /path/accepted-parent.json

MANORL_SYNTH_APPROACH_MODE=near \
MANORL_SYNTH_ACCEPTED_PARENT=/path/accepted-parent.json \
MANORL_PREDECODED_MANIFEST=/path/pre60-bundle/manifest.json \
CHECKPOINT=/path/exact-parent-checkpoint.pt \
./synthesize.sh banana 01 1 0
```

`far` samples XY 0.30–1.00 m, world-up Z 0.08–0.30 m, and ±30° azimuth around
the initial object→pre60 hand direction. `near` uses an independent seed and
maps a movement-end+15 retreat-like endpoint to the initial object. That anchor
only chooses the Near start; it never modifies the tail. Start XYZ is sampled;
start `q_ref[3:28]` is copied from raw source frame 0, then wrist orientation and
finger joints smoothly reach pre60 frame 0 along the established 4 cm arc.

For a new pinned package that intentionally differs from the checkpoint
trajectory-package signature, base-parent bootstrap may explicitly pass
`--policy-transfer` to `tools/export_manorl_synthetic_lance.py`. The manifest
records `checkpoint_loading=policy_transfer`. This boundary still validates the
checkpoint reward/environment family, model architecture, tensor finiteness, and
state-dict shapes; ordinary inference remains strict by default.

The prefix never calls policy and rejects solved right-hand/table or
right-hand/object contact above 0.2 N. After the prefix, checkpoint policy runs
the complete original reference tail. Default production has no retreat suffix
at simulation time (retreat is added offline in stage 2). Saving additionally
requires reason code 1, final XYZ Euler mean error ≤35°, and at least 101
right-hand/object contact frames strictly above 0.2 N. Reward is not an
acceptance rule; a candidate failing any rule is not written.

Inspect episodes interactively with:

```bash
python tools/view_manorl_approach_prefix.py \
  --checkpoint /path/exact-parent-checkpoint.pt \
  --accepted-parent /path/accepted-parent.json \
  --predecode-dir /path/pre60-bundle \
  --approach-mode near \
  --seed 49 \
  --speed 1.0
```

Compact output is for replay and visualization, not offline policy training.
Use `MANORL_SYNTH_OUTPUT_FORMAT=full` or `--output-format full` when
observations, actions, rewards, contact forces, or reference trajectories are
required. Existing full v2.2 and v2.3 datasets are never rewritten by synthesis;
project an existing full dataset with:

```bash
python tools/compact_manorl_synthetic_lance.py \
  --input /path/full.lance --output /path/compact.lance
```

The exporter accepts `--output-format {full,compact-replay-visual}`. Full mode
remains the explicit audit contract; compact mode changes only persisted fields,
not physics, action semantics, checkpoint restoration, or source identity.

### Synthesis run contract

By default each accepted parent targets five accepted episodes within twelve
attempts, and bounded partial yield is preserved. Attempts use consecutive
seeds; successful rows record `episode_index`, `generation_attempt`, and the
attempt seed. Each attempt round runs in a fresh process and appends one Lance
fragment. Override with `--episodes-per-identity` / `--max-attempts-per-identity`
or the `MANORL_SYNTH_EPISODES_PER_IDENTITY` /
`MANORL_SYNTH_MAX_ATTEMPTS_PER_IDENTITY` wrapper variables.

`--num-envs` controls the simultaneous source identities; set it to the number of
eligible rows to export the whole pair, or use a smaller value and advance
`--pair-assignment-cycle` across bounded batches. The exporter refuses to
overwrite an existing dataset unless `--replace` is passed. It requires a native
checkpoint sidecar and preserves the checkpoint's serialized
residual-action/CCD settings. On hosts where nested Lance row reads are natively
unstable, set `MANORL_PREDECODED_MANIFEST` to the validated isolated-predecode
manifest; every selected pickle is SHA256-checked before the GPU rollout. The
exporter seeds NumPy and Torch/CUDA with `42` by default; override it with
`MANORL_SYNTH_SEED` or `--seed`. The seed is recorded in every row. MJX-Warp GPU
contact reductions can still vary at floating-point scale, and closed-loop
rollouts can amplify that numerical variation, so the contract does not claim
bitwise replay across separate processes.

The reward contract retains positive expected-contact reward through the raw
`object_move.end_frame`, gives ten neutral grace frames, then applies `-1.2`
(`-3 * max_contact_reward`) whenever any of the 16 physical hand keypoint forces
exceeds `0.2 N`. Legacy-reward checkpoints remain loadable for inference/export,
but only a newly trained checkpoint can learn the release behavior; training
resume remains strictly bound to the new reward contract.

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
