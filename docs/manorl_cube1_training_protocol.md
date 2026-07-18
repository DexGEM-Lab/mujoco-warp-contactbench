# ManoRL Cube1 Fast-Training Protocol

## Scope

This experiment trains from scratch from a versioned Lance selector. The user
chooses one object and one gesture/action ID; the target discovers every fully
pre/post-padded matching row, sorts by source trajectory identity, and assigns
world `i` to candidate `i % candidate_count`. Assignments are fixed across
normal episode resets and reported in metrics/Rerun metadata. The formal
reference fixture remains `cube1_01_009`; it is not the training-data default.
Isaac rl-games checkpoints are not inputs. The catalog can inspect assignments
for every object/action present in Lance. Physical MJX execution currently has
an object runtime for `cube1`; another selected object is reported with its
actual assigned Lance identities and then rejected before simulation, rather
than being simulated using cube geometry.

This protocol starts from the completed Gym-to-MuJoCo aligned contract. Earlier
MuJoCo runs used different policy, action, point-cloud, timing, reward, and PPO
semantics; their poor returns and checkpoints are not comparable training
baselines. New experiments use only the current source-aligned defaults and
current native checkpoint contracts. The file-by-file implementation map is in
`docs/manorl_phase5_abi_inventory.md` under "Implementation summary".

## Preconditions

- The source semantic fixture and target verifier must pass for source action,
  observation, and termination evidence. Its source reward fields are
  reference-only: the fixture lacks pair-filtered hand-object forces and cannot
  establish reward equality.
- Training uses source-aligned reward contract
  `source_aligned_hand_object_contact_1x_threshold_2n_v1`: expected hand-object
  contacts receive weighted proportional credit only when their pair-filtered
  world-force norm is strictly greater than `2.0 N`. The direct contact
  contribution is `1.0x` its unscaled contact quality, whose maximum remains
  `0.4`. Action and one-shot deviation-failure penalties
  default to zero while deviation still terminates at the target training
  threshold `0.10 m`.
- PPO applies the source `0.5x` shaper under
  `source_aligned_hand_object_contact_1x_threshold_2n_shaper_0p5_v1`.
  Native loads, including
  inference-only visualization, reject sidecars that do not declare the current
  environment/control contract.
- Target Python is `/home/jay/anaconda3/envs/manorl_mujoco/bin/python`.
- Torch must report `2.13.0+cu129` and `torch.cuda.is_available() == True`.
- Physical environment uses MJX-Warp CUDA and the policy/value model uses CUDA.
- Training defaults to `--use_residual true` and `--terminal true`. Pass
  `--use_residual false` only for source-reference diagnostics, or
  `--terminal false` for formal source-horizon termination. Target training uses
  the validated early phase of 50 steps, movement pre-padding 250, and the
  `0.10 m` deviation threshold. Its normalized `[-1, 1]^26` action Box maps XYZ
  residual actions with per-step scale `0.005 m`, gamma `0.9`, and cap
  `+/-0.05 m`. The first two thumb scales/caps are `0.10/0.12` and `1.0/1.2`.
  FiLM and dynamic PointNet are enabled by default; GPU sampling uses the
  global CUDA Torch RNG.

## Fixed Budget

The production defaults use 2,048 vector worlds, 48 rollout steps, and 8,000
PPO updates. This is 786,432,000 environment transitions. A numbered native
checkpoint is written every 200 completed updates by default. Training stops
earlier only on non-finite values, CUDA failure, or a semantic/reset invariant
failure. An explicit `--wall-clock-seconds` safety cap may stop the run before
all requested updates complete. Smaller world/update counts remain available
as explicit smoke or diagnostic overrides; they are not convergence claims.
The Warp broadphase capacity is derived from the active world count (at least
31 contacts per world plus the configured margin).

The fast protocol uses the source model initialization: source-compatible
orthogonal actor/critic MLP initialization, untouched PointNet/condition/FiLM
initialization, and trainable `log_std=-0.99`. It does not inject a target-only
action bias or alter the source initial exploration scale.

PPO updates begin after the first completed rollout (`learning_starts=0`), as
in the source rl-games contract. The contact onset frame is an environment
phase, not a training-start gate.

The policy update uses the resolved Gym run's legacy adaptive-KL contract.
Each rollout stores the Gaussian policy mean and standard deviation. Every
completed minibatch computes the source rl-games exact Gaussian KL against
those stored parameters, updates the stored reference for that minibatch, and
immediately adjusts learning rate by factor `1.5` within `[1e-6, 1e-2]` around
the `0.016` threshold. The target does not use skrl's sampled-log-ratio KL to
stop a learning epoch early. Exact and approximate KL are both telemetry, but
only exact KL controls the scheduler.

## Evaluation

Run deterministic mean-action evaluation for one fixed 791-call episode before
and after training. Report three rows:

1. Reference baseline: zero residual action.
2. Untrained policy: initial deterministic policy mean.
3. Trained policy: final deterministic policy mean.

Every row reports return, mean reward, mean action absolute value, reset/timeout
status, final object-target distance, maximum object-target distance, and
contact-window reward. Under the target 0.10 m training deviation threshold,
the zero-residual reference baseline follows the historical 0.1 m reset boundary, so the short
run is accepted when the trained policy is finite, exceeds the untrained return,
and does not reset before that zero-reference call count. Completing all 791 calls is
reported as a stretch result, not silently assumed. The reference baseline is a
controller diagnostic, not a learned-policy comparator.

Evaluation uses one world by default, matching the Gym reference and avoiding
the worst-of-many early-reset statistic. Pass `--evaluation-num-envs <count>`
explicitly for a bounded multi-world diagnostic; values are limited to
`1..min(--num-envs, 128)`. This prevents a requested evaluation from recreating
a second full-size training runtime. The zero, untrained, and trained rows all
use the same evaluation trajectory prefix.
Their evaluation-only PPO configuration selects the largest divisor shared by
the training minibatch and evaluation rollout batch, while the training PPO
configuration and update semantics remain unchanged. The initial training
policy, value, optimizer, and normalizer state is written to an owned temporary
native checkpoint, loaded for the untrained row, then loaded again immediately
before training. That temporary checkpoint and sidecar are removed after the
boundary completes, and the initial bounded runtime is released before
training begins. The trained row is always evaluated from a fresh bounded
runtime after loading the final native checkpoint. It is the executable artifact
a user receives, and avoids reporting train-process-only normalizer or BatchNorm
state. Native training resume requires the reward, PPO, and environment contract
IDs emitted by the current runtime. Inference checks the checkpoint schema,
finite tensor state, provenance, and strict model key/shape compatibility;
sidecar PointNet, sampling, action, and timing values are versioned metadata, not
permanent equality constraints on future runtime versions.

## Launch

Start the production run with the source-compatible model initialization and
no checkpoint input. The object, gesture, world count, update budget,
checkpoint cadence, evaluation count, FiLM, residual, terminal, and W&B values
below are also the CLI defaults:

```bash
JAX_PLATFORMS=cuda /home/jay/anaconda3/envs/manorl_mujoco/bin/python \
  -m tools.train_manorl_cube1 \
  --output outputs/manorl/cube1_01_default \
  --object cube1 \
  --gesture 01 \
  --num-envs 2048 \
  --updates 8000 \
  --checkpoint-interval-updates 200 \
  --evaluation-num-envs 1 \
  --film true \
  --use_residual true \
  --terminal true \
  --wandb true
```

To apply an opt-in time cap, add `--wall-clock-seconds <positive-seconds>`.
The cap may stop the run before the fixed 8,000-update, 786,432,000-transition
budget is complete.

The validated Gym checkpoint's resolved run config uses a 4096-sample
minibatch. ManoRL therefore defaults to the largest divisor shared by `4096`
and the configured 48-step rollout batch. The production 2,048-world default
therefore uses `4096` (24 minibatches per rollout, three epochs). For smaller
explicit diagnostic world counts, omission resolves to the largest valid
divisor; `--minibatch-size` remains an explicit override. Any selected value
must divide the rollout batch and is recorded in metrics and native checkpoint
runtime configuration.

The default checkpoint cadence is 200 updates. After every 200 completed PPO
updates, the output-prefix namespace receives
`<output>/checkpoint-000200.pt` and `<output>/checkpoint-000200.pt.json`.
Pass `--checkpoint-interval-updates <count>` to select another positive cadence.
`<output>/last.pt` atomically follows the most recent completed periodic or
final checkpoint. Its fixed sidecar records native compatibility only; exact
progress remains in the numbered and final checkpoint sidecars. The legacy final
`<output>.pt` checkpoint remains separate.

### W&B Tracking

W&B tracking is enabled by default. Pass `--wandb false` for an explicit local
or diagnostic run that must not initialize W&B. Provision the exact training
interpreter from the locked W&B release before using the default:

```bash
/home/jay/anaconda3/envs/manorl_mujoco/bin/uv pip install \
  --python /home/jay/anaconda3/envs/manorl_mujoco/bin/python \
  "wandb==0.28.0"
```

An enabled run uses `project=one_policy`, `group=s02`, and the authenticated
default account when `--wandb-entity` is left empty. Its name defaults to the
output prefix plus the selected object and gesture; its default tags are
`manorl,mujoco,skrl`.

```bash
  --wandb true \
  --wandb-project one_policy \
  --wandb-group s02 \
  --wandb-tags manorl,mujoco,skrl
```

The training process initializes one W&B run after resolving trajectory
assignments and devices. It creates the output parent before initialization and
passes it as W&B's local directory, so SDK state is stored at
`<output-parent>/wandb` under ignored training outputs rather than the repository
root. It records the complete serializable training/PPO/raw reward configuration,
logs each PPO update against its completed PPO update count as the primary W&B axis, with monotonic environment transitions retained as the `transitions` secondary metric. It records scalar means
for `total`, `distance_x`, `distance_y`, `distance_z`, `rotation`,
`action_penalty`, `contact`, `object_stability`, `survival`, and
`deviation_penalty`, alongside `reward_mean`. It also records
`completed_episode_count` and, only when one or more episodes complete in that
update, `episode_return_mean`. It also records instantaneous and cumulative
environment transitions per second for each PPO update, plus final throughput. Initial and final evaluations use the corresponding completed PPO update count as their W&B step.
Every PPO update also records exact KL mean/min/max, approximate KL mean,
learning-rate start/end/min/max, scheduler increase/decrease counts, and the
number of completed minibatches. These metrics are emitted even when no episode
completes, so early scheduler failures cannot be hidden by episode logging.
Each completed vector step emits one `manorl.completed_episode_returns.v1` JSON
record with parallel `env_ids` and exact `returns` arrays to
`<output>.episodes.jsonl`; active records first accumulate in
`<output>.episodes.jsonl.partial`, which is preserved after an interruption and
atomically published only after successful training. This JSONL is the complete
per-episode evidence while W&B receives one update-level return histogram rather
than a log call for every episode. Human stdout prints only completed count,
mean, minimum, and maximum after the JSONL flush. `--console-format json`
preserves the original JSON stdout records and final result for machine
consumers. Histogram samples are retained only for the
current update, bounded by one rollout batch (196,608 values at 4096 worlds and
48 rollout steps), then discarded. W&B artifacts include that episode JSONL
alongside the checkpoint and
sidecar, metrics JSON, evaluation trace, and a completed Rerun recording when
one exists. These target-native aggregates are semantically related to IsaacGym
reward telemetry, but their logger key names are not an identity contract. It
writes the zero/untrained/trained evaluation summaries and final acceptance
values, preserving `evaluation/policy/*` as the untrained and trained policy
comparison time series for existing dashboards, then uploads those artifacts.
An enabled run fails on W&B initialization
or logging errors; cleanup failures do not mask a training error.

To record one actual training world without changing PPO actions, rollout
memory, or updates, add a Rerun output path. `--rerun-stride 4` records env 0
once every four vector control calls:

```bash
  --rerun-output outputs/manorl/cube1_03_scratch_run.rrd \
  --rerun-env-id 0 \
  --rerun-stride 4
```

The default recording is env 0 only. It keeps one stable artifact:
`cube1_03_scratch_run.rrd`. The recorder writes the active episode privately,
then atomically replaces that path when the next delayed reset is applied; it
does not retain an episode archive. Recording never opens a second GUI; open
the latest completed recording from another terminal with:

```bash
/home/jay/anaconda3/envs/manorl_mujoco/bin/rerun \
  outputs/manorl/cube1_03_scratch_run.rrd
```

A current or completed captured trainer log can be summarized without touching
the training process:

```bash
/home/jay/anaconda3/envs/manorl_mujoco/bin/python \
  tools/watch_manorl_training.py outputs/manorl/server2.train.log --follow
```

For local CUDA/X11-or-Wayland visual PPO training, run the 8-world wrapper:

```bash
JAX_PLATFORMS=cuda scripts/train_manorl_cube1_visual.sh --updates 64
```

It sets `--num-envs 8 --evaluation-num-envs 8 --minibatch-size 384`, W&B off,
and `--headless false` with an eight-world tile view. Extra arguments follow the
wrapper defaults and a timestamp/pid output prefix avoids collisions. The
training viewer mirrors state after the trainer's own vector step, never calls
policy or environment step methods, and renders every `--viewer-stride` steps.
Its controls are left-drag rotate, right-drag horizontal pan, middle-drag
vertical pan, wheel zoom, and R reset. Close the window or press Esc to request
graceful exit after the current full rollout/update; regular checkpoint,
evaluation, JSONL, and artifact publication
then proceed. It requires a CUDA-capable training environment plus `DISPLAY` or
`WAYLAND_DISPLAY`; no CPU fallback exists.

The embedded Rerun blueprint explicitly displays the object point cloud and
tracks the object with an orbital camera. It records actual state, target, 26D
action/controller target, hand keypoints, contact force vectors, named reward
terms, per-step cumulative `episode/return`, reset causes, and the thresholds
used by that recording.

## Visual Test With Background Rerun

Run the production MuJoCo viewer and record env 0 on the same environment
steps. `--loop` keeps the normal visual simulation running through delayed
resets; it does not start a Rerun GUI.

```bash
JAX_PLATFORMS=cpu /home/jay/anaconda3/envs/manorl_mujoco/bin/python \
  -m sim.manorl.view_environment \
  --device cpu --object cube1 --gesture 03 --num-envs 8 --render-env 0 \
  --use_residual true --terminal true --loop \
  --rerun-output outputs/manorl/cube1_03_latest.rrd
```

`--use_residual` and `--terminal` both default to `true` in viewer,
standalone recorder, and training CLIs. Set either explicitly with
`--use_residual false` or `--terminal false`. The normal viewer supplies zero
26D actions, so enabling residual processing alone does not alter its
reference-following motion.

From a second terminal, inspect the latest published env-0 episode without
altering the MuJoCo viewer:

```bash
/home/jay/anaconda3/envs/manorl_mujoco/bin/rerun \
  outputs/manorl/cube1_03_latest.rrd
```

## Artifacts

Write config, metrics, a final native skrl checkpoint, and a compact evaluation
trace under a unique output prefix. A requested periodic cadence creates an
owned `<output>/` checkpoint namespace; numbered and final sidecars include
completed update and environment-transition progress. Payloads are prepared as
temporary files and published with replacements. `last.pt.json` is installed
once before the first payload and remains fixed compatibility metadata, so later
updates replace only complete `last.pt` payloads. Every checkpoint sidecar
records the source-aligned environment/control contract and 0.5x PPO reward boundary. The post-training
viewer consumes the same actual `MujocoManoEnvironment` path and reports
the contact threshold plus the PPO reward scale in
Rerun metadata.
