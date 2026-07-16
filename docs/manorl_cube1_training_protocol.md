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

## Preconditions

- The source semantic fixture and target verifier must pass for source action,
  observation, and termination evidence. Its source reward fields are
  reference-only: the fixture lacks pair-filtered hand-object forces and cannot
  establish reward equality.
- Training uses target reward contract
  `target_hand_object_contact_no_action_deviation_penalty_v3`: expected
  hand-object contacts receive weighted proportional credit only when their
  pair-filtered world-force norm is strictly greater than `1.0 N`. The direct
  contact contribution is `3.0x` its unscaled contact quality, whose maximum
  remains `0.4`; the distance gate and distance reward components continue to
  use that unscaled quality. The unchanged observation contact encoding uses its
  separate `2.0 N` threshold. Action and one-shot deviation-failure penalties
  default to zero while deviation still terminates at the target training
  threshold `0.10 m`.
- PPO optimizes that environment reward at raw `1.0x` under
  `target_hand_object_contact_no_action_deviation_penalty_v3_raw_ppo_reward_1x_v3`;
  no skrl reward shaper is configured. This intentionally diverges from the
  sibling IsaacGym setup's fixed `0.5x` shaper. Native loads, including
  inference-only visualization, reject sidecars that do not declare the current
  environment/control contract.
- Target Python is `/home/jay/anaconda3/envs/manorl_mujoco/bin/python`.
- Torch must report `2.13.0+cu129` and `torch.cuda.is_available() == True`.
- Physical environment uses MJX-Warp CUDA and the policy/value model uses CUDA.
- Training defaults to `--use_residual true` and `--terminal true`. Pass
  `--use_residual false` only for source-reference diagnostics, or
  `--terminal false` for formal source-horizon termination. Target training uses
  the current-source early phase of 100 steps and the target `0.10 m` deviation threshold.
  Its normalized `[-1, 1]^26` action Box maps XYZ residual actions with scale
  `0.003 m`, gamma `0.9`, and cap `+/-0.03 m`; rotation and joint mappings retain
  their historical target values.

## Fixed Budget

Use 64 vector worlds, 48 rollout steps, and 64 PPO updates. This is 196,608
environment transitions. By default training completes all 64 updates; it
stops earlier only on non-finite values, CUDA failure, or a semantic/reset
invariant failure. An explicit `--wall-clock-seconds` safety cap may stop the
run before all requested updates complete.
The 64-world Warp broadphase uses `naconmax=2048`: Warp requires at least 31
contacts per world in this scene, and the remaining 64 slots are margin.

The fast protocol uses the source model initialization: source-compatible
orthogonal actor/critic MLP initialization, untouched PointNet/condition/FiLM
initialization, and trainable `log_std=-0.99`. It does not inject a target-only
action bias or alter the source initial exploration scale.

PPO defers its first update until source contact onset step 250. Earlier
rollouts are pure tracking phases; optimizing them changes shared actor/critic
features before the contact reward can discriminate residual corrections.

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

Evaluation uses `min(--num-envs, 128)` worlds by default; pass
`--evaluation-num-envs <count>` to override that bounded count only within
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
state. Native checkpoint sidecars must declare both
`reward_contract: target_hand_object_contact_no_action_deviation_penalty_v3` and
`ppo_reward_contract: target_hand_object_contact_no_action_deviation_penalty_v3_raw_ppo_reward_1x_v3`, and
`environment_contract: target_residual_xyz_0p003_gamma_0p9_cap_0p03_deviation_0p10_v1`.
Missing or mismatched environment contracts fail before `agent.load` for both
resume and inference, so a visualization cannot silently run under different
control or terminal dynamics.

## Launch

Start a scratch run with the source-compatible model initialization and no
checkpoint input:

```bash
JAX_PLATFORMS=cuda /home/jay/anaconda3/envs/manorl_mujoco/bin/python \
  -m tools.train_manorl_cube1 \
  --output outputs/manorl/cube1_03_scratch_run \
  --object cube1 \
  --gesture 03 \
  --num-envs 64 \
  --updates 64
```

To apply an opt-in time cap, add `--wall-clock-seconds <positive-seconds>`.
The cap may stop the run before the fixed 64-update, 196,608-transition budget
is complete.

PPO uses a 1024-sample minibatch by default. `--minibatch-size 4096` selects
the IsaacGym-sized minibatch for the 4096-world Server2 run; the selected value
must divide the 48-rollout vector batch and is recorded in both metrics and
native checkpoint runtime configuration.

For the 2,500-update Server2 run, add `--checkpoint-interval-updates 100`.
After every 100 completed PPO updates, the output-prefix namespace receives
`<output>/checkpoint-000100.pt` and `<output>/checkpoint-000100.pt.json`.
`<output>/last.pt` atomically follows the most recent completed periodic or
final checkpoint. Its fixed sidecar records native compatibility only; exact
progress remains in the numbered and final checkpoint sidecars. The legacy final
`<output>.pt` checkpoint remains separate.

### Optional W&B Tracking

W&B tracking is disabled unless `--wandb true` is passed. Provision the exact
training interpreter from the locked W&B release before enabling it:

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
logs each PPO update against monotonic environment transitions with scalar means
for `total`, `distance_x`, `distance_y`, `distance_z`, `rotation`,
`action_penalty`, `contact`, `object_stability`, `survival`, and
`deviation_penalty`, alongside `reward_mean`. It also records
`completed_episode_count` and, only when one or more episodes complete in that
update, `episode_return_mean`. It also records instantaneous and cumulative
environment transitions per second for each PPO update, plus final throughput.
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
records the target environment/control contract and raw-1.0x PPO reward boundaries. The post-training
viewer consumes the same actual `MujocoManoEnvironment` path and reports
separate observation/reward contact thresholds plus the raw PPO reward scale in
Rerun metadata.
