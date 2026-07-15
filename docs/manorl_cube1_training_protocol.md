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
- Training uses target reward contract `target_hand_object_contact_v1`: expected
  hand-object contacts receive weighted proportional credit only when their
  pair-filtered world-force norm is strictly greater than `1.0 N`. The unchanged
  observation contact encoding uses its separate `2.0 N` threshold.
- PPO optimizes that environment reward at raw `1.0x` under
  `target_hand_object_contact_v1_raw_ppo_reward_1x_v1`; no skrl reward shaper
  is configured. This intentionally diverges from the sibling IsaacGym setup's
  fixed `0.5x` shaper. Native checkpoints from that legacy objective are
  rejected before load.
- Target Python is `/home/jay/anaconda3/envs/manorl_mujoco/bin/python`.
- Torch must report `2.13.0+cu129` and `torch.cuda.is_available() == True`.
- Physical environment uses MJX-Warp CUDA and the policy/value model uses CUDA.
- Training defaults to `--use_residual true` and `--terminal true`. Pass
  `--use_residual false` only for source-reference diagnostics, or
  `--terminal false` for formal source-horizon termination. It uses the
  current-source early phase of 100 steps and source deviation threshold of 0.1 m.

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
contact-window reward. Under the source 0.1 m training deviation threshold the
zero-residual reference baseline currently resets near call 287, so the short
run is accepted when the trained policy is finite, exceeds the untrained return,
and does not reset before that zero-reference call count. Completing all 791 calls is
reported as a stretch result, not silently assumed. The reference baseline is a
controller diagnostic, not a learned-policy comparator.

The trained row is always evaluated from a fresh runtime after loading the
native checkpoint. It is the executable artifact a user receives, and avoids
reporting train-process-only normalizer or BatchNorm state. Native checkpoint sidecars must declare both
`reward_contract: target_hand_object_contact_v1` and
`ppo_reward_contract: target_hand_object_contact_v1_raw_ppo_reward_1x_v1`.
Missing or mismatched contracts fail before resume rather than silently changing
the training objective.

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

For the 2,500-update Server2 run, add `--checkpoint-interval-updates 100`.
After every 100 completed PPO updates, the dedicated output directory receives
`checkpoint-000100.pt` and `checkpoint-000100.pt.json`. `last.pt` and its
sidecar copy the most recently completed periodic or final checkpoint; the
legacy final `<output>.pt` checkpoint remains separate.

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
logs each PPO update against monotonic environment transitions, writes the
zero/untrained/trained evaluation summaries and final acceptance values, then
uploads the native checkpoint and sidecar, metrics JSON, evaluation trace, and a
completed Rerun recording when one exists. An enabled run fails on W&B
initialization or logging errors; cleanup failures do not mask a training error.

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

The embedded Rerun blueprint explicitly displays the object point cloud and
tracks the object with an orbital camera. It records actual state, target, 26D
action/controller target, hand keypoints, contact force vectors, named reward
terms, reset causes, and the thresholds used by that recording.

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
trace under a unique output prefix. A requested periodic cadence writes numbered
native checkpoints in the output directory; their sidecars include completed
update and environment-transition progress. `last.pt` and its sidecar copy the
most recent completed checkpoint. Every checkpoint sidecar records the
environment and raw-1.0x PPO reward boundaries. The post-training viewer
consumes the same actual `MujocoManoEnvironment` path and reports separate
observation/reward contact thresholds plus the raw PPO reward scale in Rerun
metadata.
