# ManoRL Cube1 Fast-Training Protocol

## Scope

This experiment trains from scratch on exactly the accepted Lance trajectory:
dataset version 132, row 1, `cube1_01_009`, source slice `[440,1232)`. It is a
single-trajectory convergence demonstration, not a multi-object or
generalization result. Isaac rl-games checkpoints are not inputs.

## Preconditions

- The source semantic fixture and target verifier must pass.
- Target Python is `/home/jay/anaconda3/envs/manorl_mujoco/bin/python`.
- Torch must report `2.13.0+cu129` and `torch.cuda.is_available() == True`.
- Physical environment uses MJX-Warp CUDA and the policy/value model uses CUDA.
- Training uses `residual_enabled=True`, current-source early phase of 100
  steps, and source deviation termination of 0.1 m.

## Fixed Budget

Use 64 vector worlds, 48 rollout steps, and 64 PPO updates. This is 196,608
environment transitions. Stop earlier only at a 20 minute wall-clock limit or
on non-finite values, CUDA failure, or a semantic/reset invariant failure.
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
reporting train-process-only normalizer or BatchNorm state.

## Launch

Start a scratch run with the source-compatible model initialization and no
checkpoint input:

```bash
JAX_PLATFORMS=cuda /home/jay/anaconda3/envs/manorl_mujoco/bin/python \
  -m tools.train_manorl_cube1 \
  --output outputs/manorl/cube1_01_009_scratch_run \
  --num-envs 64 \
  --updates 64 \
  --wall-clock-seconds 1200
```

To record one actual training world without changing PPO actions, rollout
memory, or updates, add a Rerun output path. `--rerun-stride 4` records env 0
once every four vector control calls:

```bash
  --rerun-output outputs/manorl/cube1_01_009_scratch_run.rrd \
  --rerun-env-id 0 \
  --rerun-stride 4
```

Open the resulting file with `/home/jay/anaconda3/envs/manorl_mujoco/bin/rerun outputs/manorl/cube1_01_009_scratch_run.rrd`.
The control-call timeline remains monotonic across delayed resets; it records
actual state, target, 26D action/controller target, 64-point object cloud, hand
keypoints, contact force vectors, named reward terms, reset causes, and the
thresholds used by that run.

## Artifacts

Write config, metrics, native skrl checkpoint, and a compact evaluation trace
under a unique `outputs/manorl/` prefix. The post-training viewer consumes the
native checkpoint and renders the same actual `MujocoManoEnvironment` path.
