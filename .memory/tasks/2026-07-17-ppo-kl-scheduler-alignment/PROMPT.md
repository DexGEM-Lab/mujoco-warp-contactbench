# PPO KL/Scheduler Alignment

## Objective

Align ManoRL's skrl PPO KL measurement and adaptive learning-rate schedule with
the resolved rl-games contract that trained the validated Gym checkpoint, then
run a bounded `cube1/01` 2,048-environment convergence diagnostic on Server2.

## Source contract

- Resolved Gym run configuration:
  `IsaacGymEnvs/isaacgymenvs/runs/18-09-37-36_MANOHand_all_all-new_setting-val099-new_obs/config.yaml`
- Rollout length 48, minibatch size 4096, 3 mini-epochs.
- Initial learning rate `3e-4`, adaptive KL threshold `0.016`.
- Default rl-games `schedule_type=legacy` because the resolved config does not
  override it.
- Store rollout policy mean and standard deviation. After each minibatch update,
  compute rl-games `policy_kl` from current and stored Gaussian parameters and
  immediately update LR by factor 1.5 within `[1e-6, 1e-2]`.
- Do not use skrl's action-log-ratio approximate KL early stopping.

## Ownership

- `sim/manorl/skrl_runtime.py` and a narrowly scoped PPO helper module if needed.
- `tools/train_manorl_cube1.py` for defaults and diagnostics.
- Focused files under `tests/manorl/`.
- This task-memory directory.

Do not modify environment physics, reward, observations, trajectory loading,
assets, multi-object runtime, generated outputs, or the primary `dev` worktree.

## Acceptance

- Executable unit tests match the rl-games KL formula and legacy scheduler
  transitions exactly.
- PPO stores old mean/std and schedules once per completed minibatch without
  KL-driven early termination.
- W&B/console metrics expose exact KL and scheduler behavior from the first PPO
  update, independently of completed episodes.
- A bounded Server2 `cube1/01`, 2,048-env run completes without numerical errors
  and provides evidence on early LR/contact behavior before any long run.
