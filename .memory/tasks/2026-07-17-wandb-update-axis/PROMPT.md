## Objective

Make the primary W&B step in the assigned trainer equal completed PPO update count, matching checkpoint-002000.pt semantics, while preserving environment transition telemetry.

## Workbench

- Inspect `tools/train_manorl_cube1.py` W&B helpers and all evaluation call sites.
- Update focused assertions in `tests/manorl/test_train_wandb.py`.
- Run focused W&B tests, `py_compile`, and `git diff --check`; commit and stop.

## Context

- Worktree: `/home/jay/dexrobot/FromSSH/manoRL_mujoco-worktrees/feat-wandb-update-axis`
- Branch: `feat/wandb-update-axis`
- The sibling `feat-wandb-gym-panel-schema` worktree is out of scope and must remain untouched.

## Task specifications

- `_log_wandb_update` must call `run.log(step=update)` and set `global_step` and `update` to that completed update.
- Keep environment transition count under `transitions`; retain existing throughput metrics.
- `_log_wandb_evaluations` must accept both `update` and `transitions`, and use `step/update/global_step=update`.
- Update initial and final evaluation call sites, including final acceptance logging.
- Add a JSON-serializable config declaration that the primary W&B axis is `completed_ppo_updates` and transitions are a secondary metric.
- Update focused tests to expect update steps `[1, 2]`, evaluation steps `[0, 2]`, and preserved transitions `[3072, 6144]`.
- Do not change PPO, reward, or runtime behavior; do not run remote experiments.

## Constraints

Only edit `tools/train_manorl_cube1.py`, `tests/manorl/test_train_wandb.py`, and the current protocol document if strictly required.
Do not modify the sibling W&B panel-schema worktree.
Commit normally on this feature branch after validation.
