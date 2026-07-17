# Operations Evidence

- 2026-07-17T09:37:52+08:00: Confirmed clean `feat/wandb-update-axis` worktree at `15ab11e`; read `AGENTS.md`, `.memory/project/pi-orchestration.md`, and GitGuard contribution rules. User scope is limited to trainer and focused W&B tests, with no remote experiments.
- 2026-07-17T09:42:00+08:00: Implemented completed-PPO-update W&B primary-axis logging in `tools/train_manorl_cube1.py`, preserved transitions/FPS metrics, updated evaluation and final acceptance steps, declared the axis in JSON config, and updated the current protocol doc and focused tests. Configured command `/home/jay/anaconda3/envs/manorl_mujoco/bin/python -m pytest -q tests/manorl/test_train_wandb.py` passed 12 tests; `py_compile` and `git diff --check` passed. No remote experiment was run.
