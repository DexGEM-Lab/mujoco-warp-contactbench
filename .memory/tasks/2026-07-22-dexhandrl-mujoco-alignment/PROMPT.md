# DexHandRL MuJoCo Alignment — Milestone 1 Bootstrap

## Objective

Bootstrap `feat/dexhandrl-mujoco-alignment` from the current `dev` MANO reference implementation and apply, in chronological order, the DexHandRL migration commits `f3cb5fc`, `934e48f`, and `c06ad8b` from `origin/dexhandrl-mujoco-migration`.

Preserve imported DexHandRL semantics while resolving conflicts so current `dev` MANO code and workflows remain intact.

## Acceptance target

Establish an executable baseline for **direct Gym checkpoint reuse**, with parity checkpoints across:

1. reference trajectory,
2. raw observation,
3. normalized observation,
4. policy action,
5. control targets, and
6. rollout.

The first concrete acceptance case is `cube1` / action `01`. Do **not** use retraining or reward tuning as a substitute for direct checkpoint reuse and parity. Do **not** merge this feature into `dev`.

## Required milestone work

- Initialize this task's `PROMPT.md`, append-only timestamped `OPS.md`, and causal-model `EPISTEMIC.md`.
- Run focused imports/tests for the DexHandRL surface:
  `tests/test_obs_layout.py`, `tests/test_rlgames_checkpoint.py`,
  `tests/test_skrl_env_contract.py`, `tests/test_skrl_env_parity.py`,
  `tests/test_skrl_training_defaults.py`, and
  `tests/test_skrl_trajectory_routing.py`, plus cheap directly implicated tests.
- Inventory the exact available and missing dataset, Gym source checkout,
  rl-games checkpoint, and Python/runtime prerequisites required to execute the
  `cube1` / action `01` baseline.

## Constraints

Work only in this linked feature worktree. Preserve current MANO behavior and
unrelated files. Use explicit-file staging and inspect staged diffs. Commit
coherent checkpoints. Stop after bootstrap and focused baseline checks; do not
begin speculative observation/action repairs.

All visual tests and final rollouts must use `env=1`, export `DISPLAY=:1`, and
run explicitly non-headless with the viewer enabled when the runtime supports
one. Numerical trace collection may remain headless.
