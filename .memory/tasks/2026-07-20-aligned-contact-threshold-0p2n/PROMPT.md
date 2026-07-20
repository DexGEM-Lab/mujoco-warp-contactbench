## Objective
Change the production ManoRL observation contact threshold and actual reward hand-object contact threshold from strict `> 2.0 N` to strict `> 0.2 N`, keep both aligned, version incompatible native reward/PPO contracts, update runtime/checkpoint/Rerun metadata and active documentation, add focused regression coverage, and commit the coherent result on `feat/aligned-contact-threshold-0p2n`.

## Workbench
- Trace the observation and reward contact paths, native checkpoint contract checks, checkpoint/runtime sidecar metadata, Rerun metadata, current protocol/ABI/README text, and focused tests.
- Inspect history and all refs for prior 0.2 N threshold work; determine whether IsaacGym source/checkpoint conversion constants are historical external contracts that must remain 2 N.
- Implement only within the assigned files and preserve historical experiment records.
- Run focused tests and static checks, review the diff, and commit without merging into `dev`.

## Context
- Worktree: `/home/jay/dexrobot/FromSSH/manoRL_mujoco-worktrees/feat-aligned-contact-threshold-0p2n`
- Branch: `feat/aligned-contact-threshold-0p2n`
- Assignment date: 2026-07-20.
- Initial commit: `d9a581c`.
- Owned implementation scope: `sim/manorl/observations.py`, `sim/manorl/rewards.py`, `sim/manorl/checkpoint.py` only if required, `sim/manorl/rerun_recorder.py`, `tests/manorl/test_observations_rewards.py`, `tests/manorl/test_gym_checkpoint.py`, `tests/manorl/test_environment.py`, `docs/manorl_cube1_training_protocol.md`, `docs/manorl_phase5_abi_inventory.md`, `README.md`, and this task-memory directory.

## Task specifications
- Production observation and reward contact thresholds must be exactly `0.2 N` and use strict greater-than semantics: exactly `0.2` is non-contact; values above it are contact.
- Keep observation and actual reward hand-object thresholds aligned at the same constant/value.
- Native reward and PPO contract IDs must encode `0p2n` so old native 2 N checkpoints are rejected.
- Checkpoint/runtime sidecar metadata and Rerun metadata must expose `0.2 N`.
- Update active production docs and README, plus focused tests covering threshold equality, above-threshold behavior, alignment, metadata, and contract rejection.
- Historical experiment results/memory records must remain unchanged.
- Determine from history and references whether IsaacGym source/checkpoint conversion constants are historical external contracts; leave such 2 N constants unchanged unless evidence shows they are production ManoRL contracts.
- Acceptance: focused tests pass, targeted syntax/diff checks pass, and the committed diff stays within the owned scope.

## Constraints
- Do not touch outputs, Server2, unrelated files, or historical `.memory` task results.
- Preserve unrelated changes.
- Do not merge into `dev`.
- Use focused validation only; do not run the full suite.
- Commit the coherent result on this feature branch and report the commit SHA, changed files, tests, and residual risks.
