# Current Epistemic Model

## Phenomenon
Production ManoRL currently distinguishes hand-object contact at a strict force threshold of 2 N in observation and reward behavior. The requested behavior is a lower, aligned strict threshold of 0.2 N, with native checkpoint contracts changing so policies trained under the old reward/observation semantics cannot be accepted silently.

## Live mechanisms
- Observation contact is produced by the production force comparison in `sim/manorl/observations.py`.
- Reward hand-object contact is produced by a separate force comparison in `sim/manorl/rewards.py`; both comparisons must use the same 0.2 N contract.
- Native contract IDs are the compatibility boundary for old checkpoints; encoding `0p2n` in reward and PPO IDs should make old 2 N native checkpoints fail exact contract checks.
- Checkpoint/runtime sidecars and Rerun metadata are user-visible declarations of the active runtime contract and must report 0.2 N.
- IsaacGym source/checkpoint conversion constants may describe historical external artifacts rather than the current production contract. They should change only if tracing proves they are consumed as active ManoRL thresholds.

## Current claim
The minimal correct change is to update production threshold definitions and every active metadata/document/test contract in the assigned scope, while preserving historical records and any confirmed IsaacGym compatibility constants. The acceptance-critical behavior is strict `force > 0.2`, not `>= 0.2`, in both observation and reward paths.

## Resolved compatibility boundary
The `GYM_REWARD_CONTRACT`, `GYM_PPO_REWARD_CONTRACT`, and
`CHECKPOINT_SIDECAR_REWARD_CONFIG` values describe converted external IsaacGym
artifacts and remain at 2 N. Native production ManoRL defaults move to 0.2 N.
Because the observation gate also changes, the native environment contract is
versioned in addition to the native reward and PPO reward contracts.
