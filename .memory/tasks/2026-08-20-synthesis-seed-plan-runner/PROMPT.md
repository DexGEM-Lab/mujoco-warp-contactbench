## Objective
Run the fixed banana VLA near/far augmentation plan under the established ManoRL synthesis setting and deliver the actual bounded-attempt yield as one compact v2_contact Lance dataset.

## Contract
- Inputs are the original 35 `manorl_accepted_synthetic_parent_v3` descriptors; a failed attempt never changes its parent.
- The plan has 10 target distance cells per parent and 12 globally unique candidate seeds per cell.
- Each candidate is an atomic near/far pair: accept only when both succeed; otherwise discard both and try the next candidate in the same distance cell.
- Each slot stops after the first accepted pair or after all 12 candidates fail. An exhausted slot produces no rows and does not block later slots.
- The target is at most 350 accepted pairs / 700 rows. Final yield is the number actually accepted under the fixed budget; never change physics, parent, distance cell, or success criteria to meet the target.
- Preserve the established 4 cm endpoint-smooth approach Z arc from `ApproachPrefixConfig` for both modes.
- Both modes require `synthetic_parent_movement_end_plus15_retreat_tail_xy_extra_z_extra_discrete_c2_v4`.
- Preserve compact `synthetic_mano_target_replay_visual_v2_contact`, contact/reference/command mapping, accepted-parent augmentation UUID semantics, and checkpoint policy-transfer behavior.
- Reuse one single-env MJX/checkpoint runtime within each parent worker; run one fresh process per parent to bound Warp allocator lifetime.
- Persist accepted slots, candidate failures, and exhausted slots atomically for resume without regenerating accepted rows.

## Completion semantics
- `attempts_complete`: all ten slots reached either accepted or 12-candidate exhausted state.
- `complete` / `all_slots_succeeded`: all ten target pairs succeeded.
- A parent can have `attempts_complete=true` and `complete=false`; its accepted rows remain valid partial yield.
- The batch finishes after all 350 slots have terminal states, then merges every accepted pair and reports actual yield.

## Owned surface
- `tools/run_manorl_synthesis_seed_plan.py`
- `tests/manorl/test_synthesis_seed_plan.py`
- task memory

## Validation
- Pure plan/schema/status tests.
- Server1 source/input hash preflight.
- Compact validator on the merged actual-yield dataset.
- Report overall and per-action/parent/distance-cell success and failure counts.

## Production delivery
- Server1 only, four RTX 4090 GPUs, parent-index shards.
- Merge successful parent rows deterministically into one actual-yield Lance.
- Publish with local staging then `cp -r` to NAS; do not use rsync temporary files on CIFS.

## Constraints
- Do not modify parent selection, seed-cell membership, contact/deviation thresholds, policy, retreat bounds, or approach setting to improve yield.
- Do not modify primary worktree user files such as `test.sh`.
- No subagents.
