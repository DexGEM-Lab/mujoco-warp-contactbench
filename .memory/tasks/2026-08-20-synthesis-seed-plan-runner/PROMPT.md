## Objective
Implement and deploy a plan-driven ManoRL synthesis runner for the banana VLA dataset. Consume the fixed original 35-parent seed/fallback plan and produce exactly 10 accepted paired seeds per parent: one near and one far row, both with mandatory movement-end+15 retreat, for 700 compact v2_contact rows total.

## Contract
- Inputs are the original 35 `manorl_accepted_synthetic_parent_v3` descriptors; a failed candidate never changes its parent.
- The plan has 10 target distance cells per parent and 12 globally unique candidates per cell.
- Near and far form an atomic pair: accept only when both succeed; otherwise discard both and try the next candidate in the same far/near distance cell.
- Never substitute seed+1 silently.
- Both modes use a 10 cm endpoint-smooth collision-avoidance Z arc; sampled start positions and distance coverage remain unchanged.
- Both modes require `synthetic_parent_movement_end_plus15_retreat_tail_xy_extra_z_extra_discrete_c2_v4`.
- Preserve compact `synthetic_mano_target_replay_visual_v2_contact`, contact/reference/command mapping, accepted-parent augmentation UUID semantics, and checkpoint policy-transfer behavior.
- Reuse one single-env MJX/checkpoint runtime within each parent worker; run one fresh process per parent to bound Warp allocator lifetime.
- Persist accepted slots and failures atomically for resume without regenerating accepted rows.

## Owned surface
- `tools/run_manorl_synthesis_seed_plan.py`
- `tests/manorl/test_synthesis_seed_plan.py`
- task memory

## Validation
- Pure plan/schema/identity tests.
- Real one-parent, one-slot near/far smoke under the original plan.
- Compact validator on the two-row output.
- Server1 four-GPU launch only after source/input hash preflight.

## Production delivery
- Server1 only, four RTX 4090 GPUs, parent-index shards.
- Merge 35 parent datasets deterministically into one 700-row Lance.
- Validate 350 paired seeds, near/far 350 each, five parents/action, 100 rows/action, unique UUIDs, and preserved distance-decile coverage.

## Constraints
- Work only in `feat/synthesis-seed-plan-runner` until integration.
- Do not modify primary `dev` worktree feature files or user `test.sh`.
- No subagents.
