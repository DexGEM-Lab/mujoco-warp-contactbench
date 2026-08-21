# Current model

## Phenomenon
The fixed 35 accepted parents must each yield ten atomic near/far seed pairs with mandatory movement-end+15 retreat. The optimized plan uses non-consecutive seeds and same-distance-cell fallbacks, while the ordinary exporter advances consecutive seeds and rebuilds a runtime per attempt.

## Fixed production identity
- The original 35 accepted-parent v3 descriptors are immutable production inputs.
- Each parent binds its successful source/reference, checkpoint, object offset, raw frame-0 hand pose, and movement-end+15 anchor.
- The original 350 primary seeds and 4,200 planned fallback candidates remain the coverage plan.
- Candidate failure advances only within the parent/slot's fixed far/near distance cell. It never changes the parent.

## Runtime mechanism
- Approach prefix: no policy inference, processed action, or accumulated residual.
- Both near and far use a 10 cm endpoint-smooth Z arc. This preserves their sampled start position and distance quantile while avoiding prefix hand-object contact observed with the prior 4 cm path.
- Middle: the accepted parent's checkpoint policy runs normally against the unchanged source/reference and object offset.
- Retreat: at movement-end+15, policy inference and new residual accumulation stop; entry residual decays smoothly while v4 deforms the retreat endpoint.
- Retreat is mandatory for both near and far.

## Runner design
- One parent worker owns one single-environment MJX/checkpoint runtime and persists slot status atomically.
- Near and far for one candidate seed are collected sequentially in the same runtime and accepted atomically.
- Successful pairs append two compact `synthetic_mano_target_replay_visual_v2_contact` rows.
- One fresh worker process per parent bounds Warp allocator lifetime. Server1 distributes parent workers across four GPUs.
- Planned seeds are explicit; no seed+1 fallback exists.

## Evidence
- The first original near slot (`banana_01_079`, seed 100408) failed at 4 cm due to prefix hand-object contact.
- Holding all semantic inputs fixed and changing only the arc to 10 cm produced a successful near rollout; the paired far rollout also succeeded.
- Runner smoke accepted both rows at fallback rank 0 and the compact validator passed at 120/480 Hz x4.

## Negative evidence
- Prefix geometry replay alone proves geometry, not rollout success.
- Replacing parents after candidate failure violates the production contract.
- Reusing the geometry seed as an explanation for policy point-cloud drift was tested and did not explain prior deviations.
- Rebuilding many MJX runtimes in one Python process accumulates Warp allocations; parent workers must be process-isolated.

## Current claim
Runner v2 is ready to integrate: its schema tests and real paired smoke pass. After merge and Server1 input deployment, four GPU parent shards can produce resumable parent datasets, which can then be deterministically merged and validated as a 700-row compact Lance dataset.
