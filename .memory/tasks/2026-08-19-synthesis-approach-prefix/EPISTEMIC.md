# Current model

## Phenomenon
Accepted successful manipulation rollouts can be diversified by changing free-space approach and post-task retreat while preserving the task reference and checkpoint that produced the parent success.

## Supported mechanism
Accepted-parent v3 binds semantic source/checkpoint/object state, raw source frame0 hand pose, movement timing, and a nonzero source retreat direction. Far and near starts are immutable seeded references. Near and retreat share one endpoint distribution family but independent seeds. Approach joins pre60 with exact discrete C1 velocity continuity. The task body remains unchanged. Retreat begins at the parent’s deterministic movement-end+15 state, not a candidate rollout’s future contact, and smoothly deforms the original tail.

The policy is absent in generated approach and retreat phases. Prefix residual is zero. At retreat entry, abruptly zeroing saturated residual caused a measured 3.0 cm / 0.402 rad controller-target jump; cumulative residual is retained at the first retreat command and quintic-decayed monotonically to exact zero at the final command. This preserves controller continuity while preventing new residual accumulation.

## Contracts
- Base reference: pre60/post250, 120 Hz control/reference, 480 Hz physics ×4.
- Far: XY 0.30–0.70 m, Z 0.08–0.30 m, source direction ±30°.
- Near: independent near seed. Endpoint XY maps final-object-relative→initial-object-relative through yaw; Z retains the sampled retreat endpoint’s absolute world height. This prevents lifted objects from mapping the initial hand below the table.
- Retreat: original anchor→final XY + 0.03–0.15 m, ±30°, final wrist Z + 0.04–0.10 m.
- Raw source frame0 right q_ref[3:28] is the exact approach start pose; XYZ is sampled.
- Parent movement-end+15 (125 ms) is mapped through reference.source_frame_index; command mapping only verifies it.
- Parent last solved contact must not precede movement-end; contact qualifies the parent but does not choose retreat anchor.
- Parent movement-end+15→final wrist XY distance must be positive so the retreat direction exists.
- Prefix rejects solved controlled-hand table or object contact >0.2 N.
- Compact v2_contact is validated for state/command/reference lengths, monotonic mapping, target replay, and force-frame consistency.
- Augmentation UUID identity v3 includes semantic parent identity, mode/config, episode/attempt, independent seeds, and +15 anchor contract; machine paths are excluded.

## Ruled out
- Append after original final frame: visually produced a second retreat after retreat was already complete.
- Final-hand→object retreat direction: points inward rather than away.
- Raw C2 source-acceleration splice: full-catalog capture noise produced 3.37 m/s and 139 rad/s spikes.
- Batched viewer worlds: background worlds advanced before display.
- Rebuilding Warp runtime per reset: allocator accumulation caused OOM/exit139.
- Contact-derived anchor for each generated candidate: future final contact is unavailable before reference generation and requires two simulation passes.
- Parent last-contact as production anchor: contact mapping can omit non-keypoint contact; some families lost measured contact while target still moved 11.7–28.0 cm.
- Hard residual zero at retreat entry: produced a 3 cm / 23° controller jump.
- Transition-array indexing for state→source anchor: caused a measured off-by-two; state-aligned reference mapping is authoritative.
- Translating near endpoint Z by final→initial object height: lifted cube2 mapped the hand below the floor. World-Z preservation eliminates the mechanism.

## Remaining limitations
- Raw source/pre60 finger coordinates sometimes lie slightly outside current URDF limits. Controller targets clip while reset qpos starts at raw values. Final actual audit measured peak early joint rate 1.65 rad/s and acceleration 79.9 rad/s² without instability; this is a pre-existing data↔URDF convention issue.
- A generated rollout can retain solved contact after the parent-derived anchor because randomized approach changes closed-loop physics. The contract is explicitly parent movement-end+15.
- Same-step MJX-Warp CCD overflow detection remains unavailable with the pinned DataWarp ABI; configured capacity and runtime warning remain visible.

## Current claim
The approach/retreat mechanism was physically validated on banana and an accepted cube1:02 parent using the earlier movement-end+5 offset: near/far rows succeeded, compact/full validators passed, and smooth residual discharge removed the controller discontinuity. The user then moved the anchor later to movement-end+15. Because this only preserves the task policy for 10 additional frames before invoking the same tested retreat mechanism, the +15 change is being validated by contract/index/tail-margin tests rather than repeated GPU simulation, per user decision. A 3,429-reference +15 static audit found 3,426 valid near geometries with no below-floor start and at least 236 tail states; the remaining three have no horizontal retreat direction and are rejected by parent selection.
