# Current model

The v3.1 538-D route was an incompatible experiment, not a base for the
accepted v4 contract: it built cache geometry through native `MjData` and
`mj_geomDistance`, omitted `delta_ref`, substituted origin-only relative
motion, read post-integration derived state without a final forward, and mixed
all-contact with paired-contact semantics. Any normalizer/checkpoint from that
route is invalid for v4.

The active source is `manorl.autonomy.*.v4`: seven raw blocks (119/109/123/
352/192/50/12 = 957) and 829 after a fixed PointNet code. The model input has
no trajectory identity. Reference cache geometry (`signed_gap`, proximity,
confidence, valid) is immutable demonstration content; runtime anchor error is
`delta_actual^O - delta_ref^O`, which is zero at exact replay even when the
reference pair has positive gap or penetration. The actual block carries the
all-object contact resultant, including table context; each geometry region
carries only the canonical hand→object paired force. This separation is the
relevant physical intervention because object support and grasp counterparty
are different causes.

The cache compiler uses N=1 Warp FK (`mjx.make_data`, `mjx.forward`,
`impl="warp"`) plus compiled collision geom metadata and exact cube2 box
geometry. No native state/step/forward/distance API is in its production path.
The device state distinguishes origins, COMs and point velocities using
`xpos/xipos/subtree_com/cvel`; anchor derivative includes the co-rotating
`-omega_O×delta` term. The fused runtime does four Warp steps then one Warp
forward before extracting all post-action features/reward, so the transition
is same-time.

Evidence [OPS 2026-09-10T00:00:00Z]: seven independent formula tests pass, and
the real pinned cube2_02_2833 N=1 CPU runtime resets to finite raw `(1,957)` /
encoded `(1,829)` and returns a finite valid post-action reward `1.8973923`.
This proves only the contract path starts and steps; it says nothing about
learning, grasp, lift, scale, or generalization. Training is stopped by user.

Live boundary: the pinned fast reducer intentionally supports only pyramidal
`condim=3` and fails closed for capacity/API/dimension drift. General
cone/dimension support requires bundled Warp `contact_force` parity before it
can be activated. Reference hand primitive sampling is deterministic and uses
collision geoms, but should receive an independent exact sampled-mesh/primitive
review before training is considered. The next review should inspect that
collision sampling and direct per-contact helper parity, not start an optimizer.
