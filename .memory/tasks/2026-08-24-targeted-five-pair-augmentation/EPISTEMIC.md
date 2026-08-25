# Current model

## Parent qualification has two distinct layers
`manorl_accepted_synthetic_parent_v3` proves a row can supply source identity, raw hand start, object offset, movement_end+15 mapping, and late-contact geometry. It does not prove that the historical row passes the newer persisted-float32 three-rule gate.

Every production parent must pass both:
1. current parent-row gate: completion evidence from success-only parent provenance, final XYZ mean rotation <=35 deg, and >=101 persisted right-hand/object contact frames;
2. v3 descriptor eligibility: solved contact reaches at least movement end and movement_end+15/source mapping plus nonzero later horizontal geometry remain valid.

A real GPU counterexample established the distinction. Historical descriptor `banana_02_1256` produced 12 Far and 12 Near candidates in one fixed coverage cell. Every candidate completed, had 226–237 contact frames, and had no prefix collision; all failed only final rotation at 46.03–56.27 deg. The old 15-parent/1,200-target plan is diagnostic-only and must not be produced.

Historical current-gate audit leaves enough candidates to reselect existing pairs: banana:02 25/32, banana:18 9/29, bowl:04 47/48. The replacement gate-qualified plan has digest `1dc8714355876c428a3715559cd02e5c96f6edc56fde8c001a2106a42d2b254b`.

## Cylinder6 source and policy evidence
Clean v530 provides 50 trajectories each for cylinder6:03 and cylinder6:09; canonical pre60 and pre180 packages/predecoded bundles contain all 100 with zero source rejection. Explicit policy transfer successfully runs both actions to reason 1.

Zero-offset seed42 all100 bootstrap isolates different failure mechanisms:
- action03: rotation is usually accurate (median 5.27 deg), but contact is too short (median 85.5 frames); four rows pass the atomic gate and only `cylinder6_03_3951` remains descriptor-eligible.
- action09: contact is ample (median 360 frames), but final rotation is wrong (median 89.04 deg); only `cylinder6_09_4040` passes and remains descriptor-eligible.

The established 2 cm XY variation at seed43 changes the attraction basin rather than improving one fixed direction. Atomic-gate yield is one action03 and four action09 rows. Descriptor eligibility retains `cylinder6_03_3945` and action09 identities `cylinder6_09_4013`, `cylinder6_09_4043`, `cylinder6_09_4069`; the fourth action09 gate pass loses contact before movement end. Across seed42+43, current distinct eligible counts are action03=2 and action09=4, below the required 5+5.

Successful seed43 XY offsets span multiple quadrants, so no single offset direction explains the intervention. Seed44 repeats the same symmetric XY02 distribution as an independent replication. Prediction: it should yield a small set of different source identities if XY perturbation genuinely exposes nearby policy basins. Stop when merged descriptor sets reach five distinct identities per action; if action03 remains below five, inspect late-contact dynamics before any further seed.

## Multi-round identity rule
Different bootstrap rounds may produce multiple eligible parents for the same source. They are never silently overwritten. The selected variant maximizes the weakest normalized margin among rotation headroom to 35 deg, contact headroom above 100 frames, and late-contact headroom through movement_end+15; all variants remain auditable.

## Final augmentation mechanism
For each gate-qualified selected parent, Far has 50 fixed spatial slots (5 radius x 5 azimuth x 2 height) and Near has 30 empirical slots (5 distance x 3 azimuth x 2 height), each with 12 same-cell fallback seeds. Failures cannot cross cells. The runner preserves the complete pre60 source tail, uses no retreat, applies the atomic gate plus prefix collision gate before Lance append, and journals pending writes for deterministic recovery.
