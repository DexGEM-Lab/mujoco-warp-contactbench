# Objective
Produce prefix-only ManoRL augmentation for four pairs: banana:02, banana:18, bowl:04, mayonnaisebottle:04.

All Cylinder6 actions are cancelled by the user. Never resume `cylinder6:03`, `cylinder6:09`, or the diagnostic `cylinder6:14` replacement. Preserve their accepted rows, failure sidecars, packages, parent descriptors, and smoke evidence as `publish=false`, but exclude every Cylinder6 row from final validation, merge, and publication. Mayonnaisebottle:04 replaces the cancelled Cylinder6 400-slot allocation under a separate immutable plan; do not mutate the original five-pair plan.

# Selection
For each pair select five source/accepted-parent trajectories. Prefer greater raw-reference right-wrist-to-initial-object distance at canonical pre60 frame 0, while maximizing coverage of relative wrist XYZ/direction rather than taking five clustered maxima.

# Yield targets
Per pair, across the five selected parents:
- Near: 150 accepted rows total.
- Far: 250 accepted rows total.

The retained original three-pair plan contains 1,200 bounded slots before operational exclusions. The original `banana_02_1251/Far` exclusion removes 50 slots, leaving 1,150 original-plan slots. The independent mayonnaisebottle:04 replacement contributes 400 slots, so aggregate operational target is 1,550 slots. Failed candidates are never saved and shortfall must be reported honestly.

# Production contract
- 120 Hz control/reference, 480 Hz physics x4.
- pre60, 4 cm approach.
- Far radius 0.30–1.00 m.
- Near uses movement_end+15 only for start sampling.
- Full original reference tail, no retreat.
- compact `synthetic_mano_target_replay_visual_v2_contact`.
- Atomic quality gate: reason 1, final XYZ mean <=35 deg, >=101 persisted right-hand/target-object frames >0.2 N, plus prefix collision gate.

# Parent screening
Parent bootstrap may use bounded 2 cm object-XY variation to batch-screen more policy outcomes. Screening artifacts are intermediate and `publish=false`. Select five different source identities per pair only after the atomic gate and accepted-parent late-contact/anchor eligibility, prioritizing quality margin, pre60 distance, and spatial coverage.

# Formal production XY contract
Formal Near/Far collection does not draw a new random object-XY offset (`object_init_xy_offset_range_m = 0`). It reuses each selected accepted parent's fixed object offset as part of the parent ABI. Screening-time randomness is therefore frozen into the selected parent descriptor rather than resampled during production.

# Infrastructure
Server1 only. Inspect GPU ownership before launching. Do not interrupt other users. Deploy exact committed code and explicit artifacts with SHA256.
