# Objective
Produce prefix-only ManoRL augmentation for five pairs: banana:02, banana:18, bowl:04, cylinder6:03, cylinder6:14.

The original `cylinder6:09` Far/Near tasks were explicitly retired after cross-parent, cross-cell zero yield. Preserve their failure sidecars and zero-row evidence, but never resume them. Replace their 250 Far + 150 Near slots with a separately source-bound and digest-bound `cylinder6:14` plan; do not mutate the original plan.

# Selection
For each pair select five source/accepted-parent trajectories. Prefer greater raw-reference right-wrist-to-initial-object distance at canonical pre60 frame 0, while maximizing coverage of relative wrist XYZ/direction rather than taking five clustered maxima.

# Yield targets
Per pair, across the five selected parents:
- Near: 150 accepted rows total.
- Far: 250 accepted rows total.

Total target remains 2,000 bounded slots before operational exclusions: the original `banana_02_1251/Far` exclusion removes 50 slots, while `cylinder6:14` replaces the retired `cylinder6:09` 400 slots. Failed candidates are never saved and shortfall must be reported honestly.

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
