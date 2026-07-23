# Epistemic Record

## Current model

The phase-130 checkpoint uses Gym's v3 live task clock: physical initialization is frame 61, policy decisions are frames 62–191, and post-step states are frames 63–192. The three exact artifacts are:

- Gym trace: `outputs/dexhandrl_trace/v3_phase130_gym/gym_v3_phase130.npz`, SHA256 `f04c21efff3d6e0dba55cf4cf15977906e975161ca95c74081ed53a36d4f5def`.
- MuJoCo trace: `outputs/dexhandrl_trace/v3_phase130_mujoco/mujoco_v3_phase130.npz`, SHA256 `457df82270622391af0bc320ea6e1da03899c2aa36dceec12f9cbdbc2bad4441`.
- Comparison: `outputs/dexhandrl_trace/v3_phase130_gym/compare_vs_mujoco.json`, SHA256 `e776bbde3562cf04ea1e9c95d8e9feb6efab8416a1883d0c6ece1f0bd723e528`.

All 130 `reference_active_target` rows (decisions 62–191) are exact across backends. Both corrected targets equal the reference through decision 190; the residual begins at decision 191, with cross-backend corrected-target maximum `0.00134665` (the measured value is `0.0013466477394104004`).

## Live mechanism

The first state-independent control-contract gap is Gym's post-filter target. Gym's corrected-to-applied active target becomes nonzero at decision 109 (row/index 47), before the residual at decision 191, and its overall corrected-to-applied maximum is `0.317322` (measured `0.3173224925994873`). MuJoCo has `applied_active_target == corrected_active_target`; therefore cross-backend applied targets first diverge at decision 109. At decision 191, Gym's filter correction reaches `0.0113177` and the cross-backend applied-target maximum is `0.0125070`.

The separate `full_target` mismatch starts at decision 62 because upstream physical/current-state-dependent routing selects different distal targets; it is not evidence that the active reference or corrected target contract is wrong. Full-target divergence must remain a separate physics/routing metric.

## Live intervention

Implement Gym's exact active-space ordered post-filters: `velocity_clamp` followed by `position_clamp`. Use Gym source parameters and `control_dt`: per-step maximum deltas are base xyz `0.03`, base rpy `0.1`, and fingers `0.02`; position limits are the active primary-DOF limits. Initialize the previous applied target from the reset frame-61 full target, matching Gym `reset_targets` before reset physics. This ordering and initialization are part of the control contract.

## Pre-registered prediction

After the exact filters, `applied_active_target` must be exact across backends through decision 190, moving its first divergence from 109 to 191. At decision 191, applied-target cross-backend maximum mismatch must be `<= 0.0013467`. Failure means previous-target initialization or active limits differ from Gym; trace those before changing physics. Full target is not expected to become exact because routing preserves different physical state.

Evidence provenance: `OPS.md`.
