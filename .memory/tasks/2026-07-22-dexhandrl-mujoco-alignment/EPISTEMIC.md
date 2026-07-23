# Epistemic Record

## Current model

The phase-130 checkpoint uses Gym's v3 live task clock: physical initialization is frame 61, policy decisions are frames 62–191, and post-step states are frames 63–192. The accepted filtered artifacts are:

- Gym trace: `outputs/dexhandrl_trace/v3_phase130_gym/gym_v3_phase130.npz`, SHA256 `f04c21efff3d6e0dba55cf4cf15977906e975161ca95c74081ed53a36d4f5def`.
- Filtered MuJoCo trace: `outputs/dexhandrl_trace/v3_phase130_mujoco_filtered/mujoco_v3_phase130.npz`, SHA256 `3532925b8745af6fac11966441cdc8955598d842244ad2ef7167b7e064779821`.
- Filtered comparison: `outputs/dexhandrl_trace/v3_phase130_mujoco_filtered/compare_vs_gym.json`, SHA256 `5047d8d92e69094424ead89a88ee70794135cacfdba089297e8587b8333e6c06`.

Both filtered commands exited 0 and the traces contain 130 rows. `reference_active_target` is exact on every row. Corrected targets first diverge at decision 191 with maximum `0.00134549`. Applied targets are bit/nearly exact through decision 190: decision 108 has residual numerical error `7.3e-12`, decisions 109 and 110 are exact including Gym's filter effects, and decision 190 has `9.2e-12`. Their first divergence moved from decision 109 to decision 191; the decision-191 maximum `0.00134549` is within the preregistered `0.0013467` bound. The post-filter prediction passed, so the ordered velocity/position filter contract and reset-target initialization are accepted.

## Alignment boundary

Identity, frame schedule, reference, episode time, RMS loading, policy execution, residual algorithm, post-filters, and mapping algorithm are aligned. Remaining shared-observation differences are physical state: active qpos, kinematics, object state/contact, and point cloud.

`full_target` still diverges from frame 62 because physical-state-dependent routing distributes distal targets differently. This does not falsify the accepted filter contract. A physics intervention is now permitted by the task constraint, but it must follow a demonstrated mechanism rather than precede the full-trace evidence.

## Full-trace preregistration

Generate and compare complete 631-action Gym and MuJoCo traces. Before seeing the result, the required invariants and measurements are:

- `reference_active_target` remains exact on all 631 rows.
- The first corrected-target and applied-target divergence remains decision 191.
- Quantify object position and orientation error relative to target.
- Quantify object lift, including maximum and final lift.
- Quantify hand target error, contact slices, and phase-local error, explicitly separating the interval before `movement_start=261` from the lift interval.

Prediction: if error stays low before decision 261 and rises during lift, grasp/contact dynamics dominate. If error exceeds `0.05` before decision 261, free-space servo/model mismatch dominates. No tuning is permitted before this result is recorded.

Evidence provenance: `OPS.md`.
