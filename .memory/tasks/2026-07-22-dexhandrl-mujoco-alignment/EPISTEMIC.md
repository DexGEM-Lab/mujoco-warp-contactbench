# Epistemic Record

## Current model

The v3 trace schedule is settled on Gym's live task clock. Lance initializes the physical hand/object state at frame 61. Gym reset performs its second observation pass through `post_physics_step()`, so the policy receives decision frame 62. Each trace row records a policy decision in frames 62–692 and its post-step state in frames 63–693: 631 actions in the inclusive physical window 61–693. The reset marker belongs only to decision 62. Both producers enforce this contract. Episode time is policy action-count time: zero at reset, then `n / 633` for action index `n`.

The exact v3 artifacts are:

- `outputs/dexhandrl_trace/v3_mujoco_smoke/mujoco_v3_smoke.npz` — SHA256 `9319b846804223e8d35168f6616627acc18b8d6ed7566e9f98458bf25fa60174`.
- `outputs/dexhandrl_trace/v3_gym_smoke/gym_v3_smoke.npz` — SHA256 `948ff17aeabd1fdabad5f3608db26607a4661eef89b38f2f47e804de9bfb8430`.
- `outputs/dexhandrl_trace/v3_gym_smoke/compare_v3_vs_mujoco.json` — SHA256 `cc144abd55b60784dd00e8813ebaffc8d401dcdbe4d7cc1d03dd9b9548647822`.

Reference/corrected/applied active target, target pose/lookahead, episode time, and offsets match exactly. The baseline first processed-observation divergence was a reset temporal-phase mismatch: Gym had already advanced one control frame after teleport, giving object `v_z=-0.0981`, while MuJoCo with `settle_frames=0` had not advanced and gave `v_z=0`.

The settle1 diagnostic supports this mechanism. Changing only MuJoCo reset to `settle_frames=1` reduced first-row object-linear-velocity mismatch from `0.0980999991` to `7.45e-09` and object-position mismatch from `0.0007357523` to `0.0002943054`. It also reduced processed-observation, RMS-observation, and policy-mean mismatch overall. Therefore `settle_frames=1` is the justified reset intervention for object state and policy behavior overall.

Reset timing does not explain the hand-state residual. With settle1, the three-row active-finger max/RMSE worsened from `0.4020402/0.0970351` to `0.4402826/0.1263067`; full-target max decreased slightly but RMSE worsened. Temporal phase is therefore supported for object/reset/policy overall and rejected as the explanation for hand divergence.

## Live mechanism

Gym uses 2 internal physics substeps per 0.01 s control frame, whereas MuJoCo uses 10. For a freely falling object, this integration subdivision difference exactly accounts for the residual settle1 first-row position mismatch while preserving the already aligned end-of-frame velocity. The same subdivision difference may alter hand actuator and contact evolution within each control frame.

## Next discriminating experiment

Run a diagnostic that changes only `settle_frames=1` and `substeps=2` relative to the production baseline; relative to settle1, only `substeps` changes from 10 to 2. Object velocity mismatch should remain below `1e-6`. If integration subdivision is the remaining object mechanism, first-row object-position mismatch should fall from `0.0002943054` to below `1e-6`.

For the hand mechanism, require at least 20% improvement versus settle1 in all three metrics: three-row active-finger max, active-finger RMSE, and policy-mean RMSE. If that threshold is not met, timing subdivision is not a material driver of hand divergence; the live hand mechanism becomes actuator/contact/model semantics.
