# Epistemic Record

## Current model

The v3 trace schedule is settled on Gym's live task clock. Lance initializes the physical hand/object state at frame 61. Gym reset performs its second observation pass through `post_physics_step()`, so the policy receives decision frame 62. Each trace row therefore records a policy decision in frames 62–692 and its post-step state in frames 63–693: 631 actions in the inclusive physical window 61–693. The reset marker belongs only to decision 62. Both producers enforce this contract; Gym reads and verifies `task_states["current_frame_idx"]` at the decision and after `env_step`, while MuJoCo initializes frame 61 and exposes frame 62 before its first action.

Episode time is policy action-count time: zero at reset, then `n / 633` for action index `n`. It is not derived from the logical Lance frame.

The exact v3 artifacts are:

- `outputs/dexhandrl_trace/v3_mujoco_smoke/mujoco_v3_smoke.npz` — SHA256 `9319b846804223e8d35168f6616627acc18b8d6ed7566e9f98458bf25fa60174`.
- `outputs/dexhandrl_trace/v3_gym_smoke/gym_v3_smoke.npz` — SHA256 `948ff17aeabd1fdabad5f3608db26607a4661eef89b38f2f47e804de9bfb8430`.
- `outputs/dexhandrl_trace/v3_gym_smoke/compare_v3_vs_mujoco.json` — SHA256 `cc144abd55b60784dd00e8813ebaffc8d401dcdbe4d7cc1d03dd9b9548647822`.

Both traces validate decisions 62–64, states 63–65, the reset marker, episode time `[0, 1/633, 2/633]`, finite values, and `env1`. Explicit environment close prevents the Gym shutdown segfault and preserves source/output integrity.

The comparator settles the live mechanism. Reference/corrected/applied active target, target pose/lookahead, episode time, and offsets match exactly, yet the first processed-observation divergence is at frame 62. At reset Gym's object has `v_z=-0.0981` while MuJoCo has `v_z=0`; every later row retains exactly one gravity-step difference. Object z differs by `-0.0007358` at the first row. Active-finger normalized difference reaches `0.402` across the three rows, contacts differ, and full-target difference reaches `0.00913` despite exact active-target agreement.

The causal account is a reset temporal-phase mismatch: Gym reset advances one PhysX control step after teleport, while MuJoCo with `settle_frames=0` does not. The exact active target remains aligned because it is upstream of the evolved-state-dependent distal routing; the active-finger, contacts, and full-target discrepancies arise after that one-step state evolution.

## Next discriminating experiment

Prediction was pre-registered before the experiment runs: a diagnostic MuJoCo reset with `settle_frames=1` should make the reset object-velocity mismatch approximately zero, reduce first-row z error from `0.0007358` to `<=0.0003`, and reduce active-finger and full-target divergence because state-preserving distal routing starts from an evolved state. If velocity fails to align, or hand/full-target metrics fail to shrink, the temporal-phase account is falsified for those residuals; the remaining mechanism is then integrator/contact/model behavior rather than reset timing.
