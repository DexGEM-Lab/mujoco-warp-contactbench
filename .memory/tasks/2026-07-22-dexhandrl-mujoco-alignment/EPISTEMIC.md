# Epistemic Record

## Current model

The v3 trace schedule is settled on Gym's live task clock. Lance initializes the physical hand/object state at frame 61. Gym reset performs its second observation pass through `post_physics_step()`, so the policy receives decision frame 62. Each trace row records a policy decision in frames 62–692 and its post-step state in frames 63–693. Episode time is policy action-count time: zero at reset, then `n / 633` for action index `n`.

Reference/corrected/applied active target, target pose/lookahead, episode time, and offsets match exactly. The baseline first processed-observation divergence was a reset temporal-phase mismatch: Gym had already advanced one control frame after teleport, giving object `v_z=-0.0981`, while MuJoCo with `settle_frames=0` had not advanced and gave `v_z=0`.

The settle1 diagnostic supports `settle_frames=1` as the reset intervention. It reduced first-row object-linear-velocity mismatch from `0.0980999991` to `7.45e-09`, object-position mismatch from `0.0007357523` to `0.0002943054`, and processed-observation, RMS-observation, and policy-mean mismatch overall. It did not repair the hand state: active-finger max/RMSE worsened to `0.4402826/0.1263067`.

Changing only MuJoCo subdivision from production `substeps=10` to `substeps=2` under settle1 is ruled out for production. The valid schema-v3 diagnostic preserved zero first-row velocity error, but object-position error worsened to `0.0004905015`, failing the `<1e-6` prediction. Active-finger max/RMSE improved 25.8%/35.4% to `0.3268136/0.0815956`, but policy-mean RMSE worsened 239% to `0.1061625`, RMS-observation RMSE worsened 86.8% to `0.1478189`, full-target RMSE worsened 13.6% to `0.0025172`, and contact-magnitude RMS plus hand-pose and point-cloud slices worsened sharply. The preregistered joint criterion therefore failed. Keep `substeps=10`; subdivision is not the policy-hand solution. The residual object-position mismatch is an integrator-semantics difference rather than a reason to change production subdivision. Evidence is recorded in `OPS.md`.

## Live mechanism

The dominant settle1 active-finger errors localize to ring1 and pinky-related drift. Gym normalized ring1 at rows 0–2 is `-0.0266/-0.0242/-0.0229`, while MuJoCo is `0.1648/0.3958/0.4174`. The source actuator contract differs by an order of magnitude: Gym finger stiffness/damping are `300/20`; MuJoCo finger `kp/kv` are `15/0`; both use force limit `50`. The live mechanism is therefore finger actuator response, not generic tuning. The causal prediction is specific: increasing finger gains should suppress the immediate ring1/pinky divergence and thereby improve active-finger and policy-mean agreement.

## Next discriminating experiment

Keep `settle_frames=1`, production `substeps=10`, and every other setting fixed. Change only finger `kp/kv` from MuJoCo `15/0` to Gym `300/20`.

Require a finite stable trace, at least 30% improvement versus settle1 in both active-finger maximum error and active-finger RMSE, and at least 20% improvement in policy-mean RMSE. Ring1/pinky divergence must specifically shrink. If the trace is unstable or fails these improvement thresholds, exact numeric gains are not transferable across actuator semantics; the next intervention is bounded one-step response calibration rather than blind tuning.
