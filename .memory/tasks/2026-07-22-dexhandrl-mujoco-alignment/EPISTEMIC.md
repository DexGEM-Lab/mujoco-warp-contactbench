# Epistemic Record

## Current model

The v3 trace schedule follows Gym's live task clock. Lance initializes physical hand/object state at frame 61. Gym reset advances one control frame before its second observation pass exposes policy decision frame 62. Each trace row records a decision in frames 62–692 and its post-step state in frames 63–693; episode time is policy action-count time, zero at reset and then `n / 633`.

`settle_frames=1` is adopted as the MuJoCo default. It reduced reset object-velocity mismatch from `0.0980999991` to `7.45e-09`, first-row z mismatch from `0.0007357523` to `0.0002943054`, and improved processed/RMS observation and policy-mean agreement. It represents the reset temporal phase, not a hand-control repair.

Production MuJoCo retains `substeps=10` and finger `kp/kv=15/0`. Under settle1, a `substeps=2` diagnostic retained velocity alignment but worsened first-row object position to `0.0004905015` and degraded policy, RMS, full-target, contact, and hand-pose measures despite improving active-finger error. Exact Gym numeric finger gains `300/20` are also ruled out: the stable finite trace had active-finger max/RMSE `0.461604/0.194584` (4.8%/54.1% worse than settle1), policy RMSE `0.078093` (149% worse), and hand-qvel/data maxima `50.717/50.88`. Its 94% full-target-RMSE improvement is not physical/policy-state improvement. Raw gain values are non-transferable across PhysX and MuJoCo actuator semantics.

The observation contract is now affine, not saturated: Gym `_normalize_joint_positions` emits values outside `[-1, 1]`; MuJoCo must preserve those raw values while retaining denominator protection.

## Live mechanism

After reset-phase alignment, the remaining hand gap is actuator/contact/model response. Ring1/pinky drift is evidence of the mismatch, but direct transfer of Gym gains made the response worse rather than calibrating it. Full target can differ from decision 62 because its routing depends on simulator state; it must be assessed separately from active-target contract agreement.

## Next discriminating experiment

Before running it, register a fresh v3 Gym+MuJoCo trace pair with 130 actions, decisions 62–191, using production `settle_frames=1`, `substeps=10`, and finger `kp/kv=15/0`. Predict reference active target will match exactly for all rows. Predict corrected active target will match exactly through decision 190 and its first residual-driven mismatch will be decision 191. If Gym velocity or position clamps engage earlier, `applied_active_target` will expose the first control-contract divergence; implement that contract before more physics work. Report full-target divergence separately because state-dependent routing can diverge from decision 62.

Evidence provenance: `OPS.md`.
