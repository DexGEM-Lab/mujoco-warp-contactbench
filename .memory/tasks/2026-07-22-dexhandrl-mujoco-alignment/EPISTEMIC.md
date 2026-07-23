# Epistemic Record

## Current model

The COM-fixed object physics is sufficient for successful lift under exact Gym control, but the production policy fails because physical response differences corrupt its feedback. On identical COM-fixed, hand-friction-3 physics, exact Gym control reaches `0.2859 m` lift with `0.03736 m` movement position RMSE, while production closed loop reaches `0.0261 m` with `0.1818 m` position RMSE and `1.526 rad` orientation RMSE. Offline network replay matches stored policy means at approximately `1.5e-7` RMSE, so checkpoint mapping and policy execution are correct. The remaining causal chain is simulator response → hand/object-relative observations → policy divergence (OPS, `2026-07-23T04:44:18Z`).

The strongest current upstream candidate is base-rotation actuator response. Gym uses base-rotation stiffness/damping/effort `5000/500/200`; MuJoCo uses `10000/125/100`. This mismatch is specific to rotation: base-position actuation already matches Gym at `5000/500/500`. During exact-control residual onset decisions 191–220, base-rotation cross-backend RMSE is `0.04029 rad` and raw point-cloud RMSE is `0.006472 m`. A roughly 40 mrad hand-frame rotation produces several millimeters of displacement at hand/object scale, matching the observed point-cloud error. Point-cloud substitution removes `53.7%` of onset policy error, providing the downstream link from rotational response to policy divergence (OPS, `2026-07-23T04:52:38Z`).

Point-cloud generation itself is exact: canonical Gym and MuJoCo local samples agree within `6.46e-9 m` nearest-neighbor and `3.73e-9 m` indexwise. Policy-visible point-cloud mismatch therefore measures upstream relative-pose error rather than a sampling defect (OPS, diagnostic SHA prefix `c6fca588`). Contact and object-state observations remain consequential later in the rollout, but the base-rotation mismatch offers a concrete source for the earliest relative-pose divergence.

## Ruled out

- Wrong object inertial origin caused the earlier open-loop rotation/ejection, but preserving the 7.48 mm URDF COM offset corrected it and enabled successful exact-control lift.
- Checkpoint loading, observation normalization, network mapping, and policy execution do not explain current policy-mean divergence; offline replay is numerically exact.
- Point-cloud sampling order or geometry does not explain the mismatch; the canonical local sampling contract is effectively exact.
- Scalar finger proportional stiffness is insufficient. Lowering `finger_kp` from `15` to `7.5` improved early-lift cross-backend finger qpos RMSE only `4.8%` (`0.04731→0.04504 rad`), worsened pregrasp cross-backend RMSE (`0.03648→0.05430 rad`), and reduced lift through frame 331 (`0.1434→0.1260 m`). Contact-magnitude RMSE improved (`0.01745→0.00796`), while contact direction stayed near `0.133` and object-force RMSE stayed `0.165`; changing scalar stiffness altered force magnitude without correcting pose or force direction (OPS, trace/validation/metrics SHA prefixes `ee2dfb32`/`f52c827d`/`dd8c4a14`).
- Direct transfer of Gym finger gains `300/20` is also rejected because it worsened active-finger and policy error. Numeric actuator gains are not generally portable across simulator semantics.

## Preregistered next intervention

Replay exact Gym controls for decisions 62–330 with COM fixed, hand friction `3 0.01 0.001`, finger gains `15/0`, and `substeps=10`. Change only the base-rotation actuator group from MuJoCo stiffness/damping/effort `10000/125/100` to Gym source values `5000/500/200`.

Accept the source base-rotation contract only if all three predictions hold:

1. Onset decisions 191–220 base-rotation cross-backend RMSE falls at least 50%, from `0.04029` to `<=0.02015 rad`.
2. Onset raw point-cloud RMSE falls at least 30%, from `0.006472` to `<=0.00453 m`.
3. Maximum object lift through frame 331 remains `>=0.10 m`.

If the direct source values fail, fit MuJoCo's base-rotation response to Gym rather than continuing to copy numeric gains. The decisive mechanism is matched rotational motion and its induced hand-frame point-cloud alignment, not equality of actuator parameter numbers.

Evidence provenance: `OPS.md`.
