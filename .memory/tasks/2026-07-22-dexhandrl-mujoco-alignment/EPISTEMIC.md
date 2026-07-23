# Epistemic Record

## Current model

The COM-fixed object physics is sufficient for successful lift under exact Gym control, but the production policy fails because simulator-specific physical response corrupts its feedback. On identical COM-fixed, hand-friction-3 physics, exact Gym control reaches `0.2859 m` lift with `0.03736 m` movement position RMSE, while production closed loop reaches `0.0261 m` with `0.1818 m` position RMSE and `1.526 rad` orientation RMSE. Offline network replay matches stored policy means at approximately `1.5e-7` RMSE, so checkpoint mapping and policy execution are correct. The remaining causal chain is actuator/contact response → hand/object-relative observations → policy divergence (OPS, `2026-07-23T04:44:18Z`).

Base-rotation response remains the strongest upstream candidate, but direct source gain transfer is falsified. Gym source values `5000/500/200` worsen onset base-rotation cross-backend RMSE from the MuJoCo baseline's `0.04029` to `0.12091 rad`, point-cloud RMSE from `0.006472` to `0.02058 m`, and lift through frame 331 from `0.1434` to `0.00388 m`. The associated target errors and maximum qvel identify the mechanism: Gym reaches `0.06093 rad` target RMSE with qvel `1.083`; MuJoCo baseline `10000/125/100` reaches `0.02428 rad` with qvel `2.572`; MuJoCo with source values reaches `0.16621 rad` with qvel `4.924`. Thus the source group drives MuJoCo into simulator-specific overshoot and force saturation rather than reproducing Gym's lag (OPS, `2026-07-23T05:02:02Z`, trace SHA prefix `82833334`).

The next intervention therefore estimates a MuJoCo-native proportional gain from its measured response while preserving its baseline damping and force limit. Under an approximate inverse quasi-static error relation, `kp*=10000*(0.02428/0.06093)=3985`, rounded to `4000`. This is a falsifiable response fit, not a claim that actuator parameters transfer between simulators.

Point-cloud generation itself is exact: canonical Gym and MuJoCo local samples agree within `6.46e-9 m` nearest-neighbor and `3.73e-9 m` indexwise. Policy-visible point-cloud mismatch therefore measures upstream relative-pose error rather than a sampling defect. Point-cloud substitution removes `53.7%` of onset policy error, providing the downstream link from rotational response to policy divergence (OPS, diagnostic SHA prefix `c6fca588`).

## Ruled out

- Wrong object inertial origin caused the earlier open-loop rotation/ejection, but preserving the 7.48 mm URDF COM offset corrected it and enabled successful exact-control lift.
- Checkpoint loading, observation normalization, network mapping, and policy execution do not explain current policy-mean divergence; offline replay is numerically exact.
- Point-cloud sampling order or geometry does not explain the mismatch; the canonical local sampling contract is effectively exact.
- Scalar finger proportional stiffness is insufficient. Lowering `finger_kp` from `15` to `7.5` improved early-lift cross-backend finger qpos RMSE only `4.8%` (`0.04731→0.04504 rad`), worsened pregrasp cross-backend RMSE (`0.03648→0.05430 rad`), and reduced lift through frame 331 (`0.1434→0.1260 m`).
- Direct transfer of Gym finger gains `300/20` worsened active-finger and policy error.
- Direct transfer of Gym base-rotation stiffness/damping/effort `5000/500/200` is rejected. It worsened base-rotation and point-cloud alignment and nearly eliminated lift; response metrics show overshoot/saturation rather than matched lag.

## Preregistered next intervention

Run one visible `env=1` 269-row exact-control replay for decisions 62–330 with preserved COM, hand friction `3 0.01 0.001`, finger gains `15/0`, and all other settings fixed. Change only MuJoCo `base_rot_kp` from `10000` to `4000`; retain `base_rot_kv=125` and force limit `100`.

Accept the inverse-error estimate only if all predictions hold:

1. Onset MuJoCo base-rotation target RMSE lies within 25% of Gym: `[0.0457, 0.0762] rad`.
2. Onset base-rotation cross-backend RMSE is `<=0.02015 rad`.
3. Onset raw point-cloud RMSE is `<=0.00453 m`.
4. Maximum object lift through frame 331 remains `>=0.10 m`.

If the test fails, reject the inverse-error model and inspect force saturation and damping as separate mechanisms rather than performing a linear gain search.

Evidence provenance: `OPS.md`.
