# Epistemic Record

## Current model

The COM-fixed object physics is sufficient for successful lift under exact Gym control, but the production policy fails in closed loop. On identical COM-fixed, hand-friction-3 physics, exact Gym control reaches `0.2859 m` lift with `0.03736 m` movement position RMSE, while the production closed loop reaches only `0.0261 m` with `0.1818 m` position RMSE and `1.526 rad` orientation RMSE. The remaining failure is therefore observation→policy feedback, not an inability of the object/contact physics to execute a successful trajectory (OPS, `2026-07-23T04:44:18Z`).

The network and checkpoint mapping are correct: offline replay matches stored means at approximately `1.5e-7` RMSE. Counterfactual slice replacement localizes policy divergence to physical feedback. Point cloud explains `53.7%` of policy error at residual onset (decisions 191–220); contact explains `65.4%` before movement (191–260), including `31.7%` from contact magnitude; during early lift, point cloud and object state explain `24.7%` and `21.0%`. These channels encode contact and relative hand/object pose (OPS, slice SHA `442c4dc...`).

Point-cloud generation itself is ruled out. Canonical Gym and MuJoCo local samples agree within `6.46e-9 m` nearest-neighbor and `3.73e-9 m` indexwise. Policy-visible point-cloud divergence is caused by different physical hand/object poses upstream of sampling (OPS, diagnostic SHA `c6fca588...`).

The live upstream mechanism is hand-response mismatch. Under the same exact Gym controls, MuJoCo tracks finger targets substantially more tightly than Gym: full finger target RMSE `0.0432` versus `0.0920 rad`, and movement RMSE `0.0530` versus `0.1171 rad`. Early-lift cross-backend finger qpos RMSE is `0.0473 rad`, with the thumb dominant. MuJoCo is too stiff relative to Gym, so fingers occupy different poses, engage different contact bodies and forces, and generate a different hand-frame point cloud; those observations then drive the policy onto a failing trajectory (OPS, hand-response SHA `58c3f480...`).

## Ruled out

- Wrong object inertial origin caused the earlier open-loop rotation/ejection, but preserving the 7.48 mm URDF COM offset corrected it and enabled the successful exact-control lift.
- Checkpoint loading, observation normalization, network mapping, and policy execution do not explain current policy-mean divergence; offline replay is numerically exact.
- Point-cloud sampling order or geometry does not explain the mismatch; the canonical local sampling contract is effectively exact.
- Weak MuJoCo finger actuation is contradicted by its lower target-tracking error. The relevant mismatch is excess proportional response or another mechanism that produces similarly over-tight tracking.

## Preregistered next intervention

Run exact Gym control, COM fixed, hand friction `3 0.01 0.001`, `substeps=10`, and `finger_kv=0`; change only `finger_kp` from `15` to `7.5`. Evaluate decisions 62–330 in the visible `env=1` run.

Accept proportional stiffness as the mechanism only if all three predictions hold:

1. Early-lift MuJoCo finger target RMSE enters `[0.06, 0.10] rad`.
2. Early-lift cross-backend finger qpos RMSE improves at least 25%, from `0.0473` to `<=0.0355 rad`.
3. Maximum lift through frame 330 remains `>=0.10 m`.

If target tracking worsens without cross-backend qpos improvement, reject proportional stiffness and investigate damping/contact compliance as the mechanism. The decisive quantity is cross-backend pose alignment, not target-tracking error alone.

Evidence provenance: `OPS.md`.
