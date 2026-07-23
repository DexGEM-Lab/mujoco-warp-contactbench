# Epistemic Record

## Current model

Aligned Gym contact feedback is causally sufficient for grasp formation and lift, but counterfactual late contact is only one contributor to ejection. The all-contact actual-checkpoint intervention reduced pre-movement policy-mean RMSE by `73.37%` and lifted the object `0.28963 m`. Switching from replayed Gym contacts to live MuJoCo contacts after decision 440 preserved both results, yet post-switch angular speed still reached `24.24 rad/s`, linear speed `2.126 m/s`, late position RMSE `0.24497 m`, and final error `0.24497 m`. The switch improved angular speed from `46.54 rad/s` and late RMSE from `0.385 m`, establishing a real contact-feedback contribution, but it did not eliminate the failure (OPS, `2026-07-23T05:45:00Z`).

The phase switch was late relative to the causal onset. During decisions `421..440`, live MuJoCo contact magnitude averaged `0.04766` while replayed Gym contact magnitude averaged `0.00515`, a `9.25×` mismatch. By decision 440, the finger target already differed from Gym by `0.0788 rad`. Thus the policy had already integrated counterfactual low-force evidence into a divergent command before live contact resumed. This explains the partial phase-switch improvement without making contact the sole late mechanism (OPS, `2026-07-23T05:45:00Z`).

Relative-pose feedback is now the dominant live policy-input mechanism around ejection. Offline checkpoint replay is numerically faithful (`1.58e-7` RMSE). Replacing only point cloud during decisions `421..440` reduces policy error by `63.06%`. During `441..470`, the reductions are point cloud `34.75%`, object state `22%`, velocity `14%`, and contact only `7.32%`. Because `object_point_cloud_hand_frame` expresses object pose relative to the hand, its divergence transmits the coupled hand/object physical-state mismatch back into policy actions even after contact becomes live. The next closed-loop intervention adds only this relative-pose channel to the already proven all-contact diagnostic (OPS, `2026-07-23T05:50:00Z`).

## Ruled out

- Wrong object inertial origin caused the earlier open-loop rotation/ejection. Preserving the `7.48 mm` URDF COM offset corrected it and enabled successful exact-control lift.
- Checkpoint loading, observation normalization, network mapping, and policy execution do not explain policy-mean divergence; offline replay is accurate to `1.58e-7` RMSE.
- Point-cloud sampling order or local geometry does not explain point-cloud mismatch; canonical Gym and MuJoCo samples agree at nanometer scale. The live point-cloud difference encodes relative hand/object pose divergence.
- Contact feedback being insufficient for grasp and lift is ruled out: both all-contact and phase-switch closed-loop runs preserve the `73.37%` pre-movement policy improvement and `0.28963 m` lift.
- Hidden late live contact force as the sole ejection cause is ruled out. Restoring live contact after decision 440 improves the ejection but leaves `24.24 rad/s` angular speed and `0.24497 m` final error.
- A general actuator force limit of `50` is not sufficient to explain release failure because exact Gym controls release correctly through the same MuJoCo actuators.
- Scalar finger proportional stiffness, direct Gym finger gains, direct Gym base-rotation gains, and the inverse-error `base_rot_kp=4000` estimate are rejected by prior diagnostics.
- The bad-`kp` ejection does not justify changing object/contact settings or softening `solref`; the production-gain control arm succeeds with those settings unchanged.

## Preregistered next intervention

Run one complete 631-decision visible actual-checkpoint closed loop with `env=1`, preserved COM, stable production defaults, and hand friction `3 0.01 0.001`. Starting from the proven all-contact diagnostic, make exactly one addition: import aligned Gym `object_point_cloud_hand_frame` for every row alongside the same three aligned Gym contact slices:

- `contact_force_magnitude`
- `contact_force_directions`
- `object_contact_force_magnitude`
- `object_point_cloud_hand_frame`

All policy actions, control targets, and physics remain live. This remains a diagnostic and must not create a production dependency on Gym traces.

Predictions:

1. Maximum lift remains `>=0.25 m`.
2. Policy-mean RMSE over decisions `421..470` improves by `>=40%` relative to the all-contact run.
3. The first object-position error above `0.05 m` is delayed beyond frame `575`.
4. After decision 190, maximum angular speed is `<=10 rad/s` and maximum linear speed is `<=0.8 m/s`.
5. Late position RMSE is `<=0.15 m`, and final position error is `<=0.10 m`.
6. If these criteria fail, contact plus point cloud are insufficient; object-state feedback or physical response dominates. Do not add further replay slices blindly.

Evidence provenance: `OPS.md`.
