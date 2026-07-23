# Epistemic Record

## Current model

Contact feedback is causally sufficient for checkpoint-policy grasp and lift. In a full visible actual-checkpoint closed loop, substituting only the three aligned Gym contact slices reduced policy-mean RMSE over decisions 191–260 from `0.26856` to `0.07153` (`73.37%`) and raised maximum lift from the baseline `0.0261 m` to `0.28963 m`. All three slices were bit-exact while policy actions and targets remained live. This passes both preregistered criteria and converts the earlier offline attribution into a closed-loop causal result (OPS, `2026-07-23T05:30:00Z`).

The remaining failure is release feedback after successful lift. Maximum height occurs at frame `437`; object position error first exceeds `0.05 m` at frame `451` and `0.10 m` at frame `464`; angular speed peaks at `46.54 rad/s` at frame `458`, and linear speed peaks at `2.349 m/s` at frame `469`. Movement position/orientation RMSE are `0.207 m`/`1.404 rad`, and late position RMSE reaches `0.385 m`. The temporal ordering identifies ejection immediately after peak height, rather than a failure confined to final placement (OPS, `2026-07-23T05:30:00Z`).

The leading mechanism is counterfactual late contact replay hiding the live MuJoCo release state. Late applied finger targets remain tighter than Gym, especially thumb joints 1/2/4 with signed deltas `+0.050/+0.114/+0.139 rad`; late finger qpos RMSE is `0.0873 rad`, and policy/action RMSE is approximately `0.158`. The same actuators release correctly when driven by exact Gym controls, so the general `50` force limit does not by itself produce the failure. Once the MuJoCo state diverges, prerecorded Gym contacts report forces from a different physical trajectory, preventing the policy from responding to the actual force/release condition. A phase switch from replayed to live contact at decision 441 directly tests this mechanism (OPS, `2026-07-23T05:30:00Z`).

Point cloud and object state remain secondary live mechanisms. Their values encode the diverged physical state and may continue to command bad release behavior even after live contact is restored. They become the next inspection target only if the preregistered contact phase switch fails to suppress ejection.

## Ruled out

- Wrong object inertial origin caused the earlier open-loop rotation/ejection. Preserving the `7.48 mm` URDF COM offset corrected it and enabled successful exact-control lift.
- Checkpoint loading, observation normalization, network mapping, and policy execution do not explain policy-mean divergence; offline replay is numerically exact.
- Point-cloud sampling order or geometry does not explain its mismatch; canonical local Gym and MuJoCo samples agree at nanometer scale.
- Contact feedback being insufficient for grasp/lift is ruled out by the live three-slice intervention: policy error fell `73.37%` and lift reached `0.28963 m`.
- A general actuator force limit of `50` is not sufficient to explain the release failure because exact Gym controls release correctly through the same MuJoCo actuators.
- Scalar finger proportional stiffness, direct Gym finger gains, direct Gym base-rotation gains, and the inverse-error `base_rot_kp=4000` estimate are rejected by prior diagnostics.
- The bad-`kp` ejection does not justify changing object/contact settings or softening `solref`; the production-gain control arm succeeds with those settings unchanged.

## Preregistered next intervention

Run the same visible `env=1` actual-checkpoint closed loop with preserved COM, hand friction `3 0.01 0.001`, `base_rot_kp=10000`, and all production defaults. Import the three aligned Gym contact slices only for decisions `62..440` inclusive:

- `contact_force_magnitude`
- `contact_force_directions`
- `object_contact_force_magnitude`

From decision `441` onward, use live MuJoCo contact observations. Preserve the live raw MuJoCo observation separately throughout. This is a phase-switch diagnostic and must not become a production dependency on Gym traces.

Predictions:

1. Pre-movement policy-mean RMSE reduction remains at least `60%`.
2. Maximum lift remains at least `0.25 m`.
3. After decision 440, maximum angular speed is at most `10 rad/s` and maximum linear speed is at most `0.8 m/s`.
4. Late position RMSE is at most `0.15 m`, and final position error is at most `0.10 m`.
5. If lift is preserved and ejection is suppressed, support the hidden-live-force/release-feedback mechanism.
6. If ejection persists, contact replay is not the sole late cause; inspect point-cloud and object-state feedback next.

Evidence provenance: `OPS.md`.
