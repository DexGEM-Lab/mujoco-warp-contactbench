# Epistemic Record

## Current model

Aligned Gym observation values are a major causal driver of checkpoint action divergence, but contact plus hand-frame point cloud are insufficient for stable trajectory completion. In the full visible actual-checkpoint intervention, importing those slices reduced policy-mean RMSE over decisions `421..470` from `0.313735` to `0.071386` (`77.246%`) and produced `0.311934 m` maximum lift. The rollout still became inaccurate and dynamically unstable: object-position error exceeded `0.05 m` at frame `403`, post-decision-190 angular and linear speeds reached `28.825 rad/s` and `1.1025 m/s`, and late position RMSE and final position error were both `0.2053 m`. Only two of seven preregistered criteria passed (OPS, `2026-07-23T06:08:10Z`).

The causal boundary is now narrower than observation mismatch. Exact full-control replay succeeds under the same preserved object COM, hand friction, default gains, and live MuJoCo physics, with movement position RMSE `0.03736 m` and maximum lift `0.2859 m`. Thus MuJoCo can execute the Gym trajectory when given the complete applied control. The unresolved gap lies between policy output and the full applied target or in the state-dependent physical evolution downstream of that target (OPS, `2026-07-23T06:08:10Z`).

The next discriminating intervention imports only Gym `final_action`. MuJoCo observations and every downstream stateful transformation remain live. This separates policy-action divergence from cumulative offsets, residual correction, postfilters, full-target routing, actuators, contacts, and physics. The local policy mean remains recorded as the counterfactual diagnostic, while the imported action is the sole intervention (OPS, `2026-07-23T06:08:10Z`).

## Ruled out

- Wrong object inertial origin caused the earlier open-loop rotation/ejection. Preserving the `7.48 mm` URDF COM offset corrected it and enabled successful exact-control lift.
- Checkpoint loading, observation normalization, network mapping, and policy execution do not explain stored policy-mean divergence; offline network replay is numerically faithful.
- Point-cloud sampling order or local geometry does not explain point-cloud mismatch; canonical Gym and MuJoCo samples agree at nanometer scale. The live difference encodes relative hand/object pose divergence.
- Observation mismatch being incidental is ruled out. Imported contact plus point-cloud values reduce late policy-mean RMSE by `77.246%` and restore strong lift.
- Contact plus point cloud being sufficient for trajectory stability is ruled out. Despite improved action agreement and `0.311934 m` lift, the intervention passes only two of seven criteria and ends with `0.2053 m` position error.
- General MuJoCo inability to realize the desired trajectory is ruled out at the tested COM/friction/default-gain setting: exact full-control replay achieves `0.03736 m` movement position RMSE and `0.2859 m` lift.
- Hidden late live contact force as the sole ejection cause, scalar finger proportional stiffness, direct Gym finger gains, direct Gym base-rotation gains, and the inverse-error `base_rot_kp=4000` estimate remain rejected by prior diagnostics.

## Preregistered next intervention

Run one complete 631-decision visible actual-checkpoint trace with `env=1`. Import only Gym `final_action` at aligned frames. Keep all of the following live and local to MuJoCo:

- observation generation;
- cumulative base and joint offsets;
- reference target and residual/correction processing;
- velocity and position postfilters;
- full-target construction and state-dependent routing;
- actuator, contact, and physics evolution.

Record local `policy_mean` alongside the imported `final_action`, with explicit metadata identifying an action-only intervention. Reject configurations that combine this mode with control-trace or observation-trace intervention.

If policy-action divergence is a sufficient cause of the remaining failure, predict:

1. Groupwise applied-active/full-target RMSE versus Gym is `<=0.02 rad`.
2. Movement object-position RMSE is `<=0.08 m`.
3. Maximum lift is `>=0.25 m`.
4. Maximum angular speed after decision 190 is `<=10 rad/s`.
5. Late object-position RMSE is `<=0.10 m`.
6. Final object-position error is `<=0.05 m`.

If the action-only intervention fails while exact full-control replay continues to pass, policy-action divergence is insufficient. The remaining mechanism lies in downstream state-dependent residual correction or actuator/contact evolution.

Evidence provenance: `OPS.md`.
