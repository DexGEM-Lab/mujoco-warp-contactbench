# Epistemic Record

## Final causal model

The native Gym checkpoint does not transfer behaviorally from Isaac Gym to MuJoCo by replaying its 22-dimensional action alone. The visible action-only trace imports Gym `final_action` exactly while retaining the local policy mean as a counterfactual diagnostic (RMSE `0.29145`). Its locally computed applied active targets match Gym to numerical precision (overall RMSE `2.52e-9`), yet its full 26-DoF finger targets differ by `0.06260 rad` RMSE. The resulting rollout lifts the object `0.27726 m` but fails the task: movement position RMSE is `0.31585 m`, post-decision-190 angular speed reaches `26.21 rad/s`, and late/final position error is `0.51252 m` (OPS, `2026-07-23T06:31:42Z`). Gym action is therefore an under-specified cross-simulator control interface.

The causal mechanism is plant-dependent distal routing. Both the Gym source and the MuJoCo implementation compute the same full-target mapping `R(active target, current physical distal qpos)`, and their mapping formulas agree. Because PhysX and MuJoCo produce different joint response and contact evolution, the current distal qpos supplied to `R` differs. The same 22-dimensional active target consequently expands into different 26-dimensional full finger targets, which then changes contact feedback and subsequent policy state. Routed-joint traces make this state difference concrete: MuJoCo often tracks middle/ring joint 3 nearly exactly while Gym lags by approximately `0.02–0.04 rad`.

Exact full-control, source-conditioned replay succeeds on the same corrected MuJoCo plant, with movement position RMSE `0.03736 m` and maximum lift `0.2859 m`. This demonstrates that the object/contact plant can execute the source trajectory when the missing routed target is supplied. The action-only failure is therefore caused by omitted routing state/full control rather than action indexing, target ordering, or an inability of MuJoCo to realize the grasp.

Offline routing alternatives do not justify changing semantics. Using current physical state gives full-target RMSE `0.05491`; previous-target recurrence worsens it to `0.05705`; a stateless/raw-reference construction gives `0.05730`. A semantic rewrite is rejected without evidence that it represents the source controller.

## Closed branches

- Action/checkpoint indexing, normalization, network mapping, and policy execution.
- Reference, residual, cumulative-offset, and active-space postfilter formulas.
- Full-target mapping formula mismatch.
- General MuJoCo inability to execute the trajectory.
- Scalar finger gain transfer and scalar base-rotation gain search.
- Blind observation-slice replay as a parity solution.
- Previous-target recurrence and stateless routing rewrites.

## Repository status

**Migration candidate / behavioral parity pending.** The expected simulator gap is numerically characterized and acceptable as a known migration boundary. Exact-control source-conditioned replay is validated. Native checkpoint grasp-lift-place is not behaviorally accepted and remains a known limitation.

A principled production path requires either (1) an adapter that supplies Gym-equivalent routing state/full routed targets or (2) policy adaptation/retraining against the MuJoCo plant. Neither belongs silently inside checkpoint parity because either changes the deployed control contract or the policy.

Evidence provenance: `OPS.md`.
