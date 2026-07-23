# Epistemic Record

## Current model

The object anomaly was caused by a wrong inertial-frame origin in the generated MuJoCo object body. The source URDF COM offset is `[0.005175592236, -0.005187185076, 0.001516276923]` m (7.48 mm from the geometry origin), while the previous scene conversion hardcoded the inertial position to zero but retained inertia values about the source COM. That displaced the physical COM relative to contact geometry and produced the observed rotation/ejection. The COM-fixed replay preserves the URDF inertial frame and confirms this mechanism.

## Accepted COM-fixed evidence

The exact-control visible trace has SHA256 prefix `a7f670f5...`; its comparison is `01ef742b...`, validation is `91dbc505...`, and metrics are `e919192c...`. It contains 631 finite rows and exact imported Gym control arrays.

Relative to the prior `3 0.01 0.001` replay, COM correction passes every preregistered criterion:

- Movement position RMSE: `0.2892 → 0.03736 m` (87.1% reduction), better than Gym `0.05017 m`.
- Movement orientation RMSE: `2.7247 → 0.4723 rad` (82.7% reduction).
- Maximum lift: `0.2859 m`, close to Gym `0.3087 m`.
- Movement hand-contact rows: `88 → 286`, above Gym `269`.
- Position error first exceeds 5 cm at frame `455`, versus Gym frame `422`.

This confirms the wrong 7.48 mm COM as the dominant object anomaly and validates the ManoRL-based diagnosis. The remaining exact-control gap is orientation RMSE `0.472 rad` versus Gym `0.226 rad`; grasp, lift, and translation are close. No new physics tuning has been performed.

## Production closed-loop preregistration

Run the full production checkpoint closed-loop evaluation with COM fixed and hand sliding friction `3 0.01 0.001`. Do not inject or replay a control trace. Use `DISPLAY=:1`, `env=1`, and `headless=false`; retain all other defaults.

Success requires maximum lift `>0.20 m` and movement position RMSE `<0.10 m`. Report orientation RMSE and policy/control divergence. If the run fails despite the open-loop COM-fixed success, the remaining mechanism is observation/policy closed-loop transfer rather than object contact physics.

Evidence provenance: `OPS.md`.
