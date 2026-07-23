# Epistemic Record

## Current model

The complete paired rollout covers 631 policy decisions 62–692 and post-step
states 63–693. Both artifacts are finite and end in a truncated final state:

- Gym: `outputs/dexhandrl_trace/v3_full631_gym/gym_v3_full631.npz`, SHA256
  `3fa200902b689f20664136c7988ab645a827b3f2cd6d32f79250ea657ca436ac`.
- MuJoCo: `outputs/dexhandrl_trace/v3_full631_mujoco/mujoco_v3_full631.npz`,
  SHA256 `c7fe9b357c7005eb4e74e915ff55e4ae77c5706a3e86cf443ee07542aaf136d4`.
- Comparison: `outputs/dexhandrl_trace/v3_full631_mujoco/compare_vs_gym.json`,
  SHA256 `7399bc9ffb3f5d24e9a236580ab240b430c90696ba6c6d3378df9c675ca6c150`.
- Metrics: `outputs/dexhandrl_trace/v3_full631_mujoco/full_metrics.json`,
  SHA256 `c385e7755fa65ee00a99065928ce0e5672c3bef34b000269b48a8a651e6c4e3c`.

The MuJoCo trace, comparison, and metrics exited 0. The Gym artifact and runner
completed, the environment closed, and the source checkout remained unchanged.
Python then segfaulted during Isaac Gym interpreter teardown. The artifact is
valid evidence, but a reproducible successful Gym CLI exit remains unresolved.

## What the full rollout establishes

`reference_active_target` is exact across all 631 decisions. Corrected and
applied active targets first diverge at decision 191, preserving the accepted
filter boundary.

MuJoCo tracks the free-space object trajectory adequately before movement:
states 63–260 have maximum object position error `0.008136 m` (8.14 mm), below
the preregistered `0.05 m` threshold. The failure begins with grasp and lift:

- During movement, MuJoCo object position RMSE/max are `0.197747/0.293853 m`,
  versus Gym `0.050173/0.087589 m`.
- MuJoCo maximum lift is `0.031762 m`, versus Gym `0.308728 m`.
- MuJoCo overall hand active-target RMSE is better at `0.030404 rad`, versus
  Gym `0.088467 rad`, despite MuJoCo failing to lift the object.
- During movement, hand-contact magnitude is active for 32 MuJoCo rows versus
  269 Gym rows. MuJoCo loses hand contact around frame 300 while the target
  continues rising.

The evidence rules out generic servo inability as the principal explanation.
The live mechanisms are grasp formation/retention in MuJoCo and closed-loop
policy/control divergence after decision 191. The current paired rollout cannot
separate them because each simulator follows its own state-conditioned policy,
offsets, and control targets after divergence.

## Exact-control replay preregistration

Run one no-tuning replay that forces all 631 Gym `full_target` rows and the Gym
offset/action-stage records into MuJoCo while MuJoCo physics evolves. Exact
reference and control arrays are required; approximation or regenerated policy
outputs would invalidate the discrimination.

Classify using MuJoCo's movement interval:

- **Control-dominant:** object RMSE `<0.10 m` and maximum lift `>0.20 m`.
- **Physics/contact-dominant:** object RMSE `>0.15 m` and maximum lift `<0.10 m`.
- **Mixed:** every other outcome.

A successful replay would show that divergent closed-loop policy/targets cause
the failed lift. Failure under the exact Gym control sequence would localize the
remaining gap to MuJoCo grasp/contact response. The mixed region preserves cases
where both mechanisms contribute.

All visual tests and final rollouts must use `env=1`, export `DISPLAY=:1`, and
run explicitly non-headless with the viewer enabled when supported. Numerical
trace collection may remain headless.

Evidence provenance: `OPS.md`.
