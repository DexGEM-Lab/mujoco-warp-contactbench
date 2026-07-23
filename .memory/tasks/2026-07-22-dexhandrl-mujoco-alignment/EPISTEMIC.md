# Epistemic Record

## Current model

The full exact-control replay resolves the main ambiguity in the failed MuJoCo
lift. MuJoCo fails even when driven by Gym's recorded controls, so divergent
closed-loop targets are not a sufficient explanation. The failure is
physics/contact-dominant under the preregistered thresholds, and the strongest
live mechanism is a direct hand-friction mismatch.

The replay at clean HEAD `679f1ef` covers all 631 decisions 62–692 and states
63–693, is finite, and ends in a truncated final state. All ten imported Gym
control arrays are bit-exact: `policy_mean`, `final_action`,
`base_offset_before`, `base_offset_after`, `joint_offset_before`,
`joint_offset_after`, `reference_active_target`, `corrected_active_target`,
`applied_active_target`, and `full_target`.

Evidence artifacts:

- Exact-control replay: `outputs/dexhandrl_trace/v3_full631_mujoco_gym_control/mujoco_control_replay_full631.npz`, SHA256 `3de271d6a8d652c235d07c58bcbd10f5a76e3b8e5cfbe68b38868f36aace6bf3`.
- Comparison: `outputs/dexhandrl_trace/v3_full631_mujoco_gym_control/compare_vs_gym.json`, SHA256 `7a5353101f3c6a360d5adf227fd363256199d24313d5e84585f94d075692d749`.
- Validation: `outputs/dexhandrl_trace/v3_full631_mujoco_gym_control/validation.json`, SHA256 `045291d1f15269b9d7a8b13dd3beca90399e64bdc1eab213809abb17cc924932`.
- Metrics: `outputs/dexhandrl_trace/v3_full631_mujoco_gym_control/replay_metrics.json`, SHA256 `ed473ca0a22c4e21d1ea5dd072fe95d0379ea7f669d4d09a15b00e736aee3d87`.

## What the exact-control replay establishes

With exact Gym controls, MuJoCo movement object-position RMSE is `0.1809978787
m`, maximum lift is `0.0046744756 m`, and hand contact is active for 53 movement
rows. Closed-loop MuJoCo produced RMSE `0.1977472621 m`, maximum lift
`0.0317615978 m`, and 32 contact rows. Exact Gym control modestly reduces
position RMSE and increases the count of contact rows, but reduces lift by
`0.0270871222 m`.

The replay meets the preregistered physics/contact-dominant classification:
movement RMSE is greater than `0.15 m` and maximum lift is less than `0.10 m`.
It decisively fails the control-dominant requirements of RMSE below `0.10 m` and
lift above `0.20 m`. The object still slips despite bit-exact imported controls;
therefore closed-loop target divergence cannot by itself explain the failed
lift.

## Live grasp-retention mechanism

Gym loads the canonical hand MJCF default `<geom friction="3 2 2">`. MuJoCo
`sim/dexhandrl/scene.py` rewrites this hand default to `0.5 0.01 0.001`.
Object and ground friction remain separately set at `0.5`, so the mismatch is
specific to the hand geometry rather than a global scene-friction change.

The rewrite greatly reduces hand-object sliding friction. Its causal prediction
matches the observed failure: contact can form, but the object slips as the hand
target rises, contact is lost near frame 300, and lift stays negligible even
under the exact Gym target sequence. This source mismatch is now the principal
mechanism to test.

## Hand-friction experiment preregistration

Run one single-variable exact-control experiment. Restore only hand-geometry
friction to the canonical `3 2 2`. Hold object and ground friction, actuator
gains, substeps, contact settings, and the complete imported Gym control trace
fixed.

The friction mechanism is sufficient only if all three predictions hold:

1. hand contact is retained past frame 300;
2. maximum lift exceeds `0.20 m`;
3. movement object-position RMSE falls below `0.10 m`.

If maximum lift remains below `0.10 m`, hand friction alone is insufficient.
The next investigation must inspect collision geometry and contact compliance;
do not tune additional parameters.

The next visible run must use `DISPLAY=:1`, `--no-headless`, and `env=1`. The
viewer must mirror the simulation state without stepping physics, so enabling
visualization cannot change the replay trajectory.

Evidence provenance: `OPS.md`.
