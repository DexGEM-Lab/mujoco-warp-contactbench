# Epistemic Record

## Current model

The visible exact-control replay with hand friction `3 0.01 0.001` localized the object anomaly to the object inertial frame, not to torsional or rolling friction. The user's visual observation was that the grasp looked somewhat okay while the simulated object looked wrong; the object rotated/ejected during lift. The replay used exact Gym control arrays, status 0, `env=1`, `DISPLAY=:1`, and a live viewer.

Evidence artifacts:

- Trace: `outputs/dexhandrl_trace/v3_full631_mujoco_gym_control_handfriction3_slide_visible/mujoco_control_replay_full631_handfriction3_slide_visible.npz`, SHA256 `35ba55f0...`.
- Comparison: `outputs/dexhandrl_trace/v3_full631_mujoco_gym_control_handfriction3_slide_visible/compare_vs_gym.json`, SHA256 `9944fc01...`.
- Validation: `outputs/dexhandrl_trace/v3_full631_mujoco_gym_control_handfriction3_slide_visible/validation.json`, SHA256 `91dbc505...`.

## What the `3 0.01 0.001` comparison establishes

All numeric, control, and physics arrays are bit-exact to the prior `3 2 2` replay; only metadata differs. Therefore the torsional/rolling hypothesis is falsified: the contact `condim=3` means only sliding friction participates, while friction slots 2 and 3 are inactive. Rotation/ejection cannot have been caused by those coefficients.

The ManoRL cube1 comparison rules down mesh geometry. The URDF and mesh have identical bounds, centroid, volume, and faces. The source URDF inertial origin is `[0.005175592236, -0.005187185076, 0.001516276923]` m. The Gym URDF importer and ManoRL `_link_inertial` preserve this origin. In contrast, the DexHand scene `_object_body_xml` hardcodes inertial `pos="0 0 0"` while retaining inertia values about the source COM. This displaces the physical COM by 7.48 mm from the geometry and creates an incorrect contact torque. That mechanism directly explains the observed object rotation/ejection.

## Next preregistered intervention

Run one visible exact-control replay changing only object inertial-frame handling: preserve the URDF inertial origin, together with generic origin rotation and collision origin, while leaving mesh, mass, inertia, gains, substeps, solver, control arrays, and fixed friction `3 0.01 0.001` unchanged. Use `DISPLAY=:1`, `env=1`, `--no-headless`, and a viewer.

Pass criteria relative to the `3 2 2` baseline are orientation movement RMSE improving at least 30%, from `2.7247` to `<=1.9073` rad; position movement RMSE improving at least 20%, from `0.2892` to `<=0.2314` m; and maximum lift remaining `>=0.10` m. If this fails, inspect ManoRL contact compliance/integrator and object friction separately; do not wholesale-copy ManoRL.

Evidence provenance: `OPS.md`.
