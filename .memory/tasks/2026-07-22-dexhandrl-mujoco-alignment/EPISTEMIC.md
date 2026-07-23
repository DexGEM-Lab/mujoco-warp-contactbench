# Epistemic Record

## Current model

The visible canonical-friction replay establishes that hand-object friction is
causally important to grasp retention, but rejects the complete canonical MuJoCo
friction vector as the production setting. Increasing hand friction from
`0.5 0.01 0.001` to `3 2 2` substantially increases lift and contact duration,
yet it destabilizes object rotation, ejects the object, and worsens translational
tracking.

The replay ran at clean HEAD `1a3932c`, with `DISPLAY=:1`, `env=1`, and headless
disabled. `xwininfo` verified the live MuJoCo viewer. It contains 631 finite rows
and exact imported Gym control arrays.

Evidence artifacts:

- Trace: `outputs/dexhandrl_trace/v3_full631_mujoco_gym_control_handfriction3_visible/mujoco_control_replay_full631_handfriction3_visible.npz`, SHA256 `97a8fe12b995b3e1e4704fdcd0066f781bb7eb501b16fc7bf669471e1c4ea4b4`.
- Comparison: `outputs/dexhandrl_trace/v3_full631_mujoco_gym_control_handfriction3_visible/compare_vs_gym.json`, SHA256 `9944fc01e509ab92b5dba3b45176a67328c313c98b541d703a54b19f43598fab`.
- Validation: `outputs/dexhandrl_trace/v3_full631_mujoco_gym_control_handfriction3_visible/validation.json`, SHA256 `6c2c461ebc7c49b4e1a1eedbb8457502f2fed25dc680553c56b80cdc1d428ee8`.
- Metrics: `outputs/dexhandrl_trace/v3_full631_mujoco_gym_control_handfriction3_visible/friction_metrics.json`, SHA256 prefix `51ac2fd`.

## What the friction-3 replay establishes

Maximum lift increased from the exact-control baseline `0.00467 m` to `0.12935
m`, a gain of `0.12468 m`. Movement hand-contact rows increased from 53 to 88.
These changes under identical controls establish that contact friction materially
changes grasp retention and lift.

The result fails the preregistered success criteria. Movement object-position
RMSE worsened from `0.1810 m` to `0.2892 m`, and movement orientation RMSE
worsened from `0.119 rad` to `2.725 rad`. Object orientation first exceeded `1
rad` at frame 300, the object reached maximum height at frame 330 while nearing
a pi-radian flip, and it was then ejected and dropped. Lift improved but remained
below the required `0.20 m`; position RMSE also remained above the required
`0.10 m`.

## Live mechanism

Gym PhysX exposes scalar surface friction, whereas MuJoCo represents friction as
sliding, torsional, and rolling components. Setting MuJoCo to the canonical
`3 2 2` therefore changes three contact mechanisms at once. The increased lift
and contact count identify stronger sliding resistance as beneficial. The large,
early pi-radian rotation followed by ejection identifies excessive contact moment
as the dominant failure introduced by the large torsional and rolling terms.
The complete `3 2 2` vector is rejected as a production default.

## Next experiment preregistration

Run one exact-control experiment relative to `3 2 2`. Keep hand sliding friction
at `3`, and restore only torsional and rolling friction to the prior small values:
`3 0.01 0.001`. Hold the complete imported Gym control trace and every other
simulation setting fixed. Run visibly with `DISPLAY=:1`, `env=1`, and headless
disabled.

If torsional/rolling overconstraint caused the ejection, the replay must retain
maximum lift `>=0.10 m`, reduce movement orientation RMSE below `0.5 rad`, and
improve movement position RMSE below `0.18 m`. Lift below `0.10 m` means sliding
friction alone is insufficient. Orientation remaining above `1 rad` means the
destabilizing torque arises from collision geometry or normal forces rather than
MuJoCo's torsional/rolling friction dimensions. Do not tune any parameter after
the result until the anomaly is inspected.

Evidence provenance: `OPS.md`.
