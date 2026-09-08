# ManoRL autonomous cube2 M2 diagnostic

This route is a separate `manorl.autonomy.v2` contract. The legacy residual
route and checkpoints remain unchanged. M2 is a physical diagnostic milestone;
the PPO command is explicitly smoke-only until the canonical GAE/PPO adapter is
connected.

## Physical clock and alignment

The package reference/control clock is 120 Hz. `simulation_clock(120)` supplies
four 1/480-second MJX-Warp substeps per control transition, and `compile_model`
is called with that physics timestep. The route reads `object_pos_raw` and the
initial quaternion, rotates the pinned collision vertices, and computes one
support translation to `FLOOR_TOP_Z`. That same translation is applied once to
all reference hand XYZ and object positions, including reset. Raw package arrays
are never rewritten; source-relative geometry is consequently invariant.

## Contacts and reference intent

Reference hand q is forward-kinematics evaluated for every trajectory frame in
the compiled model. Reference surface proximity, nearest collision-mesh anchors,
and distance confidence are computed independently from the measured state.
Measured and reference states use `mj_geomDistance` between every hand segment
collision geom and every object collision geom; the nearest signed-distance
witness endpoint is retained in object-local coordinates with a declared
penetration/distance confidence. Solved hand-object forces, per-keypoint
hand-object forces, supporting net force and contact count come from the
canonical `MjxWarpPhysicalProducer`; table contacts are not used as hand-object
force. The direction-labeled net force is diagnostic telemetry, not a reward.
Geometry distance is an intent/slip proxy, never a measured force.

## Control and reward

All 28 DOFs are actor-owned from step zero. The actor emits normalized actions;
`rate_limited_command` integrates the measured previous command in actuator units
per second using the actual 1/120-second control period, physical limits, and a
reference-independent tracking-error envelope around measured q. Load-induced
servo error is retained; references are not added after actor output.

Reward terms are additive before movement: object path, hand-object relationship,
reference contact/anchor correspondence, measured hand-object contact, finger
configuration, object orientation, velocity tracking, and release. There is no
force-magnitude bonus for impacts. Stability is represented only by a
measured-contact relative-motion slip proxy, so arbitrary stasis does not receive
a stability bonus. Release starts
from demonstrated intent proximity falling after the final contact-intent frame.
Episodes terminate on trajectory completion, a dropped object, or large path
divergence and report the failure phase.

## Diagnostic commands

Use the content-addressed package (the path below is an example local checkout):

```bash
PYTHONPATH=. /home/jay/dexrobot/FromSSH/manoRL_mujoco/.venv/bin/python \
  tools/train_manorl_autonomy.py diagnostic --actor zero \
  --package /path/to/cube2_02_v295_f120_pre180_post180 \
  --trace outputs/manorl/contact_conditioned_autonomy/zero_hold_v2.json

PYTHONPATH=. /home/jay/dexrobot/FromSSH/manoRL_mujoco/.venv/bin/python \
  tools/train_manorl_autonomy.py diagnostic --actor reference-pursuit \
  --package /path/to/cube2_02_v295_f120_pre180_post180 \
  --trace outputs/manorl/contact_conditioned_autonomy/reference_pursuit_v2.json

PYTHONPATH=. /home/jay/dexrobot/FromSSH/manoRL_mujoco/.venv/bin/python \
  tools/train_manorl_autonomy.py package-summary \
  --package /path/to/cube2_02_v295_f120_pre180_post180 \
  --output outputs/manorl/contact_conditioned_autonomy/cube2_02_summary_v2.json
```

Both diagnostics start from the first package frame. The reference-pursuit
actor is explicitly labeled and consumes q_ref before emitting its action; no
reference is present in the command map after that output. Traces contain full
state/target, command, solved force, supporting force, anchors, per-term reward,
path/lift errors and failure phase. The package summary is deterministic for all
50 identities and records raw object extents for future deployment selection.

The former standalone PPO smoke command has been removed because it lacked
GAE/bootstrap and was not a valid training interface. The diagnostic CLI is the
only executable in M2. The next milestone must connect the canonical GAE/PPO
implementation and formal identity splits before any training. Old v1
checkpoints are incompatible with the v2 contract IDs. This milestone provides
runnable physical evidence, not learned competence.
