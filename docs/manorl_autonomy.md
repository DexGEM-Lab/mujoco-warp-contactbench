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
Episodes terminate on trajectory horizon, a dropped object, or large path
divergence. Terminal telemetry distinguishes `task_success` (peak demonstrated
lift plus final path accuracy) from `horizon_reached`; no horizon is called a
successful grasp.

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
only executable in M2. ## M3 first formal learner

The first learner uses `RlGamesPPO` and skrl's done-aware GAE/bootstrap through an
ordinary single N=1 Gymnasium boundary over the v2 MJX-Warp environment. The
adapter reports the requested CPU/GPU device so wrapper, model and memory tensors
match. Terminal observations are recorded before an explicit reset; finite task
horizons are terminated, while rollout cuts before the horizon bootstrap through
canonical last values. This is an honest first-learner boundary, not a
vectorization claim. The deterministic
50-identity split is 40/5/5; `cube2_02_2833`, `cube2_02_2835`, and
`cube2_02_2837` are fixed in TRAIN because they were inspected during design.
The remaining 47 identities are seeded into 37 TRAIN, 5 validation, and 5 test.
The split digest is persisted in checkpoints and evaluation rejects package or
split mismatches.

```bash
WANDB_MODE=offline PYTHONPATH=. python tools/train_manorl_autonomy.py formaltrain \
  --wandb --updates 1 --rollouts 2 --total-transitions 2 \
  --identity-index 0 --checkpoint outputs/manorl/contact_conditioned_autonomy/m3_formalppo.pt

PYTHONPATH=. python tools/train_manorl_autonomy.py evaluate \
  --checkpoint outputs/manorl/contact_conditioned_autonomy/m3_formalppo.pt \
  --identity-index 0 --allow-train-eval --steps 4
```

The setting switch is `--setting contact-conditioned|state-only`; both retain
actual producer contact telemetry, while state-only removes reference contact
conditioning from its reward/observation path. W&B is enabled by default and
uses the local `WANDB_PROJECT`/`WANDB_ENTITY` policy; offline mode is explicit
for local smoke. The former handwritten PPO smoke is gone. The first learner
checkpoint stores model, optimizer, RNG, normalizer, source/package, split,
contract, seed and configuration provenance. Learned rollout metrics remain
physical diagnostics: peak/sustained airborne contact, path error, slip, release
and per-identity traces are required before any competence claim. A maximum lift
plus endpoint pose is not sufficient.

Old v1 checkpoints are incompatible with the v2 contract IDs. This milestone
provides a runnable formal learner and one-update evidence, not learned full-task
competence.
