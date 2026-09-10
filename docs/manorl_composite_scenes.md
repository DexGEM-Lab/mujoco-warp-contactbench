# Composite scenes with one manipulation target

A capture can contain multiple physical objects while controlling only one.
For modern rows, `index.scene` supplies comma-separated object names in the
same order as `objects` when `trajectory_metadata.object_names` is absent.
The unique `trajectory_info.object_move` entry identifies the manipulation
target. That object alone determines action-pair identity, target tracking,
point-cloud conditioning, grasp mapping and reward.

All scene bodies are initialized at their captured poses at the start of the
selected padded window. A shared Z translation grounds the lowest collision
surface and moves the hand and all objects together. Objects are not grounded
independently: doing so would put a supported bowl inside its supporting block.
Composite decoding requires materialized collision assets for every body.

The physical environment uses a unified MJX-Warp model containing the union of
scene object types. For each world, bodies present in its capture are placed
in the workspace; only absent types are parked outside it. Object-object
collision is enabled for these composite models, retaining real collision
meshes, mass, inertia and gravity. Non-target objects are unactuated free
bodies, not scripted kinematic props. They may move under contact forces even
when their captured reference barely moves. Repeated instances of the same
object type in a row are unsupported and rejected.

Single-object decoding and default collision masks remain unchanged. Legacy
source-path identity/scene checks remain strict. Composite right-policy
`device_transition` keeps every contact in the observation and live-contact
counts, but filters its reward-only hand-object force through each world's
immutable target-geometry row. The rows permit `-1` padding for targets with
unequal collision-piece counts; a hand touching a passive scene object cannot
earn target-contact reward. The bimanual transition path applies the same
filter to both hands while retaining the right-hand observation interface.
Standalone `device_contact_decode` remains restricted to homogeneous
single-hand scenes because its compact output cannot satisfy the composite
host-snapshot diagnostics contract. `--device-resident-controls` is a separate
setting.

## Persisted packages

Composite packages use `manorl.trajectory_package.v2`. Their hashed trajectory
records include `scene_object_types`, `scene_object_initial_pos` and
`scene_object_initial_quat_xyzw`. Old readers reject v2 instead of silently
turning a scene into a single-object task. Current readers also accept v1
single-object packages. Recompile packages created before this scene support;
an old package cannot recover object states that were omitted at compilation.

## No-policy inspection

Use the interactive viewer without a checkpoint:

```bash
DISPLAY=:1 PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m sim.manorl.view_environment \
  --dataset-path /path/to/capture.lance --dataset-version 5 \
  --object bowl --gesture 03 --reference-fps 100 \
  --device gpu --hand-side right --num-envs 1 \
  --use_residual false --terminal false --speed 0.5 --loop
```

This applies reference hand commands through the physical controller. Both
objects evolve in the contact solver; it does not force their poses to follow
the recorded object trajectory. The target remains a reference for comparison.

For `dexgem_vla_demo_guangxue_cma2lance_20260909_060700.lance` v5, row 0 has scene
`bowl,cuboid1`. At the padded initial frame, scene grounding shifts all bodies
by about +0.32 mm; the model reports bowl–cuboid1 contacts and the interactive
viewer shows the supporting block beneath the bowl.
