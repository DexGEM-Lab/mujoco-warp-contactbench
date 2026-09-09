# One policy target, multiple physical objects

Modern composite Lance scenes must retain all physical bodies, even with one
`object_move` target. `bowl,cuboid1` captures include a supporting block; dropping
it or grounding the bowl independently changes the task.

Canonical contract and inspection command: `docs/manorl_composite_scenes.md`.
Composite initial states persist in trajectory package v2; v1 single-object
packages remain supported. Recompile pre-fix composite packages.

The GPU device-transition reducer retains every physical contact for keypoint
force observations and per-world contact counts. Its reward-only hand-object
term instead uses the immutable, per-world `active_object_geom_ids` table: a
contact earns target reward only when its non-hand geom belongs to that world's
policy target object. Tables may contain `-1` padding for unequal collision
piece counts; entries must be members of the unified object-geom set. Composite
right-policy `device_transition` is therefore supported, including a passive
left hand. Standalone `device_contact_decode` remains restricted to homogeneous
single-hand scenes because it deliberately does not materialize the full host
contact snapshot required by its public diagnostics contract.

This corrects a device implementation path and does not change policy tensor or
checkpoint serialization ABI. A checkpoint trained with the former union-of-
objects reward remains loadable, but is not behaviorally equivalent to training
with target-filtered contact reward; resume composite training from a host-
correct lineage.
