# Current-A action002 approach pilot

`tools/pilot_atomic_grade_a_augmentation.py` tests27 fixed candidates: one per
current action002 A parent from the accepted compact534 v3 run. It does not
substitute ancestral parents, edit parent controls, move scene objects or run
160-slot production. Radii5/10/15cm have nine candidates each, directions use
eight azimuth-sector centers with30deg positive elevation, and rotations use
one intrinsic-XYZ wrist coordinate at±30deg. There are no reserve candidates.

## Prefix and references

The approach lasts120 frames and uses the existing physical-reference
`_discrete_c1_prefix` with the4cm smooth vertical arc. The teacher is the selected
parent's recorded physical hand state. Its old `reference` field belongs to an
ancestor (action002 starts at ancestor frame76), so copying that field as the
new approach destination would change the requested parent. Original ancestor
references stay in the frozen `parent.json`; the exported child reference maps
to its immediate current parent with `[0]*120 + range(parent_frames)`.

Every selected parent has `target[0] == recorded_qpos[0]`. The prefix endpoint
is `2*q0-q1` from its physical teacher, followed by the complete byte-identical
float32 parent target. Reference splice residual and control splice residual
are reported separately. This is physical-reference discrete-C1, not a claim
that the control stream or actual simulated velocity is exactly C1.

## Physics and evidence

The pilot calls the same `run_physics` as the accepted Grade runner: finite
`u1-table`, headroom contact2048/world,CCD2048/world,constraint8192, native28D
arrival-indexed targets at120Hz,480Hz physics,four substeps. All initial qvel
are zero as in the current parent Grade replay. Only initial wrist6D changes.
A read-only observer records qpos/qvel/ctrl, compact hand-object contacts, and
native contact geom IDs, positions, bases, friction, EFC addresses and pyramid
forces. A CPU FK mirror reconstructs21 hand points; it is never stepped and
cannot affect physics. Prefix hand-scene force is checked after every480Hz
substep, not just at120Hz frame boundaries.

`prepare` freezes parent metadata/arrays and27 candidates into six batch5 jobs
(the final batch has two real worlds and three non-counted padding worlds).
Run each batch once with `--pass-name generation` and once in another process
with `--pass-name verification`. Both passes run every fixed candidate,
including first-pass failures. Use `tools.replay_capacity_guard` for each run;
logs and guard receipts are mandatory. No hidden controller, retry or target
adjustment is performed between passes.

## Pilot acceptance

Each pass must have prefix hand-scene normal force<=0.2N, target-position error
against the extended immediate-parent trajectory<3cm, at least101 post-prefix
hand-target contact frames and final target rotation within35deg of the parent.
Executed controls must equal the frozen stream. Independent second-vs-first
whole-trajectory target-position error must also be<3cm. The first run becomes
the stored child; the second supplies genuine replay-fidelity evidence. These
are a parent-fidelity and repeatability screen, not an exhaustive new stirring
semantic classifier. A passing self-comparison is never counted as Grade A.

`collect` verifies hashes, plan identity, independent PIDs and guard receipts,
and exports all actual first-pass children to diagnostic `all_candidates.lance`.
`replay_a_candidates.lance` contains children whose independent replay is A;
that alone does not imply the parent motion was preserved. Only children passing
all frozen gates enter `accepted.lance`. If none pass, the accepted count is zero
and no accepted dataset is published. `summary.json` retains all distinctions
and rejection evidence; every exported field is compared with Lance readback.
Original534/800 inputs remain read-only.
