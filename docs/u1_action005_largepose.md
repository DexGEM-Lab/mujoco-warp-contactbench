# Action005 ten-parent strict-C1 large-pose campaign

## Deliverable

The action005 campaign uses all ten rows from the standardized complete-scene
Lance. Each row contributes exactly 16 accepted children, for 160 total. The
published four-action exact640 remains immutable; a separate exact800 dataset
interleaves the new action005 rows between actions003 and006.

Every action005 scene contains the ordered pair `mayonnaisebottle,bowl`. Exported
rows retain measured `qpos`, `qvel`, executed native-position controls, full
native contact-frame evidence, the physical parent teacher, and both parent
object tracks.

## Frozen physics

All parents and children use the U1 setting identified by
`c6db552f105d4de493880a52518772dbacad9496b105d2f4e43e87a907e692cd`:
120Hz arrival-indexed absolute 28-DoF targets, 480Hz physics, four substeps,
native position actuators, pyramidal contacts, `impratio=1`, CCD16, and capacities
1024/256/4096. There are no post-frame0 state writes, runtime feedback, hidden
PID state, attachment, weld, teleportation, or object-property changes.

## Parent qualification

Each standardized row is compiled into a pinned `model.mjb` bundle and replayed
in two processes under one frozen control array. Both replays must complete the
same semantics:

- opposing multi-finger airborne support;
- at least 100ms at or beyond 90 degrees of bottle tilt;
- peak pour above and within 14cm of the bowl;
- upright return, world support, and finger release;
- terminal 200ms p95 linear/angular speed below 2mm/s and 0.02rad/s;
- terminal position/orientation excursion below 0.5mm and 0.5 degrees.

Maximum terminal qvel remains reported, but does not decide settlement. Row7
showed why: one process emitted a single 3.83mm/s contact-solver spike while the
bottle moved only 0.031mm and 0.017 degrees over the complete terminal window.

The original row9 target was physically marginal. One process placed it upright;
the independent replay lost stability after final finger contact and settled at
89.7 degrees. The accepted row9 schedule repeats each existing source control
frame520 through560 twice. This halves terminal finger-withdrawal speed while
preserving every source control value and its order. Two diagnostic and two
formal processes then settled the bottle upright near 1.1 degrees. The original
failure and the intervention evidence remain in the staging diagnostics.

## Perturbation contract

The campaign preserves the four-action global 160-slot plan:

- 144 base cells: three radii (5/10/15cm), eight positive-elevation azimuth
  sectors, and six signed 30-degree wrist rotations;
- 16 deterministic extra draws;
- 16 fixed same-cell reserve candidates per slot;
- radius totals54/53/53 and 20 slots per sector.

A frozen assignment gives every parent exactly16 slots, two per sector, five or
six per radius, two or three per signed rotation, and one or two extra draws.
Each child prepends a 120-frame physical-teacher discrete-C1 approach and then
executes its complete parent target byte-for-byte. Prefix rejection uses solved
native hand-scene normal force above0.2N. Acceptance requires two physical
processes with identical frozen controls and distinct positive PIDs.

No exhausted slot may be replaced by a smaller perturbation, another cell, or a
padded failure.
