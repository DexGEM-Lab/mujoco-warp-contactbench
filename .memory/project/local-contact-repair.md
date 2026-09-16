# Local physical contact repair

Canonical runner/recipe contract: `docs/local_contact_repair.md`. The Cheyingtong
120Hz selection lives in
`sim/manorl/task_assets/local_contact_repairs/cheyingtong_120hz/catalog.json`.
It binds the historical43 inputs, preserves Sunke capture identity, and uses the
Cheyingtong physical hand at120/480Hz. This is not the September15 capture task.

Changing cadence requires a fresh baseline: interpolate in executed physical
time, update derivatives and the wrist integral dt, and retain the original
native model arrays. Do not credit cadence-only changes to a grasp correction.

Absolute donor preload can completely mask changes to nominal finger targets.
Additional finger-control corrections must be applied once after that blend,
with separate nominal/command/actual fields. A qpos/qvel trace alone is not a
validated continuation checkpoint; physical prefix reexecution is supported.

A merely lifted object is not necessarily a reusable successful grasp. A
misoriented grasp can wedge at placement, producing large wrist error; late
release tweaks may not resolve that upstream geometry. B045 illustrates this.
Conversely, isolated approach-joint opening may leave proximal collision links
pushing the object; small whole-wrist clearance can avoid induced yaw (A049/A052).

A046/B200/B067 expose numerical contact sensitivity: matching commands, initial
state and parameters can diverge from~1e-10 at early GPU steps into task failure.
Reject lucky-repeat selection. Endpoint gates can also accept a slipped bowl
that lands on the stove; inspect actual unsupported no-contact intervals.
Respect actuator limits when fitting offsets (B067 index abduction is saturated).
For grasp transfer, use a verified early opposed-contact state before major
slippage, not a deformed midair pose. The selected donor is already unsupported:
low lift does not mean unloaded. This repaired B067 with4.41mm wrist /2.04deg
wrist /6.34deg finger targets and two independent complete replays.

B035 remains explicitly unresolved in full reference rotation. Its baseline
extra rotation aligns with the two-finger pinch line while wrist/reference
rotation agrees. The delivered early-grasp alignment (7.30mm/6.01deg wrist,
4deg finger targets) reduces opening-axis error to~10deg and full error to~22deg,
with continued holding in a separate2s audit. The original15deg full-rotation gate
still fails. Two finger rays do not establish rotational force closure.

Bounded middle-finger IK could not add the intended contact, and a static support
pose did not necessarily survive continuous dynamic acquisition. Larger15/20deg
wrist-alignment diagnostics can pass a transient endpoint or a later hold without
passing both under the original duration; they are not included in the micro
selection. Audit continuations keep the live controller/solver state and original
last-tick derivatives, and remain separate from normal trajectory duration.
