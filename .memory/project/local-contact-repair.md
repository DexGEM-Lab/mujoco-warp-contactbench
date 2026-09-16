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

B035 remains explicitly unresolved in full orientation: rotation aligns with the
two-finger pinch line while wrist/reference rotation agrees. Two finger rays do
not establish rotational force closure. Bounded middle-finger IK failed to add
contact, and a visually plausible static support pose did not necessarily survive
continuous dynamic acquisition. Do not publish kinematic fit residuals as task
success or inherit the candidate's label without final physical replay.
