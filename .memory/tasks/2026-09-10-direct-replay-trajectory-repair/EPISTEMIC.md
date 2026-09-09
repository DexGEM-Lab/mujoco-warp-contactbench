# Current model

## Objective and supported boundary
Directly repair the human replay rather than train a policy. Source v5 has 59 rows; only representative bowl row49 and pitcher row31 have been physically characterized. No successful full-trajectory repair yet. The one-row object-only trial Lance is explicitly marked failed sustained hold; it must not be promoted as corrected production data.

## Mechanism
Objects in the existing replay are free bodies after initialization, while captured hand commands drive position actuators. Later object-reference edits do not control physics. Small initial object translations change acquisition but are insufficient: row49 baseline lifts 2.7cm; (+6,+12,0)mm reaches 17.3cm transiently, then drops. Static held-pose tests also drop the bowl, ruling out motion timing as the sole cause. Simple thumb preload does not stabilize the original grasp.

The original/retargeted pinch places some finger links deeply through the bowl wall. Native geometric contacts at the held pose include index link penetrations around 13mm; many contact normals push the bowl down or outward instead of opposing across the wall. A fingertip-only position fit reproduced intermediate-link penetration. A collision-aware fit removes those deep intersections, placing outer finger normals inward/upward; thumb must then be moved outward from inside the bowl to establish opposing preload. This is the current discriminating intervention. Native contact recomputation is geometry evidence, not MJX force telemetry.

## Constraints and interventions
No RL implementation inspected, no learned policy or physical parameter changes. Wrist/finger pose changes are task-level trajectory edits announced during continuation; their deltas are separately represented. Original source remains immutable. A support grasp may require a local wrist-pose adjustment, not merely changed finger angles. Preserve all failed trials rather than claim they worked.

## Next question
Does collision-aware opposing contact hold the freely initialized bowl for three seconds? If yes, transport that fixed object-relative grasp through acquisition/lift and test the complete replay. If no, inspect actual force directions and loss-of-contact geometry before another trajectory experiment.

See OPS.md for immutable evidence and output paths.
