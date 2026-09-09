# Multi-object scene runtime

## Objective

Correct the composite-scene adaptation so every object listed by the Lance row is physically materialized in each MJX world, while the unique `object_move` object remains the sole policy target and reward object. Re-run the policy-free `bowl,cuboid1` reference on DISPLAY=:1 and verify both bodies are present.

## Success criteria

- Preserve ordered scene-object names and initial poses from all `objects` entries at the selected source window; passive bodies then evolve under physics.
- Compute one scene-level ground shift so relative object transforms are preserved.
- Initialize every scene object in each world; park only object types absent from that world.
- Enable physical object-object collision while preserving hand/object and floor/object contacts.
- Keep active-object observations, target tracking, rewards, and policy action dimensions unchanged.
- DISPLAY=:1 replay visibly contains bowl and cuboid1 and completes the source trajectory.

## Constraints

- Work only in `feat/multi-object-scene-runtime`.
- Do not implement a visual-only passive object; both objects must participate in MJX physics.
- Preserve exact single-object behavior and trajectory package compatibility.
- Do not modify or clean unrelated files in primary dev.
