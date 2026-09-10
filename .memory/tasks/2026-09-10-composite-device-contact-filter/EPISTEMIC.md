# Current model

The host and GPU paths now share the intended reward rule: a hand contact contributes to target-contact reward only when its non-hand geom belongs to the fixed policy target object for that world. Keypoint contact observations and live per-world contact counts remain reductions over all valid contacts.

`active_object_geom_ids` is static metadata captured when the JIT reducer is built. It supports padded unequal collision-piece sets (`-1` only) and validates membership in the unified physical object-geoms during tracing. Device transition is available for composite right-policy worlds; standalone device contact decode remains intentionally excluded for composite scenes.

Checkpoint ABI remains unchanged: no observation/action shape, serialized metadata schema, reward contract identifier, or environment contract identifier changed. A legacy checkpoint can load but training under the former union-of-objects fast-path reward is not behaviorally comparable to target-filtered training.

A parent CUDA/MJX run on the dataset-v5 bowl/cuboid1 row advanced host physics once and evaluated device transition on that exact snapshot for 260 steps (including reset step 240): reward, done, and counters matched, maximum observation error was 2.17e-7, and target-contact reward was nonzero. The prior independent-simulation qpos drift of 5e-5 identifies the solver as the source of that comparison noise, not transition arithmetic. The regression is therefore dataset-gated and explicitly shares the host snapshot.
