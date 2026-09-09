# One policy target, multiple physical objects

Modern composite Lance scenes must retain all physical bodies, even with one
`object_move` target. `bowl,cuboid1` captures include a supporting block; dropping
it or grounding the bowl independently changes the task.

Canonical contract and inspection command: `docs/manorl_composite_scenes.md`.
Composite initial states persist in trajectory package v2; v1 single-object
packages remain supported. Recompile pre-fix composite packages. Device contact
fast paths currently reject composite scenes because their contact aggregation
includes non-target bodies; the host-decoded MJX path retains target filtering.
