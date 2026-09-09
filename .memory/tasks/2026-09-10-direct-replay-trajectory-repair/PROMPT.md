# Direct replay trajectory repair

## Objective
Directly inspect and repair object trajectories in `dexgem_vla_demo_guangxue_cma2lance_20260909_060700.lance` (version 5, 59 human demonstrations), using the existing dev replay and DISPLAY=:1. Start with stove-to-air bowl lifting, then bottle/pitcher interactions with a bowl.

## User constraints
- Parent model performs the work directly; no subagents authorized.
- No reinforcement learning, policy training, or reading RL implementation.
- Preserve source data. Save corrected object poses and comparison artifacts separately.
- Start with object-only corrections. The requested outcome is a repaired replay/object motion; the user did not explicitly prohibit local hand-trajectory edits. After object-only alignment failed to sustain contact and the user requested continuation, local finger closure edits were announced. Keep edits explicit and separately recorded; preserve the task's transport path where possible, and retain source timestamps, physical parameters and source data. A local wrist/grasp-pose repair was announced after static failure showed finger-only offsets insufficient.

## Success criteria
- Inspect actual replay/contact geometry, not merely logs.
- Produce corrected object trajectory data with exact source row/version provenance.
- Verify geometric plausibility and replay behavior. Distinguish constrained/kinematic object following from a free-body contact grasp.
- Do not claim all rows repaired based on a representative case.

## Ownership
Use this isolated ManoRL feature worktree for task scripts and artifacts. The primary dev checkout and its unrelated edits remain untouched. Shared read-only dev simulation/data adapters may be imported by scripts; do not inspect training implementations.
