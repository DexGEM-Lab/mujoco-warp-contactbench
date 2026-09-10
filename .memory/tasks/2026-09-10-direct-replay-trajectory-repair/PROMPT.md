# Direct replay trajectory repair

## Objective
Directly inspect and repair object trajectories in `dexgem_vla_demo_guangxue_cma2lance_20260909_060700.lance` (version 5, 59 human demonstrations), using the existing dev replay and DISPLAY=:1. The user explicitly scopes the requested first delivery to five selected trajectories, preferring tractable examples. Existing repaired rows0/19/31/36/49 cover five action categories; verify and deliver these without extending scope to the other54 rows.

## User constraints
- The user now authorizes bounded delegation for the current five-trajectory objective. Earlier trajectory construction was performed directly in the parent. Use one read-only independent artifact review for final functional verification; no nested delegation.
- No reinforcement learning, policy training, or reading RL implementation.
- Preserve source data. Save corrected object poses and comparison artifacts separately.
- Start with object-only corrections. The requested outcome is a repaired replay/object motion; the user did not explicitly prohibit local hand-trajectory edits. After object-only alignment failed to sustain contact and the user requested continuation, local finger closure edits were announced. Keep edits explicit and separately recorded; preserve the task's transport path where possible, and retain source timestamps, physical parameters and source data. A local wrist/grasp-pose repair was announced after static failure showed finger-only offsets insufficient.

## Success criteria
- Inspect actual replay/contact geometry, not merely logs.
- Produce corrected object trajectory data with exact source row/version provenance.
- Verify geometric plausibility and replay behavior. Distinguish constrained/kinematic object following from a free-body contact grasp.
- Deliver five explicitly identified repaired examples with usable replay instructions and original data preserved. Do not claim all59 repaired.
- Existing recorded physical evidence can support final verification; no new GPU simulation is necessary unless that evidence is materially invalidated. The completed replay released GPU0 to another task; do not restart GPU work without coordinating.

## Ownership
Use this isolated ManoRL feature worktree for task scripts and artifacts. The primary dev checkout and its unrelated edits remain untouched. Shared read-only dev simulation/data adapters may be imported by scripts; do not inspect training implementations.
