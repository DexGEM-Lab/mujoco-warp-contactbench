## Objective
Prepare new capture training settings (updated user request: pre60/post250,120Hz): XYZ scale0.002m/cap0.01m, five distal finger contact sites by default, only right hand. Inspect available servers without starting jobs.
## Workbench
Implement training defaults and explicit reference-hand removal; preserve source operator/asset identity and underscore object names; focused checks with new source.
## Context
Source dexgem_vla_demo_cma2lance_20260915_030735.lance v3:133 rows,8actions,all bilateral cheyingtong captures at~120Hz.16 cup-pouring rows have ambiguous comma-joined manipulated names; user must select target, do not invent.
## Task specifications
Training defaults change; EnvironmentConfig/source-map legacy route remains explicit and checkpoint-compatible. Five fingertip sites are thumb_ip/index_dip/middle_dip/ring_dip/pinky_dip, NOT every phalanx. All five finger action groups active. drop_uncontrolled_hands must remove left references/model while preserving object states and source UUID/indices. Optional source-matched manifest is process-scoped. New object names may include underscores; split identity from right into object/action/sequence.
## Constraints
No training launches, no server changes, no new delegation permission. Work only in assigned feat/sept15-right-fingertip-profile worktree. Preserve unrelated dev content. Do not silently relabel the16 ambiguous source rows.
