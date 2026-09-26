# Evidence trail

## 2026-09-26T23:32:00+08:00 — Scope and baseline
User requested fastest validated two-GPU synthesis, explicitly authorized a monitoring subagent, and asked to complete synthesis of already-trained references. Final quota is uniformly five accepted saved variants per source.
Prior live inspection (23:24–23:27) verified 16 final object checkpoints at 5000 updates, 113 pairs, and 13,540 package source records (67,700-row target); four other objects were training. Server1 GPU2/3 were idle. No new synthesis jobs were running.
Feature worktree is clean at dev base 70153d8 before task-memory initialization. Primary dev has unrelated untracked files; leave them untouched.
Operator monitoring-preflight run c948aa57-a559-4984-a68f-2130e0ebdc36 launched asynchronously, read-only, while parent investigates code semantics. It is not a persistent production supervisor and has not launched synthesis.
Inspection: historical approach config accepts required_base_pre_padding but production identity is pre60 Far/Near. Vector collector reconstructs runtime per wave and records per-step contacts on the host. New per-source offset/posture contract and runtime caching must be checked rather than assumed existing.
