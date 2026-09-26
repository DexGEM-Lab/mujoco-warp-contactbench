# Evidence trail

## 2026-09-26T23:32:00+08:00 — Scope and baseline
User requested fastest validated two-GPU synthesis, explicitly authorized a monitoring subagent, and asked to complete synthesis of already-trained references. Final quota is uniformly five accepted saved variants per source.
Prior live inspection (23:24–23:27) verified 16 final object checkpoints at 5000 updates, 113 pairs, and 13,540 package source records (67,700-row target); four other objects were training. Server1 GPU2/3 were idle. No new synthesis jobs were running.
Feature worktree is clean at dev base 70153d8 before task-memory initialization. Primary dev has unrelated untracked files; leave them untouched.
Operator monitoring-preflight run c948aa57-a559-4984-a68f-2130e0ebdc36 launched asynchronously, read-only, while parent investigates code semantics. It is not a persistent production supervisor and has not launched synthesis.
Inspection: historical approach config accepts required_base_pre_padding but production identity is pre60 Far/Near. Vector collector reconstructs runtime per wave and records per-step contacts on the host. New per-source offset/posture contract and runtime caching must be checked rather than assumed existing.

## 2026-09-27T00:07:00+08:00 — Implementation and preflight corrections
Focused source-prefix/ledger/legacy-prefix/package tests and atomic Lance publication/recovery tests passed (36 tests). These are contract/persistence evidence, not GPU physical acceptance or throughput evidence. GPU benchmarking is the next gate.
Implemented an opt-in base-early-phase field; new source-relative prefixes preserve early80 while historical Far/Near remains unchanged. New runtime restores checkpoint raw_gesture/all-joint/reward/asset settings, caches exact materialized solved buffers, and omits unused rollout observation/action storage.
Offline retreat additionally translates dependent keypoints/kinematic wrist targets and checks all scene-object contact ends plus native collision geometry. No CPU force reconstruction is used.
Monitoring preflight found dead Server1 and missing Server2 Unison watchers. Operator ensure started Server2, but could not repair Server1's extant dead-shell session. Parent restarted only that confirmed-dead sync session. Both watchers subsequently reported active Unison and clean Nothing-to-do reconciliation. Training processes were not touched.
Commit 4332446 succeeded; its optional Git auto-GC pack-refs failed under GitGuard for an unrelated historical branch. No ref bypass or branch cleanup was attempted.
