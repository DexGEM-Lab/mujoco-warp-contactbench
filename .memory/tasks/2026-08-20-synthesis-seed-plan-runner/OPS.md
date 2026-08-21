# OPS

## 2026-08-20T00:00:00Z
- Created feat/synthesis-seed-plan-runner from dev@d028c61.
- Seed plan root: /home/jay/data/manorl_banana_vla_parent_screen_20260820/seed_plan_10_per_parent.
- Plan: 35 parents, 10 globally unique episode seeds each, near/far paired, 700 rows. Every parent has one selected seed in each far-distance and near-distance empirical decile.
- Fallback plan: 12 same-2D-distance-decile candidates per slot, 4,200 globally unique candidate seeds; minimum cell population 59.
- Real augmentation replay validation: 350 parent-seed pairs, far/near/retreat geometry maximum absolute error 0.
- Decision: use single-env sequential runtime reuse per parent/mode; existing identity-keyed exporter maps make ten duplicate source identities unsafe without broader refactor.

## 2026-08-20T23:44:17+08:00
- User corrected production scope: retain the original 35 previously successful accepted parents; augmentation failures may replace only the candidate seed within its fixed distance cell, never the parent.
- Removed exploratory alternative artifacts: `prefix_clear_v2/`, `strict_175_accepted_parents/`, `strict_175_prefix_screen_chunks/`, and their derived screen/plan JSON files. No Lance dataset was removed.
- Rechecked protected production inputs after rollback: `production_shortlist_35.json` SHA256 `74fbd543cbdf0f5577f6d834969df4b027160e7be3dc969f06dc2fdc62c425fe`; `seed_plan.json` SHA256 `23552b473b783ff09d73c3cc149e86c2e7d2e3482ffdcf5b5fe9d33b331a3c07`; `paired_seed_fallbacks.json` SHA256 `a80b6863432d6be69b1545adb9e113eb0650588f4cf9f99b005c99b17499d23a`; `planned_700_rows.json` SHA256 `f37414dbc0b867c3b0063525aca55256021ff4e0a03a8f268349102ceb303527`.
- Confirmed runner changes are isolated: no tracked simulation, exporter, asset, checkpoint, or original-plan modifications; only the new plan-runner tool and task memory are untracked in the feature worktree.

## 2026-08-21T11:39:13+08:00
- Server1 preflight: all four RTX 4090 GPUs were idle; NAS `/mnt/nas-222-projects` was mounted read-write with ~14 TiB free; source Unison watcher was restored and repeatedly reported `Nothing to do`.
- Core v4 synthesis source hashes matched local and Server1 after Unison: `approach_prefix.py` `1e7c9b...`, exporter `5b8416...`, `lance_v2.py` `f219da...`, validator `6fd022...`.
- Original first production near cell (`banana_01_079`, episode seed 100408) failed under the default 4 cm approach arc because prefix hand-object contact occurred. Holding parent, seed, sampled start pose, distance cell, checkpoint, and retreat fixed while increasing only the endpoint-smooth Z arc to the contract maximum 10 cm produced a complete successful near rollout. The matching far rollout also completed.
- Production runner v2 therefore defaults both modes to a 10 cm collision-avoidance Z arc. This does not change the far/near start distributions or seed coverage. Retreat v4 remains mandatory for every mode.
- Real runner smoke used the original paired fallback plan, parent index 0, slot 0, seed 100408. Near and far were atomically accepted at fallback rank 0. Output contained 2 compact `synthetic_mano_target_replay_visual_v2_contact` rows; validator passed, 120/480 Hz x4, zero decoder retries. Artifact: `/tmp/manorl_seed_plan_runner_smoke_v2`.
- Focused validation: `29 passed` across seed-plan, accepted-parent, and approach-prefix tests.
