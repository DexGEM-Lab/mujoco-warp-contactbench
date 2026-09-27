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

## 2026-09-27T00:27:00+08:00 — First GPU anomaly and causal correction
Source snapshot 99284d1 was deployed immutably. Preparation produced 13,540 sources / 67,700 slots / 16 scene groups; 641 sources have movement_start below 80 and retain the training crop behavior. Server2 guangguan package metadata differed only in compile_report_sha256 and package_digest; array metadata and source records were identical. Its exact manifest/READY were retained with verified shared arrays. Initial preparation rejected an omitted READY marker; the exact server marker was copied and preparation then passed. CIFS rejected copy2 timestamps; byte copies with SHA256 readback succeeded instead.
Cube2 and cylinder6 B16 fixed-candidate tests each rejected all 40 candidates as sampled_prefix_joint_limits before physical rollout. Direct inspection found cube2 source index PIP q0=-0.1811rad with physical lower=0, analogous middle/ring/pinky violations, and cylinder6 negative PIPs. mjx_sim.command_target explicitly clamps final targets during training. Corrected sampling to perturb the clipped reachable finger baseline and interpolate to clipped endpoint commands, leaving the entire original reference unchanged. A regression test covers this exact mismatch.
Scheduled monitor management returned Scheduled runs are disabled by scheduledRuns.enabled=false. No scheduled run was created. The live manager uses its constructor config; current enabled disk files do not change that live result. Parent retained the partial diff before investigating; no CLI-agent fallback or runtime bypass was used.

## 2026-09-27T02:16:34+08:00 — Throughput selection and live production evidence
Controlled cylinder6 320-candidate comparison under4898a05: B64 saved149 in285.8385s (1876.58/hour), B256 saved147 in210.6727s (2511.95/hour). Earlier B16 actual-row QA covered25 rows: cube2 bottom clearance10.88–16.98cm with103–269 contact+airborne frames; cylinder6 clearance19.07–23.72cm with133–199 such frames.
Offline geometry rejection was inspected directly: cube2 source cube2_02_2252 slot0 failed at anchor265+1 with exactly zero added displacement and unchanged -1.576mm floor penetration. Cylinder6 retreat reduced a35.5um baseline overlap to33.9um yet failed the absolute check. New gate compares penetration per geom pair against the original solved baseline. New/no-deeper-penetration regression tests passed.
Native initial-pose penetration >1mm matched all33 initial-force rejections across80 inspected fixed candidates with zero false positives in that sample. Added proposal prefilter and useful-world refilling. Vector contact export matches scalar values/signs/frames/order in randomized fixtures.
Final fixed640-candidate benchmark under532dcab:248 initial geometry rejections,392 simulated; B256 andB512 each saved286. Full publication rates5359.4824 versus4922.0425 accepted rows/GPU-hour (192.1081s versus209.1815s). B256 selected. Tests did not emit native CCD-overflow warnings, but live CCD count remained unavailable; no zero-overflow guarantee inferred.
Local broad suite completed:559 passed,15 skipped,64 failed,26 errors. Root causes included missing/uninitialized worktree DexStream assets (including misleading parent-repo HEAD from an empty submodule), absent legacy Lance fixture, and historical shell-entrypoint import failure. This was not a full-suite pass.47 focused tests passed after watch/PAUSE additions.
Production source64956077c645e917676e9ec3b9a6775a461ca225 was deployed as an immutable verified archive (SHA a1727da373d60a5f5565b3eeb91f011772a93119983583c05311c64e5b418b00). Both Unison watchers were observed active/clean before launch. Persistent watch launched02:04:47+08 with GPU2/3, B256,8 waves per round. Operator startup run c72b57ef-3bc9-40c2-b221-06fc547c2fd7 confirmed process ownership and meaningful rollout progress; its first snapshot correctly reported zero rows before publication.
Parent dependency-barrier check then confirmed575 accepted rows via direct SQLite and actual NAS shards, no blocked groups/alerts. A subsequent readback validated12 sampled rows across four NAS production shards; authoritative count advanced to833 (554 cylinder6,279 cube2). Snapshot/path/PID details and artifacts are in the ignored local deployment record. Benchmark rows are not counted. All-source5-slot completion remains outstanding.

## 2026-09-27T10:56:19+08:00 — Append-only fleet expansion
User requested gradual use of all idle GPUs and automatic admission of completed object policies. Implemented host-local admission, immutable cross-host object ownership, completion-gated GPU pools and actual executing-source provenance.23 focused admission/ledger/publication/contact tests passed; physics/sampling/reward math was unchanged. Source c34ff3cebc1eaa3136d216d039d43dc41ba4430c was deployed immutably on both hosts, archive SHA6c4cc4a2c250281a0805290e03f029e70947240796a975a60006129053fdd9fa.
Operator87b0acc9 completed a read-only resource/completion snapshot. Both Unison watchers were clean. cuboid3 andsphere2 were final5000-update policies; largeclamp/cylinder5 were still training. The Server2 training-only heartbeat retained authority to restart failed bottle/pitcherbase queues despite idle GPUs. Parent verified its exact command/PID and zero live trainers, retired only that heartbeat, and preserved its scripts/logs.
Registered all16 existing Server1 object owners before Server2 admission. Server2 appended533 cuboid3 sources (2665 slots) and150 sphere2 sources (750 slots). Fleet target became14,223 sources /71,115 slots /18 objects /121 pairs. Server2 publishes under its own host subroot, avoiding cross-ledger recovery scans of Server1 shards.
Server1 management-only handoff froze and replaced watch/supervisor after ownership checks; both GPU worker PIDs remained unchanged. New supervisor explicitly logged adoption and recovered10 already-promoted rows. GPU0/1 remained outside the eligible pool while largeclamp/cylinder5 training continued; pool expansion depends on completed model admission plus live NVIDIA ownership checks.
Server2 GPU0/1 became active synthesis workers. Direct SQLite/NAS evidence confirmed85 cuboid3 +35 sphere2 accepted rows, no blocked groups. Four rows across both objects passed compact readback validation and reported executing revision c34ff3c. NAS registry contained18 unique objects, assigned16 to Server1 and2 to Server2. Exact PIDs/paths are in ignored local deployment memory.

## 2026-09-27T11:31:52+08:00 — Recurring operator monitoring enabled
Live schedule.list succeeded instead of the previous disabled-runtime response. Created/read back project schedule raw120-training-synthesis-monitor, every15m, overlap skip, catchUp latest, quiet false; bounded fresh operator workflow. Immediate operator19527856 confirmed largeclamp final5000 and admission, but JSON counters were missing because its remote inspection used unavailable python and had a NameError. Preserved clean feature worktree evidence, corrected the scheduled prompt to explicit /usr/bin/python3 and defined variables, and used one bounded corrective resume de63dd6e. It reported Server1 17525/70650 and Server2 1023/3415 with no blocked/errors, largeclamp admitted, cylinder5 still training.
A single parent check at11:30:15 clarified the previously instantaneous GPU3 idle sample: GPU0/2/3 all had synthesis workers; GPU3 cylinder4 PID1197942 had a fresh10.1s rollout heartbeat at step800. No worker was killed/restarted for the transient D/0%% observation. Schedule next fire after corrected creation was11:40:35+08; future recurring execution must be distinguished from successful definition creation.

## 2026-09-27T13:02:00+08:00 — NAS promotion failure recovered without replay
largeclamp worker exited when CPU publication directory rename returned EACCES. Its verified partial contained54 accepted rows and3 payloads (44557876 bytes); all hashes rechecked, original attempts unchanged, slots ready. Under owner lock, restored only those exact claim tokens, promoted the intact shard in a fresh process and committed54 slots once. Final Lance count54 and three sampled rows validated. Input IPC and failure logs retained. This supports transient NAS promotion refusal, not a proven permanent ACL or physics failure.
Commit331cdc2 releases readback handles before promotion and applies four logged retries only to EACCES/EPERM/EBUSY (1/2/4s backoff); persistent failure preserves partials and never overwrites finals.20 focused tests passed. Deployed immutable new snapshots and replaced only management processes on both hosts; all six GPU workers/publishers were preserved/adopted. Removed only recovered largeclamp's blocked entry; both host status files then showed no blocked groups or alerts. Largeclamp is eligible again, not yet claimed reexecuted at this barrier. Existing snapshots remain immutable. The reportedly stale cylinder1 worker showed fresh progress on direct inspection; no restart was needed. No foreign GPU process was modified.

## 2026-09-27T13:25:30+08:00 — Historical heartbeat false alarm corrected
Direct Server1 inspection localized worker_heartbeat_stale to retired largeclamp owner g0-7f605b04c167/PID1221023 (no matching live PID, age2415s). All four active supervisor owners had fresh1.7–9.5s rollouts. Root cause was watch scanning all retained two-hour worker history for stale records, including failed owners not in current assignments.
Commit961013d0cffca93126abeb0476e4c18639c2c124 restricts liveness checks to current supervisor owners, flags missing/current-stale heartbeats, retains historical audit rows, and publishes active_workers/stale_worker_owners.21 focused tests passed, including historical-failure, current-stale, and missing-heartbeat cases. Immutable snapshots deployed on both hosts; only watch/admission were replaced, leaving supervisors and GPU/CPU workers untouched.
Post-deployment: both hosts alerts=[] and stale_worker_owners=[]; S1 accepted19933 with active heartbeat ages6.8–22.7s, S2 accepted1383 with0.9–2.2s. The old failed owner remained in recent_workers but was absent from active_workers. Local monitoring guidance was curated to exact production/NAS paths, full current-owner fields, and correct completion-rate semantics; prior local history was archived, not deleted.

## 2026-09-27T20:02:49+08:00 — Batch underfill and user-authorized source selection
Server2's actual512-world run alternated independent candidate counts512/104. Read-only SQLite showed sphere2 had134 accepted and616 remaining slots, all leased by the same worker: one pending publication held512 while the next wave claimed104. Proposals per short wave were214–266, below the4096 sampling budget; lack of available leases, not the sampling cap, produced the padding. Commit0e6b5e8 settles pending publication only for short claims, then fills from legitimately released failures.28 focused tests passed; this does not implement mid-wave refill.
The user changed the objective to allow skipping raw sources below25% success. PROMPT now specifies a20-settled-attempt floor, strict below25%, preservation of existing rows and in-flight tokens, and separate original/effective/retained targets. Finished generation attempts include sampling/geometry rejection; abandonment/in-flight records remain unfinished and are excluded. The filter uses accepted slots as numerator, preserving publication authority.
Implemented source policy/exclusion tables and SQL triggers for compatibility with old in-flight workers, coherent summary snapshots, explicit CLI activation, NAS exclusion audit, and completion guards for publication blockers.50 focused tests passed before final documentation updates. The new policy is not yet deployed.
A Server1 read-only historical preview was interrupted by its35s query deadline; no DB changes were made. findmnt confirmed local ext4, and EXPLAIN showed full attempts scan plus temporary grouping tree on a2291941376-byte ledger. Initial settled-source index construction and policy activation will follow a graceful writer drain, not hold a long write lock against live workers. Prediction: retained completed sources/accepted rows remain unchanged; ready low-success deficits become skipped and no longer re-enter claims, including through legacy finish/release SQL.

## 2026-09-27T20:22:05+08:00 — Filter deployment and bounded writer drain
All78 synthesis tests passed on the final source candidate. Committed source b1cfa66facbcb61683fe2448d95da7c8621573e3 includes the prior short-batch refill checkpoint0e6b5e8. Source archive SHA256=0231150b9c9a7c5f9737243983b2cd8f040ebd98d5b3925bd2da9a918ebdc692 was verified on both hosts and extracted into new immutable source-b1cfa66 directories. Both Unison watchers were active and clean.
Initial operation helpers failed before sending any signals or touching SQLite: Conda os lacks pidfd_open. Failed logs/handoff records remain under production/recoveries/filter25-b1cfa66. Feature worktree was clean at b1cfa66 before the corrected attempt. System Python supports pidfd operations; v2 uses it for control and calls the original Conda runtime for filtering. V2 helper SHA256=80ee4c58f65fdecc7e14baf2cf1e039bca78c9490246ac72e02868e5745bc329.
Both corrected helpers reached waiting_for_current_rounds; supervisors acknowledged stop_requested=true and original GPU workers remained alive. Only campaign watch/admission managers were stopped; no worker or foreign training process was signalled and no PAUSE marker created. The jobs wait at most45min for drain, then bound activation to15min, compare accepted identity/config hashes, and launch updated managers. Current evidence is startup/drain, not activated policy or completed exclusion counts. Per-host evidence is production/source-filter-transition.json and production/recoveries/filter25-b1cfa66-v2/. Exact PIDs/paths are in private local memory.
Replaced the monitor definition without deleting history: old raw120-training-synthesis-monitor paused; new raw120-synthesis-success25-monitor runs every15min and tests retained-source completion, exclusions, leases and publication blockers. It recognizes planned migration instead of treating deliberately absent managers as an outage.

## 2026-09-27T20:42:29+08:00 — Both filters active; preservation and live refill verified
S2 transition reached managers_launched with1483 accepted identities unchanged before/after (SHAf001f5d220f3b14c54e598daa13b53b494ad5d2d5fd9cd5f20a0fb40c4da5d7a). It retains251 sources at1255 rows, excludes432 sources, preserves228 partial rows, skips1932 slots, and has zero ready/leased work. Fresh admission/watch and the432-record NAS audit agree; watch completed normally. The old supervisor status retains its pre-filter denominator because no new supervisor was needed; it is not the current quota authority.
S1 transition also reached managers_launched, preserving21962 accepted identities (SHA9938a58ebdfcecf180e1972f543e2df3c3f99bf370ecf6fcbcca83c7a779a408). Initial exclusion count8944 grew to9029 through live settlement; NAS audit matched9029. The20:31 admission snapshot showed22004 accepted,5249 retained sources (2717 full),39431 skipped slots and9955 unresolved. The changed retained/excluded row split is classification, not deletion.
S1 new cylinder3 worker on source-b1cfa66 completed a191-candidate wave, found0 ready for its next claim, waited19.6s for publication, credited42 valid rows and then claimed79 eligible candidates. This directly verifies short-claim settlement/refill without inventing256 independent candidates.512 physical execution was previously established on S2; no throughput speedup is claimed, and excluded sources will not be rerun solely for benchmarking. Mid-wave rolling refill remains unimplemented.
The new20:39 monitor reports S1 continuing and S2 complete; the fleet is not ALL_COMPLETE. The15-minute v2 schedule remains active under the user's retained-source criterion.

## 2026-09-27T21:52:56+08:00 — Last-group sharing uses a newly freed second GPU
Direct21:27 S1 inspection found only cube1 open (9346 unresolved,8912 ready) while GPU1 was idle; the original assigned-group filter prevented additional workers. Implemented distinct-group preference with a ready-capacity fallback for shared groups, per-loop launch reservations and unchanged live NVIDIA ownership checks.83 focused synthesis tests passed, including real-ledger disjoint claims after exclusions. Commit eb51e5d353e2d87c1a5208cf9f2dbad5420c1409 was immutably deployed on S1 (archive SHA c4c73e6a25cd89848de90535ca22fc3d9b7dd7b1533b4d1b991f99f140ea3c01). Both Unison watchers were active and clean.
GPU1 was taken by external training during deployment; it was not forced. Management handoff replaced only campaign watch/admission/supervisor, obtaining the supervisor's stop-scheduling acknowledgment before retiring it; no GPU/CPU worker was signalled and no PAUSE marker created. New supervisor adopted the existing GPU2 worker unchanged. When GPU3 freed, it automatically started a second cube1 worker there; both were verified in actual256-world rollout with fresh heartbeats. S1 handoff evidence is production/recoveries/management-eb51e5d.json.
At21:46, the two owners held223 and256 leases; all479 tokens and source/slot identities were unique. S1 accepted22404, remaining479 (all leased, ready0), excluded11161, skipped48507; with S2's1483 preserved rows, fleet accepted23887. The large deficit reduction also includes source exclusion, not just saved output. Source-filter/physical settings stayed unchanged; dual-GPU throughput has not yet been measured over a completed comparable interval. Empty ready alone is not ALL_COMPLETE.

## 2026-09-27T22:29:02+08:00 — Completed retained cohort and final data reconciliation
Both hosts reached zero ready/leased slots, no workers/blocked groups and complete retained quotas. Final S1:22523 accepted,2975 retained sources (14875 rows),11303 excluded sources with7648 preserved rows,48867 skipped. Final S2:1483 accepted,251 retained sources (1255 rows),432 excluded sources with228 preserved rows,1932 skipped. Total24006 accepted =3226 retained sources×5 +7876 preserved excluded-source rows;11735 excluded sources and50799 skipped slots. This completes the revised policy, not the original unfiltered74805 target.
Started read-only CPU completion audits on both hosts; helper SHAc14d4c972bbcfc6b1939a52e4fbe8e1a59e96b12026ba134afce185b88ea220e. S1 passed998 shards/22523 rows/15718904785 hashed bytes/34 samples; S2 passed197/1483/965149035/4. All1195 actual Lance UUID indices matched their ledger and commit records;3585 payload files totaling16684053820 bytes matched publication SHA256s. All accepted manifest gates and38 compact-row samples passed. No extra/unpromoted shards, duplicate UUIDs, or cross-host source overlap were found. Full row identity and retained-source five-slot assertions were checked, not inferred from file counts.
Published byte-readback-verified per-host reports/inventories/exclusion snapshots and frozen consumer artifacts under fleet/completion/20260927-success25-v1 in the NAS output root. manifest.json SHA=afe75217d8c5abcafaba8ecd8ef5b9e982aebe9f6af9b4700fb250c6b8373d72; rows.json SHA=72dbc1f29f2366ac0f8015eba3591dc3be8b3102fef031bad65e384462ce8f23. rows.json labels retained_source; manifest covers19 actually published objects/105 object-action pairs. cuboid1 has no accepted output. Snapshot boundaries explicitly exclude any future appended shards.
Paused raw120-synthesis-success25-monitor after validation; schedule.show confirmed paused. Old v1 schedule remains paused/history retained. Campaign GPU workers exited naturally; other users' training was never stopped. No worktrees, datasets, checkpoints, failed partial/IPC evidence or old snapshots were deleted. Source feature remains unmerged into protected dev.

## 2026-09-28T00:02:12+08:00 — Correction: original fixed object–action budget was not implemented
User explicitly clarified the unchanged goal:100 original sources in cylinder1:01 imply500 accepted outputs; excluding50 weak sources does NOT reduce500, and the remaining sources must supply it. Source eligibility and pair output budget are independent. Prior entries that described overall completion under a reduced retained-source target reflect the assistant's mistaken interpretation, not the user's intended scope reduction; that completion claim is withdrawn.
Direct code inspection confirms the gap: slots has CHECK(slot<5), initialize creates exactly five slots per source, excluded source deficits become skipped, and summary sets target=original_target-skipped. No redistribution exists. The original aggregate target remains74805, valid saved rows24006, outstanding50799. Existing index/payload verification remains true and data must be preserved;105 nonempty pairs out of129 is coverage, not fixed-quota completion.
Updated PROMPT/project guide/current model and todo state to the correct requirement. Correct implementation must retain fixed per-pair budgets, allow eligible sources beyond five unique variants, preserve all prior credit/lineage and classify no-eligible-source pairs as blocked rather than complete. No live ledger, source-policy setting, published data or GPU process was changed during this clarification; implementation/migration remains pending.

## 2026-09-28T12:10:00+08:00 — Fixed pair-budget implementation and migration (feature branch only)

Implemented the corrected quota model in the isolated worktree
feat/raw120-prefix-fivefold: pair-budget ledger, donor reassignment with extra
variants, explicit idempotent legacy migration with mandatory post-migration
policy reapplication, per-pair status reporting, and the widened sampler
(legacy seeds bit-identical; extra variants cycle strata within 5-50cm).
98 focused synthesis tests pass, including 100-sources/50-excluded reaching
500 from survivors, mixed-action group isolation, blocked no-donor pairs,
exact .25/floor-20 boundaries, formerly-full donor re-evaluation, migration
credit preservation/idempotence/lease refusal/historical-manifest recovery,
unique variant seeds and UUID lineage, concurrent disjoint claims without
overfill, and admission/watch fixed-target completion. All tests use real
SQLite; publication tests use real shard fixtures.

Deployment preconditions recorded for the parent: drain writers and settle
publications; run `migrate --dry-run` then `migrate` on each live-ledger COPY
under ROOT/production/recoveries/fixed-pair-copy-20260928/ first; expect
accepted24006 unchanged, formerly-exempt full donors excluded (217 S1 +75 S2
=292), eligible donors 2934 rather than3226, and blocked pairs reported for
deficits without donors. The migration reapplies the stored 25%/20 policy
before returning, so no claim can run against the uncorrected donor set.
Parent owns remote operations, final acceptance and integration; no SSH, no
protected-branch merges, no data deletion, no delegation.

## 2026-09-28T12:40:00+08:00 — Parent live inventory confirms migration expectations

Parent's independent read-only inventory of both live legacy ledgers: source
specs carry explicit `pair` (e.g. cylinder7:02) and groups span actions, so
the migration's spec-based pair extraction and mixed-action group isolation
match reality. All129 pairs are deficient. With the full5 exemption removed
and old exclusions preserved: S1 14278 sources / target71390 / accepted22523 /
deficit48867, 2758 eligible donors, 217 formerly-full donors below .25, 32
no-donor pairs (11267 blocked debt, 37600 runnable); S2 683/3415/1483/1932,
176 eligible, 75 formerly-full, 2 blocked sphere2:01/02 pairs (500/1432).
Totals: 2934 eligible donors, 34 blocked pairs /11767 debt, 95 runnable
pairs /39032 debt. Both ledgers hold accepted/skipped only (no leases,
workers), so the migration's lease-refusal precondition passes; admission
daemons are alive and the parent will quiesce them before the real run. The
existing attempts_settled_source index keeps grouped policy reapplication
counts fast (<.3s). Commit733e3a8+ reports blocked debt separately
(`blocked_pair_debt`/`runnable_pair_debt` in summary) and preserves all
target/credit; no GPU/source mutation was performed by this worker.

## Timestamp correction (appended 2026-09-28T01:14:00+08:00, from `date`)

The two OPS entries above labeled "2026-09-28T12:10:00+08:00" and
"2026-09-28T12:40:00+08:00" were written while the actual local time was
about 00:55-01:05 +08:00 on 2026-09-28; the 12:xx labels are wrong. The
entries' technical content is unchanged; only their timestamps were
mis-stated. Immutable entries are preserved as written; this correction is
appended rather than rewriting them.

## 2026-09-28T01:14:00+08:00 — Critic-review corrective pass (third commit)

Fixed all three review blockers in the same worktree starting from ec14e1c:
(1) schema conversion and stored25%/20 reapplication now commit in ONE BEGIN
IMMEDIATE transaction, so the pair-budget marker is never visible before
eligibility is corrected; an already-migrated invocation verifies and repairs
eligibility left uncorrected by an interrupted older candidate while
preserving credit/history/config and original exclusion rows. (2) `migrate
--dry-run` projects reconciliation, conversion and reapplication on an
isolated host-local temporary SQLite copy (read-only backup of the original,
shutil fallback) and never constructs a writable legacy ledger or calls finish
on the original; published shards are only read. (3) extra-variant seeds now
come from a dedicated positive signed-int64 namespace [2**31,2**63) with a
durable UNIQUE allocation registry (nullable attempts.allocated_seed column,
historical rows keep NULL) and deterministic in-transaction collision
resolution; the allocated seed is carried on the claim and propagated through
prefix sampling, runtime point-template seeding, retreat and compact
publication instead of being recomputed. Consumer audit: torch
manual_seed_all, np.default_rng, int64 astype and sha256 identity seeding all
accept full positive int63; lance provenance is pa.int64 with a roundtrip
test. 123 focused tests pass (20 pair-budget incl. forced-collision,
stale/retry/restart allocation, dry-run immutability, interruption rollback,
already-migrated repair; 7 publication incl. int64 seed roundtrip; approach
prefix suite). No live ledger or shard was touched.

## 2026-09-28T01:58:00+08:00 — Pending-aware exclusion guard + audited upgrade (fourth commit)

New bounded T2 slice from decisive live evidence (late-settlement-20260928
read-only snapshot in the primary repo): the per-settlement eligibility rule
excluded donors on settlement order alone because already-issued failures
settle immediately while GPU successes publish later. Implemented the approved
mechanism unchanged elsewhere: reported rate stays durable A / finished F,
floor F>=20, exactly .25 eligible; P = unresolved LEASED output claims for the
source (abandoned/released records never count); exclusion commits only when
(A+P)/(F+P) < minimum_rate — applied identically in the settlement trigger,
explicit configure/reapply and dead-owner release, so once excluded the best
pending outcome mathematically stays below threshold. Added append-only
source_exclusion_history audit and the explicit idempotent quiescent
`upgrade-source-policy` CLI: one transaction replaces the old instantaneous
trigger (never coexisting), creates the audit table, and reinstates only
active exclusions whose now fully-settled A/F meets the unchanged rate with
already-issued settlements after the decision, preserving each original
decision record before reinstating; skipped variants stay retired, genuinely
low sources stay excluded, credit/targets/history/seeds untouched, and
re-exclusion after repair remains possible and audited. 130 focused synthesis
tests pass (27 pair-budget incl. failure-first/delayed-success protection,
clearly-below-despite-pending, exact optimistic boundary, P=0 cutoff/floor,
abandoned-not-P/F, last-lease release decision, two-worker pending leases,
upgrade rollback/idempotence/trigger replacement and evidence-gated repair).
No worker was stopped, no ledger or shard mutated, nothing deployed; the
parent drains current rounds and runs the upgrade on both hosts.

## 2026-09-28T04:22:00+08:00 — Explicit single-object authority transfer tool (fifth commit)

Implemented the bounded authority-transfer seam for reusing S2 GPUs on
blocked S1 pairs: `sim/manorl/synthesis_transfer.py` plus five
`transfer-*` subcommands in `tools/run_manorl_synthesis_campaign.py`.
One object at a time under externally established FULL campaign quiescence
(zero leased outputs, no live worker owner locks — re-checked by every
command); export/import/detach/commit-owner/finalize with durable receipts,
idempotent replay or loud refusal, and no automatic rollback. Schema additions
are minimal: `authority_transfers` (pending rows block claims and campaign
completion on both hosts) and `transferred_credit` (exact retired-credit
receipts). Registry authority uses an append-only transfer record plus an
atomic `objects/<obj>.current.json` pointer; `ownership_record` returns the
pointer first, so admission already skips foreign-owned objects and
`register_existing` additionally skips objects mid-transfer. `reconcile_shards`
now skips ONLY exact retired receipts (source, variant, UUID, ORIGINAL shard);
unknown sources/UUID/shard conflicts still fail loudly. Only machine-local
pickle/checkpoint paths are rebased onto verified byte-identical copies;
source identity, hashes, pair targets, attempts/seeds, exclusions and history
are imported verbatim. Historical accepted rows keep their original NAS shard
paths; the destination publishes new rows under its own output root. Frozen
fleet/completion/20260927-success25-v1 paths are untouched. No physics,
sampler, reward or eligibility math changed. Focused tests
`tests/manorl/test_synthesis_transfer.py` use two real SQLite campaigns with
accepted/failed/unfinished/excluded history and real input files: global
conservation (sources, pair targets, accepted UUID/variant/shard, attempts,
seeds, exclusion audit), destination path rebase, lineage continuation
without duplication, claim/completion blocking until finalize, exact
retired-receipt reconciliation (foreign/changed/shard-conflict still fail),
precondition/identity/config-mismatch refusals, interrupted-step idempotent
replay, admission semantics before/after transfer, and CLI wiring. Validation
with /home/jay/dexrobot/FromSSH/manoRL_mujoco/.venv/bin/python: 93 passed
across test_synthesis_{transfer,pair_budget,admission,publication,
source_filter,campaign}.py, plus 23 passed in test_synthesis_{seed_plan,
acceptance}.py. The one earlier failure under bare python3 was an
environment-only `import mujoco` miss, not a regression. Operational
CLI/quiescence/recovery documentation was appended to
docs/manorl_raw120_fivefold_synthesis.md. No SSH, no live ledger, no GPU job,
no dataset deletion, no protected-branch merge; the parent owns both
production-ledger copy rehearsals and any actual object choice/transfer.

## 2026-09-28T05:56:53+08:00 — Undeployed authority-transfer prototype deferred from the active line (sixth commit)

Commit 84432142e9c362a7e32602dc9a2df0f754f87e46 (the explicit single-object
authority-transfer seam) is an UNDEPLOYED prototype that carried three BLOCK
review findings. The parent authorized a reversible defer: a scoped revert
commit removes only its executable code (`sim/manorl/synthesis_transfer.py`),
CLI wiring (`tools/run_manorl_synthesis_campaign.py` transfer-* subcommands),
tests (`tests/manorl/test_synthesis_transfer.py`) and guide documentation
(`docs/manorl_raw120_fivefold_synthesis.md` transfer section), restoring the
active feature line to the reviewed/deployed 8d7fc386955f8350ab7f06491b6a7a6443064d14
baseline semantics for those files. The prototype commit itself is preserved
verbatim and remains explicitly recoverable by reverting this defer; no
worktree, checkpoint, dataset, ignored artifact or private/local evidence was
deleted. This OPS entry is appended, not rewritten: the 04:22 entry above is
historical record. No live ledger, GPU process or publication was touched; the
transfer architecture is neither reopened nor judged here — it is deferred so
an unrelated unsafe-to-deploy prototype is not shipped alongside other work.

## 2026-09-28T06:58:18+08:00 — Parked short-wave padding candidate validated

Closed out the inherited fixed-width short-wave padding candidate without
changing real candidate semantics. Short waves retain every actual reference,
seed, physics/gate path and quota claim, while trailing physical worlds use a
stationary joint-limit-valid raised/open hand, stationary active-object
reference, and empty policy window; external hand contacts fail installation
and native hand-self overlap remains allowed. Collection stores, terminates and
credits only actual claims. Full batches keep the original no-disable-mask path.
Repaired only invalid test fixtures: active reference positions now equal their
scene slots before `ReferenceTrajectory` construction, the collision fixture
places its passive object at the actual 0.85 m parked hand location and named
per-object qpos addresses, and fake-stepper frame counts reflect increment-before-
reset behavior. Validation: padding behavior 13 passed; focused campaign,
pair-budget, admission, publication, source-filter, seed-plan, acceptance and
contacts regression suites 111 passed. This is CPU/native-MuJoCo validation;
no GPU run, deployment, remote operation, live ledger or protected branch was
touched. Paired real-input evidence remains the prior 11-real/256-world result:
active duplicate padding overflowed CCD4096 (max4401), parked fillers had zero
overflow, and both retained 7/11 base acceptance.
