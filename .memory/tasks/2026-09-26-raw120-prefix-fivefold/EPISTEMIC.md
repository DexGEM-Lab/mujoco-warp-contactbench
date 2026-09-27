# Current model — fixed pair budgets implemented on the branch, live migration pending

## Corrected understanding (authoritative)
The user clarified the ORIGINAL goal: each object–action pair keeps its target
of original training-source count×5 even after weak sources are excluded.
100 sources with50 excluded still requires500 accepted rows, not250. Healthy
sources may contribute more than five variants. Quota can move between sources
within a pair, never between different pairs.

The branch now implements this goal (commits733e3a8, ec14e1c and the corrective
pass after the critic review): `pair_targets` freeze each pair's budget,
claimable `outputs` obligations are keyed by source+variant, extra variants
(>=5) are created only for same-pair deficits on eligible donors, so a pair
can never be overclaimed or overpublished and cross-pair borrowing is
impossible. No-donor deficit pairs are reported blocked with their stranded
debt split from runnable debt; completion requires every pair met and zero
outstanding leases. The25% donor filter keeps its20-settled floor, finished
attempt denominator and durable accepted numerator, without the old
five-row exemption, and is pending-aware: P = the source's unresolved leased
output claims, exclusion only when (A+P)/(F+P) < rate at F>=20 (identical in
the settlement trigger, policy activation/reapply and dead-owner release),
with an append-only source_exclusion_history audit and an explicit quiescent
`upgrade-source-policy` command that atomically replaces the old
instantaneous trigger and reinstates only evidenced premature exclusions.
Live fixed-pair 2a ledgers ARE deployed on both hosts (24006 credits and
74805 target preserved); at01:34-01:36 all four GPUs genuinely ran our
workers. A temporal bias was then observed live: already-issued failures
settle before GPU successes publish, so the instantaneous rule made
premature exclusions (e.g. sphere2_19_4107 at5/21, later30/59=.508;
cuboid3_03_1190 5/21 -> 43/64=.672; S2 51 post-migration exclusions of
which 43 now meet the durable rate). The guard prevents new ones and the
upgrade repairs the evidenced ones; evidence lives in the primary repo at
.memory/local/raw120-prefix-fivefold/late-settlement-20260928/server2.json. The legacy per-source-slot ledger (slots CHECK(slot<5),
shrinking target, no replenishment) is converted by the explicit idempotent
`migrate` CLI, which reconciles promoted shards first, refuses outstanding
leases, converts schema AND reapplies the stored25%/20 policy in ONE
cRASH-atomic transaction (the pair-budget marker is never visible before
eligibility is corrected), verifies preserved credit/history/config lineage
afterwards, and repairs eligibility if an interrupted older candidate
committed the marker without the reapplication. `--dry-run` projects the
whole run on an isolated temporary SQLite copy and never opens the original
for writing. Extra-variant seeds are durably allocated inside the claim
transaction from a dedicated positive-int64 namespace [2**31,2**63), disjoint
from the legacy31-bit domain, with a UNIQUE registry (nullable attempt
column preserving historical rows) and deterministic collision resolution;
the allocated seed is propagated through prefix metadata, runtime seeding,
retreat and compact publication instead of being recomputed.

The verified24006 rows remain valid. Fixed target74805, deficit50799; the
parent's live inventory expects2934 eligible donors,34 blocked pairs carrying
11767 stranded debt and95 runnable pairs carrying39032 after migration. Of129
trained/admitted pairs,105 have at least one row and24 have none;105 nonempty
pairs does not mean105 fixed quotas are complete. NO live ledger has yet been
migrated: the parent will test full production-ledger copies before any
deployment, and this branch's implementation is not a completion claim.

## Valid partial-data evidence (old completion claim superseded)
The old implementation stopped after3226 sources reached five rows (16130)
and retained7876 additional previously accepted rows:24006 total. It incorrectly
removed50799 output obligations when11735 sources were excluded. These skipped
obligations are still owed under the user's fixed-pair goal. Published coverage
is19 objects/105 object-action pairs from the trained20/129 inventory; cuboid1
has no accepted output.

S1 has22523 accepted rows,2975 retained sources full,11303 excluded sources and
48867 skipped slots. S2 has1483 accepted rows,251 retained sources full,432
excluded sources and1932 skipped slots. Both have no ready/leased slots,
workers, or blocked publication under the old schema. This empty queue is a
consequence of the wrong quota model, not proof that the intended goal is done.
The v2 monitor is paused and also needs its completion criterion corrected.
External training was never stopped.

## Delivered snapshot and verification
The campaign output's `fleet/completion/20260927-success25-v1/manifest.json`
freezes1195 shard paths/counts, actual coverage and validation evidence.
`rows.json` maps every UUID/source/slot/shard and marks retained_source, allowing
the16130 uniform-five rows to be distinguished from preserved partial data.
Per-host reports and exclusion snapshots are beside them. The private local
record contains the NAS root; the manifest is authoritative if future admissions
append additional shards to its parent directory.

Every actual Lance UUID index matches its accepted ledger and publication
manifest. All3585 payload files (16684053820 bytes) match publication SHA256
records. Cross-host original source namespaces and row UUIDs are disjoint;
retained sources each have five accepted slots. No extra/unpromoted shards were
found.38 oldest/newest-per-object samples passed the existing compact validator,
including final acceptance, clocks, shapes, reference mapping and force frames.
Manifest SHA256: afe75217d8c5abcafaba8ecd8ef5b9e982aebe9f6af9b4700fb250c6b8373d72.
Rows catalog SHA256:72dbc1f29f2366ac0f8015eba3591dc3be8b3102fef031bad65e384462ce8f23.

## Invariants behind the result
- Prefix and manipulation use actual MJX-Warp physics. Preserve trained early80
  AFTER the prefix and perturb native-servo-clipped finger targets while leaving
  the raw base reference unchanged. Raw PIPs can exceed physical limits.
- Compact contacts are the original solved forces. Offline retreat translates
  wrist XYZ and dependent keypoints/targets after last contact+15, preserving
  fingers, objects, timestamps and contacts. It rejects new/deeper penetration
  against the solved baseline, not legitimate pre-existing soft overlap.
- Only checksum/readback-verified publication earns quota credit. Local SQLite,
  unique source/slot tokens, publisher-inherited owner locks and recovery before
  dead-owner release prevent duplicate credit. Host ownership is disjoint.
- The25% denominator is finished generation attempts, including sampling/geometry
  rejection; unfinished/infrastructure-abandoned attempts do not count. Require
  at least20 settlements, retain exactly25%, and do not retroactively exclude
  already-complete sources. SQL triggers prevent failed/released excluded slots
  returning to ready, even for an older worker. In-flight valid rows still save.
- Exclusion decisions and accepted data are separate. Newly excluded partial
  sources can reduce retained_accepted while increasing excluded_accepted;
  total accepted rows are not deleted. Completion also respects active workers
  and publication blockers, not just a smaller quota denominator.

## Mechanisms learned and remaining limitations
One fixed-width batch can end with very few useful candidates; terminated slots
are not refilled mid-wave and the viewer stepper can replay discarded worlds.
That remains unimplemented optimization, not unfinished work for this cohort.
S2 physically ran512 worlds without observed OOM/native capacity warnings, but
no independent valid-row speedup from512 was established. A matched early
cylinder6 benchmark actually favored256 (286 rows/192.108s vs209.181s at512).

With616 remaining slots, a pending512-slot publication made the next wave claim
only104 and pad duplicates. The short-claim fix settles publication before
padding, preserving full-batch overlap and never reclaiming accepted slots.
Live evidence showed0 ready ->42 saved rows settled ->79 eligible claims.
That is initial-fill repair, not mid-wave refill.

Filtering eventually left one large cube1 group. A one-worker-per-group rule
then left idle GPUs unused despite8912 ready slots. eb51e5d prefers distinct
groups, then shares enough ready work across idle local GPUs with per-loop
reservations and unchanged foreign-ownership checks. Two real cube1 workers
held479 disjoint source/slot tokens; the original worker was adopted unchanged.

Eligibility is not occupancy; foreign jobs limited usable GPUs.83 focused
synthesis tests pass. Historical broad local testing had missing asset/fixture
failures, so no full-repository pass is claimed. Live CCD count is unavailable
in pinned Warp: CCD16 and the final data hash audit do not independently prove
zero physics truncation. Accepted waves retain their warning/capacity evidence.

## Operational evidence
Source b1cfa66 activated filtering on both hosts; S1 later deployed eb51e5d for
same-group sharing. Code remains on the isolated feature branch, unmerged into
dev. Both initial filter migrations preserved every accepted identity/config
hash. The first helper's missing Conda pidfd failed before mutations; system
Python control plus Conda activation corrected it. A settled-source index was
built after writer drain to avoid a long live SQLite write lock. The largeclamp
54-row EACCES recovery was hash-verified and credited once; all failure evidence,
checkpoints, datasets and worktrees remain preserved. See OPS.md for provenance.

## 2026-09-28 — Fixed pair budgets implemented on the feature branch

Commit (this branch): `sim/manorl/synthesis_campaign.py` now implements
`manorl_raw120_fixed_pair_output_budget_v2`. `pair_targets` freezes each
object:action pair's target at 5x its originally admitted source count
(append-only on admission, immutable otherwise, never changed by eligibility).
Claimable `outputs` rows are keyed `(source_key, variant)`; variants 0-4 are
the historical slots, variants >=5 are extra donor variants created only for a
pair's deficit and only on eligible same-pair donors, least-loaded first.
`accepted+ready+leased` can therefore never exceed the pair target: overclaim
and overpublication are structurally impossible, and cross-pair borrowing
cannot happen because donors are drawn from the pair itself. Pairs with debt
and no eligible donors are `blocked`, never complete; completion additionally
requires zero ready/leased. The25% donor filter keeps the20-settled-attempt
floor, strict-below exclusion, finished-attempt denominator and durable
accepted numerator, and no longer exempts five-row sources.

Legacy ledgers are converted by an explicit, idempotent, one-way migration
(`sim/manorl/synthesis_migration.py`, CLI `migrate --campaign ... [--dry-run]`).
It reconciles promoted shards under legacy semantics first, refuses any
remaining outstanding lease, then converts in one transaction: drops `slots`,
adds the explicit `pair` column (legacy order preserved; admissions use
explicit column names), copies every row preserving state/attempts/owner/
UUID/shard lineage, rebuilds pair targets, and re-creates exclusion triggers.
It then MANDATORILY reapplies any stored source-success policy under corrected
semantics before returning, with a pre/post exclusion audit: live-shaped
formerly-exempt full donors (292 in the live copies) become excluded donors
while their accepted rows stay credited. Pair-budget code fails loudly on
legacy databases and vice versa; `slot` remains the published lineage name of
the variant index, so historical manifests keep reconciling. Extra donor
variants cycle the same five radial/azimuth strata within 5-50cm with a
structurally different 64-bit seed derivation: no legacy source/slot/attempt
seed or generated UUID is ever reused (slots 0-4 stay bit-identical).

Validation: 98 focused synthesis tests pass (13+2 new pair-budget/migration
tests using real SQLite and shard fixtures). Pre-existing unrelated
`test_lance_v2` fixture failures reproduce at clean HEAD. No live ledger,
policy, published data or GPU process was touched. Migration of the live
ledgers (copies exist under each host's
ROOT/production/recoveries/fixed-pair-copy-20260928/) remains the parent's
deployment step; production is not fixed and the50799-row debt is not
fulfilled by this change.
