# Current model — fixed pair quota not yet implemented

## Corrected understanding (authoritative)
The user clarified the ORIGINAL goal: each object–action pair keeps its target
of original training-source count×5 even after weak sources are excluded.
100 sources with50 excluded still requires500 accepted rows, not250. Healthy
sources may contribute more than five variants. Quota can move between sources
within a pair, never between different pairs.

The implementation does NOT meet this goal. slots has CHECK(slot<5), initialize
creates five slots per source, exclusions turn missing source slots into skipped,
and summary subtracts skipped from target. There is no donor-source allocation
or replenishment of excluded quota. The prior overall completion claim is
withdrawn; this was an assistant interpretation and implementation error, not a
user reduction of the output goal.

The verified24006 rows remain valid. Fixed target74805, deficit50799. Of129
trained/admitted pairs,105 have at least one row and24 have none;105 nonempty
pairs does not mean105 fixed quotas are complete. Next implementation must keep
pair budgets separate from source eligibility, migrate/credit all existing rows,
permit unique variants beyond five per source, and report no-eligible-source
pairs as blocked. No live ledger has yet been migrated to that design.

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
