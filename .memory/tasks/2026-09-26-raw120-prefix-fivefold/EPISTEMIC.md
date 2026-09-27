# Current model

## Objective and completion
The original inventory is14961 sources/74805 slots/20 objects/129 pairs. The user
now permits excluding incomplete raw sources below25% historical generation
success. A retained source still requires five accepted published slots. Use at
least20 finished attempts; exactly25% is retained. Keep all existing accepted
rows, including partial data from excluded sources. Original inventory, retained
source targets, skipped slots and preserved excluded-source rows must be separate.
Completion requires retained sources full, no ready/leased work and no unresolved
publication blockers. Foreign training does not define this campaign's completion.

## Physical/data invariants
- Preserve trained early80 AFTER the physical prefix. Perturb native-servo-clipped
  finger targets while leaving the base raw reference unchanged; raw PIPs can be
  negative despite a physical lower limit of zero.
- Prefix and manipulation are MJX-Warp physics. Compact contacts are actual solved
  forces. Offline retreat translates wrist XYZ and dependent keypoints/targets
  after last contact+15, preserving fingers, objects, timestamps and contacts.
- Retreat rejects new/deeper per-geom-pair penetration relative to the solved
  baseline, not pre-existing compliant overlap. Native geometry never substitutes
  reconstructed CPU forces for Warp evidence.
- Inference CCD16 is a setting, not proof of zero overflow. Live CCD count is
  unavailable; captured warnings and nacon/nefc checks remain necessary.
- Host-local SQLite and disjoint NAS ownership remain required. Only verified
  durable publication earns credit. Reconcile promoted shards before releasing
  dead-owner claims; publisher-inherited locks protect in-flight work.

## Source filtering mechanism
The ledger's finished attempt timestamp is set only by settlement. Sampling and
geometry rejections count as finished generation attempts; abandoned/in-flight
attempts do not. The numerator comes from accepted source slots, not a diagnostic
claim of success. Complete five-slot sources are not retroactively excluded.
Frozen exclusion records plus SQLite triggers prevent even a legacy worker from
making an excluded failed/released slot ready again. Leased tokens stay valid;
late valid publications remain accepted. Summary counters use one read snapshot.
Policy is separate from immutable physical campaign configuration. Admission
mirrors the per-source audit under fleet/source-filters, not ownership records.

## Efficiency and actual parallelism
Each GPU worker was256 worlds with8 sequential waves. Server2 has now executed
512-world waves without observed OOM/native capacity warnings, but higher useful
throughput is not yet established. Four recent hard-source256 rounds yielded
5 accepted rows from8192 simulated candidates; hardware utilization is distinct
from this very low yield. A live256-world batch had only one useful candidate
left at step1800. Finished slots are not refilled mid-wave and the viewer stepper
can keep resetting/replaying discarded worlds. That remains an unresolved waste.

A second, resolved-in-code problem affects initial batch filling: sphere2 had616
remaining slots, and asynchronous publication held512 leases while the next wave
claimed104 and padded to512. Logs alternated512/104 repeatedly. claim_wave now
settles pending publication only when the next claim is short, then claims more
released failures. Full claims retain overlap; accepted slots are not reclaimed.
Seven real-SQLite tests cover refill, full-batch overlap, no-ready, accepted,
completion, failure and genuinely insufficient quota cases. This fix does NOT
implement mid-wave refill.

Historical matched cylinder6 benchmark: B256 saved286/192.108s versus B512
286/209.181s; do not extrapolate strong-object throughput to remaining hard sources.
In sampled hard rounds,94–96% of rollout wall time was inside the stepper, which
includes host/device synchronization; that is not a measurement of pure GPU time.
CPU retreat/publication was roughly1s per415–719s round, not the dominant bottleneck.

## Deployment boundary and remaining verification
The512 management handoff was graceful and preserved existing workers; actual
512 execution was observed. Initial-batch refill0e6b5e8 and source filtering are
included in immutable snapshot b1cfa66 on both hosts;78 focused synthesis tests
pass. Activation jobs are now waiting for a graceful writer drain, so real
exclusion counts and resumed new-policy production still require verification.
The first operations helpers failed before any signals because Conda Python
lacks pidfd_open; corrected jobs use system Python for process control and the
original Conda interpreter for ledger activation. Both corrected jobs reached
the intended drain phase. See the local record for current identities.
Large histories lack a settled-source index:
a read-only Server1 aggregation hit a35s deadline; EXPLAIN showed a full attempts
scan and temporary grouping tree on local ext4. Initial index creation/filter
activation therefore follows a writer drain rather than a long live write lock.

Only two GPUs were recently available to this campaign (S1 GPU2, S2 GPU0); the
others were occupied by external jobs. Do not kill them. Eligible pool size is
not actual occupancy. Managers and scheduling are independent of LLM availability.
The v2 monitoring contract now uses retained-source completion and explicitly
recognizes the planned drain/index/migration phases; the old all-source schedule
is paused and retained as history. Existing EACCES recovery credited54 hash-verified rows
once; keep that evidence and never recredit it.

See OPS.md for immutable evidence and the ignored local operational record for
current PIDs, paths, deployments and dated counters. The old broad local suite
had559 passes/15 skips/64 failures/26 errors due mainly to missing assets/fixtures;
focused tests are not a claim that the entire suite passed.
