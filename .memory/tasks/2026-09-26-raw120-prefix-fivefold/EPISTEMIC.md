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
Both hosts now run the b1cfa66 source-filter policy after a successful graceful
writer drain and index build;78 focused tests pass. Accepted identity sets and
physical configuration hashes were identical before/after activation:21962 rows
on S1 and1483 on S2. The first operations helper's missing Conda pidfd support
was corrected by separating system-Python control from Conda ledger activation;
failed pre-mutation evidence is preserved.

S2 has completed its retained quota:251 sources at1255 rows,432 excluded sources,
228 preserved partial rows,1483 total accepted,1932 skipped, no remaining leases
or ready work. Its new watch exits normally. The old supervisor status can retain
the pre-filter target; newer policy-bearing admission/watch records are the quota
authority when completion required no new supervisor.

S1 continues under the new policy. Its initial8944 exclusions grew to9029 through
actual settlement. The observed current fleet retains5500 sources,2968 complete,
with23487 total saved rows and9955 unresolved slots. These are changing counters,
not a final target. Newly excluded partial sources transfer saved rows between
retained_accepted and excluded_accepted without decreasing total accepted.

The initial refill fix also has live evidence: S1 started with191 useful
candidates, found0 ready for the next batch, settled42 newly saved rows, then
claimed79 eligible candidates after19.6s. It correctly respects the smaller
filtered quota instead of inventing256 independent candidates. Mid-wave refill
remains unimplemented. S2's completed/excluded sources must not be rerun merely to
benchmark512; prior actual512 execution did not establish a valid-row speedup.

Filtering left only a large cube1 group with ready work. The original assigned-
group exclusion prevented another local GPU from helping even with8912 ready
slots. S1 now runs eb51e5d: prefer distinct groups, then share a ready group across
idle GPUs, reserving one batch per planned launch.83 synthesis tests pass.
Live evidence shows the original GPU2 worker was adopted unchanged and a newly
released GPU3 started another cube1 worker. Their479 leased slots had479 unique
tokens and source/slot pairs. Both run256 worlds; no worker or foreign process was
signalled. This removes a scheduling limit, not the mid-wave refill limitation.

S1 now uses GPU2/3. The previously free GPU1 and both S2 GPUs were subsequently
occupied by external jobs; do not kill them. Eligible pool size is not occupancy.
A later snapshot had23887 fleet accepted rows and479 unresolved slots, all leased;
no ready work is not itself completion until leases settle. Managers and scheduling are independent of LLM availability.
The v2 monitoring contract now uses retained-source completion and explicitly
recognizes the planned drain/index/migration phases; the old all-source schedule
is paused and retained as history. Existing EACCES recovery credited54 hash-verified rows
once; keep that evidence and never recredit it.

See OPS.md for immutable evidence and the ignored local operational record for
current PIDs, paths, deployments and dated counters. The old broad local suite
had559 passes/15 skips/64 failures/26 errors due mainly to missing assets/fixtures;
focused tests are not a claim that the entire suite passed.
