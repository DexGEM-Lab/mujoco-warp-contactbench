# Current model

## Objective and completion
Every admitted raw source owns five accepted published slots. The active fleet
has14,223 sources /71,115 slots /18 objects /121 pairs: original Server1 cohort
plus newly completed cuboid3 and sphere2 on Server2. Attempts, benchmark rows,
or easier sources' surplus never fill another source's quota. Four GPUs now
synthesize; the two remaining training GPUs join only after their completed
models are validated and admitted.

## Supported mechanisms
- Preserve the trained early80 AFTER the physical approach prefix. Merely
  cropping to pre80 does not preserve policy timing in the historical runner.
- Raw PIP reference angles can be negative while physical lower limits are zero.
  Prefix sampling must join the native-servo-clipped finger target, preserving
  the original reference unchanged. Both first GPU probes rejected all40 rows
  until this mismatch was corrected.
- Offline retreat translates wrist XYZ, corresponding world keypoints and
  kinematic wrist targets. Reject NEW/deeper per-geom-pair penetration relative
  to the original solved state: absolute zero-penetration rejected unchanged
  splice frames with legitimate soft floor contact (directly observed).
- Cheap native geometry at reset identified all33 initial-force-rejected starts
  in80 inspected candidates, with no false positives in that sample. It is a
  proposal filter, never a replacement for Warp physics/force acceptance.
- Vectorized contact export agrees with the scalar exporter on force signs,
  transforms, grouped totals and order. It consumes actual solved Warp buffers.
- Runtime reuse, exact host-buffer reuse, no discarded480D rollout storage and
  overlapped CPU retreat/publication reduce wasted work. Initial pose rejection
  before GPU dispatch and refilling make batches useful.

## Direct evidence
See OPS.md experiment entries and local deployment record. B16 baseline yielded
5/40 cube2 and20/40 cylinder6; all25 saved rows passed compact validation.
Their object-bottom clearance was10.88–16.98cm and19.07–23.72cm, with103–269
and133–199 contact+airborne frames respectively. These are specific physical
witnesses, not a generalization claim for all113 pairs.
Controlled320-candidate cylinder6 comparison: B64 saved149 in285.84s
(1876.6/hour), B256 saved147 in210.67s (2512.0/hour).
Final640-candidate comparison with geometry prefilter/vector export:248 starts
rejected before physics,392 simulated; B256 andB512 both saved286. End-to-end
rates5359.5 and4922.0 accepted rows/GPU-hour. Select B256; do not extrapolate
this strong-object rate to weak policies or compare different source sets as a
pure speedup attribution.
No native CCD-overflow warning was seen in these tests. A live CCD count remains
unavailable; nacon/nefc checks and captured native warnings guard publication.
A capacity number is not proof of no truncation.

## Current operational state
Controller revisionc34ff3c governs append-only admission and dynamic GPU pools;
physics/sampling/publication math is unchanged. Server1 management adopted both
live6495607 workers without changing their PIDs. New workers label their actual
executing source revision rather than the original campaign revision.
Server2 GPU0/1 actively synthesize cuboid3 (533 sources/2665 slots) and sphere2
(150/750). First confirmed publications were85 and35 rows, with four NAS rows
read back through the compact validator and matching source revision.
Each host keeps local SQLite. The NAS registry assigns16 objects to Server1 and2
to Server2; publication roots are disjoint. Server1 GPU0/1 stay gated by final
largeclamp/cylinder5 model admission and live GPU ownership. The owned Server2
training heartbeat was retired after confirming no active trainer, preventing
old failed-training queues from competing for the newly assigned synthesis GPUs.
Owner locks pass to CPU publishers, so a killed GPU parent cannot expose slots
still being published. Watch restarts departed supervisors, which adopt survivors.
PAUSE remains explicit operator intent. Full source-level completion is open.

## Open questions / limits
- Actual per-object production yield, hard-source deficits and source-level five
  slot completion remain to be measured. Prefix seed changes need not repair a
  systematically failing manipulation policy.
- Batch sizes produced small physical differences and one cube2 orientation-gate
  flip in the small baseline; each actual row still needs its own acceptance.
  Solver warm-start and clock are now cleared at each wave, without claiming
  bitwise cross-batch identity.
- Broad local suite:559 passed,15 skipped,64 failed,26 errors, dominated by an
  uninitialized DexStream worktree and absent legacy Lance fixtures; isolated
  entrypoint import failure is historical.47 focused tests pass; real-asset GPU
  validation is decisive for this pipeline, not a claim the full suite passed.
- Pi recurring subagents are NOT enabled in this live session: management still
  returns scheduledRuns.enabled=false. The extension reads config once at load;
  the actual on-disk extension config is already true. User /reload is needed
  before creating a recurring agent schedule. Server-side recovery is separate.
