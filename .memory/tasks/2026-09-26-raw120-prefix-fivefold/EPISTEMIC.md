# Current model

## Objective and completion
Every one of 13,540 currently completed-policy raw sources owns five accepted
published slots (67,700 rows, 16 objects, 113 pairs). Attempts, benchmark rows,
or easier sources' surplus never fill another source's quota. Training remains
on four GPUs; synthesis owns Server1 GPU2/3.

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
Production6495607 runs on Server1 GPU2/3 with B256 and8 waves per worker round.
Operator inspection confirmed live watch/supervisor/two workers and fresh
rollout heartbeats. Subsequent direct SQL and NAS reads confirmed833 accepted
production rows (554 cylinder6,279 cube2);12 sampled rows across four actual
NAS shards passed compact validation. Production is genuinely producing data,
but67,700-row completion remains open. Benchmark outputs are separate.
Owner locks pass to CPU publishers, so a killed GPU parent cannot expose slots
still being published. Watch restarts a departed supervisor; supervisor adopts
survivors. PAUSE is explicit operator intent.

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
