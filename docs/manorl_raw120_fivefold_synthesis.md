# Raw120 fivefold synthesis

This campaign consumes the completed per-object raw120 checkpoints. Its unit of
completion is **five accepted, durably published variants for each retained raw
source**. The current campaign permits excluding incomplete sources below25%
historical generation success after at least20 settled attempts. Attempts,
other sources' surplus rows, and benchmark outputs do not fill a quota.
Already-published rows are retained, including partial results from excluded
sources. The historical pre60 Far/Near workflow remains available separately.

## Motion and physics

- Cheyingtong right 28-DoF, 120 Hz reference/control and 480 Hz physics.
- Crop the base reference with the same pre80 loader used during training.
- Sample five source-relative displacement strata covering 5–50 cm. Each
  source uses one randomized azimuth stratum per slot, with a full-sphere
  elevation distribution; infeasible starts are rejected, not clipped into a
  different distance. Retries stay within their slot's distance stratum.
- Initial wrist orientation varies by up to 45 degrees about a sampled world
  axis; finger targets vary by up to 0.15 rad around the source's native-servo
  clipped pose, within physical joint limits. Raw reference PIP angles can be
  negative even when the physical lower limit is zero; the prefix joins the
  reachable controller target without rewriting the original base reference.
  A 4 cm endpoint-smooth arc joins the unchanged base reference. A pose-only
  native collision prefilter rejects initial penetrations deeper than 1 mm;
  it cannot accept a row or replace the subsequent Warp force/physics gates.
- The added prefix **and the checkpoint's original early80 phase** are
  policy-free. Policy/deviation gates open at `prefix_frames + early80`.
  Prefix collision rejection covers the added prefix only, not the original
  early80 segment. Historical Far/Near retains its original timing.
- Restore checkpoint residual scales, all-joint authority, raw_gesture expected
  contacts, reward, point-template mode, asset identity and clocks explicitly.
  The policy-transfer loader is used only for the intentional reference and
  inference-capacity change, not to hide incompatible observation/control fields.
- Requested inference CCD capacity is 16/world, contacts 1024/world, constraints
  4096. Training remains at CCD8. Both native stdout and stderr must be captured.
  Native overflow warnings or saturated measured contact/constraint buffers block
  publication. An unavailable live CCD counter is recorded as unavailable; a
  capacity number does not prove zero truncation.

The base save gate remains reference completion, final object rotation mean
error <=35 degrees, and >=101 persisted frames with solved right-hand/target
normal force >0.2 N. Prefix hand/table or hand/object force >0.2 N rejects it.

## Offline retreat and row semantics

Retreat starts after the last >0.2 N solved hand contact with **any scene object**,
plus 15 frames, and replaces the full remaining tail. The original object states,
contact records, timestamps, fingers and nominal reference remain unchanged.
Because this is a kinematic wrist translation, world hand keypoints and wrist
controller targets receive the same translation as actual wrist XYZ. This avoids
an internally inconsistent picture/target pair. The tail is explicitly labeled
kinematic; its new targets are not claimed to have been integrated by Warp.
Native MuJoCo kinematics/collision geometry rejects new or deeper hand
penetration per geom pair relative to the original solved tail. Existing soft
floor contact is not mistaken for a newly introduced collision (including at the
two unchanged splice frames). This check does not reconstruct or replace forces.

Compact output keeps the existing v2_contact schema and omits stored rollout
observations/actions/rewards. Raw source identity, physical hand metadata,
checkpoint metadata hash, actual CCD settings, software revision and augmentation
identity remain traceable. Per-shard external manifests contain acceptance,
perturbation, retreat, capacity observations and payload checksums.

## Execution

`tools/run_manorl_synthesis_campaign.py prepare` takes an inventory JSON:

```json
{"jobs": [{"object": "cube2", "checkpoint": "/pinned/cube2.pt",
  "packages": ["/packages/remake.mtp", "/packages/guangguan.mtp"],
  "pairs": ["cube2:01", "cube2:02"]}]}
```

The inventory must enumerate every completed pair; the example is abbreviated.
`prepare` verifies package hashes, snapshots cropped references and initializes a
local-filesystem SQLite ledger. Do not put the ledger on NAS/CIFS. Sources from
different datasets are namespaced by dataset/version/UUID/identity.

```bash
python -m tools.run_manorl_synthesis_campaign prepare \
  --inventory inventory.json --campaign /local/campaign \
  --output /nas/new-campaign --software-commit "$REV"
python -m tools.watch_manorl_synthesis_campaign \
  --campaign /local/campaign --gpus 2,3 --batch-size 256 --waves 8
```

The supervisor assigns workers to available eligible GPUs, preferring distinct
object/scene groups. If only a few groups remain, another idle GPU may share a
group with at least one full batch of ready slots. Planned launches reserve their
batch capacity until the next ledger snapshot; leased-only or blocked work never
justifies another launch. Same-group workers claim disjoint slots atomically in
the same host-local ledger. The live NVIDIA ownership check still prevents
starting on a card occupied by another job. A worker retains one homogeneous
model/policy across waves and writes only compact-required state/commands.
Each new wave clears solver warm-start and clock state. Rejected start poses
are replenished before dispatch so they do not consume full GPU episodes.
Exact solved host buffers can be reused from observation extraction; batched
contact-coordinate transforms preserve the scalar exporter contract. A separate
CPU process performs retreat and compact publication while the GPU runs its next
wave, with one pending publication as backpressure. If the next claim is short,
settle the pending publication and claim newly released failed slots before
padding the wave. For example, 616 remaining slots cannot sustain two simultaneous
512-slot leases: claiming before settlement otherwise alternates 512 real
candidates with only 104. Full claims keep asynchronous publication overlap;
accepted slots are never reclaimed. Compare `actual_candidates`, not just the
fixed `worlds` count, when measuring useful concurrency. The worker logs this
conditional wait and refill. Temporary IPC traces are reclaimed only after
verified publication; failed IPC inputs and logs remain.
Existing datasets and checkpoints are never overwritten.

NAS shards are created under unique partial names, read back, hashed and promoted
atomically with their recovery manifest. Readback handles are released before
promotion; transient CIFS permission/sharing failures receive up to four logged
rename attempts (1/2/4-second backoff), without rewriting rows or overwriting a
final shard. Persistent failure retains the partial and IPC evidence. Only after
successful promotion does SQLite mark accepted slots. Promotion-before-ledger crashes are reconciled before dead-owner claims
are released. Completion requires every retained source's five unique slots.
Excluded deficits are explicitly skipped work, never successful rows. Persistent
publication failures remain visible even when the retained quota is satisfied.

## Excluding low-success sources

Enable the opt-in policy on each host-local campaign:

```bash
python -m tools.run_manorl_synthesis_campaign set-source-filter \
  --campaign /local/campaign --minimum-rate 0.25 --minimum-attempts 20
```

The numerator is accepted source slots after verified durable publication; the
denominator is finished generation attempts, including prefix sampling/geometry
rejections and solved candidates. Unfinished and infrastructure-abandoned
attempts have no settlement timestamp and do not count. Exactly25% stays
eligible. Sources already at five saved rows are left complete.

The ledger freezes an exclusion decision and changes only ready slots to
`skipped`. In-flight tokens remain valid: a successful late publication is still
accepted, while a failed or abandoned excluded slot becomes skipped rather than
retryable. SQLite triggers enforce this even for a pre-upgrade worker finishing
its current wave. No source, attempt, shard, or accepted row is deleted.

`original_target` preserves the inventory target. `target` subtracts skipped
slots, retaining accepted rows and unresolved in-flight work in its denominator.
`retained_target`/`retained_accepted` and
`retained_sources_complete`/`sources_retained` describe the five-row obligation
for retained sources. `sources_excluded`, `skipped_slots`, and
`excluded_accepted` report exclusions and preserved partial data separately.
As additional sources reach the evidence floor, the effective target can shrink.
`retained_accepted` can decrease when a partial source is newly excluded while
its rows move to `excluded_accepted`; total `accepted` does not lose those rows.
Completion also requires no ready/leased slots, active workers, or publication
blockers; quota completion alone must not hide a failed publication.

Policy and per-source decisions live in local SQLite. Admission mirrors the
audit to NAS `fleet/source-filters/<host>.json`, outside ownership records.
Reapplying the same policy is idempotent; a changed threshold requires an
explicit policy migration. Building the settled-attempt index on a large old
ledger can hold a long write lock: drain workers and suspend manager writers
before initial activation, then restart managers from an immutable snapshot.

## Growing the fleet without duplicating quotas

Use `tools/admit_manorl_synthesis_models.py` on each host. It scans final training
reports and matching native checkpoint sidecars, requires completed5000 updates
and the raw120/pre80/early80 profile, snapshots the checkpoint by content hash,
then prepares and appends only new object sources to the live local ledger.
Existing accepted slots, outstanding leases and source snapshots are unchanged.

SQLite remains on each host's local filesystem. An immutable NAS registry assigns
each object/checkpoint to exactly one host campaign; another host skips that
object rather than generating another five rows for the same sources. Existing
Server1 objects must be registered before starting Server2 discovery. Give each
host a separate publication root (for example the original root for Server1 and
`hosts/server2` beneath it for Server2), because recovery scans are ledger-local.

A per-host GPU policy separates always-available cards from training-completion
gates, e.g. `{"always":[2,3],"after_objects":{"0":["largeclamp"],"1":["cylinder5"]}}`.
Admission writes `gpu-pool.json` only after the named model's source admission;
the supervisor additionally checks live NVIDIA process ownership before launch.
Thus releasing a card without a valid final model does not silently abandon its
training. `--ensure-watch` also starts supervision when newly admitted work arrives
after a previously completed local cohort. It respects the campaign's `PAUSE`.

Deploy new controller code in a new immutable source directory. During a
management-only handoff, surviving GPU workers and CPU publishers retain their
owner locks and are adopted, not killed. New workers record their executing
revision through `MANORL_SOFTWARE_COMMIT` or `DEPLOYED_COMMIT`; campaign configuration
retains the original generation parameters and does not relabel older shards.

## Throughput measurement

Use `tools/benchmark_manorl_synthesis_campaign.py` in separate fresh processes
for batch sizes 16, 32, 64 (and larger only when useful). Keep group, source limit,
candidate limit, sampling seeds and physical/gate configuration fixed. Pass
`--start-prefilter` to include the production initial-geometry prefilter. Both
stdout and stderr must redirect directly to the declared `--log`.

The benchmark warms the runtime before timing the fixed candidate set and waits
for CPU postprocessing, verified Lance write and promotion. Compare **accepted
saved trajectories per GPU-hour**, initialization, physics/policy, contact
materialization, collection and publication times. Benchmark rows live under
separate output directories and never silently count toward production quotas.

The persistent watch process restarts a departed supervisor; surviving workers
are adopted using PID/start-time and inherited owner locks rather than duplicated.
It exits normally only at exact per-source completion, or explicit operator pause.
Creating `/local/campaign/PAUSE` stops new claims and drains the current wave;
remove that marker and restart the watch process to resume. SIGTERM/SIGINT to the
watch process records the same explicit pause intent.

Monitor `watch-status.json`, `status.json`, worker heartbeats, per-source deficits,
fatal/error logs, actual GPU ownership and local/NAS space. Heartbeat alerts use
only current supervisor owner assignments (`active_workers` and
`stale_worker_owners`); retained historical failures remain audit evidence, not
active liveness targets. A blocked group or
idle GPU with remaining work is an intervention signal, not completion. A Pi
monitor agent performs bounded inspections; recurring agent inspection requires
an enabled live Pi scheduler and is distinct from server-side process recovery.
