# Raw120 fivefold synthesis

This campaign consumes the completed per-object raw120 checkpoints. Its unit of
completion is **five accepted, durably published variants for each raw source**.
Attempts, other sources' surplus rows, and benchmark outputs do not fill a quota.
The historical pre60 Far/Near workflow remains available separately.

## Motion and physics

- Cheyingtong right 28-DoF, 120 Hz reference/control and 480 Hz physics.
- Crop the base reference with the same pre80 loader used during training.
- Sample five source-relative displacement strata covering 5–50 cm. Each
  source uses one randomized azimuth stratum per slot, with a full-sphere
  elevation distribution; infeasible starts are rejected, not clipped into a
  different distance. Retries stay within their slot's distance stratum.
- Initial wrist orientation varies by up to 45 degrees about a sampled world
  axis; finger targets vary by up to 0.15 rad within physical joint limits.
  A 4 cm endpoint-smooth arc joins the unchanged base reference.
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
Native MuJoCo kinematics/collision geometry rejects hand penetration throughout
the changed tail; it does not reconstruct or replace contact forces.

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
python -m tools.run_manorl_synthesis_campaign supervise \
  --campaign /local/campaign --gpus 2,3 --batch-size 64 --waves 32
```

The supervisor owns two homogeneous-object worker processes. A worker retains
one model/policy across waves and writes only compact-required state/commands.
Exact solved host buffers can be reused from observation extraction. A separate
CPU process performs retreat and compact publication while the GPU runs its next
wave, with one pending publication as backpressure. Temporary IPC traces are
reclaimed only after verified publication; failed IPC inputs and logs remain.
Existing datasets and checkpoints are never overwritten.

NAS shards are created under unique partial names, read back, hashed and promoted
atomically with their recovery manifest. Only then does SQLite mark accepted
slots. Promotion-before-ledger crashes are reconciled before dead-owner claims
are released. Completion requires every source's five unique slots, not merely
an aggregate row count. Persistent failures leave visible deficits; they cannot
be turned into successful completion.

## Throughput measurement

Use `tools/benchmark_manorl_synthesis_campaign.py` in separate fresh processes
for batch sizes 16, 32, 64 (and larger only when useful). Keep group, source limit,
candidate limit, sampling seeds and physical/gate configuration fixed. Both
stdout and stderr must redirect directly to the declared `--log`.

The benchmark warms the runtime before timing the fixed candidate set and waits
for CPU postprocessing, verified Lance write and promotion. Compare **accepted
saved trajectories per GPU-hour**, initialization, physics/policy, contact
materialization, collection and publication times. Benchmark rows live under
separate output directories and never silently count toward production quotas.

Monitor `status.json`, worker heartbeat records, per-source deficits, fatal/error
logs, actual GPU process ownership and local/NAS free space. A blocked group or
idle GPU with remaining work is an intervention signal, not a completion state.
