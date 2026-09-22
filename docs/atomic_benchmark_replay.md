# Cheyingtong atomic-task benchmark replay pilot

`tools/replay_atomic_benchmark_pilot.py` adapts the existing State45 image-data
pipeline to the new multi-object compact trajectories. It does not consume the
source-camera images and it does not impose saved object states after reset.

For one action it batches five selected rows through the original runtime
contract:

```text
saved frame-0 hand + all task-scene object poses
→ absolute urdf_dof_target[t]
→ 4 × 480 Hz MJX-Warp substeps per 120 Hz control step
→ actual solved hand/object trajectory
→ benchmark host render mirror (never stepped)
→ current head + right-wrist, 640×360
```

Objects named by each compact row participate in physics. The render mirror
contains all nine task assets. Objects absent from the physical row use the
per-UUID static position in `scene_layout_per_trajectory.json`; their render
collision bits are zero. The fixed pilot scene is `clean_kitchen/seed42`.

The immutable selection is `configs/atomic_image_replay_pilot_40.json`: five
rows per available action `001/002/003/004/005/006/007/009`, chosen evenly over
each action's dataset-order population without observing replay results.
Action008 is absent and must be added as new source data.

Each action produces one 30 fps side-by-side video containing the five complete
replays, plus a storyboard, per-row dynamic traces and replay/reference error
metrics. Six GPUs may run six action processes concurrently; the remaining two
actions run on the first two cards after their first jobs finish.

Every worker uses one physical GPU for both compute and offscreen rendering:
`CUDA_VISIBLE_DEVICES=N`, `MUJOCO_EGL_DEVICE_ID=N`, and `--gpu N` must agree.
The entry point rejects a mismatch before importing JAX or MuJoCo, preventing
all EGL contexts from silently accumulating on physical GPU0.

## Table and allocation profiles

`--physics-profile u1-table` keeps the finite box tabletop and four legs. It
lowers their collision geometry by1mm so the tabletop surface is z=-0.001m,
and sets their friction to `[1,0.01,0.001]`. It does not restore an infinite
plane or change commands, hand/object properties, clock, or actuator gains.
The original `atomic-benchmark` table remains an explicit comparison profile.

For multi-object replay the default allocation profile is `headroom`:

| Buffer | Default | Batch5 allocation |
|---|---:|---:|
| Contact | 2048/world | 10240 global slots |
| CCD | 2048/world | 10240 global slots |
| Constraint | 8192/world | 8192 per-world solver width |

Grade accepts `--capacity-profile headroom` (default), `expanded` (historical
2048/512/8192 contact/CCD/constraint), or `u1` (historical1024/256/4096).
The render CLI and direct `run_physics` defaults use the same headroom values.
A source dataset's capacity metadata is provenance, not proof that its buffer
sizes suffice for a newly compiled multi-object batch. The compact534 batch5
run with256 CCD slots/world emitted actual demand up to2498 against1280 slots;
its process completion and per-row grades are not valid acceptance evidence.
The new10240-slot default provides about4x that observed demand. More/larger
objects still require checking the log; no fixed capacity guarantees all scenes.

These sizes allocate storage; they do not change friction coefficients, masses,
actuator gains, integration clocks or CCD iterations (16). Parallel contact
ordering can change with allocation shape, so bitwise replay is not promised.

### Required guarded launch

The pinned DataWarp does not expose a live CCD count. Run physics and rendering
commands through the parent log guard so a native overflow cannot masquerade as
success (the guard needs no GPU or third-party dependencies):

```sh
python3 -m tools.replay_capacity_guard --log /OUTPUT/shard0/action001.log -- \
  "$PY" -m tools.replay_atomic_grade --run-shard \
  --physics-profile u1-table --capacity-profile headroom \
  ...
```

Supply the usual dataset, plan, shard, action, model, scene and GPU arguments in
place of `...`. For the pruned534 Lance use `--lance-read-mode individual`;
physics batching remains5, including non-counted tail padding.

On the first real contact/CCD/constraint overflow, the guard terminates only its
child process group, retains the log and writes `<log>.guard.json` with
`state=rejected_overflow`, and exits86. `set -e` in the driver then stops the
queue. It never silently retries or changes capacity. Check these receipts and
all logs before final aggregation; row JSON or `state=complete` alone is not
sufficient. Preserve failed roots and use a fresh root for a changed capacity.
