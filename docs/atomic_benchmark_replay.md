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
