# Raw right-hand capture replay

`tools/replay_capture_no_policy.py` measures whether recorded right-hand joint
commands manipulate free objects. It loads no policy, residual action, grasp
repair, or object controller.

```bash
MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python tools/replay_capture_no_policy.py \
  --dataset /path/to/capture.lance --version 2 --row 0 \
  --output outputs/raw_capture/row00 --render
```

The output directory must be new. A failed setup/run is retained for inspection;
use another output directory after correcting its cause. This graph-replay tool
requires a CUDA device. Before using a non-default hand profile on a fresh clone,
materialize its meshes/skin with `git -C assets/dexstream_digital_assets lfs pull`;
the training setup only hydrates its own manifest's hand profile.

## Physical and source contract

- Load only the right 28-DoF hand. Select its source operator from `index.operator`
  and require the recorded right-hand betas to match that asset. The run stores
  its own integrity manifest; the default training hand/manifest is unchanged.
- Consume every source frame, including approach and withdrawal. Object identities
  and array slots come from explicit scene/metadata fields, including names such
  as `egg_cup`; no object identity is inferred by splitting a trajectory label.
- Preserve every scene object and enable object-object collisions. A single
  common vertical shift places the lowest source geometry above the project
  floor; it applies equally to all objects and the hand.
- Initialize object coordinates once, then advance MJX-Warp physics only. No
  subsequent object pose/velocity writes, external object wrenches, welds, or
  source-dependent object feedback are applied.
- Keep project actuator gains, joint limits, gravity, friction coefficients, and
  default solver settings. The position controller chooses equivalent nearby
  wrist angles and enforces joint limits; raw and applied commands are saved
  separately so clipping remains visible.
- Default cadence is one original command per 120-Hz frame and four physics
  steps at 480 Hz. The measured mean timestamp cadence must agree within 0.5 Hz.
  Integer `data_fps=119` does not silently slow a nominal 120-Hz capture. Original
  irregular timestamps are retained; no trajectory interpolation is performed.
- Record actual MJX contact geometry after each transition. These are not native
  contacts recomputed from rendered coordinates. Capacity saturation is an error.

## Reading the results

Each run writes `manifest.json`, `asset_manifest.json`, `trace.npz`, and
`summary.json`. With `--render`, it also writes `comparison.mp4` and acquisition,
source-apex, and terminal stills. The left panel animates recorded object poses
for comparison; only the **right panel** shows measured free-body physics.

`longest_free_grip_seconds` is a screening measure: the object is over 2 cm
above its initial body height, has hand contact, and has no other-body contact.
Inspect the trace/video to distinguish a sustained grasp from a transient push,
knock-away, or intended release. Likewise, following the recorded hand trajectory
or briefly raising an object does not establish complete task success.

The source operator and physical settings bound the result. A failed baseline
is not a claim that another controller or repaired trajectory cannot succeed.
