# Atomic-task image replay pilot

`tools/render_atomic_image_pilot.py` renders saved Cheyingtong compact states in
a fixed nine-object DexStream scene. It is the visual preflight for splitting the
complete actions into language-conditioned atomic skills.

## Contract

- Input: `manorl_cheyingtong_unified_compact_20260918/compact.lance`, version 2.
- Hand: manifest-bound Cheyingtong right MANO URDF and skin.
- Objects: egg, rack, stick, cup, bin, bowl, induction cooker, sauce bottle and
  pitcher from the same DexStream asset commit as `asset_manifest.json`.
- Motion: recorded hand and recorded active-scene object poses are restored
  exactly. Objects absent from a row remain static at
  `scene_layout_per_trajectory.json` positions.
- Rendering: one checker-floor scene, fixed current head camera and right-wrist
  camera, 640×360 by default.
- Causality: no policy and no `mj_step`. Every contact bit in the render mirror
  is set to zero after the compiled asset contract is validated. Background
  objects are visible but cannot change the saved motion.
- The original Lance `video` columns are never read.

The pilot renders one deterministic representative of each available action:
`001,002,003,004,005,006,007,009`. Action `008` is absent from this dataset.
Do not invent a button trajectory during image generation.

## Example

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

PYTHONPATH=. python -m tools.render_atomic_image_pilot \
  --dataset /mnt/podshare/all/sjq-pi05-tmptest/lora/datasets/\
manorl_cheyingtong_unified_compact_20260918/compact.lance \
  --layout /mnt/podshare/all/sjq-pi05-tmptest/lora/datasets/\
manorl_cheyingtong_unified_compact_20260918/scene_layout_per_trajectory.json \
  --asset-manifest /mnt/podshare/all/sjq-pi05-tmptest/lora/datasets/\
manorl_cheyingtong_unified_compact_20260918/asset_manifest.json \
  --asset-root /mnt/podshare/all/chensiyuan-pi05Mint/lora/\
dexstream_digital-assets \
  --output /mnt/podshare/mine/lora/results/atomic_image_replay_pilot_v1
```

Pilot video samples every four 120 Hz states and encodes at 30 fps, preserving
real time. The eventual image Lance should use `--video-stride 1` semantics and
store every control-frame head/wrist image. `--max-frames` exists only for a
bounded smoke test.

Each action directory contains `combined.mp4` (head/right-wrist side by side),
the component `head.mp4` and `right_wrist.mp4`, `storyboard.jpg` and
`metadata.json`. The top-level manifest authenticates the selected row,
layout, asset manifest, asset commit, camera contract and output hashes.

## Atomic language boundary

The compact dataset has no atomic-boundary column. Its three source capture
Lances declare `atomic_actions`, but all 394 source rows contain an empty list.
The coarse `object_move` interval cannot distinguish grasp, tap, hover, pour,
squeeze or placement. Freeze a separate per-UUID segmentation manifest only
after inspecting the target-simulator pilot and implementing event-specific
boundary proposals with sampled visual review. Never derive one fixed frame map
for every trajectory: action lengths and source identities vary.
