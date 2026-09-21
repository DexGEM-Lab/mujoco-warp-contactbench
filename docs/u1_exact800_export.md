# Strict-C1 U1 exact800 export

This workflow builds a new five-action dataset without modifying the published
exact640. It accepts only the completed ten-parent action005 campaign described
in [`u1_action005_largepose.md`](u1_action005_largepose.md). The older
`tools/export_u1_exact800.py` path belongs to the quarantined row847 pick/place
campaign and is not valid for this deliverable.

## Inputs

- immutable exact640 publication, 160 rows each for003/006/007/009;
- signed ten-parent action005 `registry.json`;
- frozen action005 `plan.json` and canonical `005.jsonl`;
- exactly160 selected action005 artifact trees, 16 from each parent.

The exporter rehashes the complete exact640 publication and every selected
physical replay. It rejects unfinished candidates, exhausted slots, identity or
UUID changes, missing artifacts, prefix force contact, failed gates, target
changes, non-distinct replay PIDs, or a changed parent suffix.

## Export and readback

```sh
PY=/home/jay/dexrobot/FromSSH/manoRL_mujoco/.venv/bin/python
EXACT640=/mnt/nas-222-project/mocap_v2/lance_datasets/manorl_cheyingtong_u1_largepose_4x160_c1_20260921
A005=/mnt/nas-222-projects/mocap_v2/lance_datasets/.manorl_u1_action005_10x16_v1_c1_20260922_staging
OUT=/mnt/nas-222-project/mocap_v2/lance_datasets/.manorl_cheyingtong_u1_largepose_5x160_c1_20260922

$PY -m tools.export_u1_largepose800 export \
  --exact640 "$EXACT640" \
  --action005-registry "$A005/registry.json" \
  --action005-plan "$A005/plan.json" \
  --action005-attempts "$A005" \
  --output "$OUT" --execute-export

$PY -m tools.export_u1_largepose800 validate \
  --exact640 "$EXACT640" \
  --action005-registry "$A005/registry.json" \
  --action005-plan "$A005/plan.json" \
  --action005-attempts "$A005" \
  --output "$OUT"
```

Rows are ordered003/005/006/007/009, 160 each. The four source blocks are Arrow-
normalized copies of exact640 rows; validation compares every new row with its
source row. Every005 row is rebuilt independently from its second physical replay
and compared field-for-field with Lance readback.

`u1_five_action_largepose800_c1_native_v1` retains the compact fields plus
float64 physical qpos/qvel/applied controls, per-frame native contact JSON,
lineage, physical teacher qpos, and complete-scene reference JSON. Action005
contains two ordered object tracks, `mayonnaisebottle,bowl`; unavailable evidence
is never represented by zeros or empty contact arrays.

## Visualization layout

Reuse the already resolved exact640 offsets and solve only the new action005
backgrounds with real collision meshes:

```sh
$PY -m tools.compute_u1_largepose800_background_offsets \
  --lance "$OUT/compact.lance" --version 1 \
  --source640-offsets "$EXACT640/background_offsets.json" \
  --asset-root /home/jay/dexrobot/FromSSH/manoRL_mujoco/assets/dexstream_digital_assets \
  --asset-manifest "$A005/contracts/assets/asset_manifest_cheyingtong.json" \
  --output "$OUT/background_offsets.json"

$PY -m tools.build_u1_largepose800_scene_layout \
  --lance "$OUT/compact.lance" --version 1 \
  --offsets "$OUT/background_offsets.json" \
  --output "$OUT/visualization_layout.json"
```

Active object poses come from each Lance row. Canonical background objects and
clearance offsets are visualization-only and are outside recorded physics.
Publication fails if any mesh-clearance pair remains unresolved or any layout
row lacks the complete nine-object scene.

## Atomic publication

```sh
FINAL=/mnt/nas-222-project/mocap_v2/lance_datasets/manorl_cheyingtong_u1_largepose_5x160_c1_20260922
$PY -m tools.publish_u1_largepose800 \
  --staging-output "$OUT" --final-output "$FINAL" \
  --action005-registry "$A005/registry.json"
```

The publisher reopens Lance, checks order/count/schema, verifies every005 row has
real physical/native/teacher/two-object evidence, confirms16 rows per parent,
validates the layout and offsets, copies runnable action005 parent bundles,
hashes every file, and performs one same-filesystem rename. The final path must
not already exist.

## Published outcome

The final publication is
`/mnt/nas-222-project/mocap_v2/lance_datasets/manorl_cheyingtong_u1_largepose_5x160_c1_20260922`.
Full readback passed all800 rows; action005 required no background offsets and the
combined layout has zero unresolved clearances. Independent final-path audit
rehashed69 files, reopened Lance v1 with800 unique UUIDs, and loaded all ten
parent bundles. The `sha256.json` digest is
`c4545e64d96eb983f5f49ebf3d45a9b32d7269a0af06f262a84e4813cc269777`.

A full903-frame GPU replay directly from exact800 row160 passed without physics
overrides in the complete bottle+bowl scene. Its maximum bottle-position and
rotation differences from recorded physical evidence were7.56mm and0.04873rad.
