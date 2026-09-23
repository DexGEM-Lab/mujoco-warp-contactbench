# Fixed-Cheyingtong raw-capture repair training

This workflow trains one residual policy from the two canonical real-world capture corpora:

```text
/mnt/nas-222-projects/mocap_v2/lance_datasets/
  human_p1_remake_v3/human_p1_remake_clean.lance       version 978
  human_p1_guangguan/human_p1_guangguan_clean.lance    version 530
```

## Physical-hand invariant

Every row uses the same physical model:

```text
physical hand = Cheyingtong right MANO
control       = raw right-hand q at 120 Hz + policy residual
physics       = MJX-Warp at 480 Hz, four substeps/control
```

`index.operator` and `mano_hand_shapes` describe the recorded source only. They remain in package provenance and never select a physical hand. A row recorded by any operator still executes on Cheyingtong. Raw capture has one kinematic q track; it has no IsaacGym-style `q_state_ref`.

## Source accounting

- Remake v978: 10,584 rows; all contain a right hand and valid increasing timestamps.
- Guangguan v530: 5,164 rows; 35 have no right hand and 9 have non-increasing timestamps. These 44 rows must be explicit compiler rejections.
- Guangguan contains 55 action18 `bottle,cap` rows. Compile them with `--target-object-overrides bottle:18`; bottle is the tracked object and cap remains a free scene body with object-object collision.
- Expected accepted total: 15,704 rows across 139 object/action pairs.

Do not combine daily, dirty, anomaly, generated, synthetic, or IsaacGym-refined Lance sources with this catalog.

## Clock and scene contract

Raw-transfer compilation keeps the complete capture: approach, manipulation, release, and withdrawal. Movement annotations define reward phases and do not crop the row.

Timestamp duration is authoritative:

- hand XYZ and object XYZ: linear interpolation;
- hand angular coordinates: unwrap, then linear interpolation;
- object orientation: quaternion SLERP;
- output: coupled 120 Hz reference/control grid;
- final source pose: retained with no more than one terminal grid hold.

Objects are initialized once from frame zero after one common scene grounding shift. They are free MJX-Warp bodies thereafter; reference object poses are never written back into physics.

## Generate the fixed physical profile

Use the materialized, pinned DexStream checkout that training will use:

```bash
PYTHONPATH=. python tools/generate_manorl_asset_manifest.py \
  --asset-root /path/to/dexstream_digital_assets \
  --hand-operator cheyingtong \
  --output /local/data/manorl_raw_transfer/cheyingtong_asset_manifest.json
```

The trainer and compiler must use the same manifest and asset root:

```bash
export MANORL_ASSET_MANIFEST=/local/data/manorl_raw_transfer/cheyingtong_asset_manifest.json
export MANORL_ASSET_ROOT=/path/to/dexstream_digital_assets
```

## Compile one verified package per source

```bash
PYTHONPATH=. python -m tools.compile_manorl_trajectory_package \
  --output /local/data/manorl_raw_transfer/remake-v978-raw120.mtp \
  --dataset-path /mnt/nas-222-projects/mocap_v2/lance_datasets/human_p1_remake_v3/human_p1_remake_clean.lance \
  --dataset-version 978 --pairs all \
  --reference-fps 120 --hand-side right --drop-uncontrolled-hands \
  --pre-padding 0 --post-padding 0 --raw-transfer \
  --target-object-overrides bottle:18 \
  --asset-manifest "$MANORL_ASSET_MANIFEST" --asset-root "$MANORL_ASSET_ROOT"

PYTHONPATH=. python -m tools.compile_manorl_trajectory_package \
  --output /local/data/manorl_raw_transfer/guangguan-v530-raw120.mtp \
  --dataset-path /mnt/nas-222-projects/mocap_v2/lance_datasets/human_p1_guangguan/human_p1_guangguan_clean.lance \
  --dataset-version 530 --pairs all \
  --reference-fps 120 --hand-side right --drop-uncontrolled-hands \
  --pre-padding 0 --post-padding 0 --raw-transfer \
  --target-object-overrides bottle:18 \
  --asset-manifest "$MANORL_ASSET_MANIFEST" --asset-root "$MANORL_ASSET_ROOT"
```

Compilation runs Lance/PyArrow only in isolated workers. Training loads JSON and mmap NPY arrays from both verified packages.

## Conservative first policy profile

The initial learned correction is intentionally bounded:

```text
wrist XYZ increment = 1 mm/control
wrist XYZ cap       = 10 mm
joint scale         = 1x base scales
joint cap           = 1x base caps
residual penalty    = 1.0
finger repair mask  = all 22 joints
contact intent      = action-specific raw gesture mapping
```

All finger joints may be corrected because avoiding an unintended collision can require moving a non-contact finger. Contact reward remains action-specific; it does not demand five fingertips for pinch, lateral, or adduction grasps.

Example package-backed smoke launch:

```bash
PYTHONPATH=. python -m tools.train_manorl_cube1 \
  --output /local/runs/raw-transfer-smoke \
  --trajectory-packages /local/data/manorl_raw_transfer/remake-v978-raw120.mtp \
  --trajectory-packages /local/data/manorl_raw_transfer/guangguan-v530-raw120.mtp \
  --all-pairs --raw-transfer \
  --reference-fps 120 --hand-side right --drop-uncontrolled-hands \
  --pre-padding 0 --post-padding 0 \
  --target-object-overrides bottle:18 \
  --expected-contact-mode raw_gesture --residual-joint-mode all \
  --action-penalty-scale 1.0 \
  --position-scale 0.001 --max-position-offset 0.01 \
  --joint-scale-multiplier 1.0 --joint-max-offset-multiplier 1.0 \
  --unified-object-batch --evaluation-enabled false
```

Set the ordinary GPU/device-transition/CCD capacity flags appropriate for the training host. Do not begin a long run until zero-residual replay has measured whether failures are primarily contact acquisition, load-bearing transport, or release.

## Acceptance evidence

A trajectory is not accepted from reward or natural completion alone. Report, by source/object/action:

- contact acquisition and distinct contacting fingers;
- free load-bearing duration and bottom clearance when the reference lifts;
- object position/orientation tracking during transport;
- final support, release, speed, and drift;
- residual magnitude and saturation frequency;
- natural completion versus deviation termination.

Increase correction bounds only when successful episodes saturate the conservative envelope and the physical failure occurs in the same controlled axis.
