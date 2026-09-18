# Refined RL episode import and training

The refined RL dataset is a collection of complete simulated episodes, not raw
capture windows and not canonical ManoRL-generated references. Its compact row
schema carries:

- `index.scene`, `index.action_code`, UUID and seed UUID;
- one right-hand 28-DoF target track;
- one object position/axis-angle track;
- exact timestamps; and
- `provenance.source_rl_*` identity.

It omits raw-capture movement annotations and canonical generated-reference
checkpoint provenance. Import it only with `--rl-episode-reference`; this keeps
both existing contracts strict.

## Clock and hand contract

All replay and training output uses coupled 120 Hz reference/control and 480 Hz
physics. The importer preserves elapsed time from `timestamp` and resamples to
120 Hz. It never treats `trajectory_metadata.data_fps` as authoritative: in the
600-row mayonnaise subset, 437 declarations disagree with timestamp spacing
(481 rows are observed at 200 Hz and 119 at 120 Hz).

The runtime hand is fixed by an explicit asset manifest. For this campaign the
manifest must name `cheyingtong`; source operator and source MANO shape do not
select the physical model. The package records the manifest hash and each row's
source RL path/version/row/UUID.

Generate the deterministic manifest:

```bash
python tools/generate_manorl_asset_manifest.py \
  --hand-operator cheyingtong \
  --output outputs/gym2mjx-refined/asset_manifest.json
```

## Compile the mayonnaise package

The compiler requires `pylance==7.0.0`; older `0.24.1` readers panic on this
dataset's `lance.encodings21.PageLayout` pages. Lance and PyArrow remain confined
to isolated compiler workers. The produced MTP is the only production trainer
input.

```bash
PYTHONPATH=$PWD python -m tools.compile_manorl_trajectory_package \
  --output /local/package/root/mtp-refined-mayo-v1-cheyingtong-f120 \
  --dataset-path /mnt/nas-222-project/sunjieqiang/new_vla/mano_rl_refined_100_per_action.lance \
  --dataset-version 1 \
  --pairs mayonnaisebottle:01,mayonnaisebottle:02,mayonnaisebottle:03,mayonnaisebottle:04,mayonnaisebottle:05,mayonnaisebottle:08 \
  --reference-fps 120 --hand-side right \
  --pre-padding 0 --post-padding 0 \
  --rl-episode-reference \
  --asset-manifest outputs/gym2mjx-refined/asset_manifest.json \
  --fixed-hand-operator cheyingtong
```

Compilation is fail-closed: every discovered row is either decoded or appears
in the hash-bound rejection ledger. Never overwrite the source Lance dataset.

## Measure physical replay success

The success metric is MJX execution under zero residual action: normal reference
horizon completion is success; the standard object-target deviation termination
is failure. Stored source object motion alone is not replay evidence.

```bash
PYTHONPATH=$PWD python -m tools.replay_rl_episode_package \
  --trajectory-package /local/package/root/mtp-refined-mayo-v1-cheyingtong-f120 \
  --asset-manifest outputs/gym2mjx-refined/asset_manifest.json \
  --output outputs/gym2mjx-refined/mayo-zero-replay.json \
  --device gpu \
  --joint-scale-multiplier 1.0 \
  --joint-max-offset-multiplier 1.0
```

The JSON result contains aggregate and per-action success rates plus every row's
termination reason, target error, UUID and original RL provenance.

## Train from the package

Use the same fixed manifest and select the standard MTP with zero source
padding. Both joint residual increments and cumulative joint caps are 1× for
this campaign. The trainer process does not import Lance or PyArrow.

```bash
export MANORL_ASSET_MANIFEST=$PWD/outputs/gym2mjx-refined/asset_manifest.json
JAX_PLATFORMS=cuda python -m tools.train_manorl_cube1 \
  --trajectory-package /local/package/root/mtp-refined-mayo-v1-cheyingtong-f120 \
  --dataset-path /mnt/nas-222-project/sunjieqiang/new_vla/mano_rl_refined_100_per_action.lance \
  --dataset-version 1 \
  --pairs mayonnaisebottle:01,mayonnaisebottle:02,mayonnaisebottle:03,mayonnaisebottle:04,mayonnaisebottle:05,mayonnaisebottle:08 \
  --reference-fps 120 --hand-side right \
  --pre-padding 0 --post-padding 0 \
  --joint-scale-multiplier 1.0 \
  --joint-max-offset-multiplier 1.0 \
  --output outputs/manorl/refined-mayo-cheyingtong-f120-joint1x
```

Choose `--num-envs`, update budget and GPU only after replay and a bounded
memory/throughput smoke on the target server. Do not displace existing jobs.
