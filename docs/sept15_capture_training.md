# September 15 right-hand capture profile

This profile prepares `dexgem_vla_demo_cma2lance_20260915_030735.lance` v3.
It does not start training. The source remains immutable.

## Requested controls

For this capture use pre60/post250 at120Hz: a0.5-second approach margin
and about2.08seconds after the annotated movement end. The shorter pre-window
does not retime the source; insufficient margins use the existing edge hold.

New trainer defaults (`TrainingBudget`, CLI, `train.sh`) are:

- XYZ action contribution ±0.002 m per axis, cumulative cap ±0.01 m;
  decay stays 0.9, so offset = clip(0.9 * previous + 0.002 * action, ±0.01).
- `--expected-contact-mode five_fingertips`: expected sites are `thumb_ip`,
  `index_dip`, `middle_dip`, `ring_dip`, `pinky_dip`. These are the existing
  distal collision bodies, not newly introduced point sensors. Their mask
  enables all22 finger residual axes, while proximal/palm contacts are not
  expected reward sites. All physical contacts remain in observations.
- Hand-joint scales/caps, wrist rotation, servo gains, contact force threshold,
  gravity and collision geometry are unchanged.

`EnvironmentConfig` and `ResidualActionConfig` retain their older generic
source-mapping/3mm/3cm defaults for existing library consumers. New training
passes its explicit2mm/1cm and five-fingertip choices into the environment.
To recreate historical trainer settings, pass `--position-scale 0.003
--max-position-offset 0.03 --expected-contact-mode source_mapping`.

The contact mode is recorded in the strict checkpoint signature. Legacy
sidecars without this field mean `source_mapping`; they cannot silently resume
as `five_fingertips`. Inference restores the recorded contact mode. Existing
residual-action signature validation also rejects mismatched XYZ scales.

## Actual source coverage

The source contains133 rows, all with `hand_names=[left,right]`, operator
`cheyingtong`, and nominal120Hz capture (integer metadata119 or120).
A process-specific source-matched hand manifest is required; the global
training manifest remains sunke for compatibility.

After allowing underscore object names and selecting the right hand,117 rows
have an unambiguous target and decode successfully:

| Target/action | Rows |
|---|---:|
| egg_ellipsoid:01 | 30 |
| cylinder7:02 | 30 |
| bowl:03 | 29 |
| egg_cup:04 | 11 |
| mayonnaisebottle:05 | 10 |
| pitcherbase:06 | 3 |
| bowl:07 | 2 |
| bowl:09 | 2 |

Remaining16 source rows (zero-based106,107,109,112–124) contain one
`object_move` record whose name is `egg_cup,bowl` or `bowl,egg_cup`.
They are NOT converted to a target by taking the first name. The owner must
confirm which object should receive policy tracking/reward before these rows
can join a complete133-row training set. Existing all-pairs discovery excludes
ineligible metadata rows; inspect the candidate count and do not label117 as
full133 coverage. Neither the source nor these16 annotations were rewritten.

## Prepare the source hand and omit the left model

From the repository root, with the configured Python environment:

```bash
python tools/generate_manorl_asset_manifest.py \
  --hand-operator cheyingtong \
  --output outputs/sept15-right/asset_manifest.json
source configs/manorl/sept15_right.env
```

`MANORL_ASSET_MANIFEST` explicitly selects the process-local physical profile.
Its source commit and file hashes are validated; modern captures must match
its operator and selected-hand betas. Do not change it inside a running
simulation. This command does not overwrite the default training manifest.

`--hand-side right` alone controls the right hand but retains passive hands.
Add **`--drop-uncontrolled-hands`** to physically omit left-hand references and
model. Both trainer and package compiler accept the flag; the selection is
persisted and mismatched package requests are rejected. Object states, UUIDs,
source-frame provenance and all passive scene bodies remain intact.

For a future package compilation of the explicitly approved selection, use:

```bash
python -m tools.compile_manorl_trajectory_package \
  --dataset-path "$MANORL_DATASET_PATH" --dataset-version 3 \
  --hand-side right --drop-uncontrolled-hands --reference-fps 120 \
  --pre-padding 60 --post-padding 250 \
  --output outputs/sept15-right/approved.mtp
```

At present this yields only the117 unambiguous candidates, not the unresolved
16 rows. Packages built with an explicit profile record its asset provenance
and require the same profile when loaded. `MANORL_ASSET_MANIFEST` must therefore
also be set in the trainer environment. Retain the process-local manifest and
use the same drop-hand flag at both compile and train time.

A future trainer command should explicitly include:

```text
--hand-side right --drop-uncontrolled-hands
--position-scale 0.002 --max-position-offset 0.01
--expected-contact-mode five_fingertips
--reference-fps 120 --pre-padding 60 --post-padding 250
```

This work did not launch training or modify either server deployment. Server
availability is an operational snapshot, not a durable GPU reservation.
