# September 15 right-hand capture profile

This profile prepares `dexgem_vla_demo_cma2lance_20260915_030735.lance` v4.
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

The source now contains132 rows, all with `hand_names=[left,right]`, operator
`cheyingtong`, and nominal120Hz capture (integer metadata119 or120).
A process-specific source-matched hand manifest is required; the global
training manifest remains sunke for compatibility.

An explicit `--target-object-overrides egg_cup:04` resolves the compound
cup/bowl annotations while retaining the annotated interval and every scene body.
All132 rows are selected with the right-hand-only profile:

| Target/action | Rows |
|---|---:|
| egg_ellipsoid:01 | 30 |
| cylinder7:02 | 30 |
| bowl:03 | 29 |
| egg_cup:04 | 27 |
| mayonnaisebottle:05 | 10 |
| pitcherbase:06 | 3 |
| bowl:07 | 2 |
| bowl:09 | 1 |

Sixteen pouring rows contain `egg_cup,bowl` or `bowl,egg_cup` in one
`object_move.object_name`. Direct inspection of all27 pouring captures establishes
`egg_cup` as the primary manipulation target: cups move roughly29–40cm and rotate
through pouring while bowls mostly remain near their initial location, with some
recorded drift. The explicit override is scoped to action04 and requires the
chosen target to appear in both scene and original movement annotation. It
never chooses the first comma-separated name or silently invents a missing
object. Original source annotations/UUIDs/poses/windows remain unchanged.

The low-lift bowl09 UUID `61e69a08-e65a-4a5a-8bfd-8d1d57e6a0f9` was removed by
the owner in v4; v3 still exists historically. Old v3 packages must not be reused
for this v4 profile. There are132 current rows and eight object/action pairs.

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
  --dataset-path "$MANORL_DATASET_PATH" --dataset-version 4 \
  --target-object-overrides egg_cup:04 \
  --hand-side right --drop-uncontrolled-hands --reference-fps 120 \
  --pre-padding 60 --post-padding 250 \
  --output outputs/sept15-right/approved.mtp
```

This compiles all132 v4 rows with the same explicit target override. The
selection is hash-bound into the package and recorded in checkpoint metadata;
a trainer requesting a different override is rejected. Packages built with an
explicit profile also require the same asset profile when loaded.
`MANORL_ASSET_MANIFEST` must be set in the trainer environment, and both
`--drop-uncontrolled-hands` and `--target-object-overrides` must match compilation.

A future trainer command should explicitly include:

```text
--hand-side right --drop-uncontrolled-hands --target-object-overrides egg_cup:04
--position-scale 0.002 --max-position-offset 0.01
--expected-contact-mode five_fingertips
--reference-fps 120 --pre-padding 60 --post-padding 250
```

Server readiness is an operational snapshot, not a durable GPU reservation.
Preparing a launcher does not start training.

## Dense-scene constraint storage

The all-object model has1450geoms (egg_cup598 collision pieces and
egg_stick_rack721). Server1's8-world zero-policy probe exceeded default
`njmax=512` at553 constraints. With `--constraint-capacity 2048`,64steps and
indexed reset passed, with roughly740constraints in the later steps. This
changes allocation capacity only; meshes, gravity, contact rules and gains
are unchanged. The Sept15 preset exports `MANORL_CONSTRAINT_CAPACITY=2048`;
the generic trainer default remains512. Full4096 training memory must still
be validated against the selected CCD and solver workspaces.

The4096-world preflight also emitted broadphase-overflow warnings with the old
128slots/world minimum: it requested up to891467total candidate slots. Increase
`--contacts-per-world` for this scene; the preset uses512, or2097152slots at4096
worlds. A zero exit code is not sufficient: preflight logs must contain no
collision/constraint overflow. No collision shape or contact rule is removed.
