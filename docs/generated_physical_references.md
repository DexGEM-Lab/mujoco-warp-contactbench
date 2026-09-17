# Generated physical trajectories as training references

The canonical v2.3 exporter can preserve measured120Hz Cheyingtong trajectories
without integrating physics again. A generated row is not a human capture. Its
operator/betas name the physical Cheyingtong right hand, while the original raw
capture index, metadata, operator, betas and UUID remain in generation provenance.
Sunke is the predecessor simulation hand, not necessarily the raw capture operator.

## Export accepted saved states

`tools.export_saved_generated_references` takes a JSON list of
`{"id":"unique-variant-id","run_path":"/path/to/physical/run"}`. Runs must contain
accepted `result.json`, hash-bound `trace.npz`, `metrics.json` and `provenance.json`;
the trace may be in a `replay/` subdirectory. Rejected candidates are refused.

```bash
PYTHONPATH=. python -m tools.export_saved_generated_references \
  --selection accepted_selection.json --manifest cheyingtong_manifest.json \
  --asset-root /path/to/dexstream_digital_assets \
  --output generated_starts.lance --expected-contact-mode five_fingertips
```

The exporter checks physical-model arrays against the saved execution, computes
native keypoints/MANO pose and CPU-reconstructed normal contact forces, and uses
the existing canonical builder/schema. No GPU integration is performed. These
forces are not the recorded Warp contact solution.480D generation observations
use the explicitly declared expected-contact mode and saved source-object targets.
A later RL run tracking measured generated motion constructs its own observations.

Actual hand/object states, source references, generated UUIDs and actuator targets
roundtrip through Arrow/Lance. Float32 fields are compared exactly at their stored
precision; rotations use quaternion/matrix equivalence. The first state is
preintegration, and N states contain N-1 transitions. Canonical source indices name
the120Hz reference grid; original fractional100Hz correspondence is retained in
checkpoint metadata rather than silently truncated to integers.

Saved `ctrl` values are arrival-frame final-substep actuator targets, not normalized
policy actions. `finger_target_delta` is a separate correction after absolute
donor preload, never folded into desired. Reproduction requires the retained NPZ
substep controls and original controller recipe; one target per120Hz frame cannot
reproduce a480Hz feedback controller exactly.

## Explicit full-episode consumption

The raw-capture loader default still excludes generated data. Opt in using
`--generated-reference`, zero pre/post padding,120Hz and the exact hand manifest:

```bash
export MANORL_ASSET_MANIFEST=/path/to/cheyingtong_manifest.json
PYTHONPATH=. python -m tools.compile_manorl_trajectory_package \
  --dataset-path generated_starts.lance --dataset-version VERSION \
  --output generated_starts.mtp --generated-reference \
  --reference-fps 120 --hand-side right --pre-padding 0 --post-padding 0
```

This mode requires canonical v2.3 provenance,120/480Hz clocks, explicit prestep
frame0 and the matching physical-hand manifest hash. It resolves action labels
from `trajectory_metadata.gesture`, decodes the canonical right/left slots, keeps
all frames and world coordinates, and does not add grounding shifts, crop away
the initial perturbation or append edge holds. Real movement annotations remain.
The mode is bound into the package selection contract; raw and generated modes
cannot silently share a package.

The trainer accepts the same opt-in flag with the package (or direct Lance),
`--reference-fps 120 --pre-padding 0 --post-padding 0`, and matching manifest.
No training is launched by export or package compilation. The training reference
is the recorded actual hand/object motion, not the stored nominal desired/actions.

The historical synthetic-row validator assumes source betas are unchanged; it is
not an appropriate cross-hand validator. This export path instead checks the
explicit physical profile, preserves raw metadata separately, verifies native
state/target/observation roundtrips and exercises actual generated discovery plus
complete-episode decoding. Do not falsify source metadata to satisfy a legacy check.
