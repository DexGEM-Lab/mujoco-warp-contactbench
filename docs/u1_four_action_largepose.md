# Four-action large-pose U1 checkpoint

Scope: new003/006/007/009 children only. No005 changes, production replay,
export, or old campaign mutation. This checkpoint implements immutable contract
preparation and pure synthesis; **physical runtime and pilots are not implemented**.

## Prepare and inspect

Use the project virtualenv Python and run from the feature worktree:

```sh
python -m tools.run_u1_largepose prepare \
  --registry /path/to/old/contracts/registry/registry.json \
  --output outputs/u1_four_action_largepose_v1/contracts
python -m tools.run_u1_largepose audit \
  --registry outputs/u1_four_action_largepose_v1/contracts/registry/registry.json \
  --output outputs/u1_four_action_largepose_v1/source_audit
python -m pytest -q tests/test_u1_largepose.py
```

Both commands refuse existing output directories. Preparation verifies parent
bundle SHA values, acceptance metadata and installed runtime versions, copies
only the four approved bundles, and records the source registry SHA/digest.
The new registry retains unchanged per-parent metadata and hashes.

The saved digest-bound plan has160 slots/action,16 directions/slot, exact
5/10/15cm radii and single-coordinate intrinsic XYZ ±30degree rotations.
Radius counts are54/53/53, octants20 each, orientations27/27/27/27/26/26.
The last16 slots use independent spatial draws. Hosts load these saved floating
values directly; replay must never regenerate a plan to compare floating bytes.

`prefix_targets` calls the historical discrete-C1 quintic constructor with a
4cm endpoint-smooth vertical arc and appends the complete unchanged parent.
Actual initial fingers and all non-hand state remain untouched. The duration
floor uses actual initial-to-target displacement/angle and the factor1.875;
callers cannot request fewer frames. Actual sampled C1/arc speed is reported,
not qualified or silently treated as bounded by the nominal residual speed.

`extend_reference` provides a pure reconstruction function and the audit stores
its arrays in `extended_reference.npz`. Parent mapping is `[0]*prefix_frames +
range(parent_frames)`. Arrival command indices are1..newN−1; parent command
indices use that explicit mapping; original source-frame IDs are separately
retained when present. Source object arrays, teacher and timestamps retain their
complete suffix. Execution time starts at0; pour560..840 and release1130 map by
adding the prefix length. Historical source metadata remains unchanged provenance.
A future exporter still needs a single execution-clock active movement interval;
this checkpoint does not present source-rate `object_move` as shifted120Hz data.

## Structural conflicts requiring resolution

Immutable source audit found initial limited-joint violations **before** any
perturbation:

| Action | Violating joints | Largest violation |
|---|---|---|
|003|none|0|
|006|thumbIP, indexPIP, ringPIP|0.180825rad|
|007|index/middle/ring/pinkyPIP|0.181486rad|
|009|indexPIP, pinkyMCP abduction, pinkyPIP|0.181425rad|

Strict initial-range rejection and unchanged initial fingers exclude three
parents simultaneously. No clamping or grandfathering is implemented.

The canonical first-target XYZ interval speeds are0,1.596707,2.154173,0.813722m/s
for003/006/007/009. Exact discrete C1 therefore cannot obey a full0.30m/s speed
cap on three parents at any duration. The0.4s zero-offset003 prefix itself peaks
at0.349484m/s because of its4cm arc, so it needs extra duration under that cap.
006/007 inherited first Euler speeds also exceed90degrees/s. These are source
facts, not failed physical trials. Every synthesized audit marks
`duration_qualified=false` and `accepted=false`.

Decision needed: does range preflight reject inherited finger violations, or
only newly introduced root violations? Which explicit residual speed definition
separates added motion from inherited C1 endpoint velocity? Until resolved,
there is no physical pilot or production command. No scientific claim of640
accepted children follows from the synthesis tests.
