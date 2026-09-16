# Local Cheyingtong trajectory repair

`tools/repair_local_contacts.py` reexecutes the43 archived Cheyingtong base
trajectories at120Hz control /480Hz physics. Each trial uses the original physical
initialization and free-object simulation. It does not execute measured hand
positions as a replacement for the archived actuator-command recipe.

## Input and time

Supply the completed100Hz Cheyingtong output directory, its pinned Cheyingtong
asset manifest, and a materialized DexStream asset root. The input directory must
contain `comparison.json` and each row's `manifest.json`, `initial.npz`, and
`replay/trace.npz`. Trace identities are checked against the comparison. The
compiled native geometry, mass, friction and actuator arrays must equal the
archived model; only the requested timestep changes.

```bash
PYTHONPATH=. python tools/repair_local_contacts.py \
  --baseline /path/to/cheyingtong_base43_v1 \
  --manifest /path/to/cheyingtong_asset_manifest.json \
  --asset-root /path/to/dexstream_digital_assets \
  --output outputs/cheyingtong120/baseline --hz 120
```

`--rows B_row045` selects one
row; omitting it selects all43 with prior failures first. An existing row output
is rejected rather than overwritten. `--hz 100` is available for controlled
controller-compatibility checks, not the new delivery clock.

Old commands span `(N-1)/100` seconds. Targets are interpolated on a120Hz grid;
hand angles are unwrapped and object-reference quaternions use SLERP. The final
source target is held until the next regular grid boundary, adding less than
1/120s, recorded as `terminal_grid_hold_s`. Velocity feedforward and controller
integration use the actual new timestep. Source timestamps/frame coordinates
remain explicit correspondence fields; `time` is the executed clock.

## Local recipe

A recipe is tied to one row and uses seconds, never ambiguous source-frame
numbers. Smooth cubic ramps preserve the unmodified prefix and avoid target
jumps. A phase without `end_s` holds through the terminal frame.

```json
{
  "schema": "manorl.local-contact-repair.v1",
  "row_id": "B_row045",
  "note": "Example schema, not a validated repair",
  "phases": [
    {
      "start_s":0.8, "rise_s":0.15, "end_s":2.8, "fall_s":0.15,
      "finger_deg":{"j2_index_mcp_flex":1.0},
      "translation_m":[0.001,0.0,0.0]
    }
  ]
}
```

Joint names must come from the selected manifest. `rotation_deg` is a small
rotation vector, with `rotation_frame` set to `world` (default) or `wrist`.
Wrist edits change desired pose. Finger edits are extra actuator target angles
applied **once after** the existing damping/dry-friction feedforward and absolute
donor preload blend. This matters because editing only desired fingers would be
masked by a full donor-preload blend. Corrections and actual physical hand states
are saved separately. The local envelope caps translation at15mm and rotations
and finger-target deltas at10deg; reported experiments should begin much smaller.

Apply with `--recipe /path/to/recipe.json --rows B_row045` and a new output root.
The first runner physically replays the untouched prefix on every trial. It does
not claim arbitrary-frame checkpoint restoration. Saved controller integrals and
warmstarts are diagnostic fields; a validated complete restore contract would be
required before using them to accelerate trials. Final candidates always require
a complete continuous replay from original initialization.

## Interpretation

`result.json` distinguishes basic functional gates and the same physical+pose
gates as the archived43 comparison. Hold requires final hand contact without
support and full reference-rotation error<15deg. Placement requires the intended
support, release, upright tilt<10deg and final position error<8cm. Both require
reference-relative lift and the prior airborne multi-finger-contact threshold.
Hand tracking and penetration are reported independently; changing a wrist target
is not mislabeled as unchanged-reference fidelity.

Contacts are native geometric reconstructions from measured states, not recorded
GPU contact force or force closure. Inspect actual motion before accepting a
repair. A transient lift is not stable transport, and container tilt does not
measure liquid delivery. Source/100Hz results remain immutable;120Hz baseline
results must be measured before a contact change can be credited.
