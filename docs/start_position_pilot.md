# Repaired-seed initial-position pilot

Run from the repository root with the GPU-enabled project interpreter:

```sh
python -m tools.pilot_start_augmentation \
  --bundle /path/to/cheyingtong120_repair43_20260917 \
  --asset-root /path/to/dexstream_digital_assets \
  --output outputs/new-start-pilot --guard .1 --radius .1
```

The output root must be new. The runner verifies all 42 accepted seeds against
published trace hashes and reconstructs their desired commands and post-preload
finger deltas exactly. B_row035 is excluded. It then runs A_row035 and B_row043,
each zero, +10cm and -10cm, serially in fresh scalar processes. Native model arrays
are checked by the existing repair model binder before any dynamics.

Only initial hand XYZ and the position-command prefix change. A quintic residual
is exactly zero from C-1 onward, preserving the byte-exact repaired suffix and
central-difference velocity at C. C is the earlier of the measured first-contact
minus guard and the source movement-start minus guard (with the historical 0.30s
movement-window floor). Commands remain at120Hz, physics at480Hz, frame0 prestep,
N-1 intervals. The repair replay runs continuously without any merge-state reset.
Wrist rotation, fingers, preload, object initialization, gains and friction are
unchanged. The .1s guard and10cm radius are experimental, not validated limits.
The plan reports analytical peak residual speed/acceleration alongside the
proposed .5m/s and5m/s² planning caps; these are not hardware limits or clamps.

Both signs must be initially penetration-free along a common direction. The
fixed search is world X, world Y, then horizontal approach tangent. Rejected
geometry is recorded; failure to find a direction is an exclusion, never a zero
fallback. Hand self-contact does not count as hand/scene penetration.

Each trial saves initial and target arrays, parent targets, actual state,
controller/substep controls, clocks, source/recipe hashes, native model hashes,
contact chronology, merge-state differences, and separate physical+pose gates.
Acceptance additionally requires no premerge hand/scene contact and no sampled
unsupported/no-hand run above2cm lasting25ms. Contact reconstruction is120Hz
geometry, not force closure or480Hz telemetry: a transient between command frames
can escape these checks. Three consecutive unsupported120Hz samples are treated
conservatively as25ms occupancy. No production claim should exceed this evidence.

A zero replay must exactly preserve commands, initial state/control, physics
clock, controller and model hashes and pass all physical and additional gates.
Mixed-unit qpos drift is diagnostic, not an equality gate: GPU contact dynamics
can amplify floating-point differences. A failed invariant or gate stops the
entire pilot for direct inspection. Exceptions are serialized to failure.json. Nonzero physical failures
are retained and the remaining bounded trials continue. summary.json lists only
completed or geometrically excluded trials; status.json distinguishes completion
from a stopped process. This is a pilot, not a production exporter or batch runner.
