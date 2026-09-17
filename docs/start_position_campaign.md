# Physical position-only augmentation at120Hz

The campaign uses only the42 strict seeds in the published Cheyingtong repair
bundle. B035 remains excluded. Every variant changes both the initial physical
handXYZ and the early desired wristXYZ; object initialization and all velocities
are unchanged. A quintic residual vanishes at C-1, leaving the repaired command
suffix and central-difference velocity at C exact. There is no physical-state
reset at the merge. Commands run at120Hz with four480Hz substeps and a prestep
frame0, so N recorded states contain N-1 integrated intervals.

## Sampling and execution

Use `tools.run_start_augmentation` from the project environment:

```bash
PYTHONPATH=. python -m tools.run_start_augmentation \
  --bundle /path/to/cheyingtong120_repair43_20260917 \
  --asset-root /path/to/dexstream_digital_assets \
  --plan outputs/start_plan.json --make-plan

PYTHONPATH=. python -m tools.run_start_augmentation \
  --bundle /path/to/cheyingtong120_repair43_20260917 \
  --asset-root /path/to/dexstream_digital_assets \
  --plan outputs/start_plan.json --output outputs/start_campaign --batch-size 32
```

The clock is fixed to120/480Hz and the default guard is0.1s before parent contact.
The plan has four
radii (3,5,7.5,10cm) and eight spatial octants:32 candidates per seed,1344 total.
Seed2026091701 gives reproducible independent draws. Each cell samples at most16
initially collision-free directions without leaving its octant or changing its
radius. All preflight rejections remain in the plan; exhaustion yields an explicit
geometry exclusion, not a fallback to a smaller or zero displacement. No physical
rejection is replaced by another draw. `--rows` and `--slot` select fixed subsets
for diagnosis; `--scalar --batch-size 1` executes a single-world comparison.

Same-parent candidates run as independent MJX-Warp worlds. Each world retains
its own qpos,qvel,controller integral,solver warmstart and controls. The wrist PID
kernel uses the same gains/equations as scalar replay, checked against the scalar
kernel; host finger control reuses the exact existing scalar function. Batch
processing amortizes model compilation and kernel launch overhead, not physics
or validation. A separate process bounds memory lifetime for each parent.

The queue pins source code and the input plan's source/model/recipe identities.
A failure to execute stops explicitly with row/log information. Physical task
failures are recorded and the remaining candidates continue. A fresh output root
is required; partially completed episodes are not silently overwritten/retried.

## Acceptance and artifacts

`status.json` exposes the running child PID, completed parents and current
accepted count. `summary.json` aggregates rowwise results. Each variant contains
`trace.npz` (actual states, desired/parent desired, repaired finger deltas,120Hz
controls and480Hz substep controls), `metrics.json`, `provenance.json`, contact
reconstruction and `result.json`. The plan binds the parent/source UUID and trace
hash independently of the generated variant ID. The predecessor replay hand is
Sunke; this does not assert the raw capture operator was Sunke.

Acceptance requires the unchanged action-specific physical+pose gates, no
hand/environment contact during the premerge interval, and no>=25ms sampled
unsupported/no-hand-contact interval above2cm lift. Initial geometry must be
nonpenetrating with the scene. Existing hand self-contact is not treated as a
scene collision. Contact evidence is reconstructed at120Hz, not480Hz force
telemetry; contact counts do not establish force closure.

The generated trajectories remain measured dynamics, not shifted recordings.
Accepted counts are measured, never inherited from the seed or guaranteed by the
1344-slot plan. Rendering must read the new measured qpos. Original repaired
parents and original captures remain immutable.
