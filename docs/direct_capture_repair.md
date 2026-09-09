# Direct repair of human-capture replay

The September 9 Guangxue capture contains real human motions that fail when
replayed through the pinned MANO hand and free-object contact solver. The
repairs here are explicit wrist/finger target curves and, for the bowl, a small
initial object translation. No learned policy is fitted. Objects are initialized
once and then moved only by the physics solver.

## Verified samples

Source dataset: `dexgem_vla_demo_guangxue_cma2lance_20260909_060700.lance`, version 5.
Rows below are zero-based and patches also bind the immutable source UUID.

| Source row | Task | Patch | Observed result |
|---|---|---|---|
| 49 | Lift bowl from support and hold | `guangxue_bowl09_row49_v3.json` | Maximum lift 25.3 cm; stays at least 20.26 cm above initial center height during an extra 10 s hold. Final bowl tilt 4.94°, last-second movement <0.2 mm. |
| 19 | Lift bottle, tilt over bowl, return | `guangxue_bottle05_row19_v2.json` | Maximum lift 12.46 cm, retained during tilt, returned upright with hand withdrawal. At source frame 487, pose error is 1.43 cm / 2.68°. Final position error 2.78 cm and tilt 1.06°. |
| 31 | Lift pitcher by handle, tilt, return and release | `guangxue_pitcher06_row31_v1.json` | Frozen command replay lifts 16.66 cm and ends upright at 0.45°, with no finger contact. Final position error 5.56 cm. |
| 0 | Move bowl from stove to front table | `guangxue_bowl03_row0_v1.json` | Repaired lift/carry/placement finishes upright at 0.39°, 2.88 cm from edited target; 25.05 s authored sequence includes an extended grip hold. |
| 36 | Move bowl from front table to stove | `guangxue_bowl07_row36_v1.json` | Bowl returns onto its supporting block, then hand releases and withdraws. Final tilt 0.35° and 4.17 mm error to edited target; duration 25.25 s. Initial bowl Z is lowered 7.74 mm onto the table. |

These five examples cover the capture's five action categories. The remaining
54 rows are unchanged and have not been repaired or validated.
The bowl's final position is about 6.2 cm from its recorded reference. The
bottle has transient position errors during transport. There is no fluid model:
“pour” refers to the bottle's position/orientation over the bowl, not measured
liquid delivery.

## Why both trajectory and contact settings matter

There are two separable mechanisms:

1. **Retargeted grasp geometry.** Original finger targets leave supporting
   fingers away from the object or put intermediate segments through it. The
   bottle additionally gets pushed away if fingers close during approach.
   Repairs define an object-relative wrist pose, smoothly acquired finger
   targets, and an open-hand approach. Release restores the source withdrawal.
2. **Soft-friction creep.** A fitted bottle grip still slid while contact support
   nearly balanced its 2.943 N weight. Keeping hand trajectory, mass, friction
   coefficients, geometry and actuator settings fixed, elliptic friction with
   `impratio=100` reduced late static descent from approximately 8 mm/s with
   elliptic/impratio=1 to 0.12 mm/s. Default pyramidal friction lost the grip.
   The higher tangential impedance is recorded explicitly in each patch.

The source bowl hand trajectory still fails with the repaired solver profile;
the repaired hand trajectory still slips with the default profile. Neither
intervention alone explains the successful held-bowl example. The patches do
not modify project-wide defaults, gravity, mass, friction coefficient or gains.

The heavier pitcher adds a third issue: its 1.28 kg load causes roughly
10–14 cm of wrist sag with the pinned 100 N/m position servo. Its target curve
includes staged load compensation after acquisition. Construction-time object
feedback helped author the curve; the delivered replay applies the frozen
hashed actuator targets without feedback. To release the handle, the thumb
first lifts clear and the fingers then withdraw sideways. Radial withdrawal
while the thumb still occupies the handle was observed to topple the pitcher.

For the moving-bowl sample, the held-bowl primitive is translated into the new
scene and then carried to the source goal. Initial object orientation is aligned
using the bowl's axial symmetry. This changes approach, timing and the wrist/
finger trajectory substantially; it is task-preserving repair, not minimal
framewise denoising. Per-command reference indices document the correspondence.

This mechanism is consistent with MuJoCo's
[Preventing slip](https://mujoco.readthedocs.io/en/latest/modeling.html#preventing-slip)
guidance. `impratio` changes relative friction-constraint hardness; it is not a
larger Coulomb coefficient and does not guarantee no slip for every grasp.

## Reproduce live physics

From a checkout with the pinned materialized assets and dependencies:

```bash
export PYTHONPATH=.
export DISPLAY=:1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
DATASET=/path/to/dexgem_vla_demo_guangxue_cma2lance_20260909_060700.lance

python tools/replay_repaired_capture.py \
  --dataset "$DATASET" \
  --patch sim/manorl/task_assets/replay_repairs/guangxue_bowl09_row49_v3.json \
  --post-padding 1000 --output outputs/object_repair/bowl-run \
  --view --loop

python tools/replay_repaired_capture.py \
  --dataset "$DATASET" \
  --patch sim/manorl/task_assets/replay_repairs/guangxue_bottle05_row19_v2.json \
  --output outputs/object_repair/bottle-run --view --loop
```

Use a new output directory for each run. Omit `--view --loop` for a bounded
headless recording. `--solver default` explicitly runs the same edit under the
old pyramidal/impratio=1 profile for comparison. It is expected to fail sustained
holding, not silently substitute another path.

The loop resets only between complete episodes. There is no object pose
forcing, weld, residual policy, or injected object wrench during the trajectory.
All objects in the capture remain free bodies, including the supporting block
or recipient bowl. The 100 Hz command path advances four MJX-Warp substeps at
400 Hz. Each selected input frame and the explicit held tail is consumed.
Recipes with `command_track` instead execute the entire frozen command asset;
the asset hash, length and monotonic reference mapping are validated before
simulation. `--post-padding` controls the source reference window, not the
length of a frozen command asset. Command assets are loaded only from the
patch directory, and hash or source-UUID mismatches fail explicitly.

## Exported object trajectory

```bash
python tools/export_repaired_motion.py \
  --run outputs/object_repair/bowl-run \
  --output outputs/object_repair/bowl-repaired.lance
```

`direct_repaired_physical_motion.v1` contains:

- measured post-step `objects[].pos` and `objects[].rot_aa` in the simulation
  world frame, including the shared grounding shift;
- measured `hands[].urdf_dof`, and a separate `command_target_dof` track;
- original frame mapping, solver profile, exact source selector and patch;
- a generated-data marker, without copying obsolete source contact labels.

The generated-motion Lance is a trajectory **output**, not a source-capture
replacement for `view_manorl_lance.sh`. Re-executing measured hand positions is
a different experiment from applying the original actuator targets. Use the
source + patch command above for verified live physical reproduction. The NPZ
saved alongside the run also carries full scene qpos/qvel, initial state and
object-to-qpos mapping for exact visualization.

## Diagnostic invariant

A successful command sequence can replay with small trajectory differences on
this GPU contact solver; success is verified by task behavior, not bitwise pose
identity. Recorded actual-motion exports remain exact to their saved trace.

A replay trace must be reconstructed with the exact body/joint order of the
recording model. The runtime sorts scene object types; capture order can differ.
Matching only `nq` is insufficient: swapping bowl and bottle free-joint slots
produces false contact reports without a shape error. Compare body and joint
names or use saved object addresses before interpreting geometry.

`host_data()` coordinate mirrors do not provide a valid native contact buffer.
Recompute geometric contacts with `mj_forward` or use the backend's explicit
contact transfer API; label recomputed native forces separately from MJX solver
telemetry.
