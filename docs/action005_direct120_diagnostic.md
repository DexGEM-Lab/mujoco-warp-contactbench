# Direct120 action005 source selection and baseline

Selected **v4 row130**, UUID `95c72d14-d854-47af-8239-849cd9757465`, as the best complete kinematic reference. The user explicitly relaxed source-contact eligibility: CPU source geometry is diagnostic; MJX-Warp determines physical acquisition. No donor is selected.

## Source audit

|Row|Peak tilt deg|Final displacement cm|Peak bowl XY cm|Airborne maximum geometric penetration mm|
|---|---:|---:|---:|---:|
|62|93.85|3.58|10.82|16.96|
|69|100.65|4.46|10.05|13.60|
|124|106.14|1.60|8.95|23.68|
|125|113.44|0.13|9.58|15.59|
|126|121.47|4.57|9.43|10.66|
|127|117.34|1.20|10.80|10.88|
|128|128.52|1.15|8.01|36.57|
|129|119.66|0.97|9.03|16.89|
|130|132.27|0.77|8.66|10.97|
|131|128.77|1.57|8.41|18.80|

Rows62/69/124 are shallow (<110deg). Rows125/126/127/129 invert less deeply than130. Row128 has a58.15mm hand-object relative step and36.57mm geometric penetration; row131 reaches128.77deg but offers no demonstrated grasp advantage. Row131 right slot/betas match the pinned profile; its airborne relative drift is1.95cm/24.30deg, not evidence of a wrong-hand identity. All ten clocks are strictly increasing and mean rates remain nominal120Hz, but instantaneous intervals jitter. No resampling was applied.

Row130 world-axis peak tilt132.27deg differs slightly from the prior132.22 estimate; full orientation change142.36deg agrees. Final world tilt0.49deg and full orientation delta6.94deg are different measurements. Native pickup interval223–657, >2cm lift229–643, deep interval378–507; returns below15deg at587. Final source hand clearance385.7mm indicates geometric release.

CPU collision confirms zero five-contact deep-interval fraction in every deep row. Row130 middle never contacts; peak441 has only pinky distal contact,5.67mm penetration. Consequently actual contact axial ordering/span and thumb opposition are undefined at peak; proximity ordering is not a grasp. Complete per-frame pair IDs, signed distances, points and normals are retained. No source force/bearing-order assertion is made.

**Distance anomaly:** mj_geomDistance can return zero with inconsistent nonzero witness lengths. Initial `audit/` distance-derived touch/topology fields are provisional; `collision/` CPU collision pair evidence supersedes them. Verification excludes invalid witnesses rather than treating zero as contact. Pinned shape/asset checks pass; no asset mismatch was found to explain the large overlaps.

## MJX baseline

One native frame0 baseline completed1164frames on cuda:0, U1 `c6db552f105d4de493880a52518772dbacad9496b105d2f4e43e87a907e692cd`. Recorded qpos with existing velocity/friction/payload-gravity target lead;120/480Hz, four substeps, CCD16, native actuators. No post-frame0 state writes, applied forces or CPU integration.

Lift **0.00cm**, reference17.81cm. First >2cm reference-lift deficit at frame229; bearing forces {'thumb': 0.0, 'index': 0.0, 'middle': 0.0, 'ring': 0.0, 'pinky': 0.0}. First >0.2N nonthumb/>1N thumb frames: {'thumb': 206, 'index': 207, 'middle': None, 'ring': None, 'pinky': None}. Peak actual tilt1.90deg. Final1.03cm/1.06deg is the bottle left on the table, not a completed pour.

All-substep capacities {'nacon': 22, 'ncollision': 54, 'nefc': 116}; finite controls/states, max control difference1.19e-07. Native contact force/basis sampled120Hz; capacity480Hz. Broadphase count is a conservative CCD-work bound, not exact CCD occupancy.

**Zero complete passes.** Correction, focused validation and independent repeat were not run: worker turn budget exhausted after the baseline. Next concrete intervention is the user-requested one local own-row nonthumb-first/thumb-delayed acquisition correction, preserving the native wrist/object timeline. No video or successful-replay claim.

## Reproduction

Use the repository manorl_mujoco Python environment and `PYTHONPATH=$PWD`; choose a new output root.
```sh
python -m tools.audit_action005_direct120 --output NEW/audit
python -m tools.verify_action005_source_contacts --audit NEW/audit --output NEW/collision
# Freeze reviewed selection.json before replay (selected_row:130).
python -m tools.run_action005_direct120 NEW
python -m pytest -q tests/test_action005_direct120_audit.py
```

Baseline reuses existing `outputs/four_action_single_v1/run_reference.py`; its source path and build are preserved. Artifacts: `outputs/action005_direct120_v1/{selection.json,compact_result.json,audit/,collision/,baseline/}`. Immutable Lance untouched. Prior row1260.41cm baseline/2.73cm ILC failures were checked, not overwritten.

## Direct120 v2: first acquisition boundary fails

One fixed-wrist solve at supported frame200 found five distal gaps of−0.400mm
(index→pinky axial span59.94mm). No wrist correction was used. Nonthumb
closure ramps140–190; earlier source thumb pose is held before the205–223
thumb closure; correction fades300–340. Wrist targets are byte-identical to
baseline throughout and every target is baseline-exact from340 onward.

The single completed focused MJX-Warp replay stopped at **frame193**:
thumb1.0412N, index/middle/ring/pinky0N. No nonthumb had previously crossed
0.2N. Maximum lift0cm; peak tilt1.217deg. The first physical-order gate fails
before reference lift229; no full episode or repeat was permitted. Finite
states/controls, maximum target error5.96e−8; capacity maxima18contacts,
50broadphase pairs,106constraints (480Hz). Nominal U1 unchanged.

The source-frame160 thumb target does not keep the thumb physically clear
while the reference wrist approaches: high loading begins at193 despite the
delayed205 target-close phase. The shallow frame200 static fit therefore did
not establish an acquired four-finger grasp. This is the exact failure boundary;
no second target or solve was attempted.

An initial launch crashed on missing telemetry local-contact coordinates at
first bearing contact. Its log/manifest/partial telemetry and INCOMPLETE marker
are retained in `interrupted/`. One explicitly authorized instrumentation-only
restart reused the identical target SHA
`3cd2a2b929c44eaca0e2b34093e7f3c26451e550eed2cfa896cd631fb2154137`.
The local-coordinate first-bearing regression test passes. Setting hashes match;
the archived baseline model-file hash is provenance, not a separately captured
interrupted runtime model binary. No claim of deterministic state reproduction.

Artifacts: `outputs/action005_direct120_v2/{fit.json,target.npy,fit_qpos.npy,
compact_result.json,focused/,interrupted/}`. Reproduce the construction and single
focused replay with `python -m tools.repair_action005_direct120 --output NEW`.
**Complete passes:0.**
