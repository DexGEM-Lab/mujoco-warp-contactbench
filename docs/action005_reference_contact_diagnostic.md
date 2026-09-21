# A030 sliding-contact diagnostic

**Action005 remains unaccepted. No complete run or fresh-process repeat exists.**
The single nominal-U1 replay stopped at frame418, the first reference50–90°
sample. Its trace localizes middle-contact loss earlier, at402.

## Corrected geometry

Donor axial-coordinate matching is not an invariant. A lower pinky increases
leverage. Continuing the preserved70° solution with signed surface distances,
opposed sides/normals, index>middle>ring>pinky order and >=80mm span closes the
pinky gap from+0.460mm to−0.400mm. At70°, all five gaps are≈−0.400mm,
span93.743mm, opposed-normal cosines0.957–0.981 and wrist correction
17.195mm/10.993° (approved limits20mm/15°).

One neighboring-frame solve runs backward/forward from70°, with four-frame
knots and C1 PCHIP corrections across the original1321-frame A030 timeline.
Prefix<225 and tail>=1050 are byte-identical to the baseline; target-phase
weights start four fingers before thumb. Original transport/deep-inversion
reference timing is retained, but physical semantics have not been completed.

All53 reference50–90° geometry samples retain opposed sides/normals, order and
span>=86.819mm. Maximum non-pad penetration1.057mm slightly exceeds the strict
1.05mm interpolation audit threshold. Across the complete fitted window,
24 samples fail that audit, including gaps up to1.307mm near frame726.
The branch is continuous, not uniformly contact-feasible. Maximum finger target
step is0.01165rad in50–90°, but0.09789rad globally. C1 interpolation alone does
not guarantee gentle actuator motion.

## Native result

At frame418: reference tilt50.573°, actual49.127°; forces thumb/index/middle/
ring/pinky=9.499/3.801/0/1.667/2.157N. Index-to-pinky span80.826mm and available
opposed-normal cosines0.927–0.980 survive. Middle contact is missing; complete
four-finger axial ordering cannot be measured from native contacts at this frame.
Actual middle surface gap is only+0.017mm; thumb penetration1.656mm exceeds the
1.5mm loaded-contact diagnostic limit. Friction utilization peaks0.704;
static wrench reserve is1.253, yet relative rotation is still accelerating:
4.090° drift since380,20.964°/s rate,87.444°/s² trailing12-frame slope.

The correction preserves lower leverage where the old fixed-patch target lost
it, but geometric closure plus fixed donor servo deflection does not maintain
balanced five-finger bearing. Middle force first falls to zero at402.
Acquisition ordering is also unverified physically: in the recorded>=265
window thumb bears at270, pinky301, ring311, middle333. Target scheduling did
not establish the required four-fingers-before-thumb acquisition.

**Localized blocker:** realized five-finger load distribution/acquisition,
particularly early thumb loading and middle unloading—not a pinky axial deficit.
No second dynamic trial, full episode or video followed the failed diagnostic.

Runtime: setting`c6db552f105d4de493880a52518772dbacad9496b105d2f4e43e87a907e692cd`,
120/480Hz, four substeps, native actuators, pyramidal/impratio1, CCD16.
Finite states; maximum applied-target error1.184e−7; zero applied forces and
no post-frame0 episode state writes. Capacity maxima25 contacts/58 broadphase
pairs/128 constraints. The broadphase guard is not exact CCD occupancy.
Forces/contact bases are120Hz last-substep samples; capacities are checked480Hz.

## Evidence and reproduction

Ignored artifacts: `outputs/action005_reference_contacts_v2/` contains
`target.npy`, `fit.npz`, `fit.json`, `contract.json`, `trace.npz`,
`telemetry.jsonl.gz`, `result.json`, `audit.json`. Sibling `_fit.log`,
`_native.log`, `_audit.log` preserve commands' output. The v1 artifacts remain.

The original runtime aggregator incorrectly zeroed *all* topology metrics when
middle contact disappeared. Raw output is preserved; `audit.json` supersedes
those derived topology fields using the same native contacts. Missing fingers
now remain explicit while available index/pinky span and opposition survive.
This does not alter the failed decision or require another physical run.

Use `/home/jay/anaconda3/envs/manorl_mujoco/bin/python` with `PYTHONPATH=$PWD`:

```sh
python -m tools.fit_action005_reference_contacts --source-root outputs/four_action_single_v1 --donor outputs/action005_balance_grip_v4/candidate_04 --base outputs/four_action_single_v1/action005_A_row030/alternate_screen/target.npy --output outputs/action005_reference_contacts_v2 --fit-only
# Same arguments with --replay-existing instead of --fit-only: one native run.
# Same arguments with --audit-existing: offline analysis, no physics.
python -m pytest -q tests/test_action005_reference_contacts.py tests/test_action005_critical_grip.py
```

Choose a new output path for reproduction; existing traces are protected from
overwrite. Focused tests:13 passed, including lower-pinky leverage, missing
middle span preservation, bounds, target phase order and retention rejection.

## One authorized load-balance correction (v3)

A single follow-up was explicitly authorized after v2: use actual frame418
geometry to solve only thumb/middle joint offsets toward−0.9/−0.4mm gaps,
with minimum-displacement regularization and±0.05rad limits. Apply offsets to
frozen controls with a smooth350–380 onset and existing grip-window taper.
Wrist/index/ring/pinky targets, prefix<350 and tail>=1050 are byte-identical
to v2. No sweep was performed. `balance.json` records the exact joint vector.

The corrected geometry predicted thumb/middle gaps
-0.900000/-0.400000mm, but native replay again
stopped418: middle0N, thumb9.212N, ring1.585N, pinky2.078N, span80.888mm.
Thumb penetration remained1.617mm; rotation acceleration81.232deg/s² versus
87.444 previously (only7.1% lower), drift4.059deg. Middle first zero after380
is frame410. A kinematic displacement applied as
a servo-target displacement mostly redistributes spring load instead of
realizing that displacement under the coupled grasp. The predicted middle
bearing was not established. This is the stopping boundary; no further trial.

Artifacts: `outputs/action005_reference_contacts_v3/` contains `balance.json`,
`target.npy`, `trace.npz`, `telemetry.jsonl.gz`, `result.json`, `audit.json`,
`contract.json`; its `fit.json` is inherited v2 geometry, not a measured v3 fit.
Run the same CLI with `--output outputs/action005_reference_contacts_v3
--balance-from outputs/action005_reference_contacts_v2` to construct/replay
(in a new output directory). Finite/canonical, no applied forces/state writes;
capacity25/58/128, control error1.184e−7. **Two complete passes: no; zero full
episodes.** Localized blocker remains middle unloading under thumb-dominated
load, not absent lower leverage.
