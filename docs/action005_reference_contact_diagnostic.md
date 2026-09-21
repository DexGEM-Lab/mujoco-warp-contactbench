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

## V4: reconstruction fails before sensitivity identification

V3 preserved qpos/qvel/ctrl, not full native-buffer checkpoints. An authorized
substitute reconstructed its frozen target once from frame0, saving all native
buffers at380/400/409. **The reconstruction failed qualification; no sensitivity
probes, inverse correction or candidate replay ran.**

Controls are bitwise equal through409; initial states and U1 setting hash match.
All four substeps retained zero external forces, with no post-frame0 state
writes. Historical compiled model bytes were not saved: the setting fingerprint
is the historical model comparison boundary. V4 saves its compiled model/hash.

| Quantity | Maximum archive difference | Predetermined tolerance |
|---|---:|---:|
| qpos, mixed native coordinates |0.000872344 at409, coordinate33|0.000001|
| qvel, mixed native coordinates |0.101655 at291, coordinate33|0.0001|
| ctrl |0, bitwise equal|0|

Both state tolerances first fail at29. Bearing masks (>0.2N) differ at
266/267/269/273/274/288;40 force samples fail0.02N+1% per-finger tolerance.
At380 index force is3.473219N versus archived3.403213N.400/409 maximum force
differences0.010086/0.009981N pass, but late force agreement cannot establish
identity of the underlying state/contact history.

Predicted versus observed: identical controls were expected to reconstruct
v3 within the established local restore tolerances. They did not. This is
consistent with native replay variability; its numerical source was not
isolated. No derivative sign/rank/conditioning conclusion follows. Planned
±0.004/±0.002rad,8-frame probes were not run; there is no predicted correction
or realized correction comparison. Complete passes:0; frozen repeats:0.

**Single blocker:** historical full-native states are unavailable and their
reconstruction fails the approved identity tolerance. V4 checkpoints belong
to this different reconstruction, not qualified historical v3 states.

`outputs/action005_reference_contacts_v4/` contains manifest/model/hash,
`checkpoint_{380,400,409}/{state.npz,workspace.json}`, reconstruction trace and
raw telemetry, `reconstruction.json`, `divergence.json`, `result.json`.
Sibling `_native.log` preserves the run. The original tool name
`identify_action005_wrench` was changed to `reconstruct_action005_native` after
the gate failed; the unexecuted probe code was removed. No second physics run.
HEAD9890d25 is an unrelated descendant of7de8fa9, preserved untouched.

Reproduce reconstruction only with a fresh output directory:
```sh
PYTHONPATH=$PWD /home/jay/anaconda3/envs/manorl_mujoco/bin/python \
  -m tools.reconstruct_action005_native --output outputs/action005_reconstruction_NEW
```
Seven new tests enforce whole-prefix comparison, exact controls, nonfinite
rejection and bearing-mask preservation. All20 scoped005 tests pass.

## V5 fresh-realization sensitivity: inverse rejected

One NEW frozen-v3 nominal-U1 realization captured155 native buffers at380/400/409.
Model hash matchesv4; U1/initial states/target matchv3. No historical identity
matching. Every branch restores all buffers exactly. Paired8-frame zero
continuations pass: maximum pose1.79e-7, velocity3.20e-5. The frame0 baseline
has no post-initialization state writes; diagnostic branches explicitly do.
All substeps check zero external forces, finite state and capacity limits.

|Frame|Thumb/middle force N|Thumb/middle gap mm|Span mm|Drift deg|
|---|---:|---:|---:|---:|
|380|9.408/0.841|−1.674/−0.336|78.660|0|
|400|9.245/0.521|−1.652/−0.126|79.236|1.853|
|409|9.165/0.448|−1.658/−0.016|79.711|2.739|
|418|9.208/0|−1.616/+0.002|80.865|4.038|

At418 index/ring/pinky=3.674/1.588/2.072N; net force
[−0.1634,+0.0684,−0.1214]N, torque[−0.01349,+0.02657,−0.01194]Nm.
Relative speed20.611deg/s and trailing12-frame acceleration80.963deg/s².
The thumb-dominated unloading basin repeats; lower leverage survives.
Original>=80mm criterion is false at checkpoints. Supervisor explicitly allowed
same-basin/no-collapse baseline qualification because archived spans were
78.989/79.259/79.734mm; fresh differences−0.330/−0.023/−0.023mm.
This is not final acceptance. Physical acquisition still fails: thumb>1N at278,
before middle/ring/pinky first>0.2N at334/310/300.

Two unit joint-space normal-gap gradients (positive opens) were probed at
±0.004/±0.002rad over8frames, with last4-frame mean forces/gaps/net wrench
and quadratic relative SO(3) acceleration. All expected signs are observed.
Normalized Jacobian conditions2.406/1.684/2.279 and task conditions
4.10/5.21/20.08 establish rank, but every checkpoint fails30% linearity:

-380: middle-direction angular symmetry error75.7%; force/gap errors4.1/8.9%.
 Its specific angular-response mechanism remains unisolated; repeat noise is
 orders smaller.
-400: task-force inconsistency53.9%. Middle+.004rad loses contact at408;
 +.002rad retains0.400N there. Small-scale middle-force slope−21.160N/rad
 extrapolates0.3755N last4 mean at+.004; observed0.2865N. This is a post-hoc
 scale-consistency comparison, not a candidate prediction/transfer result.
-409: force77.5%, gap51.4% inconsistency; middle contact activation switches
 across horizon frames depending on perturbation sign/scale.

**Single blocker:** the measured8-frame target-to-response map crosses contact
activation boundaries and exceeds30% nonlinearity. Correct signs/full rank do
not justify a smooth inverse. No correction, v5 target, independent transfer,
full episode or repeat followed. Complete passes:0. This does not establish
 global uncontrollability. Reference-required wrench was not computed because
 sensitivity failed before the inverse stage; reported wrench is measured net.

Artifacts: `outputs/action005_reference_contacts_v5/run_001/` contains baseline
trace/raw contacts, model/manifest, checkpoints,24 signed branches plus6 zero
continuations, basis arrays, sensitivity reports and result. Parent directory
contains PID/log/exit0. An initial managed-shell interruption stopped near280
before checkpoints; preserved under`interrupted_launch/`, with exactly one
approved identical operational restart. No scientific parameter variant.

Reproduction (new output only):
```sh
PYTHONPATH=$PWD /home/jay/anaconda3/envs/manorl_mujoco/bin/python -m tools.identify_action005_fresh --output outputs/action005_fresh_NEW
```
27 focused005 tests pass, covering nonlinear/unilateral rejection, explicit
sub80mm basin reporting, collapse rejection and unit normal-basis support.
