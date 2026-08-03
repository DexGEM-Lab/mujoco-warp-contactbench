# Epistemic model

## Phenomenon
ManoRL previously advanced one decoded trajectory sample per 5 ms control step regardless of source acquisition rate. A 100 Hz capture therefore played at 2× speed; a 120 Hz processed trajectory played at approximately 1.667× speed. Lance timestamps were provenance only and did not drive simulation time.

## Implemented mechanism
The runtime now has three explicit clocks:

- reference/source: user-selected `100|120` Hz;
- policy/control: fixed 200 Hz (`0.005 s`);
- physics: fixed 400 Hz (`0.0025 s`, two substeps per control target).

Every decoded source trajectory is mapped to the 200 Hz grid before environment table construction. Hand angular coordinates are unwrapped across time before linear interpolation, object positions use linear interpolation, and object quaternions use SLERP. The final source pose is retained, with at most one partial control interval held at the endpoint for 120 Hz. Source-index provenance is mapped monotonically, and the inclusive raw movement interval is converted to inclusive control steps. Policy residuals, servo cadence, PPO rollout length, and physics settings are unchanged.

## Direct evidence
For real Lance v295 identity `banana_01_052`, source slice `[168,746)` contains 578 frames. Selecting 100 Hz produced 1,155 control frames and 5.770 s duration with movement window `[200,710]`; selecting 120 Hz produced 963 frames and 4.810 s with movement window `[167,591]`. Both retain 5 ms control targets and two 2.5 ms physics substeps. A real CPU MJX-Warp transition using the 100 Hz trajectory produced a finite reward and a `(1,480)` observation with those mapped lengths/window.

Synthetic fixtures independently verify exact 100 Hz midpoints, 120 Hz fractional interpolation, quaternion midpoint orientation, endpoint retention, source-index mapping, and movement-window mapping. The final affected-path suite passes 136 tests. A broad repository run reached 387 passing tests; its failures reproduce on unchanged `dev` or require an absent legacy Lance v132 dataset and therefore do not implicate this change (OPS.md).

## Checkpoint and interface contract
Training defaults new runs to 120 Hz and accepts `--reference-fps {100,120}` / `MANORL_REFERENCE_FPS`. Viewer, inference, Rerun recording, and synthetic export use the same selector. New checkpoint sidecars include `reference_fps` in environment ABI v7. Inference restores the checkpoint value when omitted and rejects explicit conflicts. Legacy checkpoints preserve their old one-source-frame-per-control-step behavior and reject an explicit new clock, preventing silent semantic reinterpretation.

## Remaining domain uncertainty
Raw acquisition is reported as 100 fps, while cleaned Lance v295 timestamps/metadata encode approximately 120 Hz. The selector makes that scientific choice explicit but does not establish which clock is authoritative. Until raw-to-Lance processing provenance resolves the discrepancy, 100 Hz and 120 Hz runs represent distinct hypotheses rather than interchangeable configurations.

## Current claim
The requested selectable clock is implemented without changing physics or control frequency. Any training process started before this change still uses the legacy one-frame-per-control-step schedule; it cannot be converted in place and must be restarted from scratch to test either corrected clock.

## Highest-value next action
Integrate and deploy the committed branch, then start a fresh run with an explicitly named reference clock. If raw 100 fps is the intended physical truth, use 100 Hz and treat the already-running legacy-clock experiment as invalid for that hypothesis. Preserve a 120 Hz run only as the processed-Lance-clock comparison.
