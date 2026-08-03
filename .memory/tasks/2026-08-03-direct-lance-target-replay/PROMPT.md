## Objective
Replace the NPZ/JSON target replay package path with direct Lance-row replay, as explicitly requested. The replay CLI must read one specified Lance dataset/version/row on a healthy machine, extract the post-controller target DOF and recorded physical/object state, then run the existing MJX-Warp target replay and optional MuJoCo viewer.

## Workbench
- Replace package loader/source types with a direct Lance-row decoder in `sim/manorl/target_replay.py`.
- Change `tools/replay_manorl_target_dof.py` to accept dataset/version/row-index.
- Delete the NPZ package exporter and package-only tests/docs.
- Add focused fake-Lance/row-contract tests, run the direct CLI on healthy Server1, and run one local single-display GUI smoke after explicitly exporting `DISPLAY`.
- Preserve the explicit Server2 boundary: do not run direct Lance replay on Server2 while its native Lance/PyArrow path remains disqualified.

## Context
- Current `dev` includes the previous NPZ implementation at `e1b6066`.
- The validated source row is the merged Lance row 2239, object `cube1`, source identity `cube1_01_1614`, 520 frames.
- The direct decoder must request only the columns needed for replay (`index`, `trajectory_metadata`, `timestamp`, `hands`, `objects`, `provenance`) and must handle v2.2's canonical two hand slots with only `right` active.
- Current project physics semantics remain: target vectors are post-`command_target` controller targets; apply one target per 5 ms and execute `PHYSICS_SUBSTEPS_PER_TARGET=2` MJX-Warp substeps.
- Direct Lance replay is a healthy-machine operation. Server2's known native Lance/PyArrow SIGSEGV/type-corruption boundary remains documented and must not be retried. The local GUI acceptance uses exactly one X display selected from the observed session, with `export DISPLAY=...` recorded in the run evidence.

## Task specifications
- Remove NPZ/JSON package schema, loader, writer, exporter, and package-only README instructions from the integrated code.
- Build a validated direct-row source object with finite 28D qpos/target arrays, timestamps, object pose/quaternion, source/generated lineage, checkpoint metadata, movement range, and dataset version/row identity.
- Preserve fail-closed validation: one row exactly, required nested fields, right-hand selection, shape/timing/quaternion checks, finite values, object identity, and explicit Lance errors.
- Keep headless metrics and GUI behavior; report direct dataset path/version/row plus source identity and checkpoint provenance.
- Tests must not require GPU, GUI, or a real Lance dataset; use a fake dataset/table to assert requested columns and row validation.
- Update README to show direct Lance commands and state that this path is not supported on the currently disqualified Server2.
- Do not delete existing generated NPZ artifacts outside Git unless separately requested.

## Constraints
- Work only in this linked feature worktree; do not edit or commit in primary `dev`.
- Do not run direct Lance decoding on Server2.
- Do not modify authoritative Lance datasets or production training.
- Keep GPU/CPU choice explicit; no silent fallback.
- Commit one coherent change after staged-diff inspection, then fast-forward it into `dev`.
