## Objective
Formalize the validated target-DOF replay viewer as a repository-owned, reusable ManoRL tool. It must replay a versioned predecoded target package without reading Lance or loading a policy checkpoint, support headless validation and MuJoCo GUI playback, and preserve the exact 5 ms / two-substep controller semantics used by the validated smoke.

## Workbench
- Add the package schema/loader and replay CLI under `tools/`.
- Add focused tests for package validation and target-step semantics without requiring GPU/GUI.
- Document package creation/usage and the Server2 diagnostic boundary.
- Run focused tests and a compile/help check; preserve a small example package only as an external generated artifact, not in Git.

## Context
- Source worktree: `/home/jay/dexrobot/FromSSH/manoRL_mujoco-worktrees/feat-target-dof-replay-tool`.
- Integration base: `dev` at `eebcb74`.
- Validated package: `cube1_global2200_row2239_target_dof.npz` plus JSON metadata, 520 frames, source identity `cube1_01_1614`, checkpoint global update 2200.
- The package contains timestamps, recorded 28D qpos, post-controller 28D target qpos, object position, and object XYZW quaternion. Server2 must not decode the Lance source directly.
- Existing physics ownership is `sim/manorl/environment.py`; the validated runtime applies target qpos directly to `data.ctrl`, executes `PHYSICS_SUBSTEPS_PER_TARGET=2`, and renders a host snapshot through MuJoCo viewer.

## Task specifications
- Define an explicit, versioned package contract with finite arrays, exact shapes, strictly increasing timestamps, normalized quaternions, positive timestep, and metadata sufficient to identify object/checkpoint/source lineage.
- Implement a CLI that accepts `--package`, `--device cpu|gpu`, `--headless`, `--frames`, `--speed`, `--loop`, and GUI camera/display controls as appropriate.
- Keep package loading independent of Lance, W&B, and checkpoint loading. Do not silently fall back to Lance or regenerate targets.
- Use the project environment/model assets and existing `MujocoManoEnvironment` physics; isolate internal environment calls behind small adapter functions with explicit comments about the stable contract.
- Headless mode must compare replayed qpos/object pose against recorded arrays and emit a machine-readable JSON result with pass/fail metrics.
- GUI mode must use the same physics state and synchronize a native MuJoCo viewer without advancing a separate native simulation.
- Add focused unit tests for malformed packages, quaternion/timestamp validation, and CLI/package metadata behavior. Tests must run without GPU or X11.
- Update README with the package format, export boundary (Lance decoding belongs on a healthy machine), commands, and known Server2 diagnostic limitation.
- Do not add generated replay packages, checkpoints, credentials, or host-specific paths to Git.

## Constraints
- Work only in this linked feature worktree; do not edit or commit in primary `dev`.
- Preserve unrelated existing untracked files in the primary worktree.
- Do not modify the authoritative Lance datasets or production training processes.
- Do not use Server2 Lance decoding as a validation path.
- Keep CPU/GPU semantics explicit; no silent fallback from requested GPU to CPU.
- Commit one coherent feature change after staged-diff inspection.
