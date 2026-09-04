## Objective
Replace the old `assets/all_assets`, `assets/mano_hand_s02`, and copied ManoRL runtime assets with one pinned `git@github.com:DexGEM-Lab/dexstream_digital-assets.git` source for ManoRL objects and MANO hand. Preserve the 28-DoF checkpoint/runtime contract, and finish with validated code, docs, and cleanup.

## Workbench
1. Add and inspect the pinned digital-assets source.
2. Implement object and MANO path/discovery adapters.
3. Remove old submodules/copies and stale references.
4. Run asset, URDF, scene, and focused regression validation.
5. Commit the migration on this feature branch.

## Context
Repository: `/home/jay/dexrobot/FromSSH/manoRL_mujoco-worktrees/feat-dexstream-assets-migration`.
New source: `git@github.com:DexGEM-Lab/dexstream_digital-assets.git`, main HEAD `f98da997f316c8a6b4bc2931cabed19e831ef163`.
Protected integration branch: `dev`; this task must merge back through the feature workflow.

## Task specifications
### Source contract
- One canonical Git submodule/source checkout must provide both object and hand assets.
- Pin the exact source commit used for validation; do not follow a moving branch at runtime.
- Use `hand/mano/sunke/right` and `hand/mano/sunke/left` unless evidence requires another subject; preserve the existing 28-DoF joint order and z-up floating base.
- Resolve ManoRL object bundles from `objects/DexGEM/<object>` and validate URDF, visual mesh, every collision mesh, inertial data, and metadata before compilation.
- Establish an explicit grasp-mapping source/adapter because the new repository may not contain the old `object_grasps_simple.yaml`.

### Cleanup contract
- Remove old `assets/all_assets` and `assets/mano_hand_s02` submodules and the copied `sim/manorl/runtime_assets` only after the new source is wired and validated.
- Remove stale code/tests/docs/notebooks that claim the old assets are authoritative; preserve historical data provenance in memory/docs where needed.
- Do not delete generated Lance datasets or unrelated user changes.

### Acceptance
- `git submodule status` shows only the new asset source and unrelated third-party dependencies.
- No executable ManoRL path imports old asset roots or copied runtime assets.
- Both hand sides and every currently supported ManoRL object resolve from the new source.
- Representative homogeneous and unified MJX model compilation succeeds, with focused tests covering paths, dimensions, joint order, units, collision counts, and source pin.

## Constraints
Work only in the assigned feature worktree.
Preserve all unrelated user modifications in the primary worktree.
Do not delete old assets before replacement validation passes.
Do not add generated datasets, credentials, or machine-local paths to Git.

## Unattended visual acceptance extension
- Continue until a real local single-environment (`num_envs=1`) visual check succeeds.
- Use `DISPLAY=:1` for the visible MuJoCo path when the X11 session is available; save a screenshot and inspect it rather than treating process exit as proof.
- Check GPU memory before starting and avoid occupied GPUs. Keep visual tests bounded and do not generate training/synthetic datasets.
- If the visible scene cannot run because the display is unavailable, prove the exact environmental blocker and run an equivalent offscreen render only as a secondary check; do not silently call that visual acceptance.
