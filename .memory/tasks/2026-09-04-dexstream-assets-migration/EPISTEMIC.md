# Epistemic Model

## Phenomenon
ManoRL previously mixed three physical sources: the `assets/all_assets` object
submodule, the `assets/mano_hand_s02` hand submodule, and copied hand/cube files
under `sim/manorl/runtime_assets`. The requested state is one reproducible
physical source for both hand and objects, with no silent fallback that can mix
asset provenance.

## Supported mechanism
- `assets/dexstream_digital_assets` is the sole physical source, pinned to
  `f98da997f316c8a6b4bc2931cabed19e831ef163` from
  `git@github.com:DexGEM-Lab/dexstream_digital-assets.git`.
- ManoRL consumes `hand/mano/sunke/{right,left}`. Both bundles satisfy the
  28-DoF `cmc3_mcp2_28dof`, z-up, 16-link collision/visual contract, and the
  runtime reads the exact source joint axes for MANO pose conversion.
- ManoRL discovers 28 same-name rigid `objects/DexGEM` URDF bundles. The adapter
  canonicalizes only source directory case (`CAMERA1`/`CAMERA2` to
  `camera1`/`camera2`) and reads all collision pieces/scales from each URDF.
- `sim/manorl/task_assets/dexstream_manifest.json` pins required Git/LFS files
  by source commit, SHA-256, and size. Expected-contact aliases remain in the
  ManoRL-owned task mapping because the physical source does not provide them.
- Strict native checkpoint signatures include repository, source commit, and
  manifest SHA. Explicit warm-start/policy-transfer is the only intentional
  cross-physical-version boundary.

## Evidence
- Source audit found 28 valid DexGEM URDF bundles; every one has one rigid link,
  one visual mesh, at least one collision mesh, finite inertial data, and
  resolvable mesh references.
- Manifest generation is deterministic. The setup script verifies both MANO
  sides and all 28 objects, rejects source-pin drift/unmaterialized LFS files,
  and does not modify GitGuard hooks.
- Native MuJoCo compilation passed for all 28 homogeneous models, right/left/
  bimanual cube1 models, representative visual+skin scenes, and right/left/
  bimanual unified models containing all 28 objects.
- Focused migration regressions passed 159 tests (7 skipped); the final core
  rerun passed 76 tests (5 deselected). The broader suite's observed blockers
  are the pre-existing absent external Lance fixture and an unrelated dual-
  contact reducer assertion.
- Local visual acceptance passed with `DISPLAY=:1`, a materialized new source,
  and one world: the rendered banana and MANO skin were manually inspected in
  both an offscreen image and a real GLFW window. A real v530 Lance→CPU
  MJX-Warp path also loaded `banana_09_1384` with `num_envs=1`, returned 480D
  observations, and stepped through nonzero contacts.
- The current source changes MANO palm collision scale from the removed
  runtime's 0.7 to 1.0 and changes several object CoACD/inertial assets. The
  current source is therefore a new physical version even where dimensions or
  policy tensor shapes remain compatible.

## Ruled out
- Replacing only the old submodule URL cannot work: DexStream reorganizes both
  hand and object paths and no longer contains some historical object names.
- Mapping removed `bottlewithcap` or `scissor` to `bottle` or another object
  would silently change geometry and is prohibited; those names now fail
  explicitly.
- Keeping copied runtime meshes as a fallback would permit mixed provenance
  and hide missing Git LFS materialization; the copies are removed.
- Treating the task-owned grasp map as a DexStream physical asset is incorrect;
  it encodes the ManoRL observation/reward contract.
- The live-window SIGSEGV is not evidence of a bad asset: the same DexStream
  scene renders successfully, a minimal viewer exits cleanly, and the failure
  appears only when an external X11 screenshot races the GLFW teardown.

## Current claim
The feature branch has a complete, source-pinned asset migration and a
successful single-world visual/runtime acceptance. It provides one physical
source, explicit task metadata, no executable old-source fallback, generic
scene adaptation, and checkpoint provenance binding. The migration intentionally
does not claim behavioral equivalence for legacy checkpoints or trajectories
because the physical source changed.

## Remaining boundary
- The full trajectory-driven policy viewer using the historical default fixture
  remains unavailable on this machine because
  `/mnt/nas-222-project/mocap_v2/lance_datasets/human_p1_remake/npy_s02_v3.lance`
  is absent. Asset-only and modern-v530 env=1 paths are validated.
- The external-capture/GLFW teardown issue is operational; use the saved
  renderer image or the viewer's own capture path rather than racing X11
  `ImageGrab` when a clean exit code is required.
- Before integration, merge only this feature branch into `dev`; preserve the
  primary worktree's unrelated user changes.
