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
- New source audit: 28 DexGEM URDFs, each one rigid link with one visual mesh
  and at least one collision mesh; all mesh references resolve.
- Manifest setup validation passed for both MANO sides and every object.
- Native MuJoCo compilation passed for all 28 homogeneous models, right/left/
  bimanual cube1 models, representative visual+skin scenes, and right/left/
  bimanual unified models containing all 28 objects.
- Focused tests passed: asset/submodule/grasp/scene/checkpoint/Lance tests
  (56 tests), plus the updated viewer skin check.
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

## Current claim
The feature branch has a structurally complete migration: one physical asset
submodule, source-pinned manifest and LFS setup, direct MANO/object resolution,
no executable old-source fallback, generic-scene adaptation, and checkpoint
provenance binding. The migration intentionally does not claim behavioral
equivalence for legacy checkpoints or trajectories because the physical source
changed.

## Remaining validation
- Run the complete repository test target where the external reference Lance
  fixture is available; current local full-test blockers are the pre-existing
  missing `/mnt/nas-222-project/mocap_v2/lance_datasets/human_p1_remake/npy_s02_v3.lance`
  fixture and an unrelated dual-contact reducer assertion.
- Review and commit only migration files on this feature branch; do not stage
  the primary worktree's unrelated user modifications.
