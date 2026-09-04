# Epistemic Model

## Phenomenon
ManoRL currently obtains object URDF/collision/visual/grasp inputs from the `assets/all_assets` submodule, hand inputs from `assets/mano_hand_s02` plus copied `sim/manorl/runtime_assets`, and generic scene code still references the old hand submodule. The requested target is one authoritative `dexstream_digital-assets` source for both objects and hand, with old asset sources removed.

## Live model
- The new repository provides the required DexGEM object URDF/CoACD/visual assets and MANO subject bundles.
- `hand/mano/sunke/{left,right}` matches the current 28-DoF joint names, z-up floating-base contract, and current hand beta values; it is the leading compatibility choice for the checkpoint-bound ManoRL runtime.
- The new object layout is not a drop-in path replacement: objects are under `objects/DexGEM/<name>`, object URDFs are self-contained, and grasp mappings are not yet observed in the new repository.
- A migration adapter must discover/validate new assets rather than retain a large old-source SHA registry.

## Unresolved questions that tests must answer
- Whether the new MANO `sunke` URDF/meshes are numerically identical to the current curated runtime hand, including palm collision scale and left/right path semantics.
- Whether all current registered ManoRL object types have a matching valid `objects/DexGEM/<name>` bundle and whether a replacement grasp mapping exists.
- Whether the new URDF/mesh units, collision piece ordering/counts, inertial values, and visual paths compile under the existing scene builder without hidden geometry changes.
- Which non-ManoRL generic scene/tests should be migrated or removed when the old submodules are deleted.

## Acceptance claim (pending evidence)
The change is complete only when the repository has one tracked asset source, no runtime dependency on old asset submodules/copies, all required registered objects and both hand sides resolve from the new source, representative MuJoCo compilation succeeds, and focused tests/documentation state the new source and pin.
