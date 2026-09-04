# ManoRL physical asset source

## Canonical source

ManoRL has one physical asset source:

```text
git@github.com:DexGEM-Lab/dexstream_digital-assets.git
checkout: assets/dexstream_digital_assets
pin: f98da997f316c8a6b4bc2931cabed19e831ef163
```

The root repository tracks the source as a Git submodule. Large meshes are Git
LFS objects in that submodule; they must be materialized before MuJoCo loads a
model. The project-owned
`sim/manorl/task_assets/dexstream_manifest.json` records the exact source commit
and the SHA-256/size of every file required by ManoRL, including the binary MANO
skin (`.skn`), skin bind (`.npz`), and XML fragment.

```bash
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
scripts/setup_manorl_assets.sh
```

`setup_manorl_assets.sh` pulls only the LFS files named by the manifest and then
checks both MANO sides and every DexGEM object. A raw LFS pointer, missing file,
wrong size, digest mismatch, or submodule commit drift is an error.

## ManoRL hand contract

The runtime selects the `sunke` MANO bundles:

```text
hand/mano/sunke/right/
hand/mano/sunke/left/
```

Each side supplies its URDF, 16 collision meshes, metadata, and MANO skin
fragment. The adapter preserves the current policy contract:

- 28 DoF, `cmc3_mcp2_28dof` order;
- z-up floating base;
- six wrist coordinates followed by the 22 finger coordinates;
- right-then-left ordering when `hand_side="both"`.

The skin fragment is visual-only. It replaces the 16 rigid visual partitions in
viewer scenes, while collision remains the 16 URDF mesh links. `mano_pose.py`
reads the exact right-hand joint axes from this pinned URDF rather than keeping
a second hard-coded axis table.

The source snapshot currently declares `palm_collision_scale=1.0`. This differs
from the removed curated runtime (`0.7`) and is part of the new physical asset
identity.

## ManoRL object contract

The adapter discovers same-name rigid URDF bundles under
`objects/DexGEM/`. It reads the link, visual mesh/scale, every collision
mesh/scale, inertial values, and source visual color from each URDF. It does not
assume a fixed CoACD piece count. Object body and free-joint names use the
canonical lower-case object name used by trajectory identities.

The current source exposes these 28 runtime names:

```text
banana bottle bowl camera1 camera2 cap cube1 cube2 cuboid1 cuboid2 cuboid3
cylinder1 cylinder2 cylinder3 cylinder4 cylinder5 cylinder6 cylinder7
disposablecup iphone largeclamp mayonnaisebottle pitcherbase powerdrill
sphere1 sphere2 sphere3 sphere4
```

The source directories `CAMERA1` and `CAMERA2` are exposed as `camera1` and
`camera2`; this is a name normalization at the adapter boundary, not a copied
or renamed asset. `bottlewithcap` and `scissor` were removed from the current
DexStream snapshot and therefore fail explicitly. They are not silently mapped
to `bottle` or another object with different geometry.

The source URDF is authoritative for physical units and parameters:
visual meshes are millimetre OBJ files with a `0.001` URDF scale, while CoACD
collision OBJ files are metre files with scale `1`. All current bundles are
single-link rigid bodies; multiple convex pieces do not create internal object
joints.

## Task metadata boundary

DexStream does not provide ManoRL's expected-contact subset for each
object/action pair. Those aliases are observation/reward task semantics, not
physical assets, and are maintained in:

```text
sim/manorl/task_assets/object_grasps_simple.yaml
```

Keeping this file outside the source submodule makes the ownership boundary
explicit: changing geometry requires a DexStream pin change; changing expected
contact semantics requires a ManoRL task-contract change.

## Checkpoint and data compatibility

The current hand/object snapshot is a new physical version. In addition to the
palm scale change, the current source has rebuilt or re-aligned several object
CoACD meshes and inertial tensors. Shape-compatible policy tensors therefore do
not imply identical contact behavior.

New native checkpoint signatures include:

```text
asset_source_repository
asset_source_commit
asset_manifest_sha256
```

Strict resume and ordinary inference reject missing or mismatched asset
identity. `--warm-start-checkpoint` and policy-transfer are the explicit routes
for transferring weights across physical versions; they do not establish
rollout equivalence. Generated Lance manifests should retain the checkpoint
identity and current asset manifest identity together.

## Updating the source pin

1. Intentionally move the submodule to a reviewed source commit.
2. Run `python tools/generate_manorl_asset_manifest.py`.
3. Materialize the manifest's LFS paths with `scripts/setup_manorl_assets.sh`.
4. Run the focused asset and scene tests, including homogeneous, unified, both
   hand sides, and the generic `sim/scene.py` path.
5. Review object-set changes and checkpoint compatibility before committing the
   `.gitmodules` gitlink and manifest.

Do not retain a fallback to the removed asset submodules or to copied runtime
meshes. A missing new source must be surfaced as an explicit setup/validation
error.
