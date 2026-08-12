DexHandRL object and hand runtime assets come from the root repository's pinned
`assets/all_assets` submodule.

Pinned revisions:

- `all_assets` main: `7228b5cfce8d9a072ed4bded7a489cf73d521b68`
- nested `Assets/sim/mano_assets` dev: `cde03ef94816b589f574ca6f358695005d3d1a3f`

Initialize both levels after cloning:

```bash
git submodule update --init --recursive assets/all_assets
```

Every DexHandRL object resolves from:

```text
assets/all_assets/Assets/sim/decomposed/<object>/<object>.urdf
```

The URDF visual mesh supplies rendering and policy geometry/point-cloud inputs.
Every referenced `coacd/coacd_convex_piece_*.obj` is materialized as a physical
collision geom; multi-piece objects are never reduced to the first piece.

The checked-in `assets/isaac_source_root/` directory remains the default owner
of `dexhand_env/cfg/task/Dexhand021proReconstruction.yaml`, because that task
configuration is not part of all_assets. Its old copied hand/object files are
not the default runtime source.

`DEXHANDRL_ISAAC_SOURCE_ROOT` may point to a complete alternate outer source
root containing `assets/all_assets/Assets` and `dexhand_env/cfg/task`. The
runtime fails closed when the requested decomposed URDF, mesh pieces, contact
map, or task configuration is missing.
