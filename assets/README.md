The repository contains a minimal DexHand021Pro runtime bundle under
`assets/isaac_source_root/`. It includes the full DexHand021Pro hand XML/URDF
and mesh set, the `cube1` object used by the default replay examples, the
contact-body map, and the task YAML.

For local development, `DEXHANDRL_ISAAC_SOURCE_ROOT` can point at another asset
root. If it is unset, the code uses the checked-in bundle when present, then
falls back to `/tmp/dexhandrl_assets` or an external Isaac source checkout.

The MuJoCo migration code reads the hand XML, object URDFs, and contact-body
config from the local IsaacGym-compatible tree. Default root:

`assets/isaac_source_root`

Relevant paths under that tree:

- `assets/isaac_source_root/assets/all_assets/Assets/HAND/dexhand021pro/...`
- `assets/isaac_source_root/assets/all_assets/Assets/sim/mano_objects_urdf/...`
- `assets/isaac_source_root/assets/all_assets/Assets/sim/mano_assets/objects/...`
- `assets/isaac_source_root/assets/all_assets/Assets/object_grasps_simple.yaml`
- `assets/isaac_source_root/dexhand_env/cfg/task/Dexhand021proReconstruction.yaml`

Use `SOURCE_ROOT=/path/to/dexrobot_isaac bash scripts/import_isaac_assets.sh` to
refresh the checked-in bundle. The default `OBJECTS=cube1` keeps the import
small; pass for example `OBJECTS='cube1 cube2 sphere1'` to add more object
URDF/mesh assets. The upstream object repository contains much larger meshes,
so do not import the entire object tree into this repository.
