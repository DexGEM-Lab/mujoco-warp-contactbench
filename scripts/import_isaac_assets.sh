#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/assets/isaac_source_root}"
OBJECTS="${OBJECTS:-cube1}"

if [[ -z "${SOURCE_ROOT}" ]]; then
  cat >&2 <<'EOF'
SOURCE_ROOT is required. Point it at a checkout of dexrobot_isaac that contains
assets/all_assets and dexhand_env/cfg/task/Dexhand021proReconstruction.yaml.

Example:
  SOURCE_ROOT=/path/to/dexrobot_isaac scripts/import_isaac_assets.sh

OBJECTS is a space-separated list of MANO object names and defaults to cube1.
EOF
  exit 2
fi

mkdir -p "${CACHE_ROOT}/assets/all_assets/Assets/HAND"
mkdir -p "${CACHE_ROOT}/assets/all_assets/Assets/sim/mano_objects_urdf"
mkdir -p "${CACHE_ROOT}/assets/all_assets/Assets/sim/mano_assets/objects"
mkdir -p "${CACHE_ROOT}/assets/all_assets/Assets"
mkdir -p "${CACHE_ROOT}/dexhand_env/cfg/task"

rsync -a "${SOURCE_ROOT}/assets/all_assets/Assets/HAND/dexhand021pro/" \
  "${CACHE_ROOT}/assets/all_assets/Assets/HAND/dexhand021pro/"
for object_name in ${OBJECTS}; do
  rsync -a "${SOURCE_ROOT}/assets/all_assets/Assets/sim/mano_objects_urdf/${object_name}.urdf" \
    "${CACHE_ROOT}/assets/all_assets/Assets/sim/mano_objects_urdf/"
  rsync -a "${SOURCE_ROOT}/assets/all_assets/Assets/sim/mano_assets/objects/${object_name}/" \
    "${CACHE_ROOT}/assets/all_assets/Assets/sim/mano_assets/objects/${object_name}/"
done
cp "${SOURCE_ROOT}/assets/all_assets/Assets/object_grasps_simple.yaml" \
  "${CACHE_ROOT}/assets/all_assets/Assets/object_grasps_simple.yaml"
cp "${SOURCE_ROOT}/dexhand_env/cfg/task/Dexhand021proReconstruction.yaml" \
  "${CACHE_ROOT}/dexhand_env/cfg/task/Dexhand021proReconstruction.yaml"

printf 'Copied DexHand021Pro assets and objects [%s] to %s\n' "${OBJECTS}" "${CACHE_ROOT}"
