#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/assets/isaac_source_root}"
OBJECTS="${OBJECTS:-cube1}"

if [[ -z "${SOURCE_ROOT}" ]]; then
  cat >&2 <<'EOF'
SOURCE_ROOT is required. Point it at a checkout containing
assets/all_assets/Assets and dexhand_env/cfg/task/Dexhand021proReconstruction.yaml.

Example:
  SOURCE_ROOT=/path/to/dexrobot_isaac scripts/import_isaac_assets.sh

OBJECTS is a space-separated list of object names and defaults to cube1.
Every object is copied from Assets/sim/decomposed/<object>/<object>.urdf,
including every referenced CoACD collision piece and its visual source mesh.
EOF
  exit 2
fi

ALL_ASSETS_SOURCE="${SOURCE_ROOT}/assets/all_assets"
ASSETS_SOURCE="${ALL_ASSETS_SOURCE}/Assets"
if [[ ! -d "${ASSETS_SOURCE}" ]]; then
  echo "all_assets source is missing: ${ASSETS_SOURCE}" >&2
  exit 2
fi

mkdir -p "${CACHE_ROOT}/assets/all_assets/Assets/HAND"
mkdir -p "${CACHE_ROOT}/assets/all_assets/Assets/sim/decomposed"
mkdir -p "${CACHE_ROOT}/assets/all_assets/Assets/sim/mano_assets/objects"
mkdir -p "${CACHE_ROOT}/dexhand_env/cfg/task"

rsync -a "${ASSETS_SOURCE}/HAND/dexhand021pro/" \
  "${CACHE_ROOT}/assets/all_assets/Assets/HAND/dexhand021pro/"
for object_name in ${OBJECTS}; do
  source_object="${ASSETS_SOURCE}/sim/decomposed/${object_name}"
  if [[ ! -f "${source_object}/${object_name}.urdf" ]]; then
    echo "decomposed object URDF is missing: ${source_object}/${object_name}.urdf" >&2
    exit 2
  fi
  rsync -a --delete "${source_object}/" \
    "${CACHE_ROOT}/assets/all_assets/Assets/sim/decomposed/${object_name}/"
  rsync -a --delete "${ASSETS_SOURCE}/sim/mano_assets/objects/${object_name}/" \
    "${CACHE_ROOT}/assets/all_assets/Assets/sim/mano_assets/objects/${object_name}/"
done
cp "${ASSETS_SOURCE}/object_grasps_simple.yaml" \
  "${CACHE_ROOT}/assets/all_assets/Assets/object_grasps_simple.yaml"
cp "${SOURCE_ROOT}/dexhand_env/cfg/task/Dexhand021proReconstruction.yaml" \
  "${CACHE_ROOT}/dexhand_env/cfg/task/Dexhand021proReconstruction.yaml"

printf 'Copied DexHand021Pro assets and decomposed objects [%s] to %s\n' "${OBJECTS}" "${CACHE_ROOT}"
