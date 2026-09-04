#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)"
asset_root="${repo_root}/assets/dexstream_digital_assets"
manifest="${repo_root}/sim/manorl/task_assets/dexstream_manifest.json"

if ! command -v git-lfs >/dev/null 2>&1 && ! git lfs version >/dev/null 2>&1; then
  printf '%s\n' 'error: git-lfs is required to materialize DexStream assets' >&2
  exit 2
fi

GIT_LFS_SKIP_SMUDGE=1 git -C "${repo_root}" submodule update --init assets/dexstream_digital_assets

include="$(${PYTHON:-python3} - "${manifest}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
paths = []
for hand in manifest["hands"].values():
    paths.extend(record["path"] for record in hand["files"] if record["storage"] == "lfs")
for obj in manifest["objects"].values():
    records = [obj["visual"], *obj["collisions"]]
    paths.extend(record["path"] for record in records if record["storage"] == "lfs")
print(",".join(sorted(set(paths))))
PY
)"

if [[ -z "${include}" ]]; then
  printf '%s\n' 'error: ManoRL asset manifest contains no LFS paths' >&2
  exit 2
fi

git -C "${asset_root}" lfs pull --include="${include}"

PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON:-python3}" - <<'PY'
from sim.manorl.assets import supported_object_types, validate_asset_manifest

for side in ("right", "left"):
    for object_type in supported_object_types():
        validate_asset_manifest(object_type, hand_side=side)
print("ManoRL DexStream assets verified for right, left, and bimanual composition")
PY
