#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)"
asset_root="${repo_root}/assets/dexstream_digital_assets"
manifest="${repo_root}/sim/manorl/task_assets/dexstream_manifest.json"
python_bin="${PYTHON:-python3}"

if ! command -v git-lfs >/dev/null 2>&1 && ! git lfs version >/dev/null 2>&1; then
  printf '%s\n' 'error: git-lfs is required to materialize DexStream assets' >&2
  exit 2
fi
if [[ ! -f "${manifest}" ]]; then
  printf 'error: DexStream manifest is absent: %s\n' "${manifest}" >&2
  exit 2
fi

expected_commit="$(${python_bin} - "${manifest}" <<'PYMANIFEST'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["source_commit"])
PYMANIFEST
)"

# Preserve an already materialized, correctly pinned checkout. This matters for
# air-gapped/deployment snapshots whose GitHub credentials are unavailable: the
# manifest and file hashes, not a redundant network checkout, are authoritative.
actual_commit=""
if [[ -d "${asset_root}" ]]; then
  actual_commit="$(git -C "${asset_root}" rev-parse HEAD 2>/dev/null || true)"
fi
if [[ "${actual_commit}" != "${expected_commit}" ]]; then
  # Do not run `git lfs install` here: GitGuard owns repository hooks.
  GIT_LFS_SKIP_SMUDGE=1 git -C "${repo_root}" submodule update --init assets/dexstream_digital_assets
  actual_commit="$(git -C "${asset_root}" rev-parse HEAD 2>/dev/null || true)"
fi
if [[ "${actual_commit}" != "${expected_commit}" ]]; then
  printf 'error: DexStream checkout %s does not match manifest %s\n' "${actual_commit:-<absent>}" "${expected_commit}" >&2
  exit 2
fi

mapfile -t lfs_paths < <("${python_bin}" - "${manifest}" <<'PYLFS'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
records = []
for hand in manifest["hands"].values():
    records.extend(hand["files"])
for obj in manifest["objects"].values():
    records.extend([obj["visual"], *obj["collisions"]])
for record in records:
    if record.get("storage") == "lfs":
        print(record["path"])
PYLFS
)
if [[ "${#lfs_paths[@]}" -eq 0 ]]; then
  printf '%s\n' 'error: ManoRL asset manifest contains no LFS paths' >&2
  exit 2
fi

is_lfs_pointer() {
  local path="$1"
  [[ -f "${path}" ]] || return 0
  # Keep binary bytes out of command substitution; grep -a only inspects the
  # short textual pointer header and produces no null-byte warnings.
  head -c 80 "${path}" 2>/dev/null \
    | grep -a -q '^version https://git-lfs.github.com/spec/v1'
}

needs_lfs_pull=0
for relative in "${lfs_paths[@]}"; do
  path="${asset_root}/${relative}"
  if [[ ! -f "${path}" ]] || is_lfs_pointer "${path}"; then
    needs_lfs_pull=1
    break
  fi
done
if [[ "${needs_lfs_pull}" -eq 1 ]]; then
  include="$(IFS=,; printf '%s' "${lfs_paths[*]}")"
  git -C "${asset_root}" lfs pull --include="${include}"
else
  printf '%s\n' 'DexStream LFS runtime files are already materialized; skipping network pull'
fi

PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${python_bin}" - <<'PYVERIFY'
from sim.manorl.assets import supported_object_types, validate_asset_manifest

for side in ("right", "left"):
    for object_type in supported_object_types():
        validate_asset_manifest(object_type, hand_side=side)
print("ManoRL DexStream assets verified for right, left, and bimanual composition")
PYVERIFY
