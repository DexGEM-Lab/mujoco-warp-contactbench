#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-gpu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PWD}/outputs}"
OUTPUT="${OUTPUT:-${OUTPUT_ROOT}/dexhandrl_reference_pd_metrics.json}"
LANCE="${LANCE:-/mnt/nas-222-project/mocap/for_retargeting/dexhand021pro/results_all_mano/all/lance_new_all_generated_mano/lance_result_archive_full/dexhand021pro_lance_new_all_generated_mano_retarget_result_archive.lance}"
ISAAC_SOURCE_ROOT="${ISAAC_SOURCE_ROOT:-}"
PYTHON="${PYTHON:-python}"

ARGS=(sim/dexhandrl/reference_pd_replay.py --device "${DEVICE}" --lance "${LANCE}" --output "${OUTPUT}")
if [[ -n "${ISAAC_SOURCE_ROOT}" ]]; then
  ARGS+=(--isaac-source-root "${ISAAC_SOURCE_ROOT}")
fi

"${PYTHON}" "${ARGS[@]}" "$@"
