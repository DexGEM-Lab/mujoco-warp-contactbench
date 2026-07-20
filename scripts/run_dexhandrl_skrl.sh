#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
MODE="${MODE:-train}"
DEVICE="${DEVICE:-cpu}"
NUM_ENVS="${NUM_ENVS:-8}"
LANCE="${LANCE:-/mnt/nas-222-project/mocap/for_retargeting/dexhand021pro/results_all_mano/all/lance_new_all_generated_mano/lance_result_archive_full/dexhand021pro_lance_new_all_generated_mano_retarget_result_archive.lance}"
ISAAC_SOURCE_ROOT="${ISAAC_SOURCE_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PWD}/outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/skrl_dexhandrl}"

ARGS=(-m sim.dexhandrl.skrl_train --mode "${MODE}" --device "${DEVICE}" --num-envs "${NUM_ENVS}" --lance-path "${LANCE}" --output-dir "${OUTPUT_DIR}")
if [[ -n "${ISAAC_SOURCE_ROOT}" ]]; then
  ARGS+=(--isaac-source-root "${ISAAC_SOURCE_ROOT}")
fi

"${PYTHON}" "${ARGS[@]}" "$@"
