#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/python ]; then
  echo "Missing .venv. Run scripts/setup_viz_env.sh first." >&2
  exit 1
fi

.venv/bin/python tools/export_contactbench_lance.py \
  --replace \
  --output logs/mujoco_cpu_mjx_warp_gpu_ball_pit_generated.lance
