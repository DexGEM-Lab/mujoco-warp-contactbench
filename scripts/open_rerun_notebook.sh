#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/jupyter ]; then
  echo "Missing .venv/bin/jupyter. Run scripts/setup_viz_env.sh first." >&2
  exit 1
fi

.venv/bin/jupyter lab notebooks/rerun_cpu_gpu_contacts.ipynb
