#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync
.venv/bin/python -m ipykernel install --user \
  --name mujoco-warp-contactbench \
  --display-name "MuJoCo Warp ContactBench (.venv)"
