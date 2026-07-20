#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync --extra dexhandrl

cat <<'EOF'
Local uv environment is ready.
The `dexhandrl` extra installs torch, gymnasium, and skrl for the DexHandRL
reference-PD and policy workflows.

CPU smoke test:
  JAX_PLATFORMS=cpu .venv/bin/python sim/smoke_test.py --strict --device cpu

GPU smoke test needs a CUDA-capable JAX install. The Docker image is the supported GPU path:
  scripts/run_smoke_test.sh
EOF
