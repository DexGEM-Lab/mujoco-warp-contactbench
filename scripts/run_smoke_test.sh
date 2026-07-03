#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

docker run --rm \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$PWD":/workspace/mujoco-warp-contactbench \
  -w /workspace/mujoco-warp-contactbench \
  mujoco-warp-contactbench:latest \
  python docker/mujoco_mjx/smoke_test.py --strict --output logs/mujoco_mjx_live_contact.json
