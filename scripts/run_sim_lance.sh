#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs
: "${CUDA_VISIBLE_DEVICES:=0}"

BACKEND="${BACKEND:-both}"
OUTPUT="${OUTPUT:-logs/mujoco_cpu_mjx_warp_gpu_ball_pit_generated.lance}"

# Pass additional sim_to_lance.py arguments after the script name, for example:
#   BACKEND=cpu OUTPUT=logs/test.lance scripts/run_sim_lance.sh --duration-seconds 1 --ball-count 16

docker run --rm \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e MUJOCO_GL=egl \
  -e PYOPENGL_PLATFORM=egl \
  -v "$PWD":/workspace/mujoco-warp-contactbench \
  -w /workspace/mujoco-warp-contactbench \
  mujoco-warp-contactbench:latest \
  python docker/mujoco_mjx/sim_to_lance.py \
    --backend "$BACKEND" \
    --replace \
    --output "$OUTPUT" \
    "$@"
