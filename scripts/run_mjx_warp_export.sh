#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs
: "${CUDA_VISIBLE_DEVICES:=0}"

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
  python docker/mujoco_mjx/mjx_warp_contact_export.py \
    --scenario filled_tank \
    --ball-count 120 \
    --ball-radius 0.03 \
    --duration-seconds 10 \
    --output logs/mjx_warp_gpu_ball_pit_contact_native_10s.json \
    --contactbench-output logs/mjx_warp_gpu_ball_pit_contact_10s.json \
    --error-output logs/mjx_warp_gpu_contact_to_hand_mesh_error_raw.json \
    --scene-copy logs/mjx_warp_gpu_contact_error_scene.xml \
    --progress-interval 100
