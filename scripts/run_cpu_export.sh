#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

docker run --rm \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e MUJOCO_GL=egl \
  -e PYOPENGL_PLATFORM=egl \
  -v "$PWD":/workspace/mujoco-warp-contactbench \
  -w /workspace/mujoco-warp-contactbench \
  mujoco-warp-contactbench:latest \
  python docker/mujoco_mjx/ball_pit_contact.py \
    --scenario filled_tank \
    --ball-count 120 \
    --ball-radius 0.03 \
    --duration-seconds 10 \
    --output logs/mujoco_cpu_ball_pit_contact_10s.json \
    --image logs/mujoco_cpu_ball_pit_contact_10s.svg \
    --no-video
