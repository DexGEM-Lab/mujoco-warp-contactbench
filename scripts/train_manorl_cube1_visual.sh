#!/usr/bin/env bash
set -euo pipefail

python_bin="${MANORL_PYTHON:-/home/jay/anaconda3/envs/manorl_mujoco/bin/python}"
output="${MANORL_OUTPUT:-outputs/manorl/cube1_01_visual_$(date +%Y%m%d_%H%M%S)_$$}"

exec "$python_bin" -m tools.train_manorl_cube1 \
  --output "$output" \
  --object cube1 --gesture 01 \
  --num-envs 8 --evaluation-num-envs 8 --minibatch-size 384 \
  --headless false --viewer-envs 8 --viewer-stride 1 \
  --wandb false \
  "$@"
