#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

docker build -f docker/mujoco_mjx/Dockerfile -t mujoco-warp-contactbench:latest .
