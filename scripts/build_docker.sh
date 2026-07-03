#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DOCKER_BUILDKIT=1 docker build -f Dockerfile -t mujoco-warp-contactbench:latest .
