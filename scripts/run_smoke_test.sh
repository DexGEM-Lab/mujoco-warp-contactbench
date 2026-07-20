#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

: "${OUTPUT_ROOT:=${PWD}/outputs}"
: "${OUTPUT:=${OUTPUT_ROOT}/mjx_warp_${DEVICE:-gpu}_smoke.lance}"
: "${DEVICE:=gpu}"

docker_args=(
  run --rm
  --user "$(id -u):$(id -g)"
  -e HOME=/tmp
  -v "$PWD":/workspace/mujoco-warp-contactbench
  -w /workspace/mujoco-warp-contactbench
)

if [ "$DEVICE" = "gpu" ]; then
  docker_args+=(--gpus all)
elif [ "$DEVICE" = "cpu" ]; then
  docker_args+=(-e JAX_PLATFORMS=cpu)
else
  echo "DEVICE must be cpu or gpu, got: $DEVICE" >&2
  exit 2
fi

docker "${docker_args[@]}" \
  mujoco-warp-contactbench:latest \
  python sim/smoke_test.py --strict --device "$DEVICE" --output "$OUTPUT"
