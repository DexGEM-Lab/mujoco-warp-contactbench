#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

: "${OUTPUT_ROOT:=${PWD}/outputs}"
: "${DEVICE:=gpu}"
: "${CUDA_VISIBLE_DEVICES:=0}"

OUTPUT="${OUTPUT:-${OUTPUT_ROOT}/mjx_warp_contactbench_generated.lance}"

# Pass additional sim_to_lance.py arguments after the script name, for example:
#   DEVICE=cpu OUTPUT=outputs/test.lance scripts/run_sim_lance.sh --duration-seconds 1 --ball-count 16

docker_args=(
  run --rm
  --user "$(id -u):$(id -g)"
  -e HOME=/tmp
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false
  -e MUJOCO_GL=egl
  -e PYOPENGL_PLATFORM=egl
  -v "$PWD":/workspace/mujoco-warp-contactbench
  -w /workspace/mujoco-warp-contactbench
)

if [ "$DEVICE" = "gpu" ]; then
  docker_args+=(--gpus all -e CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES")
elif [ "$DEVICE" = "cpu" ]; then
  docker_args+=(-e JAX_PLATFORMS=cpu)
else
  echo "DEVICE must be cpu or gpu, got: $DEVICE" >&2
  exit 2
fi

docker "${docker_args[@]}" \
  mujoco-warp-contactbench:latest \
  python sim/sim_to_lance.py \
    --device "$DEVICE" \
    --replace \
    --output "$OUTPUT" \
    "$@"
