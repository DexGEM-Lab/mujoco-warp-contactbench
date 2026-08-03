#!/usr/bin/env bash
set -Eeuo pipefail

# Reference-following visual smoke without a policy checkpoint.
# Fixed contract: cube1/action-01, 20 worlds, residual actions disabled.
# Usage: ./test.sh [physical_gpu]

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GPU=${1:-${MANORL_GPU:-0}}
if [[ -n ${MANORL_PYTHON:-} ]]; then
  PYTHON=$MANORL_PYTHON
elif [[ -x $ROOT/.venv/bin/python ]]; then
  PYTHON=$ROOT/.venv/bin/python
else
  PYTHON=$(command -v python3 || true)
fi
DATASET=${MANORL_DATASET_PATH:-/mnt/nas-222-project/mocap_v2/lance_datasets/human_p1_guangguan/human_p1_guangguan_clean.lance}
DATASET_VERSION=${MANORL_DATASET_VERSION:-295}
REFERENCE_FPS=${MANORL_REFERENCE_FPS:-120}
SPEED=${MANORL_VIEW_SPEED:-0.5}
PRINT_EVERY=${MANORL_PRINT_EVERY:-100}

if [[ ! -x "$PYTHON" ]]; then
  echo "ManoRL Python is not executable: $PYTHON" >&2
  exit 2
fi
if [[ ! -d "$DATASET" ]]; then
  echo "Lance dataset is absent: $DATASET" >&2
  exit 2
fi
if [[ ! "$GPU" =~ ^[0-9]+$ ]]; then
  echo "physical_gpu must be a non-negative integer, got: $GPU" >&2
  exit 2
fi
if [[ "$REFERENCE_FPS" != "100" && "$REFERENCE_FPS" != "120" ]]; then
  echo "MANORL_REFERENCE_FPS must be 100 or 120, got: $REFERENCE_FPS" >&2
  exit 2
fi
if [[ -z ${DISPLAY:-} ]]; then
  if [[ -S /tmp/.X11-unix/X1 ]]; then
    export DISPLAY=:1
  elif [[ -S /tmp/.X11-unix/X0 ]]; then
    export DISPLAY=:0
  else
    echo "No graphical X11 display is available." >&2
    exit 2
  fi
fi
if [[ -z ${XAUTHORITY:-} && -f /run/user/$(id -u)/gdm/Xauthority ]]; then
  export XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority
fi

export PYTHONPATH=$ROOT
export CUDA_VISIBLE_DEVICES=$GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false

exec "$PYTHON" -m sim.manorl.view_environment \
  --device gpu \
  --speed "$SPEED" \
  --object cube1 \
  --gesture 01 \
  --dataset-path "$DATASET" \
  --dataset-version "$DATASET_VERSION" \
  --reference-fps "$REFERENCE_FPS" \
  --hand-side right \
  --num-envs 20 \
  --render-env 0 \
  --tile-envs 20 \
  --use_residual false \
  --terminal true \
  --loop \
  --print-every "$PRINT_EVERY"
