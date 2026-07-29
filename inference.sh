#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   CHECKPOINT=/path/to/checkpoint.pt ./inference.sh [object] [gesture] [render_count] [physical_gpu]
# Example:
#   CHECKPOINT=outputs/manorl/run/training/checkpoint-000900.pt ./inference.sh cube1 01 20 0

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OBJECT=${1:-${MANORL_INFERENCE_OBJECT:-cube1}}
GESTURE=${2:-${MANORL_INFERENCE_GESTURE:-01}}
RENDER_COUNT=${3:-${MANORL_RENDER_COUNT:-20}}
GPU=${4:-${MANORL_GPU:-0}}
CHECKPOINT=${CHECKPOINT:-${MANORL_CHECKPOINT:-}}
if [[ -n ${MANORL_PYTHON:-} ]]; then
  PYTHON=$MANORL_PYTHON
elif [[ -x $ROOT/.venv/bin/python ]]; then
  PYTHON=$ROOT/.venv/bin/python
else
  PYTHON=$(command -v python3 || true)
fi
DATASET=${MANORL_DATASET_PATH:-/mnt/nas-222-project/mocap_v2/lance_datasets/human_p1_guangguan/human_p1_guangguan_clean.lance}
DATASET_VERSION=${MANORL_DATASET_VERSION:-295}
HAND_SIDE=${MANORL_HAND_SIDE:-right}
SPEED=${MANORL_VIEW_SPEED:-0.5}
PRINT_EVERY=${MANORL_PRINT_EVERY:-100}

if [[ -z "$CHECKPOINT" ]]; then
  echo "Set CHECKPOINT or MANORL_CHECKPOINT to a native ManoRL checkpoint." >&2
  exit 2
fi
CHECKPOINT=$(realpath -e "$CHECKPOINT")
if [[ ! -f "$CHECKPOINT.json" ]]; then
  echo "Checkpoint sidecar is absent: $CHECKPOINT.json" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "ManoRL Python is not executable: $PYTHON" >&2
  exit 2
fi
if [[ ! "$RENDER_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "render_count must be a positive integer, got: $RENDER_COUNT" >&2
  exit 2
fi
if [[ ! "$GPU" =~ ^[0-9]+$ ]]; then
  echo "physical_gpu must be a non-negative integer, got: $GPU" >&2
  exit 2
fi

# Same-user desktop sessions usually need no explicit Xauthority. On a remote
# tmux shell, choose the active Server2 X11 display when DISPLAY is unset.
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
  --object "$OBJECT" \
  --gesture "$GESTURE" \
  --dataset-path "$DATASET" \
  --dataset-version "$DATASET_VERSION" \
  --hand-side "$HAND_SIDE" \
  --checkpoint "$CHECKPOINT" \
  --num-envs "$RENDER_COUNT" \
  --render-env 0 \
  --tile-envs "$RENDER_COUNT" \
  --use_residual true \
  --terminal true \
  --loop \
  --print-every "$PRINT_EVERY"
