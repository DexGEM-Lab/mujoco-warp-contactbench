#!/usr/bin/env bash
set -Eeuo pipefail

# GPU checkpoint -> corrected v2 synthetic Lance.
# Usage:
#   CHECKPOINT=/path/checkpoint-000500.pt ./synthesize.sh [object] [gesture] [num_envs] [gpu]

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OBJECT=${1:-${MANORL_SYNTH_OBJECT:-cube2}}
GESTURE=${2:-${MANORL_SYNTH_GESTURE:-02}}
NUM_ENVS=${3:-${MANORL_SYNTH_NUM_ENVS:-5}}
GPU=${4:-${MANORL_GPU:-0}}
CHECKPOINT=${CHECKPOINT:-${MANORL_CHECKPOINT:-}}
DATASET=${MANORL_DATASET_PATH:-/mnt/nas-222-project/mocap_v2/lance_datasets/human_p1_guangguan/human_p1_guangguan_clean.lance}
DATASET_VERSION=${MANORL_DATASET_VERSION:-295}
OUTPUT=${MANORL_SYNTH_OUTPUT:-$ROOT/outputs/manorl/synthetic_v2_${OBJECT}_${GESTURE}_$(date -u +%Y%m%dT%H%M%SZ).lance}
PYTHON=${MANORL_PYTHON:-$ROOT/.venv/bin/python}

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
if [[ ! "$NUM_ENVS" =~ ^[1-9][0-9]*$ ]]; then
  echo "num_envs must be a positive integer, got: $NUM_ENVS" >&2
  exit 2
fi
if [[ ! "$GPU" =~ ^[0-9]+$ ]]; then
  echo "physical_gpu must be a non-negative integer, got: $GPU" >&2
  exit 2
fi

export PYTHONPATH=$ROOT
export CUDA_VISIBLE_DEVICES=$GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false

exec "$PYTHON" "$ROOT/tools/export_manorl_synthetic_lance.py" \
  --device gpu \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --object "$OBJECT" \
  --gesture "$GESTURE" \
  --dataset-path "$DATASET" \
  --dataset-version "$DATASET_VERSION" \
  --num-envs "$NUM_ENVS"
