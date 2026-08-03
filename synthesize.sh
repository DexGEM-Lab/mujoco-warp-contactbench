#!/usr/bin/env bash
set -Eeuo pipefail

# GPU checkpoint -> compact replay/visual or explicit full v2.3 synthetic Lance.
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
REFERENCE_FPS=${MANORL_REFERENCE_FPS:-}
OUTPUT_FORMAT=${MANORL_SYNTH_OUTPUT_FORMAT:-compact-replay-visual}
OUTPUT=${MANORL_SYNTH_OUTPUT:-$ROOT/outputs/manorl/synthetic_${OUTPUT_FORMAT}_${OBJECT}_${GESTURE}_$(date -u +%Y%m%dT%H%M%SZ).lance}
PYTHON=${MANORL_PYTHON:-$ROOT/.venv/bin/python}
PREDECODED_MANIFEST=${MANORL_PREDECODED_MANIFEST:-}
SEED=${MANORL_SYNTH_SEED:-42}
EPISODES_PER_IDENTITY=${MANORL_SYNTH_EPISODES_PER_IDENTITY:-5}
MAX_ATTEMPTS_PER_IDENTITY=${MANORL_SYNTH_MAX_ATTEMPTS_PER_IDENTITY:-10}

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
if [[ -n "$REFERENCE_FPS" && "$REFERENCE_FPS" != "100" && "$REFERENCE_FPS" != "120" ]]; then
  echo "MANORL_REFERENCE_FPS must be 100 or 120, got: $REFERENCE_FPS" >&2
  exit 2
fi
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "seed must be a non-negative integer, got: $SEED" >&2
  exit 2
fi
if [[ ! "$EPISODES_PER_IDENTITY" =~ ^[1-9][0-9]*$ ]]; then
  echo "episodes per identity must be positive, got: $EPISODES_PER_IDENTITY" >&2
  exit 2
fi
if [[ ! "$MAX_ATTEMPTS_PER_IDENTITY" =~ ^[1-9][0-9]*$ ]] || \
   (( MAX_ATTEMPTS_PER_IDENTITY < EPISODES_PER_IDENTITY )); then
  echo "max attempts must be at least episodes per identity" >&2
  exit 2
fi
if [[ "$OUTPUT_FORMAT" != "full" && "$OUTPUT_FORMAT" != "compact-replay-visual" ]]; then
  echo "MANORL_SYNTH_OUTPUT_FORMAT must be full or compact-replay-visual, got: $OUTPUT_FORMAT" >&2
  exit 2
fi

export PYTHONPATH=$ROOT
export CUDA_VISIBLE_DEVICES=$GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false

EXTRA_ARGS=()
if [[ -n "$REFERENCE_FPS" ]]; then
  EXTRA_ARGS+=(--reference-fps "$REFERENCE_FPS")
fi
if [[ -n "$PREDECODED_MANIFEST" ]]; then
  PREDECODED_MANIFEST=$(realpath -e "$PREDECODED_MANIFEST")
  EXTRA_ARGS+=(--predecoded-manifest "$PREDECODED_MANIFEST")
fi

exec "$PYTHON" "$ROOT/tools/export_manorl_synthetic_lance.py" \
  --device gpu \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --object "$OBJECT" \
  --gesture "$GESTURE" \
  --dataset-path "$DATASET" \
  --dataset-version "$DATASET_VERSION" \
  --num-envs "$NUM_ENVS" \
  --output-format "$OUTPUT_FORMAT" \
  --seed "$SEED" \
  --episodes-per-identity "$EPISODES_PER_IDENTITY" \
  --max-attempts-per-identity "$MAX_ATTEMPTS_PER_IDENTITY" \
  "${EXTRA_ARGS[@]}"
