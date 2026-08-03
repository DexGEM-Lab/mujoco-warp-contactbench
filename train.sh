#!/usr/bin/env bash
set -Eeuo pipefail

# Usage: ./train.sh [object|all] [num_envs] [physical_gpu]
# Example: ./train.sh cube1 2048 0
# Every eligible gesture for the selected object is trained. Use "all" for all
# materialized object/action pairs. MANORL_REFERENCE_FPS couples source and
# policy/control at 100 or 120 Hz (default: 120), with four physics substeps.

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OBJECT=${1:-${MANORL_TRAIN_OBJECT:-cube1}}
NUM_ENVS=${2:-${MANORL_NUM_ENVS:-2048}}
GPU=${3:-${MANORL_GPU:-0}}
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
UPDATES=${MANORL_UPDATES:-5000}
CHECKPOINT_INTERVAL=${MANORL_CHECKPOINT_INTERVAL:-100}
WARM_START_CHECKPOINT=${MANORL_WARM_START_CHECKPOINT:-}
WARM_START_PRIOR_UPDATES=${MANORL_WARM_START_PRIOR_UPDATES:-}
HAND_SIDE=${MANORL_HAND_SIDE:-right}
WANDB_ENABLED=${MANORL_WANDB:-true}
WANDB_ENTITY=${WANDB_ENTITY:-sunjay45711-dexerto}
WANDB_PROJECT=${WANDB_PROJECT:-mujoco-mano}
TIMEOUT=${MANORL_TIMEOUT:-48h}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

if [[ ! -x "$PYTHON" ]]; then
  echo "ManoRL Python is not executable: $PYTHON" >&2
  exit 2
fi
if [[ ! -d "$DATASET" ]]; then
  echo "Lance dataset is absent: $DATASET" >&2
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
if [[ "$REFERENCE_FPS" != "100" && "$REFERENCE_FPS" != "120" ]]; then
  echo "MANORL_REFERENCE_FPS must be 100 or 120, got: $REFERENCE_FPS" >&2
  exit 2
fi
if [[ -n "$WARM_START_CHECKPOINT" || -n "$WARM_START_PRIOR_UPDATES" ]]; then
  if [[ -z "$WARM_START_CHECKPOINT" || ! -f "$WARM_START_CHECKPOINT" ]]; then
    echo "MANORL_WARM_START_CHECKPOINT must name an existing checkpoint" >&2
    exit 2
  fi
  if [[ ! "$WARM_START_PRIOR_UPDATES" =~ ^[0-9]+$ ]]; then
    echo "MANORL_WARM_START_PRIOR_UPDATES must be a non-negative integer" >&2
    exit 2
  fi
  WARM_START_CHECKPOINT=$(realpath -e "$WARM_START_CHECKPOINT")
  WARM_START_CHECKPOINT_JSON=$(
    "$PYTHON" -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$WARM_START_CHECKPOINT"
  )
  WARM_START_PRIOR_UPDATES_JSON=$WARM_START_PRIOR_UPDATES
  CHECKPOINT_ARGS=(
    --warm-start-checkpoint "$WARM_START_CHECKPOINT"
    --warm-start-prior-updates "$WARM_START_PRIOR_UPDATES"
  )
else
  WARM_START_CHECKPOINT_JSON=null
  WARM_START_PRIOR_UPDATES_JSON=null
  CHECKPOINT_ARGS=()
fi

export PYTHONPATH=$ROOT
export CUDA_VISIBLE_DEVICES=$GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=online

EXTRA_MODEL_ARGS=()
if [[ "$OBJECT" == "all" ]]; then
  SELECTOR=all
  SELECTION_ARGS=(--all-pairs)
  EXTRA_MODEL_ARGS=(--unified-object-batch --warp-persistent-ccd-workspace)
else
  SELECTOR=$("$PYTHON" - "$DATASET" "$DATASET_VERSION" "$HAND_SIDE" "$OBJECT" <<'PY'
import sys
from pathlib import Path
import lance
from sim.manorl.trajectory import TrajectorySelection, _discover_trajectory_candidates
path = Path(sys.argv[1])
version = int(sys.argv[2])
hand_side = sys.argv[3]
object_type = sys.argv[4]
dataset = lance.dataset(str(path), version=version)
pairs, _ = _discover_trajectory_candidates(
    dataset,
    TrajectorySelection(
        selector="all",
        dataset_path=path,
        expected_dataset_version=version,
        hand_side=hand_side,
    ),
)
selected = [pair.canonical for pair in pairs if pair.object_type == object_type]
if not selected:
    raise SystemExit(f"no eligible gestures for object {object_type!r}")
print(",".join(selected))
PY
)
  SELECTION_ARGS=(--pairs "$SELECTOR")
fi

SAFE_OBJECT=${OBJECT//[^[:alnum:]_-]/_}
RUN_DIR=${MANORL_OUTPUT:-$ROOT/outputs/manorl/train_${SAFE_OBJECT}_all_gestures_f${REFERENCE_FPS}_n${NUM_ENVS}_g${GPU}/run-$STAMP}
mkdir -p "$RUN_DIR"
cat > "$RUN_DIR/run_manifest.json" <<EOF
{
  "mode": "train",
  "object": "$OBJECT",
  "selector": "$SELECTOR",
  "num_envs": $NUM_ENVS,
  "physical_gpu": $GPU,
  "dataset_path": "$DATASET",
  "dataset_version": $DATASET_VERSION,
  "reference_fps": $REFERENCE_FPS,
  "control_fps": $REFERENCE_FPS,
  "physics_fps": $((REFERENCE_FPS * 4)),
  "physics_substeps_per_control": 4,
  "pre_padding": 180,
  "post_padding": 250,
  "warm_start_checkpoint": $WARM_START_CHECKPOINT_JSON,
  "warm_start_prior_updates": $WARM_START_PRIOR_UPDATES_JSON,
  "hand_side": "$HAND_SIDE",
  "updates": $UPDATES,
  "checkpoint_interval_updates": $CHECKPOINT_INTERVAL,
  "residual_action": {
    "position_scale_m": 0.003,
    "max_position_offset_m": 0.03,
    "joint_scale_multiplier": 2.0,
    "joint_max_offset_multiplier": 2.0
  },
  "wandb_entity": "$WANDB_ENTITY",
  "wandb_project": "$WANDB_PROJECT",
  "output": "$RUN_DIR/training"
}
EOF

TELEMETRY=$RUN_DIR/gpu_telemetry.csv
printf 'timestamp,index,uuid,memory_used_mib,memory_free_mib,utilization_gpu_pct,temperature_c\n' > "$TELEMETRY"
(
  while kill -0 "$$" 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,index,uuid,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits >> "$TELEMETRY" || true
    sleep 15
  done
) &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true' EXIT

set +e
timeout --signal=INT --kill-after=120 "$TIMEOUT" "$PYTHON" -m tools.train_manorl_cube1 \
  --output "$RUN_DIR/training" \
  --updates "$UPDATES" \
  --checkpoint-interval-updates "$CHECKPOINT_INTERVAL" \
  --num-envs "$NUM_ENVS" \
  --evaluation-enabled false \
  --dataset-path "$DATASET" \
  --dataset-version "$DATASET_VERSION" \
  --reference-fps "$REFERENCE_FPS" \
  "${CHECKPOINT_ARGS[@]}" \
  --hand-side "$HAND_SIDE" \
  "${SELECTION_ARGS[@]}" \
  --pair-assignment-cycle 0 \
  --use_residual true \
  --position-scale 0.003 \
  --max-position-offset 0.03 \
  --joint-scale-multiplier 2.0 \
  --joint-max-offset-multiplier 2.0 \
  --film true \
  --terminal true \
  --headless true \
  --device-resident-controls true \
  --device-transition true \
  --capture-transition-diagnostics false \
  "${EXTRA_MODEL_ARGS[@]}" \
  --warp-ccd-iterations 16 \
  --warp-ccd-contacts-per-world 16 \
  --wandb "$WANDB_ENABLED" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-group "${SAFE_OBJECT}-all-gestures" \
  --wandb-name "manorl-${SAFE_OBJECT}-all-gestures-f${REFERENCE_FPS}-u${UPDATES}-n${NUM_ENVS}-g${GPU}-${STAMP}" \
  2>&1 | tee "$RUN_DIR/console.log"
RC=${PIPESTATUS[0]}
set -e
printf 'EXIT:%s\n' "$RC" | tee -a "$RUN_DIR/console.log"
exit "$RC"
