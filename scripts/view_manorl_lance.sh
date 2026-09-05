#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)

usage() {
  cat <<'EOF'
Usage:
  scripts/view_manorl_lance.sh --dataset PATH --object NAME --gesture ACTION [options]

Required:
  --dataset PATH           Lance dataset directory
  --object NAME            ManoRL object/scene name, for example banana
  --gesture ACTION         Leading numeric action ID, for example 18

Trajectory options:
  --dataset-version N      Pin a Lance version; latest when omitted
  --reference-fps 100|120  Coupled source/control clock (default: 120)
  --pre-padding N          Repeated boundary frames before movement (default: 180)
  --post-padding N         Repeated boundary frames after movement (default: 250)
  --hand-side SIDE         auto, right, left, or both (default: auto)

Viewer options:
  --display DISPLAY        X11 display; auto-select :1 then :0 when unset
  --xauthority PATH        X11 authority file; auto-detected when omitted
  --device cpu|gpu         MJX-Warp device (default: cpu)
  --gpu N                  Physical CUDA GPU for --device gpu (default: 0)
  --speed X                Playback speed multiplier (default: 0.5)
  --loop | --no-loop       Loop after episode reset (default: --loop)
  --terminal true|false    Enable deviation termination (default: false)
  --use-residual true|false
                           Enable residual action processing (default: false)
  --print-every N          Telemetry cadence in control calls (default: 100)
  --decode-chunk-size N    Bounded Lance decode batch size (default: 32)
  --python PATH            Python executable
  --dry-run                Print the resolved environment and command only
  -h, --help               Show this help

This reference viewer always runs exactly one environment and renders env 0.
Examples:
  scripts/view_manorl_lance.sh \
    --dataset /path/to/capture.lance --dataset-version 12 \
    --object banana --gesture 18 --reference-fps 100 \
    --pre-padding 180 --post-padding 180 --display :1
EOF
}

DATASET=""
OBJECT=""
GESTURE=""
DATASET_VERSION=""
REFERENCE_FPS=120
PRE_PADDING=180
POST_PADDING=250
HAND_SIDE=auto
DISPLAY_VALUE=${DISPLAY:-}
XAUTHORITY_VALUE=${XAUTHORITY:-}
DEVICE=cpu
GPU=0
SPEED=0.5
LOOP=true
TERMINAL=false
USE_RESIDUAL=false
PRINT_EVERY=100
DECODE_CHUNK_SIZE=32
PYTHON=${MANORL_PYTHON:-}
DRY_RUN=false

require_value() {
  if [[ $# -lt 2 || -z ${2:-} ]]; then
    printf 'error: %s requires a value\n' "$1" >&2
    usage >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) require_value "$@"; DATASET=$2; shift 2 ;;
    --object) require_value "$@"; OBJECT=$2; shift 2 ;;
    --gesture) require_value "$@"; GESTURE=$2; shift 2 ;;
    --dataset-version) require_value "$@"; DATASET_VERSION=$2; shift 2 ;;
    --reference-fps) require_value "$@"; REFERENCE_FPS=$2; shift 2 ;;
    --pre-padding) require_value "$@"; PRE_PADDING=$2; shift 2 ;;
    --post-padding) require_value "$@"; POST_PADDING=$2; shift 2 ;;
    --hand-side) require_value "$@"; HAND_SIDE=$2; shift 2 ;;
    --display) require_value "$@"; DISPLAY_VALUE=$2; shift 2 ;;
    --xauthority) require_value "$@"; XAUTHORITY_VALUE=$2; shift 2 ;;
    --device) require_value "$@"; DEVICE=$2; shift 2 ;;
    --gpu) require_value "$@"; GPU=$2; shift 2 ;;
    --speed) require_value "$@"; SPEED=$2; shift 2 ;;
    --terminal) require_value "$@"; TERMINAL=$2; shift 2 ;;
    --use-residual) require_value "$@"; USE_RESIDUAL=$2; shift 2 ;;
    --print-every) require_value "$@"; PRINT_EVERY=$2; shift 2 ;;
    --decode-chunk-size) require_value "$@"; DECODE_CHUNK_SIZE=$2; shift 2 ;;
    --python) require_value "$@"; PYTHON=$2; shift 2 ;;
    --loop) LOOP=true; shift ;;
    --no-loop) LOOP=false; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'error: unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z $DATASET || -z $OBJECT || -z $GESTURE ]]; then
  printf '%s\n' 'error: --dataset, --object, and --gesture are required' >&2
  usage >&2
  exit 2
fi
if [[ ! -d $DATASET ]]; then
  printf 'error: Lance dataset is absent: %s\n' "$DATASET" >&2
  exit 2
fi
if [[ -n $DATASET_VERSION && ! $DATASET_VERSION =~ ^[1-9][0-9]*$ ]]; then
  printf 'error: --dataset-version must be a positive integer\n' >&2
  exit 2
fi
if [[ $REFERENCE_FPS != 100 && $REFERENCE_FPS != 120 ]]; then
  printf 'error: --reference-fps must be 100 or 120\n' >&2
  exit 2
fi
for item in "$PRE_PADDING" "$POST_PADDING"; do
  if [[ ! $item =~ ^[0-9]+$ ]]; then
    printf 'error: padding values must be non-negative integers\n' >&2
    exit 2
  fi
done
if [[ ! $PRINT_EVERY =~ ^[1-9][0-9]*$ || ! $DECODE_CHUNK_SIZE =~ ^[1-9][0-9]*$ ]]; then
  printf 'error: --print-every and --decode-chunk-size must be positive integers\n' >&2
  exit 2
fi
if [[ ! $GPU =~ ^[0-9]+$ ]]; then
  printf 'error: --gpu must be a non-negative integer\n' >&2
  exit 2
fi
if [[ ! $SPEED =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ || $SPEED =~ ^0*([.]0*)?$ ]]; then
  printf 'error: --speed must be positive\n' >&2
  exit 2
fi
case "$HAND_SIDE" in auto|right|left|both) ;; *) printf 'error: invalid --hand-side: %s\n' "$HAND_SIDE" >&2; exit 2 ;; esac
case "$DEVICE" in cpu|gpu) ;; *) printf 'error: --device must be cpu or gpu\n' >&2; exit 2 ;; esac
case "$TERMINAL" in true|false) ;; *) printf 'error: --terminal must be true or false\n' >&2; exit 2 ;; esac
case "$USE_RESIDUAL" in true|false) ;; *) printf 'error: --use-residual must be true or false\n' >&2; exit 2 ;; esac

if [[ -z $PYTHON ]]; then
  if [[ -x $ROOT/.venv/bin/python ]]; then
    PYTHON=$ROOT/.venv/bin/python
  else
    PYTHON=$(command -v python3 || true)
  fi
fi
if [[ -z $PYTHON || ! -x $PYTHON ]]; then
  printf 'error: ManoRL Python is not executable: %s\n' "${PYTHON:-<absent>}" >&2
  exit 2
fi

if [[ -z $DISPLAY_VALUE ]]; then
  if [[ -S /tmp/.X11-unix/X1 ]]; then
    DISPLAY_VALUE=:1
  elif [[ -S /tmp/.X11-unix/X0 ]]; then
    DISPLAY_VALUE=:0
  else
    printf '%s\n' 'error: no X11 display is available; pass --display explicitly' >&2
    exit 2
  fi
fi
if [[ -z $XAUTHORITY_VALUE ]]; then
  if [[ -f $HOME/.Xauthority ]]; then
    XAUTHORITY_VALUE=$HOME/.Xauthority
  elif [[ -f /run/user/$(id -u)/gdm/Xauthority ]]; then
    XAUTHORITY_VALUE=/run/user/$(id -u)/gdm/Xauthority
  fi
fi

command=(
  "$PYTHON" -m sim.manorl.view_environment
  --device "$DEVICE"
  --speed "$SPEED"
  --object "$OBJECT"
  --gesture "$GESTURE"
  --dataset-path "$DATASET"
  --reference-fps "$REFERENCE_FPS"
  --pre-padding "$PRE_PADDING"
  --post-padding "$POST_PADDING"
  --hand-side "$HAND_SIDE"
  --num-envs 1
  --render-env 0
  --tile-envs 1
  --use_residual "$USE_RESIDUAL"
  --terminal "$TERMINAL"
  --print-every "$PRINT_EVERY"
)
if [[ -n $DATASET_VERSION ]]; then
  command+=(--dataset-version "$DATASET_VERSION")
fi
if [[ $LOOP == true ]]; then
  command+=(--loop)
else
  command+=(--no-loop)
fi

jax_platform=cpu
cuda_visible=""
if [[ $DEVICE == gpu ]]; then
  jax_platform=cuda
  cuda_visible=$GPU
fi

if [[ $DRY_RUN == true ]]; then
  printf 'DISPLAY=%q\n' "$DISPLAY_VALUE"
  printf 'XAUTHORITY=%q\n' "$XAUTHORITY_VALUE"
  printf 'JAX_PLATFORMS=%q\n' "$jax_platform"
  printf 'CUDA_VISIBLE_DEVICES=%q\n' "$cuda_visible"
  printf 'LANCE_DECODE_CHUNK_SIZE=%q\n' "$DECODE_CHUNK_SIZE"
  printf 'command:'
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi

if ! command -v xdpyinfo >/dev/null 2>&1; then
  printf '%s\n' 'error: xdpyinfo is required to validate the X11 viewer session' >&2
  exit 2
fi
if [[ -n $XAUTHORITY_VALUE ]]; then
  DISPLAY=$DISPLAY_VALUE XAUTHORITY=$XAUTHORITY_VALUE xdpyinfo >/dev/null 2>&1 || {
    printf 'error: cannot open X11 display %s with XAUTHORITY=%s\n' "$DISPLAY_VALUE" "$XAUTHORITY_VALUE" >&2
    exit 2
  }
elif ! DISPLAY=$DISPLAY_VALUE xdpyinfo >/dev/null 2>&1; then
  printf 'error: cannot open X11 display %s\n' "$DISPLAY_VALUE" >&2
  exit 2
fi

available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
minimum_kib=$((4 * 1024 * 1024))
if [[ -z $available_kib || $available_kib -lt $minimum_kib ]]; then
  printf 'error: viewer requires at least 4 GiB available RAM; found %s KiB\n' "${available_kib:-unknown}" >&2
  exit 2
fi
printf 'ManoRL viewer preflight: envs=1 RAM_available=%.1f_GiB device=%s display=%s\n' \
  "$(awk -v kib="$available_kib" 'BEGIN {print kib / 1024 / 1024}')" "$DEVICE" "$DISPLAY_VALUE" >&2

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader >&2 || true
fi
if [[ $DEVICE == gpu ]]; then
  free_mib=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [[ ! $free_mib =~ ^[0-9]+$ || $free_mib -lt 4096 ]]; then
    printf 'error: GPU %s requires at least 4096 MiB free for this viewer; found %s\n' "$GPU" "${free_mib:-unknown}" >&2
    exit 2
  fi
fi

export DISPLAY=$DISPLAY_VALUE
if [[ -n $XAUTHORITY_VALUE ]]; then
  export XAUTHORITY=$XAUTHORITY_VALUE
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=$jax_platform
export CUDA_VISIBLE_DEVICES=$cuda_visible
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LANCE_DECODE_CHUNK_SIZE=$DECODE_CHUNK_SIZE

printf 'Launching %s:%s from %s (version=%s, reference/control=%s Hz, pre/post=%s/%s, loop=%s)\n' \
  "$OBJECT" "$GESTURE" "$DATASET" "${DATASET_VERSION:-latest}" "$REFERENCE_FPS" \
  "$PRE_PADDING" "$POST_PADDING" "$LOOP" >&2
exec "${command[@]}"
