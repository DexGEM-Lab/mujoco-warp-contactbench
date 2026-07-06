#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

: "${DISPLAY:?DISPLAY is not set. Run this from an active X11 desktop/session.}"
: "${MODEL_XML:=.tmp/mujoco_cube/cube_3x3x3.xml}"

if ! command -v xauth >/dev/null 2>&1; then
  echo "xauth is required on the host for X11 cookie forwarding." >&2
  echo "Install it, for example: sudo apt-get install xauth" >&2
  exit 2
fi

if [[ "$DISPLAY" != localhost:* && "$DISPLAY" != 127.0.0.1:* ]] && [ ! -d /tmp/.X11-unix ]; then
  echo "X11 socket directory /tmp/.X11-unix is not available for DISPLAY=$DISPLAY." >&2
  exit 2
fi

if [ ! -f "$MODEL_XML" ]; then
  echo "Model XML not found: $MODEL_XML" >&2
  echo "Set MODEL_XML=/path/to/model.xml or fetch the asset first." >&2
  exit 2
fi

mkdir -p .tmp
XAUTH_FILE="$PWD/.tmp/docker.xauth"
touch "$XAUTH_FILE"
chmod 600 "$XAUTH_FILE"

# Create a container-readable xauth file. nlist can be empty on permissive X11
# sessions, but most desktop sessions provide a MIT-MAGIC-COOKIE entry.
if xauth nlist "$DISPLAY" | sed -e 's/^..../ffff/' | xauth -f "$XAUTH_FILE" nmerge - 2>/dev/null; then
  :
else
  echo "Failed to create $XAUTH_FILE for DISPLAY=$DISPLAY" >&2
  exit 2
fi

export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"

: "${COMPOSE_SERVICE:=contactbench-gpu}"

compose_args=()
if [ "$COMPOSE_SERVICE" = "contactbench-gpu" ]; then
  compose_args+=(--profile gpu)
fi

# Build if needed, then run the MuJoCo viewer through the X11-enabled service.
docker compose "${compose_args[@]}" build "$COMPOSE_SERVICE"
docker compose "${compose_args[@]}" run --rm "$COMPOSE_SERVICE" python -m mujoco.viewer "$MODEL_XML"
