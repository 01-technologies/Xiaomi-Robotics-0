#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Launch Xiaomi-Robotics-0 PiPER stack (server + adapter, with optional client probe).

Usage:
  bash scripts/launch_piper_stack.sh [stack options] -- [piper_adapter args]

Required:
  --instruction TEXT         Language instruction passed to deploy/piper_adapter.py.

Server options:
  --model PATH_OR_HF_ID      Model to load for deploy/server.py (required unless --no-server).
  --server-host HOST         Host for server bind. Default: 127.0.0.1
  --server-port PORT         Server port and adapter model port. Default: 10086
  --no-server                Do not launch server (connect adapter to an existing server).
  --server-log PATH          Optional server log file path.

Adapter options:
  --model-host HOST          Host the adapter connects to. Default: 127.0.0.1

Runtime options:
  --source-rocm-env          Source ./set_envs.sh before launching processes.

Client probe options:
  --no-client-check          Skip client connectivity probe (enabled by default).

Common adapter passthrough args (after '--'):
  --profile {calvin,libero}  Select state/task profile (default: calvin).
  --task-id TEXT             Optional explicit task_id override.

Defaults:
  - This script uses `uv run` for all Python commands (no conda activation path).
  - If no image/camera/ROS input mode is passed, OpenCV mode is selected with `--base-cam-index 0`.
  - ROS mode is optional and not default; OpenCV + piper_sdk is the expected baseline.

Examples:
  # OpenCV mode by default (base camera index 0)
  bash scripts/launch_piper_stack.sh \
    --model XiaomiRobotics/Xiaomi-Robotics-0-Calvin-ABCD_D \
    --instruction "Pick up the red block" \
    -- --can-port can0

  # LIBERO checkpoint with LIBERO profile + explicit wrist camera
  bash scripts/launch_piper_stack.sh \
    --model XiaomiRobotics/Xiaomi-Robotics-0-LIBERO \
    --instruction "Open the drawer" \
    -- --profile libero --base-cam-index 0 --wrist-cam-index 1 --can-port can0

  # Use existing remote server; launch adapter only
  bash scripts/launch_piper_stack.sh \
    --no-server \
    --model-host 10.0.0.8 \
    --server-port 10086 \
    --instruction "Move to target" \
    -- --base-image /tmp/base.png --wrist-image /tmp/wrist.png
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH=""
INSTRUCTION=""
SERVER_HOST="127.0.0.1"
SERVER_PORT="10086"
MODEL_HOST="127.0.0.1"
SERVER_LOG=""
SOURCE_ROCM_ENV=0
START_SERVER=1
RUN_CLIENT_CHECK=1
ADAPTER_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    --instruction)
      INSTRUCTION="$2"
      shift 2
      ;;
    --server-host)
      SERVER_HOST="$2"
      shift 2
      ;;
    --server-port)
      SERVER_PORT="$2"
      shift 2
      ;;
    --model-host)
      MODEL_HOST="$2"
      shift 2
      ;;
    --server-log)
      SERVER_LOG="$2"
      shift 2
      ;;
    --source-rocm-env)
      SOURCE_ROCM_ENV=1
      shift
      ;;
    --no-server)
      START_SERVER=0
      shift
      ;;
    --no-client-check)
      RUN_CLIENT_CHECK=0
      shift
      ;;
    --conda-env)
      echo "Error: --conda-env is no longer supported. Use uv-managed environments."
      exit 1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      ADAPTER_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1"
      echo
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$INSTRUCTION" ]]; then
  echo "Error: --instruction is required."
  exit 1
fi

if [[ "$START_SERVER" -eq 1 && -z "$MODEL_PATH" ]]; then
  echo "Error: --model is required unless --no-server is set."
  exit 1
fi

if ! [[ "$SERVER_PORT" =~ ^[0-9]+$ ]]; then
  echo "Error: --server-port must be an integer, got '$SERVER_PORT'."
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is required but was not found in PATH."
  exit 1
fi

HAS_ROS_MODE=0
HAS_BASE_IMAGE=0
HAS_WRIST_IMAGE=0
HAS_BASE_CAM=0
for arg in "${ADAPTER_ARGS[@]}"; do
  case "$arg" in
    --ros-camera-mode)
      HAS_ROS_MODE=1
      ;;
    --base-image|--base-image=*)
      HAS_BASE_IMAGE=1
      ;;
    --wrist-image|--wrist-image=*)
      HAS_WRIST_IMAGE=1
      ;;
    --base-cam-index|--base-cam-index=*)
      HAS_BASE_CAM=1
      ;;
  esac
done

if [[ "$HAS_ROS_MODE" -eq 0 && "$HAS_BASE_IMAGE" -eq 0 && "$HAS_WRIST_IMAGE" -eq 0 && "$HAS_BASE_CAM" -eq 0 ]]; then
  ADAPTER_ARGS=(--base-cam-index 0 "${ADAPTER_ARGS[@]}")
  echo "No adapter input mode provided; defaulting to OpenCV with --base-cam-index 0."
fi

wait_for_server() {
  local host="$1"
  local port="$2"
  local timeout_s="$3"

  uv run python - "$host" "$port" "$timeout_s" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
timeout_s = float(sys.argv[3])
deadline = time.time() + timeout_s

while time.time() < deadline:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        try:
            sock.connect((host, port))
        except OSError:
            time.sleep(0.2)
        else:
            print(f"Server reachable at {host}:{port}")
            sys.exit(0)

print(f"Timed out waiting for server at {host}:{port}", file=sys.stderr)
sys.exit(1)
PY
}

client_probe() {
  local host="$1"
  local port="$2"

  uv run python - "$host" "$port" <<'PY'
import sys

from deploy.client import Client

host = sys.argv[1]
port = int(sys.argv[2])

client = Client(host=host, port=port)
client.close()
print(f"Client probe succeeded for {host}:{port}")
PY
}

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "Stopping model server (pid=$SERVER_PID)..."
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ "$SOURCE_ROCM_ENV" -eq 1 ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/set_envs.sh"
  echo "Sourced ROCm env vars from set_envs.sh"
fi

if [[ "$START_SERVER" -eq 1 ]]; then
  if [[ -z "$SERVER_LOG" ]]; then
    mkdir -p "$REPO_ROOT/logs"
    SERVER_LOG="$REPO_ROOT/logs/piper_stack_server_${SERVER_PORT}.log"
  fi

  mkdir -p "$(dirname "$SERVER_LOG")"

  echo "Starting model server on ${SERVER_HOST}:${SERVER_PORT}"
  echo "Model: ${MODEL_PATH}"
  echo "Server log: ${SERVER_LOG}"
  uv run python -m deploy.server \
    --model "$MODEL_PATH" \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" >"$SERVER_LOG" 2>&1 &
  SERVER_PID="$!"

  wait_for_server "$MODEL_HOST" "$SERVER_PORT" 60
fi

if [[ "$RUN_CLIENT_CHECK" -eq 1 ]]; then
  client_probe "$MODEL_HOST" "$SERVER_PORT"
fi

echo "Starting PiPER adapter..."
uv run python -m deploy.piper_adapter \
  --model-host "$MODEL_HOST" \
  --model-port "$SERVER_PORT" \
  --instruction "$INSTRUCTION" \
  "${ADAPTER_ARGS[@]}"
