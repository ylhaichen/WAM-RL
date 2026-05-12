#!/usr/bin/env bash
set -euo pipefail

umask 007

REPO_ROOT=${REPO_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
cd "${REPO_ROOT}"

PYTHON_BIN=${PYTHON_BIN:-python}
CONFIG_NAME=${CONFIG_NAME:-libero}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PORT=${PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}
LIBERO_BENCHMARK=${LIBERO_BENCHMARK:-libero_10}
TASK_START=${TASK_START:-0}
TASK_END=${TASK_END:-1}
TEST_NUM=${TEST_NUM:-1}
SAVE_ROOT=${SAVE_ROOT:-outputs/libero_smoke/server}
OUT_DIR=${OUT_DIR:-outputs/libero_smoke/client}
SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT:-900}
SERVER_POLL_INTERVAL=${SERVER_POLL_INTERVAL:-5}
KEEP_SERVER=${KEEP_SERVER:-0}

if [[ -n "${MODEL_PATH:-}" && -z "${WAN_VA_MODEL_PATH:-}" ]]; then
    export WAN_VA_MODEL_PATH="${MODEL_PATH}"
fi

if [[ -z "${WAN_VA_MODEL_PATH:-}" ]]; then
    echo "WAN_VA_MODEL_PATH is required, for example:" >&2
    echo "  WAN_VA_MODEL_PATH=/data/wam-rl/checkpoints/lingbot-va-posttrain-libero-long bash script/run_libero_smoke.sh" >&2
    exit 2
fi

if [[ ! -d "${WAN_VA_MODEL_PATH}" ]]; then
    echo "WAN_VA_MODEL_PATH does not exist: ${WAN_VA_MODEL_PATH}" >&2
    exit 2
fi

if [[ ! -d "${WAN_VA_MODEL_PATH}/transformer" ]]; then
    echo "WAN_VA_MODEL_PATH should point to a LingBot-VA checkpoint with a transformer/ subdirectory: ${WAN_VA_MODEL_PATH}" >&2
    exit 2
fi

mkdir -p "${SAVE_ROOT}" "${OUT_DIR}"
SERVER_LOG=${SERVER_LOG:-"${OUT_DIR}/server.log"}
CLIENT_LOG=${CLIENT_LOG:-"${OUT_DIR}/client.log"}

echo "Repo: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Checkpoint: ${WAN_VA_MODEL_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Benchmark: ${LIBERO_BENCHMARK}, task range: [${TASK_START}, ${TASK_END}), episodes: ${TEST_NUM}"
echo "Server log: ${SERVER_LOG}"
echo "Client log: ${CLIENT_LOG}"

"${PYTHON_BIN}" - <<'PY'
import importlib
import sys

required = [
    ("torch", "torch"),
    ("easydict", "easydict"),
    ("libero", "libero"),
    ("lerobot.datasets.utils", "lerobot"),
    ("websockets", "websockets"),
    ("cv2", "opencv-python"),
    ("imageio", "imageio"),
]

missing = []
for module_name, package_name in required:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        missing.append(f"{package_name} ({module_name}): {exc}")

if missing:
    print("Missing or broken Python dependencies:", file=sys.stderr)
    for item in missing:
        print(f"  - {item}", file=sys.stderr)
    sys.exit(2)

import torch

print(f"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

cleanup() {
    if [[ "${KEEP_SERVER}" == "1" ]]; then
        echo "KEEP_SERVER=1, leaving server running with pid ${SERVER_PID:-unknown}"
        return
    fi
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Stopping server pid ${SERVER_PID}"
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "Starting LingBot-VA LIBERO server..."
"${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port "${MASTER_PORT}" \
    -m wan_va.wan_va_server \
    --config-name "${CONFIG_NAME}" \
    --port "${PORT}" \
    --save_root "${SAVE_ROOT}" \
    "$@" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + SERVER_READY_TIMEOUT))
while (( SECONDS < deadline )); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Server exited before becoming ready. Last 120 log lines:" >&2
        tail -n 120 "${SERVER_LOG}" >&2 || true
        exit 1
    fi

    if "${PYTHON_BIN}" - "${PORT}" <<'PY' >/dev/null 2>&1
import socket
import sys

port = int(sys.argv[1])
with socket.create_connection(("127.0.0.1", port), timeout=2):
    pass
PY
    then
        echo "Server port ${PORT} is reachable."
        break
    fi
    sleep "${SERVER_POLL_INTERVAL}"
done

if (( SECONDS >= deadline )); then
    echo "Timed out waiting for server port ${PORT}. Last 120 log lines:" >&2
    tail -n 120 "${SERVER_LOG}" >&2 || true
    exit 1
fi

echo "Running LIBERO client smoke..."
"${PYTHON_BIN}" evaluation/libero/client.py \
    --libero-benchmark "${LIBERO_BENCHMARK}" \
    --port "${PORT}" \
    --test-num "${TEST_NUM}" \
    --task-range "${TASK_START}" "${TASK_END}" \
    --out-dir "${OUT_DIR}" 2>&1 | tee "${CLIENT_LOG}"

echo "LIBERO smoke finished. Outputs: ${OUT_DIR}"
