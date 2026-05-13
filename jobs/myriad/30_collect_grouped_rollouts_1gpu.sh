#!/usr/bin/env bash

# Docker workstation runner for one-GPU grouped rollout collection.
# This does not use a batch scheduler; it runs one Docker container on the
# current workstation and launches both the policy server and RoboTwin clients
# inside that container.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}"
cd "${REPO_ROOT}"

if [ -d "${HOME}/Scratch" ]; then
    DEFAULT_WAM_ROOT="${HOME}/Scratch/wam-rl"
else
    DEFAULT_WAM_ROOT="${HOME}/wam-rl"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
WAM_ROOT="${WAM_ROOT:-${DEFAULT_WAM_ROOT}}"
WAN_VA_MODEL_PATH="${WAN_VA_MODEL_PATH:-${WAM_ROOT}/checkpoints/lingbot-va-posttrain-robotwin}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${WAN_VA_ROBOTWIN_ROOT:-${WAM_ROOT}/RoboTwin}}"
WAN_VA_ROBOTWIN_ROOT="${WAN_VA_ROBOTWIN_ROOT:-${ROBOTWIN_ROOT}}"
WAN_VA_VENV="${WAN_VA_VENV:-${WAM_ROOT}/venvs/wam-rl-container}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
GROUP_SIZE="${GROUP_SIZE:-2}"
GROUPS_PER_TASK="${GROUPS_PER_TASK:-1}"
START_SEED="${START_SEED:-10000}"
PROMPT_INDEX="${PROMPT_INDEX:-0}"
START_PORT="${START_PORT:-30156}"
MASTER_PORT="${MASTER_PORT:-30261}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-1800}"
STRICT_GRPO_CAPTURE="${STRICT_GRPO_CAPTURE:-true}"
STRICT_GRPO_TRANSITION_STD="${STRICT_GRPO_TRANSITION_STD:-0.01}"
ACTION_NUM_INFERENCE_STEPS="${ACTION_NUM_INFERENCE_STEPS:-50}"
RUN_ID="${RUN_ID:-grouped_rollouts_1gpu_$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-${WAM_ROOT}/results_grouped_rollouts/${RUN_ID}}"
SELECTED_TASKS="${TASK_NAMES:-hanging_mug}"
SERVER_LOG="${SERVER_LOG:-${RESULTS_ROOT}/logs/server_0_${START_PORT}.log}"

DOCKER_IMAGE="${DOCKER_IMAGE:-pytorch/pytorch:2.9.0-cuda12.6-cudnn9-devel}"
DOCKER_GPUS="${DOCKER_GPUS:-device=${CUDA_VISIBLE_DEVICES}}"
DOCKER_SHM_SIZE="${DOCKER_SHM_SIZE:-32g}"
DOCKER_RUN_AS_USER="${DOCKER_RUN_AS_USER:-1}"

case "${STRICT_GRPO_CAPTURE}" in
    true|True|1|yes|YES|on|ON) STRICT_GRPO_CAPTURE_PY=True ;;
    *) STRICT_GRPO_CAPTURE_PY=False ;;
esac

if [ "${NUM_GPUS}" -ne 1 ]; then
    echo "This Docker workstation runner supports exactly one GPU. Got NUM_GPUS=${NUM_GPUS}." >&2
    exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is not available on PATH." >&2
    exit 2
fi

if [ ! -d "${WAN_VA_MODEL_PATH}/transformer" ]; then
    echo "WAN_VA_MODEL_PATH must point to a checkpoint with transformer/: ${WAN_VA_MODEL_PATH}" >&2
    exit 2
fi

if [ ! -d "${ROBOTWIN_ROOT}" ]; then
    echo "ROBOTWIN_ROOT does not exist: ${ROBOTWIN_ROOT}" >&2
    exit 2
fi

mkdir -p \
    "${WAM_ROOT}" \
    "${RESULTS_ROOT}/logs" \
    "${RESULTS_ROOT}/server_vis" \
    "${RESULTS_ROOT}/rollouts" \
    "${RESULTS_ROOT}/groups"

export REPO_ROOT WAM_ROOT WAN_VA_MODEL_PATH ROBOTWIN_ROOT WAN_VA_ROBOTWIN_ROOT WAN_VA_VENV
export CUDA_VISIBLE_DEVICES NUM_GPUS GROUP_SIZE GROUPS_PER_TASK START_SEED PROMPT_INDEX
export START_PORT MASTER_PORT SERVER_WAIT_SECONDS RESULTS_ROOT SELECTED_TASKS SERVER_LOG
export STRICT_GRPO_CAPTURE STRICT_GRPO_CAPTURE_PY STRICT_GRPO_TRANSITION_STD
export ACTION_NUM_INFERENCE_STEPS RUN_ID
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "HOST=$(hostname)"
echo "DATE=$(date -Is)"
echo "REPO_ROOT=${REPO_ROOT}"
echo "DOCKER_IMAGE=${DOCKER_IMAGE}"
echo "DOCKER_GPUS=${DOCKER_GPUS}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "WAN_VA_VENV=${WAN_VA_VENV}"
echo "WAN_VA_MODEL_PATH=${WAN_VA_MODEL_PATH}"
echo "ROBOTWIN_ROOT=${ROBOTWIN_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "RUN_ID=${RUN_ID}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "GROUP_SIZE=${GROUP_SIZE}"
echo "GROUPS_PER_TASK=${GROUPS_PER_TASK}"
echo "START_SEED=${START_SEED}"
echo "SELECTED_TASKS=${SELECTED_TASKS}"
echo "STRICT_GRPO_CAPTURE=${STRICT_GRPO_CAPTURE_PY}"
echo "STRICT_GRPO_TRANSITION_STD=${STRICT_GRPO_TRANSITION_STD}"

DOCKER_ARGS=(
    run
    --rm
    --gpus "${DOCKER_GPUS}"
    --network host
    --ipc host
    --shm-size "${DOCKER_SHM_SIZE}"
    -e HOME="${HOME}"
    -e REPO_ROOT
    -e WAM_ROOT
    -e WAN_VA_MODEL_PATH
    -e ROBOTWIN_ROOT
    -e WAN_VA_ROBOTWIN_ROOT
    -e WAN_VA_VENV
    -e CUDA_VISIBLE_DEVICES
    -e NUM_GPUS
    -e GROUP_SIZE
    -e GROUPS_PER_TASK
    -e START_SEED
    -e PROMPT_INDEX
    -e START_PORT
    -e MASTER_PORT
    -e SERVER_WAIT_SECONDS
    -e RESULTS_ROOT
    -e SELECTED_TASKS
    -e SERVER_LOG
    -e STRICT_GRPO_CAPTURE
    -e STRICT_GRPO_CAPTURE_PY
    -e STRICT_GRPO_TRANSITION_STD
    -e ACTION_NUM_INFERENCE_STEPS
    -e RUN_ID
    -e TOKENIZERS_PARALLELISM
    -e PYTORCH_CUDA_ALLOC_CONF
    -e PYTHON_BIN
    -v "${HOME}:${HOME}"
    -v "${REPO_ROOT}:${REPO_ROOT}"
    -v "${WAM_ROOT}:${WAM_ROOT}"
    -v "${ROBOTWIN_ROOT}:${ROBOTWIN_ROOT}"
    -w "${REPO_ROOT}"
)

if [ "${DOCKER_RUN_AS_USER}" = "1" ]; then
    DOCKER_ARGS+=(--user "$(id -u):$(id -g)")
fi

echo "Starting Docker container..."
docker run "${DOCKER_ARGS[@]:1}" "${DOCKER_IMAGE}" bash -lc '
set -euo pipefail

cd "${REPO_ROOT}"

if [ -n "${WAN_VA_VENV:-}" ]; then
    if [ ! -x "${WAN_VA_VENV}/bin/python" ]; then
        echo "WAN_VA_VENV is set but missing python: ${WAN_VA_VENV}" >&2
        exit 2
    fi
    source "${WAN_VA_VENV}/bin/activate"
    PYTHON_BIN="${WAN_VA_VENV}/bin/python"
fi

echo "Container HOST=$(hostname)"
echo "Container DATE=$(date -Is)"
echo "Container PYTHON_BIN=${PYTHON_BIN}"

"${PYTHON_BIN}" - <<'"'"'PY'"'"'
import importlib
import sys

required = [
    ("torch", "torch"),
    ("websockets", "websockets"),
    ("cv2", "opencv-python"),
    ("imageio", "imageio"),
    ("yaml", "PyYAML"),
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

"${PYTHON_BIN}" tools/set_attn_mode.py "${WAN_VA_MODEL_PATH}" torch
"${PYTHON_BIN}" tools/check_setup.py \
    --model-path "${WAN_VA_MODEL_PATH}" \
    --robotwin-root "${ROBOTWIN_ROOT}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

SERVER_PID=""
cleanup() {
    if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Stopping server pid ${SERVER_PID}"
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "Starting server: port=${START_PORT}, master_port=${MASTER_PORT}"
"${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port "${MASTER_PORT}" \
    wan_va/wan_va_server.py \
    --config-name robotwin \
    --port "${START_PORT}" \
    --save_root "${RESULTS_ROOT}/server_vis" \
    --opts \
    strict_grpo_capture="${STRICT_GRPO_CAPTURE_PY}" \
    strict_grpo_transition_std="${STRICT_GRPO_TRANSITION_STD}" \
    action_num_inference_steps="${ACTION_NUM_INFERENCE_STEPS}" \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

"${PYTHON_BIN}" - <<'"'"'PY'"'"'
import os
import sys
import time
import urllib.request

port = int(os.environ["START_PORT"])
deadline = time.time() + int(os.environ["SERVER_WAIT_SECONDS"])
while time.time() < deadline:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
            if response.status == 200:
                print(f"server ready on port {port}", flush=True)
                sys.exit(0)
    except Exception:
        pass
    time.sleep(5)
print(f"server did not open port {port}", file=sys.stderr)
sys.exit(1)
PY

read -r -a all_tasks <<< "${SELECTED_TASKS}"

export START_PORT
export WAN_VA_ROBOTWIN_ROOT="${ROBOTWIN_ROOT}"
export ROLLOUT_LOG_DIR="${RESULTS_ROOT}/rollouts"
export POLICY_CHECKPOINT="${WAN_VA_MODEL_PATH}"
export REFERENCE_CHECKPOINT="${WAN_VA_MODEL_PATH}"

for ((group_index=0; group_index<GROUPS_PER_TASK; group_index++)); do
    seed=$((START_SEED + group_index))
    export GROUP_INDEX="${group_index}"
    for ((sample_idx=0; sample_idx<GROUP_SIZE; sample_idx++)); do
        export SAMPLE_IDX="${sample_idx}"
        export SAMPLING_SEED=$((START_SEED * 1000000 + group_index * GROUP_SIZE + sample_idx))

        for ((task_index=0; task_index<${#all_tasks[@]}; task_index++)); do
            batch_tasks=("${all_tasks[@]:task_index:1}")
            export TASK_NAMES="${batch_tasks[*]}"
            export CLIENT_LOG_DIR="${RESULTS_ROOT}/logs/clients/group_${group_index}/sample_${sample_idx}/task_${task_index}"

            echo "Running group=${group_index} sample=${sample_idx} seed=${seed} tasks=${TASK_NAMES}"
            bash evaluation/robotwin/launch_client_multigpus.sh "${RESULTS_ROOT}" 0 "${seed}" 1
        done
    done
done

"${PYTHON_BIN}" tools/collect_robotwin_rollouts.py \
    "${RESULTS_ROOT}" \
    --out-jsonl "${RESULTS_ROOT}/groups/rollouts_flat.jsonl" \
    --out-csv "${RESULTS_ROOT}/groups/rollouts_flat.csv"

"${PYTHON_BIN}" tools/build_grpo_groups.py \
    "${RESULTS_ROOT}" \
    --expected-group-size "${GROUP_SIZE}" \
    --require-strict-artifacts \
    --require-existing-artifacts \
    --wait-for-artifacts-seconds 120 \
    --fail-on-validation-errors \
    --out-jsonl "${RESULTS_ROOT}/groups/grpo_groups.jsonl" \
    --out-summary "${RESULTS_ROOT}/groups/grpo_summary.json" \
    --out-manifest "${RESULTS_ROOT}/groups/grpo_manifest.json"

"${PYTHON_BIN}" tools/validate_grpo_dataset.py \
    "${RESULTS_ROOT}/groups/grpo_groups.jsonl" \
    --out-summary "${RESULTS_ROOT}/groups/grpo_dataset_validation.json" \
    --fail-on-error

echo "Rollout JSON count:"
find "${RESULTS_ROOT}/rollouts" -type f -name "*.json" | wc -l
echo "Strict GRPO artifact count:"
find "${RESULTS_ROOT}/server_vis" -type f -name "strict_grpo_*.pt" | wc -l
echo "Grouped rollout collection complete: ${RESULTS_ROOT}"
'
