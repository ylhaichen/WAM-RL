#!/bin/bash -l

# One-GPU smoke evaluation. Starts one LingBot-VA server and runs one RoboTwin task.
# Submit after assets and RoboTwin are installed.

#$ -S /bin/bash
#$ -N wam_smoke
#$ -cwd
#$ -j y
#$ -o logs/jobs
#$ -l h_rt=4:00:00
#$ -l mem=4G
#$ -pe smp 16
#$ -l tmpfs=100G
#$ -l gpu=1
#$ -ac allow=L

set -euo pipefail

MYRIAD_JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${REPO_ROOT:-}" ]; then
    REPO_ROOT="$(cd "${MYRIAD_JOB_DIR}/../.." && pwd)"
fi
source "${REPO_ROOT}/jobs/myriad/common.sh"

TASK_NAME="${TASK_NAME:-adjust_bottle}"
TEST_NUM="${TEST_NUM:-3}"
PORT="${PORT:-29056}"
MASTER_PORT="${MASTER_PORT:-29061}"
RESULTS_ROOT="${RESULTS_ROOT:-${WAM_ROOT}/results_smoke}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-1800}"

export TASK_NAME TEST_NUM PORT MASTER_PORT RESULTS_ROOT SERVER_WAIT_SECONDS

print_job_context

container_exec_gpu <<'CONTAINER'
set -euo pipefail

if [ -d "${WAN_VA_CONDA_LIBS}/lib" ]; then
    export LD_LIBRARY_PATH="${WAN_VA_CONDA_LIBS}/lib:${LD_LIBRARY_PATH:-}"
fi

cd "${REPO_ROOT}"

if [ ! -x "${WAN_VA_VENV}/bin/python" ]; then
    echo "Missing venv: ${WAN_VA_VENV}" >&2
    echo "Run jobs/myriad/00_install_container_env.sh first." >&2
    exit 1
fi

source "${WAN_VA_VENV}/bin/activate"

mkdir -p "${RESULTS_ROOT}/logs" "${RESULTS_ROOT}/server_vis" "${RESULTS_ROOT}/rollouts"

python tools/set_attn_mode.py "${WAN_VA_MODEL_PATH}" torch
python tools/check_setup.py \
    --model-path "${WAN_VA_MODEL_PATH}" \
    --robotwin-root "${ROBOTWIN_ROOT}"

nvidia-smi

SERVER_PID=""
cleanup() {
    if [ -n "${SERVER_PID}" ]; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port "${MASTER_PORT}" \
    wan_va/wan_va_server.py \
    --config-name robotwin \
    --port "${PORT}" \
    --save_root "${RESULTS_ROOT}/server_vis" \
    > "${RESULTS_ROOT}/logs/server_${PORT}.log" 2>&1 &
SERVER_PID=$!

python - <<'PY'
import os
import sys
import time
import urllib.request

port = int(os.environ["PORT"])
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

export PORT
export TEST_NUM
export ROBOTWIN_ROOT
export WAN_VA_ROBOTWIN_ROOT="${ROBOTWIN_ROOT}"
export ROLLOUT_LOG_DIR="${RESULTS_ROOT}/rollouts"

CUDA_VISIBLE_DEVICES=0 bash evaluation/robotwin/launch_client.sh \
    "${RESULTS_ROOT}" \
    "${TASK_NAME}"
CONTAINER
