#!/bin/bash -l

# Four-GPU pilot evaluation. Starts four LingBot-VA servers and runs selected tasks.
# Submit after the one-GPU smoke job passes.

#$ -S /bin/bash
#$ -N wam_pilot4
#$ -cwd
#$ -j y
#$ -o logs/jobs
#$ -l h_rt=12:00:00
#$ -l mem=4G
#$ -pe smp 32
#$ -l tmpfs=200G
#$ -l gpu=4
#$ -ac allow=L

set -euo pipefail

if [ -z "${REPO_ROOT:-}" ]; then
    REPO_ROOT="${SGE_O_WORKDIR:-$(pwd)}"
fi
source "${REPO_ROOT}/jobs/myriad/common.sh"

NUM_GPUS="${NUM_GPUS:-4}"
TASK_NAMES="${TASK_NAMES:-adjust_bottle place_mouse_pad stack_blocks_two click_bell}"
TEST_NUM="${TEST_NUM:-10}"
START_PORT="${START_PORT:-29556}"
MASTER_PORT="${MASTER_PORT:-29661}"
RESULTS_ROOT="${RESULTS_ROOT:-${WAM_ROOT}/results_pilot}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-1800}"

export NUM_GPUS TASK_NAMES TEST_NUM START_PORT MASTER_PORT RESULTS_ROOT SERVER_WAIT_SECONDS

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

SERVER_PIDS=()
cleanup() {
    if [ "${#SERVER_PIDS[@]}" -gt 0 ]; then
        kill "${SERVER_PIDS[@]}" 2>/dev/null || true
        wait "${SERVER_PIDS[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

for ((gpu_id=0; gpu_id<NUM_GPUS; gpu_id++)); do
    current_port=$((START_PORT + gpu_id))
    current_master_port=$((MASTER_PORT + gpu_id))
    echo "Starting server ${gpu_id}: port=${current_port}, master_port=${current_master_port}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python -m torch.distributed.run \
        --nproc_per_node 1 \
        --master_port "${current_master_port}" \
        wan_va/wan_va_server.py \
        --config-name robotwin \
        --port "${current_port}" \
        --save_root "${RESULTS_ROOT}/server_vis" \
        > "${RESULTS_ROOT}/logs/server_${gpu_id}_${current_port}.log" 2>&1 &
    SERVER_PIDS+=("$!")
    sleep 2
done

python - <<'PY'
import os
import sys
import time
import urllib.request

num_gpus = int(os.environ["NUM_GPUS"])
start_port = int(os.environ["START_PORT"])
deadline = time.time() + int(os.environ["SERVER_WAIT_SECONDS"])
ready = set()
while time.time() < deadline:
    for offset in range(num_gpus):
        port = start_port + offset
        if port in ready:
            continue
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
                if response.status == 200:
                    print(f"server ready on port {port}", flush=True)
                    ready.add(port)
        except Exception:
            pass
    if len(ready) == num_gpus:
        sys.exit(0)
    time.sleep(5)
missing = sorted(set(range(start_port, start_port + num_gpus)) - ready)
print(f"servers did not open ports: {missing}", file=sys.stderr)
sys.exit(1)
PY

export NUM_GPUS
export START_PORT
export TEST_NUM
export TASK_NAMES
export ROBOTWIN_ROOT
export WAN_VA_ROBOTWIN_ROOT="${ROBOTWIN_ROOT}"
export ROLLOUT_LOG_DIR="${RESULTS_ROOT}/rollouts"
export CLIENT_LOG_DIR="${RESULTS_ROOT}/logs/clients"

bash evaluation/robotwin/launch_client_multigpus.sh "${RESULTS_ROOT}" 0
CONTAINER
