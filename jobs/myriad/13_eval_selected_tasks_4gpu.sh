#!/bin/bash -l

# Four-GPU evaluation for a selected RoboTwin task set.
# Use this after baseline sweep has identified medium/hard tasks.

#$ -S /bin/bash
#$ -N wam_eval4
#$ -cwd
#$ -j y
#$ -o logs/jobs
#$ -l h_rt=48:00:00
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
TEST_NUM="${TEST_NUM:-50}"
SEED="${SEED:-0}"
EVAL_NAME="${EVAL_NAME:-selected_baseline}"
START_PORT="${START_PORT:-29956}"
MASTER_PORT="${MASTER_PORT:-30061}"
RESULTS_ROOT="${RESULTS_ROOT:-${WAM_ROOT}/results_selected_eval/${EVAL_NAME}/${JOB_ID:-manual}}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-1800}"
SELECTED_TASKS="${TASK_NAMES:-}"

if [ -z "${SELECTED_TASKS}" ]; then
    echo "TASK_NAMES is required, for example:" >&2
    echo "  qsub -v TASK_NAMES=\"place_mouse_pad hanging_mug ...\",TEST_NUM=50 jobs/myriad/13_eval_selected_tasks_4gpu.sh" >&2
    exit 1
fi

export NUM_GPUS TEST_NUM SEED EVAL_NAME START_PORT MASTER_PORT RESULTS_ROOT SERVER_WAIT_SECONDS SELECTED_TASKS

print_job_context
echo "EVAL_NAME=${EVAL_NAME}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "TEST_NUM=${TEST_NUM}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "SELECTED_TASKS=${SELECTED_TASKS}"

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

read -r -a all_tasks <<< "${SELECTED_TASKS}"

export NUM_GPUS
export START_PORT
export TEST_NUM
export ROBOTWIN_ROOT
export WAN_VA_ROBOTWIN_ROOT="${ROBOTWIN_ROOT}"
export ROLLOUT_LOG_DIR="${RESULTS_ROOT}/rollouts"

batch_index=0
for ((task_start=0; task_start<${#all_tasks[@]}; task_start+=NUM_GPUS)); do
    batch_tasks=("${all_tasks[@]:task_start:NUM_GPUS}")
    export TASK_NAMES="${batch_tasks[*]}"
    export CLIENT_LOG_DIR="${RESULTS_ROOT}/logs/clients/batch_${batch_index}"

    echo "Running selected batch ${batch_index}: ${TASK_NAMES}"
    bash evaluation/robotwin/launch_client_multigpus.sh "${RESULTS_ROOT}" 0 "${SEED}" "${TEST_NUM}"
    batch_index=$((batch_index + 1))
done

python tools/summarize_robotwin_results.py \
    "${RESULTS_ROOT}" \
    --csv "${RESULTS_ROOT}/summary.csv" \
    --sort rate

echo "Selected-task evaluation complete: ${RESULTS_ROOT}"
CONTAINER
