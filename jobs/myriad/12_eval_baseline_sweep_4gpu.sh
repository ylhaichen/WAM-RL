#!/bin/bash -l

# Four-GPU baseline sweep for selecting RoboTwin tasks.
# Runs tasks in batches of NUM_GPUS so each client has a dedicated server/GPU.

#$ -S /bin/bash
#$ -N wam_sweep4
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

MYRIAD_JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${REPO_ROOT:-}" ]; then
    if [ -n "${SGE_O_WORKDIR:-}" ] && [ -f "${SGE_O_WORKDIR}/jobs/myriad/common.sh" ]; then
        REPO_ROOT="${SGE_O_WORKDIR}"
    elif [ -n "${SGE_CWD_PATH:-}" ] && [ -f "${SGE_CWD_PATH}/jobs/myriad/common.sh" ]; then
        REPO_ROOT="${SGE_CWD_PATH}"
    elif [ -f "${PWD}/jobs/myriad/common.sh" ]; then
        REPO_ROOT="${PWD}"
    else
        REPO_ROOT="$(cd "${MYRIAD_JOB_DIR}/../.." && pwd)"
    fi
fi
source "${REPO_ROOT}/jobs/myriad/common.sh"

NUM_GPUS="${NUM_GPUS:-4}"
TEST_NUM="${TEST_NUM:-10}"
SEED="${SEED:-0}"
START_PORT="${START_PORT:-29756}"
MASTER_PORT="${MASTER_PORT:-29861}"
RESULTS_ROOT="${RESULTS_ROOT:-${WAM_ROOT}/results_baseline_sweep/${JOB_ID:-manual}}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-1800}"
BATCH_OFFSET="${BATCH_OFFSET:-0}"
MAX_BATCHES="${MAX_BATCHES:-0}"
SWEEP_TASKS="${SWEEP_TASKS:-${TASK_NAMES:-stack_bowls_three handover_block hanging_mug scan_object lift_pot put_object_cabinet stack_blocks_three place_shoe adjust_bottle place_mouse_pad dump_bin_bigbin move_pillbottle_pad pick_dual_bottles shake_bottle place_fan turn_switch shake_bottle_horizontally place_container_plate rotate_qrcode place_object_stand put_bottles_dustbin move_stapler_pad place_burger_fries place_bread_basket pick_diverse_bottles open_microwave beat_block_hammer press_stapler click_bell move_playingcard_away open_laptop move_can_pot stack_bowls_two place_a2b_right stamp_seal place_object_basket handover_mic place_bread_skillet stack_blocks_two place_cans_plasticbox click_alarmclock blocks_ranking_size place_phone_stand place_can_basket place_object_scale place_a2b_left grab_roller place_dual_shoes place_empty_cup blocks_ranking_rgb}}"

export NUM_GPUS TEST_NUM SEED START_PORT MASTER_PORT RESULTS_ROOT SERVER_WAIT_SECONDS
export BATCH_OFFSET MAX_BATCHES SWEEP_TASKS

print_job_context
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "TEST_NUM=${TEST_NUM}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "BATCH_OFFSET=${BATCH_OFFSET}"
echo "MAX_BATCHES=${MAX_BATCHES}"
echo "SWEEP_TASKS=${SWEEP_TASKS}"

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

read -r -a all_tasks <<< "${SWEEP_TASKS}"
if [ "${#all_tasks[@]}" -eq 0 ]; then
    echo "No tasks configured for sweep." >&2
    exit 1
fi

export NUM_GPUS
export START_PORT
export TEST_NUM
export ROBOTWIN_ROOT
export WAN_VA_ROBOTWIN_ROOT="${ROBOTWIN_ROOT}"
export ROLLOUT_LOG_DIR="${RESULTS_ROOT}/rollouts"

batch_index=0
ran_batches=0
for ((task_start=0; task_start<${#all_tasks[@]}; task_start+=NUM_GPUS)); do
    if [ "${batch_index}" -lt "${BATCH_OFFSET}" ]; then
        batch_index=$((batch_index + 1))
        continue
    fi
    if [ "${MAX_BATCHES}" -gt 0 ] && [ "${ran_batches}" -ge "${MAX_BATCHES}" ]; then
        break
    fi

    batch_tasks=("${all_tasks[@]:task_start:NUM_GPUS}")
    export TASK_NAMES="${batch_tasks[*]}"
    export CLIENT_LOG_DIR="${RESULTS_ROOT}/logs/clients/batch_${batch_index}"

    echo "Running batch ${batch_index}: ${TASK_NAMES}"
    bash evaluation/robotwin/launch_client_multigpus.sh "${RESULTS_ROOT}" 0 "${SEED}" "${TEST_NUM}"

    batch_index=$((batch_index + 1))
    ran_batches=$((ran_batches + 1))
done

python tools/summarize_robotwin_results.py \
    "${RESULTS_ROOT}" \
    --csv "${RESULTS_ROOT}/summary.csv" \
    --sort rate

echo "Baseline sweep complete: ${RESULTS_ROOT}"
CONTAINER
