#!/bin/bash -l

# Four-GPU grouped rollout collection for pseudo-GRPO training while saving
# full denoising strict-GRPO artifacts. This is a data collection job, not a
# baseline evaluation job.

#$ -S /bin/bash
#$ -N wam_grpo_rollouts
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
    REPO_ROOT="$(cd "${MYRIAD_JOB_DIR}/../.." && pwd)"
fi
source "${REPO_ROOT}/jobs/myriad/common.sh"

NUM_GPUS="${NUM_GPUS:-4}"
GROUP_SIZE="${GROUP_SIZE:-4}"
GROUPS_PER_TASK="${GROUPS_PER_TASK:-5}"
START_SEED="${START_SEED:-10000}"
PROMPT_INDEX="${PROMPT_INDEX:-0}"
START_PORT="${START_PORT:-30156}"
MASTER_PORT="${MASTER_PORT:-30261}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-1800}"
STRICT_GRPO_CAPTURE="${STRICT_GRPO_CAPTURE:-true}"
STRICT_GRPO_TRANSITION_STD="${STRICT_GRPO_TRANSITION_STD:-0.01}"
STRICT_GRPO_CAPTURE_SCOPE="${STRICT_GRPO_CAPTURE_SCOPE:-action_denoising_trajectory}"
ACTION_NUM_INFERENCE_STEPS="${ACTION_NUM_INFERENCE_STEPS:-50}"
GROUP_SEED_SEARCH="${GROUP_SEED_SEARCH:-true}"
GROUP_SEED_SEARCH_MAX_ATTEMPTS="${GROUP_SEED_SEARCH_MAX_ATTEMPTS:-50}"
GROUP_RETRY_MULTIPLIER="${GROUP_RETRY_MULTIPLIER:-4}"
GROUP_MAX_ATTEMPTS="${GROUP_MAX_ATTEMPTS:-$((GROUPS_PER_TASK * GROUP_RETRY_MULTIPLIER))}"
RUN_ID="${RUN_ID:-grouped_rollouts_${JOB_ID:-manual}}"
RESULTS_ROOT="${RESULTS_ROOT:-${WAM_ROOT}/results_grouped_rollouts/${RUN_ID}}"
STABLE_SEED_CACHE_DIR="${STABLE_SEED_CACHE_DIR:-${RESULTS_ROOT}/groups/stable_seeds}"
SELECTED_TASKS="${TASK_NAMES:-hanging_mug open_microwave turn_switch move_stapler_pad}"

case "${STRICT_GRPO_CAPTURE}" in
    true|True|1|yes|YES|on|ON) STRICT_GRPO_CAPTURE_PY=True ;;
    *) STRICT_GRPO_CAPTURE_PY=False ;;
esac

export NUM_GPUS GROUP_SIZE GROUPS_PER_TASK START_SEED PROMPT_INDEX
export START_PORT MASTER_PORT SERVER_WAIT_SECONDS RESULTS_ROOT SELECTED_TASKS
export STRICT_GRPO_CAPTURE STRICT_GRPO_CAPTURE_PY STRICT_GRPO_TRANSITION_STD STRICT_GRPO_CAPTURE_SCOPE
export ACTION_NUM_INFERENCE_STEPS GROUP_SEED_SEARCH GROUP_SEED_SEARCH_MAX_ATTEMPTS
export GROUP_RETRY_MULTIPLIER GROUP_MAX_ATTEMPTS
export RUN_ID STABLE_SEED_CACHE_DIR

print_job_context
echo "RUN_ID=${RUN_ID}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "GROUP_SIZE=${GROUP_SIZE}"
echo "GROUPS_PER_TASK=${GROUPS_PER_TASK}"
echo "START_SEED=${START_SEED}"
echo "PROMPT_INDEX=${PROMPT_INDEX}"
echo "SELECTED_TASKS=${SELECTED_TASKS}"
echo "STRICT_GRPO_CAPTURE=${STRICT_GRPO_CAPTURE_PY}"
echo "STRICT_GRPO_TRANSITION_STD=${STRICT_GRPO_TRANSITION_STD}"
echo "STRICT_GRPO_CAPTURE_SCOPE=${STRICT_GRPO_CAPTURE_SCOPE}"
echo "GROUP_SEED_SEARCH=${GROUP_SEED_SEARCH}"
echo "GROUP_SEED_SEARCH_MAX_ATTEMPTS=${GROUP_SEED_SEARCH_MAX_ATTEMPTS}"
echo "GROUP_MAX_ATTEMPTS=${GROUP_MAX_ATTEMPTS}"
echo "STABLE_SEED_CACHE_DIR=${STABLE_SEED_CACHE_DIR}"

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

mkdir -p \
    "${RESULTS_ROOT}/logs" \
    "${RESULTS_ROOT}/server_vis" \
    "${RESULTS_ROOT}/rollouts" \
    "${RESULTS_ROOT}/groups"

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
        --opts \
        strict_grpo_capture="${STRICT_GRPO_CAPTURE_PY}" \
        strict_grpo_transition_std="${STRICT_GRPO_TRANSITION_STD}" \
        strict_grpo_capture_scope="${STRICT_GRPO_CAPTURE_SCOPE}" \
        action_num_inference_steps="${ACTION_NUM_INFERENCE_STEPS}" \
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
export ROBOTWIN_ROOT
export WAN_VA_ROBOTWIN_ROOT="${ROBOTWIN_ROOT}"
export ROLLOUT_LOG_DIR="${RESULTS_ROOT}/rollouts"
export RUN_ID
export POLICY_CHECKPOINT="${WAN_VA_MODEL_PATH}"
export REFERENCE_CHECKPOINT="${WAN_VA_MODEL_PATH}"
export GROUP_SIZE
export PROMPT_INDEX
export ACTION_NUM_INFERENCE_STEPS
export GROUP_SEED_SEARCH
export GROUP_SEED_SEARCH_MAX_ATTEMPTS
export STABLE_SEED_CACHE_DIR

SUCCESSFUL_ROOTS_FILE="${RESULTS_ROOT}/groups/successful_attempt_roots.txt"
FAILED_ROOTS_FILE="${RESULTS_ROOT}/groups/failed_attempt_roots.txt"
: > "${SUCCESSFUL_ROOTS_FILE}"
: > "${FAILED_ROOTS_FILE}"

completed_groups=0
attempt_index=0
while (( completed_groups < GROUPS_PER_TASK && attempt_index < GROUP_MAX_ATTEMPTS )); do
    seed=$((START_SEED + attempt_index))
    group_index=${attempt_index}
    attempt_root="${RESULTS_ROOT}/attempts/group_${completed_groups}/attempt_${attempt_index}_seed_${seed}"
    mkdir -p "${attempt_root}/rollouts" "${attempt_root}/logs/clients"

    export GROUP_INDEX="${group_index}"
    export ROLLOUT_LOG_DIR="${attempt_root}/rollouts"

    echo "Starting group attempt logical_group=${completed_groups} physical_group=${group_index} seed=${seed} root=${attempt_root}"
    group_failed=0
    for ((sample_idx=0; sample_idx<GROUP_SIZE; sample_idx++)); do
        export SAMPLE_IDX="${sample_idx}"
        export SAMPLING_SEED=$((START_SEED * 1000000 + group_index * GROUP_SIZE + sample_idx))

        batch_index=0
        for ((task_start=0; task_start<${#all_tasks[@]}; task_start+=NUM_GPUS)); do
            batch_tasks=("${all_tasks[@]:task_start:NUM_GPUS}")
            export TASK_NAMES="${batch_tasks[*]}"
            export CLIENT_LOG_DIR="${attempt_root}/logs/clients/sample_${sample_idx}/batch_${batch_index}"

            echo "Running logical_group=${completed_groups} physical_group=${group_index} sample=${sample_idx} seed=${seed} tasks=${TASK_NAMES}"
            if ! bash evaluation/robotwin/launch_client_multigpus.sh "${attempt_root}" 0 "${seed}" 1; then
                group_failed=1
                break
            fi
            batch_index=$((batch_index + 1))
        done
        if (( group_failed != 0 )); then
            break
        fi
    done

    if (( group_failed != 0 )); then
        echo "Discarding failed group attempt logical_group=${completed_groups} physical_group=${group_index} seed=${seed}; logs remain under ${attempt_root}" >&2
        echo "${attempt_root}" >> "${FAILED_ROOTS_FILE}"
        attempt_index=$((attempt_index + 1))
        continue
    fi

    echo "Accepted group attempt logical_group=${completed_groups} physical_group=${group_index} seed=${seed}"
    echo "${attempt_root}" >> "${SUCCESSFUL_ROOTS_FILE}"
    completed_groups=$((completed_groups + 1))
    attempt_index=$((attempt_index + 1))
done

if (( completed_groups < GROUPS_PER_TASK )); then
    echo "Only completed ${completed_groups}/${GROUPS_PER_TASK} grouped rollout attempts after ${attempt_index}/${GROUP_MAX_ATTEMPTS} attempts." >&2
    echo "Failed roots are listed in ${FAILED_ROOTS_FILE}" >&2
    exit 1
fi

mapfile -t SUCCESSFUL_ATTEMPT_ROOTS < "${SUCCESSFUL_ROOTS_FILE}"

python tools/collect_robotwin_rollouts.py \
    "${SUCCESSFUL_ATTEMPT_ROOTS[@]}" \
    --out-jsonl "${RESULTS_ROOT}/groups/rollouts_flat.jsonl" \
    --out-csv "${RESULTS_ROOT}/groups/rollouts_flat.csv"

python tools/build_grpo_groups.py \
    "${SUCCESSFUL_ATTEMPT_ROOTS[@]}" \
    --canonicalize-legacy-group-ids \
    --expected-group-size "${GROUP_SIZE}" \
    --require-strict-artifacts \
    --require-existing-artifacts \
    --wait-for-artifacts-seconds 120 \
    --fail-on-validation-errors \
    --out-jsonl "${RESULTS_ROOT}/groups/grpo_groups.jsonl" \
    --out-summary "${RESULTS_ROOT}/groups/grpo_summary.json" \
    --out-manifest "${RESULTS_ROOT}/groups/grpo_manifest.json"

python tools/validate_grpo_dataset.py \
    "${RESULTS_ROOT}/groups/grpo_groups.jsonl" \
    --inspect-artifacts \
    --out-summary "${RESULTS_ROOT}/groups/grpo_dataset_validation.json" \
    --fail-on-error

echo "Rollout JSON count:"
for root in "${SUCCESSFUL_ATTEMPT_ROOTS[@]}"; do
    find "${root}/rollouts" -type f -name "*.json"
done | wc -l
echo "Strict GRPO artifact count:"
find "${RESULTS_ROOT}/server_vis" -type f -name "strict_grpo_*.pt" | wc -l
echo "Grouped rollout collection complete: ${RESULTS_ROOT}"
CONTAINER
