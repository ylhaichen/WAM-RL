#!/bin/bash

# Submit matched one-GPU baseline and actor RoboTwin eval smoke jobs.
# The two jobs use the same task/env/prompt/sampling controls and separate ports.

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

WAM_ROOT="${WAM_ROOT:-/home/zcably0/Scratch/wam-rl}"
RUN_ID="${RUN_ID:-actor_eval_pair_$(date +%Y%m%d_%H%M%S)}"
TASK_NAME="${TASK_NAME:-move_stapler_pad}"
TEST_NUM="${TEST_NUM:-2}"
ACTION_NUM_INFERENCE_STEPS="${ACTION_NUM_INFERENCE_STEPS:-10}"
PROMPT_INDEX="${PROMPT_INDEX:-0}"
SAMPLING_SEED="${SAMPLING_SEED:-12345}"
SAMPLING_SEED_PER_ENV="${SAMPLING_SEED_PER_ENV:-true}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-1800}"
SAVE_SERVER_DEBUG_TENSORS="${SAVE_SERVER_DEBUG_TENSORS:-false}"
SEED="${SEED:-0}"

BASELINE_JOB_NAME="${BASELINE_JOB_NAME:-wam_eval_base_pair}"
ACTOR_JOB_NAME="${ACTOR_JOB_NAME:-wam_eval_actor_pair}"
BASELINE_RESULTS_ROOT="${BASELINE_RESULTS_ROOT:-${WAM_ROOT}/results_actor_eval/baseline_${TASK_NAME}_${RUN_ID}}"
ACTOR_RESULTS_ROOT="${ACTOR_RESULTS_ROOT:-${WAM_ROOT}/results_actor_eval/actor_${TASK_NAME}_${RUN_ID}}"
COMPARE_ROOT="${COMPARE_ROOT:-${WAM_ROOT}/results_actor_eval/${RUN_ID}_comparison}"
BASELINE_REPEATABILITY_JSON="${BASELINE_REPEATABILITY_JSON:-}"

BASELINE_PORT="${BASELINE_PORT:-29656}"
BASELINE_MASTER_PORT="${BASELINE_MASTER_PORT:-29661}"
ACTOR_PORT="${ACTOR_PORT:-29756}"
ACTOR_MASTER_PORT="${ACTOR_MASTER_PORT:-29761}"
ACTOR_REPLAY_CHECKPOINT_PATH="${ACTOR_REPLAY_CHECKPOINT_PATH:-}"

QSUB_H_RT="${QSUB_H_RT:-2:00:00}"
QSUB_MEM="${QSUB_MEM:-4G}"
QSUB_SLOTS="${QSUB_SLOTS:-8}"
QSUB_TMPFS="${QSUB_TMPFS:-80G}"
QSUB_GPU="${QSUB_GPU:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [ -z "${ACTOR_REPLAY_CHECKPOINT_PATH}" ]; then
    echo "Set ACTOR_REPLAY_CHECKPOINT_PATH before submitting actor eval pair smoke." >&2
    exit 2
fi
if [ "${BASELINE_PORT}" = "${ACTOR_PORT}" ]; then
    echo "BASELINE_PORT and ACTOR_PORT must differ." >&2
    exit 2
fi
if [ "${BASELINE_MASTER_PORT}" = "${ACTOR_MASTER_PORT}" ]; then
    echo "BASELINE_MASTER_PORT and ACTOR_MASTER_PORT must differ." >&2
    exit 2
fi

JOB_SCRIPT="${REPO_ROOT}/jobs/myriad/10_eval_smoke_1gpu.sh"
if [ ! -f "${JOB_SCRIPT}" ]; then
    echo "Missing one-GPU eval job script: ${JOB_SCRIPT}" >&2
    exit 2
fi

QSUB_ARGS=(
    -V
    -l "h_rt=${QSUB_H_RT}"
    -l "mem=${QSUB_MEM}"
    -pe smp "${QSUB_SLOTS}"
    -l "tmpfs=${QSUB_TMPFS}"
)
if [ -n "${QSUB_GPU}" ] && [ "${QSUB_GPU}" != "0" ]; then
    QSUB_ARGS+=(-l "gpu=${QSUB_GPU}")
fi

COMMON_VARS=(
    "REPO_ROOT=${REPO_ROOT}"
    "TASK_NAME=${TASK_NAME}"
    "TEST_NUM=${TEST_NUM}"
    "ACTION_NUM_INFERENCE_STEPS=${ACTION_NUM_INFERENCE_STEPS}"
    "PROMPT_INDEX=${PROMPT_INDEX}"
    "SAMPLING_SEED=${SAMPLING_SEED}"
    "SAMPLING_SEED_PER_ENV=${SAMPLING_SEED_PER_ENV}"
    "SERVER_WAIT_SECONDS=${SERVER_WAIT_SECONDS}"
    "SAVE_SERVER_DEBUG_TENSORS=${SAVE_SERVER_DEBUG_TENSORS}"
    "SEED=${SEED}"
)

qsub_eval() {
    local job_name="$1"
    local results_root="$2"
    local port="$3"
    local master_port="$4"
    local checkpoint="$5"
    local vars=(
        "${COMMON_VARS[@]}"
        "RESULTS_ROOT=${results_root}"
        "PORT=${port}"
        "MASTER_PORT=${master_port}"
        "ACTOR_REPLAY_CHECKPOINT_PATH=${checkpoint}"
    )
    local cmd=(qsub "${QSUB_ARGS[@]}" -N "${job_name}")
    for value in "${vars[@]}"; do
        cmd+=(-v "${value}")
    done
    cmd+=("${JOB_SCRIPT}")

    if [ "${DRY_RUN}" = "1" ]; then
        printf '%q' "${cmd[0]}"
        printf ' %q' "${cmd[@]:1}"
        printf '\n'
    else
        "${cmd[@]}"
    fi
}

echo "Submitting matched actor eval pair smoke"
echo "  RUN_ID=${RUN_ID}"
echo "  TASK_NAME=${TASK_NAME}"
echo "  TEST_NUM=${TEST_NUM}"
echo "  ACTION_NUM_INFERENCE_STEPS=${ACTION_NUM_INFERENCE_STEPS}"
echo "  PROMPT_INDEX=${PROMPT_INDEX}"
echo "  SAMPLING_SEED=${SAMPLING_SEED}"
echo "  SAMPLING_SEED_PER_ENV=${SAMPLING_SEED_PER_ENV}"
echo "  SAVE_SERVER_DEBUG_TENSORS=${SAVE_SERVER_DEBUG_TENSORS}"
echo "  SEED=${SEED}"
echo "  BASELINE_RESULTS_ROOT=${BASELINE_RESULTS_ROOT}"
echo "  ACTOR_RESULTS_ROOT=${ACTOR_RESULTS_ROOT}"
echo "  COMPARE_ROOT=${COMPARE_ROOT}"
echo "  BASELINE_REPEATABILITY_JSON=${BASELINE_REPEATABILITY_JSON}"
echo "  ACTOR_REPLAY_CHECKPOINT_PATH=${ACTOR_REPLAY_CHECKPOINT_PATH}"

qsub_eval "${BASELINE_JOB_NAME}" "${BASELINE_RESULTS_ROOT}" "${BASELINE_PORT}" "${BASELINE_MASTER_PORT}" ""
qsub_eval "${ACTOR_JOB_NAME}" "${ACTOR_RESULTS_ROOT}" "${ACTOR_PORT}" "${ACTOR_MASTER_PORT}" "${ACTOR_REPLAY_CHECKPOINT_PATH}"

cat <<EOF

After both jobs finish, summarize and compare with:

python tools/summarize_actor_eval_pair.py \\
  --baseline "${BASELINE_RESULTS_ROOT}" \\
  --actor "${ACTOR_RESULTS_ROOT}" \\
  --out-root "${COMPARE_ROOT}"
EOF

if [ -n "${BASELINE_REPEATABILITY_JSON}" ]; then
    cat <<EOF

Then gate candidate promotion with:

python tools/gate_actor_eval_promotion.py \\
  --comparison "${COMPARE_ROOT}/comparison.json" \\
  --baseline-repeatability "${BASELINE_REPEATABILITY_JSON}" \\
  --out-json "${COMPARE_ROOT}/promotion_gate.json" \\
  --out-markdown "${COMPARE_ROOT}/promotion_gate.md"
EOF
fi
