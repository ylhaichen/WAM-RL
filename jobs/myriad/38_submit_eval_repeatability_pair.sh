#!/bin/bash

# Submit two matched one-GPU baseline RoboTwin eval jobs for repeatability checks.
# The jobs share task/env/prompt/sampling controls and use separate ports.

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
RUN_ID="${RUN_ID:-eval_repeatability_pair_$(date +%Y%m%d_%H%M%S)}"
TASK_NAME="${TASK_NAME:-move_stapler_pad}"
TEST_NUM="${TEST_NUM:-10}"
ACTION_NUM_INFERENCE_STEPS="${ACTION_NUM_INFERENCE_STEPS:-10}"
PROMPT_INDEX="${PROMPT_INDEX:-0}"
SAMPLING_SEED="${SAMPLING_SEED:-12345}"
SAMPLING_SEED_PER_ENV="${SAMPLING_SEED_PER_ENV:-true}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-1800}"
SAVE_SERVER_DEBUG_TENSORS="${SAVE_SERVER_DEBUG_TENSORS:-false}"
SEED="${SEED:-0}"

RUN_A_LABEL="${RUN_A_LABEL:-baseline_a}"
RUN_B_LABEL="${RUN_B_LABEL:-baseline_b}"
RUN_A_JOB_NAME="${RUN_A_JOB_NAME:-wam_eval_rep_a}"
RUN_B_JOB_NAME="${RUN_B_JOB_NAME:-wam_eval_rep_b}"
RUN_A_RESULTS_ROOT="${RUN_A_RESULTS_ROOT:-${WAM_ROOT}/results_actor_eval/${RUN_A_LABEL}_${TASK_NAME}_${RUN_ID}}"
RUN_B_RESULTS_ROOT="${RUN_B_RESULTS_ROOT:-${WAM_ROOT}/results_actor_eval/${RUN_B_LABEL}_${TASK_NAME}_${RUN_ID}}"
REPEATABILITY_ROOT="${REPEATABILITY_ROOT:-${WAM_ROOT}/results_actor_eval/${RUN_ID}_repeatability}"

RUN_A_PORT="${RUN_A_PORT:-29856}"
RUN_A_MASTER_PORT="${RUN_A_MASTER_PORT:-29861}"
RUN_B_PORT="${RUN_B_PORT:-29956}"
RUN_B_MASTER_PORT="${RUN_B_MASTER_PORT:-29961}"

QSUB_H_RT="${QSUB_H_RT:-3:00:00}"
QSUB_MEM="${QSUB_MEM:-4G}"
QSUB_SLOTS="${QSUB_SLOTS:-8}"
QSUB_TMPFS="${QSUB_TMPFS:-80G}"
QSUB_GPU="${QSUB_GPU:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [ "${RUN_A_PORT}" = "${RUN_B_PORT}" ]; then
    echo "RUN_A_PORT and RUN_B_PORT must differ." >&2
    exit 2
fi
if [ "${RUN_A_MASTER_PORT}" = "${RUN_B_MASTER_PORT}" ]; then
    echo "RUN_A_MASTER_PORT and RUN_B_MASTER_PORT must differ." >&2
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
    "ACTOR_REPLAY_CHECKPOINT_PATH="
)

qsub_eval() {
    local job_name="$1"
    local results_root="$2"
    local port="$3"
    local master_port="$4"
    local vars=(
        "${COMMON_VARS[@]}"
        "RESULTS_ROOT=${results_root}"
        "PORT=${port}"
        "MASTER_PORT=${master_port}"
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

echo "Submitting matched baseline eval repeatability pair"
echo "  RUN_ID=${RUN_ID}"
echo "  TASK_NAME=${TASK_NAME}"
echo "  TEST_NUM=${TEST_NUM}"
echo "  ACTION_NUM_INFERENCE_STEPS=${ACTION_NUM_INFERENCE_STEPS}"
echo "  PROMPT_INDEX=${PROMPT_INDEX}"
echo "  SAMPLING_SEED=${SAMPLING_SEED}"
echo "  SAMPLING_SEED_PER_ENV=${SAMPLING_SEED_PER_ENV}"
echo "  SAVE_SERVER_DEBUG_TENSORS=${SAVE_SERVER_DEBUG_TENSORS}"
echo "  SEED=${SEED}"
echo "  RUN_A_RESULTS_ROOT=${RUN_A_RESULTS_ROOT}"
echo "  RUN_B_RESULTS_ROOT=${RUN_B_RESULTS_ROOT}"
echo "  REPEATABILITY_ROOT=${REPEATABILITY_ROOT}"

qsub_eval "${RUN_A_JOB_NAME}" "${RUN_A_RESULTS_ROOT}" "${RUN_A_PORT}" "${RUN_A_MASTER_PORT}"
qsub_eval "${RUN_B_JOB_NAME}" "${RUN_B_RESULTS_ROOT}" "${RUN_B_PORT}" "${RUN_B_MASTER_PORT}"

cat <<EOF

After both jobs finish, summarize repeatability with:

python tools/summarize_robotwin_repeatability.py \\
  --run "${RUN_A_LABEL}=${RUN_A_RESULTS_ROOT}" \\
  --run "${RUN_B_LABEL}=${RUN_B_RESULTS_ROOT}" \\
  --out-json "${REPEATABILITY_ROOT}/repeatability.json" \\
  --out-csv "${REPEATABILITY_ROOT}/repeatability.csv" \\
  --out-markdown "${REPEATABILITY_ROOT}/repeatability.md"

Use that repeatability JSON with:

python tools/gate_actor_eval_promotion.py \\
  --comparison /path/to/baseline_vs_actor/comparison.json \\
  --baseline-repeatability "${REPEATABILITY_ROOT}/repeatability.json" \\
  --out-json /path/to/baseline_vs_actor/promotion_gate.json \\
  --out-markdown /path/to/baseline_vs_actor/promotion_gate.md
EOF
