#!/usr/bin/env bash

# Submit a storage-bounded replay-context grouped rollout collection.
# This is a conservative wrapper around 32_submit_grpo_scale_8tasks_4gpu.sh
# for actor-replay data collection, not a benchmark-scale rollout launcher.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: jobs/myriad/39_submit_grpo_replayctx_bounded_4gpu.sh [--dry-run]

Submit a storage-bounded replay-context grouped rollout collection.

Options:
  --dry-run   Print planning output and exit without submitting.
  -h, --help  Show this help text.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

MYRIAD_JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${REPO_ROOT:-}" ]; then
    REPO_ROOT="$(cd "${MYRIAD_JOB_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-jobs/myriad/32_submit_grpo_scale_8tasks_4gpu.sh}"
DRY_RUN="${DRY_RUN:-0}"

JOB_NAME="${JOB_NAME:-wam_grpo_replayctx_bounded}"
TASK_NAMES="${TASK_NAMES:-move_stapler_pad}"
GROUP_SIZE="${GROUP_SIZE:-8}"
GROUPS_PER_TASK="${GROUPS_PER_TASK:-1}"
START_SEED="${START_SEED:-99200}"
PROMPT_INDEX="${PROMPT_INDEX:-0}"
ACTION_NUM_INFERENCE_STEPS="${ACTION_NUM_INFERENCE_STEPS:-10}"
GROUP_SEED_SEARCH_MAX_ATTEMPTS="${GROUP_SEED_SEARCH_MAX_ATTEMPTS:-120}"
GROUP_RETRY_MULTIPLIER="${GROUP_RETRY_MULTIPLIER:-3}"
GROUP_MAX_ATTEMPTS="${GROUP_MAX_ATTEMPTS:-$((GROUPS_PER_TASK * GROUP_RETRY_MULTIPLIER))}"
STRICT_GRPO_CAPTURE="${STRICT_GRPO_CAPTURE:-true}"
STRICT_GRPO_TRANSITION_STD="${STRICT_GRPO_TRANSITION_STD:-0.01}"
STRICT_GRPO_CAPTURE_SCOPE="${STRICT_GRPO_CAPTURE_SCOPE:-action_denoising_trajectory}"
STRICT_GRPO_SAVE_REPLAY_CONTEXT="${STRICT_GRPO_SAVE_REPLAY_CONTEXT:-true}"
STRICT_GRPO_CAPTURE_CHUNK_STRIDE="${STRICT_GRPO_CAPTURE_CHUNK_STRIDE:-1}"
STRICT_GRPO_CAPTURE_MAX_CHUNKS="${STRICT_GRPO_CAPTURE_MAX_CHUNKS:-1}"
SAVE_SERVER_DEBUG_TENSORS="${SAVE_SERVER_DEBUG_TENSORS:-false}"
# Set ALLOW_UNBOUNDED_REPLAYCTX=1 only after explicit review of a full-capture run.
ALLOW_UNBOUNDED_REPLAYCTX="${ALLOW_UNBOUNDED_REPLAYCTX:-0}"
REPLAY_CONTEXT_ESTIMATE_GB="${REPLAY_CONTEXT_ESTIMATE_GB:-4.0}"
STRICT_GRPO_REPLAY_CONTEXT_MAX_GB="${STRICT_GRPO_REPLAY_CONTEXT_MAX_GB:-5.0}"
CHECK_SCRATCH_HEADROOM="${CHECK_SCRATCH_HEADROOM:-1}"
SCRATCH_PATH="${SCRATCH_PATH:-/home/zcably0/Scratch}"
MIN_SCRATCH_HEADROOM_GB="${MIN_SCRATCH_HEADROOM_GB:-50}"
STORAGE_BUDGET_MODE="${STORAGE_BUDGET_MODE:-attempt}"
SUCCESS_RATE="${SUCCESS_RATE:-}"
PLAN_JSON="${PLAN_JSON:-}"
QSUB_H_RT="${QSUB_H_RT:-6:00:00}"
QSUB_MEM="${QSUB_MEM:-4G}"
QSUB_SLOTS="${QSUB_SLOTS:-32}"
QSUB_TMPFS="${QSUB_TMPFS:-80G}"
RUN_ID="${RUN_ID:-grpo_replayctx_bounded_k${GROUP_SIZE}_g${GROUPS_PER_TASK}_s${ACTION_NUM_INFERENCE_STEPS}_$(date +%Y%m%d_%H%M%S)}"

if [ ! -f "${SUBMIT_SCRIPT}" ]; then
    echo "Missing submit script: ${SUBMIT_SCRIPT}" >&2
    exit 2
fi

cat <<EOF
Bounded replay-context GRPO collection
  JOB_NAME=${JOB_NAME}
  RUN_ID=${RUN_ID}
  TASK_NAMES=${TASK_NAMES}
  GROUP_SIZE=${GROUP_SIZE}
  GROUPS_PER_TASK=${GROUPS_PER_TASK}
  START_SEED=${START_SEED}
  PROMPT_INDEX=${PROMPT_INDEX}
  ACTION_NUM_INFERENCE_STEPS=${ACTION_NUM_INFERENCE_STEPS}
  STRICT_GRPO_CAPTURE_SCOPE=${STRICT_GRPO_CAPTURE_SCOPE}
  STRICT_GRPO_SAVE_REPLAY_CONTEXT=${STRICT_GRPO_SAVE_REPLAY_CONTEXT}
  STRICT_GRPO_CAPTURE_CHUNK_STRIDE=${STRICT_GRPO_CAPTURE_CHUNK_STRIDE}
  STRICT_GRPO_CAPTURE_MAX_CHUNKS=${STRICT_GRPO_CAPTURE_MAX_CHUNKS}
  SAVE_SERVER_DEBUG_TENSORS=${SAVE_SERVER_DEBUG_TENSORS}
  GROUP_MAX_ATTEMPTS=${GROUP_MAX_ATTEMPTS}
  REPLAY_CONTEXT_ESTIMATE_GB=${REPLAY_CONTEXT_ESTIMATE_GB}
  STRICT_GRPO_REPLAY_CONTEXT_MAX_GB=${STRICT_GRPO_REPLAY_CONTEXT_MAX_GB}
  CHECK_SCRATCH_HEADROOM=${CHECK_SCRATCH_HEADROOM}
  SCRATCH_PATH=${SCRATCH_PATH}
  MIN_SCRATCH_HEADROOM_GB=${MIN_SCRATCH_HEADROOM_GB}
  STORAGE_BUDGET_MODE=${STORAGE_BUDGET_MODE}
  SUCCESS_RATE=${SUCCESS_RATE}
  PLAN_JSON=${PLAN_JSON}
  QSUB_H_RT=${QSUB_H_RT}
  QSUB_MEM=${QSUB_MEM}
  QSUB_SLOTS=${QSUB_SLOTS}
  QSUB_TMPFS=${QSUB_TMPFS}
  DRY_RUN=${DRY_RUN}
EOF

PLAN_ARGS=(
    --task-names "${TASK_NAMES}" \
    --group-size "${GROUP_SIZE}" \
    --groups-per-task "${GROUPS_PER_TASK}" \
    --group-max-attempts "${GROUP_MAX_ATTEMPTS}" \
    --capture-max-chunks "${STRICT_GRPO_CAPTURE_MAX_CHUNKS}" \
    --save-replay-context "${STRICT_GRPO_SAVE_REPLAY_CONTEXT}" \
    --allow-unbounded-replayctx "${ALLOW_UNBOUNDED_REPLAYCTX}" \
    --replay-context-estimate-gb "${REPLAY_CONTEXT_ESTIMATE_GB}" \
    --storage-budget-mode "${STORAGE_BUDGET_MODE}" \
    --check-scratch-headroom "${CHECK_SCRATCH_HEADROOM}" \
    --scratch-path "${SCRATCH_PATH}" \
    --min-scratch-headroom-gb "${MIN_SCRATCH_HEADROOM_GB}" \
    --dry-run "${DRY_RUN}"
)
if [ -n "${SUCCESS_RATE}" ]; then
    PLAN_ARGS+=(--success-rate "${SUCCESS_RATE}")
fi

if [ -n "${PLAN_JSON}" ]; then
    mkdir -p "$(dirname "${PLAN_JSON}")"
    python tools/plan_replay_context_collection.py "${PLAN_ARGS[@]}" --format json > "${PLAN_JSON}"
    echo "Wrote replay-context collection plan: ${PLAN_JSON}"
fi

python tools/plan_replay_context_collection.py "${PLAN_ARGS[@]}" --format shell

run_submit_script() {
    local submit_dry_run="$1"
    JOB_NAME="${JOB_NAME}" \
    RUN_ID="${RUN_ID}" \
    TASK_NAMES="${TASK_NAMES}" \
    GROUP_SIZE="${GROUP_SIZE}" \
    GROUPS_PER_TASK="${GROUPS_PER_TASK}" \
    START_SEED="${START_SEED}" \
    PROMPT_INDEX="${PROMPT_INDEX}" \
    ACTION_NUM_INFERENCE_STEPS="${ACTION_NUM_INFERENCE_STEPS}" \
    GROUP_SEED_SEARCH=true \
    GROUP_SEED_SEARCH_MAX_ATTEMPTS="${GROUP_SEED_SEARCH_MAX_ATTEMPTS}" \
    GROUP_RETRY_MULTIPLIER="${GROUP_RETRY_MULTIPLIER}" \
    GROUP_MAX_ATTEMPTS="${GROUP_MAX_ATTEMPTS}" \
    STRICT_GRPO_CAPTURE="${STRICT_GRPO_CAPTURE}" \
    STRICT_GRPO_TRANSITION_STD="${STRICT_GRPO_TRANSITION_STD}" \
    STRICT_GRPO_CAPTURE_SCOPE="${STRICT_GRPO_CAPTURE_SCOPE}" \
    STRICT_GRPO_SAVE_REPLAY_CONTEXT="${STRICT_GRPO_SAVE_REPLAY_CONTEXT}" \
    STRICT_GRPO_CAPTURE_CHUNK_STRIDE="${STRICT_GRPO_CAPTURE_CHUNK_STRIDE}" \
    STRICT_GRPO_CAPTURE_MAX_CHUNKS="${STRICT_GRPO_CAPTURE_MAX_CHUNKS}" \
    STRICT_GRPO_REPLAY_CONTEXT_MAX_GB="${STRICT_GRPO_REPLAY_CONTEXT_MAX_GB}" \
    SAVE_SERVER_DEBUG_TENSORS="${SAVE_SERVER_DEBUG_TENSORS}" \
    REPO_ROOT="${REPO_ROOT}" \
    QSUB_H_RT="${QSUB_H_RT}" \
    QSUB_MEM="${QSUB_MEM}" \
    QSUB_SLOTS="${QSUB_SLOTS}" \
    QSUB_TMPFS="${QSUB_TMPFS}" \
    DRY_RUN="${submit_dry_run}" \
    bash "${SUBMIT_SCRIPT}"
}

if [ "${DRY_RUN}" = "1" ]; then
    echo "DRY_RUN=1, printing underlying qsub command without submitting ${JOB_NAME}."
    run_submit_script 1
    exit 0
fi

run_submit_script 0
