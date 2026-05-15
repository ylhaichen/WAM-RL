#!/usr/bin/env bash

# Submit the next hard/medium GRPO rollout collection round.
# This wrapper intentionally avoids the easy tasks from the baseline sweep.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}"
cd "${REPO_ROOT}"

SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-jobs/myriad/32_submit_grpo_scale_8tasks_4gpu.sh}"
BASE_TIMESTAMP="${BASE_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

GROUP_SIZE="${GROUP_SIZE:-8}"
GROUPS_PER_TASK="${GROUPS_PER_TASK:-8}"
PROMPT_INDEX="${PROMPT_INDEX:-0}"
ACTION_NUM_INFERENCE_STEPS="${ACTION_NUM_INFERENCE_STEPS:-50}"
STRICT_GRPO_CAPTURE="${STRICT_GRPO_CAPTURE:-true}"
STRICT_GRPO_TRANSITION_STD="${STRICT_GRPO_TRANSITION_STD:-0.01}"

SUBMIT_CORE="${SUBMIT_CORE:-1}"
SUBMIT_SECONDARY="${SUBMIT_SECONDARY:-1}"
DRY_RUN="${DRY_RUN:-0}"

CORE_JOB_NAME="${CORE_JOB_NAME:-wam_grpo_core}"
CORE_RUN_ID="${CORE_RUN_ID:-grpo_core_hard_medium_k${GROUP_SIZE}_g${GROUPS_PER_TASK}_seedsearch_${BASE_TIMESTAMP}}"
CORE_START_SEED="${CORE_START_SEED:-80000}"
CORE_TASK_NAMES="${CORE_TASK_NAMES:-hanging_mug turn_switch open_microwave put_bottles_dustbin}"
CORE_GROUP_SEED_SEARCH_MAX_ATTEMPTS="${CORE_GROUP_SEED_SEARCH_MAX_ATTEMPTS:-200}"
CORE_GROUP_RETRY_MULTIPLIER="${CORE_GROUP_RETRY_MULTIPLIER:-12}"
CORE_GROUP_MAX_ATTEMPTS="${CORE_GROUP_MAX_ATTEMPTS:-$((GROUPS_PER_TASK * CORE_GROUP_RETRY_MULTIPLIER))}"

SECONDARY_JOB_NAME="${SECONDARY_JOB_NAME:-wam_grpo_med}"
SECONDARY_RUN_ID="${SECONDARY_RUN_ID:-grpo_secondary_medium_k${GROUP_SIZE}_g${GROUPS_PER_TASK}_seedsearch_${BASE_TIMESTAMP}}"
SECONDARY_START_SEED="${SECONDARY_START_SEED:-90000}"
SECONDARY_TASK_NAMES="${SECONDARY_TASK_NAMES:-move_stapler_pad press_stapler place_dual_shoes place_fan}"
SECONDARY_GROUP_SEED_SEARCH_MAX_ATTEMPTS="${SECONDARY_GROUP_SEED_SEARCH_MAX_ATTEMPTS:-120}"
SECONDARY_GROUP_RETRY_MULTIPLIER="${SECONDARY_GROUP_RETRY_MULTIPLIER:-6}"
SECONDARY_GROUP_MAX_ATTEMPTS="${SECONDARY_GROUP_MAX_ATTEMPTS:-$((GROUPS_PER_TASK * SECONDARY_GROUP_RETRY_MULTIPLIER))}"

if [ ! -f "${SUBMIT_SCRIPT}" ]; then
    echo "Missing submit script: ${SUBMIT_SCRIPT}" >&2
    exit 2
fi

if [ "${DRY_RUN}" != "1" ] && ! command -v qsub >/dev/null 2>&1; then
    echo "qsub is not available on PATH. Run this on a Myriad login node, or set DRY_RUN=1." >&2
    exit 2
fi

submit_collection() {
    local job_name="$1"
    local run_id="$2"
    local start_seed="$3"
    local task_names="$4"
    local seed_search_attempts="$5"
    local retry_multiplier="$6"
    local max_attempts="$7"

    cat <<EOF
Next-round GRPO collection
  JOB_NAME=${job_name}
  RUN_ID=${run_id}
  TASK_NAMES=${task_names}
  GROUP_SIZE=${GROUP_SIZE}
  GROUPS_PER_TASK=${GROUPS_PER_TASK}
  START_SEED=${start_seed}
  GROUP_SEED_SEARCH_MAX_ATTEMPTS=${seed_search_attempts}
  GROUP_RETRY_MULTIPLIER=${retry_multiplier}
  GROUP_MAX_ATTEMPTS=${max_attempts}
EOF

    if [ "${DRY_RUN}" = "1" ]; then
        echo "DRY_RUN=1, not submitting ${job_name}."
        return
    fi

    JOB_NAME="${job_name}" \
    RUN_ID="${run_id}" \
    GROUP_SIZE="${GROUP_SIZE}" \
    GROUPS_PER_TASK="${GROUPS_PER_TASK}" \
    START_SEED="${start_seed}" \
    PROMPT_INDEX="${PROMPT_INDEX}" \
    ACTION_NUM_INFERENCE_STEPS="${ACTION_NUM_INFERENCE_STEPS}" \
    STRICT_GRPO_CAPTURE="${STRICT_GRPO_CAPTURE}" \
    STRICT_GRPO_TRANSITION_STD="${STRICT_GRPO_TRANSITION_STD}" \
    TASK_NAMES="${task_names}" \
    GROUP_SEED_SEARCH=true \
    GROUP_SEED_SEARCH_MAX_ATTEMPTS="${seed_search_attempts}" \
    GROUP_RETRY_MULTIPLIER="${retry_multiplier}" \
    GROUP_MAX_ATTEMPTS="${max_attempts}" \
    REPO_ROOT="${REPO_ROOT}" \
    bash "${SUBMIT_SCRIPT}"
}

if [ "${SUBMIT_CORE}" = "1" ]; then
    submit_collection \
        "${CORE_JOB_NAME}" \
        "${CORE_RUN_ID}" \
        "${CORE_START_SEED}" \
        "${CORE_TASK_NAMES}" \
        "${CORE_GROUP_SEED_SEARCH_MAX_ATTEMPTS}" \
        "${CORE_GROUP_RETRY_MULTIPLIER}" \
        "${CORE_GROUP_MAX_ATTEMPTS}"
fi

if [ "${SUBMIT_SECONDARY}" = "1" ]; then
    submit_collection \
        "${SECONDARY_JOB_NAME}" \
        "${SECONDARY_RUN_ID}" \
        "${SECONDARY_START_SEED}" \
        "${SECONDARY_TASK_NAMES}" \
        "${SECONDARY_GROUP_SEED_SEARCH_MAX_ATTEMPTS}" \
        "${SECONDARY_GROUP_RETRY_MULTIPLIER}" \
        "${SECONDARY_GROUP_MAX_ATTEMPTS}"
fi
