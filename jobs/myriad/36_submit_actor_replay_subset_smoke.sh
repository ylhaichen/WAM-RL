#!/bin/bash

# Submit a low-resource real actor replay smoke job for a materialized subset.
# The actual training logic stays in 34_train_actor_replay_grpo_robotwin.sh.

set -euo pipefail

MYRIAD_JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${REPO_ROOT:-}" ]; then
    REPO_ROOT="$(cd "${MYRIAD_JOB_DIR}/../.." && pwd)"
fi

RUN_ID="${RUN_ID:-grpo_actor_subset_smoke_$(date +%Y%m%d_%H%M%S)}"
JOB_NAME="${JOB_NAME:-wam_grpo_actor_subset}"

if [ -z "${GRPO_GROUPS_PATH:-}" ]; then
    if [ -n "${SUBSET_ROOT:-}" ]; then
        GRPO_GROUPS_PATH="${SUBSET_ROOT}/groups/grpo_groups.jsonl"
    else
        echo "Set GRPO_GROUPS_PATH or SUBSET_ROOT before submitting actor replay subset smoke." >&2
        exit 2
    fi
fi

WAM_ROOT="${WAM_ROOT:-/home/zcably0/Scratch/wam-rl}"
GRPO_OUTPUT_DIR="${GRPO_OUTPUT_DIR:-${WAM_ROOT}/results_grpo_actor_replay/${RUN_ID}}"
GRPO_STEPS="${GRPO_STEPS:-1}"
GRPO_LR="${GRPO_LR:-0.0000001}"
GRPO_ACTION_NUM_INFERENCE_STEPS="${GRPO_ACTION_NUM_INFERENCE_STEPS:-10}"
GRPO_LOGPROB_REDUCTION="${GRPO_LOGPROB_REDUCTION:-mean}"
GRPO_LOGPROB_STD_FLOOR="${GRPO_LOGPROB_STD_FLOOR:-0.1}"
GRPO_PROGRESS_EVERY="${GRPO_PROGRESS_EVERY:-1}"
GRPO_TRAINABLE_MODE="${GRPO_TRAINABLE_MODE:-action_heads}"

QSUB_H_RT="${QSUB_H_RT:-2:00:00}"
QSUB_MEM="${QSUB_MEM:-8G}"
QSUB_SLOTS="${QSUB_SLOTS:-4}"
QSUB_TMPFS="${QSUB_TMPFS:-40G}"
QSUB_GPU="${QSUB_GPU:-1}"
DRY_RUN="${DRY_RUN:-0}"

export REPO_ROOT RUN_ID GRPO_GROUPS_PATH GRPO_OUTPUT_DIR
export GRPO_STEPS GRPO_LR GRPO_ACTION_NUM_INFERENCE_STEPS
export GRPO_LOGPROB_REDUCTION GRPO_LOGPROB_STD_FLOOR GRPO_PROGRESS_EVERY
export GRPO_TRAINABLE_MODE

JOB_SCRIPT="${REPO_ROOT}/jobs/myriad/34_train_actor_replay_grpo_robotwin.sh"
if [ ! -f "${JOB_SCRIPT}" ]; then
    echo "Missing actor replay trainer job script: ${JOB_SCRIPT}" >&2
    exit 2
fi

echo "Submitting actor replay subset smoke job"
echo "  JOB_NAME=${JOB_NAME}"
echo "  RUN_ID=${RUN_ID}"
echo "  REPO_ROOT=${REPO_ROOT}"
echo "  GRPO_GROUPS_PATH=${GRPO_GROUPS_PATH}"
echo "  GRPO_OUTPUT_DIR=${GRPO_OUTPUT_DIR}"
echo "  GRPO_STEPS=${GRPO_STEPS}"
echo "  GRPO_LR=${GRPO_LR}"
echo "  GRPO_ACTION_NUM_INFERENCE_STEPS=${GRPO_ACTION_NUM_INFERENCE_STEPS}"
echo "  GRPO_LOGPROB_REDUCTION=${GRPO_LOGPROB_REDUCTION}"
echo "  GRPO_LOGPROB_STD_FLOOR=${GRPO_LOGPROB_STD_FLOOR}"
echo "  QSUB_H_RT=${QSUB_H_RT}"
echo "  QSUB_MEM=${QSUB_MEM}"
echo "  QSUB_SLOTS=${QSUB_SLOTS}"
echo "  QSUB_TMPFS=${QSUB_TMPFS}"
echo "  QSUB_GPU=${QSUB_GPU}"

QSUB_ARGS=(
    -V
    -N "${JOB_NAME}"
    -l "h_rt=${QSUB_H_RT}"
    -l "mem=${QSUB_MEM}"
    -pe smp "${QSUB_SLOTS}"
    -l "tmpfs=${QSUB_TMPFS}"
)
if [ -n "${QSUB_GPU}" ] && [ "${QSUB_GPU}" != "0" ]; then
    QSUB_ARGS+=(-l "gpu=${QSUB_GPU}")
fi

if [ "${DRY_RUN}" = "1" ]; then
    printf 'qsub'
    printf ' %q' "${QSUB_ARGS[@]}" "${JOB_SCRIPT}"
    printf '\n'
    exit 0
fi

qsub "${QSUB_ARGS[@]}" "${JOB_SCRIPT}"
