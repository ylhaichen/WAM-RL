#!/bin/bash -l

# Phase 4 real actor replay GRPO trainer.
# Requires strict artifacts collected with STRICT_GRPO_SAVE_REPLAY_CONTEXT=true.

#$ -S /bin/bash
#$ -N wam_grpo_actor
#$ -cwd
#$ -j y
#$ -o logs/jobs
#$ -l h_rt=8:00:00
#$ -l mem=8G
#$ -pe smp 8
#$ -l tmpfs=100G
#$ -l gpu=1
#$ -ac allow=L

set -euo pipefail

MYRIAD_JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${REPO_ROOT:-}" ]; then
    REPO_ROOT="$(cd "${MYRIAD_JOB_DIR}/../.." && pwd)"
fi
source "${REPO_ROOT}/jobs/myriad/common.sh"

RUN_ID="${RUN_ID:-grpo_actor_replay_${JOB_ID:-manual}}"
RESULTS_ROOT="${RESULTS_ROOT:-${WAM_ROOT}/results_grouped_rollouts/latest}"
GRPO_GROUPS_PATH="${GRPO_GROUPS_PATH:-${RESULTS_ROOT}/groups/grpo_groups.jsonl}"
GRPO_OUTPUT_DIR="${GRPO_OUTPUT_DIR:-${WAM_ROOT}/results_grpo_actor_replay/${RUN_ID}}"
GRPO_STEPS="${GRPO_STEPS:-1}"
GRPO_LR="${GRPO_LR:-0.00005}"
GRPO_CLIP_LOW="${GRPO_CLIP_LOW:-0.2}"
GRPO_CLIP_HIGH="${GRPO_CLIP_HIGH:-0.28}"
GRPO_DEVICE="${GRPO_DEVICE:-cuda}"
GRPO_DTYPE="${GRPO_DTYPE:-bfloat16}"
GRPO_SEED="${GRPO_SEED:-0}"
GRPO_TRAINABLE_MODE="${GRPO_TRAINABLE_MODE:-action_heads}"
GRPO_CONFIG_NAME="${GRPO_CONFIG_NAME:-robotwin_grpo_train}"
GRPO_ACTION_NUM_INFERENCE_STEPS="${GRPO_ACTION_NUM_INFERENCE_STEPS:-}"

export RUN_ID RESULTS_ROOT GRPO_GROUPS_PATH GRPO_OUTPUT_DIR
export GRPO_STEPS GRPO_LR GRPO_CLIP_LOW GRPO_CLIP_HIGH GRPO_DEVICE GRPO_DTYPE GRPO_SEED
export GRPO_TRAINABLE_MODE GRPO_CONFIG_NAME GRPO_ACTION_NUM_INFERENCE_STEPS

print_job_context
echo "RUN_ID=${RUN_ID}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "GRPO_GROUPS_PATH=${GRPO_GROUPS_PATH}"
echo "GRPO_OUTPUT_DIR=${GRPO_OUTPUT_DIR}"
echo "GRPO_STEPS=${GRPO_STEPS}"
echo "GRPO_LR=${GRPO_LR}"
echo "GRPO_DEVICE=${GRPO_DEVICE}"
echo "GRPO_DTYPE=${GRPO_DTYPE}"
echo "GRPO_TRAINABLE_MODE=${GRPO_TRAINABLE_MODE}"
echo "GRPO_ACTION_NUM_INFERENCE_STEPS=${GRPO_ACTION_NUM_INFERENCE_STEPS}"

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

if [ ! -f "${GRPO_GROUPS_PATH}" ]; then
    echo "Missing GRPO groups file: ${GRPO_GROUPS_PATH}" >&2
    exit 2
fi

mkdir -p "${GRPO_OUTPUT_DIR}"

ACTION_STEPS_ARGS=()
if [ -n "${GRPO_ACTION_NUM_INFERENCE_STEPS}" ]; then
    ACTION_STEPS_ARGS=(--action-num-inference-steps "${GRPO_ACTION_NUM_INFERENCE_STEPS}")
fi

python tools/validate_grpo_dataset.py \
    "${GRPO_GROUPS_PATH}" \
    --inspect-artifacts \
    --require-replay-context \
    --out-summary "${GRPO_OUTPUT_DIR}/input_dataset_validation.json" \
    --fail-on-error

python tools/train_actor_replay_grpo.py \
    --groups-jsonl "${GRPO_GROUPS_PATH}" \
    --output-dir "${GRPO_OUTPUT_DIR}" \
    --model-path "${WAN_VA_MODEL_PATH}" \
    --config-name "${GRPO_CONFIG_NAME}" \
    --steps "${GRPO_STEPS}" \
    --learning-rate "${GRPO_LR}" \
    --clip-low "${GRPO_CLIP_LOW}" \
    --clip-high "${GRPO_CLIP_HIGH}" \
    --device "${GRPO_DEVICE}" \
    --dtype "${GRPO_DTYPE}" \
    --seed "${GRPO_SEED}" \
    --trainable-mode "${GRPO_TRAINABLE_MODE}" \
    "${ACTION_STEPS_ARGS[@]}"

echo "Actor replay GRPO training complete: ${GRPO_OUTPUT_DIR}"
CONTAINER
