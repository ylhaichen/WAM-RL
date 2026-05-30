#!/bin/bash -l

# Phase 4 strict-artifact offline GRPO smoke trainer.
# This validates grouped rollout data, GRPO loss, optimizer wiring, metrics,
# and checkpoint IO before a full LingBot-VA actor replay adapter is enabled.

#$ -S /bin/bash
#$ -N wam_grpo_train
#$ -cwd
#$ -j y
#$ -o logs/jobs
#$ -l h_rt=4:00:00
#$ -l mem=4G
#$ -pe smp 8
#$ -l tmpfs=50G
#$ -l gpu=1
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

RUN_ID="${RUN_ID:-grpo_train_${JOB_ID:-manual}}"
RESULTS_ROOT="${RESULTS_ROOT:-${WAM_ROOT}/results_grouped_rollouts/latest}"
GRPO_GROUPS_PATH="${GRPO_GROUPS_PATH:-${RESULTS_ROOT}/groups/grpo_groups.jsonl}"
GRPO_OUTPUT_DIR="${GRPO_OUTPUT_DIR:-${WAM_ROOT}/results_grpo_train/${RUN_ID}}"
GRPO_STEPS="${GRPO_STEPS:-20}"
GRPO_LR="${GRPO_LR:-0.001}"
GRPO_CLIP_LOW="${GRPO_CLIP_LOW:-0.2}"
GRPO_CLIP_HIGH="${GRPO_CLIP_HIGH:-0.28}"
GRPO_DEVICE="${GRPO_DEVICE:-cpu}"
GRPO_SEED="${GRPO_SEED:-0}"

export RUN_ID RESULTS_ROOT GRPO_GROUPS_PATH GRPO_OUTPUT_DIR
export GRPO_STEPS GRPO_LR GRPO_CLIP_LOW GRPO_CLIP_HIGH GRPO_DEVICE GRPO_SEED

print_job_context
echo "RUN_ID=${RUN_ID}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "GRPO_GROUPS_PATH=${GRPO_GROUPS_PATH}"
echo "GRPO_OUTPUT_DIR=${GRPO_OUTPUT_DIR}"
echo "GRPO_STEPS=${GRPO_STEPS}"
echo "GRPO_LR=${GRPO_LR}"
echo "GRPO_DEVICE=${GRPO_DEVICE}"

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
    echo "Pass GRPO_GROUPS_PATH=/path/to/grpo_groups.jsonl or RESULTS_ROOT=/path/to/collection." >&2
    exit 2
fi

mkdir -p "${GRPO_OUTPUT_DIR}"

python tools/validate_grpo_dataset.py \
    "${GRPO_GROUPS_PATH}" \
    --out-summary "${GRPO_OUTPUT_DIR}/input_dataset_validation.json" \
    --fail-on-error

python tools/train_offline_grpo_smoke.py \
    --groups-jsonl "${GRPO_GROUPS_PATH}" \
    --output-dir "${GRPO_OUTPUT_DIR}" \
    --steps "${GRPO_STEPS}" \
    --learning-rate "${GRPO_LR}" \
    --clip-low "${GRPO_CLIP_LOW}" \
    --clip-high "${GRPO_CLIP_HIGH}" \
    --device "${GRPO_DEVICE}" \
    --seed "${GRPO_SEED}"

echo "Offline GRPO smoke training complete: ${GRPO_OUTPUT_DIR}"
CONTAINER
