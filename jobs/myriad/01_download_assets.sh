#!/bin/bash -l

# Batch script to download LingBot-VA checkpoints and the RoboTwin posttrain dataset.
# Submit after 00_install_container_env.sh has completed.

#$ -S /bin/bash
#$ -N wam_assets
#$ -cwd
#$ -j y
#$ -o logs/jobs
#$ -l h_rt=12:00:00
#$ -l mem=4G
#$ -pe smp 4
#$ -l tmpfs=100G

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

print_job_context

container_exec_cpu <<'CONTAINER'
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
python -m pip install -U "huggingface-hub[cli]==0.36.2"

mkdir -p "${WAN_VA_MODEL_PATH}" "${WAN_VA_BASE_MODEL_PATH}" "${WAN_VA_DATASET_PATH}"

if [ "${DOWNLOAD_EVAL_MODEL:-true}" = "true" ]; then
    hf download robbyant/lingbot-va-posttrain-robotwin \
        --local-dir "${WAN_VA_MODEL_PATH}"
    python tools/set_attn_mode.py "${WAN_VA_MODEL_PATH}" torch
fi

if [ "${DOWNLOAD_BASE_MODEL:-true}" = "true" ]; then
    hf download robbyant/lingbot-va-base \
        --local-dir "${WAN_VA_BASE_MODEL_PATH}"
    python tools/set_attn_mode.py "${WAN_VA_BASE_MODEL_PATH}" flex
fi

if [ "${DOWNLOAD_DATASET:-false}" = "true" ]; then
    hf download robbyant/robotwin-clean-and-aug-lerobot \
        --repo-type dataset \
        --local-dir "${WAN_VA_DATASET_PATH}"
else
    echo "Skipping 406G RoboTwin post-training dataset. Set DOWNLOAD_DATASET=true to download it."
fi

check_args=(
    --model-path "${WAN_VA_MODEL_PATH}"
)
if [ "${DOWNLOAD_DATASET:-false}" = "true" ]; then
    check_args+=(--dataset-path "${WAN_VA_DATASET_PATH}")
fi
if [ -d "${ROBOTWIN_ROOT}" ]; then
    check_args+=(--robotwin-root "${ROBOTWIN_ROOT}")
else
    echo "Skipping RoboTwin path check because ${ROBOTWIN_ROOT} does not exist yet."
fi
python tools/check_setup.py "${check_args[@]}"
CONTAINER
