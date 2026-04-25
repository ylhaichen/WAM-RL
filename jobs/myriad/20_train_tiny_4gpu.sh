#!/bin/bash -l

# Tiny four-GPU training smoke test. This is only a wiring check, not a real run.

#$ -S /bin/bash
#$ -N wam_train_tiny
#$ -cwd
#$ -j y
#$ -o logs/jobs
#$ -l h_rt=4:00:00
#$ -l mem=4G
#$ -pe smp 32
#$ -l tmpfs=200G
#$ -l gpu=4
#$ -ac allow=L

set -euo pipefail

JOB_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${JOB_SCRIPT_DIR}/common.sh"

NUM_GPUS="${NUM_GPUS:-4}"
TRAIN_STEPS="${TRAIN_STEPS:-10}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
SAVE_ROOT="${SAVE_ROOT:-${WAM_ROOT}/train_tiny}"
MASTER_PORT="${MASTER_PORT:-29761}"

export NUM_GPUS TRAIN_STEPS GRAD_ACCUM SAVE_ROOT MASTER_PORT

print_job_context

container_exec_gpu <<'CONTAINER'
set -euo pipefail

cd "${REPO_ROOT}"

if [ ! -x "${WAN_VA_VENV}/bin/python" ]; then
    echo "Missing venv: ${WAN_VA_VENV}" >&2
    echo "Run jobs/myriad/00_install_container_env.sh first." >&2
    exit 1
fi

source "${WAN_VA_VENV}/bin/activate"

mkdir -p "${SAVE_ROOT}"

python tools/set_attn_mode.py "${WAN_VA_BASE_MODEL_PATH}" flex
python tools/check_setup.py \
    --model-path "${WAN_VA_BASE_MODEL_PATH}" \
    --dataset-path "${WAN_VA_DATASET_PATH}"

nvidia-smi

export WAN_VA_MODEL_PATH="${WAN_VA_BASE_MODEL_PATH}"
export WAN_VA_DATASET_PATH
export WAN_VA_EMPTY_EMB_PATH
export WAN_VA_ENABLE_WANDB=false

NGPU="${NUM_GPUS}" \
MASTER_PORT="${MASTER_PORT}" \
CONFIG_NAME=robotwin_train \
bash script/run_va_posttrain.sh \
    num_steps="${TRAIN_STEPS}" \
    batch_size=1 \
    gradient_accumulation_steps="${GRAD_ACCUM}" \
    load_worker=4 \
    save_interval="${TRAIN_STEPS}" \
    save_root="${SAVE_ROOT}"
CONTAINER
