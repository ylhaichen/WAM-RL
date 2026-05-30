#!/bin/bash -l

# Batch script to build the Python venv inside the PyTorch Apptainer image.
# Submit from the repo root with: qsub jobs/myriad/00_install_container_env.sh

#$ -S /bin/bash
#$ -N wam_install
#$ -cwd
#$ -j y
#$ -o logs/jobs
#$ -l h_rt=4:00:00
#$ -l mem=4G
#$ -pe smp 8
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

python -V
python -m venv --system-site-packages "${WAN_VA_VENV}"
source "${WAN_VA_VENV}/bin/activate"

python -m pip install -U pip setuptools wheel ninja packaging
python -m pip install \
    websockets \
    einops \
    diffusers==0.36.0 \
    transformers==4.55.2 \
    accelerate \
    msgpack \
    opencv-python \
    matplotlib \
    ftfy \
    easydict \
    safetensors \
    Pillow \
    scipy \
    wandb \
    "imageio[ffmpeg]" \
    "numpy>=1.26.4,<2"
python -m pip install lerobot==0.3.3 --no-deps

if [ "${INSTALL_FLASH_ATTN:-false}" = "true" ]; then
    MAX_JOBS="${MAX_JOBS:-4}" python -m pip install flash-attn --no-build-isolation
else
    echo "Skipping flash-attn. Use INSTALL_FLASH_ATTN=true if you need attn_mode=flashattn."
fi

python -m pip install -e . --no-deps

python - <<'PY'
import torch
from wan_va.modules.model import WanAttention

print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("WanAttention import ok", WanAttention.__name__)
PY
CONTAINER
