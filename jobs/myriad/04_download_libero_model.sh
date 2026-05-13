#!/bin/bash -l

# Batch script to download the LingBot-VA LIBERO-LONG checkpoint.
# Submit from the repo root with:
#   qsub jobs/myriad/04_download_libero_model.sh

#$ -S /bin/bash
#$ -N wam_libero_model
#$ -cwd
#$ -j y
#$ -o logs/jobs
#$ -l h_rt=6:00:00
#$ -l mem=4G
#$ -pe smp 4
#$ -l tmpfs=80G

set -euo pipefail

MYRIAD_JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${REPO_ROOT:-}" ]; then
    REPO_ROOT="$(cd "${MYRIAD_JOB_DIR}/../.." && pwd)"
fi
source "${REPO_ROOT}/jobs/myriad/common.sh"

LIBERO_MODEL_REPO="${LIBERO_MODEL_REPO:-robbyant/lingbot-va-posttrain-libero-long}"
WAN_VA_LIBERO_MODEL_PATH="${WAN_VA_LIBERO_MODEL_PATH:-${WAM_ROOT}/checkpoints/lingbot-va-posttrain-libero-long}"
LIBERO_ATTENTION_MODE="${LIBERO_ATTENTION_MODE:-torch}"
HF_HOME="${HF_HOME:-${WAM_ROOT}/hf-cache}"

export LIBERO_MODEL_REPO
export WAN_VA_LIBERO_MODEL_PATH
export LIBERO_ATTENTION_MODE
export HF_HOME

print_job_context
echo "LIBERO_MODEL_REPO=${LIBERO_MODEL_REPO}"
echo "WAN_VA_LIBERO_MODEL_PATH=${WAN_VA_LIBERO_MODEL_PATH}"
echo "LIBERO_ATTENTION_MODE=${LIBERO_ATTENTION_MODE}"
echo "HF_HOME=${HF_HOME}"

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

mkdir -p "${WAN_VA_LIBERO_MODEL_PATH}" "${HF_HOME}"

python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["LIBERO_MODEL_REPO"]
local_dir = os.environ["WAN_VA_LIBERO_MODEL_PATH"]

print(f"Downloading {repo_id} to {local_dir}")
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
)
PY

if [ ! -d "${WAN_VA_LIBERO_MODEL_PATH}/transformer" ]; then
    echo "Download finished, but transformer/ was not found in ${WAN_VA_LIBERO_MODEL_PATH}" >&2
    exit 1
fi

python tools/set_attn_mode.py "${WAN_VA_LIBERO_MODEL_PATH}" "${LIBERO_ATTENTION_MODE}"

python - <<'PY'
import json
import os
from pathlib import Path

model_path = Path(os.environ["WAN_VA_LIBERO_MODEL_PATH"])
config_path = model_path / "transformer" / "config.json"
config = json.loads(config_path.read_text())

print(f"LIBERO checkpoint ready: {model_path}")
print(f"transformer attn_mode={config.get('attn_mode')}")
print("Use it for smoke testing with:")
print(f"  WAN_VA_MODEL_PATH={model_path} CUDA_VISIBLE_DEVICES=0 bash script/run_libero_smoke.sh")
PY
CONTAINER
