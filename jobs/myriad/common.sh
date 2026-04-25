#!/bin/bash

set -euo pipefail

JOB_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${JOB_SCRIPT_DIR}/../.." && pwd)}"

if [ -d "${HOME}/Scratch" ]; then
    DEFAULT_WAM_ROOT="${HOME}/Scratch/wam-rl"
else
    DEFAULT_WAM_ROOT="${HOME}/wam-rl"
fi

WAM_ROOT="${WAM_ROOT:-${DEFAULT_WAM_ROOT}}"
SIF="${SIF:-${HOME}/containers/pytorch-2.9.0-cu126.sif}"
WAN_VA_VENV="${WAN_VA_VENV:-${WAM_ROOT}/venvs/wam-rl-container}"
WAN_VA_CONDA_LIBS="${WAN_VA_CONDA_LIBS:-${WAM_ROOT}/conda-libs}"
WAN_VA_EGL_VENDOR_DIR="${WAN_VA_EGL_VENDOR_DIR:-${WAM_ROOT}/glvnd/egl_vendor.d}"
WAN_VA_VULKAN_ICD_DIR="${WAN_VA_VULKAN_ICD_DIR:-${WAM_ROOT}/vulkan/icd.d}"
WAN_VA_MODEL_PATH="${WAN_VA_MODEL_PATH:-${WAM_ROOT}/checkpoints/lingbot-va-posttrain-robotwin}"
WAN_VA_BASE_MODEL_PATH="${WAN_VA_BASE_MODEL_PATH:-${WAM_ROOT}/checkpoints/lingbot-va-base}"
WAN_VA_DATASET_PATH="${WAN_VA_DATASET_PATH:-${WAM_ROOT}/datasets/robotwin-clean-and-aug-lerobot}"
WAN_VA_EMPTY_EMB_PATH="${WAN_VA_EMPTY_EMB_PATH:-${WAN_VA_DATASET_PATH}/empty_emb.pt}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${WAN_VA_ROBOTWIN_ROOT:-${WAM_ROOT}/RoboTwin}}"
WAN_VA_ROBOTWIN_ROOT="${WAN_VA_ROBOTWIN_ROOT:-${ROBOTWIN_ROOT}}"

export REPO_ROOT
export WAM_ROOT
export SIF
export WAN_VA_VENV
export WAN_VA_CONDA_LIBS
export WAN_VA_EGL_VENDOR_DIR
export WAN_VA_VULKAN_ICD_DIR
export WAN_VA_MODEL_PATH
export WAN_VA_BASE_MODEL_PATH
export WAN_VA_DATASET_PATH
export WAN_VA_EMPTY_EMB_PATH
export ROBOTWIN_ROOT
export WAN_VA_ROBOTWIN_ROOT
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WAN_VA_ENABLE_WANDB="${WAN_VA_ENABLE_WANDB:-false}"

if [ -d "${WAN_VA_CONDA_LIBS}/lib" ]; then
    export LD_LIBRARY_PATH="${WAN_VA_CONDA_LIBS}/lib:${LD_LIBRARY_PATH:-}"
fi

mkdir -p "${WAN_VA_EGL_VENDOR_DIR}" "${WAN_VA_VULKAN_ICD_DIR}"
cat > "${WAN_VA_EGL_VENDOR_DIR}/10_nvidia.json" <<'JSON'
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "libEGL_nvidia.so.0"
  }
}
JSON
cat > "${WAN_VA_VULKAN_ICD_DIR}/nvidia_icd.json" <<'JSON'
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "libGLX_nvidia.so.0",
    "api_version": "1.3.204"
  }
}
JSON
export __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
export VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json"

if command -v module >/dev/null 2>&1; then
    module load apptainer/1.2.4-1 2>/dev/null || true
fi

if ! command -v apptainer >/dev/null 2>&1; then
    echo "apptainer is not available. Try: module load apptainer/1.2.4-1" >&2
    exit 1
fi

if [ ! -f "${SIF}" ]; then
    echo "Missing Apptainer image: ${SIF}" >&2
    echo "Build it first, for example:" >&2
    echo "  mkdir -p ${HOME}/containers" >&2
    echo "  apptainer pull ${SIF} docker://pytorch/pytorch:2.9.0-cuda12.6-cudnn9-devel" >&2
    exit 1
fi

mkdir -p "${WAM_ROOT}" "${REPO_ROOT}/logs/jobs"

APPTAINER_ARGS=(--bind "${HOME}:${HOME}")
if [ -d /shared ]; then
    APPTAINER_ARGS+=(--bind /shared:/shared)
fi
if [ -n "${TMPDIR:-}" ] && [ -d "${TMPDIR}" ]; then
    APPTAINER_ARGS+=(--bind "${TMPDIR}:${TMPDIR}")
fi
if [ -d "${WAM_ROOT}" ]; then
    APPTAINER_ARGS+=(--bind "${WAM_ROOT}:${WAM_ROOT}")
fi
if [ -d "${REPO_ROOT}" ]; then
    APPTAINER_ARGS+=(--bind "${REPO_ROOT}:${REPO_ROOT}")
fi
if [ -d "${ROBOTWIN_ROOT}" ]; then
    APPTAINER_ARGS+=(--bind "${ROBOTWIN_ROOT}:${ROBOTWIN_ROOT}")
fi
APPTAINER_ARGS+=(--bind "${WAN_VA_EGL_VENDOR_DIR}:/usr/share/glvnd/egl_vendor.d")
APPTAINER_ARGS+=(--bind "${WAN_VA_VULKAN_ICD_DIR}:/usr/share/vulkan/icd.d")
APPTAINER_ARGS+=(--bind "${WAN_VA_VULKAN_ICD_DIR}:/etc/vulkan/icd.d")

print_job_context() {
    echo "JOB_ID=${JOB_ID:-local}"
    echo "HOST=$(hostname)"
    echo "DATE=$(date -Is)"
    echo "REPO_ROOT=${REPO_ROOT}"
    echo "WAM_ROOT=${WAM_ROOT}"
    echo "SIF=${SIF}"
    echo "WAN_VA_VENV=${WAN_VA_VENV}"
    echo "WAN_VA_MODEL_PATH=${WAN_VA_MODEL_PATH}"
    echo "WAN_VA_BASE_MODEL_PATH=${WAN_VA_BASE_MODEL_PATH}"
    echo "WAN_VA_DATASET_PATH=${WAN_VA_DATASET_PATH}"
    echo "ROBOTWIN_ROOT=${ROBOTWIN_ROOT}"
}

container_exec_cpu() {
    apptainer exec "${APPTAINER_ARGS[@]}" "${SIF}" bash -s
}

container_exec_gpu() {
    apptainer exec --nv "${APPTAINER_ARGS[@]}" "${SIF}" bash -s
}
