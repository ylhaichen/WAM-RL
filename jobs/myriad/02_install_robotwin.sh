#!/bin/bash -l

# Install RoboTwin into ROBOTWIN_ROOT. This is needed for evaluation clients,
# not for starting the LingBot-VA inference server.

#$ -S /bin/bash
#$ -N wam_robotwin
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

JOB_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${JOB_SCRIPT_DIR}/common.sh"

ROBOTWIN_REPO_URL="${ROBOTWIN_REPO_URL:-https://github.com/RoboTwin-Platform/RoboTwin.git}"
ROBOTWIN_COMMIT="${ROBOTWIN_COMMIT:-2eeec322}"
DOWNLOAD_ROBOTWIN_ASSETS="${DOWNLOAD_ROBOTWIN_ASSETS:-true}"

export ROBOTWIN_REPO_URL ROBOTWIN_COMMIT DOWNLOAD_ROBOTWIN_ASSETS

print_job_context
echo "ROBOTWIN_REPO_URL=${ROBOTWIN_REPO_URL}"
echo "ROBOTWIN_COMMIT=${ROBOTWIN_COMMIT}"
echo "DOWNLOAD_ROBOTWIN_ASSETS=${DOWNLOAD_ROBOTWIN_ASSETS}"

mkdir -p "$(dirname "${ROBOTWIN_ROOT}")"
if [ ! -d "${ROBOTWIN_ROOT}/.git" ]; then
    git clone "${ROBOTWIN_REPO_URL}" "${ROBOTWIN_ROOT}"
fi

cd "${ROBOTWIN_ROOT}"
git fetch --all --tags
git checkout "${ROBOTWIN_COMMIT}"

mkdir -p envs
if [ ! -d envs/curobo/.git ]; then
    git clone https://github.com/NVlabs/curobo.git envs/curobo
fi

container_exec_gpu <<'CONTAINER'
set -euo pipefail

if [ ! -x "${WAN_VA_VENV}/bin/python" ]; then
    echo "Missing venv: ${WAN_VA_VENV}" >&2
    echo "Run jobs/myriad/00_install_container_env.sh first." >&2
    exit 1
fi

source "${WAN_VA_VENV}/bin/activate"

cd "${ROBOTWIN_ROOT}"
export CC="${ROBOTWIN_CC:-gcc}"
export CXX="${ROBOTWIN_CXX:-g++}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
if [ ! -f "${WAN_VA_CONDA_LIBS}/lib/libX11.so.6" ]; then
    /opt/conda/bin/conda create -y -p "${WAN_VA_CONDA_LIBS}" -c conda-forge \
        xorg-libx11 xorg-libxext xorg-libxrender xorg-libxi xorg-libxrandr \
        xorg-libxinerama xorg-libxcursor xorg-libxfixes xorg-libsm xorg-libice \
        libxcb libglvnd glib
fi
if [ -d "${WAN_VA_CONDA_LIBS}/lib" ]; then
    export LD_LIBRARY_PATH="${WAN_VA_CONDA_LIBS}/lib:${LD_LIBRARY_PATH:-}"
fi
which "${CC}"
which "${CXX}"

cat > script/requirements.txt <<'REQ'
transforms3d==0.4.2
sapien==3.0.0b1
scipy==1.10.1
mplib==0.2.1
gymnasium==0.29.1
trimesh==4.4.3
open3d==0.18.0
imageio==2.34.2
pydantic
zarr
openai
huggingface_hub==0.36.2
h5py
azure==4.0.0
azure-ai-inference
pyglet<2
wandb
moviepy
imageio
termcolor
av
matplotlib
ffmpeg
REQ

python - <<'PY'
from pathlib import Path

path = Path("script/_install.sh")
text = path.read_text()
text = text.replace(
    'pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"',
    'pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation',
)
text = text.replace(
    """cd envs
git clone https://github.com/NVlabs/curobo.git
cd curobo
pip install -e . --no-build-isolation
cd ../..""",
    """if [ ! -d envs/curobo ]; then
    echo "Missing envs/curobo. Clone it on the host before running this script." >&2
    exit 1
fi
cd envs/curobo
pip install -e . --no-build-isolation
cd ../..""",
)
path.write_text(text)
print("patched script/_install.sh")
PY

bash script/_install.sh

if ! python -c "import curobo" >/dev/null 2>&1; then
    if [ ! -d envs/curobo ]; then
        echo "Missing envs/curobo. Clone it on the host before running this script." >&2
        exit 1
    fi
    python -m pip install ninja packaging
    python -m pip install -e envs/curobo --no-build-isolation
fi

python - <<'PY'
import curobo
print("curobo import ok", getattr(curobo, "__version__", "unknown"))
PY

if [ -f script/update_embodiment_config_path.py ]; then
    python script/update_embodiment_config_path.py || true
fi

if [ "${DOWNLOAD_ROBOTWIN_ASSETS}" = "true" ]; then
    cd assets
    python _download.py
    python - <<'PY'
from pathlib import Path
from zipfile import ZipFile

for name in ["background_texture.zip", "embodiments.zip", "objects.zip"]:
    path = Path(name)
    print(f"extracting {path} ...")
    if not path.exists():
        raise FileNotFoundError(path)
    with ZipFile(path) as zf:
        zf.extractall(".")
    path.unlink()
    print(f"done {path}")
PY
    cd ..
    python ./script/update_embodiment_config_path.py
else
    echo "Skipping RoboTwin assets. Set DOWNLOAD_ROBOTWIN_ASSETS=true to download them."
fi

python - <<'PY'
import importlib

for name in ["sapien", "mplib", "gymnasium", "transforms3d"]:
    module = importlib.import_module(name)
    print(f"{name} import ok: {getattr(module, '__version__', 'unknown')}")
PY
CONTAINER
