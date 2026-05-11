# Portable Environment Setup for WAM-RL

This document is for building a reproducible WAM-RL training/evaluation image
on a fresh compute provider machine. It assumes the provider has no existing
WAM-RL checkout, LingBot-VA checkpoints, RoboTwin install, RoboTwin assets, or
post-training dataset.

The current provider profile is:

- OS: Ubuntu 22.04.4 LTS (Jammy)
- GPU driver: NVIDIA 535.216.03
- CUDA reported by `nvidia-smi`: 12.4
- Test GPU: NVIDIA L20, 45 GB VRAM
- Target training hardware: 8 GPUs
- CPU: 2 x Intel Xeon Gold 6548Y+
- CPU cores: 32 cores per socket, 64 physical cores total

## 0. Provider Deliverables

The compute provider should complete all items in this document and return:

- Final image name/tag or image digest.
- Exact WAM-RL git commit used in the image.
- Exact RoboTwin commit used in the image.
- Exact CuRobo commit/tag used in the image.
- Output logs from all smoke tests in Section 8.
- Final paths for checkpoints, dataset, RoboTwin assets, and result storage.
- Final 8-GPU hardware profile:
  - GPU model and VRAM per GPU.
  - Driver version from the production 8-GPU node.
  - Runtime launcher type, such as Docker, Kubernetes, Slurm, or internal
    launcher.
- Whether production jobs have internet access or must use internal mirrors.

The image is not ready for training until all smoke tests pass.

## 1. Build Goal

Build one image that can run:

- LingBot-VA model import and server inference.
- RoboTwin evaluation clients.
- Selected-task baseline evaluation.
- PEFT / SFT training smoke.
- Future grouped-rollout RL training.

Use the Myriad-specific `docs/SERVER_SETUP.md` only as a reference. That file
uses Apptainer and SGE-specific paths. This document is scheduler-agnostic and
is intended for a Docker/OCI-style company image.

## 1.1 Required Repositories

The provider must clone this project repository. The repo contains model code,
training code, evaluation bridge code, job references, tools, and these
environment files.

Recommended checkout:

```bash
export REPO_ROOT=/workspace/WAM-RL
git clone https://github.com/ylhaichen/WAM-RL.git "$REPO_ROOT"
cd "$REPO_ROOT"
git checkout main
git rev-parse HEAD
```

If the company network cannot access GitHub directly, mirror this repository
into the company Git service and keep the commit hash visible in the image
build log.

The provider must also clone RoboTwin separately:

```bash
export WAM_ROOT=/data/wam-rl
export ROBOTWIN_ROOT=$WAM_ROOT/RoboTwin
git clone https://github.com/RoboTwin-Platform/RoboTwin.git "$ROBOTWIN_ROOT"
cd "$ROBOTWIN_ROOT"
git checkout 2eeec322
git rev-parse HEAD
```

RoboTwin is intentionally not vendored inside WAM-RL because it has large assets
and simulator-specific dependencies.

## 2. Base Image

Recommended base image:

```bash
nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
```

Minimum system packages:

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential \
  git \
  git-lfs \
  curl \
  wget \
  ca-certificates \
  python3 \
  python3-dev \
  python3-venv \
  python3-pip \
  ninja-build \
  cmake \
  pkg-config \
  ffmpeg \
  unzip \
  libgl1 \
  libglib2.0-0 \
  libx11-6 \
  libxext6 \
  libxrender1 \
  libxi6 \
  libxrandr2 \
  libxinerama1 \
  libxcursor1 \
  libsm6 \
  libice6 \
  libegl1 \
  libglvnd0 \
  libvulkan1 \
  vulkan-tools
```

Recommended Python version:

```text
Python 3.10 or 3.11
```

Do not use Python 3.12 for the first image unless RoboTwin/SAPIEN/CuRobo have
already been validated with it.

Minimal Dockerfile-style skeleton:

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git git-lfs curl wget ca-certificates \
    python3 python3-dev python3-venv python3-pip \
    ninja-build cmake pkg-config ffmpeg unzip \
    libgl1 libglib2.0-0 libx11-6 libxext6 libxrender1 libxi6 \
    libxrandr2 libxinerama1 libxcursor1 libsm6 libice6 \
    libegl1 libglvnd0 libvulkan1 vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/wam-rl-venv
ENV PATH=/opt/wam-rl-venv/bin:$PATH

# Copy or clone WAM-RL into /workspace/WAM-RL before running the Python install
# steps from Section 4.
```

## 3. Repository Layout

Use these paths inside the image or mounted training workspace:

```bash
export WAM_ROOT=/data/wam-rl
export REPO_ROOT=/workspace/WAM-RL
export WAN_VA_MODEL_PATH=$WAM_ROOT/checkpoints/lingbot-va-posttrain-robotwin
export WAN_VA_BASE_MODEL_PATH=$WAM_ROOT/checkpoints/lingbot-va-base
export WAN_VA_DATASET_PATH=$WAM_ROOT/datasets/robotwin-clean-and-aug-lerobot
export WAN_VA_EMPTY_EMB_PATH=$WAN_VA_DATASET_PATH/empty_emb.pt
export ROBOTWIN_ROOT=$WAM_ROOT/RoboTwin
export WAN_VA_ROBOTWIN_ROOT=$ROBOTWIN_ROOT
export RESULTS_ROOT=$WAM_ROOT/results
export TMPDIR=$WAM_ROOT/tmp
export TORCH_EXTENSIONS_DIR=$WAM_ROOT/torch_extensions
export CUDA_CACHE_PATH=$WAM_ROOT/cuda_cache
export TOKENIZERS_PARALLELISM=false
export WAN_VA_ENABLE_WANDB=false
```

Create directories:

```bash
mkdir -p \
  "$WAM_ROOT" \
  "$WAN_VA_MODEL_PATH" \
  "$WAN_VA_BASE_MODEL_PATH" \
  "$WAN_VA_DATASET_PATH" \
  "$RESULTS_ROOT" \
  "$TMPDIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$CUDA_CACHE_PATH"
```

Recommended persistent storage size:

- Checkpoints: at least 100 GB.
- RoboTwin assets: at least 100 GB.
- Post-training dataset: at least 500 GB.
- Results, rollouts, videos, and logs: at least 500 GB for iterative RL.
- Temporary build/cache space: at least 100 GB, preferably local SSD.

## 4. Python Environment

Create and activate a virtual environment:

```bash
python3 -m venv /opt/wam-rl-venv
source /opt/wam-rl-venv/bin/activate
python -m pip install -U pip setuptools wheel
```

Install PyTorch first. The CUDA wheel index must be explicit:

```bash
python -m pip install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

Why PyTorch 2.6.0:

- It has official CUDA 12.4 wheels.
- It is new enough for the `torch.nn.attention.flex_attention` import path used
  by LingBot-VA.
- It is a safer first target for the provider's CUDA 12.4 image than the
  Myriad-specific `torch==2.9.0` setup.

Install WAM-RL base dependencies:

```bash
cd "$REPO_ROOT"
python -m pip install -r requirements-cu124.txt
python -m pip install -e . --no-deps
```

Do not install WAM-RL with unrestricted dependency resolution before PyTorch is
installed. The CUDA wheel must be pinned first so the image does not silently
fall back to a CPU wheel or a mismatched CUDA build.

If the company mirror cannot access `https://download.pytorch.org/whl/cu124`,
the provider should mirror the exact `torch/torchvision/torchaudio` CUDA 12.4
wheels internally and keep the versions above unchanged for the first image.

## 5. RoboTwin Install

Clone RoboTwin:

```bash
mkdir -p "$WAM_ROOT"
git clone https://github.com/RoboTwin-Platform/RoboTwin.git "$ROBOTWIN_ROOT"
cd "$ROBOTWIN_ROOT"
git checkout 2eeec322
```

Install RoboTwin Python dependencies:

```bash
source /opt/wam-rl-venv/bin/activate
python -m pip install -r "$REPO_ROOT/requirements-robotwin.txt"
```

If `lerobot==0.3.3` or RoboTwin dependencies try to downgrade core packages,
install the conflicting dependency with `--no-deps` and keep the already pinned
PyTorch/Numpy stack unchanged unless there is a concrete import failure.

Install PyTorch3D from source:

```bash
MAX_JOBS=8 python -m pip install \
  "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
  --no-build-isolation
```

Install CuRobo V1-compatible source:

```bash
cd "$ROBOTWIN_ROOT"
mkdir -p envs
git clone https://github.com/NVlabs/curobo.git envs/curobo
cd envs/curobo
git checkout v0.7.8
MAX_JOBS=8 python -m pip install -e . --no-build-isolation
```

The `v0.7.8` pin matters because the RoboTwin commit used here imports V1
CuRobo paths such as `curobo.types.math.Pose`.

Validate simulator imports:

```bash
python - <<'PY'
import torch
import sapien
import mplib
import gymnasium
import transforms3d
import curobo
from curobo.types.math import Pose

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("curobo", getattr(curobo, "__version__", "unknown"), Pose.__name__)
print("simulator imports ok")
PY
```

If SAPIEN, OpenCV, or Vulkan reports missing display/runtime libraries, fix the
image by adding the missing Ubuntu runtime packages rather than relying on
interactive manual exports.

## 6. Assets And Checkpoints

Required assets for baseline RoboTwin evaluation:

- LingBot-VA RoboTwin checkpoint:
  `robbyant/lingbot-va-posttrain-robotwin`
- RoboTwin repository and assets.

Required assets for SFT/RL training:

- Base LingBot-VA checkpoint:
  `robbyant/lingbot-va-base`
- RoboTwin post-training dataset:
  `robbyant/robotwin-clean-and-aug-lerobot`

Download with Hugging Face CLI if external access is available:

```bash
source /opt/wam-rl-venv/bin/activate

hf download robbyant/lingbot-va-posttrain-robotwin \
  --local-dir "$WAN_VA_MODEL_PATH"

hf download robbyant/lingbot-va-base \
  --local-dir "$WAN_VA_BASE_MODEL_PATH"

hf download robbyant/robotwin-clean-and-aug-lerobot \
  --repo-type dataset \
  --local-dir "$WAN_VA_DATASET_PATH"
```

If external access is unavailable, mirror these repositories into the company
artifact store and preserve the same directory contents under the paths above.

The provider should treat these as required artifact inputs:

```text
robbyant/lingbot-va-posttrain-robotwin
robbyant/lingbot-va-base
robbyant/robotwin-clean-and-aug-lerobot
RoboTwin assets: background_texture.zip, embodiments.zip, objects.zip
```

The post-training dataset is about 406 GB. It is not required for pretrained
RoboTwin evaluation, but it is required for PEFT/SFT and future RL training.

Set attention modes:

```bash
cd "$REPO_ROOT"
python tools/set_attn_mode.py "$WAN_VA_MODEL_PATH" torch
python tools/set_attn_mode.py "$WAN_VA_BASE_MODEL_PATH" flex
```

RoboTwin assets:

```bash
cd "$ROBOTWIN_ROOT/assets"
python _download.py
unzip background_texture.zip
unzip embodiments.zip
unzip objects.zip
rm -f background_texture.zip embodiments.zip objects.zip

cd "$ROBOTWIN_ROOT"
python ./script/update_embodiment_config_path.py
```

If the company network blocks RoboTwin asset downloads, mirror these asset zip
files internally and unpack them into `$ROBOTWIN_ROOT/assets`.

## 7. Runtime Environment

Set these variables in the job launcher before running evaluation or training:

```bash
source /opt/wam-rl-venv/bin/activate

export WAM_ROOT=/data/wam-rl
export REPO_ROOT=/workspace/WAM-RL
export WAN_VA_MODEL_PATH=$WAM_ROOT/checkpoints/lingbot-va-posttrain-robotwin
export WAN_VA_BASE_MODEL_PATH=$WAM_ROOT/checkpoints/lingbot-va-base
export WAN_VA_DATASET_PATH=$WAM_ROOT/datasets/robotwin-clean-and-aug-lerobot
export WAN_VA_EMPTY_EMB_PATH=$WAN_VA_DATASET_PATH/empty_emb.pt
export ROBOTWIN_ROOT=$WAM_ROOT/RoboTwin
export WAN_VA_ROBOTWIN_ROOT=$ROBOTWIN_ROOT
export TOKENIZERS_PARALLELISM=false
export WAN_VA_ENABLE_WANDB=false
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.0.0
export TMPDIR=$WAM_ROOT/tmp
export TORCH_EXTENSIONS_DIR=$WAM_ROOT/torch_extensions
export CUDA_CACHE_PATH=$WAM_ROOT/cuda_cache
```

For 8-GPU training/evaluation, the launcher must also set:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=8
```

Distributed training should be launched with `torchrun` or the provider's
scheduler wrapper. Example:

```bash
torchrun \
  --nproc_per_node=8 \
  --master_port="${MASTER_PORT:-29500}" \
  wan_va/train.py \
  --config robotwin_peft_train
```

The exact training command may change as RL training code is added. The image
should still pass the smoke tests below before any long job is submitted.

## 8. Smoke Tests

Run these inside the final image.

All smoke tests should be run on the same runtime type that will be used for
real jobs. If production uses an 8-GPU launcher, also run the basic GPU check
on the production 8-GPU node.

Basic GPU check:

```bash
python - <<'PY'
import torch

print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("device count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0", torch.cuda.get_device_name(0))
PY
```

LingBot-VA import check:

```bash
cd "$REPO_ROOT"
python - <<'PY'
from wan_va.modules.model import WanAttention
from wan_va.configs import VA_CONFIGS

print("WanAttention", WanAttention.__name__)
print("robotwin_peft_train", VA_CONFIGS["robotwin_peft_train"].trainable_mode)
PY
```

Project setup check:

```bash
cd "$REPO_ROOT"
python tools/check_setup.py \
  --model-path "$WAN_VA_MODEL_PATH" \
  --dataset-path "$WAN_VA_DATASET_PATH" \
  --robotwin-root "$ROBOTWIN_ROOT"
```

RoboTwin render/import check:

```bash
cd "$REPO_ROOT"
python evaluation/robotwin/test_render.py
```

If `test_render.py` requires a GPU display stack, run it on a GPU node with
NVIDIA container runtime enabled.

8-GPU distributed launch check:

```bash
cd "$REPO_ROOT"
cat > /tmp/wam_dist_check.py <<'PY'
import os
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
x = torch.ones(1, device="cuda")
dist.all_reduce(x)
if dist.get_rank() == 0:
    print("distributed ok", "world_size", dist.get_world_size(), "sum", x.item())
dist.destroy_process_group()
PY

torchrun --standalone --nproc_per_node=8 /tmp/wam_dist_check.py
```

If the provider's launcher wraps `torchrun`, replace this with their official
8-GPU no-op distributed check and include the command in the handoff notes.

## 9. Expected First Evaluation

After the image and assets are validated, run a 1-task smoke evaluation first:

```bash
cd "$REPO_ROOT"
export RESULTS_ROOT=$WAM_ROOT/results/smoke_1gpu
mkdir -p "$RESULTS_ROOT"

# Use the provider's launcher to start one LingBot-VA server and one RoboTwin
# client. The Myriad reference is jobs/myriad/10_eval_smoke_1gpu.sh, but the
# provider should translate it to their scheduler.
```

Then run the selected baseline:

```bash
TASK_NAMES="hanging_mug turn_switch open_microwave put_bottles_dustbin move_stapler_pad press_stapler blocks_ranking_rgb place_dual_shoes place_fan put_object_cabinet stack_bowls_three adjust_bottle click_bell"
TEST_NUM=50
SEED=1
```

Expected selected-task rollout count:

```text
13 tasks * 50 episodes = 650 result records
```

Summarize results:

```bash
python tools/summarize_robotwin_results.py "$RESULTS_ROOT" --sort rate
```

Provider handoff should include:

- The exact command used to start LingBot-VA server processes.
- The exact command used to start RoboTwin clients.
- Where client logs are written.
- Where `metrics/*/res.json` is written.
- Where `rollouts/*/*.json` and action tensors are written.
- How to stop a stuck evaluation job.

The Myriad scripts under `jobs/myriad/` are references. The provider should
translate them to the production scheduler rather than requiring Myriad/SGE.

## 10. Information Still Needed From The Provider

Ask the provider for:

- Final 8-GPU model name and VRAM per GPU.
- Final 8-GPU driver version, not only the one-GPU image-build notebook.
- Whether the production runtime uses Docker, Kubernetes, Slurm, or a custom
  launcher.
- Whether the container runs as root or non-root.
- Persistent storage path and capacity for checkpoints, RoboTwin assets,
  datasets, and results.
- Whether the production job has internet access.
- Internal package mirror URLs for PyPI, PyTorch wheels, GitHub source mirrors,
  Hugging Face assets, and RoboTwin assets.
- Maximum shared memory, `/dev/shm`, and local temporary storage.
- Whether outbound websocket connections between local server/client processes
  are allowed inside a job.

## 11. Known Risks

- Driver `535.216.03` with CUDA 12.4 should be validated with the exact PyTorch
  CUDA 12.4 wheel. If `torch.cuda.is_available()` fails, the provider should
  either update the driver or provide a known-good PyTorch/CUDA wheel pair.
- `flash-attn` is optional. Do not install it for the first image unless the
  team explicitly wants `attn_mode=flashattn`.
- `pytorch3d` and `curobo` compile native extensions. Build them during image
  creation, not at runtime, and keep build logs.
- The 406 GB post-training dataset is not needed for pretrained evaluation but
  is needed for SFT/RL training.
- RoboTwin asset paths can become absolute-path-sensitive. Always run
  `python ./script/update_embodiment_config_path.py` after moving or unpacking
  assets.
