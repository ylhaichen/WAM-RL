# Server Setup for WAM-RL on UCL Myriad

This server uses an old host `glibc` (`2.17` in our test), so native `pip`
installation of current PyTorch wheels fails. Use the PyTorch Apptainer image
and submit GPU work through SGE job scripts.

UCL Myriad uses `qsub`/`qrsh`. GPU jobs request GPUs with `#$ -l gpu=<N>`.
The L-type nodes have 4x A100 40GB GPUs; request them with `#$ -ac allow=L`.

References:

- <https://www.rc.ucl.ac.uk/docs/Clusters/Myriad/>
- <https://www.rc.ucl.ac.uk/docs/Example_Jobscripts/>

## 1. Build the Apptainer image

Run this once on the login node:

```bash
mkdir -p "$HOME/containers"
apptainer pull "$HOME/containers/pytorch-2.9.0-cu126.sif" \
  docker://pytorch/pytorch:2.9.0-cuda12.6-cudnn9-devel
```

CPU-only sanity check on the login node:

```bash
apptainer exec \
  --bind "$HOME:$HOME" \
  --bind /shared:/shared \
  "$HOME/containers/pytorch-2.9.0-cu126.sif" \
  python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

On the login node, `torch.cuda.is_available()` is expected to be `False`.

## 2. Batch-first workflow

Submit jobs from the repo root:

```bash
cd "$HOME/Scratch/WAM-RL"
mkdir -p logs/jobs
```

Build the container-side Python venv:

```bash
qsub jobs/myriad/00_install_container_env.sh
```

Download checkpoints. The 406G post-training dataset is skipped by default:

```bash
qsub jobs/myriad/01_download_assets.sh
```

Install RoboTwin simulator and download RoboTwin assets:

```bash
qsub jobs/myriad/02_install_robotwin.sh
```

This job pins CuRobo to `v0.7.8`, because current CuRobo V2 changed its public
API and RoboTwin commit `2eeec322` imports V1 paths such as
`curobo.types.math.Pose`.

The job script extracts RoboTwin asset zip files with Python `zipfile`, because
the PyTorch Apptainer image may not include a system `unzip` command.

If asset extraction is interrupted after the zip files have downloaded, resume
it on a CPU node:

```bash
qsub jobs/myriad/03_unpack_robotwin_assets.sh
```

Stop any manual extraction first. The unpack job uses
`$ROBOTWIN_ROOT/assets/.unpack.lock` to avoid two processes writing the same
asset directory. It also runs RoboTwin's embodiment path updater inside the
Apptainer container so generated CuRobo config files use `/home/...` paths that
are visible during evaluation.

If an older unpack job wrote `/myriadfs/home/...` paths into RoboTwin YAML files,
fix them inside the running Apptainer shell before launching evaluation:

```bash
cd "$ROBOTWIN_ROOT"
python ./script/update_embodiment_config_path.py

python - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["ROBOTWIN_ROOT"])
old = f"/myriadfs/home/{os.environ['USER']}"
new = f"/home/{os.environ['USER']}"

for path in (root / "assets" / "embodiments").glob("**/*.yml"):
    text = path.read_text()
    if old in text:
        path.write_text(text.replace(old, new))
        print("fixed", path)
PY

grep -R "/myriadfs/home" -n assets/embodiments/*.yml assets/embodiments/*/*.yml || true
```

Download the full post-training dataset only when you are ready to run SFT/RL
training:

```bash
qsub -v DOWNLOAD_DATASET=true jobs/myriad/01_download_assets.sh
```

Run a one-GPU smoke evaluation:

```bash
qsub jobs/myriad/10_eval_smoke_1gpu.sh
```

Run the four-GPU pilot evaluation:

```bash
qsub jobs/myriad/11_eval_pilot_4gpu.sh
```

Run a tiny four-GPU training wiring check:

```bash
qsub jobs/myriad/20_train_tiny_4gpu.sh
```

Useful scheduler commands:

```bash
qstat
tail -f logs/jobs/wam_smoke.o<JOB_ID>
qdel <JOB_ID>
```

## 3. Default paths and overrides

The job scripts default to:

```bash
export WAM_ROOT="$HOME/Scratch/wam-rl"
export SIF="$HOME/containers/pytorch-2.9.0-cu126.sif"
export WAN_VA_VENV="$WAM_ROOT/venvs/wam-rl-container"
export WAN_VA_CONDA_LIBS="$WAM_ROOT/conda-libs"
export WAN_VA_MODEL_PATH="$WAM_ROOT/checkpoints/lingbot-va-posttrain-robotwin"
export WAN_VA_BASE_MODEL_PATH="$WAM_ROOT/checkpoints/lingbot-va-base"
export WAN_VA_DATASET_PATH="$WAM_ROOT/datasets/robotwin-clean-and-aug-lerobot"
export ROBOTWIN_ROOT="$WAM_ROOT/RoboTwin"
```

Override paths at submission time with `qsub -v`:

```bash
qsub -v WAM_ROOT="$HOME/Scratch/wam-rl",ROBOTWIN_ROOT="$HOME/Scratch/RoboTwin" \
  jobs/myriad/10_eval_smoke_1gpu.sh
```

For quick smoke tests:

```bash
qsub -v TEST_NUM=1,TASK_NAME=adjust_bottle \
  jobs/myriad/10_eval_smoke_1gpu.sh
```

For the four-task pilot:

```bash
qsub -v TASK_NAMES="adjust_bottle place_mouse_pad stack_blocks_two click_bell",TEST_NUM=10 \
  jobs/myriad/11_eval_pilot_4gpu.sh
```

## 4. Interactive one-GPU debug session

Use interactive sessions only for short debugging because they can wait in the
queue for a long time:

```bash
qrsh \
  -l gpu=1 \
  -ac allow=L \
  -l h_rt=4:00:00 \
  -l mem=4G \
  -pe smp 8 \
  -l tmpfs=100G
```

Inside the allocated node:

```bash
hostname
nvidia-smi

cd "$HOME/Scratch/WAM-RL"
apptainer shell --nv \
  --bind "$HOME:$HOME" \
  --bind /shared:/shared \
  "$HOME/containers/pytorch-2.9.0-cu126.sif"
```

Inside `Apptainer>`:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

If RoboTwin/SAPIEN/OpenCV reports missing shared libraries such as
`libX11.so.6`, `libgthread-2.0.so.0`, or `libGL.so.1`, install runtime
libraries into a user-writable conda prefix and add it to `LD_LIBRARY_PATH`:

```bash
export WAM_ROOT="$HOME/Scratch/wam-rl"
export WAN_VA_CONDA_LIBS="$WAM_ROOT/conda-libs"

if [ -d "$WAN_VA_CONDA_LIBS/conda-meta" ]; then
  /opt/conda/bin/conda install -y -p "$WAN_VA_CONDA_LIBS" -c conda-forge \
    xorg-libx11 xorg-libxext xorg-libxrender xorg-libxi xorg-libxrandr \
    xorg-libxinerama xorg-libxcursor xorg-libxfixes xorg-libsm xorg-libice \
    libxcb libglvnd glib
else
  /opt/conda/bin/conda create -y -p "$WAN_VA_CONDA_LIBS" -c conda-forge \
    xorg-libx11 xorg-libxext xorg-libxrender xorg-libxi xorg-libxrandr \
    xorg-libxinerama xorg-libxcursor xorg-libxfixes xorg-libsm xorg-libice \
    libxcb libglvnd glib
fi

export LD_LIBRARY_PATH="$WAN_VA_CONDA_LIBS/lib:${LD_LIBRARY_PATH:-}"
```

If SAPIEN then reports missing GLVND/Vulkan ICD paths such as
`/usr/share/glvnd/egl_vendor.d`, exit the container, create user-writable ICD
files, and re-enter Apptainer with extra binds:

```bash
exit

export WAM_ROOT="$HOME/Scratch/wam-rl"
export REPO_ROOT="$HOME/Scratch/WAM-RL"
export WAN_VA_EGL_VENDOR_DIR="$WAM_ROOT/glvnd/egl_vendor.d"
export WAN_VA_VULKAN_ICD_DIR="$WAM_ROOT/vulkan/icd.d"

mkdir -p "$WAN_VA_EGL_VENDOR_DIR" "$WAN_VA_VULKAN_ICD_DIR"
cat > "$WAN_VA_EGL_VENDOR_DIR/10_nvidia.json" <<'JSON'
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "libEGL_nvidia.so.0"
  }
}
JSON
cat > "$WAN_VA_VULKAN_ICD_DIR/nvidia_icd.json" <<'JSON'
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "libGLX_nvidia.so.0",
    "api_version": "1.3.204"
  }
}
JSON

apptainer shell --nv \
  --bind "$HOME:$HOME" \
  --bind /shared:/shared \
  --bind "$WAN_VA_EGL_VENDOR_DIR:/usr/share/glvnd/egl_vendor.d" \
  --bind "$WAN_VA_VULKAN_ICD_DIR:/usr/share/vulkan/icd.d" \
  --bind "$WAN_VA_VULKAN_ICD_DIR:/etc/vulkan/icd.d" \
  "$HOME/containers/pytorch-2.9.0-cu126.sif"
```

Inside `Apptainer>`:

```bash
export WAM_ROOT="$HOME/Scratch/wam-rl"
export WAN_VA_CONDA_LIBS="$WAM_ROOT/conda-libs"
export LD_LIBRARY_PATH="$WAN_VA_CONDA_LIBS/lib:${LD_LIBRARY_PATH:-}"
export __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
export VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json"
export SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO="0.0.0"
export TMPDIR="$WAM_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export TORCH_EXTENSIONS_DIR="$WAM_ROOT/torch_extensions"
export CUDA_CACHE_PATH="$WAM_ROOT/cuda_cache"
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH"
```

The `SETUPTOOLS_SCM_*` variables avoid a CuRobo runtime version lookup failure
inside the PyTorch container, which does not include `git`.
The temporary/cache variables avoid CUDA extension builds writing to stale SGE
paths such as `/tmpdir/job/<old-job>.undefined`.

## 5. Model and dataset assets

Recommended baseline checkpoint for RoboTwin evaluation:

```bash
hf download robbyant/lingbot-va-posttrain-robotwin \
  --local-dir "$WAN_VA_MODEL_PATH"
```

Base checkpoint for future SFT/RL experiments:

```bash
hf download robbyant/lingbot-va-base \
  --local-dir "$WAN_VA_BASE_MODEL_PATH"
```

Post-training dataset with WAN 2.2 latents:

```bash
hf download robbyant/robotwin-clean-and-aug-lerobot \
  --repo-type dataset \
  --local-dir "$WAN_VA_DATASET_PATH"
```

This dataset is about 406G. It is not needed for pretrained RoboTwin
evaluation; it is only needed for post-training, BC anchoring, or dataset-based
training smoke tests.

The model repos are Apache-2.0. The dataset page lists CC BY-NC-SA 4.0, so
check that this matches the project requirements before using it beyond
research/coursework.

## 6. Prepare RoboTwin

Follow the RoboTwin install described in `README.md`, then export the root:

```bash
export ROBOTWIN_ROOT="$HOME/Scratch/wam-rl/RoboTwin"
export WAN_VA_ROBOTWIN_ROOT="$ROBOTWIN_ROOT"
```

The evaluation job scripts assume RoboTwin is already installed at
`ROBOTWIN_ROOT`.

## 7. Attention mode

LingBot-VA reads `attn_mode` from the checkpoint config.

For inference/evaluation:

```bash
python tools/set_attn_mode.py "$WAN_VA_MODEL_PATH" torch
```

For training:

```bash
python tools/set_attn_mode.py "$WAN_VA_BASE_MODEL_PATH" flex
```

`flash-attn` is optional. Install it only if you explicitly want
`attn_mode=flashattn`.

## 8. Sanity checks

```bash
python tools/check_setup.py \
  --model-path "$WAN_VA_MODEL_PATH" \
  --dataset-path "$WAN_VA_DATASET_PATH" \
  --robotwin-root "$ROBOTWIN_ROOT"
```

## 9. Pilot task set

For the first 4x A100 40GB pilot, use:

- `adjust_bottle`
- `place_mouse_pad`
- `stack_blocks_two`
- `click_bell`

Rollout metadata is written under `RESULTS_ROOT/rollouts`.
