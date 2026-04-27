#!/bin/bash -l

# Resume/extract RoboTwin asset zip files on a CPU node.
# Stop any interactive extraction process before submitting this job.

#$ -S /bin/bash
#$ -N unpack_robotwin
#$ -cwd
#$ -j y
#$ -o logs/jobs
#$ -l h_rt=4:00:00
#$ -l mem=4G
#$ -pe smp 4
#$ -l tmpfs=20G

set -euo pipefail

if [ -z "${REPO_ROOT:-}" ]; then
    REPO_ROOT="${SGE_O_WORKDIR:-$(pwd)}"
fi
source "${REPO_ROOT}/jobs/myriad/common.sh"

ASSETS_ROOT="${ASSETS_ROOT:-${ROBOTWIN_ROOT}/assets}"
ZIP_NAMES="${ZIP_NAMES:-background_texture.zip embodiments.zip objects.zip}"
PROGRESS_EVERY="${PROGRESS_EVERY:-500}"
LOCK_DIR="${ASSETS_ROOT}/.unpack.lock"

export ZIP_NAMES
export PROGRESS_EVERY

print_job_context
echo "ASSETS_ROOT=${ASSETS_ROOT}"
echo "ZIP_NAMES=${ZIP_NAMES}"

if [ ! -d "${ASSETS_ROOT}" ]; then
    echo "Missing assets directory: ${ASSETS_ROOT}" >&2
    exit 1
fi

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "Another RoboTwin asset unpack appears to be running: ${LOCK_DIR}" >&2
    echo "Remove this directory only if you are sure no unpack process is active." >&2
    exit 1
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

cd "${ASSETS_ROOT}"

python3 - <<'PY'
import os
import sys
from pathlib import Path
from zipfile import ZipFile

zip_names = os.environ["ZIP_NAMES"].split()
progress_every = int(os.environ.get("PROGRESS_EVERY", "500"))

for zip_name in zip_names:
    zip_path = Path(zip_name)
    if not zip_path.exists():
        print(f"missing {zip_path}; skipping", flush=True)
        continue

    print(f"resuming {zip_path} ...", flush=True)
    skipped = 0
    extracted = 0

    with ZipFile(zip_path) as zf:
        members = zf.infolist()
        total = len(members)

        for i, member in enumerate(members, 1):
            target = Path(member.filename)

            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            if target.exists() and target.stat().st_size == member.file_size:
                skipped += 1
            else:
                zf.extract(member, ".")
                extracted += 1

            if i % progress_every == 0 or i == total:
                print(
                    f"{zip_name}: {i}/{total}, skipped={skipped}, extracted={extracted}",
                    flush=True,
                )

    print(f"done {zip_path}: skipped={skipped}, extracted={extracted}", flush=True)

print("asset unpack complete", flush=True)
PY

cd "${ROBOTWIN_ROOT}"

if [ -f ./script/update_embodiment_config_path.py ]; then
    echo "Updating RoboTwin embodiment config paths inside Apptainer ..."
    container_exec_cpu <<'CONTAINER'
set -euo pipefail

if [ -d "${WAN_VA_CONDA_LIBS}/lib" ]; then
    export LD_LIBRARY_PATH="${WAN_VA_CONDA_LIBS}/lib:${LD_LIBRARY_PATH:-}"
fi

source "${WAN_VA_VENV}/bin/activate"
cd "${ROBOTWIN_ROOT}"

python ./script/update_embodiment_config_path.py

python - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["ROBOTWIN_ROOT"])
user = os.environ.get("USER", "")
replacements = [
    (f"/myriadfs/home/{user}", f"/home/{user}"),
]

changed = 0
for path in (root / "assets" / "embodiments").glob("**/*.yml"):
    text = path.read_text()
    new_text = text
    for old, new in replacements:
        new_text = new_text.replace(old, new)
    if new_text != text:
        path.write_text(new_text)
        changed += 1
        print(f"normalized {path}", flush=True)

print(f"path normalization complete: changed={changed}", flush=True)
PY
CONTAINER
fi

ls -ld assets/background_texture assets/embodiments assets/objects
du -sh assets/background_texture assets/embodiments assets/objects
