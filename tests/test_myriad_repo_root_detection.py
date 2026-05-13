from pathlib import Path


def test_myriad_scheduler_jobs_resolve_repo_root_from_script_path():
    scripts = [
        path
        for path in Path("jobs/myriad").glob("*.sh")
        if path.name not in {"common.sh", "30_collect_grouped_rollouts_1gpu.sh"}
    ]

    assert scripts
    for script in scripts:
        text = script.read_text()
        assert 'MYRIAD_JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"' in text, script
        assert 'REPO_ROOT="$(cd "${MYRIAD_JOB_DIR}/../.." && pwd)"' in text, script
        assert 'REPO_ROOT="${SGE_O_WORKDIR:-$(pwd)}"' not in text, script
