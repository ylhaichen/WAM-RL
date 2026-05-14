import subprocess
import sys


def test_build_grpo_groups_script_entrypoint_can_resolve_repo_imports():
    result = subprocess.run(
        [sys.executable, "tools/build_grpo_groups.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Build mixed GRPO groups" in result.stdout


def test_validate_grpo_dataset_script_entrypoint_can_resolve_repo_imports():
    result = subprocess.run(
        [sys.executable, "tools/validate_grpo_dataset.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Validate GRPO groups" in result.stdout


def test_train_offline_grpo_smoke_help_does_not_require_external_pythonpath():
    result = subprocess.run(
        [sys.executable, "tools/train_offline_grpo_smoke.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "strict-artifact offline GRPO smoke training" in result.stdout
