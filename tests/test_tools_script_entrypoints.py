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


def test_summarize_grpo_groups_help_does_not_require_external_pythonpath():
    result = subprocess.run(
        [sys.executable, "tools/summarize_grpo_groups.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Summarize GRPO group JSONL files" in result.stdout


def test_summarize_robotwin_results_help_does_not_require_external_pythonpath():
    result = subprocess.run(
        [sys.executable, "tools/summarize_robotwin_results.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Summarize RoboTwin res.json files" in result.stdout
    assert "--episodes-csv" in result.stdout


def test_compare_robotwin_eval_episodes_help_does_not_require_external_pythonpath():
    result = subprocess.run(
        [sys.executable, "tools/compare_robotwin_eval_episodes.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Compare RoboTwin eval runs on matched episode keys" in result.stdout
    assert "--run" in result.stdout


def test_subset_grpo_groups_help_does_not_require_external_pythonpath():
    result = subprocess.run(
        [sys.executable, "tools/subset_grpo_groups.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Create small GRPO group JSONL subsets" in result.stdout
    assert "--samples-per-reward" in result.stdout


def test_materialize_grpo_artifacts_help_does_not_require_external_pythonpath():
    result = subprocess.run(
        [sys.executable, "tools/materialize_grpo_artifacts.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Materialize referenced GRPO artifacts" in result.stdout
    assert "--include-replay-context" in result.stdout


def test_audit_grpo_artifact_storage_help_does_not_require_external_pythonpath():
    result = subprocess.run(
        [sys.executable, "tools/audit_grpo_artifact_storage.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Audit filesystem footprint of GRPO artifact references" in result.stdout
    assert "--materialize-manifest" in result.stdout


def test_train_actor_replay_grpo_help_does_not_require_external_pythonpath():
    result = subprocess.run(
        [sys.executable, "tools/train_actor_replay_grpo.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "real LingBot actor replay GRPO training" in result.stdout


def test_summarize_actor_replay_training_help_does_not_require_external_pythonpath():
    result = subprocess.run(
        [sys.executable, "tools/summarize_actor_replay_training.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Summarize actor replay GRPO training output directories" in result.stdout
    assert "--out-markdown" in result.stdout


def test_diagnose_actor_replay_help_does_not_require_external_pythonpath():
    result = subprocess.run(
        [sys.executable, "tools/diagnose_actor_replay.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Diagnose real actor replay" in result.stdout
