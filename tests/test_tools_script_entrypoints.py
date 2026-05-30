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


def test_summarize_actor_eval_pair_help_does_not_require_external_pythonpath():
    result = subprocess.run(
        [sys.executable, "tools/summarize_actor_eval_pair.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Summarize a matched baseline-vs-actor RoboTwin eval pair" in result.stdout
    assert "--baseline" in result.stdout
    assert "--min-matched-episodes" in result.stdout


def test_summarize_robotwin_repeatability_help_does_not_require_external_pythonpath():
    result = subprocess.run(
        [sys.executable, "tools/summarize_robotwin_repeatability.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Summarize RoboTwin eval repeatability across matched episodes" in result.stdout
    assert "--run" in result.stdout


def test_gate_actor_eval_promotion_help_does_not_require_external_pythonpath():
    result = subprocess.run(
        [sys.executable, "tools/gate_actor_eval_promotion.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Gate actor-replay eval promotion" in result.stdout
    assert "--baseline-repeatability" in result.stdout


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
    assert "--max-replay-context-gb" in result.stdout


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
    assert "--dry-run" in result.stdout


def test_merge_grpo_groups_help_does_not_require_external_pythonpath():
    result = subprocess.run(
        [sys.executable, "tools/merge_grpo_groups.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Merge GRPO group JSONL files" in result.stdout
    assert "--allow-duplicate-group-ids" in result.stdout


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
    assert "--inspect-replay-contexts" in result.stdout
    assert "--print-summary" in result.stdout


def test_plan_myriad_storage_cleanup_help_does_not_require_external_pythonpath():
    result = subprocess.run(
        [sys.executable, "tools/plan_myriad_storage_cleanup.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Plan non-destructive WAM-RL Myriad storage cleanup candidates" in result.stdout
    assert "--out-markdown" in result.stdout


def test_plan_replay_context_collection_help_does_not_require_external_pythonpath():
    result = subprocess.run(
        [sys.executable, "tools/plan_replay_context_collection.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Plan bounded replay-context rollout collection storage" in result.stdout
    assert "--storage-budget-mode" in result.stdout
    assert "--format" in result.stdout


def test_estimate_group_mixing_help_does_not_require_external_pythonpath():
    result = subprocess.run(
        [sys.executable, "tools/estimate_group_mixing.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Estimate mixed-group probabilities" in result.stdout
    assert "--group-sizes" in result.stdout


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
    assert "--out-csv" in result.stdout
    assert "--out-markdown" in result.stdout


def test_inspect_actor_replay_checkpoint_help_does_not_require_external_pythonpath():
    result = subprocess.run(
        [sys.executable, "tools/inspect_actor_replay_checkpoint.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Inspect real actor replay checkpoint tensor statistics" in result.stdout
    assert "--reference" in result.stdout


def test_inspect_grpo_replay_context_help_does_not_require_external_pythonpath():
    result = subprocess.run(
        [sys.executable, "tools/inspect_grpo_replay_context.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Inspect tensor storage inside a strict GRPO replay-context artifact" in result.stdout
    assert "--top-k" in result.stdout
    assert "--metadata-only" in result.stdout
    assert "--print-summary" in result.stdout


def test_summarize_grpo_replay_contexts_help_does_not_require_external_pythonpath():
    result = subprocess.run(
        [sys.executable, "tools/summarize_grpo_replay_contexts.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Summarize replay-context storage and tensor metadata" in result.stdout
    assert "--inspect-artifacts" in result.stdout
    assert "--inspect-context-tensors" in result.stdout
    assert "--print-summary" in result.stdout


def test_diagnose_actor_replay_help_does_not_require_external_pythonpath():
    result = subprocess.run(
        [sys.executable, "tools/diagnose_actor_replay.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Diagnose real actor replay" in result.stdout


def test_report_grpo_run_status_help_does_not_require_external_pythonpath():
    result = subprocess.run(
        [sys.executable, "tools/report_grpo_run_status.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Report WAM-RL GRPO job and result status" in result.stdout
    assert "--inspect-files" in result.stdout
