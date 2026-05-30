import subprocess
import sys

import pytest

from tools.plan_replay_context_collection import (
    format_shell_summary,
    plan_replay_context_collection,
)


def test_plan_replay_context_collection_attempt_budget():
    report = plan_replay_context_collection(
        task_names="move_stapler_pad turn_switch",
        group_size=4,
        groups_per_task=1,
        group_max_attempts=3,
        capture_max_chunks=1,
        save_replay_context=True,
        replay_context_estimate_gb=4.0,
        storage_budget_mode="attempt",
    )

    assert report["task_count"] == 2
    assert report["accepted_contexts"] == 8
    assert report["attempt_budget_contexts"] == 24
    assert report["accepted_estimate_gb"] == 32.0
    assert report["attempt_budget_estimate_gb"] == 96.0
    assert report["storage_budget_estimate_gb"] == 96.0


def test_plan_replay_context_collection_accepted_budget():
    report = plan_replay_context_collection(
        task_names="move_stapler_pad",
        group_size=4,
        groups_per_task=1,
        group_max_attempts=3,
        capture_max_chunks=1,
        save_replay_context=True,
        replay_context_estimate_gb=4.0,
        storage_budget_mode="accepted",
    )

    assert report["accepted_estimate_gb"] == 16.0
    assert report["attempt_budget_estimate_gb"] == 48.0
    assert report["storage_budget_estimate_gb"] == 16.0


def test_plan_replay_context_collection_rejects_unbounded_capture_without_override():
    with pytest.raises(ValueError, match="STRICT_GRPO_CAPTURE_MAX_CHUNKS"):
        plan_replay_context_collection(
            task_names="move_stapler_pad",
            group_size=4,
            groups_per_task=1,
            group_max_attempts=1,
            capture_max_chunks=0,
            save_replay_context=True,
            replay_context_estimate_gb=4.0,
            storage_budget_mode="attempt",
        )


def test_plan_replay_context_collection_reports_headroom(tmp_path):
    report = plan_replay_context_collection(
        task_names="move_stapler_pad",
        group_size=4,
        groups_per_task=1,
        group_max_attempts=1,
        capture_max_chunks=1,
        save_replay_context=True,
        replay_context_estimate_gb=1_000_000.0,
        storage_budget_mode="attempt",
        check_scratch_headroom=True,
        scratch_path=tmp_path,
        min_scratch_headroom_gb=1.0,
    )

    assert report["scratch_path_missing"] is False
    assert report["headroom_ok"] is False
    assert report["required_for_budget_plus_headroom_gb"] == 4_000_001.0


def test_format_shell_summary_matches_submitter_output():
    report = plan_replay_context_collection(
        task_names="move_stapler_pad",
        group_size=4,
        groups_per_task=1,
        group_max_attempts=3,
        capture_max_chunks=1,
        save_replay_context=True,
        replay_context_estimate_gb=4.0,
        storage_budget_mode="attempt",
    )

    text = format_shell_summary(report)

    assert "Replay-context storage estimate" in text
    assert "accepted_estimate_gb=16.00" in text
    assert "attempt_budget_estimate_gb=48.00" in text
    assert "storage_budget_estimate_gb=48.00" in text


def test_plan_replay_context_collection_cli_shell_format(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "tools/plan_replay_context_collection.py",
            "--task-names",
            "move_stapler_pad",
            "--group-size",
            "4",
            "--groups-per-task",
            "1",
            "--group-max-attempts",
            "3",
            "--capture-max-chunks",
            "1",
            "--save-replay-context",
            "true",
            "--replay-context-estimate-gb",
            "4",
            "--storage-budget-mode",
            "attempt",
            "--check-scratch-headroom",
            "true",
            "--scratch-path",
            str(tmp_path),
            "--min-scratch-headroom-gb",
            "0",
            "--dry-run",
            "true",
            "--format",
            "shell",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Replay-context storage estimate" in result.stdout
    assert "headroom_ok=" in result.stdout
