import json
import subprocess
import sys
from pathlib import Path

from tools.plan_myriad_storage_cleanup import compact_cleanup_summary, plan_myriad_storage_cleanup, write_markdown_report


def _write_run(root: Path, name: str, *, groups: list[dict] | None, summary: dict | None = None) -> Path:
    run = root / name
    groups_dir = run / "groups"
    groups_dir.mkdir(parents=True)
    if groups is not None:
        (groups_dir / "grpo_groups.jsonl").write_text(
            "".join(json.dumps(group) + "\n" for group in groups),
            encoding="utf-8",
        )
    if summary is not None:
        (groups_dir / "grpo_summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
    return run


def _group() -> dict:
    return {
        "group_id": "g0",
        "task": "move_stapler_pad",
        "samples": [{"sample_idx": 0, "reward": 1.0, "strict_grpo_artifact_paths": ["/tmp/a.pt"]}],
    }


def test_plan_myriad_storage_cleanup_marks_empty_replayctx_server_vis_candidate(tmp_path):
    run = _write_run(
        tmp_path,
        "grpo_replayctx_smoke",
        groups=[],
        summary={"total_groups": 1, "mixed_groups": 0, "skipped_all_success": 1},
    )
    (run / "server_vis").mkdir()
    (run / "server_vis" / "strict_grpo_replay_context_0.pt").write_bytes(b"x" * 4096)
    (run / "groups" / "successful_attempt_roots.txt").write_text(str(run / "attempts/a") + "\n", encoding="utf-8")

    report = plan_myriad_storage_cleanup([tmp_path], min_candidate_gb=0)

    assert report["run_count"] == 1
    assert report["cleanup_candidate_count"] == 2
    actions = {candidate["action"] for candidate in report["cleanup_candidates"]}
    assert "review_delete_server_vis_after_metadata_backup" in actions
    assert "review_delete_whole_debug_run_after_metadata_backup" in actions
    assert report["runs"][0]["grpo_group_line_count"] == 0
    assert report["runs"][0]["grpo_group_total_line_count"] == 0


def test_plan_myriad_storage_cleanup_protects_non_empty_groups(tmp_path):
    run = _write_run(tmp_path, "grpo_replayctx_staplerpad", groups=[_group()])
    (run / "server_vis").mkdir()
    (run / "server_vis" / "strict_grpo_replay_context_0.pt").write_bytes(b"x" * 4096)

    report = plan_myriad_storage_cleanup([run], min_candidate_gb=0, large_run_gb=0)

    assert report["cleanup_candidate_count"] == 0
    assert report["protected_run_count"] == 1
    assert report["runs"][0]["protection_reasons"] == [
        "non-empty groups/grpo_groups*.jsonl may reference trainable artifacts"
    ]
    assert report["runs"][0]["notes"] == [
        "large trainable source run; materialize/archive subsets before considering server_vis cleanup"
    ]


def test_plan_myriad_storage_cleanup_protects_non_empty_partial_group_files(tmp_path):
    run = _write_run(tmp_path, "grpo_scale_tasks_a_k8_g8_retry", groups=[])
    (run / "groups" / "grpo_groups_partial.jsonl").write_text(json.dumps(_group()) + "\n", encoding="utf-8")
    (run / "server_vis").mkdir()
    (run / "server_vis" / "strict_grpo_0.pt").write_bytes(b"x" * 4096)

    report = plan_myriad_storage_cleanup([run], min_candidate_gb=0, large_run_gb=0)

    assert report["cleanup_candidate_count"] == 0
    assert report["protected_run_count"] == 1
    assert report["runs"][0]["grpo_group_line_count"] == 0
    assert report["runs"][0]["grpo_group_total_line_count"] == 1
    assert report["runs"][0]["protection_reasons"] == [
        "non-empty groups/grpo_groups*.jsonl may reference trainable artifacts",
        "name matches curated dataset/source run pattern",
    ]


def test_write_markdown_report_has_candidate_and_protected_sections(tmp_path):
    empty = _write_run(tmp_path, "grpo_debug_empty", groups=[])
    (empty / "server_vis").mkdir()
    (empty / "server_vis" / "artifact.pt").write_bytes(b"x")
    trainable = _write_run(
        tmp_path,
        "grpo_replayctx_trainable",
        groups=[_group()],
    )
    (trainable / "server_vis").mkdir()
    (trainable / "server_vis" / "artifact.pt").write_bytes(b"x")

    report = plan_myriad_storage_cleanup([tmp_path], min_candidate_gb=0, large_run_gb=0)
    text = write_markdown_report(report)

    assert "Cleanup Candidates" in text
    assert "Protected Runs" in text
    assert "grpo_debug_empty" in text
    assert "grpo_replayctx_trainable" in text


def test_plan_myriad_storage_cleanup_cli_prints_summary(tmp_path):
    run = _write_run(tmp_path, "grpo_replayctx_smoke", groups=[])
    (run / "server_vis").mkdir()
    (run / "server_vis" / "artifact.pt").write_bytes(b"x")

    result = subprocess.run(
        [
            sys.executable,
            "tools/plan_myriad_storage_cleanup.py",
            str(tmp_path),
            "--min-candidate-gb",
            "0",
            "--print-summary",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["run_count"] == 1
    assert summary["cleanup_candidate_count"] == 2
    assert summary["top_candidates"][0]["run_name"] == "grpo_replayctx_smoke"


def test_plan_myriad_storage_cleanup_summary_reports_large_protected_runs(tmp_path):
    run = _write_run(tmp_path, "grpo_replayctx_trainable", groups=[_group()])
    (run / "server_vis").mkdir()
    (run / "server_vis" / "artifact.pt").write_bytes(b"x" * 4096)

    report = plan_myriad_storage_cleanup([tmp_path], min_candidate_gb=0, large_run_gb=0)
    summary = compact_cleanup_summary(report)

    assert summary["cleanup_candidate_count"] == 0
    assert summary["protected_disk_gb"] > 0
    assert summary["top_protected_runs"][0]["run_name"] == "grpo_replayctx_trainable"
    assert summary["top_protected_runs"][0]["group_line_count"] == 1
    assert summary["top_protected_runs"][0]["protection_reasons"]
    assert summary["top_protected_runs"][0]["notes"]
