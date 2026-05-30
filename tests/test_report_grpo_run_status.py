import json
import subprocess
import sys
from pathlib import Path

from tools.report_grpo_run_status import parse_job_log, report_grpo_run_status, select_latest_job_log


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_parse_job_log_extracts_direct_and_env_values(tmp_path):
    log = tmp_path / "wam_grpo_replayctx.o123"
    root = tmp_path / "results"
    log.write_text(
        "\n".join(
            [
                "JOB_ID=123",
                "env_list: TERM=NONE,RUN_ID=run_a,RESULTS_ROOT="
                + str(root)
                + ",GROUP_SIZE=4,ACTION_NUM_INFERENCE_STEPS=10",
                "Accepted group attempt logical_group=0 physical_group=0 seed=1",
                "Discarding failed group attempt logical_group=1 physical_group=2 seed=3; "
                "logs remain under /tmp/attempt",
                "Traceback (most recent call last):",
                "OSError: [Errno 122] Disk quota exceeded",
                "Grouped rollout collection complete: " + str(root),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = parse_job_log(log)

    assert report["values"]["JOB_ID"] == "123"
    assert report["values"]["RUN_ID"] == "run_a"
    assert report["values"]["RESULTS_ROOT"] == str(root)
    assert report["values"]["GROUP_SIZE"] == "4"
    assert report["values"]["ACTION_NUM_INFERENCE_STEPS"] == "10"
    assert report["counters"]["accepted_group_attempt_count"] == 1
    assert report["counters"]["discarded_group_attempt_count"] == 1
    assert report["counters"]["traceback_count"] == 1
    assert report["counters"]["disk_quota_count"] == 1
    assert report["failed_attempt_roots"] == ["/tmp/attempt"]
    assert report["completion_paths"] == [str(root)]
    assert report["grouped_rollout_completion_paths"] == [str(root)]
    assert report["training_completion_paths"] == []


def test_report_grpo_run_status_summarizes_results_and_training(tmp_path):
    root = tmp_path / "results"
    groups = root / "groups"
    output = tmp_path / "train"
    log = tmp_path / "wam_grpo_actor.o456"
    group = {
        "group_id": "g0",
        "task": "move_stapler_pad",
        "samples": [
            {"sample_idx": 0, "reward": 1.0, "strict_grpo_artifact_paths": ["a.pt"]},
            {"sample_idx": 1, "reward": 0.0, "strict_grpo_artifact_paths": ["b.pt"]},
        ],
    }
    groups.mkdir(parents=True)
    (groups / "grpo_groups.jsonl").write_text(json.dumps(group) + "\n", encoding="utf-8")
    (groups / "successful_attempt_roots.txt").write_text("/tmp/ok\n", encoding="utf-8")
    _write_json(groups / "grpo_summary.json", {"mixed_groups": 1, "transition_count": 12})
    _write_json(groups / "grpo_dataset_validation.json", {"ok": True, "transition_count": 12, "error_count": 0})
    output.mkdir()
    _write_json(output / "metrics.json", {"history": [{"step": 1, "loss": 0.5, "ratio_mean": 1.0}]})
    (output / "checkpoint.pt").write_bytes(b"checkpoint")
    log.write_text(
        "\n".join(
            [
                "JOB_ID=456",
                f"RESULTS_ROOT={root}",
                f"GRPO_OUTPUT_DIR={output}",
                "Actor replay GRPO training complete: " + str(output),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = report_grpo_run_status(job_log=log, results_root=None, training_output_dir=None)

    assert report["status"]["state"] == "training_checkpoint_written"
    assert report["status"]["trainable_group_count"] == 1
    assert report["status"]["transition_count"] == 12
    assert report["results_root"]["successful_attempt_count"] == 1
    assert report["training_output_dir"]["checkpoint_exists"] is True
    assert report["training_output_dir"]["final_metrics"]["loss"] == 0.5


def test_report_grpo_run_status_infers_paths_from_completion_markers(tmp_path):
    root = tmp_path / "results"
    output = tmp_path / "train"
    groups = root / "groups"
    groups.mkdir(parents=True)
    output.mkdir()
    (groups / "grpo_groups.jsonl").write_text("{}\n", encoding="utf-8")
    _write_json(groups / "grpo_dataset_validation.json", {"ok": True, "transition_count": 3})
    _write_json(output / "metrics.json", {"history": [{"step": 1, "loss": 0.25}]})
    (output / "checkpoint.pt").write_bytes(b"checkpoint")
    log = tmp_path / "job.o1"
    log.write_text(
        "\n".join(
            [
                "Grouped rollout collection complete: " + str(root),
                "Actor replay GRPO training complete: " + str(output),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = report_grpo_run_status(job_log=log, results_root=None, training_output_dir=None)

    assert report["results_root"]["path"] == str(root)
    assert report["training_output_dir"]["path"] == str(output)
    assert report["status"]["state"] == "training_checkpoint_written"


def test_report_grpo_run_status_cli_markdown(tmp_path):
    root = tmp_path / "results"
    groups = root / "groups"
    groups.mkdir(parents=True)
    (groups / "grpo_groups.jsonl").write_text("", encoding="utf-8")
    _write_json(groups / "grpo_summary.json", {"mixed_groups": 0})

    result = subprocess.run(
        [
            sys.executable,
            "tools/report_grpo_run_status.py",
            "--results-root",
            str(root),
            "--print-markdown",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "# GRPO Run Status" in result.stdout
    assert "completed_without_trainable_groups" not in result.stderr


def test_select_latest_job_log_uses_mtime(tmp_path):
    older = tmp_path / "wam_grpo_replayctx.o1"
    newer = tmp_path / "wam_grpo_replayctx.o2"
    older.write_text("JOB_ID=1\n", encoding="utf-8")
    newer.write_text("JOB_ID=2\n", encoding="utf-8")
    older.touch()
    newer.touch()

    selected = select_latest_job_log([str(tmp_path / "wam_grpo_replayctx.o*")])

    assert selected == newer


def test_report_grpo_run_status_cli_accepts_job_log_glob(tmp_path):
    root = tmp_path / "results"
    groups = root / "groups"
    groups.mkdir(parents=True)
    (groups / "grpo_groups.jsonl").write_text("{}\n", encoding="utf-8")
    _write_json(groups / "grpo_dataset_validation.json", {"ok": True, "transition_count": 1})
    log = tmp_path / "wam_grpo_replayctx.o1"
    log.write_text("Grouped rollout collection complete: " + str(root) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "tools/report_grpo_run_status.py",
            "--job-log-glob",
            str(tmp_path / "wam_grpo_replayctx.o*"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["job_log"]["path"] == str(log)
    assert payload["status"]["state"] == "trainable_groups_available"
