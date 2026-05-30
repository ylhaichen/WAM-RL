import json
import os
from pathlib import Path

import torch

from tools.summarize_actor_replay_training import (
    discover_output_dirs,
    format_text_report,
    summarize_actor_replay_output,
    write_csv_report,
    write_markdown_report,
)


def _write_output(root: Path) -> None:
    root.mkdir()
    checkpoint = root / "checkpoint.pt"
    checkpoint.write_bytes(b"ckpt")
    (root / "input_dataset_validation.json").write_text(
        json.dumps({"ok": True, "error_count": 0, "transition_count": 2}) + "\n",
        encoding="utf-8",
    )
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "result": {
                    "transition_count": 2,
                    "steps": 1,
                    "final_loss": 0.25,
                    "final_ratio_mean": 1.1,
                    "checkpoint_path": str(checkpoint),
                    "trainable_param_count": 3,
                    "total_param_count": 5,
                },
                "config": {
                    "learning_rate": 1e-7,
                    "action_num_inference_steps": 10,
                    "logprob_reduction": "mean",
                    "logprob_std_floor": 0.1,
                    "trainable_mode": "action_heads",
                },
                "history": [
                    {
                        "step": 1,
                        "loss": 0.25,
                        "ratio_mean": 1.1,
                        "grad_norm": 0.2,
                        "param_update_norm": 0.01,
                        "param_update_max": 0.005,
                        "param_update_param_count": 3,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_summarize_actor_replay_output_reports_complete_run(tmp_path):
    root = tmp_path / "run"
    _write_output(root)

    summary = summarize_actor_replay_output(root)

    assert summary["ok"] is True
    assert summary["validation_ok"] is True
    assert summary["checkpoint_exists"] is True
    assert summary["checkpoint_bytes"] == len(b"ckpt")
    assert summary["transition_count"] == 2
    assert summary["steps"] == 1
    assert summary["final_loss"] == 0.25
    assert summary["config_source"] == "metrics"
    assert summary["learning_rate"] == 1e-7
    assert summary["action_num_inference_steps"] == 10
    assert summary["logprob_reduction"] == "mean"
    assert summary["logprob_std_floor"] == 0.1
    assert summary["trainable_mode"] == "action_heads"
    assert summary["final_grad_norm"] == 0.2
    assert summary["final_param_update_norm"] == 0.01
    assert summary["final_param_update_max"] == 0.005
    assert summary["parameter_update_measured"] is True
    assert summary["parameter_update_detected"] is True
    assert summary["warnings"] == []
    assert summary["last_step"]["grad_norm"] == 0.2


def test_discover_output_dirs_filters_pattern_and_latest(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    old_run = root / "actor_old"
    new_run = root / "actor_new"
    ignored = root / "baseline_new"
    for idx, path in enumerate([old_run, ignored, new_run], start=1):
        path.mkdir()
        os.utime(path, (idx, idx))

    discovered = discover_output_dirs(root, pattern="actor_*", latest=1)

    assert discovered == [new_run]


def test_summarize_actor_replay_output_marks_failure_diagnostics(tmp_path):
    root = tmp_path / "run"
    _write_output(root)
    (root / "failure_diagnostics.json").write_text("{}\n", encoding="utf-8")

    summary = summarize_actor_replay_output(root)

    assert summary["ok"] is False
    assert summary["failure_diagnostics_exists"] is True


def test_write_markdown_report(tmp_path):
    root = tmp_path / "run"
    _write_output(root)
    summary = summarize_actor_replay_output(root)
    out = tmp_path / "summary.md"

    write_markdown_report([summary], out)

    text = out.read_text(encoding="utf-8")
    assert "Actor Replay Training Summary" in text
    assert "| output_dir | ok | validation |" in text
    assert "metrics" in text
    assert "action_steps" in text
    assert "mean" in text
    assert "update_norm" in text
    assert "0.25" in text


def test_write_csv_report(tmp_path):
    root = tmp_path / "run"
    _write_output(root)
    summary = summarize_actor_replay_output(root)
    out = tmp_path / "summary.csv"

    write_csv_report([summary], out)

    text = out.read_text(encoding="utf-8")
    assert "output_dir,ok,validation_ok" in text
    assert "config_source" in text
    assert "learning_rate" in text
    assert "1e-07" in text
    assert "mean" in text


def test_format_text_report_is_concise_for_terminal_triage(tmp_path):
    root = tmp_path / "run"
    _write_output(root)
    summary = summarize_actor_replay_output(root)

    text = format_text_report([summary])

    assert "run" in text
    assert "update_norm" in text
    assert "yes" in text
    assert "0.01" in text
    assert str(root) not in text


def test_summarize_actor_replay_output_falls_back_to_checkpoint_config(tmp_path):
    root = tmp_path / "run"
    _write_output(root)

    metrics_path = root / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    del metrics["config"]
    metrics_path.write_text(json.dumps(metrics) + "\n", encoding="utf-8")
    torch.save(
        {
            "trainable_state_dict": {"action_head.weight": torch.tensor([1.0])},
            "config": {
                "learning_rate": 2e-7,
                "action_num_inference_steps": 8,
                "logprob_reduction": "sum",
                "logprob_std_floor": 0.2,
                "trainable_mode": "last_block",
                "ignored_list": [1, 2],
            },
        },
        root / "checkpoint.pt",
    )

    summary = summarize_actor_replay_output(root)

    assert summary["config_source"] == "checkpoint"
    assert summary["learning_rate"] == 2e-7
    assert summary["action_num_inference_steps"] == 8
    assert summary["logprob_reduction"] == "sum"
    assert summary["logprob_std_floor"] == 0.2
    assert summary["trainable_mode"] == "last_block"
    assert "missing_training_config" not in summary["warnings"]


def test_summarize_actor_replay_output_warns_when_update_is_zero(tmp_path):
    root = tmp_path / "run"
    _write_output(root)
    metrics_path = root / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["history"][0]["param_update_norm"] = 0.0
    metrics["history"][0]["param_update_max"] = 0.0
    metrics_path.write_text(json.dumps(metrics) + "\n", encoding="utf-8")

    summary = summarize_actor_replay_output(root)

    assert summary["ok"] is True
    assert summary["parameter_update_measured"] is True
    assert summary["parameter_update_detected"] is False
    assert "no_parameter_update_detected" in summary["warnings"]


def test_summarize_actor_replay_output_warns_when_update_metric_is_missing(tmp_path):
    root = tmp_path / "run"
    _write_output(root)
    metrics_path = root / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    del metrics["history"][0]["param_update_norm"]
    del metrics["history"][0]["param_update_max"]
    metrics_path.write_text(json.dumps(metrics) + "\n", encoding="utf-8")

    summary = summarize_actor_replay_output(root)

    assert summary["ok"] is True
    assert summary["parameter_update_measured"] is False
    assert summary["parameter_update_detected"] is False
    assert "missing_parameter_update_metric" in summary["warnings"]
