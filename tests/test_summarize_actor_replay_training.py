import json
from pathlib import Path

from tools.summarize_actor_replay_training import (
    summarize_actor_replay_output,
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
                "history": [
                    {
                        "step": 1,
                        "loss": 0.25,
                        "ratio_mean": 1.1,
                        "grad_norm": 0.2,
                        "param_update_norm": 0.01,
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
    assert summary["last_step"]["grad_norm"] == 0.2


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
    assert "0.25" in text
