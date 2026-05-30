import json
import subprocess
import sys
from pathlib import Path

import torch

from tools.inspect_actor_replay_checkpoint import (
    build_report,
    compare_checkpoints,
    summarize_checkpoint,
)


def _write_checkpoint(path: Path, *, weight: torch.Tensor, bias: torch.Tensor, config: dict | None = None) -> None:
    torch.save(
        {
            "trainable_state_dict": {
                "action_head.weight": weight,
                "action_head.bias": bias,
            },
            "config": config or {},
        },
        path,
    )


def test_summarize_checkpoint_reports_tensor_stats(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    _write_checkpoint(
        checkpoint,
        weight=torch.tensor([[3.0, 4.0]], dtype=torch.float32),
        bias=torch.tensor([2.0], dtype=torch.float32),
        config={
            "learning_rate": 1e-7,
            "action_num_inference_steps": 10,
            "logprob_reduction": "mean",
            "trainable_mode": "action_heads",
            "ignored_list": [1, 2],
        },
    )

    summary = summarize_checkpoint(checkpoint)

    assert summary["tensor_count"] == 2
    assert summary["param_count"] == 3
    assert summary["total_l2_norm"] == (3.0**2 + 4.0**2 + 2.0**2) ** 0.5
    assert summary["max_abs"] == 4.0
    assert summary["dtype_counts"] == {"float32": 2}
    assert summary["top_key_prefix_counts"] == {"action_head": 2}
    assert summary["config"]["learning_rate"] == 1e-7
    assert summary["config"]["action_num_inference_steps"] == 10
    assert summary["config"]["logprob_reduction"] == "mean"
    assert summary["config"]["trainable_mode"] == "action_heads"
    assert "ignored_list" not in summary["config"]


def test_compare_checkpoints_reports_deltas(tmp_path):
    reference = tmp_path / "reference.pt"
    candidate = tmp_path / "candidate.pt"
    _write_checkpoint(
        reference,
        weight=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        bias=torch.tensor([0.5], dtype=torch.float32),
    )
    _write_checkpoint(
        candidate,
        weight=torch.tensor([[2.0, 4.0]], dtype=torch.float32),
        bias=torch.tensor([0.5], dtype=torch.float32),
    )

    comparison = compare_checkpoints(candidate, reference)

    assert comparison["shared_key_count"] == 2
    assert comparison["compared_param_count"] == 3
    assert comparison["delta_l2_norm"] == (1.0**2 + 2.0**2) ** 0.5
    assert comparison["delta_max_abs"] == 2.0
    assert comparison["changed_tensor_count"] == 1
    assert comparison["missing_in_candidate_count"] == 0
    assert comparison["extra_in_candidate_count"] == 0


def test_build_report_and_cli_write_json(tmp_path):
    reference = tmp_path / "reference.pt"
    candidate = tmp_path / "candidate.pt"
    out_json = tmp_path / "report.json"
    out_markdown = tmp_path / "report.md"
    _write_checkpoint(reference, weight=torch.zeros((1, 2)), bias=torch.zeros(1))
    _write_checkpoint(candidate, weight=torch.ones((1, 2)), bias=torch.zeros(1))

    report = build_report([candidate], reference=reference)

    assert len(report["checkpoints"]) == 1
    assert len(report["comparisons"]) == 1

    result = subprocess.run(
        [
            sys.executable,
            "tools/inspect_actor_replay_checkpoint.py",
            str(candidate),
            "--reference",
            str(reference),
            "--out-json",
            str(out_json),
            "--out-markdown",
            str(out_markdown),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    written = json.loads(out_json.read_text(encoding="utf-8"))
    assert written["comparisons"][0]["shared_key_count"] == 2
    markdown = out_markdown.read_text(encoding="utf-8")
    assert "Actor Replay Checkpoint Inspection" in markdown
    assert "action_steps" in markdown
