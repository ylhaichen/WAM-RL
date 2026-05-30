import json
import subprocess
import sys

import torch

from tools.summarize_grpo_replay_contexts import (
    compact_summary,
    format_markdown,
    summarize_replay_contexts,
)


def _write_context(path):
    torch.save(
        {
            "schema_version": 2,
            "cache_name": "pos",
            "use_cfg": True,
            "cfg_pruned_to_conditional": False,
            "action_guidance_scale": 1.0,
            "action_num_inference_steps": 10,
            "frame_chunk_size": 21,
            "text_emb": torch.zeros((2, 3), dtype=torch.float32),
            "transformer_cache": [
                {
                    "k": torch.zeros((2, 4, 5), dtype=torch.float16),
                    "v": torch.zeros((2, 4, 5), dtype=torch.float16),
                }
            ],
        },
        path,
    )


def _write_groups(path, context_path=None, artifact_path=None, capture_stride=None, capture_max_chunks=None):
    sample = {
        "sample_idx": 0,
        "reward": 1.0,
        "strict_grpo_artifact_paths": [str(artifact_path)] if artifact_path else [],
    }
    if context_path is not None:
        sample["strict_grpo_replay_context_paths"] = [str(context_path)]
    if capture_stride is not None:
        sample["strict_grpo_capture_chunk_stride"] = capture_stride
    if capture_max_chunks is not None:
        sample["strict_grpo_capture_max_chunks"] = capture_max_chunks
    path.write_text(
        json.dumps(
            {
                "group_id": "g0",
                "task": "move_stapler_pad",
                "samples": [sample],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_summarize_replay_contexts_aggregates_group_metadata_paths(tmp_path):
    context = tmp_path / "strict_grpo_replay_context_0.pt"
    groups = tmp_path / "grpo_groups.jsonl"
    _write_context(context)
    _write_groups(groups, context_path=context)

    report = summarize_replay_contexts([groups], inspect_context_tensors=True)

    assert report["group_count"] == 1
    assert report["sample_count"] == 1
    assert report["unique_context_count"] == 1
    assert report["inspected_context_count"] == 1
    assert report["file_total_bytes"] == context.stat().st_size
    assert report["tensor_total_bytes"] == 184
    assert report["conditional_branch_estimated_savings_gib"] == 80 / 1024**3
    assert report["context_tensor_inspected"] is True
    assert report["config_count"] == 1
    config = report["configs"][0]["config"]
    assert config["action_num_inference_steps"] == 10
    assert config["kv_batch_sizes"] == [2]
    assert report["configs"][0]["tasks"] == ["move_stapler_pad"]

    summary = compact_summary(report)
    assert "missing_context_paths" not in summary
    assert summary["unique_context_count"] == 1


def test_summarize_replay_contexts_can_resolve_paths_from_strict_artifacts(tmp_path):
    context = tmp_path / "strict_grpo_replay_context_0.pt"
    artifact = tmp_path / "strict_grpo_0.pt"
    groups = tmp_path / "grpo_groups.jsonl"
    _write_context(context)
    torch.save({"replay_context_path": context.name}, artifact)
    _write_groups(groups, artifact_path=artifact)

    report = summarize_replay_contexts([groups], inspect_artifacts=True, inspect_context_tensors=True)

    assert report["unique_context_count"] == 1
    assert report["artifact_error_count"] == 0
    assert report["context_error_count"] == 0
    assert report["configs"][0]["context_count"] == 1


def test_summarize_replay_contexts_default_only_stats_context_files(tmp_path):
    context = tmp_path / "strict_grpo_replay_context_0.pt"
    groups = tmp_path / "grpo_groups.jsonl"
    _write_context(context)
    _write_groups(groups, context_path=context, capture_stride=2, capture_max_chunks=1)

    report = summarize_replay_contexts([groups])

    assert report["context_tensor_inspected"] is False
    assert report["file_total_bytes"] == context.stat().st_size
    assert report["tensor_total_bytes"] == 0
    config = report["configs"][0]["config"]
    assert config["action_num_inference_steps"] is None
    assert config["strict_grpo_capture_chunk_stride"] == 2
    assert config["strict_grpo_capture_max_chunks"] == 1


def test_summarize_replay_contexts_formats_markdown(tmp_path):
    context = tmp_path / "strict_grpo_replay_context_0.pt"
    groups = tmp_path / "grpo_groups.jsonl"
    _write_context(context)
    _write_groups(groups, context_path=context)

    text = format_markdown(summarize_replay_contexts([groups], inspect_context_tensors=True))

    assert "# GRPO Replay Context Summary" in text
    assert "move_stapler_pad" in text
    assert "conditional_branch_estimated_savings_gib" in text


def test_summarize_replay_contexts_cli_writes_reports(tmp_path):
    context = tmp_path / "strict_grpo_replay_context_0.pt"
    groups = tmp_path / "grpo_groups.jsonl"
    out_json = tmp_path / "summary.json"
    out_csv = tmp_path / "summary.csv"
    out_md = tmp_path / "summary.md"
    _write_context(context)
    _write_groups(groups, context_path=context)

    result = subprocess.run(
        [
            sys.executable,
            "tools/summarize_grpo_replay_contexts.py",
            str(groups),
            "--print-summary",
            "--out-json",
            str(out_json),
            "--out-csv",
            str(out_csv),
            "--out-markdown",
            str(out_md),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout.split("Wrote JSON:")[0])
    assert summary["unique_context_count"] == 1
    assert "contexts" not in summary
    assert json.loads(out_json.read_text())["unique_context_count"] == 1
    assert "action_num_inference_steps" in out_csv.read_text()
    assert "GRPO Replay Context Summary" in out_md.read_text()
