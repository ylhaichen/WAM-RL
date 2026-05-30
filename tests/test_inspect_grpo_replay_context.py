import json
import subprocess
import sys

import torch

from tools.inspect_grpo_replay_context import (
    compact_replay_context_summary,
    inspect_replay_context,
)


def test_inspect_replay_context_reports_tensor_bytes_by_top_level(tmp_path):
    path = tmp_path / "strict_grpo_replay_context_0.pt"
    torch.save(
        {
            "schema_version": 2,
            "cache_name": "pos",
            "use_cfg": True,
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

    report = inspect_replay_context(path, top_k=2)

    assert report["schema_version"] == 2
    assert report["metadata_only"] is False
    assert report["tensor_count"] == 3
    assert report["dtype_counts"] == {"float16": 2, "float32": 1}
    assert report["scalar_fields"]["cache_name"] == "pos"
    assert report["scalar_fields"]["action_num_inference_steps"] == 10
    assert report["top_level_tensor_bytes"]["transformer_cache"] == 160
    assert report["top_level_tensor_bytes"]["text_emb"] == 24
    cache_summary = report["transformer_cache_summary"]
    assert cache_summary["block_count"] == 1
    assert cache_summary["kv_batch_sizes"] == [2]
    assert cache_summary["conditional_branch_estimate"]["estimated_savings_bytes"] == 80
    assert report["top_tensors"][0]["path"].startswith("transformer_cache.0.")
    assert report["top_tensors"][0]["device"] == "cpu"

    summary = compact_replay_context_summary(report)
    assert summary["file_gib"] > 0
    assert summary["tensor_gib"] == report["tensor_bytes"] / 1024**3
    assert summary["top_level_tensor_gib"]["transformer_cache"] == 160 / 1024**3
    assert summary["transformer_cache"]["kv_batch_sizes"] == [2]
    assert summary["transformer_cache"]["conditional_branch_estimate"]["estimated_savings_gib"] == 80 / 1024**3


def test_inspect_replay_context_can_use_metadata_only_load(tmp_path):
    path = tmp_path / "strict_grpo_replay_context_0.pt"
    torch.save({"cache": torch.zeros((4, 5), dtype=torch.float16)}, path)

    report = inspect_replay_context(path, metadata_only=True)

    assert report["metadata_only"] is True
    assert report["tensor_bytes"] == 40
    assert report["top_tensors"][0]["device"] == "meta"


def test_inspect_replay_context_cli_writes_outputs(tmp_path):
    context = tmp_path / "ctx.pt"
    out_json = tmp_path / "ctx.json"
    out_md = tmp_path / "ctx.md"
    torch.save({"cache": torch.ones((2, 2), dtype=torch.float32)}, context)

    result = subprocess.run(
        [
            sys.executable,
            "tools/inspect_grpo_replay_context.py",
            str(context),
            "--out-json",
            str(out_json),
            "--out-markdown",
            str(out_md),
            "--metadata-only",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(out_json.read_text())
    assert report["metadata_only"] is True
    assert report["tensor_bytes"] == 16
    assert "GRPO Replay Context Inspection" in out_md.read_text()
    assert "Scalar Fields" in out_md.read_text()


def test_inspect_replay_context_cli_can_print_compact_summary(tmp_path):
    context = tmp_path / "ctx.pt"
    torch.save(
        {
            "action_num_inference_steps": 10,
            "transformer_cache": [
                {
                    "k": torch.zeros((2, 2), dtype=torch.float16),
                    "v": torch.zeros((2, 2), dtype=torch.float16),
                }
            ],
        },
        context,
    )

    result = subprocess.run(
        [
            sys.executable,
            "tools/inspect_grpo_replay_context.py",
            str(context),
            "--metadata-only",
            "--print-summary",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert "top_tensors" not in summary
    assert summary["metadata_only"] is True
    assert summary["scalar_fields"]["action_num_inference_steps"] == 10
    assert summary["transformer_cache"]["kv_batch_sizes"] == [2]
