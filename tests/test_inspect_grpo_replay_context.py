import json
import subprocess
import sys

import torch

from tools.inspect_grpo_replay_context import inspect_replay_context


def test_inspect_replay_context_reports_tensor_bytes_by_top_level(tmp_path):
    path = tmp_path / "strict_grpo_replay_context_0.pt"
    torch.save(
        {
            "schema_version": 2,
            "text_emb": torch.zeros((2, 3), dtype=torch.float32),
            "transformer_cache": [
                {
                    "k": torch.zeros((4, 5), dtype=torch.float16),
                    "v": torch.zeros((4, 5), dtype=torch.float16),
                }
            ],
        },
        path,
    )

    report = inspect_replay_context(path, top_k=2)

    assert report["schema_version"] == 2
    assert report["tensor_count"] == 3
    assert report["dtype_counts"] == {"float16": 2, "float32": 1}
    assert report["top_level_tensor_bytes"]["transformer_cache"] == 80
    assert report["top_level_tensor_bytes"]["text_emb"] == 24
    assert report["top_tensors"][0]["path"].startswith("transformer_cache.0.")


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
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(out_json.read_text())["tensor_bytes"] == 16
    assert "GRPO Replay Context Inspection" in out_md.read_text()
