import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.merge_grpo_groups import merge_grpo_group_files, write_outputs


def _write_group(path: Path, group_id: str, *, sample_count: int = 1) -> None:
    samples = [
        {
            "task": "move_stapler_pad",
            "group_id": group_id,
            "sample_idx": idx,
            "reward": float(idx % 2),
            "advantage": 0.0,
            "success": bool(idx % 2),
            "record_path": f"/tmp/{group_id}_{idx}.json",
            "strict_grpo_artifact_paths": [],
            "strict_grpo_artifact_count": 0,
        }
        for idx in range(sample_count)
    ]
    group = {
        "group_id": group_id,
        "task": "move_stapler_pad",
        "group_size": sample_count,
        "reward_mean": 0.5,
        "reward_std": 0.5,
        "samples": samples,
    }
    path.write_text(json.dumps(group) + "\n", encoding="utf-8")


def test_merge_grpo_group_files_writes_manifest(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    out_jsonl = tmp_path / "merged" / "grpo_groups.jsonl"
    out_manifest = tmp_path / "merged" / "manifest.json"
    _write_group(first, "g0", sample_count=2)
    _write_group(second, "g1", sample_count=3)

    groups, manifest = merge_grpo_group_files([first, second])
    write_outputs(groups, manifest, out_jsonl=out_jsonl, out_manifest=out_manifest)

    assert out_jsonl.read_text(encoding="utf-8").count("\n") == 2
    payload = json.loads(out_manifest.read_text(encoding="utf-8"))
    assert payload["group_count"] == 2
    assert payload["sample_count"] == 5
    assert payload["duplicate_group_id_count"] == 0
    assert payload["source_counts"] == [
        {"source_file": str(first), "group_count": 1},
        {"source_file": str(second), "group_count": 1},
    ]


def test_merge_grpo_group_files_rejects_duplicate_group_ids(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_group(first, "g0")
    _write_group(second, "g0")

    with pytest.raises(ValueError, match="duplicate group_id"):
        merge_grpo_group_files([first, second])

    groups, manifest = merge_grpo_group_files([first, second], allow_duplicate_group_ids=True)
    assert len(groups) == 2
    assert manifest["duplicate_group_id_count"] == 1
    assert manifest["duplicate_group_ids"] == ["g0"]


def test_merge_grpo_groups_script_entrypoint(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    out_jsonl = tmp_path / "out" / "grpo_groups.jsonl"
    _write_group(first, "g0")
    _write_group(second, "g1")

    result = subprocess.run(
        [
            sys.executable,
            "tools/merge_grpo_groups.py",
            str(first),
            str(second),
            "--out-jsonl",
            str(out_jsonl),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["group_count"] == 2
    assert payload["output_jsonl"] == str(out_jsonl)
    assert out_jsonl.exists()
