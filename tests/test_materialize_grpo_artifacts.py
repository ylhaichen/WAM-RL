import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import tools.materialize_grpo_artifacts as materialize_tool
from tools.materialize_grpo_artifacts import materialize_grpo_artifacts, write_materialized_outputs


def _write_group(path: Path, artifact_path: Path) -> None:
    group = {
        "group_id": "g0",
        "task": "move_stapler_pad",
        "group_size": 1,
        "reward_mean": 1.0,
        "reward_std": 0.0,
        "samples": [
            {
                "task": "move_stapler_pad",
                "group_id": "g0",
                "sample_idx": 0,
                "reward": 1.0,
                "advantage": 1.0,
                "success": True,
                "record_path": "/tmp/record.json",
                "strict_grpo_artifact_count": 1,
                "strict_grpo_artifact_paths": [str(artifact_path)],
            }
        ],
    }
    path.write_text(json.dumps(group) + "\n", encoding="utf-8")


def test_materialize_grpo_artifacts_symlinks_artifact_and_replay_context(tmp_path):
    source_dir = tmp_path / "source" / "run with spaces"
    source_dir.mkdir(parents=True)
    artifact = source_dir / "strict_grpo_0.pt"
    context = source_dir / "strict_grpo_replay_context_0.pt"
    torch.save({"replay_context_path": context.name}, artifact)
    torch.save({"context": True}, context)
    groups_jsonl = tmp_path / "groups.jsonl"
    _write_group(groups_jsonl, artifact)

    groups, manifest = materialize_grpo_artifacts(
        groups_jsonl,
        out_root=tmp_path / "materialized",
        include_replay_context=True,
    )

    rewritten_path = Path(groups[0]["samples"][0]["strict_grpo_artifact_paths"][0])
    rewritten_context = rewritten_path.parent / context.name
    assert rewritten_path.is_symlink()
    assert rewritten_path.resolve() == artifact
    assert rewritten_context.is_symlink()
    assert rewritten_context.resolve() == context
    assert manifest["unique_artifact_count"] == 1
    assert manifest["unique_replay_context_count"] == 1
    assert manifest["planned_copy_bytes"] == 0
    assert manifest["source_artifacts_plus_replay_contexts"]["resolved_bytes"] == artifact.stat().st_size + context.stat().st_size


def test_materialize_grpo_artifacts_can_copy_and_write_outputs(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    artifact = source_dir / "strict.pt"
    torch.save({"schema_version": 2}, artifact)
    groups_jsonl = tmp_path / "groups.jsonl"
    out_jsonl = tmp_path / "out" / "groups" / "grpo_groups.jsonl"
    out_manifest = tmp_path / "out" / "manifest.json"
    _write_group(groups_jsonl, artifact)

    groups, manifest = materialize_grpo_artifacts(
        groups_jsonl,
        out_root=tmp_path / "out",
        link_mode="copy",
    )
    write_materialized_outputs(groups, manifest, out_jsonl=out_jsonl, out_manifest=out_manifest)

    rewritten_path = Path(json.loads(out_jsonl.read_text())["samples"][0]["strict_grpo_artifact_paths"][0])
    assert rewritten_path.exists()
    assert not rewritten_path.is_symlink()
    assert torch.load(rewritten_path, map_location="cpu") == {"schema_version": 2}
    assert json.loads(out_manifest.read_text())["output_jsonl"] == str(out_jsonl)


def test_materialize_grpo_artifacts_dry_run_only_reports_paths(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    artifact = source_dir / "strict_grpo_0.pt"
    context = source_dir / "strict_grpo_replay_context_0.pt"
    torch.save({"replay_context_path": context.name}, artifact)
    torch.save({"context": True}, context)
    groups_jsonl = tmp_path / "groups.jsonl"
    out_root = tmp_path / "dry_run_out"
    _write_group(groups_jsonl, artifact)

    groups, manifest = materialize_grpo_artifacts(
        groups_jsonl,
        out_root=out_root,
        link_mode="copy",
        include_replay_context=True,
        dry_run=True,
    )

    rewritten_path = Path(groups[0]["samples"][0]["strict_grpo_artifact_paths"][0])
    assert manifest["dry_run"] is True
    assert manifest["unique_artifact_count"] == 1
    assert manifest["unique_replay_context_count"] == 1
    assert manifest["source_artifacts"]["resolved_bytes"] == artifact.stat().st_size
    assert manifest["source_replay_contexts"]["resolved_bytes"] == context.stat().st_size
    assert manifest["planned_copy_bytes"] == artifact.stat().st_size + context.stat().st_size
    assert rewritten_path.is_relative_to(out_root)
    assert not out_root.exists()


def test_materialize_grpo_artifacts_cli_dry_run_does_not_write_outputs(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    artifact = source_dir / "strict_grpo_0.pt"
    torch.save({"schema_version": 2}, artifact)
    groups_jsonl = tmp_path / "groups.jsonl"
    out_root = tmp_path / "cli_dry_run_out"
    _write_group(groups_jsonl, artifact)

    result = subprocess.run(
        [
            sys.executable,
            "tools/materialize_grpo_artifacts.py",
            str(groups_jsonl),
            "--out-root",
            str(out_root),
            "--link-mode",
            "copy",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["output_jsonl"] == str(out_root / "groups" / "grpo_groups.jsonl")
    assert payload["output_manifest"] == str(out_root / "manifest.json")
    assert payload["source_artifacts"]["resolved_bytes"] == artifact.stat().st_size
    assert payload["planned_copy_bytes"] == artifact.stat().st_size
    assert not out_root.exists()


def test_materialize_grpo_artifacts_checks_torch_before_partial_output(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    artifact = source_dir / "strict.pt"
    torch.save({"replay_context_path": "ctx.pt"}, artifact)
    groups_jsonl = tmp_path / "groups.jsonl"
    out_root = tmp_path / "out"
    _write_group(groups_jsonl, artifact)

    def fail_torch():
        raise RuntimeError("missing torch")

    monkeypatch.setattr(materialize_tool, "_ensure_torch_available", fail_torch)

    with pytest.raises(RuntimeError, match="missing torch"):
        materialize_grpo_artifacts(
            groups_jsonl,
            out_root=out_root,
            include_replay_context=True,
        )

    assert not out_root.exists()


def test_materialize_reads_replay_context_path_with_meta_map_location(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    artifact = source_dir / "strict.pt"
    artifact.write_bytes(b"artifact-placeholder")
    context = source_dir / "ctx.pt"
    context.write_bytes(b"context")
    groups_jsonl = tmp_path / "groups.jsonl"
    _write_group(groups_jsonl, artifact)

    seen_map_locations = []

    def fake_load(path, *, map_location):
        seen_map_locations.append(map_location)
        assert path == artifact
        return {"replay_context_path": context.name}

    monkeypatch.setattr(torch, "load", fake_load)

    materialize_grpo_artifacts(
        groups_jsonl,
        out_root=tmp_path / "out",
        include_replay_context=True,
    )

    assert seen_map_locations == ["meta"]
