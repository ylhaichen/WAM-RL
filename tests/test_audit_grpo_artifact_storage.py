import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from tools.audit_grpo_artifact_storage import audit_grpo_artifact_storage, compact_storage_summary


def _write_group(path: Path, artifact_paths: list[Path]) -> None:
    group = {
        "group_id": "g0",
        "task": "move_stapler_pad",
        "samples": [
            {
                "task": "move_stapler_pad",
                "group_id": "g0",
                "sample_idx": 0,
                "reward": 1.0,
                "strict_grpo_artifact_paths": [str(path) for path in artifact_paths],
            },
            {
                "task": "move_stapler_pad",
                "group_id": "g0",
                "sample_idx": 1,
                "reward": 0.0,
                "strict_grpo_artifact_paths": [str(artifact_paths[0])],
            },
        ],
    }
    path.write_text(json.dumps(group) + "\n", encoding="utf-8")


def test_audit_grpo_artifact_storage_counts_refs_and_symlink_targets(tmp_path):
    source = tmp_path / "source.pt"
    source.write_bytes(b"1234567890")
    linked = tmp_path / "linked.pt"
    os.symlink(source, linked)
    groups_jsonl = tmp_path / "groups.jsonl"
    _write_group(groups_jsonl, [source, linked])

    report = audit_grpo_artifact_storage(groups_jsonl)

    assert report["group_count"] == 1
    assert report["sample_count"] == 2
    assert report["artifact_ref_count"] == 3
    assert report["unique_artifact_count"] == 2
    assert report["tasks"] == [
        {
            "task": "move_stapler_pad",
            "group_count": 1,
            "sample_count": 2,
            "success_count": 1,
            "failure_count": 1,
            "artifact_ref_count": 3,
        }
    ]
    assert report["artifacts"]["existing_count"] == 2
    assert report["artifacts"]["symlink_count"] == 1
    assert report["artifacts"]["resolved_bytes"] == 20
    assert report["artifacts"]["apparent_bytes"] >= 10


def test_audit_grpo_artifact_storage_reads_materialization_manifest(tmp_path):
    artifact = tmp_path / "artifact.pt"
    context = tmp_path / "context.pt"
    materialized_context = tmp_path / "materialized" / "context.pt"
    artifact.write_bytes(b"artifact")
    context.write_bytes(b"context")
    materialized_context.parent.mkdir()
    os.symlink(context, materialized_context)

    groups_jsonl = tmp_path / "groups.jsonl"
    _write_group(groups_jsonl, [artifact])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "link_mode": "symlink",
                "replay_context_mapping": {str(context): str(materialized_context)},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = audit_grpo_artifact_storage(groups_jsonl, materialize_manifest=manifest)

    assert report["materialize_link_mode"] == "symlink"
    assert report["manifest_unique_replay_context_count"] == 1
    assert report["materialized_replay_contexts"]["symlink_count"] == 1
    assert report["materialized_replay_contexts"]["resolved_bytes"] == len(b"context")
    assert report["source_replay_contexts"]["regular_file_count"] == 1


def test_audit_grpo_artifact_storage_reports_broken_symlink(tmp_path):
    missing_target = tmp_path / "missing.pt"
    broken = tmp_path / "broken.pt"
    os.symlink(missing_target, broken)
    groups_jsonl = tmp_path / "groups.jsonl"
    _write_group(groups_jsonl, [broken])

    report = audit_grpo_artifact_storage(groups_jsonl)

    assert report["artifacts"]["missing_count"] == 1
    assert report["artifacts"]["broken_symlink_count"] == 1
    assert report["artifacts"]["missing_paths"] == [str(broken)]


def test_audit_grpo_artifact_storage_can_inspect_replay_contexts(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    artifact = source_dir / "strict_grpo_0.pt"
    context = source_dir / "strict_grpo_replay_context_0.pt"
    torch.save({"replay_context_path": context.name}, artifact)
    context.write_bytes(b"context-bytes")
    groups_jsonl = tmp_path / "groups.jsonl"
    _write_group(groups_jsonl, [artifact])

    report = audit_grpo_artifact_storage(groups_jsonl, inspect_replay_contexts=True)

    assert report["replay_context_ref_count"] == 1
    assert report["unique_replay_context_count"] == 1
    assert report["replay_contexts"]["resolved_bytes"] == len(b"context-bytes")
    assert report["artifacts_plus_replay_contexts"]["existing_count"] == 2
    assert report["replay_context_errors"] == []
    assert report["replay_context_mapping"] == {str(artifact): str(context)}

    summary = compact_storage_summary(report)
    assert summary["unique_replay_context_count"] == 1
    assert summary["combined_resolved_gb"] == report["artifacts_plus_replay_contexts"]["resolved_bytes"] / 1024**3


def test_audit_grpo_artifact_storage_inspects_replay_contexts_with_meta_map_location(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    artifact = source_dir / "strict_grpo_0.pt"
    artifact.write_bytes(b"artifact-placeholder")
    context = source_dir / "strict_grpo_replay_context_0.pt"
    context.write_bytes(b"context")
    groups_jsonl = tmp_path / "groups.jsonl"
    _write_group(groups_jsonl, [artifact])

    seen_map_locations = []

    def fake_load(path, *, map_location):
        seen_map_locations.append(map_location)
        assert path == artifact
        return {"replay_context_path": context.name}

    monkeypatch.setattr(torch, "load", fake_load)

    report = audit_grpo_artifact_storage(groups_jsonl, inspect_replay_contexts=True)

    assert seen_map_locations == ["meta"]
    assert report["replay_context_mapping"] == {str(artifact): str(context)}


def test_audit_grpo_artifact_storage_can_omit_replay_context_mapping(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    artifact = source_dir / "strict_grpo_0.pt"
    context = source_dir / "strict_grpo_replay_context_0.pt"
    torch.save({"replay_context_path": context.name}, artifact)
    context.write_bytes(b"context")
    groups_jsonl = tmp_path / "groups.jsonl"
    _write_group(groups_jsonl, [artifact])

    report = audit_grpo_artifact_storage(
        groups_jsonl,
        inspect_replay_contexts=True,
        include_replay_context_mapping=False,
    )

    assert "replay_context_mapping" not in report
    assert report["unique_replay_context_count"] == 1


def test_audit_grpo_artifact_storage_reports_invalid_replay_context_artifact(tmp_path):
    artifact = tmp_path / "strict_grpo_0.pt"
    artifact.write_text("not a torch artifact", encoding="utf-8")
    groups_jsonl = tmp_path / "groups.jsonl"
    _write_group(groups_jsonl, [artifact])

    report = audit_grpo_artifact_storage(groups_jsonl, inspect_replay_contexts=True)

    assert report["replay_context_ref_count"] == 0
    assert report["replay_context_error_count"] == 1
    assert report["replay_context_errors"][0]["artifact_path"] == str(artifact)


def test_audit_grpo_artifact_storage_cli_fails_over_budget(tmp_path):
    artifact = tmp_path / "strict_grpo_0.pt"
    artifact.write_bytes(b"artifact")
    groups_jsonl = tmp_path / "groups.jsonl"
    _write_group(groups_jsonl, [artifact])

    result = subprocess.run(
        [
            sys.executable,
            "tools/audit_grpo_artifact_storage.py",
            str(groups_jsonl),
            "--max-resolved-gb",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert '"ok": false' in result.stdout


def test_audit_grpo_artifact_storage_cli_prints_compact_summary(tmp_path):
    artifact = tmp_path / "strict_grpo_0.pt"
    artifact.write_bytes(b"artifact")
    groups_jsonl = tmp_path / "groups.jsonl"
    _write_group(groups_jsonl, [artifact])

    result = subprocess.run(
        [
            sys.executable,
            "tools/audit_grpo_artifact_storage.py",
            str(groups_jsonl),
            "--print-summary",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["artifact_ref_count"] == 2
    assert "tasks" not in summary
