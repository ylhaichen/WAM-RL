import json
import os
from pathlib import Path

from tools.audit_grpo_artifact_storage import audit_grpo_artifact_storage


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
