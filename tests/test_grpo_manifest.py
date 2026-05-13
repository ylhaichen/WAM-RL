from __future__ import annotations

import json
from dataclasses import dataclass

from wan_va.rl.group_builder import build_grpo_groups
from wan_va.rl.manifest import build_grpo_manifest, write_grpo_manifest
from wan_va.rl.validation import validate_rollout_records


@dataclass
class DummyRollout:
    task: str
    group_id: str
    sample_idx: int
    group_size: int
    reward: float
    success: bool
    record_path: str
    strict_grpo_ready: bool = True
    strict_grpo_artifact_paths: list[str] | None = None
    env_seed: int = 10000
    sampling_seed: int = 730000


def _record(group_id: str, sample_idx: int, reward: float) -> DummyRollout:
    return DummyRollout(
        task="open_microwave",
        group_id=group_id,
        sample_idx=sample_idx,
        group_size=2,
        reward=reward,
        success=reward > 0.0,
        record_path=f"/tmp/{group_id}_{sample_idx}.json",
        strict_grpo_artifact_paths=[f"/tmp/strict_{sample_idx}.pt"],
    )


def test_build_grpo_manifest_summarizes_records_groups_and_validation(tmp_path):
    records = [_record("mixed", 0, 0.0), _record("mixed", 1, 1.0)]
    groups = build_grpo_groups(records, expected_group_size=2, require_strict_artifacts=True)
    validation = validate_rollout_records(records, expected_group_size=2, require_strict_artifacts=True)

    manifest = build_grpo_manifest(
        roots=[tmp_path],
        records=records,
        group_result=groups,
        validation_report=validation,
        groups_jsonl=tmp_path / "groups" / "grpo_groups.jsonl",
        summary_json=tmp_path / "groups" / "grpo_summary.json",
    )
    out = tmp_path / "groups" / "grpo_manifest.json"
    write_grpo_manifest(manifest, out)

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["record_count"] == 2
    assert data["group_summary"]["mixed_groups"] == 1
    assert data["validation"]["ok"] is True
    assert data["outputs"]["groups_jsonl"].endswith("grpo_groups.jsonl")
