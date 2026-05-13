import json

from tools.validate_grpo_dataset import validate_dataset


def test_validate_grpo_dataset_tool_reports_ok_for_existing_artifact(tmp_path):
    artifact = tmp_path / "strict.pt"
    artifact.write_bytes(b"pt")
    groups = tmp_path / "grpo_groups.jsonl"
    groups.write_text(
        json.dumps(
            {
                "group_id": "g0",
                "task": "open_microwave",
                "samples": [
                    {
                        "sample_idx": 0,
                        "reward": 1.0,
                        "advantage": 1.0,
                        "record_path": "/tmp/r0.json",
                        "strict_grpo_artifact_paths": [str(artifact)],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_dataset(groups, require_existing_artifacts=True, inspect_artifacts=False)

    assert report.ok
    assert report.transition_count == 1


def test_validate_grpo_dataset_tool_reports_missing_artifact(tmp_path):
    groups = tmp_path / "grpo_groups.jsonl"
    groups.write_text(
        json.dumps(
            {
                "group_id": "g0",
                "task": "open_microwave",
                "samples": [
                    {
                        "sample_idx": 0,
                        "reward": 1.0,
                        "advantage": 1.0,
                        "record_path": "/tmp/r0.json",
                        "strict_grpo_artifact_paths": [str(tmp_path / "missing.pt")],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_dataset(groups, require_existing_artifacts=True, inspect_artifacts=False)

    assert report.ok is False
    assert report.issues[0].code == "missing_transition_artifact"
