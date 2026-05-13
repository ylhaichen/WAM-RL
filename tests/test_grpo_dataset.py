import json

from wan_va.rl.dataset import (
    REQUIRED_STRICT_ARTIFACT_KEYS,
    inspect_strict_artifacts,
    load_strict_artifact,
    read_transition_refs,
    validate_transition_refs,
)


def test_read_transition_refs_flattens_group_samples(tmp_path):
    path = tmp_path / "grpo_groups.jsonl"
    path.write_text(
        json.dumps(
            {
                "group_id": "g0",
                "task": "open_microwave",
                "samples": [
                    {
                        "sample_idx": 0,
                        "reward": 0.0,
                        "advantage": -1.0,
                        "record_path": "/tmp/r0.json",
                        "strict_grpo_artifact_paths": ["/tmp/a0.pt"],
                    },
                    {
                        "sample_idx": 1,
                        "reward": 1.0,
                        "advantage": 1.0,
                        "record_path": "/tmp/r1.json",
                        "strict_grpo_artifact_paths": ["/tmp/a1.pt", "/tmp/a1b.pt"],
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    refs = list(read_transition_refs(path))

    assert [(ref.sample_idx, ref.artifact_path, ref.advantage) for ref in refs] == [
        (0, "/tmp/a0.pt", -1.0),
        (1, "/tmp/a1.pt", 1.0),
        (1, "/tmp/a1b.pt", 1.0),
    ]


def test_load_strict_artifact_validates_required_keys(tmp_path):
    artifact = {key: object() for key in REQUIRED_STRICT_ARTIFACT_KEYS}
    loaded = load_strict_artifact(tmp_path / "strict.pt", loader=lambda path: artifact)

    assert loaded is artifact


def test_load_strict_artifact_rejects_missing_keys(tmp_path):
    try:
        load_strict_artifact(tmp_path / "strict.pt", loader=lambda path: {"schema_version": 1})
    except ValueError as exc:
        assert "missing required strict artifact keys" in str(exc)
    else:
        raise AssertionError("expected missing-key ValueError")


def test_validate_transition_refs_reports_missing_files(tmp_path):
    path = tmp_path / "grpo_groups.jsonl"
    path.write_text(
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

    report = validate_transition_refs(read_transition_refs(path), require_existing_artifacts=True)

    assert report.ok is False
    assert report.error_count == 1
    assert report.issues[0].code == "missing_transition_artifact"


def test_inspect_strict_artifacts_reports_schema_errors(tmp_path):
    path = tmp_path / "grpo_groups.jsonl"
    artifact_path = tmp_path / "strict.pt"
    artifact_path.write_bytes(b"pt")
    path.write_text(
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
                        "strict_grpo_artifact_paths": [str(artifact_path)],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = inspect_strict_artifacts(
        read_transition_refs(path),
        loader=lambda _: {"schema_version": 1},
    )

    assert report.ok is False
    assert report.issues[0].code == "invalid_transition_artifact"
