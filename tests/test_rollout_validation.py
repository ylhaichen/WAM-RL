from __future__ import annotations

from dataclasses import dataclass

from wan_va.rl.validation import validate_rollout_records


@dataclass
class DummyRollout:
    task: str
    group_id: str
    sample_idx: int | None
    group_size: int | None
    reward: float
    success: bool
    record_path: str
    strict_grpo_ready: bool = True
    strict_grpo_artifact_paths: list[str] | None = None
    env_seed: int = 10000
    sampling_seed: int = 730000


def _record(group_id: str, sample_idx: int | None, reward: float, *, artifact_path: str = "/tmp/a.pt"):
    return DummyRollout(
        task="open_microwave",
        group_id=group_id,
        sample_idx=sample_idx,
        group_size=2,
        reward=reward,
        success=reward > 0.0,
        record_path=f"/tmp/{group_id}_{sample_idx}.json",
        strict_grpo_artifact_paths=[artifact_path],
    )


def test_validate_rollout_records_accepts_complete_mixed_group_without_path_check():
    report = validate_rollout_records(
        [_record("g0", 0, 0.0), _record("g0", 1, 1.0)],
        expected_group_size=2,
        require_strict_artifacts=True,
        require_existing_artifacts=False,
    )

    assert report.ok
    assert report.error_count == 0
    assert report.group_count == 1
    assert report.record_count == 2


def test_validate_rollout_records_reports_duplicate_missing_and_incomplete_groups(tmp_path):
    existing_artifact = tmp_path / "strict.pt"
    existing_artifact.write_bytes(b"pt")
    report = validate_rollout_records(
        [
            _record("dup", 0, 0.0, artifact_path=str(existing_artifact)),
            _record("dup", 0, 1.0, artifact_path=str(existing_artifact)),
            _record("missing_sample", None, 0.0, artifact_path=str(existing_artifact)),
            _record("missing_artifact", 0, 1.0, artifact_path=str(tmp_path / "missing.pt")),
            _record("incomplete", 0, 0.0, artifact_path=str(existing_artifact)),
        ],
        expected_group_size=2,
        require_strict_artifacts=True,
        require_existing_artifacts=True,
    )

    codes = {issue.code for issue in report.issues}
    assert report.ok is False
    assert "duplicate_sample_idx" in codes
    assert "missing_sample_idx" in codes
    assert "missing_artifact_path" in codes
    assert "incomplete_group" in codes
