from __future__ import annotations

from dataclasses import dataclass

from wan_va.rl.group_builder import build_grpo_groups, canonicalize_legacy_group_id


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


def _record(group_id: str, sample_idx: int, reward: float, *, ready: bool = True) -> DummyRollout:
    return DummyRollout(
        task="open_microwave",
        group_id=group_id,
        sample_idx=sample_idx,
        group_size=4,
        reward=reward,
        success=reward > 0.0,
        record_path=f"/tmp/{group_id}_{sample_idx}.json",
        strict_grpo_ready=ready,
        strict_grpo_artifact_paths=[f"/tmp/strict_{sample_idx}.pt"] if ready else [],
    )


def test_build_grpo_groups_keeps_mixed_groups_and_computes_advantages():
    result = build_grpo_groups(
        [
            _record("mixed", 2, 1.0),
            _record("mixed", 0, 0.0),
            _record("mixed", 1, 1.0),
            _record("mixed", 3, 0.0),
            _record("all_success", 0, 1.0),
            _record("all_success", 1, 1.0),
            _record("all_success", 2, 1.0),
            _record("all_success", 3, 1.0),
            _record("all_failure", 0, 0.0),
            _record("all_failure", 1, 0.0),
            _record("all_failure", 2, 0.0),
            _record("all_failure", 3, 0.0),
        ],
        expected_group_size=4,
    )

    assert result.summary.total_groups == 3
    assert result.summary.mixed_groups == 1
    assert result.summary.skipped_all_success == 1
    assert result.summary.skipped_all_failure == 1
    assert result.summary.mixed_group_ratio == 1 / 3

    group = result.groups[0]
    assert group.group_id == "mixed"
    assert [sample.sample_idx for sample in group.samples] == [0, 1, 2, 3]
    assert [sample.reward for sample in group.samples] == [0.0, 1.0, 1.0, 0.0]
    assert [sample.advantage for sample in group.samples] == [-1.0, 1.0, 1.0, -1.0]


def test_canonicalize_legacy_group_id_strips_prompt_hash_suffix():
    assert (
        canonicalize_legacy_group_id("open_microwave_seed200010000_group000000_3afe141b81")
        == "open_microwave_seed200010000_group000000"
    )
    assert canonicalize_legacy_group_id("already_stable_group") == "already_stable_group"


def test_build_grpo_groups_can_repair_legacy_hashed_group_ids():
    records = [
        _record("open_microwave_seed1_group000000_a000000000", 0, 0.0),
        _record("open_microwave_seed1_group000000_b000000000", 1, 1.0),
        _record("open_microwave_seed1_group000000_c000000000", 2, 0.0),
        _record("open_microwave_seed1_group000000_d000000000", 3, 1.0),
    ]

    result = build_grpo_groups(records, expected_group_size=4, canonicalize_legacy_ids=True)

    assert result.summary.mixed_groups == 1
    assert result.groups[0].group_id == "open_microwave_seed1_group000000"


def test_build_grpo_groups_skips_non_contiguous_sample_indices():
    result = build_grpo_groups(
        [
            _record("bad_indices", 0, 0.0),
            _record("bad_indices", 2, 1.0),
        ],
        expected_group_size=2,
    )

    assert result.summary.mixed_groups == 0
    assert result.summary.skipped_incomplete == 1


def test_build_grpo_groups_can_require_strict_artifacts_and_skip_incomplete_groups():
    result = build_grpo_groups(
        [
            _record("missing_artifact", 0, 0.0, ready=True),
            _record("missing_artifact", 1, 1.0, ready=False),
            _record("complete", 0, 0.0, ready=True),
            _record("complete", 1, 1.0, ready=True),
        ],
        expected_group_size=2,
        require_strict_artifacts=True,
    )

    assert [group.group_id for group in result.groups] == ["complete"]
    assert result.summary.total_groups == 2
    assert result.summary.mixed_groups == 1
    assert result.summary.skipped_missing_artifacts == 1
