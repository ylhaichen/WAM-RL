"""Framework-neutral schemas for offline GRPO rollout groups."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GrpoSample:
    task: str
    group_id: str
    sample_idx: int
    reward: float
    advantage: float
    success: bool
    record_path: str
    env_seed: int | None = None
    sampling_seed: int | None = None
    strict_grpo_scope: str = ""
    strict_grpo_artifact_count: int = 0
    strict_grpo_artifact_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        data = asdict(self)
        data["strict_grpo_artifact_paths"] = list(self.strict_grpo_artifact_paths)
        return data


@dataclass(frozen=True)
class GrpoGroup:
    group_id: str
    task: str
    group_size: int
    reward_mean: float
    reward_std: float
    samples: tuple[GrpoSample, ...]

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "task": self.task,
            "group_size": self.group_size,
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
            "samples": [sample.to_dict() for sample in self.samples],
        }


@dataclass(frozen=True)
class GrpoSummary:
    total_groups: int
    mixed_groups: int
    skipped_all_success: int
    skipped_all_failure: int
    skipped_incomplete: int
    skipped_missing_artifacts: int

    @property
    def mixed_group_ratio(self) -> float:
        if self.total_groups == 0:
            return 0.0
        return self.mixed_groups / self.total_groups

    def to_dict(self) -> dict:
        data = asdict(self)
        data["mixed_group_ratio"] = self.mixed_group_ratio
        return data


@dataclass(frozen=True)
class GrpoGroupBuildResult:
    groups: tuple[GrpoGroup, ...]
    summary: GrpoSummary
