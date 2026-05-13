"""Offline dataset utilities for denoising-step GRPO training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable


REQUIRED_STRICT_ARTIFACT_KEYS = frozenset(
    {
        "schema_version",
        "scope",
        "sampling_seed",
        "frame_st_id",
        "timestep",
        "action_xt",
        "action_xt_next",
        "transition_mean",
        "transition_std",
        "old_logprob_sum",
        "old_logprob_mean",
        "old_logprob_count",
        "logprob_mask",
    }
)


@dataclass(frozen=True)
class GrpoTransitionRef:
    task: str
    group_id: str
    sample_idx: int
    reward: float
    advantage: float
    record_path: str
    artifact_path: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DatasetIssue:
    severity: str
    code: str
    message: str
    group_id: str = ""
    sample_idx: int | None = None
    artifact_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DatasetValidationReport:
    transition_count: int
    issues: tuple[DatasetIssue, ...]

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def ok(self) -> bool:
        return self.error_count == 0

    def to_dict(self) -> dict:
        return {
            "transition_count": self.transition_count,
            "error_count": self.error_count,
            "ok": self.ok,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def read_grpo_group_dicts(path: Path) -> Iterable[dict]:
    with path.expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_transition_refs(path: Path) -> Iterable[GrpoTransitionRef]:
    for group in read_grpo_group_dicts(path):
        task = str(group.get("task", ""))
        group_id = str(group["group_id"])
        for sample in group.get("samples", []):
            for artifact_path in sample.get("strict_grpo_artifact_paths", []) or []:
                yield GrpoTransitionRef(
                    task=task,
                    group_id=group_id,
                    sample_idx=int(sample["sample_idx"]),
                    reward=float(sample["reward"]),
                    advantage=float(sample["advantage"]),
                    record_path=str(sample["record_path"]),
                    artifact_path=str(artifact_path),
                )


def load_strict_artifact(path: Path, *, loader: Callable[[Path], dict] | None = None) -> dict:
    expanded = path.expanduser()
    if loader is None:
        import torch

        data = torch.load(expanded, map_location="cpu")
    else:
        data = loader(expanded)

    if not isinstance(data, dict):
        raise ValueError(f"strict artifact must be a dict: {expanded}")
    missing = sorted(REQUIRED_STRICT_ARTIFACT_KEYS - set(data))
    if missing:
        raise ValueError(f"strict artifact {expanded} missing required strict artifact keys: {missing}")
    return data


def validate_transition_refs(
    refs: Iterable[GrpoTransitionRef],
    *,
    require_existing_artifacts: bool = True,
) -> DatasetValidationReport:
    items = list(refs)
    issues: list[DatasetIssue] = []
    for ref in items:
        if require_existing_artifacts and not Path(ref.artifact_path).expanduser().exists():
            issues.append(
                DatasetIssue(
                    severity="error",
                    code="missing_transition_artifact",
                    message=f"transition artifact does not exist: {ref.artifact_path}",
                    group_id=ref.group_id,
                    sample_idx=ref.sample_idx,
                    artifact_path=ref.artifact_path,
                )
            )
    return DatasetValidationReport(transition_count=len(items), issues=tuple(issues))


def inspect_strict_artifacts(
    refs: Iterable[GrpoTransitionRef],
    *,
    loader: Callable[[Path], dict] | None = None,
) -> DatasetValidationReport:
    items = list(refs)
    issues: list[DatasetIssue] = []
    for ref in items:
        try:
            load_strict_artifact(Path(ref.artifact_path), loader=loader)
        except Exception as exc:
            issues.append(
                DatasetIssue(
                    severity="error",
                    code="invalid_transition_artifact",
                    message=str(exc),
                    group_id=ref.group_id,
                    sample_idx=ref.sample_idx,
                    artifact_path=ref.artifact_path,
                )
            )
    return DatasetValidationReport(transition_count=len(items), issues=tuple(issues))
