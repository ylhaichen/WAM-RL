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
    _validate_strict_artifact_shapes(data, expanded)
    _validate_strict_artifact_values(data, expanded)
    return data


def _validate_strict_artifact_shapes(data: dict, path: Path) -> None:
    state_keys = ("action_xt", "action_xt_next", "transition_mean", "logprob_mask")
    state_shapes = {key: _shape_of(data[key]) for key in state_keys}
    known_state_shapes = {shape for shape in state_shapes.values() if shape is not None}
    if len(known_state_shapes) > 1:
        raise ValueError(f"strict artifact {path} has incompatible state tensor shapes: {state_shapes}")

    state_shape = next(iter(known_state_shapes), None)
    if state_shape is not None and len(state_shape) == 0:
        raise ValueError(f"strict artifact {path} state tensors must have a batch dimension")
    batch_size = None if state_shape is None else state_shape[0]

    for key in ("transition_std", "old_logprob_sum", "old_logprob_mean", "old_logprob_count"):
        shape = _shape_of(data[key])
        if shape is None or batch_size is None:
            continue
        if shape not in {(), (batch_size,)}:
            raise ValueError(
                f"strict artifact {path} field {key} must be scalar or batch vector of length {batch_size}; "
                f"got shape {shape}"
            )


def _validate_strict_artifact_values(data: dict, path: Path) -> None:
    for key in ("action_xt", "action_xt_next", "transition_mean", "old_logprob_sum", "old_logprob_mean", "old_logprob_count"):
        _validate_tensor_finite(data[key], path=path, key=key)
    _validate_tensor_finite(data["transition_std"], path=path, key="transition_std", positive=True)


def _shape_of(value: object) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except TypeError:
        return None


def _validate_tensor_finite(value: object, *, path: Path, key: str, positive: bool = False) -> None:
    try:
        import torch
    except ImportError:
        return

    try:
        tensor = torch.as_tensor(value)
    except Exception:
        return
    if tensor.numel() == 0:
        raise ValueError(f"strict artifact {path} field {key} must not be empty")
    numeric = tensor.float()
    if not torch.isfinite(numeric).all():
        raise ValueError(f"strict artifact {path} field {key} contains non-finite values")
    if positive and not (numeric > 0).all():
        raise ValueError(f"strict artifact {path} field {key} must be positive")


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
