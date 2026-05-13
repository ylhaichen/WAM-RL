"""Validation for collected RoboTwin GRPO rollout records."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    record_path: str = ""
    group_id: str = ""
    sample_idx: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ValidationReport:
    record_count: int
    group_count: int
    issues: tuple[ValidationIssue, ...]

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    @property
    def ok(self) -> bool:
        return self.error_count == 0

    def to_dict(self) -> dict:
        return {
            "record_count": self.record_count,
            "group_count": self.group_count,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "ok": self.ok,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def validate_rollout_records(
    records: Iterable[object],
    *,
    expected_group_size: int | None = None,
    require_strict_artifacts: bool = False,
    require_existing_artifacts: bool = False,
) -> ValidationReport:
    items = list(records)
    issues: list[ValidationIssue] = []
    by_group: dict[str, list[object]] = defaultdict(list)

    for record in items:
        group_id = _string_attr(record, "group_id")
        sample_idx = _optional_int_attr(record, "sample_idx")
        record_path = _string_attr(record, "record_path")

        if not group_id:
            issues.append(_issue("error", "missing_group_id", "rollout record is missing group_id", record))
            continue
        by_group[group_id].append(record)

        if sample_idx is None:
            issues.append(_issue("error", "missing_sample_idx", "rollout record is missing sample_idx", record))
        if _optional_int_attr(record, "group_size") is None:
            issues.append(_issue("error", "missing_group_size", "rollout record is missing group_size", record))

        reward = float(getattr(record, "reward"))
        success = bool(getattr(record, "success"))
        if reward not in {0.0, 1.0}:
            issues.append(_issue("error", "non_binary_reward", f"reward must be 0.0 or 1.0, got {reward}", record))
        if (reward > 0.0) != success:
            issues.append(_issue("error", "reward_success_mismatch", "reward and success flag disagree", record))

        artifact_paths = _artifact_paths(record)
        if require_strict_artifacts and not artifact_paths:
            issues.append(_issue("error", "missing_strict_artifacts", "strict GRPO artifacts are required", record))
        if require_strict_artifacts and not bool(getattr(record, "strict_grpo_ready", False)):
            issues.append(_issue("error", "strict_grpo_not_ready", "strict_grpo_ready is false", record))
        if require_existing_artifacts:
            for artifact_path in artifact_paths:
                if not Path(artifact_path).expanduser().exists():
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="missing_artifact_path",
                            message=f"strict GRPO artifact path does not exist: {artifact_path}",
                            record_path=record_path,
                            group_id=group_id,
                            sample_idx=sample_idx,
                        )
                    )

    for group_id, group_items in by_group.items():
        sample_indices = [_optional_int_attr(record, "sample_idx") for record in group_items]
        known_indices = [idx for idx in sample_indices if idx is not None]
        duplicate_indices = sorted({idx for idx in known_indices if known_indices.count(idx) > 1})
        for idx in duplicate_indices:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="duplicate_sample_idx",
                    message=f"group has duplicate sample_idx={idx}",
                    group_id=group_id,
                    sample_idx=idx,
                )
            )

        group_sizes = {_optional_int_attr(record, "group_size") for record in group_items if _optional_int_attr(record, "group_size") is not None}
        if len(group_sizes) > 1:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="inconsistent_group_size",
                    message=f"group has inconsistent group_size values: {sorted(group_sizes)}",
                    group_id=group_id,
                )
            )

        target_size = expected_group_size
        if target_size is None and len(group_sizes) == 1:
            target_size = next(iter(group_sizes))
        if target_size is not None and len(group_items) != target_size:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="incomplete_group",
                    message=f"group has {len(group_items)} records, expected {target_size}",
                    group_id=group_id,
                )
            )

    return ValidationReport(
        record_count=len(items),
        group_count=len(by_group),
        issues=tuple(issues),
    )


def _issue(severity: str, code: str, message: str, record: object) -> ValidationIssue:
    return ValidationIssue(
        severity=severity,
        code=code,
        message=message,
        record_path=_string_attr(record, "record_path"),
        group_id=_string_attr(record, "group_id"),
        sample_idx=_optional_int_attr(record, "sample_idx"),
    )


def _artifact_paths(record: object) -> list[str]:
    value = getattr(record, "strict_grpo_artifact_paths", None)
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _string_attr(record: object, name: str) -> str:
    value = getattr(record, name, "")
    if value is None:
        return ""
    return str(value)


def _optional_int_attr(record: object, name: str) -> int | None:
    value = getattr(record, name, None)
    if value is None:
        return None
    return int(value)
