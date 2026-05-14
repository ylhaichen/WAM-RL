"""Build dynamic-sampling GRPO groups from RoboTwin rollout records."""

from __future__ import annotations

import re
from collections import defaultdict
from math import sqrt
from typing import Iterable

from .trajectory_schema import GrpoGroup, GrpoGroupBuildResult, GrpoSample, GrpoSummary

_LEGACY_PROMPT_HASH_SUFFIX = re.compile(r"^(?P<prefix>.+_group\d{6})_[0-9a-f]{10}$")


def build_grpo_groups(
    records: Iterable[object],
    *,
    expected_group_size: int | None = None,
    require_strict_artifacts: bool = False,
    canonicalize_legacy_ids: bool = False,
    advantage_eps: float = 1e-6,
    advantage_clip: float = 2.0,
) -> GrpoGroupBuildResult:
    by_group: dict[str, list[object]] = defaultdict(list)
    for record in records:
        group_id = str(getattr(record, "group_id", ""))
        if not group_id:
            continue
        if canonicalize_legacy_ids:
            group_id = canonicalize_legacy_group_id(group_id)
        by_group[group_id].append(record)

    groups: list[GrpoGroup] = []
    skipped_all_success = 0
    skipped_all_failure = 0
    skipped_incomplete = 0
    skipped_missing_artifacts = 0

    for group_id in sorted(by_group):
        items = sorted(by_group[group_id], key=lambda item: _sample_idx(item))

        if expected_group_size is not None and len(items) != expected_group_size:
            skipped_incomplete += 1
            continue
        if expected_group_size is not None and {_sample_idx(item) for item in items} != set(range(expected_group_size)):
            skipped_incomplete += 1
            continue

        if require_strict_artifacts and not all(_has_strict_artifacts(item) for item in items):
            skipped_missing_artifacts += 1
            continue

        rewards = [float(getattr(item, "reward")) for item in items]
        success_count = sum(1 for reward in rewards if reward > 0.0)
        if success_count == 0:
            skipped_all_failure += 1
            continue
        if success_count == len(items):
            skipped_all_success += 1
            continue

        reward_mean = sum(rewards) / len(rewards)
        reward_std = sqrt(sum((reward - reward_mean) ** 2 for reward in rewards) / len(rewards))
        advantage_denominator = reward_std if reward_std > 0.0 else advantage_eps
        samples = tuple(
            GrpoSample(
                task=str(getattr(item, "task", "")),
                group_id=group_id,
                sample_idx=_sample_idx(item),
                reward=float(getattr(item, "reward")),
                advantage=_clip((float(getattr(item, "reward")) - reward_mean) / advantage_denominator, advantage_clip),
                success=bool(getattr(item, "success")),
                record_path=str(getattr(item, "record_path")),
                env_seed=_optional_int(getattr(item, "env_seed", None)),
                sampling_seed=_optional_int(getattr(item, "sampling_seed", None)),
                strict_grpo_artifact_paths=tuple(str(path) for path in (getattr(item, "strict_grpo_artifact_paths", None) or [])),
            )
            for item in items
        )
        groups.append(
            GrpoGroup(
                group_id=group_id,
                task=samples[0].task,
                group_size=len(samples),
                reward_mean=reward_mean,
                reward_std=reward_std,
                samples=samples,
            )
        )

    summary = GrpoSummary(
        total_groups=len(by_group),
        mixed_groups=len(groups),
        skipped_all_success=skipped_all_success,
        skipped_all_failure=skipped_all_failure,
        skipped_incomplete=skipped_incomplete,
        skipped_missing_artifacts=skipped_missing_artifacts,
    )
    return GrpoGroupBuildResult(groups=tuple(groups), summary=summary)


def canonicalize_legacy_group_id(group_id: str) -> str:
    """Strip the legacy prompt-hash suffix from generated rollout group ids.

    Early grouped rollout jobs generated ids from the rendered prompt text:
    ``task_seed<env>_group<idx>_<sha1-prefix>``. Prompt rendering can differ
    across samples, so those ids split one intended dynamic-sampling group into
    many one-record groups. The canonicalizer only removes that exact suffix
    shape and leaves all other ids untouched.
    """
    match = _LEGACY_PROMPT_HASH_SUFFIX.match(group_id)
    if match is None:
        return group_id
    return match.group("prefix")


def _sample_idx(record: object) -> int:
    value = getattr(record, "sample_idx", None)
    if value is None:
        return 0
    return int(value)


def _has_strict_artifacts(record: object) -> bool:
    if not bool(getattr(record, "strict_grpo_ready", False)):
        return False
    return bool(getattr(record, "strict_grpo_artifact_paths", None) or [])


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _clip(value: float, limit: float) -> float:
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value
