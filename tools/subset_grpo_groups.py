#!/usr/bin/env python3
"""Create small GRPO group JSONL subsets for replay debugging/training."""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
from typing import Iterable

try:
    from tools._repo_root import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _repo_root import ensure_repo_root_on_path

ensure_repo_root_on_path()

from wan_va.rl.dataset import read_grpo_group_dicts


def subset_grpo_groups(
    groups_jsonl: Path,
    *,
    tasks: set[str] | None = None,
    max_groups: int | None = None,
    samples_per_reward: int | None = None,
    max_artifacts_per_sample: int | None = None,
    require_artifacts: bool = False,
    preserve_advantages: bool = False,
    group_id_suffix: str = "_subset",
    preserve_group_id: bool = False,
    include_file_sizes: bool = False,
) -> tuple[list[dict], dict]:
    """Return selected groups and a manifest.

    The function only rewrites JSON group references. It never copies or deletes
    artifact files, which keeps it safe to run against large replay-context
    datasets on Scratch.
    """

    if max_groups is not None and max_groups <= 0:
        raise ValueError("max_groups must be positive when set")
    if samples_per_reward is not None and samples_per_reward <= 0:
        raise ValueError("samples_per_reward must be positive when set")
    if max_artifacts_per_sample is not None and max_artifacts_per_sample <= 0:
        raise ValueError("max_artifacts_per_sample must be positive when set")

    source_groups = list(read_grpo_group_dicts(groups_jsonl.expanduser()))
    selected: list[dict] = []
    skipped_task = 0
    skipped_unmixed = 0
    skipped_empty = 0

    for group in source_groups:
        task = str(group.get("task", ""))
        if tasks is not None and task not in tasks:
            skipped_task += 1
            continue
        samples = _select_samples(
            group.get("samples", []),
            samples_per_reward=samples_per_reward,
            max_artifacts_per_sample=max_artifacts_per_sample,
            require_artifacts=require_artifacts,
        )
        if not samples:
            skipped_empty += 1
            continue
        if not _is_mixed(samples):
            skipped_unmixed += 1
            continue
        selected_group = _rewrite_group(
            group,
            samples,
            preserve_advantages=preserve_advantages,
            group_id_suffix=group_id_suffix,
            preserve_group_id=preserve_group_id,
        )
        selected.append(selected_group)
        if max_groups is not None and len(selected) >= max_groups:
            break

    manifest = _build_manifest(
        source_path=groups_jsonl.expanduser(),
        source_groups=source_groups,
        selected_groups=selected,
        tasks=tasks,
        max_groups=max_groups,
        samples_per_reward=samples_per_reward,
        max_artifacts_per_sample=max_artifacts_per_sample,
        require_artifacts=require_artifacts,
        preserve_advantages=preserve_advantages,
        preserve_group_id=preserve_group_id,
        group_id_suffix=group_id_suffix,
        skipped_task=skipped_task,
        skipped_unmixed=skipped_unmixed,
        skipped_empty=skipped_empty,
        include_file_sizes=include_file_sizes,
    )
    return selected, manifest


def write_outputs(groups: list[dict], *, out_jsonl: Path, out_manifest: Path | None, manifest: dict) -> None:
    out_jsonl.expanduser().parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.expanduser().open("w", encoding="utf-8") as f:
        for group in groups:
            f.write(json.dumps(group, ensure_ascii=False) + "\n")
    if out_manifest is not None:
        out_manifest.expanduser().parent.mkdir(parents=True, exist_ok=True)
        out_manifest.expanduser().write_text(
            json.dumps({**manifest, "output_jsonl": str(out_jsonl.expanduser())}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )


def _select_samples(
    samples: Iterable[dict],
    *,
    samples_per_reward: int | None,
    max_artifacts_per_sample: int | None,
    require_artifacts: bool,
) -> list[dict]:
    copied = []
    for sample in samples:
        item = dict(sample)
        paths = list(item.get("strict_grpo_artifact_paths", []) or [])
        if max_artifacts_per_sample is not None:
            paths = paths[:max_artifacts_per_sample]
        item["strict_grpo_artifact_paths"] = paths
        item["strict_grpo_artifact_count"] = len(paths)
        if require_artifacts and not paths:
            continue
        copied.append(item)

    if samples_per_reward is None:
        return copied

    failures = [sample for sample in copied if float(sample.get("reward", 0.0)) <= 0.0]
    successes = [sample for sample in copied if float(sample.get("reward", 0.0)) > 0.0]
    selected = failures[:samples_per_reward] + successes[:samples_per_reward]
    return sorted(selected, key=lambda sample: int(sample.get("sample_idx", 0)))


def _rewrite_group(
    group: dict,
    samples: list[dict],
    *,
    preserve_advantages: bool,
    group_id_suffix: str,
    preserve_group_id: bool,
) -> dict:
    rewards = [float(sample.get("reward", 0.0)) for sample in samples]
    reward_mean = sum(rewards) / len(rewards)
    reward_std = sqrt(sum((reward - reward_mean) ** 2 for reward in rewards) / len(rewards))
    group_id = str(group["group_id"])
    if not preserve_group_id:
        group_id = group_id + group_id_suffix

    rewritten_samples = []
    for sample in samples:
        item = dict(sample)
        item["group_id"] = group_id
        item["success"] = bool(float(item.get("reward", 0.0)) > 0.0)
        if not preserve_advantages:
            item["advantage"] = _advantage(float(item.get("reward", 0.0)), reward_mean, reward_std)
        rewritten_samples.append(item)

    return {
        **{key: value for key, value in group.items() if key != "samples"},
        "group_id": group_id,
        "group_size": len(rewritten_samples),
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "samples": rewritten_samples,
    }


def _advantage(reward: float, reward_mean: float, reward_std: float, *, eps: float = 1e-6, clip: float = 2.0) -> float:
    denom = reward_std if reward_std > 0.0 else eps
    value = (reward - reward_mean) / denom
    return max(-clip, min(clip, value))


def _is_mixed(samples: list[dict]) -> bool:
    success_count = sum(1 for sample in samples if float(sample.get("reward", 0.0)) > 0.0)
    return 0 < success_count < len(samples)


def _build_manifest(
    *,
    source_path: Path,
    source_groups: list[dict],
    selected_groups: list[dict],
    tasks: set[str] | None,
    max_groups: int | None,
    samples_per_reward: int | None,
    max_artifacts_per_sample: int | None,
    require_artifacts: bool,
    preserve_advantages: bool,
    preserve_group_id: bool,
    group_id_suffix: str,
    skipped_task: int,
    skipped_unmixed: int,
    skipped_empty: int,
    include_file_sizes: bool,
) -> dict:
    source_artifacts = _artifact_paths(source_groups)
    selected_artifacts = _artifact_paths(selected_groups)
    manifest = {
        "source_jsonl": str(source_path),
        "options": {
            "tasks": sorted(tasks) if tasks is not None else None,
            "max_groups": max_groups,
            "samples_per_reward": samples_per_reward,
            "max_artifacts_per_sample": max_artifacts_per_sample,
            "require_artifacts": require_artifacts,
            "preserve_advantages": preserve_advantages,
            "preserve_group_id": preserve_group_id,
            "group_id_suffix": group_id_suffix,
        },
        "input_group_count": len(source_groups),
        "output_group_count": len(selected_groups),
        "input_sample_count": _sample_count(source_groups),
        "output_sample_count": _sample_count(selected_groups),
        "input_artifact_ref_count": len(source_artifacts),
        "output_artifact_ref_count": len(selected_artifacts),
        "output_unique_artifact_count": len(set(selected_artifacts)),
        "skipped_task_group_count": skipped_task,
        "skipped_empty_group_count": skipped_empty,
        "skipped_unmixed_group_count": skipped_unmixed,
        "tasks": _task_summary(selected_groups),
    }
    if include_file_sizes:
        manifest["output_existing_artifact_bytes"] = _existing_bytes(selected_artifacts)
    return manifest


def _sample_count(groups: list[dict]) -> int:
    return sum(len(group.get("samples", []) or []) for group in groups)


def _artifact_paths(groups: list[dict]) -> list[str]:
    paths: list[str] = []
    for group in groups:
        for sample in group.get("samples", []) or []:
            paths.extend(str(path) for path in sample.get("strict_grpo_artifact_paths", []) or [])
    return paths


def _existing_bytes(paths: list[str]) -> int:
    total = 0
    for value in set(paths):
        path = Path(value).expanduser()
        if path.exists():
            total += path.stat().st_size
    return total


def _task_summary(groups: list[dict]) -> list[dict]:
    by_task: dict[str, dict] = {}
    for group in groups:
        task = str(group.get("task", ""))
        item = by_task.setdefault(
            task,
            {
                "task": task,
                "group_count": 0,
                "sample_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "artifact_ref_count": 0,
            },
        )
        samples = group.get("samples", []) or []
        item["group_count"] += 1
        item["sample_count"] += len(samples)
        for sample in samples:
            if float(sample.get("reward", 0.0)) > 0.0:
                item["success_count"] += 1
            else:
                item["failure_count"] += 1
            item["artifact_ref_count"] += len(sample.get("strict_grpo_artifact_paths", []) or [])
    return [by_task[key] for key in sorted(by_task)]


def _task_filter(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    tasks: set[str] = set()
    for value in values:
        tasks.update(item for item in value.split() if item)
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Create small GRPO group JSONL subsets.")
    parser.add_argument("groups_jsonl", type=Path, help="Input GRPO groups JSONL.")
    parser.add_argument("--tasks", nargs="*", help="Optional task names to keep.")
    parser.add_argument("--max-groups", type=int, help="Maximum number of mixed groups to write.")
    parser.add_argument("--samples-per-reward", type=int, help="Keep up to N failures and N successes per group.")
    parser.add_argument("--max-artifacts-per-sample", type=int, help="Keep only the first N artifact refs per sample.")
    parser.add_argument("--require-artifacts", action="store_true", help="Drop samples that have no artifact refs.")
    parser.add_argument("--preserve-advantages", action="store_true", help="Keep original sample advantages.")
    parser.add_argument("--preserve-group-id", action="store_true", help="Do not suffix output group ids.")
    parser.add_argument("--group-id-suffix", default="_subset", help="Suffix appended to output group ids.")
    parser.add_argument("--include-file-sizes", action="store_true", help="Stat selected direct artifact refs.")
    parser.add_argument("--out-jsonl", type=Path, required=True, help="Output subset groups JSONL.")
    parser.add_argument("--out-manifest", type=Path, help="Optional output manifest JSON.")
    args = parser.parse_args()

    groups, manifest = subset_grpo_groups(
        args.groups_jsonl,
        tasks=_task_filter(args.tasks),
        max_groups=args.max_groups,
        samples_per_reward=args.samples_per_reward,
        max_artifacts_per_sample=args.max_artifacts_per_sample,
        require_artifacts=args.require_artifacts,
        preserve_advantages=args.preserve_advantages,
        group_id_suffix=args.group_id_suffix,
        preserve_group_id=args.preserve_group_id,
        include_file_sizes=args.include_file_sizes,
    )
    write_outputs(groups, out_jsonl=args.out_jsonl, out_manifest=args.out_manifest, manifest=manifest)
    print(json.dumps({**manifest, "output_jsonl": str(args.out_jsonl.expanduser())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
