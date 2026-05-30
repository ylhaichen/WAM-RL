#!/usr/bin/env python3
"""Summarize GRPO group JSONL files for paper tables and audits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

try:
    from tools._repo_root import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _repo_root import ensure_repo_root_on_path

ensure_repo_root_on_path()


@dataclass(frozen=True)
class TaskGrpoStats:
    task: str
    group_count: int
    sample_count: int
    success_count: int
    failure_count: int
    success_rate: float
    transition_count: int
    mean_transitions_per_sample: float
    mean_reward_std: float
    balance_rate: float
    replay_context_count: int = 0
    replay_context_total_tensor_bytes: int = 0
    replay_context_total_file_bytes: int = 0
    replay_context_total_tensor_gib: float = 0.0
    replay_context_total_file_gib: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GrpoGroupSummary:
    source_files: tuple[str, ...]
    group_count: int
    sample_count: int
    success_count: int
    failure_count: int
    success_rate: float
    transition_count: int
    mean_transitions_per_sample: float
    duplicate_group_id_count: int
    tasks: tuple[TaskGrpoStats, ...]
    replay_context_count: int = 0
    replay_context_total_tensor_bytes: int = 0
    replay_context_total_file_bytes: int = 0
    replay_context_total_tensor_gib: float = 0.0
    replay_context_total_file_gib: float = 0.0

    def to_dict(self) -> dict:
        return {
            "source_files": list(self.source_files),
            "group_count": self.group_count,
            "sample_count": self.sample_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "transition_count": self.transition_count,
            "mean_transitions_per_sample": self.mean_transitions_per_sample,
            "duplicate_group_id_count": self.duplicate_group_id_count,
            "replay_context_count": self.replay_context_count,
            "replay_context_total_tensor_bytes": self.replay_context_total_tensor_bytes,
            "replay_context_total_file_bytes": self.replay_context_total_file_bytes,
            "replay_context_total_tensor_gib": self.replay_context_total_tensor_gib,
            "replay_context_total_file_gib": self.replay_context_total_file_gib,
            "tasks": [task.to_dict() for task in self.tasks],
        }


def read_group_dicts(paths: list[Path]) -> list[dict]:
    groups: list[dict] = []
    for path in paths:
        expanded = path.expanduser()
        with expanded.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    groups.append(json.loads(line))
    return groups


def summarize_groups(paths: list[Path], *, inspect_artifacts: bool = False) -> GrpoGroupSummary:
    groups = read_group_dicts(paths)
    group_ids = [str(group.get("group_id", "")) for group in groups]
    duplicate_group_id_count = sum(count - 1 for count in Counter(group_ids).values() if count > 1)

    task_groups: dict[str, list[dict]] = defaultdict(list)
    for group in groups:
        task = str(group.get("task", ""))
        task_groups[task].append(group)

    task_stats = tuple(
        _summarize_task(task, task_groups[task], inspect_artifacts=inspect_artifacts)
        for task in sorted(task_groups)
    )
    sample_count = sum(item.sample_count for item in task_stats)
    success_count = sum(item.success_count for item in task_stats)
    transition_count = sum(item.transition_count for item in task_stats)
    replay_context_count = sum(item.replay_context_count for item in task_stats)
    replay_context_total_tensor_bytes = sum(item.replay_context_total_tensor_bytes for item in task_stats)
    replay_context_total_file_bytes = sum(item.replay_context_total_file_bytes for item in task_stats)
    return GrpoGroupSummary(
        source_files=tuple(str(path.expanduser()) for path in paths),
        group_count=len(groups),
        sample_count=sample_count,
        success_count=success_count,
        failure_count=sample_count - success_count,
        success_rate=_safe_div(success_count, sample_count),
        transition_count=transition_count,
        mean_transitions_per_sample=_safe_div(transition_count, sample_count),
        duplicate_group_id_count=duplicate_group_id_count,
        replay_context_count=replay_context_count,
        replay_context_total_tensor_bytes=replay_context_total_tensor_bytes,
        replay_context_total_file_bytes=replay_context_total_file_bytes,
        replay_context_total_tensor_gib=_gib(replay_context_total_tensor_bytes),
        replay_context_total_file_gib=_gib(replay_context_total_file_bytes),
        tasks=task_stats,
    )


def _summarize_task(task: str, groups: list[dict], *, inspect_artifacts: bool = False) -> TaskGrpoStats:
    sample_count = 0
    success_count = 0
    transition_count = 0
    replay_context_count = 0
    replay_context_total_tensor_bytes = 0
    replay_context_total_file_bytes = 0
    reward_stds: list[float] = []
    for group in groups:
        reward_stds.append(float(group.get("reward_std", 0.0) or 0.0))
        for sample in group.get("samples", []) or []:
            sample_count += 1
            reward = float(sample.get("reward", 1.0 if sample.get("success") else 0.0))
            if reward > 0.0:
                success_count += 1
            transition_count += _count_sample_transitions(sample, inspect_artifacts=inspect_artifacts)
            context_stats = _sample_replay_context_stats(sample, inspect_artifacts=inspect_artifacts)
            replay_context_count += context_stats["count"]
            replay_context_total_tensor_bytes += context_stats["tensor_bytes"]
            replay_context_total_file_bytes += context_stats["file_bytes"]
    failure_count = sample_count - success_count
    return TaskGrpoStats(
        task=task,
        group_count=len(groups),
        sample_count=sample_count,
        success_count=success_count,
        failure_count=failure_count,
        success_rate=_safe_div(success_count, sample_count),
        transition_count=transition_count,
        mean_transitions_per_sample=_safe_div(transition_count, sample_count),
        mean_reward_std=mean(reward_stds) if reward_stds else 0.0,
        balance_rate=_safe_div(min(success_count, failure_count), sample_count),
        replay_context_count=replay_context_count,
        replay_context_total_tensor_bytes=replay_context_total_tensor_bytes,
        replay_context_total_file_bytes=replay_context_total_file_bytes,
        replay_context_total_tensor_gib=_gib(replay_context_total_tensor_bytes),
        replay_context_total_file_gib=_gib(replay_context_total_file_bytes),
    )


def _count_sample_transitions(sample: dict, *, inspect_artifacts: bool) -> int:
    artifact_paths = sample.get("strict_grpo_artifact_paths", []) or []
    if not inspect_artifacts:
        return len(artifact_paths)

    from wan_va.rl.dataset import count_strict_artifact_transitions, load_strict_artifact

    transition_count = 0
    for artifact_path in artifact_paths:
        artifact = load_strict_artifact(Path(artifact_path))
        transition_count += count_strict_artifact_transitions(artifact)
    return transition_count


def _sample_replay_context_stats(sample: dict, *, inspect_artifacts: bool) -> dict[str, int]:
    context_paths = sample.get("strict_grpo_replay_context_paths", []) or []
    tensor_bytes = int(sample.get("strict_grpo_replay_context_total_tensor_bytes", 0) or 0)
    if not inspect_artifacts:
        return {
            "count": len(context_paths),
            "tensor_bytes": tensor_bytes,
            "file_bytes": 0,
        }

    resolved_paths = set(str(path) for path in context_paths)
    for path in _resolve_replay_context_paths_from_artifacts(sample.get("strict_grpo_artifact_paths", []) or []):
        resolved_paths.add(path)
    return {
        "count": len(resolved_paths),
        "tensor_bytes": tensor_bytes,
        "file_bytes": sum(_path_size(Path(path)) for path in resolved_paths),
    }


def _resolve_replay_context_paths_from_artifacts(artifact_paths: list[str]) -> list[str]:
    if not artifact_paths:
        return []

    import torch

    resolved: list[str] = []
    for artifact_value in artifact_paths:
        artifact_path = Path(artifact_value).expanduser()
        if not artifact_path.exists():
            continue
        artifact = torch.load(artifact_path, map_location="meta")
        if not isinstance(artifact, dict):
            continue
        context_value = artifact.get("replay_context_path")
        if not isinstance(context_value, str) or not context_value:
            continue
        context_path = Path(context_value).expanduser()
        if not context_path.is_absolute():
            context_path = artifact_path.parent / context_path
        resolved.append(str(context_path))
    return resolved


def _path_size(path: Path) -> int:
    try:
        return int(path.expanduser().stat().st_size)
    except FileNotFoundError:
        return 0


def _gib(value: int) -> float:
    return value / 1024**3


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def format_markdown(summary: GrpoGroupSummary) -> str:
    lines = [
        "# GRPO Group Summary",
        "",
        "## Overall",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| group_count | {summary.group_count} |",
        f"| sample_count | {summary.sample_count} |",
        f"| success_count | {summary.success_count} |",
        f"| failure_count | {summary.failure_count} |",
        f"| success_rate | {summary.success_rate:.3f} |",
        f"| transition_count | {summary.transition_count} |",
        f"| mean_transitions_per_sample | {summary.mean_transitions_per_sample:.3f} |",
        f"| duplicate_group_id_count | {summary.duplicate_group_id_count} |",
        f"| replay_context_count | {summary.replay_context_count} |",
        f"| replay_context_total_tensor_gib | {summary.replay_context_total_tensor_gib:.3f} |",
        f"| replay_context_total_file_gib | {summary.replay_context_total_file_gib:.3f} |",
        "",
        "## Per-Task",
        "",
        "| task | groups | samples | success | failure | success_rate | transitions | transitions/sample | reward_std_mean | balance_rate | replay_contexts | replay_context_file_gib |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary.tasks:
        lines.append(
            f"| {item.task} | {item.group_count} | {item.sample_count} | "
            f"{item.success_count} | {item.failure_count} | {item.success_rate:.3f} | "
            f"{item.transition_count} | {item.mean_transitions_per_sample:.3f} | "
            f"{item.mean_reward_std:.3f} | {item.balance_rate:.3f} | "
            f"{item.replay_context_count} | {item.replay_context_total_file_gib:.3f} |"
        )
    lines.append("")
    lines.append("## Source Files")
    lines.append("")
    for source in summary.source_files:
        lines.append(f"- `{source}`")
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, summary: GrpoGroupSummary) -> None:
    path.expanduser().parent.mkdir(parents=True, exist_ok=True)
    with path.expanduser().open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(TaskGrpoStats.__dataclass_fields__)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in summary.tasks:
            writer.writerow(item.to_dict())


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize GRPO group JSONL files for paper tables.")
    parser.add_argument("groups_jsonl", nargs="+", type=Path, help="One or more grpo_groups.jsonl files.")
    parser.add_argument(
        "--inspect-artifacts",
        action="store_true",
        help="Load strict GRPO .pt artifacts and count v2 trajectory transitions instead of only artifact paths.",
    )
    parser.add_argument("--out-json", type=Path, help="Optional summary JSON output.")
    parser.add_argument("--out-csv", type=Path, help="Optional per-task CSV output.")
    parser.add_argument("--out-markdown", type=Path, help="Optional Markdown table output.")
    args = parser.parse_args()

    summary = summarize_groups([path.expanduser() for path in args.groups_jsonl], inspect_artifacts=args.inspect_artifacts)
    payload = summary.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.out_json:
        args.out_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.out_json.expanduser().write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote JSON: {args.out_json}")
    if args.out_csv:
        write_csv(args.out_csv, summary)
        print(f"Wrote CSV: {args.out_csv}")
    if args.out_markdown:
        args.out_markdown.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.out_markdown.expanduser().write_text(format_markdown(summary), encoding="utf-8")
        print(f"Wrote Markdown: {args.out_markdown}")


if __name__ == "__main__":
    main()
