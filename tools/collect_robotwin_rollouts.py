#!/usr/bin/env python3
"""Collect RoboTwin rollout records into a flat reward dataset."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class RolloutRecord:
    result_root: str
    run_id: str
    policy_checkpoint: str
    reference_checkpoint: str
    task: str
    seed: int
    env_seed: int
    episode_index: int
    group_id: str
    sample_idx: int | None
    group_size: int | None
    sampling_seed: int | None
    success: bool
    reward: float
    prompt: str
    actions_path: str
    initial_obs_path: str
    visualization_path: str
    record_path: str
    server_action_paths: list[str]
    server_latent_paths: list[str]
    strict_grpo_ready: bool
    strict_grpo_scope: str
    strict_grpo_artifact_paths: list[str]
    video_guidance_scale: float | None = None
    action_guidance_scale: float | None = None
    action_num_inference_steps: int | None = None
    action_count: int | None = None
    obs_count: int | None = None
    take_action_cnt: int | None = None
    step_lim: int | None = None


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _task_filter(values: Iterable[str] | None) -> set[str] | None:
    if not values:
        return None
    out: set[str] = set()
    for value in values:
        out.update(item for item in value.split() if item)
    return out


def iter_rollout_records(root: Path, tasks: set[str] | None = None) -> Iterable[RolloutRecord]:
    rollout_root = root / "rollouts"
    if not rollout_root.exists():
        return

    for path in sorted(rollout_root.glob("*/*.json")):
        data = _load_json(path)
        task = str(data.get("task_name") or path.parent.name)
        if tasks is not None and task not in tasks:
            continue
        yield RolloutRecord(
            result_root=str(root),
            run_id=str(data.get("run_id", "")),
            policy_checkpoint=str(data.get("policy_checkpoint", "")),
            reference_checkpoint=str(data.get("reference_checkpoint", "")),
            task=task,
            seed=int(data["seed"]),
            env_seed=int(data.get("env_seed", data["seed"])),
            episode_index=int(data["episode_index"]),
            group_id=str(data.get("group_id", "")),
            sample_idx=_optional_int(data.get("sample_idx")),
            group_size=_optional_int(data.get("group_size")),
            sampling_seed=_optional_int(data.get("sampling_seed")),
            success=bool(data["success"]),
            reward=float(data.get("reward", 1.0 if data["success"] else 0.0)),
            prompt=str(data.get("prompt", "")),
            actions_path=str(data.get("actions_path", "")),
            initial_obs_path=str(data.get("initial_obs_path", "")),
            visualization_path=str(data.get("visualization_path", "")),
            record_path=str(path),
            server_action_paths=_string_list(data.get("server_action_paths")),
            server_latent_paths=_string_list(data.get("server_latent_paths")),
            strict_grpo_ready=bool(data.get("strict_grpo_ready", False)),
            strict_grpo_scope=str(data.get("strict_grpo_scope", "")),
            strict_grpo_artifact_paths=_string_list(data.get("strict_grpo_artifact_paths")),
            video_guidance_scale=_optional_float(data.get("video_guidance_scale")),
            action_guidance_scale=_optional_float(data.get("action_guidance_scale")),
            action_num_inference_steps=_optional_int(data.get("action_num_inference_steps")),
            action_count=_optional_int(data.get("action_count")),
            obs_count=_optional_int(data.get("obs_count")),
            take_action_cnt=_optional_int(data.get("take_action_cnt")),
            step_lim=_optional_int(data.get("step_lim")),
        )


def _optional_int(value) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def summarize(records: list[RolloutRecord]) -> str:
    by_task: dict[str, list[RolloutRecord]] = {}
    for record in records:
        by_task.setdefault(record.task, []).append(record)

    lines = []
    header = f"{'task':28s} {'succ':>5s} {'total':>5s} {'rate':>7s}"
    lines.append(header)
    lines.append("-" * len(header))
    total_succ = 0
    total = 0
    for task in sorted(by_task):
        items = by_task[task]
        succ = sum(1 for item in items if item.success)
        total_succ += succ
        total += len(items)
        rate = succ / len(items) if items else 0.0
        lines.append(f"{task:28s} {succ:5d} {len(items):5d} {rate:7.1%}")
    if total:
        lines.append("-" * len(header))
        lines.append(f"{'overall':28s} {total_succ:5d} {total:5d} {total_succ / total:7.1%}")
    return "\n".join(lines)


def write_jsonl(path: Path, records: list[RolloutRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: list[RolloutRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(RolloutRecord.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect RoboTwin rollout JSON into JSONL/CSV.")
    parser.add_argument("roots", nargs="+", type=Path, help="RoboTwin result roots.")
    parser.add_argument("--tasks", nargs="*", help="Optional selected task names.")
    parser.add_argument("--out-jsonl", type=Path, help="Output JSONL path.")
    parser.add_argument("--out-csv", type=Path, help="Output CSV path.")
    args = parser.parse_args()

    tasks = _task_filter(args.tasks)
    records: list[RolloutRecord] = []
    for root in args.roots:
        records.extend(iter_rollout_records(root.expanduser(), tasks=tasks))

    records.sort(key=lambda item: (item.task, item.seed, item.episode_index, item.record_path))
    if not records:
        raise SystemExit("No rollout records found.")

    print(summarize(records))
    if args.out_jsonl:
        write_jsonl(args.out_jsonl.expanduser(), records)
        print(f"\nWrote JSONL: {args.out_jsonl}")
    if args.out_csv:
        write_csv(args.out_csv.expanduser(), records)
        print(f"Wrote CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
