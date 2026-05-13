#!/usr/bin/env python3
"""Build dynamic-sampling GRPO groups from collected RoboTwin rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.collect_robotwin_rollouts import iter_rollout_records
from wan_va.rl.group_builder import build_grpo_groups
from wan_va.rl.trajectory_schema import GrpoGroupBuildResult


def build_groups_from_roots(
    roots: list[Path],
    *,
    expected_group_size: int | None = None,
    require_strict_artifacts: bool = False,
    tasks: set[str] | None = None,
) -> GrpoGroupBuildResult:
    records = []
    for root in roots:
        records.extend(iter_rollout_records(root.expanduser(), tasks=tasks))
    return build_grpo_groups(
        records,
        expected_group_size=expected_group_size,
        require_strict_artifacts=require_strict_artifacts,
    )


def write_group_outputs(
    result: GrpoGroupBuildResult,
    *,
    out_jsonl: Path,
    out_summary: Path,
) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for group in result.groups:
            f.write(json.dumps(group.to_dict(), ensure_ascii=False) + "\n")

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(
        json.dumps(result.summary.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _task_filter(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    tasks: set[str] = set()
    for value in values:
        tasks.update(item for item in value.split() if item)
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mixed GRPO groups from RoboTwin rollout JSON.")
    parser.add_argument("roots", nargs="+", type=Path, help="RoboTwin result roots containing rollouts/.")
    parser.add_argument("--expected-group-size", type=int, help="Drop groups that do not contain exactly this many samples.")
    parser.add_argument("--require-strict-artifacts", action="store_true", help="Drop groups missing strict GRPO tensor artifacts.")
    parser.add_argument("--tasks", nargs="*", help="Optional selected task names.")
    parser.add_argument("--out-jsonl", type=Path, help="Output grouped JSONL path.")
    parser.add_argument("--out-summary", type=Path, help="Output summary JSON path.")
    args = parser.parse_args()

    roots = [root.expanduser() for root in args.roots]
    default_out_root = roots[0] / "groups"
    out_jsonl = (args.out_jsonl or default_out_root / "grpo_groups.jsonl").expanduser()
    out_summary = (args.out_summary or default_out_root / "grpo_summary.json").expanduser()

    result = build_groups_from_roots(
        roots,
        expected_group_size=args.expected_group_size,
        require_strict_artifacts=args.require_strict_artifacts,
        tasks=_task_filter(args.tasks),
    )
    write_group_outputs(result, out_jsonl=out_jsonl, out_summary=out_summary)

    summary = result.summary.to_dict()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote groups: {out_jsonl}")
    print(f"Wrote summary: {out_summary}")


if __name__ == "__main__":
    main()
