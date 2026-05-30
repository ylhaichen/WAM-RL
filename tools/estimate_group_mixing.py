#!/usr/bin/env python3
"""Estimate mixed-group probability for grouped rollout planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def mixed_group_probability(success_rate: float, group_size: int) -> float:
    if not 0.0 <= success_rate <= 1.0:
        raise ValueError("success_rate must be in [0, 1]")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    return 1.0 - success_rate**group_size - (1.0 - success_rate) ** group_size


def estimate_group_mixing(summary: dict[str, Any], group_sizes: list[int]) -> dict[str, Any]:
    rows = []
    if "success_rate" in summary:
        rows.append(_estimate_row("overall", float(summary["success_rate"]), group_sizes))
    for task in summary.get("tasks", []) or []:
        if "task" not in task or "success_rate" not in task:
            continue
        rows.append(_estimate_row(str(task["task"]), float(task["success_rate"]), group_sizes))
    return {
        "source": summary.get("source_files", []),
        "group_sizes": group_sizes,
        "rows": rows,
    }


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    group_sizes = report["group_sizes"]
    lines = [
        "# Mixed Group Probability Estimate",
        "",
        "| task | success rate | " + " | ".join(f"k={size}" for size in group_sizes) + " |",
        "|---|---:|" + "|".join("---:" for _ in group_sizes) + "|",
    ]
    for row in report["rows"]:
        values = [
            row["task"],
            f"{row['success_rate']:.4f}",
            *[f"{row['by_group_size'][str(size)]['mixed_probability']:.4f}" for size in group_sizes],
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    path.expanduser().parent.mkdir(parents=True, exist_ok=True)
    path.expanduser().write_text("\n".join(lines), encoding="utf-8")


def _estimate_row(task: str, success_rate: float, group_sizes: list[int]) -> dict[str, Any]:
    by_group_size = {}
    for group_size in group_sizes:
        mixed_prob = mixed_group_probability(success_rate, group_size)
        by_group_size[str(group_size)] = {
            "mixed_probability": mixed_prob,
            "expected_attempts_per_mixed_group": None if mixed_prob <= 0.0 else 1.0 / mixed_prob,
        }
    return {
        "task": task,
        "success_rate": success_rate,
        "by_group_size": by_group_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate mixed-group probabilities from GRPO success rates.")
    parser.add_argument("--summary", type=Path, help="Summary JSON from tools/summarize_grpo_groups.py.")
    parser.add_argument("--success-rate", type=float, help="Single success rate to estimate without a summary file.")
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[4, 8], help="Group sizes to estimate.")
    parser.add_argument("--out-json", type=Path, help="Optional JSON output path.")
    parser.add_argument("--out-markdown", type=Path, help="Optional Markdown output path.")
    args = parser.parse_args()

    if args.summary is None and args.success_rate is None:
        parser.error("set --summary or --success-rate")

    if args.summary is not None:
        summary = json.loads(args.summary.expanduser().read_text(encoding="utf-8"))
    else:
        summary = {"success_rate": args.success_rate}

    report = estimate_group_mixing(summary, args.group_sizes)
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    if args.out_json:
        args.out_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.out_json.expanduser().write_text(text, encoding="utf-8")
    if args.out_markdown:
        write_markdown_report(args.out_markdown, report)


if __name__ == "__main__":
    main()
