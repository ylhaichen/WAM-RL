#!/usr/bin/env python3
import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TaskResult:
    task: str
    succ: int
    total: int
    paths: list[Path]

    @property
    def rate(self) -> float:
        return self.succ / self.total if self.total else 0.0


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def band(rate: float) -> str:
    if rate >= 0.90:
        return "easy"
    if rate >= 0.50:
        return "medium"
    if rate >= 0.10:
        return "hard"
    return "too_hard"


def task_name_from_res_json(path: Path) -> str:
    # Expected: .../metrics/<task>/res.json
    if path.parent.parent.name == "metrics":
        return path.parent.name
    return path.parent.name


def load_results(root: Path) -> list[TaskResult]:
    grouped: dict[str, TaskResult] = {}
    for path in sorted(root.rglob("res.json")):
        data = json.loads(path.read_text())
        task = task_name_from_res_json(path)
        succ = int(data["succ_num"])
        total = int(data["total_num"])
        if task not in grouped:
            grouped[task] = TaskResult(task=task, succ=0, total=0, paths=[])
        grouped[task].succ += succ
        grouped[task].total += total
        grouped[task].paths.append(path)
    return list(grouped.values())


def format_table(results: list[TaskResult]) -> str:
    lines = []
    header = f"{'task':28s} {'succ':>5s} {'total':>5s} {'rate':>7s} {'95% CI':>17s} {'band':>8s}"
    lines.append(header)
    lines.append("-" * len(header))
    total_succ = 0
    total = 0
    for item in results:
        low, high = wilson_interval(item.succ, item.total)
        total_succ += item.succ
        total += item.total
        lines.append(
            f"{item.task:28s} {item.succ:5d} {item.total:5d} "
            f"{item.rate:7.1%} [{low:5.1%}, {high:5.1%}] {band(item.rate):>8s}"
        )
    if total:
        low, high = wilson_interval(total_succ, total)
        lines.append("-" * len(header))
        lines.append(
            f"{'overall':28s} {total_succ:5d} {total:5d} "
            f"{(total_succ / total):7.1%} [{low:5.1%}, {high:5.1%}] {'':>8s}"
        )
    return "\n".join(lines)


def write_csv(path: Path, results: list[TaskResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "succ",
                "total",
                "rate",
                "ci_low",
                "ci_high",
                "band",
                "num_result_files",
            ],
        )
        writer.writeheader()
        for item in results:
            low, high = wilson_interval(item.succ, item.total)
            writer.writerow(
                {
                    "task": item.task,
                    "succ": item.succ,
                    "total": item.total,
                    "rate": item.rate,
                    "ci_low": low,
                    "ci_high": high,
                    "band": band(item.rate),
                    "num_result_files": len(item.paths),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RoboTwin res.json files.")
    parser.add_argument("root", type=Path, help="Result root containing metrics/*/res.json files.")
    parser.add_argument("--csv", type=Path, help="Optional CSV output path.")
    parser.add_argument(
        "--sort",
        choices=["rate", "task", "total"],
        default="rate",
        help="Sort table by success rate, task name, or total count.",
    )
    parser.add_argument("--descending", action="store_true", help="Reverse sort order.")
    args = parser.parse_args()

    results = load_results(args.root.expanduser())
    if not results:
        raise SystemExit(f"No res.json files found under {args.root}")

    if args.sort == "rate":
        key = lambda item: (item.rate, item.task)
    elif args.sort == "task":
        key = lambda item: item.task
    else:
        key = lambda item: (item.total, item.task)
    results = sorted(results, key=key, reverse=args.descending)

    print(format_table(results))
    if args.csv:
        write_csv(args.csv.expanduser(), results)
        print(f"\nWrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
