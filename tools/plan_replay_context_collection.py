#!/usr/bin/env python3
"""Plan storage pressure for bounded replay-context rollout collection."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


def plan_replay_context_collection(
    *,
    task_names: str,
    group_size: int,
    groups_per_task: int,
    group_max_attempts: int,
    capture_max_chunks: int,
    save_replay_context: bool,
    replay_context_estimate_gb: float,
    storage_budget_mode: str,
    success_rate: float | None = None,
    check_scratch_headroom: bool = False,
    scratch_path: Path | None = None,
    min_scratch_headroom_gb: float = 0.0,
    dry_run: bool = False,
    allow_unbounded_replayctx: bool = False,
) -> dict[str, Any]:
    tasks = [task for task in task_names.split() if task]
    mode = storage_budget_mode.strip().lower()
    if mode not in {"attempt", "accepted"}:
        raise ValueError("storage_budget_mode must be 'attempt' or 'accepted'")
    if success_rate is not None and not 0.0 <= success_rate <= 1.0:
        raise ValueError("success_rate must be in [0, 1]")
    if save_replay_context and capture_max_chunks <= 0 and not allow_unbounded_replayctx:
        raise ValueError(
            "Refusing bounded replay-context submission with STRICT_GRPO_CAPTURE_MAX_CHUNKS<=0. "
            "Set ALLOW_UNBOUNDED_REPLAYCTX=1 only for an explicitly reviewed full-capture run."
        )

    chunks_per_rollout = max(capture_max_chunks, 0)
    accepted_contexts = len(tasks) * groups_per_task * group_size * chunks_per_rollout if save_replay_context else 0
    attempt_budget_contexts = len(tasks) * group_max_attempts * group_size * chunks_per_rollout if save_replay_context else 0
    accepted_estimate_gb = accepted_contexts * replay_context_estimate_gb
    attempt_budget_estimate_gb = attempt_budget_contexts * replay_context_estimate_gb
    storage_budget_estimate_gb = (
        attempt_budget_estimate_gb if mode == "attempt" else accepted_estimate_gb
    )

    report: dict[str, Any] = {
        "task_count": len(tasks),
        "tasks": tasks,
        "group_size": group_size,
        "groups_per_task": groups_per_task,
        "group_max_attempts": group_max_attempts,
        "capture_max_chunks": capture_max_chunks,
        "save_replay_context": save_replay_context,
        "replay_context_estimate_gb": replay_context_estimate_gb,
        "accepted_contexts": accepted_contexts,
        "attempt_budget_contexts": attempt_budget_contexts,
        "accepted_estimate_gb": accepted_estimate_gb,
        "attempt_budget_estimate_gb": attempt_budget_estimate_gb,
        "storage_budget_mode": mode,
        "storage_budget_estimate_gb": storage_budget_estimate_gb,
        "check_scratch_headroom": check_scratch_headroom,
        "dry_run": dry_run,
    }

    if success_rate is not None:
        mixed_probability = _mixed_group_probability(success_rate, group_size)
        report["mixing"] = {
            "success_rate": success_rate,
            "mixed_group_probability": mixed_probability,
            "expected_attempts_per_mixed_group": None
            if mixed_probability <= 0.0
            else 1.0 / mixed_probability,
            "probability_at_least_one_mixed_with_attempt_budget": 1.0
            - (1.0 - mixed_probability) ** group_max_attempts,
        }

    if check_scratch_headroom:
        if scratch_path is None:
            raise ValueError("scratch_path is required when check_scratch_headroom is true")
        scratch = scratch_path.expanduser()
        report["scratch_path"] = str(scratch)
        report["min_scratch_headroom_gb"] = min_scratch_headroom_gb
        try:
            usage = shutil.disk_usage(scratch)
        except FileNotFoundError:
            report["scratch_available_gb"] = None
            report["required_for_budget_plus_headroom_gb"] = storage_budget_estimate_gb + min_scratch_headroom_gb
            report["headroom_ok"] = None
            report["scratch_path_missing"] = True
        else:
            available_gb = usage.free / 1024**3
            required_gb = storage_budget_estimate_gb + min_scratch_headroom_gb
            report["scratch_available_gb"] = available_gb
            report["required_for_budget_plus_headroom_gb"] = required_gb
            report["headroom_ok"] = available_gb >= required_gb
            report["scratch_path_missing"] = False

    return report


def format_shell_summary(report: dict[str, Any]) -> str:
    lines = [
        "Replay-context storage estimate",
        f"  task_count={report['task_count']}",
        f"  accepted_contexts={report['accepted_contexts']}",
        f"  attempt_budget_contexts={report['attempt_budget_contexts']}",
        f"  accepted_estimate_gb={report['accepted_estimate_gb']:.2f}",
        f"  attempt_budget_estimate_gb={report['attempt_budget_estimate_gb']:.2f}",
        f"  storage_budget_mode={report['storage_budget_mode']}",
        f"  storage_budget_estimate_gb={report['storage_budget_estimate_gb']:.2f}",
    ]
    if report.get("check_scratch_headroom"):
        available = report.get("scratch_available_gb")
        lines.append(f"  scratch_path={report['scratch_path']}")
        lines.append(
            "  scratch_available_gb=unavailable"
            if available is None
            else f"  scratch_available_gb={available:.2f}"
        )
        lines.append(
            "  required_for_budget_plus_headroom_gb="
            f"{report['required_for_budget_plus_headroom_gb']:.2f}"
        )
        if report.get("headroom_ok") is not None:
            lines.append(f"  headroom_ok={str(report['headroom_ok']).lower()}")
    if report.get("mixing"):
        mixing = report["mixing"]
        expected = mixing.get("expected_attempts_per_mixed_group")
        lines.extend(
            [
                "Mixed-group estimate",
                f"  success_rate={mixing['success_rate']:.4f}",
                f"  mixed_group_probability={mixing['mixed_group_probability']:.4f}",
                "  expected_attempts_per_mixed_group="
                + ("unavailable" if expected is None else f"{expected:.2f}"),
                "  probability_at_least_one_mixed_with_attempt_budget="
                f"{mixing['probability_at_least_one_mixed_with_attempt_budget']:.4f}",
            ]
        )
    return "\n".join(lines)


def _mixed_group_probability(success_rate: float, group_size: int) -> float:
    return 1.0 - success_rate**group_size - (1.0 - success_rate) ** group_size


def _str_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean-like value, got: {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan bounded replay-context rollout collection storage.")
    parser.add_argument("--task-names", required=True, help="Space-separated task names.")
    parser.add_argument("--group-size", required=True, type=int)
    parser.add_argument("--groups-per-task", required=True, type=int)
    parser.add_argument("--group-max-attempts", required=True, type=int)
    parser.add_argument("--capture-max-chunks", required=True, type=int)
    parser.add_argument("--save-replay-context", required=True, type=_str_bool)
    parser.add_argument("--allow-unbounded-replayctx", default="false", type=_str_bool)
    parser.add_argument("--replay-context-estimate-gb", required=True, type=float)
    parser.add_argument("--storage-budget-mode", default="attempt", choices=("attempt", "accepted"))
    parser.add_argument(
        "--success-rate",
        type=float,
        help="Optional recent task success rate for mixed-group probability planning.",
    )
    parser.add_argument("--check-scratch-headroom", default="false", type=_str_bool)
    parser.add_argument("--scratch-path", type=Path)
    parser.add_argument("--min-scratch-headroom-gb", default=0.0, type=float)
    parser.add_argument("--dry-run", default="false", type=_str_bool)
    parser.add_argument(
        "--format",
        choices=("json", "shell"),
        default="json",
        help="Output format. Use shell for job-wrapper-compatible text.",
    )
    args = parser.parse_args()

    try:
        report = plan_replay_context_collection(
            task_names=args.task_names,
            group_size=args.group_size,
            groups_per_task=args.groups_per_task,
            group_max_attempts=args.group_max_attempts,
            capture_max_chunks=args.capture_max_chunks,
            save_replay_context=args.save_replay_context,
            replay_context_estimate_gb=args.replay_context_estimate_gb,
            storage_budget_mode=args.storage_budget_mode,
            success_rate=args.success_rate,
            check_scratch_headroom=args.check_scratch_headroom,
            scratch_path=args.scratch_path,
            min_scratch_headroom_gb=args.min_scratch_headroom_gb,
            dry_run=args.dry_run,
            allow_unbounded_replayctx=args.allow_unbounded_replayctx,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc

    if args.format == "shell":
        print(format_shell_summary(report))
    else:
        print(json.dumps(report, indent=2))

    if report.get("scratch_path_missing") and not args.dry_run:
        print(f"Scratch path does not exist: {report['scratch_path']}", file=sys.stderr)
        raise SystemExit(2)
    if report.get("headroom_ok") is False and not args.dry_run:
        print(
            "Insufficient Scratch headroom for replay-context storage budget. "
            "Free space, lower GROUP_SIZE/GROUPS_PER_TASK/STRICT_GRPO_CAPTURE_MAX_CHUNKS, "
            "lower GROUP_MAX_ATTEMPTS, set STORAGE_BUDGET_MODE=accepted, "
            "or set CHECK_SCRATCH_HEADROOM=0 after explicit review.",
            file=sys.stderr,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
