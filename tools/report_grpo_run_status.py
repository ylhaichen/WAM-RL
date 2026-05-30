#!/usr/bin/env python3
"""Summarize WAM-RL GRPO job logs and result directories."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


GiB = 1024**3
LOG_VALUE_KEYS = (
    "JOB_ID",
    "RUN_ID",
    "RESULTS_ROOT",
    "GRPO_GROUPS_PATH",
    "GRPO_OUTPUT_DIR",
    "GRPO_STEPS",
    "GRPO_ACTION_NUM_INFERENCE_STEPS",
    "GROUP_SIZE",
    "GROUPS_PER_TASK",
    "GROUP_MAX_ATTEMPTS",
    "TASK_NAMES",
    "ACTION_NUM_INFERENCE_STEPS",
    "STRICT_GRPO_CAPTURE_MAX_CHUNKS",
    "STRICT_GRPO_SAVE_REPLAY_CONTEXT",
    "STRICT_GRPO_REPLAY_CONTEXT_MAX_GB",
)


def parse_job_log(path: Path) -> dict[str, Any]:
    expanded = path.expanduser()
    values: dict[str, str] = {}
    counters = {
        "traceback_count": 0,
        "error_count": 0,
        "accepted_group_attempt_count": 0,
        "discarded_group_attempt_count": 0,
        "disk_quota_count": 0,
        "pytorch_stream_writer_count": 0,
        "completion_marker_count": 0,
    }
    failed_attempt_roots: list[str] = []
    completion_paths: list[str] = []

    with expanded.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            for key in LOG_VALUE_KEYS:
                for value in _extract_key_values(line, key):
                    values[key] = value

            if "Traceback" in line:
                counters["traceback_count"] += 1
            if "ERROR" in line:
                counters["error_count"] += 1
            if "Accepted group attempt" in line:
                counters["accepted_group_attempt_count"] += 1
            if "Discarding failed group attempt" in line:
                counters["discarded_group_attempt_count"] += 1
                root = _extract_failed_attempt_root(line)
                if root:
                    failed_attempt_roots.append(root)
            if "Disk quota exceeded" in line:
                counters["disk_quota_count"] += 1
            if "PytorchStreamWriter" in line:
                counters["pytorch_stream_writer_count"] += 1
            if "Grouped rollout collection complete:" in line:
                counters["completion_marker_count"] += 1
                completion_paths.append(line.split("Grouped rollout collection complete:", 1)[1].strip())
            elif "Actor replay GRPO training complete:" in line:
                counters["completion_marker_count"] += 1
                completion_paths.append(line.split("Actor replay GRPO training complete:", 1)[1].strip())
            elif "Offline GRPO smoke training complete:" in line:
                counters["completion_marker_count"] += 1
                completion_paths.append(line.split("Offline GRPO smoke training complete:", 1)[1].strip())

    return {
        "path": str(expanded),
        "exists": expanded.exists(),
        "values": values,
        "counters": counters,
        "failed_attempt_roots": failed_attempt_roots,
        "completion_paths": completion_paths,
    }


def report_grpo_run_status(
    *,
    job_log: Path | None = None,
    results_root: Path | None = None,
    training_output_dir: Path | None = None,
    inspect_files: bool = False,
) -> dict[str, Any]:
    log_report = parse_job_log(job_log) if job_log is not None else None
    inferred_results_root = _first_present_path(
        results_root,
        _value_path(log_report, "RESULTS_ROOT"),
    )
    inferred_training_output_dir = _first_present_path(
        training_output_dir,
        _value_path(log_report, "GRPO_OUTPUT_DIR"),
    )

    report: dict[str, Any] = {
        "job_log": log_report,
        "results_root": _summarize_results_root(inferred_results_root, inspect_files=inspect_files)
        if inferred_results_root is not None
        else None,
        "training_output_dir": _summarize_training_output(inferred_training_output_dir)
        if inferred_training_output_dir is not None
        else None,
    }
    report["status"] = _derive_status(report)
    return report


def format_markdown(report: dict[str, Any]) -> str:
    status = report.get("status", {})
    lines = [
        "# GRPO Run Status",
        "",
        "## Overall",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| state | {_md(status.get('state', 'unknown'))} |",
        f"| trainable_group_count | {status.get('trainable_group_count', 0)} |",
        f"| transition_count | {status.get('transition_count', 0)} |",
        f"| validation_ok | {_md(status.get('validation_ok'))} |",
        f"| traceback_count | {status.get('traceback_count', 0)} |",
        f"| disk_quota_count | {status.get('disk_quota_count', 0)} |",
        "",
    ]

    log_report = report.get("job_log")
    if log_report:
        values = log_report.get("values", {})
        lines.extend(["## Job Log", "", "| key | value |", "|---|---|"])
        for key in LOG_VALUE_KEYS:
            if key in values:
                lines.append(f"| {_md(key)} | {_md(values[key])} |")
        lines.append("")

    results = report.get("results_root")
    if results:
        lines.extend(
            [
                "## Results Root",
                "",
                "| metric | value |",
                "|---|---:|",
                f"| path | {_md(results.get('path'))} |",
                f"| exists | {_md(results.get('exists'))} |",
                f"| grpo_group_line_count | {results.get('grpo_group_line_count', 0)} |",
                f"| successful_attempt_count | {results.get('successful_attempt_count', 0)} |",
                f"| failed_attempt_count | {results.get('failed_attempt_count', 0)} |",
                f"| strict_artifact_count | {results.get('strict_artifact_count', 'not inspected')} |",
                f"| replay_context_count | {results.get('replay_context_count', 'not inspected')} |",
                f"| disk_gib | {_format_optional_gib(results.get('disk_bytes'))} |",
                f"| server_vis_gib | {_format_optional_gib(results.get('server_vis_bytes'))} |",
                "",
            ]
        )

    training = report.get("training_output_dir")
    if training:
        final_metrics = training.get("final_metrics") or {}
        lines.extend(["## Training Output", "", "| metric | value |", "|---|---:|"])
        lines.append(f"| path | {_md(training.get('path'))} |")
        lines.append(f"| exists | {_md(training.get('exists'))} |")
        for key in sorted(final_metrics):
            lines.append(f"| {_md(key)} | {_md(final_metrics[key])} |")
        lines.append("")

    return "\n".join(lines)


def _derive_status(report: dict[str, Any]) -> dict[str, Any]:
    log_report = report.get("job_log") or {}
    counters = log_report.get("counters") or {}
    results = report.get("results_root") or {}
    validation = results.get("validation") or {}
    summary = results.get("grpo_summary") or {}
    training = report.get("training_output_dir") or {}

    group_lines = int(results.get("grpo_group_line_count", 0) or 0)
    transition_count = int(validation.get("transition_count", 0) or summary.get("transition_count", 0) or 0)
    validation_ok = validation.get("ok")
    traceback_count = int(counters.get("traceback_count", 0) or 0)
    disk_quota_count = int(counters.get("disk_quota_count", 0) or 0)

    if training and training.get("checkpoint_exists"):
        state = "training_checkpoint_written"
    elif group_lines > 0 and validation_ok is not False:
        state = "trainable_groups_available"
    elif counters.get("completion_marker_count", 0) and group_lines == 0:
        state = "completed_without_trainable_groups"
    elif traceback_count or disk_quota_count:
        state = "errors_seen"
    else:
        state = "unknown_or_queued"

    return {
        "state": state,
        "trainable_group_count": group_lines,
        "transition_count": transition_count,
        "validation_ok": validation_ok,
        "traceback_count": traceback_count,
        "disk_quota_count": disk_quota_count,
        "accepted_group_attempt_count": int(counters.get("accepted_group_attempt_count", 0) or 0),
        "discarded_group_attempt_count": int(counters.get("discarded_group_attempt_count", 0) or 0),
    }


def _summarize_results_root(root: Path, *, inspect_files: bool) -> dict[str, Any]:
    expanded = root.expanduser()
    groups_dir = expanded / "groups"
    report: dict[str, Any] = {
        "path": str(expanded),
        "exists": expanded.exists(),
        "groups_dir": str(groups_dir),
        "grpo_groups_path": str(groups_dir / "grpo_groups.jsonl"),
        "grpo_group_line_count": _line_count(groups_dir / "grpo_groups.jsonl"),
        "successful_attempt_count": _line_count(groups_dir / "successful_attempt_roots.txt"),
        "failed_attempt_count": _line_count(groups_dir / "failed_attempt_roots.txt"),
        "grpo_summary": _read_json(groups_dir / "grpo_summary.json"),
        "validation": _read_first_json(
            [
                groups_dir / "grpo_dataset_validation_actor_replay.json",
                groups_dir / "grpo_dataset_validation.json",
                groups_dir / "validation_actor_replay.json",
            ]
        ),
    }
    if inspect_files:
        report.update(_inspect_result_files(expanded))
    return report


def _summarize_training_output(output_dir: Path) -> dict[str, Any]:
    expanded = output_dir.expanduser()
    metrics = _read_json(expanded / "metrics.json")
    final_metrics = _final_metrics(metrics)
    return {
        "path": str(expanded),
        "exists": expanded.exists(),
        "input_dataset_validation": _read_json(expanded / "input_dataset_validation.json"),
        "metrics_path": str(expanded / "metrics.json"),
        "metrics_exists": (expanded / "metrics.json").exists(),
        "checkpoint_path": str(expanded / "checkpoint.pt"),
        "checkpoint_exists": (expanded / "checkpoint.pt").exists(),
        "final_metrics": final_metrics,
    }


def _inspect_result_files(root: Path) -> dict[str, Any]:
    strict_artifact_count = 0
    replay_context_count = 0
    disk_bytes = 0
    server_vis_bytes = 0
    server_vis = root / "server_vis"

    if not root.exists():
        return {
            "strict_artifact_count": 0,
            "replay_context_count": 0,
            "disk_bytes": 0,
            "server_vis_bytes": 0,
        }

    disk_bytes = _path_disk_bytes(root)
    if _is_relative_to(root, server_vis):
        server_vis_bytes += _path_disk_bytes(root)

    for base, dirs, files in os.walk(root, followlinks=False):
        base_path = Path(base)
        for name in dirs:
            path = base_path / name
            size = _path_disk_bytes(path)
            disk_bytes += size
            if _is_relative_to(path, server_vis):
                server_vis_bytes += size
        for name in files:
            path = base_path / name
            size = _path_disk_bytes(path)
            disk_bytes += size
            if _is_relative_to(path, server_vis):
                server_vis_bytes += size
            if _is_strict_artifact_name(name):
                strict_artifact_count += 1
            elif name.startswith("strict_grpo_replay_context_") and name.endswith(".pt"):
                replay_context_count += 1

    return {
        "strict_artifact_count": strict_artifact_count,
        "replay_context_count": replay_context_count,
        "disk_bytes": disk_bytes,
        "disk_gib": disk_bytes / GiB,
        "server_vis_bytes": server_vis_bytes,
        "server_vis_gib": server_vis_bytes / GiB,
    }


def _extract_key_values(line: str, key: str) -> list[str]:
    pattern = re.compile(rf"(?:^|[,\s]){re.escape(key)}=([^,\s]+)")
    return [match.group(1) for match in pattern.finditer(line)]


def _extract_failed_attempt_root(line: str) -> str | None:
    marker = "logs remain under "
    if marker not in line:
        return None
    return line.split(marker, 1)[1].strip()


def _value_path(log_report: dict[str, Any] | None, key: str) -> Path | None:
    if not log_report:
        return None
    value = (log_report.get("values") or {}).get(key)
    if not value:
        return None
    return Path(value)


def _first_present_path(*paths: Path | None) -> Path | None:
    for path in paths:
        if path is not None:
            return path
    return None


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.strip():
                    count += 1
    except OSError:
        return 0
    return count


def _read_first_json(paths: list[Path]) -> dict[str, Any]:
    for path in paths:
        value = _read_json(path)
        if value:
            return value
    return {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _final_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    if not metrics:
        return {}
    if isinstance(metrics.get("final_metrics"), dict):
        return metrics["final_metrics"]
    history = metrics.get("history")
    if isinstance(history, list) and history and isinstance(history[-1], dict):
        return history[-1]
    steps = metrics.get("steps")
    if isinstance(steps, list) and steps and isinstance(steps[-1], dict):
        return steps[-1]
    return {
        key: value
        for key, value in metrics.items()
        if key.startswith("final_") or key in {"checkpoint_path", "transition_count"}
    }


def _path_disk_bytes(path: Path) -> int:
    try:
        stat = path.lstat()
    except FileNotFoundError:
        return 0
    blocks = getattr(stat, "st_blocks", None)
    if blocks is not None:
        return int(blocks) * 512
    return int(stat.st_size)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_strict_artifact_name(name: str) -> bool:
    return name.startswith("strict_grpo_") and name.endswith(".pt") and not name.startswith(
        "strict_grpo_replay_context_"
    )


def _format_optional_gib(value: Any) -> str:
    if value is None:
        return "not inspected"
    return f"{float(value) / GiB:.3f} GiB"


def _md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report WAM-RL GRPO job and result status.")
    parser.add_argument("--job-log", type=Path, help="Optional SGE job log to parse.")
    parser.add_argument("--results-root", type=Path, help="Optional grouped rollout result root.")
    parser.add_argument("--training-output-dir", type=Path, help="Optional GRPO training output directory.")
    parser.add_argument(
        "--inspect-files",
        action="store_true",
        help="Walk the result root to count strict artifacts/replay contexts and disk blocks.",
    )
    parser.add_argument("--out-json", type=Path, help="Optional JSON report path.")
    parser.add_argument("--out-markdown", type=Path, help="Optional Markdown report path.")
    parser.add_argument("--print-markdown", action="store_true", help="Print Markdown instead of JSON.")
    args = parser.parse_args()

    if args.job_log is None and args.results_root is None and args.training_output_dir is None:
        parser.error("at least one of --job-log, --results-root, or --training-output-dir is required")

    report = report_grpo_run_status(
        job_log=args.job_log,
        results_root=args.results_root,
        training_output_dir=args.training_output_dir,
        inspect_files=args.inspect_files,
    )

    if args.out_json is not None:
        args.out_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.out_json.expanduser().write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.out_markdown is not None:
        args.out_markdown.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.out_markdown.expanduser().write_text(format_markdown(report), encoding="utf-8")

    if args.print_markdown:
        print(format_markdown(report))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
