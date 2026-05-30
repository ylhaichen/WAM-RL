#!/usr/bin/env python3
"""Summarize WAM-RL GRPO job logs and result directories."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


GiB = 1024**3
LOG_VALUE_KEYS = (
    "JOB_ID",
    "GIT_COMMIT",
    "SUBMIT_GIT_COMMIT",
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
    grouped_rollout_completion_paths: list[str] = []
    training_completion_paths: list[str] = []

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
                path = line.split("Grouped rollout collection complete:", 1)[1].strip()
                completion_paths.append(path)
                grouped_rollout_completion_paths.append(path)
            elif "Actor replay GRPO training complete:" in line:
                counters["completion_marker_count"] += 1
                path = line.split("Actor replay GRPO training complete:", 1)[1].strip()
                completion_paths.append(path)
                training_completion_paths.append(path)
            elif "Offline GRPO smoke training complete:" in line:
                counters["completion_marker_count"] += 1
                path = line.split("Offline GRPO smoke training complete:", 1)[1].strip()
                completion_paths.append(path)
                training_completion_paths.append(path)

    return {
        "path": str(expanded),
        "exists": expanded.exists(),
        "values": values,
        "counters": counters,
        "failed_attempt_roots": failed_attempt_roots,
        "completion_paths": completion_paths,
        "grouped_rollout_completion_paths": grouped_rollout_completion_paths,
        "training_completion_paths": training_completion_paths,
    }


def parse_qstat_job_detail_text(text: str) -> dict[str, Any]:
    fields: dict[str, str] = {}
    active_key: str | None = None
    for raw_line in text.splitlines():
        if ":" not in raw_line:
            if active_key and raw_line.strip():
                fields[active_key] = f"{fields[active_key]},{raw_line.strip()}"
            continue
        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key:
            fields[key] = value
            active_key = key

    env = _parse_key_value_list(fields.get("env_list", ""))
    resources = _parse_key_value_list(fields.get("hard resource_list", ""))
    values = {key: env[key] for key in LOG_VALUE_KEYS if key in env}
    if "job_number" in fields and "JOB_ID" not in values:
        values["JOB_ID"] = fields["job_number"]
    return {
        "exists": bool(fields),
        "fields": fields,
        "values": values,
        "resources": resources,
        "job_number": fields.get("job_number"),
        "job_name": fields.get("job_name"),
        "owner": fields.get("owner"),
        "submission_time": fields.get("submission_time"),
        "cwd": fields.get("cwd"),
        "script_file": fields.get("script_file"),
        "parallel_environment": fields.get("parallel environment"),
        "project": fields.get("project"),
    }


def parse_qstat_job_detail_file(path: Path) -> dict[str, Any]:
    expanded = path.expanduser()
    try:
        text = expanded.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return {"exists": False, "path": str(expanded), "fields": {}, "values": {}, "resources": {}}
    report = parse_qstat_job_detail_text(text)
    report["path"] = str(expanded)
    return report


def load_qstat_job_detail(job_id: str) -> dict[str, Any]:
    result = subprocess.run(
        ["qstat", "-j", str(job_id)],
        check=False,
        capture_output=True,
        text=True,
    )
    report = parse_qstat_job_detail_text(result.stdout)
    report.update(
        {
            "job_id": str(job_id),
            "command": ["qstat", "-j", str(job_id)],
            "returncode": result.returncode,
            "stderr": result.stderr.strip(),
        }
    )
    if result.returncode != 0:
        report["exists"] = False
    return report


def report_grpo_run_status(
    *,
    job_log: Path | None = None,
    results_root: Path | None = None,
    training_output_dir: Path | None = None,
    qstat_job: dict[str, Any] | None = None,
    inspect_files: bool = False,
) -> dict[str, Any]:
    log_report = parse_job_log(job_log) if job_log is not None else None
    qstat_values = (qstat_job or {}).get("values") or {}
    inferred_results_root = _first_present_path(
        results_root,
        _value_path(log_report, "RESULTS_ROOT"),
        _last_path(log_report, "grouped_rollout_completion_paths"),
        _dict_value_path(qstat_values, "RESULTS_ROOT"),
        _groups_file_results_root(_dict_value_path(qstat_values, "GRPO_GROUPS_PATH")),
        _qstat_grouped_rollout_results_root(qstat_job),
    )
    inferred_training_output_dir = _first_present_path(
        training_output_dir,
        _value_path(log_report, "GRPO_OUTPUT_DIR"),
        _last_path(log_report, "training_completion_paths"),
        _dict_value_path(qstat_values, "GRPO_OUTPUT_DIR"),
    )

    report: dict[str, Any] = {
        "job_log": log_report,
        "qstat_job": qstat_job,
        "results_root": _summarize_results_root(inferred_results_root, inspect_files=inspect_files)
        if inferred_results_root is not None
        else None,
        "training_output_dir": _summarize_training_output(inferred_training_output_dir)
        if inferred_training_output_dir is not None
        else None,
    }
    report["status"] = _derive_status(report)
    return report


def select_latest_job_log(patterns: list[str]) -> Path:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(Path(path) for path in glob.glob(str(Path(pattern).expanduser())))
    files = [path for path in matches if path.is_file()]
    if not files:
        raise FileNotFoundError(f"no job logs matched: {patterns}")
    return max(files, key=lambda path: (path.stat().st_mtime, str(path)))


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
    warnings = status.get("warnings") or []
    if warnings:
        lines.extend(["## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {_md(warning)}")
        lines.append("")

    log_report = report.get("job_log")
    if log_report:
        values = log_report.get("values", {})
        lines.extend(["## Job Log", "", "| key | value |", "|---|---|"])
        for key in LOG_VALUE_KEYS:
            if key in values:
                lines.append(f"| {_md(key)} | {_md(values[key])} |")
        lines.append("")

    qstat_report = report.get("qstat_job")
    if qstat_report:
        lines.extend(["## Qstat Job", "", "| key | value |", "|---|---|"])
        for key in ("job_number", "job_name", "owner", "submission_time", "cwd", "script_file", "project"):
            if qstat_report.get(key):
                lines.append(f"| {_md(key)} | {_md(qstat_report[key])} |")
        for key in LOG_VALUE_KEYS:
            value = (qstat_report.get("values") or {}).get(key)
            if value is not None:
                lines.append(f"| {_md(key)} | {_md(value)} |")
        resources = qstat_report.get("resources") or {}
        if resources:
            lines.append(f"| hard_resources | {_md(_format_key_values(resources))} |")
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
    qstat_job = report.get("qstat_job") or {}

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
    elif qstat_job and qstat_job.get("exists"):
        state = "scheduler_known_no_log"
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
        "qstat_job_number": qstat_job.get("job_number"),
        "qstat_job_name": qstat_job.get("job_name"),
        "warnings": _status_warnings(qstat_job),
    }


def _status_warnings(qstat_job: dict[str, Any]) -> list[str]:
    if not qstat_job or not qstat_job.get("exists"):
        return []
    warnings: list[str] = []
    values = qstat_job.get("values") or {}
    resources = qstat_job.get("resources") or {}
    job_name = str(qstat_job.get("job_name") or "")

    if (
        _looks_like_grouped_rollout_qstat(qstat_job)
        and values.get("RUN_ID")
        and not values.get("RESULTS_ROOT")
        and not values.get("GRPO_GROUPS_PATH")
    ):
        warnings.append(
            "qstat env does not include explicit RESULTS_ROOT/GRPO_GROUPS_PATH; "
            "result root was inferred from RUN_ID and Myriad path conventions"
        )

    if "replayctx_bounded" in job_name:
        h_rt_seconds = _parse_seconds(resources.get("h_rt"))
        if h_rt_seconds is not None and h_rt_seconds > 6 * 60 * 60:
            warnings.append(
                f"bounded replay-context job requests h_rt={resources.get('h_rt')}, "
                "above the current 6:00:00 smoke default"
            )
        tmpfs_gb = _parse_size_gb(resources.get("tmpfs"))
        if tmpfs_gb is not None and tmpfs_gb > 80:
            warnings.append(
                f"bounded replay-context job requests tmpfs={resources.get('tmpfs')}, "
                "above the current 80G smoke default"
            )
    return warnings


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


def _parse_key_value_list(value: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in value.split(","):
        if "=" not in item:
            continue
        key, item_value = item.split("=", 1)
        key = key.strip()
        item_value = item_value.strip()
        if key:
            parsed[key] = item_value
    return parsed


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


def _last_path(log_report: dict[str, Any] | None, key: str) -> Path | None:
    if not log_report:
        return None
    values = log_report.get(key) or []
    if not values:
        return None
    return Path(str(values[-1]))


def _dict_value_path(values: dict[str, Any], key: str) -> Path | None:
    value = values.get(key)
    if not value:
        return None
    return Path(str(value))


def _groups_file_results_root(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.name != "grpo_groups.jsonl":
        return None
    if path.parent.name != "groups":
        return None
    return path.parent.parent


def _qstat_grouped_rollout_results_root(qstat_job: dict[str, Any] | None) -> Path | None:
    if not qstat_job:
        return None
    values = qstat_job.get("values") or {}
    run_id = values.get("RUN_ID")
    if not run_id:
        return None
    if not _looks_like_grouped_rollout_qstat(qstat_job):
        return None

    wam_root = values.get("WAM_ROOT")
    if wam_root:
        data_root = Path(str(wam_root))
    else:
        cwd = qstat_job.get("cwd")
        if not cwd:
            return None
        data_root = Path(str(cwd)).expanduser().parent / "wam-rl"
    return data_root / "results_grouped_rollouts" / str(run_id)


def _looks_like_grouped_rollout_qstat(qstat_job: dict[str, Any]) -> bool:
    values = qstat_job.get("values") or {}
    if "GROUP_SIZE" in values and "GROUPS_PER_TASK" in values:
        return True
    script_file = str(qstat_job.get("script_file") or "")
    return "collect_grouped_rollouts" in script_file


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


def _format_key_values(values: dict[str, str]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(values.items()))


def _parse_seconds(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    parts = text.split(":")
    if len(parts) == 3 and all(part.isdigit() for part in parts):
        hours, minutes, seconds = [int(part) for part in parts]
        return hours * 3600 + minutes * 60 + seconds
    return None


def _parse_size_gb(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    multiplier = 1.0
    if text.endswith("G"):
        text = text[:-1]
    elif text.endswith("M"):
        text = text[:-1]
        multiplier = 1.0 / 1024.0
    elif text.endswith("K"):
        text = text[:-1]
        multiplier = 1.0 / (1024.0 * 1024.0)
    try:
        return float(text) * multiplier
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Report WAM-RL GRPO job and result status.")
    parser.add_argument("--job-log", type=Path, help="Optional SGE job log to parse.")
    parser.add_argument(
        "--job-log-glob",
        action="append",
        default=[],
        help="Glob for selecting the latest SGE job log by mtime. Can be passed more than once.",
    )
    parser.add_argument("--qstat-job-id", help="Run `qstat -j JOB_ID` and include scheduler metadata.")
    parser.add_argument("--qstat-job-file", type=Path, help="Parse a saved `qstat -j JOB_ID` output file.")
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

    if args.job_log is not None and args.job_log_glob:
        parser.error("use either --job-log or --job-log-glob, not both")
    if args.qstat_job_id and args.qstat_job_file:
        parser.error("use either --qstat-job-id or --qstat-job-file, not both")
    job_log = args.job_log
    if job_log is None and args.job_log_glob:
        try:
            job_log = select_latest_job_log(args.job_log_glob)
        except FileNotFoundError as exc:
            parser.error(str(exc))

    qstat_job = None
    if args.qstat_job_id:
        qstat_job = load_qstat_job_detail(args.qstat_job_id)
    elif args.qstat_job_file:
        qstat_job = parse_qstat_job_detail_file(args.qstat_job_file)

    if job_log is None and args.results_root is None and args.training_output_dir is None and qstat_job is None:
        parser.error(
            "at least one of --job-log, --job-log-glob, --qstat-job-id, "
            "--qstat-job-file, --results-root, or --training-output-dir is required"
        )

    report = report_grpo_run_status(
        job_log=job_log,
        results_root=args.results_root,
        training_output_dir=args.training_output_dir,
        qstat_job=qstat_job,
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
