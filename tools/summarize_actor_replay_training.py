#!/usr/bin/env python3
"""Summarize actor replay GRPO training output directories."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Iterable

CONFIG_FIELDS = (
    "model_path",
    "config_name",
    "git_commit",
    "learning_rate",
    "action_num_inference_steps",
    "logprob_reduction",
    "logprob_std_floor",
    "trainable_mode",
)


def summarize_actor_replay_output(output_dir: Path, *, job_log_configs: dict[str, dict] | None = None) -> dict:
    root = output_dir.expanduser()
    metrics_path = root / "metrics.json"
    validation_path = root / "input_dataset_validation.json"
    failure_path = root / "failure_diagnostics.json"

    metrics = _read_json(metrics_path)
    validation = _read_json(validation_path)
    result = metrics.get("result", {}) if isinstance(metrics, dict) else {}
    history = metrics.get("history", []) if isinstance(metrics, dict) else []
    last_step = history[-1] if history else {}
    last_step_summary = _last_step_summary(last_step)
    checkpoint_path = Path(result.get("checkpoint_path") or root / "checkpoint.pt").expanduser()
    config, config_source = _training_config(metrics, checkpoint_path, root, job_log_configs or {})

    summary = {
        "output_dir": str(root),
        "metrics_path": str(metrics_path),
        "metrics_exists": metrics_path.exists(),
        "validation_path": str(validation_path),
        "validation_exists": validation_path.exists(),
        "validation_ok": bool(validation.get("ok", False)) if isinstance(validation, dict) else False,
        "validation_error_count": int(validation.get("error_count", -1)) if isinstance(validation, dict) else -1,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_exists": checkpoint_path.exists(),
        "checkpoint_bytes": checkpoint_path.stat().st_size if checkpoint_path.exists() else 0,
        "failure_diagnostics_path": str(failure_path),
        "failure_diagnostics_exists": failure_path.exists(),
        "transition_count": _number(result.get("transition_count")),
        "steps": _number(result.get("steps")),
        "final_loss": _number(result.get("final_loss")),
        "final_ratio_mean": _number(result.get("final_ratio_mean")),
        "trainable_param_count": _number(result.get("trainable_param_count")),
        "total_param_count": _number(result.get("total_param_count")),
        "config_source": config_source,
        "model_path": config.get("model_path"),
        "config_name": config.get("config_name"),
        "git_commit": config.get("git_commit"),
        "learning_rate": _number(config.get("learning_rate")),
        "action_num_inference_steps": _number(config.get("action_num_inference_steps")),
        "logprob_reduction": config.get("logprob_reduction"),
        "logprob_std_floor": _number(config.get("logprob_std_floor")),
        "trainable_mode": config.get("trainable_mode"),
        "final_grad_norm": last_step_summary.get("grad_norm"),
        "final_param_update_norm": last_step_summary.get("param_update_norm"),
        "final_param_update_max": last_step_summary.get("param_update_max"),
        "final_param_update_param_count": last_step_summary.get("param_update_param_count"),
        "parameter_update_measured": "param_update_norm" in last_step_summary,
        "parameter_update_detected": _positive_number(last_step_summary.get("param_update_norm")),
        "last_step": last_step_summary,
    }
    summary["ok"] = (
        summary["metrics_exists"]
        and summary["validation_ok"]
        and summary["checkpoint_exists"]
        and not summary["failure_diagnostics_exists"]
    )
    summary["warnings"] = _warnings(summary)
    return summary


def discover_output_dirs(root: Path, *, pattern: str = "*", latest: int | None = None) -> list[Path]:
    expanded_root = root.expanduser()
    if not expanded_root.exists():
        raise FileNotFoundError(f"discover root does not exist: {expanded_root}")
    if not expanded_root.is_dir():
        raise NotADirectoryError(f"discover root is not a directory: {expanded_root}")
    if latest is not None and latest <= 0:
        raise ValueError("latest must be positive when set")

    dirs = [path for path in expanded_root.glob(pattern) if path.is_dir()]
    dirs = sorted(dirs, key=lambda path: (path.stat().st_mtime, str(path)))
    if latest is not None:
        dirs = dirs[-latest:]
    return dirs


def load_job_log_configs(paths: Iterable[Path]) -> dict[str, dict]:
    configs: dict[str, dict] = {}
    for path in paths:
        output_dir, config = parse_actor_replay_job_log(path)
        if output_dir and config:
            configs[_path_key(output_dir)] = config
    return configs


def parse_actor_replay_job_log(path: Path) -> tuple[Path | None, dict]:
    output_dir: Path | None = None
    config: dict = {}
    if not path.expanduser().exists():
        return None, {}
    for raw_line in path.expanduser().read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line.startswith("GRPO_OUTPUT_DIR="):
            output_dir = Path(line.split("=", 1)[1])
            continue
        if line.startswith("Actor replay GRPO training complete:"):
            output_dir = Path(line.split(":", 1)[1].strip())
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        field = _JOB_LOG_CONFIG_FIELDS.get(key)
        if field is None:
            continue
        config[field] = _job_log_config_value(field, value)
    return output_dir, config


def write_json_report(summaries: list[dict], out_json: Path) -> None:
    out_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
    out_json.expanduser().write_text(json.dumps({"runs": summaries}, indent=2) + "\n", encoding="utf-8")


def write_markdown_report(summaries: list[dict], out_markdown: Path) -> None:
    lines = [
        "# Actor Replay Training Summary",
        "",
        "| output_dir | ok | validation | transitions | steps | config | config_name | git_commit | lr | action_steps | logprob | final_loss | ratio | grad_norm | update_norm | update_max | update | checkpoint | failure_diag | warnings |",
        "|---|---:|---:|---:|---:|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        lines.append(
            "| {output_dir} | {ok} | {validation_ok} | {transition_count} | {steps} | {config_source} | "
            "{config_name} | {git_commit} | {learning_rate} | {action_num_inference_steps} | {logprob_reduction} | "
            "{final_loss} | {final_ratio_mean} | {final_grad_norm} | "
            "{final_param_update_norm} | {final_param_update_max} | {parameter_update_detected} | {checkpoint_exists} | "
            "{failure_diagnostics_exists} | {warnings} |".format(
                output_dir=item["output_dir"],
                ok=_bool_cell(item["ok"]),
                validation_ok=_bool_cell(item["validation_ok"]),
                transition_count=_cell(item["transition_count"]),
                steps=_cell(item["steps"]),
                config_source=_cell(item["config_source"]),
                config_name=_cell(item["config_name"]),
                git_commit=_short_commit(item["git_commit"]),
                learning_rate=_cell(item["learning_rate"]),
                action_num_inference_steps=_cell(item["action_num_inference_steps"]),
                logprob_reduction=_cell(item["logprob_reduction"]),
                final_loss=_cell(item["final_loss"]),
                final_ratio_mean=_cell(item["final_ratio_mean"]),
                final_grad_norm=_cell(item["final_grad_norm"]),
                final_param_update_norm=_cell(item["final_param_update_norm"]),
                final_param_update_max=_cell(item["final_param_update_max"]),
                parameter_update_detected=_bool_cell(item["parameter_update_detected"]),
                checkpoint_exists=_bool_cell(item["checkpoint_exists"]),
                failure_diagnostics_exists=_bool_cell(item["failure_diagnostics_exists"]),
                warnings=", ".join(item["warnings"]),
            )
        )
    out_markdown.expanduser().parent.mkdir(parents=True, exist_ok=True)
    out_markdown.expanduser().write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv_report(summaries: list[dict], out_csv: Path) -> None:
    fields = [
        "output_dir",
        "ok",
        "validation_ok",
        "transition_count",
        "steps",
        "config_source",
        "model_path",
        "config_name",
        "git_commit",
        "learning_rate",
        "action_num_inference_steps",
        "logprob_reduction",
        "logprob_std_floor",
        "trainable_mode",
        "final_loss",
        "final_ratio_mean",
        "final_grad_norm",
        "final_param_update_norm",
        "final_param_update_max",
        "parameter_update_detected",
        "checkpoint_exists",
        "failure_diagnostics_exists",
        "warnings",
    ]
    out_csv.expanduser().parent.mkdir(parents=True, exist_ok=True)
    with out_csv.expanduser().open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            row = {field: item.get(field) for field in fields}
            row["warnings"] = ";".join(item.get("warnings", []))
            writer.writerow(row)


def format_text_report(summaries: list[dict]) -> str:
    """Return a compact terminal table for quick training-run triage."""

    columns = [
        ("run", lambda item: Path(item["output_dir"]).name),
        ("ok", lambda item: _bool_cell(item["ok"])),
        ("val", lambda item: _bool_cell(item["validation_ok"])),
        ("trans", lambda item: _cell(item["transition_count"])),
        ("steps", lambda item: _cell(item["steps"])),
        ("cfg", lambda item: _cell(item["config_name"])),
        ("git", lambda item: _short_commit(item["git_commit"])),
        ("lr", lambda item: _cell(item["learning_rate"])),
        ("logprob", lambda item: _cell(item["logprob_reduction"])),
        ("std_floor", lambda item: _cell(item["logprob_std_floor"])),
        ("loss", lambda item: _cell(item["final_loss"])),
        ("ratio", lambda item: _cell(item["final_ratio_mean"])),
        ("update", lambda item: _bool_cell(item["parameter_update_detected"])),
        ("update_norm", lambda item: _cell(item["final_param_update_norm"])),
        ("warnings", lambda item: ",".join(item["warnings"])),
    ]
    rows = [[formatter(item) for _, formatter in columns] for item in summaries]
    widths = [
        max(len(header), *(len(row[idx]) for row in rows)) if rows else len(header)
        for idx, (header, _) in enumerate(columns)
    ]
    header = "  ".join(name.ljust(widths[idx]) for idx, (name, _) in enumerate(columns))
    separator = "  ".join("-" * width for width in widths)
    lines = [header, separator]
    for row in rows:
        lines.append("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))
    return "\n".join(lines)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


_JOB_LOG_CONFIG_FIELDS = {
    "GRPO_CONFIG_NAME": "config_name",
    "GRPO_LR": "learning_rate",
    "GRPO_ACTION_NUM_INFERENCE_STEPS": "action_num_inference_steps",
    "GRPO_LOGPROB_REDUCTION": "logprob_reduction",
    "GRPO_LOGPROB_STD_FLOOR": "logprob_std_floor",
    "GRPO_TRAINABLE_MODE": "trainable_mode",
}


def _job_log_config_value(field: str, value: str):
    if field in {"learning_rate", "logprob_std_floor"}:
        return _number(value)
    if field == "action_num_inference_steps":
        number = _number(value)
        return int(number) if isinstance(number, (int, float)) and not isinstance(number, bool) else number
    return value


def _training_config(
    metrics: dict,
    checkpoint_path: Path,
    output_dir: Path,
    job_log_configs: dict[str, dict],
) -> tuple[dict, str]:
    if isinstance(metrics, dict) and isinstance(metrics.get("config"), dict) and metrics["config"]:
        return _scalar_config(metrics["config"]), "metrics"

    checkpoint_config = _read_checkpoint_config(checkpoint_path)
    if checkpoint_config:
        return checkpoint_config, "checkpoint"

    job_log_config = job_log_configs.get(_path_key(output_dir))
    if job_log_config:
        return _scalar_config(job_log_config), "job_log"

    return {}, "missing"


def _read_checkpoint_config(checkpoint_path: Path) -> dict:
    if not checkpoint_path.exists():
        return {}
    try:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return {}
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
        return {}
    return _scalar_config(checkpoint["config"])


def _scalar_config(config: dict) -> dict:
    return {key: config[key] for key in CONFIG_FIELDS if key in config and _is_json_scalar(config[key])}


def _is_json_scalar(value) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _path_key(path: Path) -> str:
    return str(path.expanduser())


def _number(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _last_step_summary(step: dict) -> dict:
    keys = [
        "step",
        "loss",
        "ratio_mean",
        "ratio_min",
        "ratio_max",
        "clip_fraction",
        "logratio_mean",
        "grad_norm",
        "param_update_norm",
        "param_update_max",
    ]
    return {key: _number(step.get(key)) for key in keys if key in step}


def _positive_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0.0


def _warnings(summary: dict) -> list[str]:
    warnings = []
    if not summary["metrics_exists"]:
        warnings.append("missing_metrics")
    if not summary["validation_exists"]:
        warnings.append("missing_validation")
    if summary["validation_exists"] and not summary["validation_ok"]:
        warnings.append("validation_failed")
    if not summary["checkpoint_exists"]:
        warnings.append("missing_checkpoint")
    if summary["failure_diagnostics_exists"]:
        warnings.append("failure_diagnostics_present")

    steps = summary.get("steps")
    trainable_param_count = summary.get("trainable_param_count")
    should_have_update_metric = (
        summary["metrics_exists"]
        and isinstance(steps, (int, float))
        and steps > 0
        and isinstance(trainable_param_count, (int, float))
        and trainable_param_count > 0
    )
    if should_have_update_metric and not summary["parameter_update_measured"]:
        warnings.append("missing_parameter_update_metric")
    if should_have_update_metric and summary["parameter_update_measured"] and not summary["parameter_update_detected"]:
        warnings.append("no_parameter_update_detected")
    if summary["metrics_exists"] and summary["checkpoint_exists"] and summary.get("config_source") == "missing":
        warnings.append("missing_training_config")
    return warnings


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _short_commit(value) -> str:
    if value is None:
        return ""
    text = str(value)
    return text[:12] if len(text) > 12 else text


def _bool_cell(value: bool) -> str:
    return "yes" if value else "no"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize actor replay GRPO training output directories.")
    parser.add_argument("output_dirs", nargs="*", type=Path, help="Actor replay training output directories.")
    parser.add_argument(
        "--discover-root",
        action="append",
        type=Path,
        default=[],
        help="Discover immediate child run directories under this root. Can be passed more than once.",
    )
    parser.add_argument("--discover-pattern", default="*", help="Glob pattern for --discover-root child directories.")
    parser.add_argument("--latest", type=int, default=None, help="Keep only the latest N discovered/explicit run dirs.")
    parser.add_argument(
        "--job-log-glob",
        action="append",
        default=[],
        help="Optional job-log glob used to recover config for older actor replay outputs.",
    )
    parser.add_argument(
        "--print-format",
        choices=("json", "table"),
        default="json",
        help="Stdout format. JSON is machine-readable; table is concise for terminal triage.",
    )
    parser.add_argument("--out-json", type=Path, help="Optional JSON summary path.")
    parser.add_argument("--out-csv", type=Path, help="Optional CSV summary path.")
    parser.add_argument("--out-markdown", type=Path, help="Optional Markdown summary path.")
    args = parser.parse_args()

    output_dirs = list(args.output_dirs)
    for root in args.discover_root:
        output_dirs.extend(discover_output_dirs(root, pattern=args.discover_pattern))
    if args.latest is not None:
        if args.latest <= 0:
            parser.error("--latest must be positive")
        expanded_dirs = [path.expanduser() for path in output_dirs]
        output_dirs = sorted(expanded_dirs, key=lambda path: (path.stat().st_mtime, str(path)))[-args.latest :]
    if not output_dirs:
        parser.error("provide at least one output directory or --discover-root")

    job_log_paths: list[Path] = []
    for pattern in args.job_log_glob:
        job_log_paths.extend(Path(item) for item in glob.glob(pattern))
    job_log_configs = load_job_log_configs(job_log_paths)

    summaries = [summarize_actor_replay_output(path, job_log_configs=job_log_configs) for path in output_dirs]
    report = {"runs": summaries}
    if args.out_json is not None:
        write_json_report(summaries, args.out_json)
    if args.out_csv is not None:
        write_csv_report(summaries, args.out_csv)
    if args.out_markdown is not None:
        write_markdown_report(summaries, args.out_markdown)
    if args.print_format == "table":
        print(format_text_report(summaries))
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
