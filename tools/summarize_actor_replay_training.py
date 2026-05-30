#!/usr/bin/env python3
"""Summarize actor replay GRPO training output directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CONFIG_FIELDS = (
    "learning_rate",
    "action_num_inference_steps",
    "logprob_reduction",
    "logprob_std_floor",
    "trainable_mode",
)


def summarize_actor_replay_output(output_dir: Path) -> dict:
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
    config, config_source = _training_config(metrics, checkpoint_path)

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


def write_json_report(summaries: list[dict], out_json: Path) -> None:
    out_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
    out_json.expanduser().write_text(json.dumps({"runs": summaries}, indent=2) + "\n", encoding="utf-8")


def write_markdown_report(summaries: list[dict], out_markdown: Path) -> None:
    lines = [
        "# Actor Replay Training Summary",
        "",
        "| output_dir | ok | validation | transitions | steps | config | lr | action_steps | logprob | final_loss | ratio | grad_norm | update_norm | update_max | update | checkpoint | failure_diag | warnings |",
        "|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        lines.append(
            "| {output_dir} | {ok} | {validation_ok} | {transition_count} | {steps} | {config_source} | "
            "{learning_rate} | {action_num_inference_steps} | {logprob_reduction} | "
            "{final_loss} | {final_ratio_mean} | {final_grad_norm} | "
            "{final_param_update_norm} | {final_param_update_max} | {parameter_update_detected} | {checkpoint_exists} | "
            "{failure_diagnostics_exists} | {warnings} |".format(
                output_dir=item["output_dir"],
                ok=_bool_cell(item["ok"]),
                validation_ok=_bool_cell(item["validation_ok"]),
                transition_count=_cell(item["transition_count"]),
                steps=_cell(item["steps"]),
                config_source=_cell(item["config_source"]),
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


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _training_config(metrics: dict, checkpoint_path: Path) -> tuple[dict, str]:
    if isinstance(metrics, dict) and isinstance(metrics.get("config"), dict) and metrics["config"]:
        return _scalar_config(metrics["config"]), "metrics"

    checkpoint_config = _read_checkpoint_config(checkpoint_path)
    if checkpoint_config:
        return checkpoint_config, "checkpoint"

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

    summaries = [summarize_actor_replay_output(path) for path in output_dirs]
    report = {"runs": summaries}
    if args.out_json is not None:
        write_json_report(summaries, args.out_json)
    if args.out_csv is not None:
        write_csv_report(summaries, args.out_csv)
    if args.out_markdown is not None:
        write_markdown_report(summaries, args.out_markdown)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
