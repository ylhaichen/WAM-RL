#!/usr/bin/env python3
"""Summarize actor replay GRPO training output directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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
    checkpoint_path = Path(result.get("checkpoint_path") or root / "checkpoint.pt").expanduser()

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
        "last_step": _last_step_summary(last_step),
    }
    summary["ok"] = (
        summary["metrics_exists"]
        and summary["validation_ok"]
        and summary["checkpoint_exists"]
        and not summary["failure_diagnostics_exists"]
    )
    return summary


def write_json_report(summaries: list[dict], out_json: Path) -> None:
    out_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
    out_json.expanduser().write_text(json.dumps({"runs": summaries}, indent=2) + "\n", encoding="utf-8")


def write_markdown_report(summaries: list[dict], out_markdown: Path) -> None:
    lines = [
        "# Actor Replay Training Summary",
        "",
        "| output_dir | ok | validation | transitions | steps | final_loss | ratio | checkpoint | failure_diag |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {output_dir} | {ok} | {validation_ok} | {transition_count} | {steps} | {final_loss} | "
            "{final_ratio_mean} | {checkpoint_exists} | {failure_diagnostics_exists} |".format(
                output_dir=item["output_dir"],
                ok=_bool_cell(item["ok"]),
                validation_ok=_bool_cell(item["validation_ok"]),
                transition_count=_cell(item["transition_count"]),
                steps=_cell(item["steps"]),
                final_loss=_cell(item["final_loss"]),
                final_ratio_mean=_cell(item["final_ratio_mean"]),
                checkpoint_exists=_bool_cell(item["checkpoint_exists"]),
                failure_diagnostics_exists=_bool_cell(item["failure_diagnostics_exists"]),
            )
        )
    out_markdown.expanduser().parent.mkdir(parents=True, exist_ok=True)
    out_markdown.expanduser().write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
    parser.add_argument("output_dirs", nargs="+", type=Path, help="Actor replay training output directories.")
    parser.add_argument("--out-json", type=Path, help="Optional JSON summary path.")
    parser.add_argument("--out-markdown", type=Path, help="Optional Markdown summary path.")
    args = parser.parse_args()

    summaries = [summarize_actor_replay_output(path) for path in args.output_dirs]
    report = {"runs": summaries}
    if args.out_json is not None:
        write_json_report(summaries, args.out_json)
    if args.out_markdown is not None:
        write_markdown_report(summaries, args.out_markdown)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
