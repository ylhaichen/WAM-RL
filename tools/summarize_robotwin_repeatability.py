#!/usr/bin/env python3
"""Summarize RoboTwin closed-loop repeatability across matched eval runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from tools._repo_root import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _repo_root import ensure_repo_root_on_path

ensure_repo_root_on_path()

from tools.compare_robotwin_eval_episodes import DEFAULT_MATCH_FIELDS
from tools.summarize_robotwin_results import EpisodeResult, load_episode_results


def summarize_repeatability(
    runs: list[tuple[str, Path]],
    *,
    match_fields: tuple[str, ...] = DEFAULT_MATCH_FIELDS,
    min_matched_episodes: int = 1,
    include_episodes: bool = True,
) -> dict[str, Any]:
    if len(runs) < 2:
        raise ValueError("at least two runs are required")
    labels = [label for label, _ in runs]
    if len(set(labels)) != len(labels):
        raise ValueError("run labels must be unique")

    indexed = {label: _index_by_key(load_episode_results(root.expanduser()), match_fields) for label, root in runs}
    duplicate_counts = {label: sum(1 for values in by_key.values() if len(values) > 1) for label, by_key in indexed.items()}
    valid_keys_by_label = {
        label: {key for key, values in by_key.items() if len(values) == 1}
        for label, by_key in indexed.items()
    }
    matched_keys = set.intersection(*valid_keys_by_label.values()) if valid_keys_by_label else set()
    if len(matched_keys) < min_matched_episodes:
        raise ValueError(
            f"only matched {len(matched_keys)} episodes; expected at least {min_matched_episodes}. "
            "Check SEED, PROMPT_INDEX, SAMPLING_SEED, and match fields, or pass "
            "--min-matched-episodes 0 for aggregate-only inspection."
        )

    episode_rows = []
    stable_success_count = 0
    stable_failure_count = 0
    flipped_count = 0
    for key in sorted(matched_keys):
        by_run = {label: indexed[label][key][0] for label in labels}
        success_count = sum(1 for episode in by_run.values() if episode.success)
        failure_count = len(labels) - success_count
        if success_count == len(labels):
            status = "stable_success"
            stable_success_count += 1
        elif success_count == 0:
            status = "stable_failure"
            stable_failure_count += 1
        else:
            status = "flipped"
            flipped_count += 1
        episode_rows.append(
            {
                "key": _key_to_dict(match_fields, key),
                "status": status,
                "success_count": success_count,
                "failure_count": failure_count,
                "success_rate": success_count / len(labels),
                "runs": {label: _episode_to_dict(episode) for label, episode in by_run.items()},
            }
        )

    run_summaries = {}
    for label in labels:
        episodes = [indexed[label][key][0] for key in matched_keys]
        success_count = sum(1 for episode in episodes if episode.success)
        run_summaries[label] = {
            "matched_episode_count": len(episodes),
            "success_count": success_count,
            "failure_count": len(episodes) - success_count,
            "success_rate": success_count / len(episodes) if episodes else 0.0,
            "available_unique_key_count": len(valid_keys_by_label[label]),
            "duplicate_key_count": duplicate_counts[label],
            "unmatched_unique_key_count": len(valid_keys_by_label[label] - matched_keys),
        }

    payload = {
        "runs": [{"label": label, "root": str(root.expanduser())} for label, root in runs],
        "match_fields": list(match_fields),
        "run_count": len(labels),
        "matched_episode_count": len(episode_rows),
        "stable_success_count": stable_success_count,
        "stable_failure_count": stable_failure_count,
        "flipped_count": flipped_count,
        "flip_rate": flipped_count / len(episode_rows) if episode_rows else 0.0,
        "run_summaries": run_summaries,
    }
    if include_episodes:
        payload["episodes"] = episode_rows
    return payload


def write_repeatability_csv(path: Path, summary: dict[str, Any]) -> None:
    path.expanduser().parent.mkdir(parents=True, exist_ok=True)
    labels = [item["label"] for item in summary["runs"]]
    match_fields = summary["match_fields"]
    fieldnames = list(match_fields) + ["status", "success_count", "failure_count", "success_rate"]
    for label in labels:
        fieldnames.extend([f"{label}_success", f"{label}_action_count", f"{label}_episode_file"])
    with path.expanduser().open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in summary.get("episodes", []):
            row = dict(item["key"])
            row.update(
                {
                    "status": item["status"],
                    "success_count": item["success_count"],
                    "failure_count": item["failure_count"],
                    "success_rate": item["success_rate"],
                }
            )
            for label in labels:
                episode = item["runs"][label]
                row[f"{label}_success"] = episode["success"]
                row[f"{label}_action_count"] = episode["action_count"]
                row[f"{label}_episode_file"] = episode["episode_file"]
            writer.writerow(row)


def write_repeatability_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# RoboTwin Eval Repeatability Summary",
        "",
        "## Overview",
        "",
        f"- runs: {summary['run_count']}",
        f"- matched episodes: {summary['matched_episode_count']}",
        f"- stable success: {summary['stable_success_count']}",
        f"- stable failure: {summary['stable_failure_count']}",
        f"- flipped: {summary['flipped_count']}",
        f"- flip rate: {summary['flip_rate']:.6g}",
        "",
        "## Run Summaries",
        "",
        "| run | matched | success | failure | success_rate | unmatched | duplicates |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, item in summary["run_summaries"].items():
        lines.append(
            "| {label} | {matched_episode_count} | {success_count} | {failure_count} | "
            "{success_rate:.6g} | {unmatched_unique_key_count} | {duplicate_key_count} |".format(
                label=label,
                **item,
            )
        )
    lines.extend(
        [
            "",
            "Tiny repeatability summaries are diagnostics only. Use them to decide whether larger paired evals are meaningful before interpreting policy deltas.",
            "",
        ]
    )
    path.expanduser().write_text("\n".join(lines), encoding="utf-8")


def _index_by_key(episodes: list[EpisodeResult], fields: tuple[str, ...]) -> dict[tuple, list[EpisodeResult]]:
    by_key: dict[tuple, list[EpisodeResult]] = defaultdict(list)
    for episode in episodes:
        by_key[_episode_key(episode, fields)].append(episode)
    return by_key


def _episode_key(episode: EpisodeResult, fields: tuple[str, ...]) -> tuple:
    return tuple(getattr(episode, field) for field in fields)


def _key_to_dict(fields: tuple[str, ...], key: tuple) -> dict[str, Any]:
    return {field: value for field, value in zip(fields, key, strict=True)}


def _episode_to_dict(episode: EpisodeResult) -> dict[str, Any]:
    return {
        "task": episode.task,
        "episode_file": episode.episode_file,
        "episode_index": episode.episode_index,
        "seed": episode.seed,
        "planned_seed": episode.planned_seed,
        "success": episode.success,
        "action_count": episode.action_count,
        "take_action_cnt": episode.take_action_cnt,
        "step_lim": episode.step_lim,
        "sampling_seed": episode.sampling_seed,
        "prompt_index": episode.prompt_index,
        "prompt": episode.prompt,
    }


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("run label must not be empty")
    return label, Path(path)


def _parse_match_fields(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return DEFAULT_MATCH_FIELDS
    fields: list[str] = []
    for value in values:
        fields.extend(item for item in value.replace(",", " ").split() if item)
    allowed = set(EpisodeResult.__dataclass_fields__)
    unknown = sorted(set(fields) - allowed)
    if unknown:
        raise ValueError(f"unknown match fields: {unknown}")
    return tuple(fields)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RoboTwin eval repeatability across matched episodes.")
    parser.add_argument("--run", action="append", type=_parse_run, required=True, help="Run in LABEL=ROOT form.")
    parser.add_argument(
        "--match-fields",
        nargs="*",
        help="Episode fields used for matching. Default: task seed prompt_index sampling_seed.",
    )
    parser.add_argument(
        "--min-matched-episodes",
        type=int,
        default=1,
        help="Fail if fewer matched episodes are found. Use 0 for aggregate-only inspection.",
    )
    parser.add_argument("--out-json", type=Path, help="Optional JSON output path.")
    parser.add_argument("--out-csv", type=Path, help="Optional matched episode CSV output path.")
    parser.add_argument("--out-markdown", type=Path, help="Optional Markdown report path.")
    parser.add_argument("--no-episodes", action="store_true", help="Omit matched episode details from JSON output.")
    args = parser.parse_args()

    summary = summarize_repeatability(
        args.run,
        match_fields=_parse_match_fields(args.match_fields),
        min_matched_episodes=args.min_matched_episodes,
        include_episodes=not args.no_episodes or bool(args.out_csv),
    )
    if args.out_json:
        args.out_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.out_json.expanduser().write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.out_csv:
        if "episodes" not in summary:
            summary = summarize_repeatability(
                args.run,
                match_fields=tuple(summary["match_fields"]),
                min_matched_episodes=args.min_matched_episodes,
                include_episodes=True,
            )
        write_repeatability_csv(args.out_csv, summary)
    if args.out_markdown:
        write_repeatability_markdown(args.out_markdown, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
