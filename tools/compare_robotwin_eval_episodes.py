#!/usr/bin/env python3
"""Compare RoboTwin eval runs on matched per-episode keys."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

try:
    from tools._repo_root import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _repo_root import ensure_repo_root_on_path

ensure_repo_root_on_path()

from tools.summarize_robotwin_results import EpisodeResult, load_episode_results


DEFAULT_MATCH_FIELDS = ("task", "seed", "prompt_index", "sampling_seed")


def compare_eval_runs(
    runs: list[tuple[str, Path]],
    *,
    match_fields: tuple[str, ...] = DEFAULT_MATCH_FIELDS,
    include_episodes: bool = True,
) -> dict:
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
    matched = []
    for key in sorted(matched_keys):
        by_run = {label: indexed[label][key][0] for label in labels}
        matched.append((key, by_run))

    run_summaries = {}
    for label in labels:
        episodes = [by_run[label] for _, by_run in matched]
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

    pairwise = []
    baseline_label = labels[0]
    for label in labels[1:]:
        improved = regressed = same_success = same_failure = 0
        for _, by_run in matched:
            baseline_success = by_run[baseline_label].success
            current_success = by_run[label].success
            if current_success and not baseline_success:
                improved += 1
            elif baseline_success and not current_success:
                regressed += 1
            elif current_success and baseline_success:
                same_success += 1
            else:
                same_failure += 1
        pairwise.append(
            {
                "baseline": baseline_label,
                "candidate": label,
                "matched_episode_count": len(matched),
                "improved_count": improved,
                "regressed_count": regressed,
                "same_success_count": same_success,
                "same_failure_count": same_failure,
                "net_improvement_count": improved - regressed,
            }
        )

    payload = {
        "runs": [{"label": label, "root": str(root.expanduser())} for label, root in runs],
        "match_fields": list(match_fields),
        "matched_episode_count": len(matched),
        "run_summaries": run_summaries,
        "pairwise_vs_first": pairwise,
    }
    if include_episodes:
        payload["episodes"] = [
            {
                "key": _key_to_dict(match_fields, key),
                "runs": {label: _episode_to_dict(episode) for label, episode in by_run.items()},
            }
            for key, by_run in matched
        ]
    return payload


def write_comparison_csv(path: Path, comparison: dict) -> None:
    path.expanduser().parent.mkdir(parents=True, exist_ok=True)
    labels = [item["label"] for item in comparison["runs"]]
    match_fields = comparison["match_fields"]
    fieldnames = list(match_fields)
    for label in labels:
        fieldnames.extend(
            [
                f"{label}_success",
                f"{label}_seed",
                f"{label}_sampling_seed",
                f"{label}_prompt_index",
                f"{label}_action_count",
                f"{label}_episode_file",
            ]
        )
    with path.expanduser().open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in comparison.get("episodes", []):
            row = dict(item["key"])
            for label in labels:
                episode = item["runs"][label]
                row[f"{label}_success"] = episode["success"]
                row[f"{label}_seed"] = episode["seed"]
                row[f"{label}_sampling_seed"] = episode["sampling_seed"]
                row[f"{label}_prompt_index"] = episode["prompt_index"]
                row[f"{label}_action_count"] = episode["action_count"]
                row[f"{label}_episode_file"] = episode["episode_file"]
            writer.writerow(row)


def _index_by_key(episodes: list[EpisodeResult], fields: tuple[str, ...]) -> dict[tuple, list[EpisodeResult]]:
    by_key: dict[tuple, list[EpisodeResult]] = defaultdict(list)
    for episode in episodes:
        by_key[_episode_key(episode, fields)].append(episode)
    return by_key


def _episode_key(episode: EpisodeResult, fields: tuple[str, ...]) -> tuple:
    return tuple(getattr(episode, field) for field in fields)


def _key_to_dict(fields: tuple[str, ...], key: tuple) -> dict:
    return {field: value for field, value in zip(fields, key, strict=True)}


def _episode_to_dict(episode: EpisodeResult) -> dict:
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
    parser = argparse.ArgumentParser(description="Compare RoboTwin eval runs on matched episode keys.")
    parser.add_argument("--run", action="append", type=_parse_run, required=True, help="Run in LABEL=ROOT form.")
    parser.add_argument(
        "--match-fields",
        nargs="*",
        help="Episode fields used for matching. Default: task seed prompt_index sampling_seed.",
    )
    parser.add_argument("--out-json", type=Path, help="Optional JSON output path.")
    parser.add_argument("--out-csv", type=Path, help="Optional matched episode CSV output path.")
    parser.add_argument("--no-episodes", action="store_true", help="Omit matched episode details from JSON output.")
    args = parser.parse_args()

    comparison = compare_eval_runs(
        args.run,
        match_fields=_parse_match_fields(args.match_fields),
        include_episodes=not args.no_episodes or bool(args.out_csv),
    )
    if args.out_json:
        args.out_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.out_json.expanduser().write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    if args.out_csv:
        if "episodes" not in comparison:
            comparison = compare_eval_runs(
                args.run,
                match_fields=tuple(comparison["match_fields"]),
                include_episodes=True,
            )
        write_comparison_csv(args.out_csv, comparison)
    print(json.dumps({key: value for key, value in comparison.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
