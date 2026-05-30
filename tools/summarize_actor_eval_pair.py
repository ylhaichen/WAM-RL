#!/usr/bin/env python3
"""Summarize matched baseline-vs-actor RoboTwin evaluation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools._repo_root import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _repo_root import ensure_repo_root_on_path

ensure_repo_root_on_path()

from tools.compare_robotwin_eval_episodes import compare_eval_runs, write_comparison_csv
from tools.summarize_robotwin_results import (
    EpisodeResult,
    format_table,
    load_episode_results,
    load_results,
    write_csv,
    write_episode_csv,
    write_episode_json,
)

PROVENANCE_FIELDS = (
    "run_id",
    "policy_checkpoint",
    "reference_checkpoint",
    "action_num_inference_steps",
    "sampling_seed",
    "prompt_index",
    "video_guidance_scale",
    "action_guidance_scale",
)
REQUIRED_PROVENANCE_FIELDS = (
    "run_id",
    "policy_checkpoint",
    "reference_checkpoint",
    "action_num_inference_steps",
    "sampling_seed",
    "prompt_index",
)


def summarize_eval_pair(
    baseline_root: Path,
    actor_root: Path,
    out_root: Path,
    *,
    baseline_label: str = "baseline",
    actor_label: str = "actor",
    match_fields: tuple[str, ...] | None = None,
    min_matched_episodes: int = 1,
) -> dict:
    baseline_root = baseline_root.expanduser()
    actor_root = actor_root.expanduser()
    out_root = out_root.expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    baseline_results = load_results(baseline_root)
    actor_results = load_results(actor_root)
    if not baseline_results:
        raise ValueError(f"No baseline res.json files found under {baseline_root}")
    if not actor_results:
        raise ValueError(f"No actor res.json files found under {actor_root}")

    baseline_episodes = load_episode_results(baseline_root)
    actor_episodes = load_episode_results(actor_root)
    run_provenance = {
        baseline_label: summarize_episode_provenance(baseline_episodes),
        actor_label: summarize_episode_provenance(actor_episodes),
    }
    provenance_warnings = build_provenance_warnings(
        {
            baseline_label: baseline_episodes,
            actor_label: actor_episodes,
        },
        run_provenance,
    )

    write_csv(out_root / f"{baseline_label}_summary.csv", baseline_results)
    write_csv(out_root / f"{actor_label}_summary.csv", actor_results)
    write_episode_csv(out_root / f"{baseline_label}_episodes.csv", baseline_episodes)
    write_episode_csv(out_root / f"{actor_label}_episodes.csv", actor_episodes)
    write_episode_json(out_root / f"{baseline_label}_episodes.json", baseline_episodes)
    write_episode_json(out_root / f"{actor_label}_episodes.json", actor_episodes)

    compare_kwargs = {}
    if match_fields is not None:
        compare_kwargs["match_fields"] = match_fields
    comparison = compare_eval_runs(
        [(baseline_label, baseline_root), (actor_label, actor_root)],
        **compare_kwargs,
    )
    if comparison["matched_episode_count"] < min_matched_episodes:
        raise ValueError(
            f"only matched {comparison['matched_episode_count']} episodes; "
            f"expected at least {min_matched_episodes}. Check SEED, PROMPT_INDEX, "
            "SAMPLING_SEED, and match fields, or pass --min-matched-episodes 0 "
            "for aggregate-only inspection."
        )
    (out_root / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    write_comparison_csv(out_root / "comparison.csv", comparison)
    write_pair_markdown(
        out_root / "summary.md",
        baseline_label=baseline_label,
        actor_label=actor_label,
        baseline_root=baseline_root,
        actor_root=actor_root,
        baseline_results=baseline_results,
        actor_results=actor_results,
        comparison=comparison,
        run_provenance=run_provenance,
        provenance_warnings=provenance_warnings,
    )

    summary = {
        "baseline_root": str(baseline_root),
        "actor_root": str(actor_root),
        "out_root": str(out_root),
        "baseline_episode_count": len(baseline_episodes),
        "actor_episode_count": len(actor_episodes),
        "matched_episode_count": comparison["matched_episode_count"],
        "run_provenance": run_provenance,
        "provenance_warnings": provenance_warnings,
        "run_summaries": comparison["run_summaries"],
        "pairwise_vs_first": comparison["pairwise_vs_first"],
        "outputs": {
            "baseline_summary_csv": str(out_root / f"{baseline_label}_summary.csv"),
            "actor_summary_csv": str(out_root / f"{actor_label}_summary.csv"),
            "baseline_episodes_csv": str(out_root / f"{baseline_label}_episodes.csv"),
            "actor_episodes_csv": str(out_root / f"{actor_label}_episodes.csv"),
            "comparison_json": str(out_root / "comparison.json"),
            "comparison_csv": str(out_root / "comparison.csv"),
            "summary_markdown": str(out_root / "summary.md"),
            "summary_json": str(out_root / "summary.json"),
        },
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def write_pair_markdown(
    path: Path,
    *,
    baseline_label: str,
    actor_label: str,
    baseline_root: Path,
    actor_root: Path,
    baseline_results: list,
    actor_results: list,
    comparison: dict,
    run_provenance: dict,
    provenance_warnings: list[dict],
) -> None:
    lines = [
        "# Actor Replay Eval Pair Summary",
        "",
        "## Inputs",
        "",
        f"- `{baseline_label}`: `{baseline_root}`",
        f"- `{actor_label}`: `{actor_root}`",
        "",
        "## Run Provenance",
        "",
        f"### {baseline_label}",
        "",
        *_format_provenance_lines(run_provenance.get(baseline_label, {})),
        "",
        f"### {actor_label}",
        "",
        *_format_provenance_lines(run_provenance.get(actor_label, {})),
        "",
    ]
    if provenance_warnings:
        lines.extend(
            [
                "## Provenance Warnings",
                "",
                *[
                    f"- {warning['label']}: missing {', '.join(warning['missing_fields'])} "
                    f"across {warning['episode_count']} exported episodes"
                    for warning in provenance_warnings
                ],
                "",
            ]
        )
    lines.extend(
        [
            "## Aggregate Results",
            "",
            f"### {baseline_label}",
            "",
            "```text",
            format_table(baseline_results),
            "```",
            "",
            f"### {actor_label}",
            "",
            "```text",
            format_table(actor_results),
            "```",
            "",
            "## Matched Episode Comparison",
            "",
            f"- matched episodes: {comparison['matched_episode_count']}",
        ]
    )
    for item in comparison["pairwise_vs_first"]:
        lines.extend(
            [
                f"- {item['candidate']} vs {item['baseline']}: "
                f"improved={item['improved_count']}, "
                f"regressed={item['regressed_count']}, "
                f"same_success={item['same_success_count']}, "
                f"same_failure={item['same_failure_count']}, "
                f"net={item['net_improvement_count']}",
            ]
        )
    lines.extend(
        [
            "",
            "Tiny evals are smoke checks only. Treat the paired counts as a wiring and regression signal, not as a benchmark-improvement claim.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_episode_provenance(episodes: list[EpisodeResult]) -> dict[str, list]:
    return {field: _unique_episode_values(episodes, field) for field in PROVENANCE_FIELDS}


def build_provenance_warnings(
    episode_sets: dict[str, list[EpisodeResult]],
    provenance_by_label: dict[str, dict[str, list]],
) -> list[dict]:
    warnings = []
    for label, episodes in episode_sets.items():
        if not episodes:
            warnings.append(
                {
                    "label": label,
                    "episode_count": 0,
                    "missing_fields": list(REQUIRED_PROVENANCE_FIELDS),
                    "message": f"{label} has no exported episode provenance.",
                }
            )
            continue

        provenance = provenance_by_label.get(label, {})
        missing_fields = [
            field
            for field in REQUIRED_PROVENANCE_FIELDS
            if not provenance.get(field)
        ]
        if not missing_fields:
            continue
        warnings.append(
            {
                "label": label,
                "episode_count": len(episodes),
                "missing_fields": missing_fields,
                "message": (
                    f"{label} is missing provenance fields: "
                    + ", ".join(missing_fields)
                ),
            }
        )
    return warnings


def _unique_episode_values(episodes: list[EpisodeResult], field: str) -> list:
    values = []
    seen = set()
    for episode in episodes:
        value = getattr(episode, field)
        if value in (None, ""):
            continue
        key = json.dumps(value, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        values.append(value)
    return sorted(values, key=lambda item: str(item))


def _format_provenance_lines(provenance: dict[str, list]) -> list[str]:
    return [f"- {field}: {_format_values(provenance.get(field, []))}" for field in PROVENANCE_FIELDS]


def _format_values(values: list) -> str:
    if not values:
        return "`<missing>`"
    shown = values[:5]
    text = ", ".join(f"`{value}`" for value in shown)
    if len(values) > len(shown):
        text += f", ... (+{len(values) - len(shown)} more)"
    return text


def _parse_match_fields(values: list[str] | None) -> tuple[str, ...] | None:
    if not values:
        return None
    fields: list[str] = []
    for value in values:
        fields.extend(item for item in value.replace(",", " ").split() if item)
    return tuple(fields)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a matched baseline-vs-actor RoboTwin eval pair.")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline eval result root.")
    parser.add_argument("--actor", type=Path, required=True, help="Actor eval result root.")
    parser.add_argument("--out-root", type=Path, required=True, help="Directory for summaries and comparison files.")
    parser.add_argument("--baseline-label", default="baseline", help="Label for the baseline run.")
    parser.add_argument("--actor-label", default="actor", help="Label for the actor run.")
    parser.add_argument(
        "--min-matched-episodes",
        type=int,
        default=1,
        help="Fail if fewer matched episodes are found. Use 0 for aggregate-only inspection.",
    )
    parser.add_argument(
        "--match-fields",
        nargs="*",
        help="Episode fields used for matching. Default comes from compare_robotwin_eval_episodes.py.",
    )
    args = parser.parse_args()

    summary = summarize_eval_pair(
        args.baseline,
        args.actor,
        args.out_root,
        baseline_label=args.baseline_label,
        actor_label=args.actor_label,
        match_fields=_parse_match_fields(args.match_fields),
        min_matched_episodes=args.min_matched_episodes,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
