#!/usr/bin/env python3
"""Gate actor-replay eval promotion using paired eval and baseline repeatability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def gate_actor_eval_promotion(
    comparison: dict[str, Any],
    baseline_repeatability: dict[str, Any],
    *,
    candidate: str | None = None,
    min_eval_matched_episodes: int = 10,
    min_repeatability_matched_episodes: int = 10,
    max_baseline_flip_rate: float = 0.1,
    min_net_improvement_count: int = 1,
    min_net_improvement_rate: float = 0.05,
) -> dict[str, Any]:
    """Return a conservative promotion decision for a baseline-vs-actor eval.

    This is intentionally a guardrail, not a significance test. It prevents
    tiny or repeatability-dominated RoboTwin evals from being treated as policy
    improvement evidence.
    """

    pairwise = comparison.get("pairwise_vs_first", [])
    if not pairwise:
        raise ValueError("comparison has no pairwise_vs_first entries")

    selected = _select_pairwise(pairwise, candidate)
    matched = int(comparison.get("matched_episode_count", selected.get("matched_episode_count", 0)))
    repeat_matched = int(baseline_repeatability.get("matched_episode_count", 0))
    flip_rate = float(baseline_repeatability.get("flip_rate", 0.0))

    improved = int(selected.get("improved_count", 0))
    regressed = int(selected.get("regressed_count", 0))
    net = int(selected.get("net_improvement_count", improved - regressed))
    net_rate = net / matched if matched else 0.0

    blockers: list[str] = []
    warnings: list[str] = []

    if matched < min_eval_matched_episodes:
        blockers.append(
            f"matched eval episodes {matched} < required {min_eval_matched_episodes}"
        )
    if repeat_matched < min_repeatability_matched_episodes:
        blockers.append(
            f"baseline repeatability episodes {repeat_matched} < required {min_repeatability_matched_episodes}"
        )
    if flip_rate > max_baseline_flip_rate:
        blockers.append(
            f"baseline repeatability flip_rate {flip_rate:.6g} > allowed {max_baseline_flip_rate:.6g}"
        )
    if net < min_net_improvement_count:
        blockers.append(
            f"net improvement count {net} < required {min_net_improvement_count}"
        )
    if net_rate < min_net_improvement_rate:
        blockers.append(
            f"net improvement rate {net_rate:.6g} < required {min_net_improvement_rate:.6g}"
        )
    if net_rate <= flip_rate:
        blockers.append(
            f"net improvement rate {net_rate:.6g} is not above baseline flip_rate {flip_rate:.6g}"
        )

    if regressed:
        warnings.append(f"candidate has {regressed} matched regressions")

    decision = "promote" if not blockers else "blocked"
    return {
        "decision": decision,
        "candidate": selected.get("candidate", candidate),
        "baseline": selected.get("baseline"),
        "blockers": blockers,
        "warnings": warnings,
        "thresholds": {
            "min_eval_matched_episodes": min_eval_matched_episodes,
            "min_repeatability_matched_episodes": min_repeatability_matched_episodes,
            "max_baseline_flip_rate": max_baseline_flip_rate,
            "min_net_improvement_count": min_net_improvement_count,
            "min_net_improvement_rate": min_net_improvement_rate,
        },
        "metrics": {
            "matched_episode_count": matched,
            "baseline_repeatability_matched_episode_count": repeat_matched,
            "baseline_flip_rate": flip_rate,
            "improved_count": improved,
            "regressed_count": regressed,
            "same_success_count": int(selected.get("same_success_count", 0)),
            "same_failure_count": int(selected.get("same_failure_count", 0)),
            "net_improvement_count": net,
            "net_improvement_rate": net_rate,
        },
    }


def write_gate_markdown(path: Path, decision: dict[str, Any]) -> None:
    lines = [
        "# Actor Eval Promotion Gate",
        "",
        f"- decision: `{decision['decision']}`",
        f"- baseline: `{decision.get('baseline')}`",
        f"- candidate: `{decision.get('candidate')}`",
        "",
        "## Metrics",
        "",
    ]
    for key, value in decision["metrics"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Thresholds", ""])
    for key, value in decision["thresholds"].items():
        lines.append(f"- {key}: `{value}`")
    if decision["blockers"]:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {item}" for item in decision["blockers"])
    if decision["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in decision["warnings"])
    lines.extend(
        [
            "",
            "This gate is a conservative engineering guardrail. It does not prove statistical significance; it prevents repeatability-dominated or tiny evals from being promoted.",
            "",
        ]
    )
    path.expanduser().parent.mkdir(parents=True, exist_ok=True)
    path.expanduser().write_text("\n".join(lines), encoding="utf-8")


def _select_pairwise(pairwise: list[dict[str, Any]], candidate: str | None) -> dict[str, Any]:
    if candidate is None:
        if len(pairwise) != 1:
            raise ValueError("comparison has multiple candidates; pass --candidate")
        return pairwise[0]
    for item in pairwise:
        if item.get("candidate") == candidate:
            return item
    raise ValueError(f"candidate {candidate!r} not found in comparison")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate actor-replay eval promotion using paired eval and baseline repeatability."
    )
    parser.add_argument("--comparison", type=Path, required=True, help="Path to comparison.json.")
    parser.add_argument(
        "--baseline-repeatability",
        type=Path,
        required=True,
        help="Path to baseline repeatability summary JSON.",
    )
    parser.add_argument("--candidate", help="Candidate label when comparison has multiple candidates.")
    parser.add_argument("--min-eval-matched-episodes", type=int, default=10)
    parser.add_argument("--min-repeatability-matched-episodes", type=int, default=10)
    parser.add_argument("--max-baseline-flip-rate", type=float, default=0.1)
    parser.add_argument("--min-net-improvement-count", type=int, default=1)
    parser.add_argument("--min-net-improvement-rate", type=float, default=0.05)
    parser.add_argument("--out-json", type=Path, help="Optional JSON output path.")
    parser.add_argument("--out-markdown", type=Path, help="Optional Markdown report path.")
    args = parser.parse_args()

    decision = gate_actor_eval_promotion(
        _load_json(args.comparison),
        _load_json(args.baseline_repeatability),
        candidate=args.candidate,
        min_eval_matched_episodes=args.min_eval_matched_episodes,
        min_repeatability_matched_episodes=args.min_repeatability_matched_episodes,
        max_baseline_flip_rate=args.max_baseline_flip_rate,
        min_net_improvement_count=args.min_net_improvement_count,
        min_net_improvement_rate=args.min_net_improvement_rate,
    )
    if args.out_json:
        args.out_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.out_json.expanduser().write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    if args.out_markdown:
        write_gate_markdown(args.out_markdown, decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
