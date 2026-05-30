#!/usr/bin/env python3
"""Plan non-destructive WAM-RL storage cleanup candidates on Myriad."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


GiB = 1024**3


def plan_myriad_storage_cleanup(
    paths: list[Path],
    *,
    min_candidate_gb: float = 1.0,
    large_run_gb: float = 10.0,
) -> dict:
    """Inspect grouped rollout directories and return cleanup recommendations.

    The planner never deletes files and never emits executable cleanup commands.
    It only reports candidates that still need human review.
    """

    runs = []
    for path in paths:
        runs.extend(_discover_runs(path.expanduser()))

    run_reports = [
        _inspect_run(run, min_candidate_bytes=int(min_candidate_gb * GiB), large_run_bytes=int(large_run_gb * GiB))
        for run in sorted(set(runs))
    ]
    candidates = [candidate for run in run_reports for candidate in run["cleanup_candidates"]]
    protected = [run for run in run_reports if run["protection_reasons"]]
    return {
        "input_paths": [str(path.expanduser()) for path in paths],
        "run_count": len(run_reports),
        "total_disk_bytes": sum(run["disk_bytes"] for run in run_reports),
        "candidate_reclaimable_bytes": sum(candidate["target_disk_bytes"] for candidate in candidates),
        "candidate_reclaimable_gb": sum(candidate["target_disk_bytes"] for candidate in candidates) / GiB,
        "cleanup_candidate_count": len(candidates),
        "protected_run_count": len(protected),
        "runs": run_reports,
        "cleanup_candidates": candidates,
    }


def write_markdown_report(report: dict) -> str:
    lines = [
        "# Myriad Storage Cleanup Plan",
        "",
        "This is a non-destructive planning report. It does not delete files and",
        "does not include executable cleanup commands.",
        "",
        "## Summary",
        "",
        f"- scanned runs: {report['run_count']}",
        f"- cleanup candidates: {report['cleanup_candidate_count']}",
        f"- protected runs: {report['protected_run_count']}",
        f"- candidate reclaimable size: {_format_gb(report['candidate_reclaimable_bytes'])}",
        "",
        "## Cleanup Candidates",
        "",
    ]
    if report["cleanup_candidates"]:
        lines.extend(
            [
                "| Run | Action | Target | Size | Reason |",
                "|---|---|---|---:|---|",
            ]
        )
        for candidate in report["cleanup_candidates"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md(candidate["run_name"]),
                        _md(candidate["action"]),
                        _md(candidate["target_path"]),
                        _format_gb(candidate["target_disk_bytes"]),
                        _md(candidate["reason"]),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No cleanup candidates met the configured threshold.")

    lines.extend(["", "## Protected Runs", ""])
    protected = [run for run in report["runs"] if run["protection_reasons"]]
    if protected:
        lines.extend(["| Run | Size | Groups | Reason |", "|---|---:|---:|---|"])
        for run in protected:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md(run["name"]),
                        _format_gb(run["disk_bytes"]),
                        str(run["grpo_group_line_count"]),
                        _md("; ".join(run["protection_reasons"])),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No protected runs found.")

    lines.append("")
    return "\n".join(lines)


def compact_cleanup_summary(report: dict) -> dict:
    return {
        "run_count": report["run_count"],
        "cleanup_candidate_count": report["cleanup_candidate_count"],
        "protected_run_count": report["protected_run_count"],
        "candidate_reclaimable_gb": report["candidate_reclaimable_gb"],
        "top_candidates": [
            {
                "run_name": candidate["run_name"],
                "action": candidate["action"],
                "target_gb": candidate["target_disk_bytes"] / GiB,
                "reason": candidate["reason"],
            }
            for candidate in sorted(
                report["cleanup_candidates"],
                key=lambda item: int(item["target_disk_bytes"]),
                reverse=True,
            )[:10]
        ],
    }


def _discover_runs(path: Path) -> list[Path]:
    if (path / "groups").is_dir() or (path / "server_vis").is_dir():
        return [path]
    if not path.is_dir():
        return []
    return [child for child in path.iterdir() if child.is_dir()]


def _inspect_run(run: Path, *, min_candidate_bytes: int, large_run_bytes: int) -> dict:
    groups_dir = run / "groups"
    server_vis = run / "server_vis"
    grpo_groups = groups_dir / "grpo_groups.jsonl"
    group_files = _group_jsonl_files(groups_dir)
    summary_path = groups_dir / "grpo_summary.json"

    disk_bytes = _dir_disk_bytes(run)
    server_vis_bytes = _dir_disk_bytes(server_vis)
    attempts_bytes = _dir_disk_bytes(run / "attempts")
    groups_bytes = _dir_disk_bytes(groups_dir)
    group_line_count = _line_count(grpo_groups)
    total_group_line_count = sum(_line_count(path) for path in group_files)
    summary = _read_json(summary_path)
    successful_attempt_count = _line_count(groups_dir / "successful_attempt_roots.txt")
    failed_attempt_count = _line_count(groups_dir / "failed_attempt_roots.txt")

    protection_reasons: list[str] = []
    notes: list[str] = []
    candidates: list[dict] = []

    if total_group_line_count > 0:
        protection_reasons.append("non-empty groups/grpo_groups*.jsonl may reference trainable artifacts")
    if _is_known_curated_run_name(run.name):
        protection_reasons.append("name matches curated dataset/source run pattern")

    if server_vis_bytes >= min_candidate_bytes and total_group_line_count == 0:
        reason_parts = ["groups/grpo_groups*.jsonl files are empty or missing"]
        if int(summary.get("mixed_groups", 0) or 0) == 0 and int(summary.get("total_groups", 0) or 0) > 0:
            reason_parts.append("summary reports no mixed trainable groups")
        if failed_attempt_count > 0 and successful_attempt_count == 0:
            reason_parts.append("only failed attempts were recorded")
        candidates.append(
            _candidate(
                run=run,
                action="review_delete_server_vis_after_metadata_backup",
                target=server_vis,
                target_disk_bytes=server_vis_bytes,
                reason="; ".join(reason_parts),
            )
        )
    elif total_group_line_count > 0 and server_vis_bytes >= large_run_bytes:
        notes.append("large trainable source run; materialize/archive subsets before considering server_vis cleanup")

    if _looks_like_debug_run(run.name) and total_group_line_count == 0 and disk_bytes >= min_candidate_bytes:
        candidates.append(
            _candidate(
                run=run,
                action="review_delete_whole_debug_run_after_metadata_backup",
                target=run,
                target_disk_bytes=disk_bytes,
                reason="debug/sanity run without non-empty GRPO groups",
            )
        )

    return {
        "path": str(run),
        "name": run.name,
        "disk_bytes": disk_bytes,
        "disk_gb": disk_bytes / GiB,
        "server_vis_bytes": server_vis_bytes,
        "server_vis_gb": server_vis_bytes / GiB,
        "groups_bytes": groups_bytes,
        "attempts_bytes": attempts_bytes,
        "grpo_group_line_count": group_line_count,
        "grpo_group_total_line_count": total_group_line_count,
        "grpo_group_files": [str(path) for path in group_files],
        "grpo_summary": summary,
        "successful_attempt_count": successful_attempt_count,
        "failed_attempt_count": failed_attempt_count,
        "protection_reasons": protection_reasons,
        "notes": notes,
        "cleanup_candidates": candidates,
    }


def _candidate(*, run: Path, action: str, target: Path, target_disk_bytes: int, reason: str) -> dict:
    return {
        "run_path": str(run),
        "run_name": run.name,
        "action": action,
        "target_path": str(target),
        "target_disk_bytes": target_disk_bytes,
        "target_gb": target_disk_bytes / GiB,
        "reason": reason,
        "guardrails": [
            "check qstat before cleanup",
            "preserve groups/, attempts/, and relevant logs first",
            "do not delete if any active subset or training dataset references this run",
        ],
    }


def _group_jsonl_files(groups_dir: Path) -> list[Path]:
    if not groups_dir.exists():
        return []
    return sorted(path for path in groups_dir.glob("grpo_groups*.jsonl") if path.is_file())


def _dir_disk_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        return _path_disk_bytes(path)
    total = _path_disk_bytes(path)
    for root, dirs, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in dirs:
            total += _path_disk_bytes(root_path / name)
        for name in files:
            total += _path_disk_bytes(root_path / name)
    return total


def _path_disk_bytes(path: Path) -> int:
    try:
        stat = path.lstat()
    except FileNotFoundError:
        return 0
    blocks = getattr(stat, "st_blocks", None)
    if blocks is not None:
        return int(blocks) * 512
    return int(stat.st_size)


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
    except OSError:
        return 0
    return count


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _is_known_curated_run_name(name: str) -> bool:
    protected_tokens = (
        "grpo_core_no_mw",
        "grpo_core_hard_medium",
        "grpo_secondary_medium",
        "grpo_scale_tasks_a",
        "grpo_scale_tasks_b",
        "grpo_open_microwave",
    )
    return any(token in name for token in protected_tokens)


def _looks_like_debug_run(name: str) -> bool:
    return any(token in name.lower() for token in ("debug", "sanity", "smoke"))


def _format_gb(num_bytes: int | float) -> str:
    return f"{float(num_bytes) / GiB:.3f} GiB"


def _md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan non-destructive WAM-RL Myriad storage cleanup candidates.")
    parser.add_argument("paths", nargs="+", type=Path, help="Grouped rollout run directories or their parent root.")
    parser.add_argument("--min-candidate-gb", type=float, default=1.0, help="Minimum target size to report.")
    parser.add_argument("--large-run-gb", type=float, default=10.0, help="Size threshold for large protected run notes.")
    parser.add_argument("--out-json", type=Path, help="Optional JSON report path.")
    parser.add_argument("--out-markdown", type=Path, help="Optional Markdown report path.")
    parser.add_argument("--print-summary", action="store_true", help="Print compact JSON summary instead of full JSON.")
    args = parser.parse_args()

    report = plan_myriad_storage_cleanup(
        args.paths,
        min_candidate_gb=args.min_candidate_gb,
        large_run_gb=args.large_run_gb,
    )

    if args.out_json is not None:
        args.out_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.out_json.expanduser().write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.out_markdown is not None:
        args.out_markdown.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.out_markdown.expanduser().write_text(write_markdown_report(report), encoding="utf-8")

    output = compact_cleanup_summary(report) if args.print_summary else report
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
