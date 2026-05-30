#!/usr/bin/env python3
"""Audit filesystem footprint of GRPO artifact references."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

try:
    from tools._repo_root import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _repo_root import ensure_repo_root_on_path

ensure_repo_root_on_path()

from wan_va.rl.dataset import REPLAY_CONTEXT_PATH_KEY, read_grpo_group_dicts


def audit_grpo_artifact_storage(
    groups_jsonl: Path,
    *,
    materialize_manifest: Path | None = None,
    inspect_replay_contexts: bool = False,
    include_replay_context_mapping: bool = True,
) -> dict:
    groups = list(read_grpo_group_dicts(groups_jsonl.expanduser()))
    artifact_paths = _artifact_paths(groups)
    report = {
        "groups_jsonl": str(groups_jsonl.expanduser()),
        "group_count": len(groups),
        "sample_count": _sample_count(groups),
        "artifact_ref_count": len(artifact_paths),
        "unique_artifact_count": len(set(artifact_paths)),
        "tasks": _task_summary(groups),
        "artifacts": _path_summary(artifact_paths),
    }

    if inspect_replay_contexts:
        replay_context_mapping, replay_context_errors = _inspect_replay_contexts(artifact_paths)
        replay_context_paths = list(replay_context_mapping.values())
        report.update(
            {
                "replay_context_ref_count": len(replay_context_paths),
                "unique_replay_context_count": len(set(replay_context_paths)),
                "replay_context_error_count": len(replay_context_errors),
                "replay_context_errors": replay_context_errors,
                "replay_contexts": _path_summary(replay_context_paths),
                "artifacts_plus_replay_contexts": _path_summary([*artifact_paths, *replay_context_paths]),
            }
        )
        if include_replay_context_mapping:
            report["replay_context_mapping"] = replay_context_mapping

    if materialize_manifest is not None:
        manifest = json.loads(materialize_manifest.expanduser().read_text(encoding="utf-8"))
        replay_context_mapping = manifest.get("replay_context_mapping", {}) or {}
        report["materialize_manifest"] = str(materialize_manifest.expanduser())
        report["materialize_link_mode"] = manifest.get("link_mode")
        report["manifest_unique_replay_context_count"] = len(replay_context_mapping)
        report["materialized_replay_contexts"] = _path_summary(str(path) for path in replay_context_mapping.values())
        report["source_replay_contexts"] = _path_summary(str(path) for path in replay_context_mapping.keys())
        report["artifacts_plus_materialized_replay_contexts"] = _path_summary(
            [*artifact_paths, *(str(path) for path in replay_context_mapping.values())]
        )

    return report


def _artifact_paths(groups: list[dict]) -> list[str]:
    paths: list[str] = []
    for group in groups:
        for sample in group.get("samples", []) or []:
            paths.extend(str(path) for path in sample.get("strict_grpo_artifact_paths", []) or [])
    return paths


def _sample_count(groups: list[dict]) -> int:
    return sum(len(group.get("samples", []) or []) for group in groups)


def _task_summary(groups: list[dict]) -> list[dict]:
    by_task: dict[str, dict] = {}
    for group in groups:
        task = str(group.get("task", ""))
        item = by_task.setdefault(
            task,
            {
                "task": task,
                "group_count": 0,
                "sample_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "artifact_ref_count": 0,
            },
        )
        item["group_count"] += 1
        samples = group.get("samples", []) or []
        item["sample_count"] += len(samples)
        for sample in samples:
            if float(sample.get("reward", 0.0)) > 0.0:
                item["success_count"] += 1
            else:
                item["failure_count"] += 1
            item["artifact_ref_count"] += len(sample.get("strict_grpo_artifact_paths", []) or [])
    return [by_task[key] for key in sorted(by_task)]


def _path_summary(paths: Iterable[str]) -> dict:
    unique_paths = sorted(set(str(path) for path in paths))
    items = [_stat_path(Path(value).expanduser()) for value in unique_paths]
    existing = [item for item in items if item["exists"]]
    return {
        "unique_count": len(unique_paths),
        "existing_count": len(existing),
        "missing_count": len(unique_paths) - len(existing),
        "symlink_count": sum(1 for item in existing if item["is_symlink"]),
        "broken_symlink_count": sum(1 for item in items if item["is_symlink"] and not item["exists"]),
        "regular_file_count": sum(1 for item in existing if item["is_file"] and not item["is_symlink"]),
        "apparent_bytes": sum(int(item["lstat_size"] or 0) for item in existing),
        "resolved_bytes": sum(int(item["stat_size"] or 0) for item in existing),
        "missing_paths": [item["path"] for item in items if not item["exists"]],
    }


def _inspect_replay_contexts(artifact_paths: Iterable[str]) -> tuple[dict[str, str], list[dict]]:
    import torch

    mapping: dict[str, str] = {}
    errors: list[dict] = []
    for value in sorted(set(str(path) for path in artifact_paths)):
        artifact_path = Path(value).expanduser()
        if not artifact_path.exists():
            continue
        try:
            artifact = torch.load(artifact_path, map_location="meta")
            if not isinstance(artifact, dict):
                raise ValueError("strict artifact is not a dict")
            replay_context_value = artifact.get(REPLAY_CONTEXT_PATH_KEY)
            if replay_context_value is None:
                continue
            if not isinstance(replay_context_value, str) or not replay_context_value:
                raise ValueError(f"invalid {REPLAY_CONTEXT_PATH_KEY!r}: {replay_context_value!r}")
            replay_context_path = Path(replay_context_value).expanduser()
            if not replay_context_path.is_absolute():
                replay_context_path = artifact_path.parent / replay_context_path
            mapping[str(artifact_path)] = str(replay_context_path)
        except Exception as exc:  # noqa: BLE001 - audit report should collect all artifact issues.
            errors.append({"artifact_path": str(artifact_path), "error": str(exc)})
    return mapping, errors


def _stat_path(path: Path) -> dict:
    item = {
        "path": str(path),
        "exists": False,
        "is_symlink": False,
        "is_file": False,
        "lstat_size": None,
        "stat_size": None,
    }
    try:
        lstat = path.lstat()
    except FileNotFoundError:
        return item
    item["exists"] = path.exists()
    item["is_symlink"] = path.is_symlink()
    item["lstat_size"] = lstat.st_size
    if not item["exists"]:
        return item
    stat = path.stat()
    item["is_file"] = path.is_file()
    item["stat_size"] = stat.st_size
    return item


def _has_missing(report: dict) -> bool:
    summaries = [report["artifacts"]]
    if "replay_contexts" in report:
        summaries.append(report["replay_contexts"])
    if "materialized_replay_contexts" in report:
        summaries.append(report["materialized_replay_contexts"])
    return any(int(summary["missing_count"]) > 0 for summary in summaries)


def _budget_summary(report: dict) -> dict:
    if "artifacts_plus_replay_contexts" in report:
        return report["artifacts_plus_replay_contexts"]
    if "artifacts_plus_materialized_replay_contexts" in report:
        return report["artifacts_plus_materialized_replay_contexts"]
    return report["artifacts"]


def compact_storage_summary(report: dict) -> dict:
    summary = {
        "groups_jsonl": report["groups_jsonl"],
        "group_count": report["group_count"],
        "sample_count": report["sample_count"],
        "artifact_ref_count": report["artifact_ref_count"],
        "unique_artifact_count": report["unique_artifact_count"],
        "artifact_resolved_gb": report["artifacts"]["resolved_bytes"] / 1024**3,
        "artifact_missing_count": report["artifacts"]["missing_count"],
    }
    if "replay_contexts" in report:
        summary.update(
            {
                "replay_context_ref_count": report["replay_context_ref_count"],
                "unique_replay_context_count": report["unique_replay_context_count"],
                "replay_context_resolved_gb": report["replay_contexts"]["resolved_bytes"] / 1024**3,
                "combined_resolved_gb": report["artifacts_plus_replay_contexts"]["resolved_bytes"] / 1024**3,
                "replay_context_missing_count": report["replay_contexts"]["missing_count"],
                "replay_context_error_count": report["replay_context_error_count"],
            }
        )
    if "materialized_replay_contexts" in report:
        summary.update(
            {
                "materialize_link_mode": report.get("materialize_link_mode"),
                "manifest_unique_replay_context_count": report["manifest_unique_replay_context_count"],
                "materialized_replay_context_resolved_gb": report["materialized_replay_contexts"][
                    "resolved_bytes"
                ]
                / 1024**3,
                "materialized_replay_context_missing_count": report["materialized_replay_contexts"][
                    "missing_count"
                ],
                "source_replay_context_resolved_gb": report["source_replay_contexts"]["resolved_bytes"] / 1024**3,
                "source_replay_context_missing_count": report["source_replay_contexts"]["missing_count"],
                "combined_materialized_resolved_gb": report["artifacts_plus_materialized_replay_contexts"][
                    "resolved_bytes"
                ]
                / 1024**3,
            }
        )
    if "storage_budget" in report:
        summary["storage_budget"] = report["storage_budget"]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit filesystem footprint of GRPO artifact references.")
    parser.add_argument("groups_jsonl", type=Path, help="Input GRPO groups JSONL.")
    parser.add_argument("--materialize-manifest", type=Path, help="Optional materialization manifest JSON.")
    parser.add_argument(
        "--inspect-replay-contexts",
        action="store_true",
        help="Load strict artifacts and include replay_context_path file footprint in the report.",
    )
    parser.add_argument(
        "--omit-replay-context-mapping",
        action="store_true",
        help="Do not include the per-artifact replay-context mapping in JSON output.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print compact summary JSON to stdout instead of the full report.",
    )
    parser.add_argument("--out-json", type=Path, help="Optional output JSON report.")
    parser.add_argument("--fail-on-missing", action="store_true", help="Exit non-zero if referenced files are missing.")
    parser.add_argument(
        "--max-resolved-gb",
        type=float,
        help="Exit non-zero if resolved artifact footprint exceeds this many GB. "
        "With --inspect-replay-contexts, this includes replay-context files.",
    )
    args = parser.parse_args()

    report = audit_grpo_artifact_storage(
        args.groups_jsonl,
        materialize_manifest=args.materialize_manifest,
        inspect_replay_contexts=args.inspect_replay_contexts,
        include_replay_context_mapping=not args.omit_replay_context_mapping,
    )
    if args.max_resolved_gb is not None:
        max_resolved_bytes = int(args.max_resolved_gb * 1024**3)
        budget_summary = _budget_summary(report)
        report["storage_budget"] = {
            "max_resolved_gb": args.max_resolved_gb,
            "max_resolved_bytes": max_resolved_bytes,
            "resolved_bytes": budget_summary["resolved_bytes"],
            "ok": int(budget_summary["resolved_bytes"]) <= max_resolved_bytes,
        }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out_json is not None:
        args.out_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.out_json.expanduser().write_text(text, encoding="utf-8")
    if args.print_summary:
        print(json.dumps(compact_storage_summary(report), ensure_ascii=False, indent=2))
    else:
        print(text, end="")
    if args.fail_on_missing and _has_missing(report):
        raise SystemExit(1)
    if "storage_budget" in report and not report["storage_budget"]["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
