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

from wan_va.rl.dataset import read_grpo_group_dicts


def audit_grpo_artifact_storage(
    groups_jsonl: Path,
    *,
    materialize_manifest: Path | None = None,
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

    if materialize_manifest is not None:
        manifest = json.loads(materialize_manifest.expanduser().read_text(encoding="utf-8"))
        replay_context_mapping = manifest.get("replay_context_mapping", {}) or {}
        report["materialize_manifest"] = str(materialize_manifest.expanduser())
        report["materialize_link_mode"] = manifest.get("link_mode")
        report["manifest_unique_replay_context_count"] = len(replay_context_mapping)
        report["materialized_replay_contexts"] = _path_summary(str(path) for path in replay_context_mapping.values())
        report["source_replay_contexts"] = _path_summary(str(path) for path in replay_context_mapping.keys())

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
    if "materialized_replay_contexts" in report:
        summaries.append(report["materialized_replay_contexts"])
    return any(int(summary["missing_count"]) > 0 for summary in summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit filesystem footprint of GRPO artifact references.")
    parser.add_argument("groups_jsonl", type=Path, help="Input GRPO groups JSONL.")
    parser.add_argument("--materialize-manifest", type=Path, help="Optional materialization manifest JSON.")
    parser.add_argument("--out-json", type=Path, help="Optional output JSON report.")
    parser.add_argument("--fail-on-missing", action="store_true", help="Exit non-zero if referenced files are missing.")
    args = parser.parse_args()

    report = audit_grpo_artifact_storage(args.groups_jsonl, materialize_manifest=args.materialize_manifest)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out_json is not None:
        args.out_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.out_json.expanduser().write_text(text, encoding="utf-8")
    print(text, end="")
    if args.fail_on_missing and _has_missing(report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
