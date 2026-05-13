"""Manifest helpers for offline GRPO rollout datasets."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .trajectory_schema import GrpoGroupBuildResult
from .validation import ValidationReport


def build_grpo_manifest(
    *,
    roots: Iterable[Path],
    records: Iterable[object],
    group_result: GrpoGroupBuildResult,
    validation_report: ValidationReport,
    groups_jsonl: Path,
    summary_json: Path,
) -> dict:
    record_items = list(records)
    task_counts: dict[str, int] = {}
    for record in record_items:
        task = str(getattr(record, "task", ""))
        task_counts[task] = task_counts.get(task, 0) + 1

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "roots": [str(Path(root).expanduser()) for root in roots],
        "record_count": len(record_items),
        "task_counts": dict(sorted(task_counts.items())),
        "group_summary": group_result.summary.to_dict(),
        "validation": validation_report.to_dict(),
        "outputs": {
            "groups_jsonl": str(groups_jsonl),
            "summary_json": str(summary_json),
        },
    }


def write_grpo_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
