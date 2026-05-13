#!/usr/bin/env python3
"""Build dynamic-sampling GRPO groups from collected RoboTwin rollouts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tools.collect_robotwin_rollouts import iter_rollout_records
from wan_va.rl.group_builder import build_grpo_groups
from wan_va.rl.manifest import build_grpo_manifest, write_grpo_manifest
from wan_va.rl.trajectory_schema import GrpoGroupBuildResult
from wan_va.rl.validation import validate_rollout_records


def build_groups_from_roots(
    roots: list[Path],
    *,
    expected_group_size: int | None = None,
    require_strict_artifacts: bool = False,
    tasks: set[str] | None = None,
) -> GrpoGroupBuildResult:
    records = collect_records_from_roots(roots, tasks=tasks)
    return build_grpo_groups(
        records,
        expected_group_size=expected_group_size,
        require_strict_artifacts=require_strict_artifacts,
    )


def collect_records_from_roots(roots: list[Path], *, tasks: set[str] | None = None) -> list[object]:
    records = []
    for root in roots:
        records.extend(iter_rollout_records(root.expanduser(), tasks=tasks))
    return records


def write_group_outputs(
    result: GrpoGroupBuildResult,
    *,
    out_jsonl: Path,
    out_summary: Path,
    out_manifest: Path | None = None,
    roots: list[Path] | None = None,
    expected_group_size: int | None = None,
    require_strict_artifacts: bool = False,
    require_existing_artifacts: bool = False,
    records: list[object] | None = None,
) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for group in result.groups:
            f.write(json.dumps(group.to_dict(), ensure_ascii=False) + "\n")

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(
        json.dumps(result.summary.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if out_manifest is not None:
        if roots is None:
            raise ValueError("roots are required when writing a GRPO manifest")
        record_items = records if records is not None else collect_records_from_roots(roots)
        validation = validate_rollout_records(
            record_items,
            expected_group_size=expected_group_size,
            require_strict_artifacts=require_strict_artifacts,
            require_existing_artifacts=require_existing_artifacts,
        )
        manifest = build_grpo_manifest(
            roots=roots,
            records=record_items,
            group_result=result,
            validation_report=validation,
            groups_jsonl=out_jsonl,
            summary_json=out_summary,
        )
        write_grpo_manifest(manifest, out_manifest)


def wait_for_strict_artifacts(records: list[object], *, timeout_seconds: float, poll_interval_seconds: float = 2.0) -> list[str]:
    deadline = time.time() + max(0.0, timeout_seconds)
    missing = _missing_artifact_paths(records)
    while missing and time.time() < deadline:
        time.sleep(poll_interval_seconds)
        missing = _missing_artifact_paths(records)
    return missing


def _missing_artifact_paths(records: list[object]) -> list[str]:
    missing: list[str] = []
    for record in records:
        for path in getattr(record, "strict_grpo_artifact_paths", None) or []:
            expanded = Path(str(path)).expanduser()
            if not expanded.exists():
                missing.append(str(path))
    return sorted(set(missing))


def _task_filter(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    tasks: set[str] = set()
    for value in values:
        tasks.update(item for item in value.split() if item)
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mixed GRPO groups from RoboTwin rollout JSON.")
    parser.add_argument("roots", nargs="+", type=Path, help="RoboTwin result roots containing rollouts/.")
    parser.add_argument("--expected-group-size", type=int, help="Drop groups that do not contain exactly this many samples.")
    parser.add_argument("--require-strict-artifacts", action="store_true", help="Drop groups missing strict GRPO tensor artifacts.")
    parser.add_argument("--require-existing-artifacts", action="store_true", help="Require strict GRPO artifact paths to exist on disk.")
    parser.add_argument("--wait-for-artifacts-seconds", type=float, default=0.0, help="Poll for async strict GRPO artifact writes before validating.")
    parser.add_argument("--fail-on-validation-errors", action="store_true", help="Exit nonzero if rollout validation reports errors.")
    parser.add_argument("--tasks", nargs="*", help="Optional selected task names.")
    parser.add_argument("--out-jsonl", type=Path, help="Output grouped JSONL path.")
    parser.add_argument("--out-summary", type=Path, help="Output summary JSON path.")
    parser.add_argument("--out-manifest", type=Path, help="Output manifest JSON path.")
    args = parser.parse_args()

    roots = [root.expanduser() for root in args.roots]
    default_out_root = roots[0] / "groups"
    out_jsonl = (args.out_jsonl or default_out_root / "grpo_groups.jsonl").expanduser()
    out_summary = (args.out_summary or default_out_root / "grpo_summary.json").expanduser()
    out_manifest = (args.out_manifest or default_out_root / "grpo_manifest.json").expanduser()
    tasks = _task_filter(args.tasks)
    records = collect_records_from_roots(roots, tasks=tasks)
    if args.wait_for_artifacts_seconds > 0:
        wait_for_strict_artifacts(records, timeout_seconds=args.wait_for_artifacts_seconds)

    result = build_grpo_groups(
        records,
        expected_group_size=args.expected_group_size,
        require_strict_artifacts=args.require_strict_artifacts,
    )
    write_group_outputs(
        result,
        out_jsonl=out_jsonl,
        out_summary=out_summary,
        out_manifest=out_manifest,
        roots=roots,
        expected_group_size=args.expected_group_size,
        require_strict_artifacts=args.require_strict_artifacts,
        require_existing_artifacts=args.require_existing_artifacts,
        records=records,
    )

    summary = result.summary.to_dict()
    validation = validate_rollout_records(
        records,
        expected_group_size=args.expected_group_size,
        require_strict_artifacts=args.require_strict_artifacts,
        require_existing_artifacts=args.require_existing_artifacts,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(validation.to_dict(), ensure_ascii=False, indent=2))
    print(f"Wrote groups: {out_jsonl}")
    print(f"Wrote summary: {out_summary}")
    print(f"Wrote manifest: {out_manifest}")
    if args.fail_on_validation_errors and validation.error_count:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
