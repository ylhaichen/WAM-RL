#!/usr/bin/env python3
"""Validate trainer-facing GRPO group datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools._repo_root import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _repo_root import ensure_repo_root_on_path

ensure_repo_root_on_path()

from wan_va.rl.dataset import (
    DatasetValidationReport,
    inspect_strict_artifacts,
    read_transition_refs,
    validate_transition_refs,
)


def validate_dataset(
    groups_jsonl: Path,
    *,
    require_existing_artifacts: bool = True,
    inspect_artifacts: bool = False,
) -> DatasetValidationReport:
    refs = list(read_transition_refs(groups_jsonl.expanduser()))
    report = validate_transition_refs(refs, require_existing_artifacts=require_existing_artifacts)
    if inspect_artifacts and report.ok:
        report = inspect_strict_artifacts(refs)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate GRPO groups from the trainer/dataset perspective.")
    parser.add_argument("groups_jsonl", type=Path, help="Path to groups/grpo_groups.jsonl.")
    parser.add_argument("--allow-missing-artifacts", action="store_true", help="Do not require artifact paths to exist.")
    parser.add_argument("--inspect-artifacts", action="store_true", help="Load .pt artifacts with torch and validate schema keys.")
    parser.add_argument("--out-summary", type=Path, help="Optional output validation summary JSON.")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit nonzero if validation reports errors.")
    args = parser.parse_args()

    report = validate_dataset(
        args.groups_jsonl,
        require_existing_artifacts=not args.allow_missing_artifacts,
        inspect_artifacts=args.inspect_artifacts,
    )
    payload = report.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.out_summary:
        args.out_summary.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.out_summary.expanduser().write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.fail_on_error and report.error_count:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
