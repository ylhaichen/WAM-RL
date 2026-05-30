#!/usr/bin/env python3
"""Merge GRPO group JSONL files with duplicate group-id checks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

try:
    from tools._repo_root import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _repo_root import ensure_repo_root_on_path

ensure_repo_root_on_path()

from wan_va.rl.dataset import read_grpo_group_dicts


def merge_grpo_group_files(
    paths: list[Path],
    *,
    allow_duplicate_group_ids: bool = False,
) -> tuple[list[dict], dict]:
    groups: list[dict] = []
    source_counts = []
    for path in paths:
        expanded = path.expanduser()
        source_groups = list(read_grpo_group_dicts(expanded))
        groups.extend(source_groups)
        source_counts.append({"source_file": str(expanded), "group_count": len(source_groups)})

    group_ids = [str(group.get("group_id", "")) for group in groups]
    duplicates = sorted(group_id for group_id, count in Counter(group_ids).items() if count > 1)
    if duplicates and not allow_duplicate_group_ids:
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"duplicate group_id values: {preview}")

    manifest = {
        "source_files": [str(path.expanduser()) for path in paths],
        "source_counts": source_counts,
        "group_count": len(groups),
        "sample_count": sum(len(group.get("samples", []) or []) for group in groups),
        "duplicate_group_id_count": len(duplicates),
        "duplicate_group_ids": duplicates,
        "allow_duplicate_group_ids": allow_duplicate_group_ids,
    }
    return groups, manifest


def write_outputs(groups: list[dict], manifest: dict, *, out_jsonl: Path, out_manifest: Path | None) -> None:
    expanded_jsonl = out_jsonl.expanduser()
    expanded_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with expanded_jsonl.open("w", encoding="utf-8") as f:
        for group in groups:
            f.write(json.dumps(group, ensure_ascii=False) + "\n")

    if out_manifest is not None:
        expanded_manifest = out_manifest.expanduser()
        expanded_manifest.parent.mkdir(parents=True, exist_ok=True)
        payload = {**manifest, "output_jsonl": str(expanded_jsonl)}
        expanded_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge GRPO group JSONL files with duplicate group-id checks.")
    parser.add_argument("groups_jsonl", nargs="+", type=Path, help="Input GRPO groups JSONL files.")
    parser.add_argument("--out-jsonl", required=True, type=Path, help="Output merged groups JSONL.")
    parser.add_argument("--out-manifest", type=Path, help="Optional merge manifest JSON.")
    parser.add_argument(
        "--allow-duplicate-group-ids",
        action="store_true",
        help="Allow duplicate group_id values instead of failing.",
    )
    args = parser.parse_args()

    try:
        groups, manifest = merge_grpo_group_files(
            args.groups_jsonl,
            allow_duplicate_group_ids=args.allow_duplicate_group_ids,
        )
    except ValueError as exc:
        parser.error(str(exc))
    write_outputs(groups, manifest, out_jsonl=args.out_jsonl, out_manifest=args.out_manifest)
    print(json.dumps({**manifest, "output_jsonl": str(args.out_jsonl.expanduser())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
