#!/usr/bin/env python3
"""Inspect tensor storage inside a strict GRPO replay-context artifact."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class TensorRecord:
    path: str
    dtype: str
    device: str
    shape: list[int]
    numel: int
    bytes: int


def inspect_replay_context(path: Path, *, top_k: int = 20, metadata_only: bool = False) -> dict[str, Any]:
    expanded = path.expanduser()
    if not expanded.exists():
        raise FileNotFoundError(f"missing replay context artifact: {expanded}")

    map_location = "meta" if metadata_only else "cpu"
    payload = torch.load(expanded, map_location=map_location)
    records = list(_iter_tensors(payload))
    dtype_counts: Counter[str] = Counter()
    dtype_bytes: Counter[str] = Counter()
    top_level_bytes: defaultdict[str, int] = defaultdict(int)
    tensor_bytes = 0

    for record in records:
        dtype_counts[record.dtype] += 1
        dtype_bytes[record.dtype] += record.bytes
        top_level_bytes[_top_level(record.path)] += record.bytes
        tensor_bytes += record.bytes

    top_tensors = sorted(records, key=lambda item: item.bytes, reverse=True)[:top_k]
    return {
        "path": str(expanded),
        "file_bytes": expanded.stat().st_size,
        "metadata_only": metadata_only,
        "schema_version": payload.get("schema_version") if isinstance(payload, dict) else None,
        "top_level_keys": sorted(str(key) for key in payload.keys()) if isinstance(payload, dict) else [],
        "tensor_count": len(records),
        "tensor_bytes": tensor_bytes,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "dtype_bytes": dict(sorted(dtype_bytes.items())),
        "top_level_tensor_bytes": dict(sorted(top_level_bytes.items(), key=lambda item: item[1], reverse=True)),
        "top_tensors": [_record_to_dict(record) for record in top_tensors],
    }


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# GRPO Replay Context Inspection",
        "",
        f"- path: `{report['path']}`",
        f"- file bytes: `{report['file_bytes']}`",
        f"- tensor bytes: `{report['tensor_bytes']}`",
        f"- tensor count: `{report['tensor_count']}`",
        f"- metadata only: `{report['metadata_only']}`",
        "",
        "## Top-Level Tensor Bytes",
        "",
        "| key | bytes | GiB |",
        "|---|---:|---:|",
    ]
    for key, value in report["top_level_tensor_bytes"].items():
        lines.append(f"| {key} | {value} | {value / 1024**3:.6g} |")
    lines.extend(
        [
            "",
            "## Top Tensors",
            "",
            "| path | dtype | device | shape | bytes | GiB |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for item in report["top_tensors"]:
        lines.append(
            "| {path} | {dtype} | {device} | {shape} | {bytes} | {gib:.6g} |".format(
                path=item["path"],
                dtype=item["dtype"],
                device=item["device"],
                shape=item["shape"],
                bytes=item["bytes"],
                gib=item["bytes"] / 1024**3,
            )
        )
    lines.append("")
    path.expanduser().parent.mkdir(parents=True, exist_ok=True)
    path.expanduser().write_text("\n".join(lines), encoding="utf-8")


def _iter_tensors(value: Any, prefix: str = "") -> Iterable[TensorRecord]:
    if torch.is_tensor(value):
        yield TensorRecord(
            path=prefix or "<root>",
            dtype=str(value.dtype).replace("torch.", ""),
            device=str(value.device),
            shape=list(value.shape),
            numel=int(value.numel()),
            bytes=int(value.numel() * value.element_size()),
        )
        return
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            child_prefix = _join_path(prefix, str(key))
            yield from _iter_tensors(value[key], child_prefix)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_tensors(item, _join_path(prefix, str(index)))


def _join_path(prefix: str, child: str) -> str:
    return child if not prefix else f"{prefix}.{child}"


def _top_level(path: str) -> str:
    return path.split(".", 1)[0] if path else "<root>"


def _record_to_dict(record: TensorRecord) -> dict[str, Any]:
    return {
        "path": record.path,
        "dtype": record.dtype,
        "device": record.device,
        "shape": record.shape,
        "numel": record.numel,
        "bytes": record.bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect tensor storage inside a strict GRPO replay-context artifact.")
    parser.add_argument("path", type=Path, help="strict_grpo_replay_context_*.pt path.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of largest tensors to include.")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Load tensors on the meta device to inspect shapes/dtypes without allocating tensor storage.",
    )
    parser.add_argument("--out-json", type=Path, help="Optional JSON output path.")
    parser.add_argument("--out-markdown", type=Path, help="Optional Markdown output path.")
    args = parser.parse_args()

    report = inspect_replay_context(args.path, top_k=args.top_k, metadata_only=args.metadata_only)
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    if args.out_json:
        args.out_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.out_json.expanduser().write_text(text, encoding="utf-8")
    if args.out_markdown:
        write_markdown_report(args.out_markdown, report)


if __name__ == "__main__":
    main()
