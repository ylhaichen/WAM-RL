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
        "scalar_fields": _scalar_fields(payload),
        "tensor_count": len(records),
        "tensor_bytes": tensor_bytes,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "dtype_bytes": dict(sorted(dtype_bytes.items())),
        "top_level_tensor_bytes": dict(sorted(top_level_bytes.items(), key=lambda item: item[1], reverse=True)),
        "transformer_cache_summary": _transformer_cache_summary(payload, records, tensor_bytes),
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
        "## Scalar Fields",
        "",
        "| key | value |",
        "|---|---|",
    ]
    for key, value in report["scalar_fields"].items():
        lines.append(f"| {key} | `{value}` |")
    cache_summary = report["transformer_cache_summary"]
    if cache_summary:
        branch_estimate = cache_summary.get("conditional_branch_estimate", {})
        lines.extend(
            [
                "",
                "## Transformer Cache Summary",
                "",
                f"- block count: `{cache_summary.get('block_count')}`",
                f"- kv tensor bytes: `{cache_summary.get('kv_tensor_bytes')}`",
                f"- kv batch sizes: `{cache_summary.get('kv_batch_sizes')}`",
                f"- conditional-only estimated tensor bytes: "
                f"`{branch_estimate.get('estimated_tensor_bytes')}`",
                f"- conditional-only estimated savings bytes: "
                f"`{branch_estimate.get('estimated_savings_bytes')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Top-Level Tensor Bytes",
            "",
            "| key | bytes | GiB |",
            "|---|---:|---:|",
        ]
    )
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


def compact_replay_context_summary(report: dict[str, Any]) -> dict[str, Any]:
    cache_summary = report.get("transformer_cache_summary") or {}
    branch_estimate = cache_summary.get("conditional_branch_estimate") or {}
    return {
        "path": report["path"],
        "metadata_only": report["metadata_only"],
        "schema_version": report["schema_version"],
        "file_gib": _gib(report["file_bytes"]),
        "tensor_gib": _gib(report["tensor_bytes"]),
        "scalar_fields": report["scalar_fields"],
        "top_level_tensor_gib": {
            key: _gib(value) for key, value in report["top_level_tensor_bytes"].items()
        },
        "transformer_cache": {
            "block_count": cache_summary.get("block_count"),
            "kv_tensor_count": cache_summary.get("kv_tensor_count"),
            "kv_gib": _gib(cache_summary.get("kv_tensor_bytes", 0)),
            "kv_batch_sizes": cache_summary.get("kv_batch_sizes", []),
            "conditional_branch_estimate": {
                "eligible_kv_tensor_count": branch_estimate.get("eligible_kv_tensor_count", 0),
                "estimated_tensor_gib": _gib(branch_estimate.get("estimated_tensor_bytes", 0)),
                "estimated_savings_gib": _gib(branch_estimate.get("estimated_savings_bytes", 0)),
            },
        },
    }


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


def _gib(value: int | float) -> float:
    return float(value) / 1024**3


def _scalar_fields(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    keys = (
        "schema_version",
        "cache_name",
        "use_cfg",
        "cfg_pruned_to_conditional",
        "action_guidance_scale",
        "action_num_inference_steps",
        "frame_chunk_size",
    )
    fields = {}
    for key in keys:
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, (str, int, float, bool)) or value is None:
            fields[key] = value
    return fields


def _transformer_cache_summary(payload: Any, records: list[TensorRecord], tensor_bytes: int) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("transformer_cache"), list):
        return {}

    cache_records = [record for record in records if record.path.startswith("transformer_cache.")]
    kv_records = [record for record in cache_records if _cache_record_key(record.path) in {"k", "v"}]
    kv_bytes = sum(record.bytes for record in kv_records)
    eligible_savings = 0
    eligible_count = 0
    batch_sizes = set()
    shape_counts: defaultdict[tuple[int, ...], int] = defaultdict(int)
    shape_bytes: defaultdict[tuple[int, ...], int] = defaultdict(int)

    for record in kv_records:
        shape = tuple(record.shape)
        shape_counts[shape] += 1
        shape_bytes[shape] += record.bytes
        if record.shape:
            batch_sizes.add(record.shape[0])
        if record.shape and record.shape[0] > 1:
            eligible_count += 1
            eligible_savings += record.bytes - (record.bytes // record.shape[0])

    estimated_tensor_bytes = tensor_bytes - eligible_savings
    return {
        "block_count": len(payload["transformer_cache"]),
        "cache_tensor_count": len(cache_records),
        "kv_tensor_count": len(kv_records),
        "kv_tensor_bytes": kv_bytes,
        "kv_batch_sizes": sorted(batch_sizes),
        "kv_shape_counts": [
            {"shape": list(shape), "count": shape_counts[shape], "bytes": shape_bytes[shape]}
            for shape in sorted(shape_counts, key=lambda item: shape_bytes[item], reverse=True)
        ],
        "conditional_branch_estimate": {
            "eligible_kv_tensor_count": eligible_count,
            "current_tensor_bytes": tensor_bytes,
            "estimated_tensor_bytes": estimated_tensor_bytes,
            "estimated_savings_bytes": eligible_savings,
            "estimated_savings_gib": eligible_savings / 1024**3,
        },
    }


def _cache_record_key(path: str) -> str:
    parts = path.split(".")
    return parts[2] if len(parts) >= 3 and parts[0] == "transformer_cache" else ""


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
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print compact summary JSON to stdout instead of the full report.",
    )
    args = parser.parse_args()

    report = inspect_replay_context(args.path, top_k=args.top_k, metadata_only=args.metadata_only)
    text = json.dumps(report, indent=2) + "\n"
    if args.out_json:
        args.out_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.out_json.expanduser().write_text(text, encoding="utf-8")
    if args.out_markdown:
        write_markdown_report(args.out_markdown, report)
    if args.print_summary:
        print(json.dumps(compact_replay_context_summary(report), indent=2))
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
