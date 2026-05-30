#!/usr/bin/env python3
"""Summarize replay-context storage and tensor metadata from GRPO groups."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from tools._repo_root import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _repo_root import ensure_repo_root_on_path

ensure_repo_root_on_path()

from tools.inspect_grpo_replay_context import compact_replay_context_summary, inspect_replay_context
from wan_va.rl.dataset import REPLAY_CONTEXT_PATH_KEY, read_grpo_group_dicts


GiB = 1024**3


def summarize_replay_contexts(
    groups_jsonl: list[Path],
    *,
    inspect_artifacts: bool = False,
    inspect_context_tensors: bool = False,
) -> dict[str, Any]:
    """Return metadata-only replay-context footprint summary for GRPO groups."""

    groups = _read_groups(groups_jsonl)
    refs, artifact_errors = _collect_context_refs(groups, inspect_artifacts=inspect_artifacts)
    by_context: dict[str, dict[str, Any]] = {}
    for ref in refs:
        item = by_context.setdefault(
            ref["path"],
            {
                "path": ref["path"],
                "tasks": set(),
                "capture_chunk_strides": set(),
                "capture_max_chunks": set(),
                "replay_context_max_gb": set(),
                "sample_refs": 0,
                "artifact_refs": 0,
            },
        )
        item["tasks"].add(ref["task"])
        _add_optional(item["capture_chunk_strides"], ref.get("capture_chunk_stride"))
        _add_optional(item["capture_max_chunks"], ref.get("capture_max_chunks"))
        _add_optional(item["replay_context_max_gb"], ref.get("replay_context_max_gb"))
        item["sample_refs"] += 1
        item["artifact_refs"] += len(ref["artifact_paths"])

    contexts: list[dict[str, Any]] = []
    context_errors: list[dict[str, str]] = []
    missing_paths: list[str] = []
    for path_value in sorted(by_context):
        path = Path(path_value).expanduser()
        if not path.exists():
            missing_paths.append(str(path))
            continue
        compact: dict[str, Any] | None = None
        file_bytes = path.stat().st_size
        tensor_bytes = 0
        if inspect_context_tensors:
            try:
                report = inspect_replay_context(path, top_k=0, metadata_only=True)
            except Exception as exc:  # noqa: BLE001 - summary should collect every bad context.
                context_errors.append({"path": str(path), "error": str(exc)})
                continue
            compact = compact_replay_context_summary(report)
            file_bytes = int(report["file_bytes"])
            tensor_bytes = int(report["tensor_bytes"])
        context_item = {
            **by_context[path_value],
            "tasks": sorted(by_context[path_value]["tasks"]),
            "capture_chunk_strides": sorted(by_context[path_value]["capture_chunk_strides"]),
            "capture_max_chunks": sorted(by_context[path_value]["capture_max_chunks"]),
            "replay_context_max_gb": sorted(by_context[path_value]["replay_context_max_gb"]),
            "file_bytes": file_bytes,
            "tensor_bytes": tensor_bytes,
            "file_gib": file_bytes / GiB,
            "tensor_gib": tensor_bytes / GiB,
            "summary": compact,
        }
        contexts.append(context_item)

    config_groups = _group_by_config(contexts)
    return {
        "source_files": [str(path.expanduser()) for path in groups_jsonl],
        "group_count": len(groups),
        "sample_count": sum(len(group.get("samples", []) or []) for group in groups),
        "context_ref_count": len(refs),
        "unique_context_count": len(by_context),
        "inspected_context_count": len(contexts),
        "missing_context_count": len(missing_paths),
        "missing_context_paths": missing_paths,
        "artifact_error_count": len(artifact_errors),
        "artifact_errors": artifact_errors,
        "context_error_count": len(context_errors),
        "context_errors": context_errors,
        "context_tensor_inspected": inspect_context_tensors,
        "file_total_bytes": sum(item["file_bytes"] for item in contexts),
        "tensor_total_bytes": sum(item["tensor_bytes"] for item in contexts),
        "file_total_gib": sum(item["file_bytes"] for item in contexts) / GiB,
        "tensor_total_gib": sum(item["tensor_bytes"] for item in contexts) / GiB,
        "conditional_branch_estimated_savings_gib": sum(
            _conditional_savings_bytes(item) for item in contexts
        )
        / GiB,
        "config_count": len(config_groups),
        "configs": config_groups,
    }


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# GRPO Replay Context Summary",
        "",
        "## Overall",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| group_count | {report['group_count']} |",
        f"| sample_count | {report['sample_count']} |",
        f"| unique_context_count | {report['unique_context_count']} |",
        f"| inspected_context_count | {report['inspected_context_count']} |",
        f"| missing_context_count | {report['missing_context_count']} |",
        f"| context_error_count | {report['context_error_count']} |",
        f"| context_tensor_inspected | {report['context_tensor_inspected']} |",
        f"| file_total_gib | {report['file_total_gib']:.3f} |",
        f"| tensor_total_gib | {report['tensor_total_gib']:.3f} |",
        f"| conditional_branch_estimated_savings_gib | "
        f"{report['conditional_branch_estimated_savings_gib']:.3f} |",
        "",
        "## Configs",
        "",
        "| contexts | file GiB | tensor GiB | cond-branch savings GiB | steps | action guidance | use cfg | pruned | frame chunk | capture stride | capture max | max GB | blocks | kv batch sizes | tasks |",
        "|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in report["configs"]:
        config = item["config"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["context_count"]),
                    f"{item['file_total_gib']:.3f}",
                    f"{item['tensor_total_gib']:.3f}",
                    f"{item['conditional_branch_estimated_savings_gib']:.3f}",
                    _md_value(config.get("action_num_inference_steps")),
                    _md_value(config.get("action_guidance_scale")),
                    _md_value(config.get("use_cfg")),
                    _md_value(config.get("cfg_pruned_to_conditional")),
                    _md_value(config.get("frame_chunk_size")),
                    _md_value(config.get("strict_grpo_capture_chunk_stride")),
                    _md_value(config.get("strict_grpo_capture_max_chunks")),
                    _md_value(config.get("strict_grpo_replay_context_max_gb")),
                    _md_value(config.get("block_count")),
                    _md_value(config.get("kv_batch_sizes")),
                    ", ".join(item["tasks"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Source Files", ""])
    for source in report["source_files"]:
        lines.append(f"- `{source}`")
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, report: dict[str, Any]) -> None:
    path.expanduser().parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "context_count",
        "file_total_gib",
        "tensor_total_gib",
        "conditional_branch_estimated_savings_gib",
        "action_num_inference_steps",
        "action_guidance_scale",
        "use_cfg",
        "cfg_pruned_to_conditional",
        "frame_chunk_size",
        "strict_grpo_capture_chunk_stride",
        "strict_grpo_capture_max_chunks",
        "strict_grpo_replay_context_max_gb",
        "block_count",
        "kv_batch_sizes",
        "tasks",
    ]
    with path.expanduser().open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in report["configs"]:
            config = item["config"]
            writer.writerow(
                {
                    "context_count": item["context_count"],
                    "file_total_gib": item["file_total_gib"],
                    "tensor_total_gib": item["tensor_total_gib"],
                    "conditional_branch_estimated_savings_gib": item[
                        "conditional_branch_estimated_savings_gib"
                    ],
                    "action_num_inference_steps": config.get("action_num_inference_steps"),
                    "action_guidance_scale": config.get("action_guidance_scale"),
                    "use_cfg": config.get("use_cfg"),
                    "cfg_pruned_to_conditional": config.get("cfg_pruned_to_conditional"),
                    "frame_chunk_size": config.get("frame_chunk_size"),
                    "strict_grpo_capture_chunk_stride": config.get("strict_grpo_capture_chunk_stride"),
                    "strict_grpo_capture_max_chunks": config.get("strict_grpo_capture_max_chunks"),
                    "strict_grpo_replay_context_max_gb": config.get("strict_grpo_replay_context_max_gb"),
                    "block_count": config.get("block_count"),
                    "kv_batch_sizes": json.dumps(config.get("kv_batch_sizes", [])),
                    "tasks": ",".join(item["tasks"]),
                }
            )


def compact_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "group_count": report["group_count"],
        "sample_count": report["sample_count"],
        "unique_context_count": report["unique_context_count"],
        "inspected_context_count": report["inspected_context_count"],
        "missing_context_count": report["missing_context_count"],
        "context_error_count": report["context_error_count"],
        "context_tensor_inspected": report["context_tensor_inspected"],
        "file_total_gib": report["file_total_gib"],
        "tensor_total_gib": report["tensor_total_gib"],
        "conditional_branch_estimated_savings_gib": report[
            "conditional_branch_estimated_savings_gib"
        ],
        "config_count": report["config_count"],
        "configs": report["configs"],
    }


def _read_groups(paths: list[Path]) -> list[dict]:
    groups: list[dict] = []
    for path in paths:
        groups.extend(read_grpo_group_dicts(path.expanduser()))
    return groups


def _collect_context_refs(groups: list[dict], *, inspect_artifacts: bool) -> tuple[list[dict], list[dict]]:
    refs: list[dict] = []
    errors: list[dict] = []
    for group in groups:
        task = str(group.get("task", ""))
        for sample in group.get("samples", []) or []:
            artifact_paths = [str(path) for path in sample.get("strict_grpo_artifact_paths", []) or []]
            context_paths = [str(path) for path in sample.get("strict_grpo_replay_context_paths", []) or []]
            if inspect_artifacts:
                artifact_contexts, artifact_errors = _resolve_context_paths_from_artifacts(artifact_paths)
                context_paths.extend(artifact_contexts)
                errors.extend(artifact_errors)
            for context_path in sorted(set(context_paths)):
                refs.append(
                    {
                        "path": context_path,
                        "task": task,
                        "sample_idx": sample.get("sample_idx"),
                        "artifact_paths": artifact_paths,
                        "capture_chunk_stride": sample.get("strict_grpo_capture_chunk_stride"),
                        "capture_max_chunks": sample.get("strict_grpo_capture_max_chunks"),
                        "replay_context_max_gb": sample.get("strict_grpo_replay_context_max_gb"),
                    }
                )
    return refs, errors


def _resolve_context_paths_from_artifacts(artifact_paths: list[str]) -> tuple[list[str], list[dict]]:
    import torch

    contexts: list[str] = []
    errors: list[dict] = []
    for path_value in sorted(set(artifact_paths)):
        artifact_path = Path(path_value).expanduser()
        if not artifact_path.exists():
            continue
        try:
            artifact = torch.load(artifact_path, map_location="meta")
            if not isinstance(artifact, dict):
                raise ValueError("strict artifact is not a dict")
            context_value = artifact.get(REPLAY_CONTEXT_PATH_KEY)
            if not isinstance(context_value, str) or not context_value:
                continue
            context_path = Path(context_value).expanduser()
            if not context_path.is_absolute():
                context_path = artifact_path.parent / context_path
            contexts.append(str(context_path))
        except Exception as exc:  # noqa: BLE001 - collect all artifact issues.
            errors.append({"artifact_path": str(artifact_path), "error": str(exc)})
    return contexts, errors


def _group_by_config(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for item in contexts:
        config = _context_config(item["summary"])
        config.update(
            {
                "strict_grpo_capture_chunk_stride": _single_or_list(item["capture_chunk_strides"]),
                "strict_grpo_capture_max_chunks": _single_or_list(item["capture_max_chunks"]),
                "strict_grpo_replay_context_max_gb": _single_or_list(item["replay_context_max_gb"]),
            }
        )
        key = json.dumps(config, sort_keys=True)
        group = by_key.setdefault(
            key,
            {
                "config": config,
                "context_count": 0,
                "file_total_bytes": 0,
                "tensor_total_bytes": 0,
                "conditional_branch_estimated_savings_bytes": 0,
                "tasks": set(),
            },
        )
        group["context_count"] += 1
        group["file_total_bytes"] += item["file_bytes"]
        group["tensor_total_bytes"] += item["tensor_bytes"]
        group["conditional_branch_estimated_savings_bytes"] += _conditional_savings_bytes(item)
        group["tasks"].update(item["tasks"])

    rows = []
    for group in by_key.values():
        rows.append(
            {
                "config": group["config"],
                "context_count": group["context_count"],
                "file_total_bytes": group["file_total_bytes"],
                "tensor_total_bytes": group["tensor_total_bytes"],
                "conditional_branch_estimated_savings_bytes": group[
                    "conditional_branch_estimated_savings_bytes"
                ],
                "file_total_gib": group["file_total_bytes"] / GiB,
                "tensor_total_gib": group["tensor_total_bytes"] / GiB,
                "conditional_branch_estimated_savings_gib": group[
                    "conditional_branch_estimated_savings_bytes"
                ]
                / GiB,
                "tasks": sorted(group["tasks"]),
            }
        )
    return sorted(rows, key=lambda value: int(value["file_total_bytes"]), reverse=True)


def _context_config(summary: dict[str, Any]) -> dict[str, Any]:
    if summary is None:
        return {
            "schema_version": None,
            "cache_name": None,
            "use_cfg": None,
            "cfg_pruned_to_conditional": None,
            "action_guidance_scale": None,
            "action_num_inference_steps": None,
            "frame_chunk_size": None,
            "block_count": None,
            "kv_batch_sizes": [],
        }
    scalar = summary.get("scalar_fields") or {}
    cache = summary.get("transformer_cache") or {}
    return {
        "schema_version": summary.get("schema_version"),
        "cache_name": scalar.get("cache_name"),
        "use_cfg": scalar.get("use_cfg"),
        "cfg_pruned_to_conditional": scalar.get("cfg_pruned_to_conditional"),
        "action_guidance_scale": scalar.get("action_guidance_scale"),
        "action_num_inference_steps": scalar.get("action_num_inference_steps"),
        "frame_chunk_size": scalar.get("frame_chunk_size"),
        "block_count": cache.get("block_count"),
        "kv_batch_sizes": cache.get("kv_batch_sizes", []),
    }


def _conditional_savings_bytes(item: dict[str, Any]) -> int:
    if not item.get("summary"):
        return 0
    estimate = (
        item.get("summary", {})
        .get("transformer_cache", {})
        .get("conditional_branch_estimate", {})
    )
    return int(float(estimate.get("estimated_savings_gib", 0.0)) * GiB)


def _add_optional(values: set, value: Any) -> None:
    if value is not None:
        values.add(value)


def _md_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def _single_or_list(values: list[Any]) -> Any:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize replay-context storage and tensor metadata from GRPO groups."
    )
    parser.add_argument("groups_jsonl", nargs="+", type=Path, help="One or more grpo_groups.jsonl files.")
    parser.add_argument(
        "--inspect-artifacts",
        action="store_true",
        help="Resolve replay_context_path from strict artifacts when group metadata lacks direct context paths.",
    )
    parser.add_argument(
        "--inspect-context-tensors",
        action="store_true",
        help="Open each replay-context .pt with torch metadata-only loading to aggregate tensor/config details. "
        "This can still do heavy filesystem IO on multi-GB contexts; leave it off for large legacy runs.",
    )
    parser.add_argument("--out-json", type=Path, help="Optional JSON output path.")
    parser.add_argument("--out-csv", type=Path, help="Optional per-config CSV output path.")
    parser.add_argument("--out-markdown", type=Path, help="Optional Markdown output path.")
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print compact summary JSON instead of the full per-context report.",
    )
    args = parser.parse_args()

    report = summarize_replay_contexts(
        [path.expanduser() for path in args.groups_jsonl],
        inspect_artifacts=args.inspect_artifacts,
        inspect_context_tensors=args.inspect_context_tensors,
    )
    payload = compact_summary(report) if args.print_summary else report
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.out_json:
        args.out_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.out_json.expanduser().write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote JSON: {args.out_json}")
    if args.out_csv:
        write_csv(args.out_csv, report)
        print(f"Wrote CSV: {args.out_csv}")
    if args.out_markdown:
        args.out_markdown.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.out_markdown.write_text(format_markdown(report), encoding="utf-8")
        print(f"Wrote Markdown: {args.out_markdown}")


if __name__ == "__main__":
    main()
