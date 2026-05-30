#!/usr/bin/env python3
"""Inspect and compare real actor replay checkpoint trainable tensors."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch


def load_trainable_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    path = checkpoint_path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"missing actor replay checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("trainable_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"checkpoint has no trainable_state_dict: {path}")

    tensors: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            tensors[str(key)] = value.detach().cpu()
    if not tensors:
        raise ValueError(f"checkpoint trainable_state_dict contains no tensors: {path}")
    return tensors


def summarize_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    state_dict = load_trainable_state_dict(checkpoint_path)
    total_sq = 0.0
    max_abs = 0.0
    dtype_counts: Counter[str] = Counter()
    prefix_counts: Counter[str] = Counter()
    param_count = 0
    total_bytes = 0

    for key, tensor in state_dict.items():
        data = tensor.float()
        total_sq += float(data.pow(2).sum().item())
        max_abs = max(max_abs, float(data.abs().max().item()) if data.numel() else 0.0)
        dtype_counts[str(tensor.dtype).replace("torch.", "")] += 1
        prefix_counts[key.split(".", 1)[0]] += 1
        param_count += int(tensor.numel())
        total_bytes += int(tensor.numel() * tensor.element_size())

    return {
        "checkpoint_path": str(checkpoint_path.expanduser()),
        "tensor_count": len(state_dict),
        "param_count": param_count,
        "total_bytes": total_bytes,
        "total_l2_norm": math.sqrt(total_sq),
        "max_abs": max_abs,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "top_key_prefix_counts": dict(prefix_counts.most_common(20)),
        "first_keys": sorted(state_dict.keys())[:20],
    }


def compare_checkpoints(candidate_path: Path, reference_path: Path) -> dict[str, Any]:
    candidate = load_trainable_state_dict(candidate_path)
    reference = load_trainable_state_dict(reference_path)

    candidate_keys = set(candidate)
    reference_keys = set(reference)
    shared_keys = sorted(candidate_keys & reference_keys)
    missing_keys = sorted(reference_keys - candidate_keys)
    extra_keys = sorted(candidate_keys - reference_keys)

    delta_sq = 0.0
    reference_sq = 0.0
    candidate_sq = 0.0
    delta_max_abs = 0.0
    changed_tensor_count = 0
    compared_param_count = 0
    shape_mismatches: list[dict[str, Any]] = []

    for key in shared_keys:
        candidate_tensor = candidate[key]
        reference_tensor = reference[key]
        if tuple(candidate_tensor.shape) != tuple(reference_tensor.shape):
            shape_mismatches.append(
                {
                    "key": key,
                    "candidate_shape": list(candidate_tensor.shape),
                    "reference_shape": list(reference_tensor.shape),
                }
            )
            continue
        candidate_float = candidate_tensor.float()
        reference_float = reference_tensor.float()
        delta = candidate_float - reference_float
        delta_sq += float(delta.pow(2).sum().item())
        reference_sq += float(reference_float.pow(2).sum().item())
        candidate_sq += float(candidate_float.pow(2).sum().item())
        current_delta_max = float(delta.abs().max().item()) if delta.numel() else 0.0
        delta_max_abs = max(delta_max_abs, current_delta_max)
        if current_delta_max > 0.0:
            changed_tensor_count += 1
        compared_param_count += int(delta.numel())

    delta_l2 = math.sqrt(delta_sq)
    reference_l2 = math.sqrt(reference_sq)
    return {
        "candidate_path": str(candidate_path.expanduser()),
        "reference_path": str(reference_path.expanduser()),
        "shared_key_count": len(shared_keys),
        "missing_in_candidate_count": len(missing_keys),
        "extra_in_candidate_count": len(extra_keys),
        "shape_mismatch_count": len(shape_mismatches),
        "compared_param_count": compared_param_count,
        "delta_l2_norm": delta_l2,
        "reference_l2_norm": reference_l2,
        "candidate_l2_norm": math.sqrt(candidate_sq),
        "relative_delta_l2": delta_l2 / reference_l2 if reference_l2 > 0.0 else None,
        "delta_max_abs": delta_max_abs,
        "changed_tensor_count": changed_tensor_count,
        "missing_in_candidate_first_keys": missing_keys[:20],
        "extra_in_candidate_first_keys": extra_keys[:20],
        "shape_mismatches_first": shape_mismatches[:20],
    }


def write_markdown_report(report: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Actor Replay Checkpoint Inspection",
        "",
        "## Checkpoints",
        "",
        "| checkpoint | tensors | params | bytes | l2_norm | max_abs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["checkpoints"]:
        lines.append(
            "| {checkpoint_path} | {tensor_count} | {param_count} | {total_bytes} | {total_l2_norm:.6g} | {max_abs:.6g} |".format(
                **item
            )
        )

    comparisons = report.get("comparisons", [])
    if comparisons:
        lines.extend(
            [
                "",
                "## Comparisons",
                "",
                "| candidate | reference | shared_keys | params | delta_l2 | rel_delta_l2 | delta_max | changed_tensors | missing | extra | shape_mismatch |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in comparisons:
            rel = item["relative_delta_l2"]
            rel_text = "" if rel is None else f"{rel:.6g}"
            lines.append(
                "| {candidate_path} | {reference_path} | {shared_key_count} | {compared_param_count} | {delta_l2_norm:.6g} | "
                f"{rel_text} | "
                "{delta_max_abs:.6g} | {changed_tensor_count} | {missing_in_candidate_count} | "
                "{extra_in_candidate_count} | {shape_mismatch_count} |".format(**item)
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(checkpoints: list[Path], reference: Path | None = None) -> dict[str, Any]:
    report = {
        "checkpoints": [summarize_checkpoint(path) for path in checkpoints],
        "comparisons": [],
    }
    if reference is not None:
        report["comparisons"] = [compare_checkpoints(path, reference) for path in checkpoints]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect real actor replay checkpoint tensor statistics without loading the full LingBot model."
    )
    parser.add_argument("checkpoints", nargs="+", type=Path, help="Actor replay checkpoint.pt files to inspect.")
    parser.add_argument("--reference", type=Path, default=None, help="Optional reference checkpoint for tensor deltas.")
    parser.add_argument("--out-json", type=Path, default=None, help="Write full report JSON.")
    parser.add_argument("--out-markdown", type=Path, default=None, help="Write Markdown report.")
    args = parser.parse_args()

    report = build_report(args.checkpoints, reference=args.reference)
    print(json.dumps(report, indent=2))
    if args.out_json is not None:
        args.out_json.expanduser().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.out_markdown is not None:
        write_markdown_report(report, args.out_markdown.expanduser())


if __name__ == "__main__":
    main()
