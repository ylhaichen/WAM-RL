#!/usr/bin/env python3
"""Materialize referenced GRPO artifacts into a portable subset directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path

try:
    from tools._repo_root import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _repo_root import ensure_repo_root_on_path

ensure_repo_root_on_path()

from wan_va.rl.dataset import REPLAY_CONTEXT_PATH_KEY, read_grpo_group_dicts


def materialize_grpo_artifacts(
    groups_jsonl: Path,
    *,
    out_root: Path,
    link_mode: str = "symlink",
    include_replay_context: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> tuple[list[dict], dict]:
    """Link or copy referenced artifacts and return rewritten groups.

    The output preserves each source artifact basename inside a hashed parent
    directory. This keeps relative `replay_context_path` references valid when a
    matching replay-context file is materialized next to the strict artifact.
    """

    if link_mode not in {"symlink", "copy"}:
        raise ValueError("link_mode must be 'symlink' or 'copy'")
    if include_replay_context:
        _ensure_torch_available()

    source_groups = list(read_grpo_group_dicts(groups_jsonl.expanduser()))
    expanded_out = out_root.expanduser()
    artifacts_root = expanded_out / "artifacts"
    mapping: dict[str, str] = {}
    context_mapping: dict[str, str] = {}

    for source_path in sorted(set(_artifact_paths(source_groups))):
        src = Path(source_path).expanduser()
        if not src.exists():
            raise FileNotFoundError(f"referenced artifact does not exist: {src}")
        dest_dir = artifacts_root / _stable_parent_key(src)
        dest = dest_dir / src.name
        if not dry_run:
            _materialize_file(src, dest, link_mode=link_mode, overwrite=overwrite)
        mapping[source_path] = str(dest)

        if include_replay_context:
            context_src, context_dest = _context_materialization_paths(src, dest)
            if context_src is not None and context_dest is not None:
                if not dry_run:
                    _materialize_file(context_src, context_dest, link_mode=link_mode, overwrite=overwrite)
                context_mapping[str(context_src)] = str(context_dest)

    rewritten_groups = _rewrite_artifact_paths(source_groups, mapping)
    source_artifacts = _file_size_summary(mapping.keys())
    source_replay_contexts = _file_size_summary(context_mapping.keys())
    source_artifacts_plus_replay_contexts = _file_size_summary([*mapping.keys(), *context_mapping.keys()])
    planned_copy_bytes = (
        int(source_artifacts_plus_replay_contexts["resolved_bytes"]) if link_mode == "copy" else 0
    )
    manifest = {
        "source_jsonl": str(groups_jsonl.expanduser()),
        "out_root": str(expanded_out),
        "link_mode": link_mode,
        "include_replay_context": include_replay_context,
        "dry_run": dry_run,
        "group_count": len(rewritten_groups),
        "sample_count": _sample_count(rewritten_groups),
        "artifact_ref_count": len(_artifact_paths(rewritten_groups)),
        "unique_artifact_count": len(mapping),
        "unique_replay_context_count": len(context_mapping),
        "artifact_mapping": mapping,
        "replay_context_mapping": context_mapping,
        "source_artifacts": source_artifacts,
        "source_replay_contexts": source_replay_contexts,
        "source_artifacts_plus_replay_contexts": source_artifacts_plus_replay_contexts,
        "planned_copy_bytes": planned_copy_bytes,
        "planned_copy_gb": planned_copy_bytes / 1024**3,
    }
    return rewritten_groups, manifest


def write_materialized_outputs(
    groups: list[dict],
    manifest: dict,
    *,
    out_jsonl: Path,
    out_manifest: Path,
) -> None:
    out_jsonl.expanduser().parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.expanduser().open("w", encoding="utf-8") as f:
        for group in groups:
            f.write(json.dumps(group, ensure_ascii=False) + "\n")
    out_manifest.expanduser().parent.mkdir(parents=True, exist_ok=True)
    out_manifest.expanduser().write_text(
        json.dumps({**manifest, "output_jsonl": str(out_jsonl.expanduser())}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _context_materialization_paths(source_artifact: Path, dest_artifact: Path) -> tuple[Path | None, Path | None]:
    context_value = _load_replay_context_path(source_artifact)
    if context_value is None:
        return None, None
    context_path = Path(context_value).expanduser()
    if context_path.is_absolute():
        return context_path, dest_artifact.parent / context_path.name
    return source_artifact.parent / context_path, dest_artifact.parent / context_path


def _load_replay_context_path(artifact_path: Path) -> str | None:
    import torch

    artifact = torch.load(artifact_path, map_location="meta")
    if not isinstance(artifact, dict):
        raise ValueError(f"strict artifact must be a dict: {artifact_path}")
    value = artifact.get(REPLAY_CONTEXT_PATH_KEY)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"artifact {artifact_path} has invalid {REPLAY_CONTEXT_PATH_KEY!r}: {value!r}")
    return value


def _ensure_torch_available() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "--include-replay-context requires torch to read strict artifact metadata; "
            "run this tool inside the WAM-RL container or a Python env with torch."
        ) from exc


def _materialize_file(source: Path, dest: Path, *, link_mode: str, overwrite: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        if not overwrite:
            return
        dest.unlink()
    if link_mode == "symlink":
        os.symlink(source, dest)
    else:
        shutil.copy2(source, dest)


def _rewrite_artifact_paths(groups: list[dict], mapping: dict[str, str]) -> list[dict]:
    rewritten = []
    for group in groups:
        group_copy = {key: value for key, value in group.items() if key != "samples"}
        samples = []
        for sample in group.get("samples", []) or []:
            sample_copy = dict(sample)
            paths = [mapping[str(path)] for path in sample_copy.get("strict_grpo_artifact_paths", []) or []]
            sample_copy["strict_grpo_artifact_paths"] = paths
            sample_copy["strict_grpo_artifact_count"] = len(paths)
            samples.append(sample_copy)
        group_copy["samples"] = samples
        rewritten.append(group_copy)
    return rewritten


def _stable_parent_key(path: Path) -> str:
    digest = hashlib.sha1(str(path.parent).encode("utf-8")).hexdigest()[:12]
    safe_parent = path.parent.name.replace(" ", "_")[:40] or "root"
    return f"{safe_parent}_{digest}"


def _artifact_paths(groups: list[dict]) -> list[str]:
    paths: list[str] = []
    for group in groups:
        for sample in group.get("samples", []) or []:
            paths.extend(str(path) for path in sample.get("strict_grpo_artifact_paths", []) or [])
    return paths


def _sample_count(groups: list[dict]) -> int:
    return sum(len(group.get("samples", []) or []) for group in groups)


def _file_size_summary(paths: Iterable[str]) -> dict:
    unique_paths = sorted(set(str(path) for path in paths))
    existing_count = 0
    missing_paths = []
    resolved_bytes = 0
    for value in unique_paths:
        path = Path(value).expanduser()
        try:
            stat = path.stat()
        except FileNotFoundError:
            missing_paths.append(str(path))
            continue
        existing_count += 1
        resolved_bytes += stat.st_size
    return {
        "unique_count": len(unique_paths),
        "existing_count": existing_count,
        "missing_count": len(missing_paths),
        "resolved_bytes": resolved_bytes,
        "resolved_gb": resolved_bytes / 1024**3,
        "missing_paths": missing_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize referenced GRPO artifacts into a subset directory.")
    parser.add_argument("groups_jsonl", type=Path, help="Input GRPO groups JSONL.")
    parser.add_argument("--out-root", type=Path, required=True, help="Output root for artifacts/groups/manifest.")
    parser.add_argument("--out-jsonl", type=Path, help="Output rewritten groups JSONL.")
    parser.add_argument("--out-manifest", type=Path, help="Output materialization manifest JSON.")
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--include-replay-context", action="store_true", help="Also materialize replay_context_path files.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing materialized files.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve mappings and print the manifest without writing files.")
    args = parser.parse_args()

    out_root = args.out_root.expanduser()
    out_jsonl = (args.out_jsonl or out_root / "groups" / "grpo_groups.jsonl").expanduser()
    out_manifest = (args.out_manifest or out_root / "manifest.json").expanduser()
    groups, manifest = materialize_grpo_artifacts(
        args.groups_jsonl,
        out_root=out_root,
        link_mode=args.link_mode,
        include_replay_context=args.include_replay_context,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    payload = {**manifest, "output_jsonl": str(out_jsonl), "output_manifest": str(out_manifest)}
    if not args.dry_run:
        write_materialized_outputs(groups, manifest, out_jsonl=out_jsonl, out_manifest=out_manifest)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
