"""Helpers for RoboTwin rollout metadata.

The rollout JSON is the contract between expensive simulator evaluation and
later pseudo/strict GRPO training. Keep this module lightweight so it can be
tested without importing RoboTwin or SAPIEN.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def _as_path_strings(paths: Iterable[str | Path] | None) -> list[str]:
    if not paths:
        return []
    return [str(Path(path)) for path in paths if str(path)]


def _as_ints(values: Iterable[int | str] | None) -> list[int]:
    if not values:
        return []
    return [int(value) for value in values]


def build_group_id(
    *,
    task_name: str,
    env_seed: int,
    prompt: str,
    group_index: int,
    prompt_key: str | None = None,
) -> str:
    """Create a stable, path-safe group id for one GRPO context."""
    digest_source = prompt if prompt_key is None else prompt_key
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:10]
    return f"{task_name}_seed{int(env_seed)}_group{int(group_index):06d}_{digest}"


def build_rollout_metadata(
    *,
    task_name: str,
    episode_index: int,
    env_seed: int,
    prompt: str,
    success: bool,
    action_count: int,
    obs_count: int,
    take_action_cnt: int,
    step_lim: int,
    executed_actions_path: str | Path,
    visualization_path: str | Path,
    initial_obs_path: str | Path | None = None,
    run_id: str | None = None,
    policy_checkpoint: str | None = None,
    reference_checkpoint: str | None = None,
    group_id: str | None = None,
    group_index: int | None = None,
    sample_idx: int | None = None,
    group_size: int | None = None,
    sampling_seed: int | None = None,
    prompt_index: int | None = None,
    planned_seed: int | None = None,
    video_guidance_scale: float | None = None,
    action_guidance_scale: float | None = None,
    action_num_inference_steps: int | None = None,
    server_action_paths: Iterable[str | Path] | None = None,
    server_latent_paths: Iterable[str | Path] | None = None,
    strict_grpo_artifact_paths: Iterable[str | Path] | None = None,
    strict_grpo_scope: str | None = None,
    strict_grpo_replay_context_paths: Iterable[str | Path] | None = None,
    strict_grpo_replay_context_tensor_bytes: Iterable[int | str] | None = None,
    strict_grpo_replay_context_max_gb: float | None = None,
    strict_grpo_capture_chunk_indices: Iterable[int | str] | None = None,
    strict_grpo_capture_chunk_stride: int | None = None,
    strict_grpo_capture_max_chunks: int | None = None,
) -> dict:
    """Build one JSON-serializable rollout record."""
    server_action_path_list = _as_path_strings(server_action_paths)
    server_latent_path_list = _as_path_strings(server_latent_paths)
    strict_path_list = _as_path_strings(strict_grpo_artifact_paths)
    replay_context_path_list = _as_path_strings(strict_grpo_replay_context_paths)
    replay_context_tensor_bytes = _as_ints(strict_grpo_replay_context_tensor_bytes)
    replay_context_total_tensor_bytes = sum(replay_context_tensor_bytes)
    capture_chunk_indices = _as_ints(strict_grpo_capture_chunk_indices)

    return {
        "schema_version": 2,
        "run_id": run_id or "",
        "policy_checkpoint": policy_checkpoint or "",
        "reference_checkpoint": reference_checkpoint or "",
        "task_name": task_name,
        "episode_index": int(episode_index),
        "seed": int(env_seed),
        "env_seed": int(env_seed),
        "planned_seed": None if planned_seed is None else int(planned_seed),
        "sampling_seed": None if sampling_seed is None else int(sampling_seed),
        "prompt": prompt,
        "group_id": group_id or "",
        "group_index": None if group_index is None else int(group_index),
        "sample_idx": None if sample_idx is None else int(sample_idx),
        "group_size": None if group_size is None else int(group_size),
        "prompt_index": None if prompt_index is None else int(prompt_index),
        "success": bool(success),
        "reward": 1.0 if success else 0.0,
        "obs_count": int(obs_count),
        "action_count": int(action_count),
        "take_action_cnt": int(take_action_cnt),
        "step_lim": int(step_lim),
        "actions_path": str(Path(executed_actions_path)),
        "executed_actions_path": str(Path(executed_actions_path)),
        "initial_obs_path": "" if initial_obs_path is None else str(Path(initial_obs_path)),
        "visualization_path": str(Path(visualization_path)),
        "server_action_paths": server_action_path_list,
        "server_latent_paths": server_latent_path_list,
        "strict_grpo_ready": bool(strict_path_list),
        "strict_grpo_scope": strict_grpo_scope or ("first_action_denoising_step" if strict_path_list else ""),
        "strict_grpo_artifact_count": len(strict_path_list),
        "strict_grpo_artifact_paths": strict_path_list,
        "strict_grpo_replay_context_count": len(replay_context_path_list),
        "strict_grpo_replay_context_paths": replay_context_path_list,
        "strict_grpo_replay_context_tensor_bytes": replay_context_tensor_bytes,
        "strict_grpo_replay_context_total_tensor_bytes": replay_context_total_tensor_bytes,
        "strict_grpo_replay_context_max_gb": strict_grpo_replay_context_max_gb,
        "strict_grpo_capture_chunk_indices": capture_chunk_indices,
        "strict_grpo_capture_chunk_stride": (
            None if strict_grpo_capture_chunk_stride is None else int(strict_grpo_capture_chunk_stride)
        ),
        "strict_grpo_capture_max_chunks": (
            None if strict_grpo_capture_max_chunks is None else int(strict_grpo_capture_max_chunks)
        ),
        "video_guidance_scale": video_guidance_scale,
        "action_guidance_scale": action_guidance_scale,
        "action_num_inference_steps": action_num_inference_steps,
    }
