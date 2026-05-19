"""Replay helpers for saved strict denoising-step GRPO artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .dataset import (
    GrpoTransitionRef,
    iter_strict_artifact_transitions,
    load_strict_artifact,
    read_transition_refs,
)


@dataclass(frozen=True)
class TransitionLogprob:
    logprob_sum: torch.Tensor
    logprob_mean: torch.Tensor
    logprob_count: torch.Tensor


@dataclass(frozen=True)
class TransitionBatch:
    refs: tuple[GrpoTransitionRef, ...]
    advantages: torch.Tensor
    rewards: torch.Tensor
    old_logprob_sum: torch.Tensor
    old_logprob_count: torch.Tensor
    transition_mean: torch.Tensor
    action_xt_next: torch.Tensor
    transition_std: torch.Tensor
    logprob_mask: torch.Tensor

    @property
    def transition_count(self) -> int:
        return int(self.old_logprob_sum.numel())


def compute_gaussian_transition_logprob(
    *,
    transition_mean: torch.Tensor,
    action_xt_next: torch.Tensor,
    transition_std: torch.Tensor | float,
    logprob_mask: torch.Tensor | None = None,
) -> TransitionLogprob:
    """Compute diagonal Gaussian logprob for `action_xt_next`.

    The policy distribution is `N(transition_mean, transition_std^2 I)`. The
    mask excludes conditioning dimensions such as the first action frame.
    """

    mean = transition_mean.float()
    target = action_xt_next.float().to(mean.device)
    std = torch.as_tensor(transition_std, dtype=mean.dtype, device=mean.device)
    if torch.any(std <= 0):
        raise ValueError("transition_std must be positive")

    while std.ndim < mean.ndim:
        std = std.view(*std.shape, *([1] * (mean.ndim - std.ndim)))

    logprob = -0.5 * ((target - mean) / std).pow(2) - torch.log(std) - 0.5 * torch.log(
        torch.tensor(2.0 * torch.pi, dtype=mean.dtype, device=mean.device)
    )
    if logprob_mask is None:
        mask = torch.ones_like(logprob, dtype=torch.bool)
    else:
        mask = logprob_mask.to(device=mean.device, dtype=torch.bool)
        logprob = logprob.masked_fill(~mask, 0.0)

    reduce_dims = tuple(range(1, logprob.ndim))
    count = mask.sum(dim=reduce_dims).clamp_min(1)
    total = logprob.sum(dim=reduce_dims)
    return TransitionLogprob(
        logprob_sum=total,
        logprob_mean=total / count,
        logprob_count=count,
    )


def load_transition_batch(
    groups_jsonl: Path,
    *,
    loader=None,
    device: torch.device | str | None = None,
) -> TransitionBatch:
    """Load all strict GRPO artifacts referenced by grouped rollout JSONL."""

    target_device = torch.device(device or "cpu")
    artifact_refs = tuple(read_transition_refs(groups_jsonl))
    if not artifact_refs:
        raise ValueError(f"no transition artifacts referenced by {groups_jsonl}")

    refs: list[GrpoTransitionRef] = []
    advantages: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    old_logprob_sum: list[torch.Tensor] = []
    old_logprob_count: list[torch.Tensor] = []
    transition_mean: list[torch.Tensor] = []
    action_xt_next: list[torch.Tensor] = []
    transition_std: list[torch.Tensor] = []
    logprob_mask: list[torch.Tensor] = []

    for ref in artifact_refs:
        artifact = load_strict_artifact(Path(ref.artifact_path), loader=loader)
        for transition in iter_strict_artifact_transitions(artifact):
            mean = _ensure_batch_dim(transition["transition_mean"]).to(target_device).float()
            next_state = _ensure_batch_dim(transition["action_xt_next"]).to(target_device).float()
            mask = _ensure_batch_dim(transition["logprob_mask"]).to(target_device).bool()
            batch_size = mean.shape[0]
            refs.append(ref)
            transition_mean.append(mean)
            action_xt_next.append(next_state)
            logprob_mask.append(mask)
            old_logprob_sum.append(_as_batch_vector(transition["old_logprob_sum"], batch_size, target_device))
            old_logprob_count.append(_as_batch_vector(transition["old_logprob_count"], batch_size, target_device))
            transition_std.append(_as_batch_vector(transition["transition_std"], batch_size, target_device))
            advantages.append(torch.full((batch_size,), float(ref.advantage), dtype=torch.float32, device=target_device))
            rewards.append(torch.full((batch_size,), float(ref.reward), dtype=torch.float32, device=target_device))

    return TransitionBatch(
        refs=tuple(refs),
        advantages=torch.cat(advantages),
        rewards=torch.cat(rewards),
        old_logprob_sum=torch.cat(old_logprob_sum),
        old_logprob_count=torch.cat(old_logprob_count),
        transition_mean=torch.cat(transition_mean, dim=0),
        action_xt_next=torch.cat(action_xt_next, dim=0),
        transition_std=torch.cat(transition_std),
        logprob_mask=torch.cat(logprob_mask, dim=0),
    )


def _ensure_batch_dim(value: torch.Tensor) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    return tensor


def _as_batch_vector(value: torch.Tensor | float, batch_size: int, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device).flatten()
    if tensor.numel() == 1:
        return tensor.repeat(batch_size)
    if tensor.numel() != batch_size:
        raise ValueError(f"expected scalar or {batch_size} values, got shape {tuple(tensor.shape)}")
    return tensor
