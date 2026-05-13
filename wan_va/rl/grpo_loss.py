"""Clipped GRPO objective for denoising-step transition log probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GrpoLossOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


def compute_clipped_grpo_loss(
    *,
    new_logprob_sum: torch.Tensor,
    old_logprob_sum: torch.Tensor,
    advantages: torch.Tensor,
    clip_low: float = 0.2,
    clip_high: float = 0.28,
    clamp_logratio: float = 20.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the asymmetric clipped GRPO loss.

    Inputs are per-transition log probability sums. This matches the current
    strict artifact contract, where one saved artifact corresponds to one
    first action denoising transition for an action chunk.
    """

    new_logprob_sum, old_logprob_sum, advantages = torch.broadcast_tensors(
        new_logprob_sum.float(),
        old_logprob_sum.float(),
        advantages.float(),
    )
    if new_logprob_sum.numel() == 0:
        raise ValueError("cannot compute GRPO loss for an empty batch")
    if clip_low < 0 or clip_high < 0:
        raise ValueError("clip_low and clip_high must be non-negative")
    if clamp_logratio <= 0:
        raise ValueError("clamp_logratio must be positive")

    raw_logratio = new_logprob_sum - old_logprob_sum
    if not torch.isfinite(raw_logratio).all():
        raise ValueError("non-finite logratio in GRPO loss input")
    if not torch.isfinite(advantages).all():
        raise ValueError("non-finite advantages in GRPO loss input")

    logratio = raw_logratio.clamp(min=-clamp_logratio, max=clamp_logratio)
    ratio = torch.exp(logratio)
    clipped_ratio = ratio.clamp(min=1.0 - clip_low, max=1.0 + clip_high)
    unclipped_objective = ratio * advantages
    clipped_objective = clipped_ratio * advantages
    objective = torch.minimum(unclipped_objective, clipped_objective)
    loss = -objective.mean()

    clipped = (ratio < (1.0 - clip_low)) | (ratio > (1.0 + clip_high))
    metrics = {
        "loss": loss.detach(),
        "ratio_mean": ratio.detach().mean(),
        "ratio_min": ratio.detach().min(),
        "ratio_max": ratio.detach().max(),
        "clip_fraction": clipped.float().mean().detach(),
        "advantage_mean": advantages.detach().mean(),
        "logratio_mean": raw_logratio.detach().mean(),
    }
    return loss, metrics
