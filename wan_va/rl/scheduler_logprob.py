"""Stochastic FlowMatch transition helpers for strict denoising-step GRPO."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .denoising_replay import compute_gaussian_transition_logprob


@dataclass(frozen=True)
class StochasticFlowMatchStep:
    transition_mean: torch.Tensor
    next_state: torch.Tensor
    transition_std: torch.Tensor
    logprob_sum: torch.Tensor
    logprob_mean: torch.Tensor
    logprob_count: torch.Tensor
    logprob_mask: torch.Tensor


def stochastic_flowmatch_step(
    *,
    scheduler,
    model_output: torch.Tensor,
    timestep: torch.Tensor,
    sample: torch.Tensor,
    transition_std: float,
    logprob_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> StochasticFlowMatchStep:
    """Run a deterministic FlowMatch step, then sample a Gaussian transition."""

    if transition_std <= 0:
        raise ValueError("transition_std must be positive")
    transition_mean = scheduler.step(model_output, timestep, sample, return_dict=False)
    noise = torch.randn(
        transition_mean.shape,
        dtype=transition_mean.dtype,
        device=transition_mean.device,
        generator=generator,
    )
    std = torch.tensor(transition_std, dtype=transition_mean.dtype, device=transition_mean.device)
    next_state = transition_mean + noise * std
    mask = logprob_mask if logprob_mask is not None else torch.ones_like(next_state, dtype=torch.bool)
    logprob = compute_gaussian_transition_logprob(
        transition_mean=transition_mean,
        action_xt_next=next_state,
        transition_std=std,
        logprob_mask=mask,
    )
    return StochasticFlowMatchStep(
        transition_mean=transition_mean,
        next_state=next_state,
        transition_std=std,
        logprob_sum=logprob.logprob_sum,
        logprob_mean=logprob.logprob_mean,
        logprob_count=logprob.logprob_count,
        logprob_mask=mask,
    )
