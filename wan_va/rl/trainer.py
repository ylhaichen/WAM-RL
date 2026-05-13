"""Offline GRPO trainer core for strict denoising-step artifacts.

This module intentionally separates the GRPO math from LingBot-VA model replay.
The included scalar policy is a smoke adapter that validates dataset loading,
ratio/loss computation, optimizer wiring, metrics, and checkpoint IO. A real
LingBot actor adapter must provide current transition means for the same saved
states before this becomes a production model update path.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .denoising_replay import TransitionBatch, compute_gaussian_transition_logprob, load_transition_batch
from .grpo_loss import compute_clipped_grpo_loss


@dataclass(frozen=True)
class OfflineGrpoTrainerConfig:
    groups_jsonl: Path
    output_dir: Path
    steps: int = 10
    learning_rate: float = 1e-3
    clip_low: float = 0.2
    clip_high: float = 0.28
    device: str = "cpu"
    seed: int = 0


@dataclass(frozen=True)
class OfflineGrpoTrainingResult:
    transition_count: int
    steps: int
    final_loss: float
    final_ratio_mean: float
    checkpoint_path: str
    metrics_path: str

    def to_dict(self) -> dict:
        return asdict(self)


class StrictArtifactScalarPolicy(torch.nn.Module):
    """Minimal trainable policy adapter for strict artifact smoke tests."""

    def __init__(self) -> None:
        super().__init__()
        self.mean_shift = torch.nn.Parameter(torch.zeros(()))

    def forward(self, batch: TransitionBatch) -> torch.Tensor:
        current_mean = batch.transition_mean + self.mean_shift
        return compute_gaussian_transition_logprob(
            transition_mean=current_mean,
            action_xt_next=batch.action_xt_next,
            transition_std=batch.transition_std,
            logprob_mask=batch.logprob_mask,
        ).logprob_sum


class OfflineGrpoTrainer:
    def __init__(self, config: OfflineGrpoTrainerConfig, policy: torch.nn.Module | None = None) -> None:
        if config.steps <= 0:
            raise ValueError("steps must be positive")
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        self.batch = load_transition_batch(config.groups_jsonl, device=self.device)
        self.policy = (policy or StrictArtifactScalarPolicy()).to(self.device)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=config.learning_rate)

    def train(self) -> OfflineGrpoTrainingResult:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, float]] = []
        last_metrics: dict[str, torch.Tensor] | None = None
        last_loss: torch.Tensor | None = None

        for step in range(1, self.config.steps + 1):
            self.optimizer.zero_grad(set_to_none=True)
            new_logprob_sum = self.policy(self.batch)
            loss, metrics = compute_clipped_grpo_loss(
                new_logprob_sum=new_logprob_sum,
                old_logprob_sum=self.batch.old_logprob_sum,
                advantages=self.batch.advantages,
                clip_low=self.config.clip_low,
                clip_high=self.config.clip_high,
            )
            loss.backward()
            self.optimizer.step()
            last_loss = loss.detach()
            last_metrics = metrics
            history.append({"step": step, **{key: float(value.detach().cpu()) for key, value in metrics.items()}})

        if last_loss is None or last_metrics is None:
            raise RuntimeError("training loop did not run")

        checkpoint_path = self.config.output_dir / "checkpoint.pt"
        metrics_path = self.config.output_dir / "metrics.json"
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "config": {
                    **asdict(self.config),
                    "groups_jsonl": str(self.config.groups_jsonl),
                    "output_dir": str(self.config.output_dir),
                },
                "history": history,
            },
            checkpoint_path,
        )
        result = OfflineGrpoTrainingResult(
            transition_count=self.batch.transition_count,
            steps=self.config.steps,
            final_loss=float(last_loss.detach().cpu()),
            final_ratio_mean=float(last_metrics["ratio_mean"].detach().cpu()),
            checkpoint_path=str(checkpoint_path),
            metrics_path=str(metrics_path),
        )
        metrics_path.write_text(
            json.dumps({"result": result.to_dict(), "history": history}, indent=2) + "\n",
            encoding="utf-8",
        )
        return result
