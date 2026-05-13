"""Checkpoint promotion rules for iterative RL runs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class EvalMetrics:
    success_rate: float
    task_success: dict[str, float]


class PromotionDecision(str, Enum):
    PROMOTE = "promote"
    REJECT_NO_IMPROVEMENT = "reject_no_improvement"
    REJECT_REGRESSION = "reject_regression"
    REJECT_INSUFFICIENT_DATA = "reject_insufficient_data"


def decide_checkpoint_promotion(
    baseline: EvalMetrics,
    candidate: EvalMetrics,
    *,
    min_success_rate_delta: float = 0.02,
    max_task_regression: float = 0.05,
) -> PromotionDecision:
    if not baseline.task_success or not candidate.task_success:
        return PromotionDecision.REJECT_INSUFFICIENT_DATA

    common_tasks = sorted(set(baseline.task_success) & set(candidate.task_success))
    if not common_tasks:
        return PromotionDecision.REJECT_INSUFFICIENT_DATA

    for task in common_tasks:
        regression = baseline.task_success[task] - candidate.task_success[task]
        if regression > max_task_regression:
            return PromotionDecision.REJECT_REGRESSION

    if candidate.success_rate - baseline.success_rate < min_success_rate_delta:
        return PromotionDecision.REJECT_NO_IMPROVEMENT

    return PromotionDecision.PROMOTE
