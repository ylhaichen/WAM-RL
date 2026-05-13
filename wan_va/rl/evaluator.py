"""Evaluation summaries used by RL checkpoint gates."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .checkpoint_gate import EvalMetrics


def summarize_rollout_success(records: Iterable[object]) -> EvalMetrics:
    by_task: dict[str, list[float]] = defaultdict(list)
    rewards: list[float] = []
    for record in records:
        value = 1.0 if bool(getattr(record, "success")) else 0.0
        rewards.append(value)
        by_task[str(getattr(record, "task", getattr(record, "task_name", "")))].append(value)
    if not rewards:
        return EvalMetrics(success_rate=0.0, task_success={})
    return EvalMetrics(
        success_rate=sum(rewards) / len(rewards),
        task_success={task: sum(values) / len(values) for task, values in sorted(by_task.items())},
    )
