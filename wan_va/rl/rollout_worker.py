"""Framework-neutral grouped rollout planning utilities.

This module deliberately does not import RoboTwin, Ray, torch, or the LingBot
server. It defines the deterministic rollout schedule contract used by shell,
future multiprocessing, and future Ray workers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RolloutTaskAssignment:
    task_name: str
    task_index: int
    gpu_id: int
    port: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RolloutBatch:
    group_index: int
    sample_idx: int
    env_seed: int
    sampling_seed: int
    prompt_index: int
    tasks: tuple[RolloutTaskAssignment, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["tasks"] = [task.to_dict() for task in self.tasks]
        return data


@dataclass(frozen=True)
class GroupedRolloutPlan:
    task_names: tuple[str, ...]
    group_size: int
    groups_per_task: int
    start_seed: int
    prompt_index: int
    num_gpus: int
    start_port: int
    batches: tuple[RolloutBatch, ...]

    @property
    def total_context_groups(self) -> int:
        return self.groups_per_task

    @property
    def total_rollouts(self) -> int:
        return len(self.task_names) * self.group_size * self.groups_per_task

    def to_dict(self) -> dict:
        return {
            "task_names": list(self.task_names),
            "group_size": self.group_size,
            "groups_per_task": self.groups_per_task,
            "start_seed": self.start_seed,
            "prompt_index": self.prompt_index,
            "num_gpus": self.num_gpus,
            "start_port": self.start_port,
            "total_context_groups": self.total_context_groups,
            "total_rollouts": self.total_rollouts,
            "batches": [batch.to_dict() for batch in self.batches],
        }


def build_grouped_rollout_plan(
    *,
    task_names: list[str] | tuple[str, ...],
    group_size: int,
    groups_per_task: int,
    start_seed: int,
    prompt_index: int,
    num_gpus: int,
    start_port: int,
) -> GroupedRolloutPlan:
    if not task_names:
        raise ValueError("task_names must not be empty")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if groups_per_task <= 0:
        raise ValueError("groups_per_task must be positive")
    if num_gpus <= 0:
        raise ValueError("num_gpus must be positive")

    task_tuple = tuple(str(task) for task in task_names)
    batches: list[RolloutBatch] = []
    for group_index in range(groups_per_task):
        env_seed = start_seed + group_index
        for sample_idx in range(group_size):
            sampling_seed = start_seed * 1_000_000 + group_index * group_size + sample_idx
            assignments = tuple(
                RolloutTaskAssignment(
                    task_name=task_name,
                    task_index=task_index,
                    gpu_id=task_index % num_gpus,
                    port=start_port + (task_index % num_gpus),
                )
                for task_index, task_name in enumerate(task_tuple)
            )
            batches.append(
                RolloutBatch(
                    group_index=group_index,
                    sample_idx=sample_idx,
                    env_seed=env_seed,
                    sampling_seed=sampling_seed,
                    prompt_index=prompt_index,
                    tasks=assignments,
                )
            )

    return GroupedRolloutPlan(
        task_names=task_tuple,
        group_size=group_size,
        groups_per_task=groups_per_task,
        start_seed=start_seed,
        prompt_index=prompt_index,
        num_gpus=num_gpus,
        start_port=start_port,
        batches=tuple(batches),
    )
