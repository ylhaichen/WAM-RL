"""Filesystem planning helpers for staged native RL iterations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RLIterationPaths:
    root: Path
    actor_old: Path
    rollouts: Path
    groups: Path
    train: Path
    checkpoints: Path
    eval_selected: Path
    eval_heldout: Path
    reports: Path

    def mkdirs(self) -> None:
        for path in (
            self.actor_old,
            self.rollouts,
            self.groups,
            self.train,
            self.checkpoints,
            self.eval_selected,
            self.eval_heldout,
            self.reports,
        ):
            path.mkdir(parents=True, exist_ok=True)


def build_iteration_paths(base_dir: Path, *, iteration: int) -> RLIterationPaths:
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    root = base_dir / f"rl_iter_{iteration:04d}"
    return RLIterationPaths(
        root=root,
        actor_old=root / "actor_old",
        rollouts=root / "rollouts",
        groups=root / "groups",
        train=root / "train",
        checkpoints=root / "checkpoints",
        eval_selected=root / "eval_selected",
        eval_heldout=root / "eval_heldout",
        reports=root / "reports",
    )
