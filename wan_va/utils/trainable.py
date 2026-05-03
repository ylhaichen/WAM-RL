# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Iterable


ACTION_HEAD_PATTERNS = (
    "action_embedder",
    "condition_embedder_action",
    "action_proj_out",
)


@dataclass(frozen=True)
class TrainableSummary:
    mode: str
    trainable_params: int
    total_params: int
    trainable_tensors: int
    total_tensors: int
    trainable_names: tuple[str, ...]

    @property
    def trainable_ratio(self) -> float:
        if self.total_params <= 0:
            return 0.0
        return self.trainable_params / self.total_params


def _as_tuple(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    return tuple(str(item) for item in value if str(item))


def _matches(name: str, patterns: Iterable[str]) -> bool:
    for pattern in patterns:
        if pattern in name or fnmatch(name, pattern):
            return True
    return False


def configure_trainable_parameters(model, config) -> TrainableSummary:
    """Apply the conservative PEFT/freeze policy requested by config.

    Supported modes:
    - full: default, train every parameter.
    - action_heads: train action-specific input/output/time embedding modules.
    - patterns: train parameters whose names match trainable_param_patterns.
    - frozen: train nothing; useful for dry-run inspection only.
    """
    mode = str(getattr(config, "trainable_mode", "full")).strip().lower()
    trainable_patterns = _as_tuple(getattr(config, "trainable_param_patterns", ()))
    frozen_patterns = _as_tuple(getattr(config, "frozen_param_patterns", ()))

    if mode == "full":
        for _, param in model.named_parameters():
            param.requires_grad = True
    elif mode == "frozen":
        for _, param in model.named_parameters():
            param.requires_grad = False
    else:
        if mode == "action_heads":
            trainable_patterns = trainable_patterns or ACTION_HEAD_PATTERNS
        elif mode != "patterns":
            raise ValueError(
                f"Unsupported trainable_mode={mode!r}; expected full, frozen, "
                "action_heads, or patterns."
            )
        if not trainable_patterns:
            raise ValueError("trainable_param_patterns must be set for patterns mode")
        for name, param in model.named_parameters():
            param.requires_grad = _matches(name, trainable_patterns)

    if frozen_patterns:
        for name, param in model.named_parameters():
            if _matches(name, frozen_patterns):
                param.requires_grad = False

    total_params = 0
    trainable_params = 0
    total_tensors = 0
    trainable_tensors = 0
    trainable_names: list[str] = []
    for name, param in model.named_parameters():
        numel = param.numel()
        total_params += numel
        total_tensors += 1
        if param.requires_grad:
            trainable_params += numel
            trainable_tensors += 1
            trainable_names.append(name)

    if trainable_params <= 0:
        raise ValueError(
            f"trainable_mode={mode!r} left zero trainable parameters; "
            "check trainable_param_patterns."
        )

    return TrainableSummary(
        mode=mode,
        trainable_params=trainable_params,
        total_params=total_params,
        trainable_tensors=trainable_tensors,
        total_tensors=total_tensors,
        trainable_names=tuple(trainable_names),
    )


def format_trainable_summary(summary: TrainableSummary, max_names: int = 20) -> str:
    names = list(summary.trainable_names[:max_names])
    suffix = ""
    if len(summary.trainable_names) > max_names:
        suffix = f"\n  ... {len(summary.trainable_names) - max_names} more tensors"
    joined_names = "\n  ".join(names) if names else "<none>"
    return (
        f"trainable_mode={summary.mode}\n"
        f"trainable_params={summary.trainable_params:,}/"
        f"{summary.total_params:,} ({summary.trainable_ratio:.3%})\n"
        f"trainable_tensors={summary.trainable_tensors}/"
        f"{summary.total_tensors}\n"
        f"trainable_tensor_names:\n  {joined_names}{suffix}"
    )
