"""RL utilities for WAM-RL.

Keep this package import lightweight. Some submodules depend on `torch`, but
metadata-only CLI tools such as `tools/build_grpo_groups.py` should run without
importing tensor training code.
"""

from importlib import import_module

from .dataset import (
    DatasetIssue,
    DatasetValidationReport,
    GrpoTransitionRef,
    inspect_strict_artifacts,
    load_strict_artifact,
    read_transition_refs,
    validate_transition_refs,
)
from .checkpoint_gate import EvalMetrics, PromotionDecision, decide_checkpoint_promotion
from .evaluator import summarize_rollout_success
from .group_builder import build_grpo_groups
from .iteration_controller import RLIterationPaths, build_iteration_paths
from .manifest import build_grpo_manifest, write_grpo_manifest
from .rollout_worker import GroupedRolloutPlan, RolloutBatch, RolloutTaskAssignment, build_grouped_rollout_plan
from .reward import binary_success_reward
from .trajectory_schema import GrpoGroup, GrpoGroupBuildResult, GrpoSample, GrpoSummary
from .validation import ValidationIssue, ValidationReport, validate_rollout_records

_LAZY_EXPORTS = {
    "GrpoLossOutput": ("grpo_loss", "GrpoLossOutput"),
    "compute_clipped_grpo_loss": ("grpo_loss", "compute_clipped_grpo_loss"),
    "TransitionBatch": ("denoising_replay", "TransitionBatch"),
    "TransitionLogprob": ("denoising_replay", "TransitionLogprob"),
    "compute_gaussian_transition_logprob": ("denoising_replay", "compute_gaussian_transition_logprob"),
    "load_transition_batch": ("denoising_replay", "load_transition_batch"),
    "StochasticFlowMatchStep": ("scheduler_logprob", "StochasticFlowMatchStep"),
    "stochastic_flowmatch_step": ("scheduler_logprob", "stochastic_flowmatch_step"),
    "OfflineGrpoTrainer": ("trainer", "OfflineGrpoTrainer"),
    "OfflineGrpoTrainerConfig": ("trainer", "OfflineGrpoTrainerConfig"),
    "OfflineGrpoTrainingResult": ("trainer", "OfflineGrpoTrainingResult"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(f".{module_name}", __name__), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "EvalMetrics",
    "GrpoGroup",
    "GrpoGroupBuildResult",
    "GrpoLossOutput",
    "GrpoSample",
    "GrpoSummary",
    "GrpoTransitionRef",
    "GroupedRolloutPlan",
    "OfflineGrpoTrainer",
    "OfflineGrpoTrainerConfig",
    "OfflineGrpoTrainingResult",
    "PromotionDecision",
    "RLIterationPaths",
    "RolloutBatch",
    "RolloutTaskAssignment",
    "StochasticFlowMatchStep",
    "TransitionBatch",
    "TransitionLogprob",
    "DatasetIssue",
    "DatasetValidationReport",
    "ValidationIssue",
    "ValidationReport",
    "build_grpo_manifest",
    "build_grpo_groups",
    "build_grouped_rollout_plan",
    "build_iteration_paths",
    "binary_success_reward",
    "compute_clipped_grpo_loss",
    "compute_gaussian_transition_logprob",
    "decide_checkpoint_promotion",
    "inspect_strict_artifacts",
    "load_strict_artifact",
    "load_transition_batch",
    "read_transition_refs",
    "summarize_rollout_success",
    "stochastic_flowmatch_step",
    "validate_transition_refs",
    "validate_rollout_records",
    "write_grpo_manifest",
]
