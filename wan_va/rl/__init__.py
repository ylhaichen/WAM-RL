"""RL utilities for WAM-RL."""

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
from .denoising_replay import TransitionBatch, TransitionLogprob, compute_gaussian_transition_logprob, load_transition_batch
from .evaluator import summarize_rollout_success
from .grpo_loss import GrpoLossOutput, compute_clipped_grpo_loss
from .group_builder import build_grpo_groups
from .iteration_controller import RLIterationPaths, build_iteration_paths
from .manifest import build_grpo_manifest, write_grpo_manifest
from .rollout_worker import GroupedRolloutPlan, RolloutBatch, RolloutTaskAssignment, build_grouped_rollout_plan
from .reward import binary_success_reward
from .scheduler_logprob import StochasticFlowMatchStep, stochastic_flowmatch_step
from .trajectory_schema import GrpoGroup, GrpoGroupBuildResult, GrpoSample, GrpoSummary
from .trainer import OfflineGrpoTrainer, OfflineGrpoTrainerConfig, OfflineGrpoTrainingResult
from .validation import ValidationIssue, ValidationReport, validate_rollout_records

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
