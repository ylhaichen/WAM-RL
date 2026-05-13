from wan_va.rl.checkpoint_gate import EvalMetrics, PromotionDecision, decide_checkpoint_promotion
from wan_va.rl.iteration_controller import build_iteration_paths


def test_checkpoint_gate_promotes_only_when_candidate_improves_without_regression():
    baseline = EvalMetrics(success_rate=0.30, task_success={"easy": 0.9, "hard": 0.2})
    candidate = EvalMetrics(success_rate=0.38, task_success={"easy": 0.88, "hard": 0.36})

    decision = decide_checkpoint_promotion(
        baseline,
        candidate,
        min_success_rate_delta=0.05,
        max_task_regression=0.05,
    )

    assert decision == PromotionDecision.PROMOTE


def test_checkpoint_gate_rejects_large_task_regression():
    baseline = EvalMetrics(success_rate=0.30, task_success={"easy": 0.9, "hard": 0.2})
    candidate = EvalMetrics(success_rate=0.42, task_success={"easy": 0.7, "hard": 0.55})

    decision = decide_checkpoint_promotion(
        baseline,
        candidate,
        min_success_rate_delta=0.05,
        max_task_regression=0.05,
    )

    assert decision == PromotionDecision.REJECT_REGRESSION


def test_build_iteration_paths_is_stable_and_complete(tmp_path):
    paths = build_iteration_paths(tmp_path / "runs", iteration=3)

    assert paths.root == tmp_path / "runs" / "rl_iter_0003"
    assert paths.rollouts == paths.root / "rollouts"
    assert paths.groups == paths.root / "groups"
    assert paths.reports == paths.root / "reports"
