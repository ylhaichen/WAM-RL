from dataclasses import dataclass

from wan_va.rl.evaluator import summarize_rollout_success
from wan_va.rl.reward import binary_success_reward


@dataclass
class Rollout:
    task: str
    success: bool


def test_binary_success_reward_maps_success_to_float_reward():
    assert binary_success_reward(True) == 1.0
    assert binary_success_reward(False) == 0.0


def test_summarize_rollout_success_reports_overall_and_per_task_rates():
    summary = summarize_rollout_success(
        [
            Rollout("easy", True),
            Rollout("easy", False),
            Rollout("hard", True),
        ]
    )

    assert summary.success_rate == 2 / 3
    assert summary.task_success == {"easy": 0.5, "hard": 1.0}
