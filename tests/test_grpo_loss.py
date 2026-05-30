import math

import torch

from wan_va.rl.grpo_loss import compute_clipped_grpo_loss


def test_clipped_grpo_loss_uses_asymmetric_clip_for_positive_and_negative_advantages():
    new_logprob = torch.tensor([math.log(2.0), math.log(0.5)])
    old_logprob = torch.zeros(2)
    advantages = torch.tensor([1.0, -1.0])

    loss, metrics = compute_clipped_grpo_loss(
        new_logprob_sum=new_logprob,
        old_logprob_sum=old_logprob,
        advantages=advantages,
        clip_low=0.2,
        clip_high=0.28,
    )

    assert torch.isclose(loss, torch.tensor((-1.28 + 0.8) / 2))
    assert torch.isclose(metrics["ratio_mean"], torch.tensor(1.25))
    assert torch.isclose(metrics["clip_fraction"], torch.tensor(1.0))
    assert "logratio_min" in metrics
    assert "logratio_max" in metrics
    assert "logratio_clamp_fraction" in metrics


def test_clipped_grpo_loss_reports_logratio_clamp_saturation():
    _, metrics = compute_clipped_grpo_loss(
        new_logprob_sum=torch.tensor([-30.0, 30.0]),
        old_logprob_sum=torch.zeros(2),
        advantages=torch.ones(2),
    )

    assert torch.isclose(metrics["logratio_min"], torch.tensor(-30.0))
    assert torch.isclose(metrics["logratio_max"], torch.tensor(30.0))
    assert torch.isclose(metrics["logratio_clamp_fraction"], torch.tensor(1.0))


def test_clipped_grpo_loss_rejects_nonfinite_logratios():
    try:
        compute_clipped_grpo_loss(
            new_logprob_sum=torch.tensor([float("nan")]),
            old_logprob_sum=torch.tensor([0.0]),
            advantages=torch.tensor([1.0]),
        )
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("expected non-finite logratio ValueError")
