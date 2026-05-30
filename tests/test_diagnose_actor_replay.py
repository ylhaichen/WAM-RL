import torch

from tools.diagnose_actor_replay import _logprob_for_mean, _normalize_logprob_std_floor


def _transition(std: float = 0.01):
    state = torch.zeros(1, 1, 1, 1, 1)
    return {
        "action_xt_next": torch.full_like(state, 0.1),
        "transition_std": torch.tensor(std),
        "logprob_mask": torch.ones_like(state, dtype=torch.bool),
    }


def test_diagnose_logprob_std_floor_matches_trainer_floor_semantics():
    transition = _transition()
    mean = torch.zeros(1, 1, 1, 1, 1)

    raw = _logprob_for_mean(
        mean,
        transition,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    floored = _logprob_for_mean(
        mean,
        transition,
        device=torch.device("cpu"),
        dtype=torch.float32,
        logprob_std_floor=0.1,
    )

    assert _normalize_logprob_std_floor(None) is None
    assert _normalize_logprob_std_floor(0.0) is None
    assert _normalize_logprob_std_floor(-1.0) is None
    assert _normalize_logprob_std_floor(0.1) == 0.1
    assert floored.item() > raw.item()
