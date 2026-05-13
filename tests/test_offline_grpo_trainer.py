import json

import torch

from tools.train_offline_grpo_smoke import run_smoke_training


def _write_artifact(path, mean_value, next_value):
    mean = torch.full((1, 1, 1, 1, 1), mean_value)
    next_state = torch.full_like(mean, next_value)
    std = torch.tensor(0.2)
    noise = (next_state - mean) / std
    logprob = -0.5 * noise.pow(2) - torch.log(std) - 0.5 * torch.log(torch.tensor(2 * torch.pi))
    torch.save(
        {
            "schema_version": 1,
            "scope": "first_action_denoising_step",
            "sampling_seed": 123,
            "frame_st_id": 0,
            "timestep": torch.tensor(999.0),
            "action_xt": torch.zeros_like(mean),
            "action_xt_next": next_state,
            "transition_mean": mean,
            "transition_std": std,
            "old_logprob_sum": logprob.flatten(1).sum(dim=1),
            "old_logprob_mean": logprob.flatten(1).mean(dim=1),
            "old_logprob_count": torch.tensor([1]),
            "logprob_mask": torch.ones_like(mean, dtype=torch.bool),
        },
        path,
    )


def test_smoke_training_writes_metrics_and_checkpoint(tmp_path):
    artifact0 = tmp_path / "strict0.pt"
    artifact1 = tmp_path / "strict1.pt"
    _write_artifact(artifact0, mean_value=0.0, next_value=0.1)
    _write_artifact(artifact1, mean_value=0.0, next_value=-0.1)
    groups = tmp_path / "grpo_groups.jsonl"
    groups.write_text(
        json.dumps(
            {
                "group_id": "g0",
                "task": "open_microwave",
                "samples": [
                    {
                        "sample_idx": 0,
                        "reward": 1.0,
                        "advantage": 1.0,
                        "record_path": str(tmp_path / "r0.json"),
                        "strict_grpo_artifact_paths": [str(artifact0)],
                    },
                    {
                        "sample_idx": 1,
                        "reward": 0.0,
                        "advantage": -1.0,
                        "record_path": str(tmp_path / "r1.json"),
                        "strict_grpo_artifact_paths": [str(artifact1)],
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_smoke_training(groups, tmp_path / "train", steps=3, learning_rate=0.05)

    assert result["transition_count"] == 2
    assert result["steps"] == 3
    assert result["final_loss"] == result["final_loss"]
    assert (tmp_path / "train" / "metrics.json").exists()
    assert (tmp_path / "train" / "checkpoint.pt").exists()
