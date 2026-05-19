import json
import math

import torch

from wan_va.rl.denoising_replay import (
    compute_gaussian_transition_logprob,
    load_transition_batch,
)


def _artifact(mean, next_state, std=0.5):
    noise = (next_state - mean) / std
    logprob = -0.5 * noise.pow(2) - math.log(std) - 0.5 * math.log(2 * math.pi)
    mask = torch.ones_like(mean, dtype=torch.bool)
    return {
        "schema_version": 1,
        "scope": "first_action_denoising_step",
        "sampling_seed": 123,
        "frame_st_id": 0,
        "timestep": torch.tensor(999.0),
        "action_xt": torch.zeros_like(mean),
        "action_xt_next": next_state,
        "transition_mean": mean,
        "transition_std": torch.tensor(std),
        "old_logprob_sum": logprob.sum(dim=tuple(range(1, logprob.ndim))),
        "old_logprob_mean": logprob.flatten(1).mean(dim=1),
        "old_logprob_count": torch.tensor([mean[0].numel()]),
        "logprob_mask": mask,
    }


def _trajectory_artifact(transitions):
    payload = []
    for index, artifact in enumerate(transitions):
        item = dict(artifact)
        item.pop("schema_version", None)
        item.pop("scope", None)
        item.pop("sampling_seed", None)
        item.pop("frame_st_id", None)
        item["denoising_step_index"] = index
        payload.append(item)
    return {
        "schema_version": 2,
        "scope": "action_denoising_trajectory",
        "sampling_seed": 123,
        "frame_st_id": 0,
        "num_transitions": len(payload),
        "transitions": payload,
    }


def test_compute_gaussian_transition_logprob_matches_saved_old_logprob():
    mean = torch.zeros(1, 2, 1, 2, 1)
    next_state = torch.full_like(mean, 0.25)
    artifact = _artifact(mean, next_state, std=0.5)

    out = compute_gaussian_transition_logprob(
        transition_mean=artifact["transition_mean"],
        action_xt_next=artifact["action_xt_next"],
        transition_std=artifact["transition_std"],
        logprob_mask=artifact["logprob_mask"],
    )

    assert torch.allclose(out.logprob_sum, artifact["old_logprob_sum"])
    assert torch.equal(out.logprob_count, artifact["old_logprob_count"])


def test_load_transition_batch_reads_group_jsonl_and_artifacts(tmp_path):
    artifact_path = tmp_path / "strict.pt"
    torch.save(_artifact(torch.zeros(1, 1, 1, 1, 1), torch.ones(1, 1, 1, 1, 1) * 0.1), artifact_path)
    group_path = tmp_path / "grpo_groups.jsonl"
    group_path.write_text(
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
                        "strict_grpo_artifact_paths": [str(artifact_path)],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    batch = load_transition_batch(group_path)

    assert batch.transition_count == 1
    assert batch.advantages.tolist() == [1.0]
    assert batch.old_logprob_sum.shape == (1,)
    assert batch.transition_mean.shape[0] == 1


def test_load_transition_batch_expands_trajectory_artifacts(tmp_path):
    artifact_path = tmp_path / "strict_trajectory.pt"
    first = _artifact(torch.zeros(1, 1, 1, 1, 1), torch.ones(1, 1, 1, 1, 1) * 0.1)
    second = _artifact(torch.zeros(1, 1, 1, 1, 1), torch.ones(1, 1, 1, 1, 1) * 0.2)
    torch.save(_trajectory_artifact([first, second]), artifact_path)
    group_path = tmp_path / "grpo_groups.jsonl"
    group_path.write_text(
        json.dumps(
            {
                "group_id": "g0",
                "task": "open_microwave",
                "samples": [
                    {
                        "sample_idx": 0,
                        "reward": 0.0,
                        "advantage": -1.0,
                        "record_path": str(tmp_path / "r0.json"),
                        "strict_grpo_artifact_paths": [str(artifact_path)],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    batch = load_transition_batch(group_path)

    assert batch.transition_count == 2
    assert len(batch.refs) == 2
    assert batch.advantages.tolist() == [-1.0, -1.0]
    assert torch.allclose(batch.action_xt_next.flatten(), torch.tensor([0.1, 0.2]))
