import json
from pathlib import Path

import torch

from wan_va.rl.actor_replay import (
    ActorReplayGrpoTrainer,
    ActorReplayTrainerConfig,
    MissingReplayContextError,
    iter_actor_replay_examples,
)


class ToyAttn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_caches = {
            "pos": {
                "k": torch.zeros(1, 1, 1, 1),
                "v": torch.zeros(1, 1, 1, 1),
                "id": torch.zeros(1, dtype=torch.long),
                "mask": torch.ones(1, dtype=torch.bool),
                "is_pred": torch.zeros(1, dtype=torch.bool),
            }
        }


class ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = ToyAttn()


class ToyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([ToyBlock()])
        self.action_embedder = torch.nn.Linear(1, 1)
        self.condition_embedder_action = torch.nn.Linear(1, 1)
        self.action_proj_out = torch.nn.Linear(1, 1)

    def forward(self, input_dict, update_cache=0, cache_name="pos", action_mode=True):
        del update_cache, cache_name, action_mode
        x = input_dict["noisy_latents"].permute(0, 2, 3, 4, 1).reshape(1, 1, 1)
        return self.action_proj_out(self.action_embedder(x))


def _transition():
    state = torch.zeros(1, 1, 1, 1, 1)
    return {
        "denoising_step_index": 0,
        "timestep": torch.tensor(1000.0),
        "action_xt": state,
        "action_xt_next": torch.full_like(state, 0.2),
        "transition_mean": state,
        "transition_std": torch.tensor(0.5),
        "old_logprob_sum": torch.tensor([0.0]),
        "old_logprob_mean": torch.tensor([0.0]),
        "old_logprob_count": torch.tensor([1]),
        "logprob_mask": torch.ones_like(state, dtype=torch.bool),
        "replay_input": {
            "noisy_latents": state,
            "timesteps": torch.tensor([1000.0]),
        },
    }


def _replay_context():
    return {
        "schema_version": 1,
        "cache_name": "pos",
        "transformer_cache": [
            {
                "k": torch.zeros(1, 1, 1, 1),
                "v": torch.zeros(1, 1, 1, 1),
                "id": torch.zeros(1, dtype=torch.long),
                "mask": torch.ones(1, dtype=torch.bool),
                "is_pred": torch.zeros(1, dtype=torch.bool),
            }
        ],
        "grid_id": torch.zeros(1, 3, dtype=torch.long),
        "text_emb": torch.zeros(1, 1, 1),
        "negative_text_emb": None,
        "use_cfg": False,
        "action_guidance_scale": 1.0,
        "action_num_inference_steps": 2,
        "frame_chunk_size": 1,
    }


def _write_group(tmp_path: Path, artifact_path: Path):
    group_path = tmp_path / "grpo_groups.jsonl"
    group_path.write_text(
        json.dumps(
            {
                "group_id": "g0",
                "task": "toy",
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
    return group_path


def test_actor_replay_requires_replay_context(tmp_path):
    artifact_path = tmp_path / "strict.pt"
    torch.save(
        {
            "schema_version": 1,
            "scope": "first_action_denoising_step",
            "sampling_seed": 1,
            "frame_st_id": 0,
            **{key: value for key, value in _transition().items() if key != "replay_input"},
        },
        artifact_path,
    )
    group_path = _write_group(tmp_path, artifact_path)

    try:
        list(iter_actor_replay_examples(group_path))
    except MissingReplayContextError as exc:
        assert "replay_context" in str(exc)
    else:
        raise AssertionError("expected MissingReplayContextError")


def test_actor_replay_trainer_updates_trainable_action_modules(tmp_path):
    artifact_path = tmp_path / "strict.pt"
    torch.save(
        {
            "schema_version": 2,
            "scope": "action_denoising_trajectory",
            "sampling_seed": 1,
            "frame_st_id": 0,
            "num_transitions": 1,
            "transitions": [_transition()],
            "replay_context": _replay_context(),
        },
        artifact_path,
    )
    group_path = _write_group(tmp_path, artifact_path)
    transformer = ToyTransformer()

    result = ActorReplayGrpoTrainer(
        ActorReplayTrainerConfig(
            groups_jsonl=group_path,
            output_dir=tmp_path / "train",
            steps=1,
            learning_rate=1e-3,
            clip_low=100.0,
            clip_high=100.0,
            device="cpu",
            dtype="float32",
            action_num_inference_steps=2,
        ),
        transformer=transformer,
    ).train()

    assert result.transition_count == 1
    assert result.trainable_param_count > 0
    assert (tmp_path / "train" / "checkpoint.pt").exists()
