import json
from pathlib import Path

import torch

from wan_va.rl.actor_replay import (
    ActorReplayGrpoTrainer,
    ActorReplayTrainerConfig,
    MissingReplayContextError,
    build_replay_context,
    check_replay_context_tensor_budget,
    count_actor_replay_transition_items,
    iter_actor_replay_examples,
    load_actor_replay_checkpoint_into_transformer,
    snapshot_transformer_cache,
    tensor_tree_nbytes,
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


class ZeroGradientTransformer(ToyTransformer):
    def forward(self, input_dict, update_cache=0, cache_name="pos", action_mode=True):
        return super().forward(input_dict, update_cache, cache_name, action_mode) * 0.0


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


def _compact_transition():
    transition = _transition()
    transition["replay_input"] = {
        "timesteps": transition["replay_input"]["timesteps"],
    }
    return transition


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


def test_build_replay_context_prunes_unused_cfg_branch_for_action_scale_one():
    transformer = ToyTransformer()
    transformer.blocks[0].attn1.attn_caches["pos"] = {
        "k": torch.arange(2.0).reshape(2, 1, 1, 1),
        "v": torch.arange(2.0, 4.0).reshape(2, 1, 1, 1),
        "id": torch.arange(2, dtype=torch.long),
        "mask": torch.tensor([True, False]),
    }

    context = build_replay_context(
        transformer=transformer,
        cache_name="pos",
        action_input_template={
            "grid_id": torch.zeros(1, 3, dtype=torch.long),
            "text_emb": torch.zeros(1, 1, 1),
        },
        negative_prompt_embeds=torch.ones(1, 1, 1),
        use_cfg=True,
        action_guidance_scale=1.0,
        action_num_inference_steps=10,
        frame_chunk_size=1,
    )

    cache = context["transformer_cache"][0]
    assert context["use_cfg"] is False
    assert context["negative_text_emb"] is None
    assert context["cfg_pruned_to_conditional"] is True
    assert cache["k"].shape[0] == 1
    assert cache["v"].shape[0] == 1
    assert cache["id"].shape[0] == 2
    assert cache["mask"].shape[0] == 2
    assert cache["k"].item() == 0.0
    assert cache["v"].item() == 2.0
    assert cache["k"].untyped_storage().nbytes() == (
        cache["k"].numel() * cache["k"].element_size()
    )


def test_snapshot_transformer_cache_can_prune_kv_branch_during_clone():
    transformer = ToyTransformer()
    transformer.blocks[0].attn1.attn_caches["pos"] = {
        "k": torch.arange(8.0).reshape(2, 4, 1, 1),
        "v": torch.arange(8.0, 16.0).reshape(2, 4, 1, 1),
        "id": torch.arange(2, dtype=torch.long),
        "mask": torch.tensor([True, False]),
    }

    cache = snapshot_transformer_cache(transformer, kv_batch_index=1)[0]

    assert cache["k"].shape[0] == 1
    assert cache["v"].shape[0] == 1
    assert cache["id"].shape[0] == 2
    assert cache["mask"].shape[0] == 2
    assert torch.equal(cache["k"], torch.arange(4.0, 8.0).reshape(1, 4, 1, 1))
    assert cache["k"].untyped_storage().nbytes() == cache["k"].numel() * cache["k"].element_size()


def test_replay_context_tensor_budget_fails_before_oversized_save():
    context = {
        "transformer_cache": [
            {
                "k": torch.zeros(4, dtype=torch.float32),
                "v": torch.zeros(2, dtype=torch.float16),
            }
        ],
        "metadata": "ignored",
    }

    assert tensor_tree_nbytes(context) == 20
    assert check_replay_context_tensor_budget(context, None) == 20
    assert check_replay_context_tensor_budget(context, 1e-6) == 20
    try:
        check_replay_context_tensor_budget(context, 1e-12, label="toy_context")
    except ValueError as exc:
        assert "toy_context tensor storage" in str(exc)
        assert "strict_grpo_replay_context_max_gb" in str(exc)
    else:
        raise AssertionError("expected replay-context budget failure")


def test_build_replay_context_keeps_cfg_branch_for_action_guidance():
    transformer = ToyTransformer()
    transformer.blocks[0].attn1.attn_caches["pos"] = {
        "k": torch.zeros(2, 1, 1, 1),
        "v": torch.zeros(2, 1, 1, 1),
    }

    context = build_replay_context(
        transformer=transformer,
        cache_name="pos",
        action_input_template={
            "grid_id": torch.zeros(1, 3, dtype=torch.long),
            "text_emb": torch.zeros(1, 1, 1),
        },
        negative_prompt_embeds=torch.ones(1, 1, 1),
        use_cfg=True,
        action_guidance_scale=5.0,
        action_num_inference_steps=10,
        frame_chunk_size=1,
    )

    assert context["use_cfg"] is True
    assert context["negative_text_emb"] is not None
    assert context["cfg_pruned_to_conditional"] is False
    assert context["transformer_cache"][0]["k"].shape[0] == 2


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
            "transitions": [_compact_transition()],
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
    metrics = json.loads((tmp_path / "train" / "metrics.json").read_text())
    step_metrics = metrics["history"][0]
    assert step_metrics["param_update_norm"] > 0.0
    assert step_metrics["param_update_max"] > 0.0
    assert step_metrics["param_update_param_count"] == result.trainable_param_count


def test_load_actor_replay_checkpoint_into_transformer_loads_trainable_state(tmp_path):
    source = ToyTransformer()
    with torch.no_grad():
        source.action_embedder.weight.fill_(2.0)
        source.action_embedder.bias.fill_(3.0)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "trainable_state_dict": {
                "action_embedder.weight": source.action_embedder.weight.detach().clone(),
                "action_embedder.bias": source.action_embedder.bias.detach().clone(),
            }
        },
        checkpoint_path,
    )
    target = ToyTransformer()

    summary = load_actor_replay_checkpoint_into_transformer(target, checkpoint_path)

    assert summary["tensor_count"] == 2
    assert summary["param_count"] == 2
    assert torch.equal(target.action_embedder.weight, source.action_embedder.weight)
    assert torch.equal(target.action_embedder.bias, source.action_embedder.bias)


def test_load_actor_replay_checkpoint_rejects_unexpected_keys(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"trainable_state_dict": {"not_a_real_param": torch.ones(1)}}, checkpoint_path)

    try:
        load_actor_replay_checkpoint_into_transformer(ToyTransformer(), checkpoint_path)
    except ValueError as exc:
        assert "unexpected transformer keys" in str(exc)
    else:
        raise AssertionError("expected unexpected-key ValueError")


def test_actor_replay_trainer_can_use_mean_logprob_reduction(tmp_path):
    transition = _compact_transition()
    transition["old_logprob_sum"] = torch.tensor([100.0])
    transition["old_logprob_mean"] = torch.tensor([0.0])
    transition["old_logprob_count"] = torch.tensor([100])
    artifact_path = tmp_path / "strict.pt"
    torch.save(
        {
            "schema_version": 2,
            "scope": "action_denoising_trajectory",
            "sampling_seed": 1,
            "frame_st_id": 0,
            "num_transitions": 1,
            "transitions": [transition],
            "replay_context": _replay_context(),
        },
        artifact_path,
    )
    group_path = _write_group(tmp_path, artifact_path)

    result = ActorReplayGrpoTrainer(
        ActorReplayTrainerConfig(
            groups_jsonl=group_path,
            output_dir=tmp_path / "train_mean",
            steps=1,
            learning_rate=1e-3,
            clip_low=100.0,
            clip_high=100.0,
            device="cpu",
            dtype="float32",
            action_num_inference_steps=2,
            logprob_reduction="mean",
        ),
        transformer=ToyTransformer(),
    ).train()

    assert result.transition_count == 1
    checkpoint = torch.load(tmp_path / "train_mean" / "checkpoint.pt", map_location="cpu")
    assert checkpoint["config"]["logprob_reduction"] == "mean"


def test_actor_replay_trainer_rejects_unknown_logprob_reduction(tmp_path):
    artifact_path = tmp_path / "strict.pt"
    torch.save(
        {
            "schema_version": 2,
            "scope": "action_denoising_trajectory",
            "sampling_seed": 1,
            "frame_st_id": 0,
            "num_transitions": 1,
            "transitions": [_compact_transition()],
            "replay_context": _replay_context(),
        },
        artifact_path,
    )
    group_path = _write_group(tmp_path, artifact_path)

    try:
        ActorReplayGrpoTrainer(
            ActorReplayTrainerConfig(
                groups_jsonl=group_path,
                output_dir=tmp_path / "train_bad_reduction",
                steps=1,
                device="cpu",
                dtype="float32",
                action_num_inference_steps=2,
                logprob_reduction="median",
            ),
            transformer=ToyTransformer(),
        )
    except ValueError as exc:
        assert "logprob_reduction" in str(exc)
    else:
        raise AssertionError("expected logprob_reduction ValueError")


def test_actor_replay_trainer_can_use_logprob_std_floor(tmp_path):
    transition = _compact_transition()
    transition["transition_std"] = torch.tensor(0.01)
    artifact_path = tmp_path / "strict.pt"
    torch.save(
        {
            "schema_version": 2,
            "scope": "action_denoising_trajectory",
            "sampling_seed": 1,
            "frame_st_id": 0,
            "num_transitions": 1,
            "transitions": [transition],
            "replay_context": _replay_context(),
        },
        artifact_path,
    )
    group_path = _write_group(tmp_path, artifact_path)

    result = ActorReplayGrpoTrainer(
        ActorReplayTrainerConfig(
            groups_jsonl=group_path,
            output_dir=tmp_path / "train_std_floor",
            steps=1,
            learning_rate=1e-3,
            clip_low=100.0,
            clip_high=100.0,
            device="cpu",
            dtype="float32",
            action_num_inference_steps=2,
            logprob_reduction="mean",
            logprob_std_floor=0.1,
        ),
        transformer=ToyTransformer(),
    ).train()

    assert result.transition_count == 1
    checkpoint = torch.load(tmp_path / "train_std_floor" / "checkpoint.pt", map_location="cpu")
    assert checkpoint["config"]["logprob_std_floor"] == 0.1


def test_actor_replay_trainer_writes_zero_gradient_diagnostics(tmp_path):
    artifact_path = tmp_path / "strict.pt"
    torch.save(
        {
            "schema_version": 2,
            "scope": "action_denoising_trajectory",
            "sampling_seed": 1,
            "frame_st_id": 0,
            "num_transitions": 1,
            "transitions": [_compact_transition()],
            "replay_context": _replay_context(),
        },
        artifact_path,
    )
    group_path = _write_group(tmp_path, artifact_path)
    output_dir = tmp_path / "train"

    try:
        ActorReplayGrpoTrainer(
            ActorReplayTrainerConfig(
                groups_jsonl=group_path,
                output_dir=output_dir,
                steps=1,
                learning_rate=1e-3,
                clip_low=100.0,
                clip_high=100.0,
                device="cpu",
                dtype="float32",
                action_num_inference_steps=2,
            ),
            transformer=ZeroGradientTransformer(),
        ).train()
    except ValueError as exc:
        assert "zero gradients" in str(exc)
        assert "grad_norm=0" in str(exc)
    else:
        raise AssertionError("expected zero-gradient ValueError")

    diagnostics = json.loads((output_dir / "failure_diagnostics.json").read_text())
    assert diagnostics["metrics"]["grad_norm"] == 0.0
    assert diagnostics["transition_count"] == 1


def test_actor_replay_accepts_external_replay_context(tmp_path):
    artifact_path = tmp_path / "strict.pt"
    context_path = tmp_path / "context.pt"
    torch.save(_replay_context(), context_path)
    torch.save(
        {
            "schema_version": 2,
            "scope": "action_denoising_trajectory",
            "sampling_seed": 1,
            "frame_st_id": 0,
            "num_transitions": 1,
            "transitions": [_compact_transition()],
            "replay_context_path": "context.pt",
        },
        artifact_path,
    )
    group_path = _write_group(tmp_path, artifact_path)

    examples = list(iter_actor_replay_examples(group_path))

    assert len(examples) == 1
    assert examples[0].replay_context["action_num_inference_steps"] == 2


def test_actor_replay_count_does_not_load_external_replay_context(tmp_path):
    artifact_path = tmp_path / "strict.pt"
    context_path = tmp_path / "context.pt"
    torch.save(_replay_context(), context_path)
    torch.save(
        {
            "schema_version": 2,
            "scope": "action_denoising_trajectory",
            "sampling_seed": 1,
            "frame_st_id": 0,
            "num_transitions": 1,
            "transitions": [_compact_transition()],
            "replay_context_path": "context.pt",
        },
        artifact_path,
    )
    group_path = _write_group(tmp_path, artifact_path)
    loaded_paths = []

    def loader(path):
        loaded_paths.append(path.name)
        if path.name == "context.pt":
            raise AssertionError("counting should not load replay_context")
        return torch.load(path, map_location="cpu")

    assert count_actor_replay_transition_items(group_path, loader=loader) == 1
    assert loaded_paths == ["strict.pt"]
