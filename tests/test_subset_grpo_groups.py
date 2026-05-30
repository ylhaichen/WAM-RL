import json

import torch

from tools.subset_grpo_groups import subset_grpo_groups, write_outputs


def _sample(sample_idx, reward, artifact_count=3):
    return {
        "task": "move_stapler_pad",
        "group_id": "g0",
        "sample_idx": sample_idx,
        "reward": float(reward),
        "advantage": 99.0,
        "success": bool(reward > 0),
        "record_path": f"/tmp/rollout_{sample_idx}.json",
        "env_seed": 10000,
        "sampling_seed": 740000 + sample_idx,
        "strict_grpo_scope": "action_denoising_trajectory",
        "strict_grpo_artifact_count": artifact_count,
        "strict_grpo_artifact_paths": [f"/tmp/s{sample_idx}_{i}.pt" for i in range(artifact_count)],
    }


def _write_groups(path):
    groups = [
        {
            "group_id": "g0",
            "task": "move_stapler_pad",
            "group_size": 4,
            "reward_mean": 0.5,
            "reward_std": 0.5,
            "samples": [_sample(0, 0), _sample(1, 0), _sample(2, 1), _sample(3, 1)],
        },
        {
            "group_id": "g1",
            "task": "turn_switch",
            "group_size": 2,
            "reward_mean": 1.0,
            "reward_std": 0.0,
            "samples": [_sample(0, 1), _sample(1, 1)],
        },
    ]
    path.write_text("".join(json.dumps(group) + "\n" for group in groups), encoding="utf-8")


def test_subset_grpo_groups_selects_balanced_samples_and_recomputes_advantages(tmp_path):
    source = tmp_path / "groups.jsonl"
    _write_groups(source)

    groups, manifest = subset_grpo_groups(
        source,
        tasks={"move_stapler_pad"},
        max_groups=1,
        samples_per_reward=1,
        max_artifacts_per_sample=2,
        require_artifacts=True,
    )

    assert manifest["input_group_count"] == 2
    assert manifest["output_group_count"] == 1
    assert manifest["output_sample_count"] == 2
    assert manifest["output_artifact_ref_count"] == 4
    assert manifest["tasks"][0]["task"] == "move_stapler_pad"

    group = groups[0]
    assert group["group_id"] == "g0_subset"
    assert group["group_size"] == 2
    assert [sample["sample_idx"] for sample in group["samples"]] == [0, 2]
    assert [sample["advantage"] for sample in group["samples"]] == [-1.0, 1.0]
    assert [sample["strict_grpo_artifact_count"] for sample in group["samples"]] == [2, 2]


def test_subset_grpo_groups_can_preserve_advantages_and_write_manifest(tmp_path):
    source = tmp_path / "groups.jsonl"
    out_jsonl = tmp_path / "subset.jsonl"
    out_manifest = tmp_path / "manifest.json"
    _write_groups(source)

    groups, manifest = subset_grpo_groups(
        source,
        samples_per_reward=1,
        preserve_advantages=True,
        preserve_group_id=True,
    )
    write_outputs(groups, out_jsonl=out_jsonl, out_manifest=out_manifest, manifest=manifest)

    written_group = json.loads(out_jsonl.read_text().strip())
    written_manifest = json.loads(out_manifest.read_text())
    assert written_group["group_id"] == "g0"
    assert [sample["advantage"] for sample in written_group["samples"]] == [99.0, 99.0]
    assert written_manifest["output_jsonl"] == str(out_jsonl)
    assert written_manifest["skipped_unmixed_group_count"] == 1


def _write_replay_context_artifact(root, name, *, context_bytes):
    artifact = root / f"{name}.pt"
    context = root / f"{name}_context.pt"
    context.write_bytes(b"x" * context_bytes)
    torch.save({"schema_version": 2, "replay_context_path": context.name, "transitions": []}, artifact)
    return artifact


def test_subset_grpo_groups_trims_by_replay_context_budget_round_robin(tmp_path):
    source_dir = tmp_path / "artifacts"
    source_dir.mkdir()
    failure_a = _write_replay_context_artifact(source_dir, "failure_a", context_bytes=10)
    failure_b = _write_replay_context_artifact(source_dir, "failure_b", context_bytes=30)
    success_a = _write_replay_context_artifact(source_dir, "success_a", context_bytes=10)
    success_b = _write_replay_context_artifact(source_dir, "success_b", context_bytes=30)

    group = {
        "group_id": "g0",
        "task": "move_stapler_pad",
        "group_size": 2,
        "reward_mean": 0.5,
        "reward_std": 0.5,
        "samples": [
            {
                **_sample(0, 0, artifact_count=0),
                "strict_grpo_artifact_paths": [str(failure_a), str(failure_b)],
            },
            {
                **_sample(1, 1, artifact_count=0),
                "strict_grpo_artifact_paths": [str(success_a), str(success_b)],
            },
        ],
    }
    source = tmp_path / "groups.jsonl"
    source.write_text(json.dumps(group) + "\n", encoding="utf-8")

    groups, manifest = subset_grpo_groups(
        source,
        samples_per_reward=1,
        require_artifacts=True,
        max_replay_context_gb=20 / 1024**3,
    )

    assert manifest["output_group_count"] == 1
    assert manifest["output_artifact_ref_count"] == 2
    assert manifest["replay_context_budget"]["selected_replay_context_bytes"] == 20
    assert manifest["replay_context_budget"]["selected_unique_replay_context_count"] == 2
    assert manifest["replay_context_budget"]["skipped_artifact_ref_count_over_budget"] == 2
    assert [len(sample["strict_grpo_artifact_paths"]) for sample in groups[0]["samples"]] == [1, 1]


def test_subset_grpo_groups_drops_unmixed_group_when_budget_removes_required_sample(tmp_path):
    source_dir = tmp_path / "artifacts"
    source_dir.mkdir()
    failure = _write_replay_context_artifact(source_dir, "failure", context_bytes=10)
    success = _write_replay_context_artifact(source_dir, "success", context_bytes=10)
    group = {
        "group_id": "g0",
        "task": "move_stapler_pad",
        "group_size": 2,
        "reward_mean": 0.5,
        "reward_std": 0.5,
        "samples": [
            {
                **_sample(0, 0, artifact_count=0),
                "strict_grpo_artifact_paths": [str(failure)],
            },
            {
                **_sample(1, 1, artifact_count=0),
                "strict_grpo_artifact_paths": [str(success)],
            },
        ],
    }
    source = tmp_path / "groups.jsonl"
    source.write_text(json.dumps(group) + "\n", encoding="utf-8")

    groups, manifest = subset_grpo_groups(
        source,
        samples_per_reward=1,
        require_artifacts=True,
        max_replay_context_gb=10 / 1024**3,
    )

    assert groups == []
    assert manifest["output_group_count"] == 0
    assert manifest["skipped_unmixed_group_count"] == 1
    assert manifest["replay_context_budget"]["selected_replay_context_bytes"] == 0
