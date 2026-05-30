import json

from tools.collect_robotwin_rollouts import iter_rollout_records


def test_collect_rollouts_preserves_group_and_strict_fields(tmp_path):
    rollout_dir = tmp_path / "rollouts" / "open_microwave"
    rollout_dir.mkdir(parents=True)
    record_path = rollout_dir / "episode_000000_seed_10000_sample_000.json"
    record_path.write_text(
        json.dumps(
            {
                "task_name": "open_microwave",
                "seed": 10000,
                "episode_index": 0,
                "success": True,
                "reward": 1.0,
                "prompt": "Open the microwave door.",
                "group_id": "open_microwave_seed10000_group000000_abcd",
                "sample_idx": 0,
                "group_size": 4,
                "sampling_seed": 10000000000,
                "actions_path": "/tmp/actions.npy",
                "initial_obs_path": "/tmp/initial_obs.npy",
                "visualization_path": "/tmp/video.mp4",
                "server_action_paths": ["/tmp/server/actions_0.pt"],
                "server_latent_paths": ["/tmp/server/latents_0.pt"],
                "strict_grpo_ready": True,
                "strict_grpo_scope": "action_denoising_trajectory",
                "strict_grpo_artifact_count": 1,
                "strict_grpo_artifact_paths": ["/tmp/server/strict_grpo_0.pt"],
                "strict_grpo_replay_context_count": 1,
                "strict_grpo_replay_context_paths": ["/tmp/server/strict_grpo_replay_context_0.pt"],
                "strict_grpo_replay_context_tensor_bytes": [1234],
                "strict_grpo_replay_context_total_tensor_bytes": 1234,
                "strict_grpo_replay_context_max_gb": 5.0,
                "strict_grpo_capture_chunk_indices": [0],
                "strict_grpo_capture_chunk_stride": 2,
                "strict_grpo_capture_max_chunks": 4,
                "action_count": 12,
                "obs_count": 4,
                "take_action_cnt": 48,
                "step_lim": 200,
            }
        ),
        encoding="utf-8",
    )

    records = list(iter_rollout_records(tmp_path))

    assert len(records) == 1
    record = records[0]
    assert record.group_id == "open_microwave_seed10000_group000000_abcd"
    assert record.sample_idx == 0
    assert record.group_size == 4
    assert record.sampling_seed == 10000000000
    assert record.initial_obs_path == "/tmp/initial_obs.npy"
    assert record.strict_grpo_ready is True
    assert record.strict_grpo_scope == "action_denoising_trajectory"
    assert record.strict_grpo_artifact_count == 1
    assert record.strict_grpo_artifact_paths == ["/tmp/server/strict_grpo_0.pt"]
    assert record.strict_grpo_replay_context_count == 1
    assert record.strict_grpo_replay_context_paths == ["/tmp/server/strict_grpo_replay_context_0.pt"]
    assert record.strict_grpo_replay_context_tensor_bytes == [1234]
    assert record.strict_grpo_replay_context_total_tensor_bytes == 1234
    assert record.strict_grpo_replay_context_max_gb == 5.0
    assert record.strict_grpo_capture_chunk_indices == [0]
    assert record.strict_grpo_capture_chunk_stride == 2
    assert record.strict_grpo_capture_max_chunks == 4
