from pathlib import Path

from evaluation.robotwin.rollout_logging import (
    build_group_id,
    build_rollout_metadata,
)


def test_build_group_id_is_stable_and_path_safe():
    group_id = build_group_id(
        task_name="open_microwave",
        env_seed=10042,
        prompt="Open the microwave door.",
        group_index=3,
    )

    assert group_id.startswith("open_microwave_seed10042_group000003_")
    assert " " not in group_id
    assert "." not in group_id


def test_build_group_id_can_use_stable_prompt_key_instead_of_prompt_text():
    first = build_group_id(
        task_name="open_microwave",
        env_seed=10042,
        prompt="Open the microwave door.",
        group_index=3,
        prompt_key="prompt0",
    )
    second = build_group_id(
        task_name="open_microwave",
        env_seed=10042,
        prompt="A slightly different rendered instruction.",
        group_index=3,
        prompt_key="prompt0",
    )

    assert first == second


def test_build_rollout_metadata_includes_pseudo_and_strict_artifacts(tmp_path):
    action_path = tmp_path / "episode_000003_actions.npy"
    action_path.write_bytes(b"npy")
    video_path = tmp_path / "episode_000003.mp4"
    video_path.write_bytes(b"mp4")
    initial_obs_path = tmp_path / "episode_000003_initial_obs.npy"
    initial_obs_path.write_bytes(b"npy")
    server_action = tmp_path / "server" / "actions_0.pt"
    server_action.parent.mkdir()
    server_action.write_bytes(b"pt")
    server_latent = tmp_path / "server" / "latents_0.pt"
    server_latent.write_bytes(b"pt")
    strict_path = tmp_path / "server" / "strict_grpo_0.pt"
    strict_path.write_bytes(b"pt")

    data = build_rollout_metadata(
        task_name="open_microwave",
        episode_index=3,
        env_seed=10042,
        prompt="Open the microwave door.",
        success=True,
        action_count=12,
        obs_count=4,
        take_action_cnt=48,
        step_lim=200,
        executed_actions_path=action_path,
        visualization_path=video_path,
        initial_obs_path=initial_obs_path,
        run_id="rl_debug_0001",
        policy_checkpoint="/ckpts/actor",
        reference_checkpoint="/ckpts/ref",
        group_id="open_microwave_seed10042_group000003_abcd1234",
        sample_idx=2,
        group_size=4,
        sampling_seed=730002,
        video_guidance_scale=5.0,
        action_guidance_scale=1.0,
        action_num_inference_steps=50,
        server_action_paths=[server_action],
        server_latent_paths=[server_latent],
        strict_grpo_artifact_paths=[strict_path],
        strict_grpo_scope="action_denoising_trajectory",
    )

    assert data["reward"] == 1.0
    assert data["env_seed"] == 10042
    assert data["seed"] == 10042
    assert data["sample_idx"] == 2
    assert data["group_size"] == 4
    assert data["strict_grpo_ready"] is True
    assert data["initial_obs_path"] == str(initial_obs_path)
    assert data["strict_grpo_scope"] == "action_denoising_trajectory"
    assert data["strict_grpo_artifact_count"] == 1
    assert data["server_action_paths"] == [str(server_action)]
    assert data["server_latent_paths"] == [str(server_latent)]
    assert data["strict_grpo_artifact_paths"] == [str(strict_path)]
