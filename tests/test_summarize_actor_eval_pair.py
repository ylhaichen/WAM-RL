import json

from tools.summarize_actor_eval_pair import summarize_eval_pair


def _res(root, task, succ, total):
    path = root / "metrics" / task / "res.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"succ_num": succ, "total_num": total}) + "\n", encoding="utf-8")


def _episode(
    root,
    task,
    index,
    seed,
    success,
    sampling_seed=12345,
    prompt_index=0,
    run_id="eval_pair_test",
    policy_checkpoint="/tmp/policy.pt",
    reference_checkpoint="/tmp/reference",
    action_num_inference_steps=10,
):
    path = root / "rollouts" / task / f"episode_{index}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "task": task,
                "episode_index": index,
                "env_seed": seed,
                "planned_seed": seed,
                "success": success,
                "action_count": 10 + index,
                "take_action_cnt": 9 + index,
                "step_lim": 400,
                "sampling_seed": sampling_seed,
                "prompt_index": prompt_index,
                "prompt": "move the stapler pad",
                "run_id": run_id,
                "policy_checkpoint": policy_checkpoint,
                "reference_checkpoint": reference_checkpoint,
                "action_num_inference_steps": action_num_inference_steps,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_summarize_eval_pair_writes_summaries_and_comparison(tmp_path):
    baseline = tmp_path / "baseline"
    actor = tmp_path / "actor"
    out = tmp_path / "pair"
    _res(baseline, "move_stapler_pad", 1, 2)
    _res(actor, "move_stapler_pad", 2, 2)
    _episode(baseline, "move_stapler_pad", 0, 10000, False)
    _episode(baseline, "move_stapler_pad", 1, 10001, True)
    _episode(actor, "move_stapler_pad", 0, 10000, True)
    _episode(actor, "move_stapler_pad", 1, 10001, True)

    summary = summarize_eval_pair(baseline, actor, out)

    assert summary["matched_episode_count"] == 2
    assert summary["run_provenance"]["baseline"]["policy_checkpoint"] == ["/tmp/policy.pt"]
    assert summary["run_provenance"]["actor"]["reference_checkpoint"] == ["/tmp/reference"]
    assert summary["run_provenance"]["actor"]["action_num_inference_steps"] == [10]
    assert summary["pairwise_vs_first"][0]["improved_count"] == 1
    assert summary["pairwise_vs_first"][0]["regressed_count"] == 0
    assert (out / "baseline_summary.csv").exists()
    assert (out / "actor_summary.csv").exists()
    assert json.loads((out / "comparison.json").read_text())["matched_episode_count"] == 2
    summary_md = (out / "summary.md").read_text()
    assert "## Run Provenance" in summary_md
    assert "policy_checkpoint: `/tmp/policy.pt`" in summary_md
    assert "Tiny evals are smoke checks only" in summary_md


def test_summarize_eval_pair_rejects_zero_matched_episodes_by_default(tmp_path):
    baseline = tmp_path / "baseline"
    actor = tmp_path / "actor"
    out = tmp_path / "pair"
    _res(baseline, "move_stapler_pad", 1, 1)
    _res(actor, "move_stapler_pad", 1, 1)
    _episode(baseline, "move_stapler_pad", 0, 10000, True, sampling_seed=12345)
    _episode(actor, "move_stapler_pad", 0, 20000, True, sampling_seed=22345)

    try:
        summarize_eval_pair(baseline, actor, out)
    except ValueError as exc:
        assert "only matched 0 episodes" in str(exc)
        assert "Check SEED" in str(exc)
    else:
        raise AssertionError("expected zero-match ValueError")


def test_summarize_eval_pair_can_allow_aggregate_only_inspection(tmp_path):
    baseline = tmp_path / "baseline"
    actor = tmp_path / "actor"
    out = tmp_path / "pair"
    _res(baseline, "move_stapler_pad", 1, 1)
    _res(actor, "move_stapler_pad", 1, 1)
    _episode(baseline, "move_stapler_pad", 0, 10000, True, sampling_seed=12345)
    _episode(actor, "move_stapler_pad", 0, 20000, True, sampling_seed=22345)

    summary = summarize_eval_pair(baseline, actor, out, min_matched_episodes=0)

    assert summary["matched_episode_count"] == 0
    assert (out / "comparison.json").exists()
