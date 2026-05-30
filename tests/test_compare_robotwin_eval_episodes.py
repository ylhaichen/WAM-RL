import json

from tools.compare_robotwin_eval_episodes import compare_eval_runs, write_comparison_csv


def _episode(root, task, index, seed, success, sampling_seed=740000, prompt_index=0):
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
                "prompt": "move the stapler",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_compare_eval_runs_reports_matched_improvements(tmp_path):
    baseline = tmp_path / "baseline"
    actor = tmp_path / "actor"
    _episode(baseline, "move_stapler_pad", 0, 10000, False)
    _episode(baseline, "move_stapler_pad", 1, 10001, True)
    _episode(actor, "move_stapler_pad", 0, 10000, True)
    _episode(actor, "move_stapler_pad", 1, 10001, False)
    _episode(actor, "move_stapler_pad", 2, 10002, True)

    comparison = compare_eval_runs([("baseline", baseline), ("actor", actor)])

    assert comparison["matched_episode_count"] == 2
    assert comparison["run_summaries"]["baseline"]["success_count"] == 1
    assert comparison["run_summaries"]["actor"]["unmatched_unique_key_count"] == 1
    assert comparison["pairwise_vs_first"] == [
        {
            "baseline": "baseline",
            "candidate": "actor",
            "matched_episode_count": 2,
            "improved_count": 1,
            "regressed_count": 1,
            "same_success_count": 0,
            "same_failure_count": 0,
            "net_improvement_count": 0,
        }
    ]


def test_compare_eval_runs_excludes_duplicate_keys_and_writes_csv(tmp_path):
    baseline = tmp_path / "baseline"
    actor = tmp_path / "actor"
    csv_path = tmp_path / "comparison.csv"
    _episode(baseline, "move_stapler_pad", 0, 10000, True)
    _episode(baseline, "move_stapler_pad", 1, 10000, False)
    _episode(actor, "move_stapler_pad", 0, 10000, True)

    comparison = compare_eval_runs([("baseline", baseline), ("actor", actor)])
    write_comparison_csv(csv_path, comparison)

    assert comparison["matched_episode_count"] == 0
    assert comparison["run_summaries"]["baseline"]["duplicate_key_count"] == 1
    assert csv_path.read_text().startswith("task,seed,prompt_index,sampling_seed")
