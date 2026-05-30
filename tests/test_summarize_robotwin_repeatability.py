import json
import subprocess
import sys

from tools.summarize_robotwin_repeatability import summarize_repeatability


def _episode(root, task, index, seed, success, sampling_seed=12345, prompt_index=0):
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
                "action_count": 100 + index,
                "take_action_cnt": 90 + index,
                "step_lim": 400,
                "sampling_seed": sampling_seed,
                "prompt_index": prompt_index,
                "prompt": "move the stapler pad",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_summarize_repeatability_reports_flipped_and_stable_keys(tmp_path):
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    _episode(run_a, "move_stapler_pad", 0, 10000, True)
    _episode(run_b, "move_stapler_pad", 0, 10000, True)
    _episode(run_a, "move_stapler_pad", 1, 10001, True, sampling_seed=12346)
    _episode(run_b, "move_stapler_pad", 1, 10001, False, sampling_seed=12346)

    summary = summarize_repeatability([("a", run_a), ("b", run_b)])

    assert summary["run_count"] == 2
    assert summary["matched_episode_count"] == 2
    assert summary["stable_success_count"] == 1
    assert summary["stable_failure_count"] == 0
    assert summary["flipped_count"] == 1
    assert summary["flip_rate"] == 0.5
    assert [item["status"] for item in summary["episodes"]] == ["stable_success", "flipped"]


def test_summarize_repeatability_rejects_zero_matches_by_default(tmp_path):
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    _episode(run_a, "move_stapler_pad", 0, 10000, True, sampling_seed=12345)
    _episode(run_b, "move_stapler_pad", 0, 20000, True, sampling_seed=22345)

    try:
        summarize_repeatability([("a", run_a), ("b", run_b)])
    except ValueError as exc:
        assert "only matched 0 episodes" in str(exc)
    else:
        raise AssertionError("expected zero-match ValueError")


def test_summarize_robotwin_repeatability_cli_writes_outputs(tmp_path):
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    out_json = tmp_path / "repeatability.json"
    out_csv = tmp_path / "repeatability.csv"
    out_md = tmp_path / "repeatability.md"
    _episode(run_a, "move_stapler_pad", 0, 10000, True)
    _episode(run_b, "move_stapler_pad", 0, 10000, False)

    result = subprocess.run(
        [
            sys.executable,
            "tools/summarize_robotwin_repeatability.py",
            "--run",
            f"a={run_a}",
            "--run",
            f"b={run_b}",
            "--out-json",
            str(out_json),
            "--out-csv",
            str(out_csv),
            "--out-markdown",
            str(out_md),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    written = json.loads(out_json.read_text(encoding="utf-8"))
    assert written["flipped_count"] == 1
    assert "status" in out_csv.read_text(encoding="utf-8")
    assert "RoboTwin Eval Repeatability Summary" in out_md.read_text(encoding="utf-8")
