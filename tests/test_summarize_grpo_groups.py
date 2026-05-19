import json
import subprocess
import sys

from tools.summarize_grpo_groups import format_markdown, summarize_groups


def test_summarize_grpo_groups_counts_task_samples_and_transitions(tmp_path):
    path = tmp_path / "grpo_groups.jsonl"
    rows = [
        {
            "group_id": "open_microwave_seed1_group000000",
            "task": "open_microwave",
            "group_size": 2,
            "reward_mean": 0.5,
            "reward_std": 0.5,
            "samples": [
                {
                    "sample_idx": 0,
                    "reward": 1.0,
                    "success": True,
                    "strict_grpo_artifact_paths": ["a.pt", "b.pt"],
                },
                {
                    "sample_idx": 1,
                    "reward": 0.0,
                    "success": False,
                    "strict_grpo_artifact_paths": ["c.pt"],
                },
            ],
        },
        {
            "group_id": "turn_switch_seed1_group000000",
            "task": "turn_switch",
            "group_size": 2,
            "reward_mean": 0.5,
            "reward_std": 0.5,
            "samples": [
                {
                    "sample_idx": 0,
                    "reward": 1.0,
                    "success": True,
                    "strict_grpo_artifact_paths": [],
                },
                {
                    "sample_idx": 1,
                    "reward": 1.0,
                    "success": True,
                    "strict_grpo_artifact_paths": ["d.pt"],
                },
            ],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = summarize_groups([path])

    assert summary.group_count == 2
    assert summary.sample_count == 4
    assert summary.success_count == 3
    assert summary.failure_count == 1
    assert summary.transition_count == 4
    by_task = {item.task: item for item in summary.tasks}
    assert by_task["open_microwave"].sample_count == 2
    assert by_task["open_microwave"].transition_count == 3
    assert by_task["open_microwave"].balance_rate == 0.5
    assert by_task["turn_switch"].success_count == 2


def test_summarize_grpo_groups_formats_markdown(tmp_path):
    path = tmp_path / "grpo_groups.jsonl"
    path.write_text(
        json.dumps(
            {
                "group_id": "g0",
                "task": "open_microwave",
                "group_size": 1,
                "reward_mean": 1.0,
                "reward_std": 0.0,
                "samples": [{"sample_idx": 0, "reward": 1.0, "strict_grpo_artifact_paths": ["a.pt"]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    text = format_markdown(summarize_groups([path]))

    assert "# GRPO Group Summary" in text
    assert "| open_microwave | 1 | 1 | 1 | 0 | 1.000 | 1 | 1.000 | 0.000 | 0.000 |" in text


def test_summarize_grpo_groups_script_entrypoint(tmp_path):
    path = tmp_path / "grpo_groups.jsonl"
    path.write_text(
        json.dumps(
            {
                "group_id": "g0",
                "task": "open_microwave",
                "group_size": 1,
                "reward_mean": 1.0,
                "reward_std": 0.0,
                "samples": [{"sample_idx": 0, "reward": 1.0, "strict_grpo_artifact_paths": ["a.pt"]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "tools/summarize_grpo_groups.py", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert '"group_count": 1' in result.stdout
