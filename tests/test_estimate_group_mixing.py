import json
import subprocess
import sys

import pytest

from tools.estimate_group_mixing import estimate_group_mixing, mixed_group_probability


def test_mixed_group_probability_excludes_all_success_and_all_failure():
    assert mixed_group_probability(0.5, 4) == pytest.approx(0.875)
    assert mixed_group_probability(0.0, 4) == 0.0
    assert mixed_group_probability(1.0, 4) == 0.0


def test_estimate_group_mixing_reports_per_task_rows():
    report = estimate_group_mixing(
        {
            "success_rate": 0.5,
            "tasks": [
                {"task": "hanging_mug", "success_rate": 0.34375},
                {"task": "move_stapler_pad", "success_rate": 0.625},
            ],
        },
        [4, 8],
    )

    rows = {row["task"]: row for row in report["rows"]}
    assert rows["overall"]["by_group_size"]["4"]["mixed_probability"] == pytest.approx(0.875)
    assert rows["hanging_mug"]["by_group_size"]["8"]["mixed_probability"] > rows["hanging_mug"][
        "by_group_size"
    ]["4"]["mixed_probability"]
    assert rows["move_stapler_pad"]["by_group_size"]["4"]["expected_attempts_per_mixed_group"] > 1.0


def test_estimate_group_mixing_cli_from_summary(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"success_rate": 0.5, "tasks": [{"task": "turn_switch", "success_rate": 0.6}]}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "tools/estimate_group_mixing.py",
            "--summary",
            str(summary),
            "--group-sizes",
            "4",
            "8",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["group_sizes"] == [4, 8]
    assert report["rows"][1]["task"] == "turn_switch"
