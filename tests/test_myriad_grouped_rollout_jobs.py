from pathlib import Path


def test_four_gpu_grouped_rollout_job_builds_grpo_groups():
    text = Path("jobs/myriad/30_collect_grouped_rollouts_4gpu.sh").read_text()

    assert "tools/collect_robotwin_rollouts.py" in text
    assert "tools/build_grpo_groups.py" in text
    assert "--expected-group-size \"${GROUP_SIZE}\"" in text
    assert "--require-strict-artifacts" in text
    assert "--require-existing-artifacts" in text
    assert "--wait-for-artifacts-seconds 120" in text
    assert "--fail-on-validation-errors" in text
    assert "--out-manifest \"${RESULTS_ROOT}/groups/grpo_manifest.json\"" in text
    assert "tools/validate_grpo_dataset.py" in text
    assert "--out-summary \"${RESULTS_ROOT}/groups/grpo_dataset_validation.json\"" in text
    assert "--fail-on-error" in text
