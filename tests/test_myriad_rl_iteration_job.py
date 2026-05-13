from pathlib import Path


def test_rl_iteration_job_runs_collect_then_grpo_train():
    text = Path("jobs/myriad/40_rl_iteration_robotwin.sh").read_text(encoding="utf-8")

    assert "jobs/myriad/30_collect_grouped_rollouts_4gpu.sh" in text
    assert "jobs/myriad/31_train_denoising_grpo_robotwin.sh" in text
    assert 'GRPO_GROUPS_PATH="${RESULTS_ROOT}/groups/grpo_groups.jsonl"' in text
    assert 'GRPO_OUTPUT_DIR="${ITERATION_ROOT}/train"' in text
    assert 'RESULTS_ROOT="${ITERATION_ROOT}/rollout_collection"' in text
