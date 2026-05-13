from pathlib import Path


def test_grpo_train_job_validates_dataset_and_runs_smoke_trainer():
    text = Path("jobs/myriad/31_train_denoising_grpo_robotwin.sh").read_text(encoding="utf-8")

    assert "tools/validate_grpo_dataset.py" in text
    assert "--fail-on-error" in text
    assert "tools/train_offline_grpo_smoke.py" in text
    assert "--groups-jsonl \"${GRPO_GROUPS_PATH}\"" in text
    assert "--output-dir \"${GRPO_OUTPUT_DIR}\"" in text
    assert 'GRPO_GROUPS_PATH="${GRPO_GROUPS_PATH:-${RESULTS_ROOT}/groups/grpo_groups.jsonl}"' in text
