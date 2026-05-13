from pathlib import Path


def test_single_gpu_collect_job_has_safe_defaults():
    script = Path("jobs/myriad/30_collect_grouped_rollouts_1gpu.sh")
    text = script.read_text()

    assert "#$" not in text
    assert "container_exec_gpu" not in text
    assert "qsub" not in text
    assert "PYTHON_BIN=" in text
    assert 'NUM_GPUS="${NUM_GPUS:-1}"' in text
    assert 'GROUP_SIZE="${GROUP_SIZE:-2}"' in text
    assert 'GROUPS_PER_TASK="${GROUPS_PER_TASK:-1}"' in text
    assert 'SELECTED_TASKS="${TASK_NAMES:-hanging_mug}"' in text
    assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"' in text
    assert 'bash evaluation/robotwin/launch_client_multigpus.sh "${RESULTS_ROOT}" 0 "${seed}" 1' in text
    assert "tools/collect_robotwin_rollouts.py" in text
