from pathlib import Path


def test_single_gpu_collect_job_has_safe_defaults():
    script = Path("jobs/myriad/30_collect_grouped_rollouts_1gpu.sh")
    text = script.read_text()

    assert "#$" not in text
    assert "container_exec_gpu" not in text
    assert "qsub" not in text
    assert "docker run" in text
    assert 'DOCKER_IMAGE="${DOCKER_IMAGE:-pytorch/pytorch:2.9.0-cuda12.6-cudnn9-devel}"' in text
    assert 'DOCKER_GPUS="${DOCKER_GPUS:-device=${CUDA_VISIBLE_DEVICES}}"' in text
    assert "--gpus" in text
    assert "--network host" in text
    assert "--ipc host" in text
    assert 'DOCKER_RUN_AS_USER="${DOCKER_RUN_AS_USER:-1}"' in text
    assert "PYTHON_BIN=" in text
    assert 'NUM_GPUS="${NUM_GPUS:-1}"' in text
    assert 'GROUP_SIZE="${GROUP_SIZE:-2}"' in text
    assert 'GROUPS_PER_TASK="${GROUPS_PER_TASK:-1}"' in text
    assert 'SELECTED_TASKS="${TASK_NAMES:-hanging_mug}"' in text
    assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"' in text
    assert 'STRICT_GRPO_CAPTURE_SCOPE="${STRICT_GRPO_CAPTURE_SCOPE:-action_denoising_trajectory}"' in text
    assert 'strict_grpo_capture_scope="${STRICT_GRPO_CAPTURE_SCOPE}"' in text
    assert 'STRICT_GRPO_CAPTURE_CHUNK_STRIDE="${STRICT_GRPO_CAPTURE_CHUNK_STRIDE:-1}"' in text
    assert 'STRICT_GRPO_CAPTURE_MAX_CHUNKS="${STRICT_GRPO_CAPTURE_MAX_CHUNKS:-0}"' in text
    assert 'strict_grpo_capture_chunk_stride="${STRICT_GRPO_CAPTURE_CHUNK_STRIDE}"' in text
    assert 'strict_grpo_capture_max_chunks="${STRICT_GRPO_CAPTURE_MAX_CHUNKS}"' in text
    assert 'bash evaluation/robotwin/launch_client_multigpus.sh "${RESULTS_ROOT}" 0 "${seed}" 1' in text
    assert "tools/collect_robotwin_rollouts.py" in text
    assert "tools/build_grpo_groups.py" in text
    assert "--expected-group-size \"${GROUP_SIZE}\"" in text
    assert "--require-existing-artifacts" in text
    assert "--wait-for-artifacts-seconds 120" in text
    assert "--fail-on-validation-errors" in text
    assert "--out-manifest \"${RESULTS_ROOT}/groups/grpo_manifest.json\"" in text
    assert "tools/validate_grpo_dataset.py" in text
    assert "--inspect-artifacts" in text
    assert "--out-summary \"${RESULTS_ROOT}/groups/grpo_dataset_validation.json\"" in text
    assert "--fail-on-error" in text
