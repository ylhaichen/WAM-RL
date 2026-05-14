from pathlib import Path


def test_four_gpu_grouped_rollout_job_builds_grpo_groups():
    text = Path("jobs/myriad/30_collect_grouped_rollouts_4gpu.sh").read_text()

    assert "tools/collect_robotwin_rollouts.py" in text
    assert "tools/build_grpo_groups.py" in text
    assert "--canonicalize-legacy-group-ids" in text
    assert "--expected-group-size \"${GROUP_SIZE}\"" in text
    assert "--require-strict-artifacts" in text
    assert "--require-existing-artifacts" in text
    assert "--wait-for-artifacts-seconds 120" in text
    assert "--fail-on-validation-errors" in text
    assert "--out-manifest \"${RESULTS_ROOT}/groups/grpo_manifest.json\"" in text
    assert "GROUP_SEED_SEARCH=\"${GROUP_SEED_SEARCH:-true}\"" in text
    assert "STABLE_SEED_CACHE_DIR=\"${STABLE_SEED_CACHE_DIR:-${RESULTS_ROOT}/groups/stable_seeds}\"" in text
    assert "tools/validate_grpo_dataset.py" in text
    assert "--out-summary \"${RESULTS_ROOT}/groups/grpo_dataset_validation.json\"" in text
    assert "--fail-on-error" in text


def test_robotwin_client_launcher_generates_task_specific_group_ids():
    text = Path("evaluation/robotwin/launch_client_multigpus.sh").read_text()

    assert "effective_group_id=" in text
    assert "${task_name}_seed${seed}_prompt${prompt_part}_group${group_part}" in text
    assert "--group_id" in text
    assert "--group_seed_search" in text
    assert "--stable_seed_cache_dir" in text


def test_robotwin_eval_client_searches_and_caches_group_stable_seed():
    text = Path("evaluation/robotwin/eval_polict_client_openpi.py").read_text()

    assert "grouped_rollout =" in text
    assert "group_seed_search =" in text
    assert "_load_cached_group_env_seed" in text
    assert "_write_cached_group_env_seed" in text
    assert "grouped rollout seed search exhausted" in text
    assert "grouped rollout seed {now_seed} failed during expert precheck" in text
