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
    assert "SUCCESSFUL_ROOTS_FILE=\"${RESULTS_ROOT}/groups/successful_attempt_roots.txt\"" in text
    assert "FAILED_ROOTS_FILE=\"${RESULTS_ROOT}/groups/failed_attempt_roots.txt\"" in text
    assert "while (( completed_groups < GROUPS_PER_TASK && attempt_index < GROUP_MAX_ATTEMPTS ))" in text
    assert "Discarding failed group attempt" in text
    assert "\"${SUCCESSFUL_ATTEMPT_ROOTS[@]}\"" in text
    assert "tools/validate_grpo_dataset.py" in text
    assert "--inspect-artifacts" in text
    assert "--out-summary \"${RESULTS_ROOT}/groups/grpo_dataset_validation.json\"" in text
    assert "--fail-on-error" in text
    assert 'STRICT_GRPO_CAPTURE_SCOPE="${STRICT_GRPO_CAPTURE_SCOPE:-action_denoising_trajectory}"' in text
    assert 'strict_grpo_capture_scope="${STRICT_GRPO_CAPTURE_SCOPE}"' in text
    assert 'STRICT_GRPO_SAVE_REPLAY_CONTEXT="${STRICT_GRPO_SAVE_REPLAY_CONTEXT:-false}"' in text
    assert 'strict_grpo_save_replay_context="${STRICT_GRPO_SAVE_REPLAY_CONTEXT}"' in text
    assert 'STRICT_GRPO_CAPTURE_CHUNK_STRIDE="${STRICT_GRPO_CAPTURE_CHUNK_STRIDE:-1}"' in text
    assert 'STRICT_GRPO_CAPTURE_MAX_CHUNKS="${STRICT_GRPO_CAPTURE_MAX_CHUNKS:-0}"' in text
    assert 'strict_grpo_capture_chunk_stride="${STRICT_GRPO_CAPTURE_CHUNK_STRIDE}"' in text
    assert 'strict_grpo_capture_max_chunks="${STRICT_GRPO_CAPTURE_MAX_CHUNKS}"' in text
    assert 'ACTOR_REPLAY_CHECKPOINT_PATH="${ACTOR_REPLAY_CHECKPOINT_PATH:-}"' in text
    assert 'actor_replay_checkpoint_path="${ACTOR_REPLAY_CHECKPOINT_PATH}"' in text


def test_one_gpu_grouped_rollout_job_uses_successful_attempt_roots():
    text = Path("jobs/myriad/30_collect_grouped_rollouts_1gpu.sh").read_text()

    assert "SUCCESSFUL_ROOTS_FILE=\"${RESULTS_ROOT}/groups/successful_attempt_roots.txt\"" in text
    assert "FAILED_ROOTS_FILE=\"${RESULTS_ROOT}/groups/failed_attempt_roots.txt\"" in text
    assert "while (( completed_groups < GROUPS_PER_TASK && attempt_index < GROUP_MAX_ATTEMPTS ))" in text
    assert "Discarding failed group attempt" in text
    assert "\"${SUCCESSFUL_ATTEMPT_ROOTS[@]}\"" in text
    assert "--inspect-artifacts" in text
    assert 'STRICT_GRPO_CAPTURE_SCOPE="${STRICT_GRPO_CAPTURE_SCOPE:-action_denoising_trajectory}"' in text
    assert 'strict_grpo_capture_scope="${STRICT_GRPO_CAPTURE_SCOPE}"' in text
    assert 'STRICT_GRPO_SAVE_REPLAY_CONTEXT="${STRICT_GRPO_SAVE_REPLAY_CONTEXT:-false}"' in text
    assert 'strict_grpo_save_replay_context="${STRICT_GRPO_SAVE_REPLAY_CONTEXT}"' in text
    assert 'STRICT_GRPO_CAPTURE_CHUNK_STRIDE="${STRICT_GRPO_CAPTURE_CHUNK_STRIDE:-1}"' in text
    assert 'STRICT_GRPO_CAPTURE_MAX_CHUNKS="${STRICT_GRPO_CAPTURE_MAX_CHUNKS:-0}"' in text
    assert 'strict_grpo_capture_chunk_stride="${STRICT_GRPO_CAPTURE_CHUNK_STRIDE}"' in text
    assert 'strict_grpo_capture_max_chunks="${STRICT_GRPO_CAPTURE_MAX_CHUNKS}"' in text
    assert 'ACTOR_REPLAY_CHECKPOINT_PATH="${ACTOR_REPLAY_CHECKPOINT_PATH:-}"' in text
    assert 'actor_replay_checkpoint_path="${ACTOR_REPLAY_CHECKPOINT_PATH}"' in text


def test_selected_eval_job_can_load_actor_replay_checkpoint():
    text = Path("jobs/myriad/13_eval_selected_tasks_4gpu.sh").read_text()

    assert 'ACTOR_REPLAY_CHECKPOINT_PATH="${ACTOR_REPLAY_CHECKPOINT_PATH:-}"' in text
    assert 'actor_replay_checkpoint_path="${ACTOR_REPLAY_CHECKPOINT_PATH}"' in text
    assert 'ACTION_NUM_INFERENCE_STEPS="${ACTION_NUM_INFERENCE_STEPS:-}"' in text
    assert 'action_num_inference_steps="${ACTION_NUM_INFERENCE_STEPS}"' in text
    assert 'SERVER_HOST="${SERVER_HOST:-127.0.0.1}"' in text
    assert 'SAVE_SERVER_DEBUG_TENSORS="${SAVE_SERVER_DEBUG_TENSORS:-false}"' in text
    assert 'save_server_debug_tensors="${SAVE_SERVER_DEBUG_TENSORS}"' in text
    assert 'SEED="${SEED:-0}"' in text
    assert "echo \"SEED=${SEED}\"" in text


def test_one_gpu_eval_smoke_job_can_load_actor_replay_checkpoint():
    text = Path("jobs/myriad/10_eval_smoke_1gpu.sh").read_text()
    launch_text = Path("evaluation/robotwin/launch_client.sh").read_text()

    assert 'ACTOR_REPLAY_CHECKPOINT_PATH="${ACTOR_REPLAY_CHECKPOINT_PATH:-}"' in text
    assert 'actor_replay_checkpoint_path="${ACTOR_REPLAY_CHECKPOINT_PATH}"' in text
    assert 'ACTION_NUM_INFERENCE_STEPS="${ACTION_NUM_INFERENCE_STEPS:-}"' in text
    assert 'action_num_inference_steps="${ACTION_NUM_INFERENCE_STEPS}"' in text
    assert 'SERVER_HOST="${SERVER_HOST:-127.0.0.1}"' in text
    assert 'PROMPT_INDEX="${PROMPT_INDEX:-}"' in text
    assert 'SAMPLING_SEED="${SAMPLING_SEED:-}"' in text
    assert 'SAMPLING_SEED_PER_ENV="${SAMPLING_SEED_PER_ENV:-}"' in text
    assert 'SEED="${SEED:-0}"' in text
    assert "echo \"SEED=${SEED}\"" in text
    assert "PROMPT_INDEX=${PROMPT_INDEX:-}" in launch_text
    assert "SAMPLING_SEED=${SAMPLING_SEED:-}" in launch_text
    assert "SAMPLING_SEED_PER_ENV=${SAMPLING_SEED_PER_ENV:-}" in launch_text
    assert "--prompt_index" in launch_text
    assert "--sampling_seed" in launch_text
    assert "--sampling_seed_per_env" in launch_text
    assert "SERVER_HOST=${SERVER_HOST:-127.0.0.1}" in launch_text
    assert "--server_host ${SERVER_HOST}" in launch_text
    assert "extra_args=()" in launch_text
    assert "--action_num_inference_steps" in launch_text
    assert 'SAVE_SERVER_DEBUG_TENSORS="${SAVE_SERVER_DEBUG_TENSORS:-false}"' in text
    assert 'save_server_debug_tensors="${SAVE_SERVER_DEBUG_TENSORS}"' in text


def test_robotwin_client_launcher_generates_task_specific_group_ids():
    text = Path("evaluation/robotwin/launch_client_multigpus.sh").read_text()

    assert "server_host=${SERVER_HOST:-127.0.0.1}" in text
    assert "--server_host ${server_host}" in text
    assert "effective_group_id=" in text
    assert "${task_name}_seed${seed}_prompt${prompt_part}_group${group_part}" in text
    assert "--group_id" in text
    assert "--sampling_seed_per_env" in text
    assert "--group_seed_search" in text
    assert "--stable_seed_cache_dir" in text


def test_robotwin_eval_client_searches_and_caches_group_stable_seed():
    text = Path("evaluation/robotwin/eval_polict_client_openpi.py").read_text()

    assert 'parser.add_argument("--server_host"' in text
    assert 'parser.add_argument("--port", type=int, default=None' in text
    assert "CLI_CONFIG_DEFAULTS" in text
    assert "config.setdefault(key, value)" in text
    assert 'host=usr_args.get("server_host", "127.0.0.1")' in text
    assert "grouped_rollout =" in text
    assert "sampling_seed_per_env" in text
    assert "_episode_sampling_seed" in text
    assert "group_seed_search =" in text
    assert "_load_cached_group_env_seed" in text
    assert "_write_cached_group_env_seed" in text
    assert "grouped rollout seed search exhausted" in text
    assert "grouped rollout seed {now_seed} failed during expert precheck" in text


def test_scale_submit_wrapper_does_not_inherit_stale_output_roots_by_default():
    text = Path("jobs/myriad/32_submit_grpo_scale_8tasks_4gpu.sh").read_text()

    assert 'USE_EXISTING_RESULTS_ROOT="${USE_EXISTING_RESULTS_ROOT:-0}"' in text
    assert 'USE_EXISTING_STABLE_SEED_CACHE_DIR="${USE_EXISTING_STABLE_SEED_CACHE_DIR:-0}"' in text
    assert 'GROUP_MAX_ATTEMPTS="${GROUP_MAX_ATTEMPTS:-$((GROUPS_PER_TASK * GROUP_RETRY_MULTIPLIER))}"' in text
    assert "export GROUP_MAX_ATTEMPTS" in text
    assert "export STRICT_GRPO_CAPTURE_SCOPE" in text
    assert "export STRICT_GRPO_SAVE_REPLAY_CONTEXT" in text
    assert "export STRICT_GRPO_CAPTURE_CHUNK_STRIDE" in text
    assert "export STRICT_GRPO_CAPTURE_MAX_CHUNKS" in text
    assert 'unset RESULTS_ROOT' in text
    assert 'unset STABLE_SEED_CACHE_DIR' in text


def test_next_round_submit_wrapper_targets_hard_medium_tasks():
    text = Path("jobs/myriad/33_submit_grpo_next_round_4gpu.sh").read_text()

    assert 'GROUP_SIZE="${GROUP_SIZE:-8}"' in text
    assert 'GROUPS_PER_TASK="${GROUPS_PER_TASK:-8}"' in text
    assert "hanging_mug turn_switch open_microwave put_bottles_dustbin" in text
    assert "move_stapler_pad press_stapler place_dual_shoes place_fan" in text
    assert 'CORE_GROUP_RETRY_MULTIPLIER="${CORE_GROUP_RETRY_MULTIPLIER:-12}"' in text
    assert 'SECONDARY_GROUP_RETRY_MULTIPLIER="${SECONDARY_GROUP_RETRY_MULTIPLIER:-6}"' in text
    assert 'DRY_RUN="${DRY_RUN:-0}"' in text
    assert 'STRICT_GRPO_CAPTURE_SCOPE="${STRICT_GRPO_CAPTURE_SCOPE:-action_denoising_trajectory}"' in text
    assert 'STRICT_GRPO_SAVE_REPLAY_CONTEXT="${STRICT_GRPO_SAVE_REPLAY_CONTEXT:-false}"' in text
    assert 'STRICT_GRPO_CAPTURE_CHUNK_STRIDE="${STRICT_GRPO_CAPTURE_CHUNK_STRIDE:-1}"' in text
    assert 'STRICT_GRPO_CAPTURE_MAX_CHUNKS="${STRICT_GRPO_CAPTURE_MAX_CHUNKS:-0}"' in text
    assert 'STRICT_GRPO_CAPTURE_CHUNK_STRIDE="${STRICT_GRPO_CAPTURE_CHUNK_STRIDE}"' in text
    assert 'STRICT_GRPO_CAPTURE_MAX_CHUNKS="${STRICT_GRPO_CAPTURE_MAX_CHUNKS}"' in text
    assert 'bash "${SUBMIT_SCRIPT}"' in text


def test_actor_replay_training_job_runs_real_actor_trainer():
    text = Path("jobs/myriad/34_train_actor_replay_grpo_robotwin.sh").read_text()

    assert "tools/train_actor_replay_grpo.py" in text
    assert "--groups-jsonl \"${GRPO_GROUPS_PATH}\"" in text
    assert "--trainable-mode \"${GRPO_TRAINABLE_MODE}\"" in text
    assert "tools/validate_grpo_dataset.py" in text
    assert "--inspect-artifacts" in text
    assert "--require-replay-context" in text
    assert "STRICT_GRPO_SAVE_REPLAY_CONTEXT=true" in text
    assert 'GRPO_LOGPROB_REDUCTION="${GRPO_LOGPROB_REDUCTION:-mean}"' in text
    assert 'GRPO_LOGPROB_STD_FLOOR="${GRPO_LOGPROB_STD_FLOOR:-0.1}"' in text
    assert "--logprob-reduction \"${GRPO_LOGPROB_REDUCTION}\"" in text
    assert "--logprob-std-floor \"${GRPO_LOGPROB_STD_FLOOR}\"" in text
    assert "--progress-every \"${GRPO_PROGRESS_EVERY}\"" in text


def test_actor_replay_subset_job_materializes_lightweight_dataset():
    text = Path("jobs/myriad/35_prepare_actor_replay_subset.sh").read_text()

    assert "tools/subset_grpo_groups.py" in text
    assert "tools/materialize_grpo_artifacts.py" in text
    assert "tools/validate_grpo_dataset.py" in text
    assert 'MATERIALIZE_LINK_MODE="${MATERIALIZE_LINK_MODE:-symlink}"' in text
    assert 'MATERIALIZE_INCLUDE_REPLAY_CONTEXT="${MATERIALIZE_INCLUDE_REPLAY_CONTEXT:-true}"' in text
    assert '--link-mode "${MATERIALIZE_LINK_MODE}"' in text
    assert "--include-replay-context" in text
    assert 'VALIDATE_INSPECT_ARTIFACTS="${VALIDATE_INSPECT_ARTIFACTS:-false}"' in text
    assert 'STORAGE_AUDIT_JSON="${STORAGE_AUDIT_JSON:-${SUBSET_ROOT}/storage_audit.json}"' in text
    assert "tools/audit_grpo_artifact_storage.py" in text
    assert '--materialize-manifest "${MATERIALIZED_MANIFEST}"' in text
    assert '--out-json "${STORAGE_AUDIT_JSON}"' in text
    assert "--fail-on-missing" in text
    assert "Actor replay subset preparation complete" in text


def test_actor_replay_subset_smoke_submitter_uses_low_resource_defaults():
    text = Path("jobs/myriad/36_submit_actor_replay_subset_smoke.sh").read_text()

    assert "34_train_actor_replay_grpo_robotwin.sh" in text
    assert 'GRPO_GROUPS_PATH="${SUBSET_ROOT}/groups/grpo_groups.jsonl"' in text
    assert 'GRPO_STEPS="${GRPO_STEPS:-1}"' in text
    assert 'GRPO_LR="${GRPO_LR:-0.0000001}"' in text
    assert 'GRPO_ACTION_NUM_INFERENCE_STEPS="${GRPO_ACTION_NUM_INFERENCE_STEPS:-10}"' in text
    assert 'GRPO_LOGPROB_REDUCTION="${GRPO_LOGPROB_REDUCTION:-mean}"' in text
    assert 'GRPO_LOGPROB_STD_FLOOR="${GRPO_LOGPROB_STD_FLOOR:-0.1}"' in text
    assert 'GRPO_PROGRESS_EVERY="${GRPO_PROGRESS_EVERY:-1}"' in text
    assert 'QSUB_H_RT="${QSUB_H_RT:-2:00:00}"' in text
    assert 'QSUB_SLOTS="${QSUB_SLOTS:-4}"' in text
    assert 'QSUB_TMPFS="${QSUB_TMPFS:-40G}"' in text
    assert 'DRY_RUN="${DRY_RUN:-0}"' in text
    assert 'qsub "${QSUB_ARGS[@]}" "${JOB_SCRIPT}"' in text


def test_actor_eval_pair_smoke_submitter_uses_matched_eval_controls():
    text = Path("jobs/myriad/37_submit_actor_eval_pair_smoke.sh").read_text()

    assert "10_eval_smoke_1gpu.sh" in text
    assert 'TASK_NAME="${TASK_NAME:-move_stapler_pad}"' in text
    assert 'TEST_NUM="${TEST_NUM:-2}"' in text
    assert 'ACTION_NUM_INFERENCE_STEPS="${ACTION_NUM_INFERENCE_STEPS:-10}"' in text
    assert 'PROMPT_INDEX="${PROMPT_INDEX:-0}"' in text
    assert 'SAMPLING_SEED="${SAMPLING_SEED:-12345}"' in text
    assert 'SAMPLING_SEED_PER_ENV="${SAMPLING_SEED_PER_ENV:-true}"' in text
    assert 'SAVE_SERVER_DEBUG_TENSORS="${SAVE_SERVER_DEBUG_TENSORS:-false}"' in text
    assert 'SEED="${SEED:-0}"' in text
    assert '"SEED=${SEED}"' in text
    assert 'BASELINE_PORT="${BASELINE_PORT:-29656}"' in text
    assert 'ACTOR_PORT="${ACTOR_PORT:-29756}"' in text
    assert "BASELINE_PORT and ACTOR_PORT must differ" in text
    assert "Set ACTOR_REPLAY_CHECKPOINT_PATH" in text
    assert '"ACTOR_REPLAY_CHECKPOINT_PATH=${checkpoint}"' in text
    assert "tools/summarize_actor_eval_pair.py" in text
    assert 'DRY_RUN="${DRY_RUN:-0}"' in text
    assert '"${cmd[@]}"' in text


def test_myriad_common_initializes_modules_for_interactive_shells():
    text = Path("jobs/myriad/common.sh").read_text()

    assert "if ! command -v apptainer" in text
    assert "/shared/ucl/apps/modules/5.3.1/init/bash" in text
    assert "[ -x /usr/bin/tclsh ]" in text
    assert "module load apptainer/1.2.4-1" in text


def test_myriad_job_scripts_resolve_repo_root_from_sge_workdir():
    for path in sorted(Path("jobs/myriad").glob("*.sh")):
        text = path.read_text()
        if 'source "${REPO_ROOT}/jobs/myriad/common.sh"' not in text:
            continue
        assert "SGE_O_WORKDIR" in text, path
        assert "SGE_CWD_PATH" in text, path
        assert '[ -f "${PWD}/jobs/myriad/common.sh" ]' in text, path
        assert 'REPO_ROOT="$(cd "${MYRIAD_JOB_DIR}/../.." && pwd)"' in text, path
