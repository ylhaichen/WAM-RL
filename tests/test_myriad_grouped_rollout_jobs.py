import json
import os
import subprocess
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
    assert 'STRICT_GRPO_REPLAY_CONTEXT_MAX_GB="${STRICT_GRPO_REPLAY_CONTEXT_MAX_GB:-0}"' in text
    assert 'strict_grpo_replay_context_max_gb="${STRICT_GRPO_REPLAY_CONTEXT_MAX_GB}"' in text
    assert 'SAVE_SERVER_DEBUG_TENSORS="${SAVE_SERVER_DEBUG_TENSORS:-true}"' in text
    assert 'save_server_debug_tensors="${SAVE_SERVER_DEBUG_TENSORS}"' in text
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
    assert 'STRICT_GRPO_REPLAY_CONTEXT_MAX_GB="${STRICT_GRPO_REPLAY_CONTEXT_MAX_GB:-0}"' in text
    assert 'strict_grpo_replay_context_max_gb="${STRICT_GRPO_REPLAY_CONTEXT_MAX_GB}"' in text
    assert 'SAVE_SERVER_DEBUG_TENSORS="${SAVE_SERVER_DEBUG_TENSORS:-true}"' in text
    assert 'save_server_debug_tensors="${SAVE_SERVER_DEBUG_TENSORS}"' in text
    assert 'ACTOR_REPLAY_CHECKPOINT_PATH="${ACTOR_REPLAY_CHECKPOINT_PATH:-}"' in text
    assert 'actor_replay_checkpoint_path="${ACTOR_REPLAY_CHECKPOINT_PATH}"' in text


def test_selected_eval_job_can_load_actor_replay_checkpoint():
    text = Path("jobs/myriad/13_eval_selected_tasks_4gpu.sh").read_text()

    assert 'ACTOR_REPLAY_CHECKPOINT_PATH="${ACTOR_REPLAY_CHECKPOINT_PATH:-}"' in text
    assert 'actor_replay_checkpoint_path="${ACTOR_REPLAY_CHECKPOINT_PATH}"' in text
    assert 'ACTION_NUM_INFERENCE_STEPS="${ACTION_NUM_INFERENCE_STEPS:-}"' in text
    assert 'action_num_inference_steps="${ACTION_NUM_INFERENCE_STEPS}"' in text
    assert 'POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-${ACTOR_REPLAY_CHECKPOINT_PATH:-${WAN_VA_MODEL_PATH}}}"' in text
    assert 'REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-${WAN_VA_MODEL_PATH}}"' in text
    assert 'RUN_ID="${RUN_ID:-}"' in text
    assert "echo \"POLICY_CHECKPOINT=${POLICY_CHECKPOINT}\"" in text
    assert "echo \"REFERENCE_CHECKPOINT=${REFERENCE_CHECKPOINT}\"" in text
    assert "echo \"RUN_ID=${RUN_ID}\"" in text
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
    assert "POLICY_CHECKPOINT=${POLICY_CHECKPOINT:-}" in launch_text
    assert "REFERENCE_CHECKPOINT=${REFERENCE_CHECKPOINT:-}" in launch_text
    assert "RUN_ID=${RUN_ID:-}" in launch_text
    assert "--run_id" in launch_text
    assert "--prompt_index" in launch_text
    assert "--sampling_seed" in launch_text
    assert "--sampling_seed_per_env" in launch_text
    assert "--policy_checkpoint" in launch_text
    assert "--reference_checkpoint" in launch_text
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
    assert "export STRICT_GRPO_REPLAY_CONTEXT_MAX_GB" in text
    assert "export SAVE_SERVER_DEBUG_TENSORS" in text
    assert 'ACTOR_REPLAY_CHECKPOINT_PATH="${ACTOR_REPLAY_CHECKPOINT_PATH:-}"' in text
    assert "export ACTOR_REPLAY_CHECKPOINT_PATH" in text
    assert 'QSUB_EXPORT_CURRENT_ENV="${QSUB_EXPORT_CURRENT_ENV:-0}"' in text
    assert 'DRY_RUN="${DRY_RUN:-0}"' in text
    assert 'if [ "${QSUB_EXPORT_CURRENT_ENV}" = "1" ]; then' in text
    assert '"ACTOR_REPLAY_CHECKPOINT_PATH=${ACTOR_REPLAY_CHECKPOINT_PATH}"' in text
    assert 'cmd=(qsub "${QSUB_ARGS[@]}")' in text
    assert 'unset RESULTS_ROOT' in text
    assert 'unset STABLE_SEED_CACHE_DIR' in text


def test_scale_submit_wrapper_dry_run_uses_explicit_qsub_vars(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    qsub_called = tmp_path / "qsub_called"
    qsub = fake_bin / "qsub"
    qsub.write_text(
        f"#!/usr/bin/env bash\ntouch {qsub_called}\nexit 0\n",
        encoding="utf-8",
    )
    qsub.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "REPO_ROOT": str(Path.cwd()),
            "TASK_NAMES": "move_stapler_pad",
            "DRY_RUN": "1",
            "ACTOR_REPLAY_CHECKPOINT_PATH": str(tmp_path / "actor.pt"),
        }
    )

    result = subprocess.run(
        ["bash", "jobs/myriad/32_submit_grpo_scale_8tasks_4gpu.sh"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "qsub -V" not in result.stdout
    assert f"ACTOR_REPLAY_CHECKPOINT_PATH={tmp_path / 'actor.pt'}" in result.stdout
    assert "TASK_NAMES=move_stapler_pad" in result.stdout
    assert not qsub_called.exists()


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
    assert 'STRICT_GRPO_REPLAY_CONTEXT_MAX_GB="${STRICT_GRPO_REPLAY_CONTEXT_MAX_GB:-0}"' in text
    assert 'STRICT_GRPO_REPLAY_CONTEXT_MAX_GB="${STRICT_GRPO_REPLAY_CONTEXT_MAX_GB}"' in text
    assert 'SAVE_SERVER_DEBUG_TENSORS="${SAVE_SERVER_DEBUG_TENSORS:-true}"' in text
    assert 'SAVE_SERVER_DEBUG_TENSORS="${SAVE_SERVER_DEBUG_TENSORS}"' in text
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
    assert 'GRPO_STORAGE_AUDIT_JSON="${GRPO_STORAGE_AUDIT_JSON:-${GRPO_OUTPUT_DIR}/input_storage_audit.json}"' in text
    assert 'GRPO_AUDIT_REPLAY_CONTEXTS="${GRPO_AUDIT_REPLAY_CONTEXTS:-true}"' in text
    assert 'GRPO_MAX_RESOLVED_GB="${GRPO_MAX_RESOLVED_GB:-0}"' in text
    assert "--logprob-reduction \"${GRPO_LOGPROB_REDUCTION}\"" in text
    assert "--logprob-std-floor \"${GRPO_LOGPROB_STD_FLOOR}\"" in text
    assert "--progress-every \"${GRPO_PROGRESS_EVERY}\"" in text
    assert "tools/audit_grpo_artifact_storage.py" in text
    assert "--inspect-replay-contexts --omit-replay-context-mapping" in text
    assert "--max-resolved-gb \"${GRPO_MAX_RESOLVED_GB}\"" in text


def test_actor_replay_subset_job_materializes_lightweight_dataset():
    text = Path("jobs/myriad/35_prepare_actor_replay_subset.sh").read_text()

    assert "tools/subset_grpo_groups.py" in text
    assert "tools/materialize_grpo_artifacts.py" in text
    assert "tools/validate_grpo_dataset.py" in text
    assert 'MATERIALIZE_LINK_MODE="${MATERIALIZE_LINK_MODE:-symlink}"' in text
    assert 'MATERIALIZE_INCLUDE_REPLAY_CONTEXT="${MATERIALIZE_INCLUDE_REPLAY_CONTEXT:-true}"' in text
    assert 'SUBSET_MAX_REPLAY_CONTEXT_GB="${SUBSET_MAX_REPLAY_CONTEXT_GB:-30}"' in text
    assert 'SUBSET_STORAGE_MAX_RESOLVED_GB="${SUBSET_STORAGE_MAX_RESOLVED_GB:-40}"' in text
    assert 'MATERIALIZE_PLAN_JSON="${MATERIALIZE_PLAN_JSON:-${SUBSET_ROOT}/materialize_plan.json}"' in text
    assert '"$(dirname "${MATERIALIZE_PLAN_JSON}")"' in text
    assert '--max-replay-context-gb "${SUBSET_MAX_REPLAY_CONTEXT_GB}"' in text
    assert '--link-mode "${MATERIALIZE_LINK_MODE}"' in text
    assert "--include-replay-context" in text
    assert '--dry-run > "${MATERIALIZE_PLAN_JSON}"' in text
    assert "source_artifacts_plus_replay_contexts" in text
    assert "Materialization preflight exceeds SUBSET_STORAGE_MAX_RESOLVED_GB" in text
    assert 'VALIDATE_INSPECT_ARTIFACTS="${VALIDATE_INSPECT_ARTIFACTS:-true}"' in text
    assert 'STORAGE_AUDIT_JSON="${STORAGE_AUDIT_JSON:-${SUBSET_ROOT}/storage_audit.json}"' in text
    assert "tools/audit_grpo_artifact_storage.py" in text
    assert '--materialize-manifest "${MATERIALIZED_MANIFEST}"' in text
    assert '--out-json "${STORAGE_AUDIT_JSON}"' in text
    assert '--max-resolved-gb "${SUBSET_STORAGE_MAX_RESOLVED_GB}"' in text
    assert "--fail-on-missing" in text
    assert "Actor replay subset preparation complete" in text


def test_actor_replay_subset_prepare_submitter_uses_explicit_qsub_vars():
    text = Path("jobs/myriad/35_submit_prepare_actor_replay_subset.sh").read_text()

    assert "35_prepare_actor_replay_subset.sh" in text
    assert "Set SOURCE_GROUPS_PATH or RESULTS_ROOT" in text
    assert 'SUBSET_ROOT="${SUBSET_ROOT:-${WAM_ROOT}/results_grpo_actor_replay_subsets/${RUN_ID}}"' in text
    assert 'SUBSET_MAX_REPLAY_CONTEXT_GB="${SUBSET_MAX_REPLAY_CONTEXT_GB:-30}"' in text
    assert 'SUBSET_STORAGE_MAX_RESOLVED_GB="${SUBSET_STORAGE_MAX_RESOLVED_GB:-40}"' in text
    assert 'MATERIALIZE_PLAN_JSON="${MATERIALIZE_PLAN_JSON:-${SUBSET_ROOT}/materialize_plan.json}"' in text
    assert 'MATERIALIZE_LINK_MODE="${MATERIALIZE_LINK_MODE:-symlink}"' in text
    assert 'MATERIALIZE_INCLUDE_REPLAY_CONTEXT="${MATERIALIZE_INCLUDE_REPLAY_CONTEXT:-true}"' in text
    assert 'QSUB_EXPORT_CURRENT_ENV="${QSUB_EXPORT_CURRENT_ENV:-0}"' in text
    assert 'if [ "${QSUB_EXPORT_CURRENT_ENV}" = "1" ]; then' in text
    assert '"SOURCE_GROUPS_PATH=${SOURCE_GROUPS_PATH}"' in text
    assert '"SUBSET_ROOT=${SUBSET_ROOT}"' in text
    assert '"MATERIALIZE_PLAN_JSON=${MATERIALIZE_PLAN_JSON}"' in text
    assert '"SUBSET_MAX_ARTIFACTS_PER_SAMPLE=${SUBSET_MAX_ARTIFACTS_PER_SAMPLE}"' in text
    assert '"MATERIALIZE_LINK_MODE=${MATERIALIZE_LINK_MODE}"' in text
    assert 'DRY_RUN="${DRY_RUN:-0}"' in text
    assert "--dry-run" in text
    assert '"${cmd[@]}"' in text


def test_actor_replay_subset_prepare_submitter_dry_run_does_not_call_qsub(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    qsub_called = tmp_path / "qsub_called"
    qsub = fake_bin / "qsub"
    qsub.write_text(
        f"#!/usr/bin/env bash\ntouch {qsub_called}\nexit 0\n",
        encoding="utf-8",
    )
    qsub.chmod(0o755)

    source_groups = tmp_path / "source" / "groups" / "grpo_groups.jsonl"
    source_groups.parent.mkdir(parents=True)
    source_groups.write_text("", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "REPO_ROOT": str(Path.cwd()),
            "SOURCE_GROUPS_PATH": str(source_groups),
            "WAM_ROOT": str(tmp_path),
            "ACTOR_REPLAY_CHECKPOINT_PATH": "/tmp/should-not-leak.pt",
        }
    )

    result = subprocess.run(
        ["bash", "jobs/myriad/35_submit_prepare_actor_replay_subset.sh", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "qsub" in result.stdout
    assert "qsub -V" not in result.stdout
    assert "SOURCE_GROUPS_PATH=" in result.stdout
    assert "SUBSET_MAX_REPLAY_CONTEXT_GB=30" in result.stdout
    assert "ACTOR_REPLAY_CHECKPOINT_PATH=/tmp/should-not-leak.pt" not in result.stdout
    assert not qsub_called.exists()


def test_actor_replay_subset_smoke_submitter_uses_low_resource_defaults():
    text = Path("jobs/myriad/36_submit_actor_replay_subset_smoke.sh").read_text()

    assert "34_train_actor_replay_grpo_robotwin.sh" in text
    assert 'GRPO_GROUPS_PATH="${SUBSET_ROOT}/groups/grpo_groups.jsonl"' in text
    assert 'GRPO_STEPS="${GRPO_STEPS:-1}"' in text
    assert 'GRPO_LR="${GRPO_LR:-0.0000001}"' in text
    assert 'GRPO_CLIP_LOW="${GRPO_CLIP_LOW:-0.2}"' in text
    assert 'GRPO_CLIP_HIGH="${GRPO_CLIP_HIGH:-0.28}"' in text
    assert 'GRPO_DEVICE="${GRPO_DEVICE:-cuda}"' in text
    assert 'GRPO_DTYPE="${GRPO_DTYPE:-bfloat16}"' in text
    assert 'GRPO_ACTION_NUM_INFERENCE_STEPS="${GRPO_ACTION_NUM_INFERENCE_STEPS:-10}"' in text
    assert 'GRPO_LOGPROB_REDUCTION="${GRPO_LOGPROB_REDUCTION:-mean}"' in text
    assert 'GRPO_LOGPROB_STD_FLOOR="${GRPO_LOGPROB_STD_FLOOR:-0.1}"' in text
    assert 'GRPO_MAX_RESOLVED_GB="${GRPO_MAX_RESOLVED_GB:-40}"' in text
    assert 'GRPO_PROGRESS_EVERY="${GRPO_PROGRESS_EVERY:-1}"' in text
    assert 'GRPO_CONFIG_NAME="${GRPO_CONFIG_NAME:-robotwin_grpo_train}"' in text
    assert 'GRPO_STORAGE_AUDIT_JSON="${GRPO_STORAGE_AUDIT_JSON:-${GRPO_OUTPUT_DIR}/input_storage_audit.json}"' in text
    assert 'GRPO_AUDIT_REPLAY_CONTEXTS="${GRPO_AUDIT_REPLAY_CONTEXTS:-true}"' in text
    assert 'PRECHECK_SUBSET_AUDIT="${PRECHECK_SUBSET_AUDIT:-true}"' in text
    assert 'SUBSET_STORAGE_AUDIT_JSON="${SUBSET_STORAGE_AUDIT_JSON:-}"' in text
    assert "Missing GRPO groups file" in text
    assert "Subset storage audit budget failed" in text
    assert "Subset storage audit precheck ok" in text
    assert 'QSUB_H_RT="${QSUB_H_RT:-4:00:00}"' in text
    assert 'QSUB_MEM="${QSUB_MEM:-16G}"' in text
    assert 'QSUB_SLOTS="${QSUB_SLOTS:-4}"' in text
    assert 'QSUB_TMPFS="${QSUB_TMPFS:-60G}"' in text
    assert 'QSUB_EXPORT_CURRENT_ENV="${QSUB_EXPORT_CURRENT_ENV:-0}"' in text
    assert 'if [ "${QSUB_EXPORT_CURRENT_ENV}" = "1" ]; then' in text
    assert '"GRPO_GROUPS_PATH=${GRPO_GROUPS_PATH}"' in text
    assert '"GRPO_OUTPUT_DIR=${GRPO_OUTPUT_DIR}"' in text
    assert '"GRPO_DEVICE=${GRPO_DEVICE}"' in text
    assert '"GRPO_STORAGE_AUDIT_JSON=${GRPO_STORAGE_AUDIT_JSON}"' in text
    assert 'DRY_RUN="${DRY_RUN:-0}"' in text
    assert "--dry-run" in text
    assert '"${cmd[@]}"' in text


def test_actor_replay_subset_smoke_dry_run_flag_does_not_call_qsub(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    qsub_called = tmp_path / "qsub_called"
    qsub = fake_bin / "qsub"
    qsub.write_text(
        f"#!/usr/bin/env bash\ntouch {qsub_called}\nexit 0\n",
        encoding="utf-8",
    )
    qsub.chmod(0o755)

    subset_root = tmp_path / "subset"
    (subset_root / "groups").mkdir(parents=True)
    (subset_root / "groups" / "grpo_groups.jsonl").write_text("", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "REPO_ROOT": str(Path.cwd()),
            "SUBSET_ROOT": str(subset_root),
            "WAM_ROOT": str(tmp_path),
        }
    )

    result = subprocess.run(
        ["bash", "jobs/myriad/36_submit_actor_replay_subset_smoke.sh", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "qsub" in result.stdout
    assert "qsub -V" not in result.stdout
    assert "GRPO_GROUPS_PATH=" in result.stdout
    assert "GRPO_OUTPUT_DIR=" in result.stdout
    assert "GRPO_ACTION_NUM_INFERENCE_STEPS=10" in result.stdout
    assert not qsub_called.exists()


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
    assert 'BASELINE_REPEATABILITY_JSON="${BASELINE_REPEATABILITY_JSON:-}"' in text
    assert 'REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-${WAM_ROOT}/checkpoints/lingbot-va-posttrain-robotwin}"' in text
    assert 'BASELINE_POLICY_CHECKPOINT="${BASELINE_POLICY_CHECKPOINT:-${REFERENCE_CHECKPOINT}}"' in text
    assert 'ACTOR_POLICY_CHECKPOINT="${ACTOR_POLICY_CHECKPOINT:-${ACTOR_REPLAY_CHECKPOINT_PATH}}"' in text
    assert 'QSUB_EXPORT_CURRENT_ENV="${QSUB_EXPORT_CURRENT_ENV:-0}"' in text
    assert 'if [ "${QSUB_EXPORT_CURRENT_ENV}" = "1" ]; then' in text
    assert '"RUN_ID=${RUN_ID}"' in text
    assert '"SEED=${SEED}"' in text
    assert 'BASELINE_PORT="${BASELINE_PORT:-29656}"' in text
    assert 'ACTOR_PORT="${ACTOR_PORT:-29756}"' in text
    assert "BASELINE_PORT and ACTOR_PORT must differ" in text
    assert "Set ACTOR_REPLAY_CHECKPOINT_PATH" in text
    assert '"ACTOR_REPLAY_CHECKPOINT_PATH=${checkpoint}"' in text
    assert '"POLICY_CHECKPOINT=${policy_checkpoint}"' in text
    assert '"REFERENCE_CHECKPOINT=${reference_checkpoint}"' in text
    assert '"${BASELINE_POLICY_CHECKPOINT}"' in text
    assert '"${ACTOR_POLICY_CHECKPOINT}"' in text
    assert "tools/summarize_actor_eval_pair.py" in text
    assert "tools/gate_actor_eval_promotion.py" in text
    assert 'DRY_RUN="${DRY_RUN:-0}"' in text
    assert "--dry-run" in text
    assert '"${cmd[@]}"' in text


def test_actor_eval_pair_dry_run_flag_does_not_call_qsub(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    qsub_called = tmp_path / "qsub_called"
    qsub = fake_bin / "qsub"
    qsub.write_text(
        f"#!/usr/bin/env bash\ntouch {qsub_called}\nexit 0\n",
        encoding="utf-8",
    )
    qsub.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "REPO_ROOT": str(Path.cwd()),
            "ACTOR_REPLAY_CHECKPOINT_PATH": str(tmp_path / "checkpoint.pt"),
            "WAM_ROOT": str(tmp_path),
        }
    )

    result = subprocess.run(
        ["bash", "jobs/myriad/37_submit_actor_eval_pair_smoke.sh", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert sum(line.startswith("qsub ") for line in result.stdout.splitlines()) == 2
    assert "qsub -V" not in result.stdout
    assert "RUN_ID=actor_eval_pair_" in result.stdout
    assert not qsub_called.exists()


def test_eval_repeatability_pair_submitter_uses_matched_baseline_controls():
    text = Path("jobs/myriad/38_submit_eval_repeatability_pair.sh").read_text()

    assert "10_eval_smoke_1gpu.sh" in text
    assert 'TASK_NAME="${TASK_NAME:-move_stapler_pad}"' in text
    assert 'TEST_NUM="${TEST_NUM:-10}"' in text
    assert 'ACTION_NUM_INFERENCE_STEPS="${ACTION_NUM_INFERENCE_STEPS:-10}"' in text
    assert 'PROMPT_INDEX="${PROMPT_INDEX:-0}"' in text
    assert 'SAMPLING_SEED="${SAMPLING_SEED:-12345}"' in text
    assert 'SAMPLING_SEED_PER_ENV="${SAMPLING_SEED_PER_ENV:-true}"' in text
    assert 'SAVE_SERVER_DEBUG_TENSORS="${SAVE_SERVER_DEBUG_TENSORS:-false}"' in text
    assert 'SEED="${SEED:-0}"' in text
    assert '"ACTOR_REPLAY_CHECKPOINT_PATH="' in text
    assert 'REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-${WAM_ROOT}/checkpoints/lingbot-va-posttrain-robotwin}"' in text
    assert 'POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-${REFERENCE_CHECKPOINT}}"' in text
    assert '"POLICY_CHECKPOINT=${POLICY_CHECKPOINT}"' in text
    assert '"REFERENCE_CHECKPOINT=${REFERENCE_CHECKPOINT}"' in text
    assert 'QSUB_EXPORT_CURRENT_ENV="${QSUB_EXPORT_CURRENT_ENV:-0}"' in text
    assert 'if [ "${QSUB_EXPORT_CURRENT_ENV}" = "1" ]; then' in text
    assert '"RUN_ID=${RUN_ID}"' in text
    assert 'RUN_A_PORT="${RUN_A_PORT:-29856}"' in text
    assert 'RUN_B_PORT="${RUN_B_PORT:-29956}"' in text
    assert "RUN_A_PORT and RUN_B_PORT must differ" in text
    assert "tools/summarize_robotwin_repeatability.py" in text
    assert "tools/gate_actor_eval_promotion.py" in text
    assert 'DRY_RUN="${DRY_RUN:-0}"' in text
    assert "--dry-run" in text
    assert '"${cmd[@]}"' in text


def test_eval_repeatability_pair_dry_run_flag_does_not_call_qsub(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    qsub_called = tmp_path / "qsub_called"
    qsub = fake_bin / "qsub"
    qsub.write_text(
        f"#!/usr/bin/env bash\ntouch {qsub_called}\nexit 0\n",
        encoding="utf-8",
    )
    qsub.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "REPO_ROOT": str(Path.cwd()),
            "WAM_ROOT": str(tmp_path),
        }
    )

    result = subprocess.run(
        ["bash", "jobs/myriad/38_submit_eval_repeatability_pair.sh", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert sum(line.startswith("qsub ") for line in result.stdout.splitlines()) == 2
    assert "qsub -V" not in result.stdout
    assert "RUN_ID=eval_repeatability_pair_" in result.stdout
    assert not qsub_called.exists()


def test_bounded_replayctx_submitter_uses_storage_safe_defaults():
    text = Path("jobs/myriad/39_submit_grpo_replayctx_bounded_4gpu.sh").read_text()

    assert "32_submit_grpo_scale_8tasks_4gpu.sh" in text
    assert 'TASK_NAMES="${TASK_NAMES:-move_stapler_pad}"' in text
    assert 'GROUP_SIZE="${GROUP_SIZE:-8}"' in text
    assert 'GROUPS_PER_TASK="${GROUPS_PER_TASK:-1}"' in text
    assert 'ACTION_NUM_INFERENCE_STEPS="${ACTION_NUM_INFERENCE_STEPS:-10}"' in text
    assert 'GROUP_RETRY_MULTIPLIER="${GROUP_RETRY_MULTIPLIER:-3}"' in text
    assert 'STRICT_GRPO_SAVE_REPLAY_CONTEXT="${STRICT_GRPO_SAVE_REPLAY_CONTEXT:-true}"' in text
    assert 'STRICT_GRPO_CAPTURE_MAX_CHUNKS="${STRICT_GRPO_CAPTURE_MAX_CHUNKS:-1}"' in text
    assert 'SAVE_SERVER_DEBUG_TENSORS="${SAVE_SERVER_DEBUG_TENSORS:-false}"' in text
    assert 'REPLAY_CONTEXT_ESTIMATE_GB="${REPLAY_CONTEXT_ESTIMATE_GB:-4.0}"' in text
    assert 'STRICT_GRPO_REPLAY_CONTEXT_MAX_GB="${STRICT_GRPO_REPLAY_CONTEXT_MAX_GB:-5.0}"' in text
    assert 'CHECK_SCRATCH_HEADROOM="${CHECK_SCRATCH_HEADROOM:-1}"' in text
    assert 'MIN_SCRATCH_HEADROOM_GB="${MIN_SCRATCH_HEADROOM_GB:-50}"' in text
    assert 'STORAGE_BUDGET_MODE="${STORAGE_BUDGET_MODE:-attempt}"' in text
    assert 'PLAN_JSON="${PLAN_JSON:-}"' in text
    assert 'PLAN_ARGS=(' in text
    assert 'Wrote replay-context collection plan' in text
    assert "tools/plan_replay_context_collection.py" in text
    assert '--task-names "${TASK_NAMES}"' in text
    assert '--group-size "${GROUP_SIZE}"' in text
    assert '--group-max-attempts "${GROUP_MAX_ATTEMPTS}"' in text
    assert '--capture-max-chunks "${STRICT_GRPO_CAPTURE_MAX_CHUNKS}"' in text
    assert '--save-replay-context "${STRICT_GRPO_SAVE_REPLAY_CONTEXT}"' in text
    assert '--replay-context-estimate-gb "${REPLAY_CONTEXT_ESTIMATE_GB}"' in text
    assert '--format shell' in text
    assert "STRICT_GRPO_REPLAY_CONTEXT_MAX_GB" in text
    assert "ALLOW_UNBOUNDED_REPLAYCTX=1" in text
    assert 'DRY_RUN="${DRY_RUN:-0}"' in text
    assert "--dry-run" in text
    assert "DRY_RUN=1, not submitting" in text
    assert 'SAVE_SERVER_DEBUG_TENSORS="${SAVE_SERVER_DEBUG_TENSORS}"' in text
    assert 'STRICT_GRPO_REPLAY_CONTEXT_MAX_GB="${STRICT_GRPO_REPLAY_CONTEXT_MAX_GB}"' in text
    assert 'bash "${SUBMIT_SCRIPT}"' in text


def test_bounded_replayctx_dry_run_budgets_attempts(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    qsub = fake_bin / "qsub"
    qsub.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    qsub.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "TASK_NAMES": "move_stapler_pad",
            "GROUP_SIZE": "4",
            "GROUPS_PER_TASK": "1",
            "GROUP_MAX_ATTEMPTS": "3",
            "STRICT_GRPO_CAPTURE_MAX_CHUNKS": "1",
            "REPLAY_CONTEXT_ESTIMATE_GB": "4",
            "CHECK_SCRATCH_HEADROOM": "0",
        }
    )

    result = subprocess.run(
        ["bash", "jobs/myriad/39_submit_grpo_replayctx_bounded_4gpu.sh", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "accepted_estimate_gb=16.00" in result.stdout
    assert "attempt_budget_estimate_gb=48.00" in result.stdout
    assert "storage_budget_mode=attempt" in result.stdout
    assert "storage_budget_estimate_gb=48.00" in result.stdout
    assert "DRY_RUN=1, not submitting" in result.stdout


def test_bounded_replayctx_dry_run_can_write_plan_json(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    qsub = fake_bin / "qsub"
    qsub.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    qsub.chmod(0o755)
    plan_json = tmp_path / "plans" / "collection_plan.json"

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "TASK_NAMES": "move_stapler_pad",
            "GROUP_SIZE": "4",
            "GROUPS_PER_TASK": "1",
            "GROUP_MAX_ATTEMPTS": "1",
            "STRICT_GRPO_CAPTURE_MAX_CHUNKS": "1",
            "REPLAY_CONTEXT_ESTIMATE_GB": "4",
            "CHECK_SCRATCH_HEADROOM": "0",
            "PLAN_JSON": str(plan_json),
        }
    )

    result = subprocess.run(
        ["bash", "jobs/myriad/39_submit_grpo_replayctx_bounded_4gpu.sh", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert f"Wrote replay-context collection plan: {plan_json}" in result.stdout
    payload = json.loads(plan_json.read_text(encoding="utf-8"))
    assert payload["attempt_budget_estimate_gb"] == 16.0
    assert payload["dry_run"] is True


def test_bounded_replayctx_dry_run_can_budget_accepted_only(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    qsub = fake_bin / "qsub"
    qsub.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    qsub.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "DRY_RUN": "1",
            "TASK_NAMES": "move_stapler_pad",
            "GROUP_SIZE": "4",
            "GROUPS_PER_TASK": "1",
            "GROUP_MAX_ATTEMPTS": "3",
            "STRICT_GRPO_CAPTURE_MAX_CHUNKS": "1",
            "REPLAY_CONTEXT_ESTIMATE_GB": "4",
            "CHECK_SCRATCH_HEADROOM": "0",
            "STORAGE_BUDGET_MODE": "accepted",
        }
    )

    result = subprocess.run(
        ["bash", "jobs/myriad/39_submit_grpo_replayctx_bounded_4gpu.sh"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "storage_budget_mode=accepted" in result.stdout
    assert "storage_budget_estimate_gb=16.00" in result.stdout


def test_bounded_replayctx_non_dry_run_blocks_insufficient_headroom_before_qsub(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    qsub_called = tmp_path / "qsub_called"
    qsub = fake_bin / "qsub"
    qsub.write_text(
        f"#!/usr/bin/env bash\ntouch {qsub_called}\nexit 0\n",
        encoding="utf-8",
    )
    qsub.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "DRY_RUN": "0",
            "TASK_NAMES": "move_stapler_pad",
            "GROUP_SIZE": "4",
            "GROUPS_PER_TASK": "1",
            "GROUP_MAX_ATTEMPTS": "1",
            "STRICT_GRPO_CAPTURE_MAX_CHUNKS": "1",
            "REPLAY_CONTEXT_ESTIMATE_GB": "1000000",
            "CHECK_SCRATCH_HEADROOM": "1",
            "SCRATCH_PATH": str(tmp_path),
        }
    )

    result = subprocess.run(
        ["bash", "jobs/myriad/39_submit_grpo_replayctx_bounded_4gpu.sh"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 2
    assert "headroom_ok=false" in result.stdout
    assert "Insufficient Scratch headroom" in result.stderr
    assert not qsub_called.exists()


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
