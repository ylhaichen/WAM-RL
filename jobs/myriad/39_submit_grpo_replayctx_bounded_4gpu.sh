#!/usr/bin/env bash

# Submit a storage-bounded replay-context grouped rollout collection.
# This is a conservative wrapper around 32_submit_grpo_scale_8tasks_4gpu.sh
# for actor-replay data collection, not a benchmark-scale rollout launcher.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}"
cd "${REPO_ROOT}"

SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-jobs/myriad/32_submit_grpo_scale_8tasks_4gpu.sh}"
DRY_RUN="${DRY_RUN:-0}"

JOB_NAME="${JOB_NAME:-wam_grpo_replayctx_bounded}"
TASK_NAMES="${TASK_NAMES:-move_stapler_pad}"
GROUP_SIZE="${GROUP_SIZE:-8}"
GROUPS_PER_TASK="${GROUPS_PER_TASK:-1}"
START_SEED="${START_SEED:-99200}"
PROMPT_INDEX="${PROMPT_INDEX:-0}"
ACTION_NUM_INFERENCE_STEPS="${ACTION_NUM_INFERENCE_STEPS:-10}"
GROUP_SEED_SEARCH_MAX_ATTEMPTS="${GROUP_SEED_SEARCH_MAX_ATTEMPTS:-120}"
GROUP_RETRY_MULTIPLIER="${GROUP_RETRY_MULTIPLIER:-3}"
GROUP_MAX_ATTEMPTS="${GROUP_MAX_ATTEMPTS:-$((GROUPS_PER_TASK * GROUP_RETRY_MULTIPLIER))}"
STRICT_GRPO_CAPTURE="${STRICT_GRPO_CAPTURE:-true}"
STRICT_GRPO_TRANSITION_STD="${STRICT_GRPO_TRANSITION_STD:-0.01}"
STRICT_GRPO_CAPTURE_SCOPE="${STRICT_GRPO_CAPTURE_SCOPE:-action_denoising_trajectory}"
STRICT_GRPO_SAVE_REPLAY_CONTEXT="${STRICT_GRPO_SAVE_REPLAY_CONTEXT:-true}"
STRICT_GRPO_CAPTURE_CHUNK_STRIDE="${STRICT_GRPO_CAPTURE_CHUNK_STRIDE:-1}"
STRICT_GRPO_CAPTURE_MAX_CHUNKS="${STRICT_GRPO_CAPTURE_MAX_CHUNKS:-1}"
SAVE_SERVER_DEBUG_TENSORS="${SAVE_SERVER_DEBUG_TENSORS:-false}"
ALLOW_UNBOUNDED_REPLAYCTX="${ALLOW_UNBOUNDED_REPLAYCTX:-0}"
REPLAY_CONTEXT_ESTIMATE_GB="${REPLAY_CONTEXT_ESTIMATE_GB:-4.0}"
STRICT_GRPO_REPLAY_CONTEXT_MAX_GB="${STRICT_GRPO_REPLAY_CONTEXT_MAX_GB:-5.0}"
CHECK_SCRATCH_HEADROOM="${CHECK_SCRATCH_HEADROOM:-1}"
SCRATCH_PATH="${SCRATCH_PATH:-/home/zcably0/Scratch}"
MIN_SCRATCH_HEADROOM_GB="${MIN_SCRATCH_HEADROOM_GB:-50}"
STORAGE_BUDGET_MODE="${STORAGE_BUDGET_MODE:-attempt}"
RUN_ID="${RUN_ID:-grpo_replayctx_bounded_k${GROUP_SIZE}_g${GROUPS_PER_TASK}_s${ACTION_NUM_INFERENCE_STEPS}_$(date +%Y%m%d_%H%M%S)}"

if [ ! -f "${SUBMIT_SCRIPT}" ]; then
    echo "Missing submit script: ${SUBMIT_SCRIPT}" >&2
    exit 2
fi

cat <<EOF
Bounded replay-context GRPO collection
  JOB_NAME=${JOB_NAME}
  RUN_ID=${RUN_ID}
  TASK_NAMES=${TASK_NAMES}
  GROUP_SIZE=${GROUP_SIZE}
  GROUPS_PER_TASK=${GROUPS_PER_TASK}
  START_SEED=${START_SEED}
  PROMPT_INDEX=${PROMPT_INDEX}
  ACTION_NUM_INFERENCE_STEPS=${ACTION_NUM_INFERENCE_STEPS}
  STRICT_GRPO_CAPTURE_SCOPE=${STRICT_GRPO_CAPTURE_SCOPE}
  STRICT_GRPO_SAVE_REPLAY_CONTEXT=${STRICT_GRPO_SAVE_REPLAY_CONTEXT}
  STRICT_GRPO_CAPTURE_CHUNK_STRIDE=${STRICT_GRPO_CAPTURE_CHUNK_STRIDE}
  STRICT_GRPO_CAPTURE_MAX_CHUNKS=${STRICT_GRPO_CAPTURE_MAX_CHUNKS}
  SAVE_SERVER_DEBUG_TENSORS=${SAVE_SERVER_DEBUG_TENSORS}
  GROUP_MAX_ATTEMPTS=${GROUP_MAX_ATTEMPTS}
  REPLAY_CONTEXT_ESTIMATE_GB=${REPLAY_CONTEXT_ESTIMATE_GB}
  STRICT_GRPO_REPLAY_CONTEXT_MAX_GB=${STRICT_GRPO_REPLAY_CONTEXT_MAX_GB}
  CHECK_SCRATCH_HEADROOM=${CHECK_SCRATCH_HEADROOM}
  SCRATCH_PATH=${SCRATCH_PATH}
  MIN_SCRATCH_HEADROOM_GB=${MIN_SCRATCH_HEADROOM_GB}
  STORAGE_BUDGET_MODE=${STORAGE_BUDGET_MODE}
  DRY_RUN=${DRY_RUN}
EOF

export TASK_NAMES GROUP_SIZE GROUPS_PER_TASK GROUP_MAX_ATTEMPTS
export STRICT_GRPO_SAVE_REPLAY_CONTEXT STRICT_GRPO_CAPTURE_MAX_CHUNKS ALLOW_UNBOUNDED_REPLAYCTX
export REPLAY_CONTEXT_ESTIMATE_GB STRICT_GRPO_REPLAY_CONTEXT_MAX_GB CHECK_SCRATCH_HEADROOM SCRATCH_PATH MIN_SCRATCH_HEADROOM_GB STORAGE_BUDGET_MODE DRY_RUN

python - <<'PY'
import os
import shutil
import sys

task_count = len(os.environ["TASK_NAMES"].split())
group_size = int(os.environ["GROUP_SIZE"])
groups_per_task = int(os.environ["GROUPS_PER_TASK"])
group_max_attempts = int(os.environ["GROUP_MAX_ATTEMPTS"])
max_chunks = int(os.environ["STRICT_GRPO_CAPTURE_MAX_CHUNKS"])
save_replay_context = os.environ["STRICT_GRPO_SAVE_REPLAY_CONTEXT"].lower() in {"1", "true", "yes", "on"}
allow_unbounded = os.environ["ALLOW_UNBOUNDED_REPLAYCTX"] == "1"
estimate_gb = float(os.environ["REPLAY_CONTEXT_ESTIMATE_GB"])
dry_run = os.environ["DRY_RUN"] == "1"
storage_budget_mode = os.environ["STORAGE_BUDGET_MODE"].strip().lower()
if storage_budget_mode not in {"attempt", "accepted"}:
    print("STORAGE_BUDGET_MODE must be 'attempt' or 'accepted'.", file=sys.stderr)
    sys.exit(2)

if save_replay_context and max_chunks <= 0 and not allow_unbounded:
    print(
        "Refusing bounded replay-context submission with STRICT_GRPO_CAPTURE_MAX_CHUNKS<=0. "
        "Set ALLOW_UNBOUNDED_REPLAYCTX=1 only for an explicitly reviewed full-capture run.",
        file=sys.stderr,
    )
    sys.exit(2)

chunks_per_rollout = max(max_chunks, 0)
accepted_contexts = task_count * groups_per_task * group_size * chunks_per_rollout if save_replay_context else 0
attempt_budget_contexts = task_count * group_max_attempts * group_size * chunks_per_rollout if save_replay_context else 0
accepted_gb = accepted_contexts * estimate_gb
attempt_budget_gb = attempt_budget_contexts * estimate_gb

print("Replay-context storage estimate")
print(f"  task_count={task_count}")
print(f"  accepted_contexts={accepted_contexts}")
print(f"  attempt_budget_contexts={attempt_budget_contexts}")
print(f"  accepted_estimate_gb={accepted_gb:.2f}")
print(f"  attempt_budget_estimate_gb={attempt_budget_gb:.2f}")
budget_gb = attempt_budget_gb if storage_budget_mode == "attempt" else accepted_gb
print(f"  storage_budget_mode={storage_budget_mode}")
print(f"  storage_budget_estimate_gb={budget_gb:.2f}")

if os.environ["CHECK_SCRATCH_HEADROOM"] == "1":
    scratch_path = os.environ["SCRATCH_PATH"]
    min_headroom_gb = float(os.environ["MIN_SCRATCH_HEADROOM_GB"])
    try:
        usage = shutil.disk_usage(scratch_path)
    except FileNotFoundError:
        print(f"  scratch_path={scratch_path}")
        print("  scratch_available_gb=unavailable")
        if not dry_run:
            print(f"Scratch path does not exist: {scratch_path}", file=sys.stderr)
            sys.exit(2)
        sys.exit(0)
    available_gb = usage.free / 1024**3
    required_gb = budget_gb + min_headroom_gb
    ok = available_gb >= required_gb
    print(f"  scratch_path={scratch_path}")
    print(f"  scratch_available_gb={available_gb:.2f}")
    print(f"  required_for_budget_plus_headroom_gb={required_gb:.2f}")
    print(f"  headroom_ok={str(ok).lower()}")
    if not dry_run and not ok:
        print(
            "Insufficient Scratch headroom for replay-context storage budget. "
            "Free space, lower GROUP_SIZE/GROUPS_PER_TASK/STRICT_GRPO_CAPTURE_MAX_CHUNKS, "
            "lower GROUP_MAX_ATTEMPTS, set STORAGE_BUDGET_MODE=accepted, "
            "or set CHECK_SCRATCH_HEADROOM=0 after explicit review.",
            file=sys.stderr,
        )
        sys.exit(2)
PY

if [ "${DRY_RUN}" = "1" ]; then
    echo "DRY_RUN=1, not submitting ${JOB_NAME}."
    exit 0
fi

JOB_NAME="${JOB_NAME}" \
RUN_ID="${RUN_ID}" \
TASK_NAMES="${TASK_NAMES}" \
GROUP_SIZE="${GROUP_SIZE}" \
GROUPS_PER_TASK="${GROUPS_PER_TASK}" \
START_SEED="${START_SEED}" \
PROMPT_INDEX="${PROMPT_INDEX}" \
ACTION_NUM_INFERENCE_STEPS="${ACTION_NUM_INFERENCE_STEPS}" \
GROUP_SEED_SEARCH=true \
GROUP_SEED_SEARCH_MAX_ATTEMPTS="${GROUP_SEED_SEARCH_MAX_ATTEMPTS}" \
GROUP_RETRY_MULTIPLIER="${GROUP_RETRY_MULTIPLIER}" \
GROUP_MAX_ATTEMPTS="${GROUP_MAX_ATTEMPTS}" \
STRICT_GRPO_CAPTURE="${STRICT_GRPO_CAPTURE}" \
STRICT_GRPO_TRANSITION_STD="${STRICT_GRPO_TRANSITION_STD}" \
STRICT_GRPO_CAPTURE_SCOPE="${STRICT_GRPO_CAPTURE_SCOPE}" \
STRICT_GRPO_SAVE_REPLAY_CONTEXT="${STRICT_GRPO_SAVE_REPLAY_CONTEXT}" \
STRICT_GRPO_CAPTURE_CHUNK_STRIDE="${STRICT_GRPO_CAPTURE_CHUNK_STRIDE}" \
STRICT_GRPO_CAPTURE_MAX_CHUNKS="${STRICT_GRPO_CAPTURE_MAX_CHUNKS}" \
STRICT_GRPO_REPLAY_CONTEXT_MAX_GB="${STRICT_GRPO_REPLAY_CONTEXT_MAX_GB}" \
SAVE_SERVER_DEBUG_TENSORS="${SAVE_SERVER_DEBUG_TENSORS}" \
REPO_ROOT="${REPO_ROOT}" \
bash "${SUBMIT_SCRIPT}"
