#!/bin/bash

# Submit a storage-bounded actor replay subset preparation job without
# inheriting the whole interactive shell environment by default.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: jobs/myriad/35_submit_prepare_actor_replay_subset.sh [--dry-run]

Submit jobs/myriad/35_prepare_actor_replay_subset.sh with an explicit qsub
environment. Set SOURCE_GROUPS_PATH, or RESULTS_ROOT with groups/grpo_groups.jsonl.

Options:
  --dry-run   Print the qsub command and exit without submitting.
  -h, --help  Show this help text.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

MYRIAD_JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${REPO_ROOT:-}" ]; then
    if [ -n "${SGE_O_WORKDIR:-}" ] && [ -f "${SGE_O_WORKDIR}/jobs/myriad/common.sh" ]; then
        REPO_ROOT="${SGE_O_WORKDIR}"
    elif [ -n "${SGE_CWD_PATH:-}" ] && [ -f "${SGE_CWD_PATH}/jobs/myriad/common.sh" ]; then
        REPO_ROOT="${SGE_CWD_PATH}"
    elif [ -f "${PWD}/jobs/myriad/common.sh" ]; then
        REPO_ROOT="${PWD}"
    else
        REPO_ROOT="$(cd "${MYRIAD_JOB_DIR}/../.." && pwd)"
    fi
fi

WAM_ROOT="${WAM_ROOT:-/home/zcably0/Scratch/wam-rl}"
RUN_ID="${RUN_ID:-grpo_actor_subset_$(date +%Y%m%d_%H%M%S)}"
if [ -z "${SOURCE_GROUPS_PATH:-}" ]; then
    if [ -n "${RESULTS_ROOT:-}" ]; then
        SOURCE_GROUPS_PATH="${RESULTS_ROOT}/groups/grpo_groups.jsonl"
    else
        echo "Set SOURCE_GROUPS_PATH or RESULTS_ROOT before submitting subset preparation." >&2
        exit 2
    fi
fi
SUBSET_ROOT="${SUBSET_ROOT:-${WAM_ROOT}/results_grpo_actor_replay_subsets/${RUN_ID}}"
RAW_SUBSET_JSONL="${RAW_SUBSET_JSONL:-${SUBSET_ROOT}/source_subset/grpo_groups.jsonl}"
RAW_SUBSET_MANIFEST="${RAW_SUBSET_MANIFEST:-${SUBSET_ROOT}/source_subset/manifest.json}"
MATERIALIZED_GROUPS_PATH="${MATERIALIZED_GROUPS_PATH:-${SUBSET_ROOT}/groups/grpo_groups.jsonl}"
MATERIALIZED_MANIFEST="${MATERIALIZED_MANIFEST:-${SUBSET_ROOT}/manifest.json}"
MATERIALIZE_PLAN_JSON="${MATERIALIZE_PLAN_JSON:-${SUBSET_ROOT}/materialize_plan.json}"
VALIDATION_JSON="${VALIDATION_JSON:-${SUBSET_ROOT}/validation_actor_replay.json}"
STORAGE_AUDIT_JSON="${STORAGE_AUDIT_JSON:-${SUBSET_ROOT}/storage_audit.json}"
SUBSET_STORAGE_MAX_RESOLVED_GB="${SUBSET_STORAGE_MAX_RESOLVED_GB:-40}"

SUBSET_TASKS="${SUBSET_TASKS:-}"
SUBSET_MAX_GROUPS="${SUBSET_MAX_GROUPS:-1}"
SUBSET_SAMPLES_PER_REWARD="${SUBSET_SAMPLES_PER_REWARD:-1}"
SUBSET_MAX_ARTIFACTS_PER_SAMPLE="${SUBSET_MAX_ARTIFACTS_PER_SAMPLE:-2}"
SUBSET_MAX_REPLAY_CONTEXT_GB="${SUBSET_MAX_REPLAY_CONTEXT_GB:-30}"
SUBSET_REQUIRE_ARTIFACTS="${SUBSET_REQUIRE_ARTIFACTS:-true}"
SUBSET_PRESERVE_ADVANTAGES="${SUBSET_PRESERVE_ADVANTAGES:-false}"
SUBSET_PRESERVE_GROUP_ID="${SUBSET_PRESERVE_GROUP_ID:-false}"
SUBSET_GROUP_ID_SUFFIX="${SUBSET_GROUP_ID_SUFFIX:-_subset}"
MATERIALIZE_LINK_MODE="${MATERIALIZE_LINK_MODE:-symlink}"
MATERIALIZE_INCLUDE_REPLAY_CONTEXT="${MATERIALIZE_INCLUDE_REPLAY_CONTEXT:-true}"
MATERIALIZE_OVERWRITE="${MATERIALIZE_OVERWRITE:-true}"
VALIDATE_INSPECT_ARTIFACTS="${VALIDATE_INSPECT_ARTIFACTS:-true}"

JOB_NAME="${JOB_NAME:-wam_grpo_subset}"
QSUB_H_RT="${QSUB_H_RT:-1:00:00}"
QSUB_MEM="${QSUB_MEM:-8G}"
QSUB_SLOTS="${QSUB_SLOTS:-2}"
QSUB_TMPFS="${QSUB_TMPFS:-20G}"
QSUB_EXPORT_CURRENT_ENV="${QSUB_EXPORT_CURRENT_ENV:-0}"
DRY_RUN="${DRY_RUN:-0}"

JOB_SCRIPT="${REPO_ROOT}/jobs/myriad/35_prepare_actor_replay_subset.sh"
if [ ! -f "${JOB_SCRIPT}" ]; then
    echo "Missing actor replay subset preparation job script: ${JOB_SCRIPT}" >&2
    exit 2
fi
if [ ! -f "${SOURCE_GROUPS_PATH}" ]; then
    echo "Missing source groups file: ${SOURCE_GROUPS_PATH}" >&2
    exit 2
fi

echo "Submitting actor replay subset preparation job"
echo "  JOB_NAME=${JOB_NAME}"
echo "  RUN_ID=${RUN_ID}"
echo "  REPO_ROOT=${REPO_ROOT}"
echo "  WAM_ROOT=${WAM_ROOT}"
echo "  SOURCE_GROUPS_PATH=${SOURCE_GROUPS_PATH}"
echo "  SUBSET_ROOT=${SUBSET_ROOT}"
echo "  MATERIALIZE_PLAN_JSON=${MATERIALIZE_PLAN_JSON}"
echo "  SUBSET_TASKS=${SUBSET_TASKS}"
echo "  SUBSET_MAX_GROUPS=${SUBSET_MAX_GROUPS}"
echo "  SUBSET_SAMPLES_PER_REWARD=${SUBSET_SAMPLES_PER_REWARD}"
echo "  SUBSET_MAX_ARTIFACTS_PER_SAMPLE=${SUBSET_MAX_ARTIFACTS_PER_SAMPLE}"
echo "  SUBSET_MAX_REPLAY_CONTEXT_GB=${SUBSET_MAX_REPLAY_CONTEXT_GB}"
echo "  SUBSET_STORAGE_MAX_RESOLVED_GB=${SUBSET_STORAGE_MAX_RESOLVED_GB}"
echo "  MATERIALIZE_LINK_MODE=${MATERIALIZE_LINK_MODE}"
echo "  MATERIALIZE_INCLUDE_REPLAY_CONTEXT=${MATERIALIZE_INCLUDE_REPLAY_CONTEXT}"
echo "  VALIDATE_INSPECT_ARTIFACTS=${VALIDATE_INSPECT_ARTIFACTS}"
echo "  QSUB_H_RT=${QSUB_H_RT}"
echo "  QSUB_MEM=${QSUB_MEM}"
echo "  QSUB_SLOTS=${QSUB_SLOTS}"
echo "  QSUB_TMPFS=${QSUB_TMPFS}"
echo "  QSUB_EXPORT_CURRENT_ENV=${QSUB_EXPORT_CURRENT_ENV}"

QSUB_ARGS=(
    -N "${JOB_NAME}"
    -l "h_rt=${QSUB_H_RT}"
    -l "mem=${QSUB_MEM}"
    -pe smp "${QSUB_SLOTS}"
    -l "tmpfs=${QSUB_TMPFS}"
)
if [ "${QSUB_EXPORT_CURRENT_ENV}" = "1" ]; then
    QSUB_ARGS=(-V "${QSUB_ARGS[@]}")
fi

QSUB_VARS=(
    "REPO_ROOT=${REPO_ROOT}"
    "WAM_ROOT=${WAM_ROOT}"
    "RUN_ID=${RUN_ID}"
    "SOURCE_GROUPS_PATH=${SOURCE_GROUPS_PATH}"
    "SUBSET_ROOT=${SUBSET_ROOT}"
    "RAW_SUBSET_JSONL=${RAW_SUBSET_JSONL}"
    "RAW_SUBSET_MANIFEST=${RAW_SUBSET_MANIFEST}"
    "MATERIALIZED_GROUPS_PATH=${MATERIALIZED_GROUPS_PATH}"
    "MATERIALIZED_MANIFEST=${MATERIALIZED_MANIFEST}"
    "MATERIALIZE_PLAN_JSON=${MATERIALIZE_PLAN_JSON}"
    "VALIDATION_JSON=${VALIDATION_JSON}"
    "STORAGE_AUDIT_JSON=${STORAGE_AUDIT_JSON}"
    "SUBSET_STORAGE_MAX_RESOLVED_GB=${SUBSET_STORAGE_MAX_RESOLVED_GB}"
    "SUBSET_TASKS=${SUBSET_TASKS}"
    "SUBSET_MAX_GROUPS=${SUBSET_MAX_GROUPS}"
    "SUBSET_SAMPLES_PER_REWARD=${SUBSET_SAMPLES_PER_REWARD}"
    "SUBSET_MAX_ARTIFACTS_PER_SAMPLE=${SUBSET_MAX_ARTIFACTS_PER_SAMPLE}"
    "SUBSET_MAX_REPLAY_CONTEXT_GB=${SUBSET_MAX_REPLAY_CONTEXT_GB}"
    "SUBSET_REQUIRE_ARTIFACTS=${SUBSET_REQUIRE_ARTIFACTS}"
    "SUBSET_PRESERVE_ADVANTAGES=${SUBSET_PRESERVE_ADVANTAGES}"
    "SUBSET_PRESERVE_GROUP_ID=${SUBSET_PRESERVE_GROUP_ID}"
    "SUBSET_GROUP_ID_SUFFIX=${SUBSET_GROUP_ID_SUFFIX}"
    "MATERIALIZE_LINK_MODE=${MATERIALIZE_LINK_MODE}"
    "MATERIALIZE_INCLUDE_REPLAY_CONTEXT=${MATERIALIZE_INCLUDE_REPLAY_CONTEXT}"
    "MATERIALIZE_OVERWRITE=${MATERIALIZE_OVERWRITE}"
    "VALIDATE_INSPECT_ARTIFACTS=${VALIDATE_INSPECT_ARTIFACTS}"
)
if [ -n "${RESULTS_ROOT:-}" ]; then
    QSUB_VARS+=("RESULTS_ROOT=${RESULTS_ROOT}")
fi

cmd=(qsub "${QSUB_ARGS[@]}")
for value in "${QSUB_VARS[@]}"; do
    cmd+=(-v "${value}")
done
cmd+=("${JOB_SCRIPT}")

if [ "${DRY_RUN}" = "1" ]; then
    printf '%q' "${cmd[0]}"
    printf ' %q' "${cmd[@]:1}"
    printf '\n'
    exit 0
fi

"${cmd[@]}"
