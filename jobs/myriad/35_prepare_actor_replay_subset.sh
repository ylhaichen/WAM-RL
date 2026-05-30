#!/bin/bash -l

# Prepare a lightweight real actor replay subset from a large replay-context run.
# This job only rewrites JSONL and links/copies referenced artifacts. It does not
# run actor training and does not delete source data.

#$ -S /bin/bash
#$ -N wam_grpo_subset
#$ -cwd
#$ -j y
#$ -o logs/jobs
#$ -l h_rt=1:00:00
#$ -l mem=8G
#$ -pe smp 2
#$ -l tmpfs=20G

set -euo pipefail

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
source "${REPO_ROOT}/jobs/myriad/common.sh"

RUN_ID="${RUN_ID:-grpo_actor_subset_${JOB_ID:-manual}}"
RESULTS_ROOT="${RESULTS_ROOT:-${WAM_ROOT}/results_grouped_rollouts/latest}"
SOURCE_GROUPS_PATH="${SOURCE_GROUPS_PATH:-${RESULTS_ROOT}/groups/grpo_groups.jsonl}"
SUBSET_ROOT="${SUBSET_ROOT:-${WAM_ROOT}/results_grpo_actor_replay_subsets/${RUN_ID}}"
RAW_SUBSET_JSONL="${RAW_SUBSET_JSONL:-${SUBSET_ROOT}/source_subset/grpo_groups.jsonl}"
RAW_SUBSET_MANIFEST="${RAW_SUBSET_MANIFEST:-${SUBSET_ROOT}/source_subset/manifest.json}"
MATERIALIZED_GROUPS_PATH="${MATERIALIZED_GROUPS_PATH:-${SUBSET_ROOT}/groups/grpo_groups.jsonl}"
MATERIALIZED_MANIFEST="${MATERIALIZED_MANIFEST:-${SUBSET_ROOT}/manifest.json}"
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

export RUN_ID RESULTS_ROOT SOURCE_GROUPS_PATH SUBSET_ROOT RAW_SUBSET_JSONL RAW_SUBSET_MANIFEST
export MATERIALIZED_GROUPS_PATH MATERIALIZED_MANIFEST VALIDATION_JSON STORAGE_AUDIT_JSON
export SUBSET_STORAGE_MAX_RESOLVED_GB
export SUBSET_TASKS SUBSET_MAX_GROUPS SUBSET_SAMPLES_PER_REWARD SUBSET_MAX_ARTIFACTS_PER_SAMPLE
export SUBSET_MAX_REPLAY_CONTEXT_GB SUBSET_REQUIRE_ARTIFACTS SUBSET_PRESERVE_ADVANTAGES SUBSET_PRESERVE_GROUP_ID SUBSET_GROUP_ID_SUFFIX
export MATERIALIZE_LINK_MODE MATERIALIZE_INCLUDE_REPLAY_CONTEXT MATERIALIZE_OVERWRITE VALIDATE_INSPECT_ARTIFACTS

print_job_context
echo "RUN_ID=${RUN_ID}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "SOURCE_GROUPS_PATH=${SOURCE_GROUPS_PATH}"
echo "SUBSET_ROOT=${SUBSET_ROOT}"
echo "RAW_SUBSET_JSONL=${RAW_SUBSET_JSONL}"
echo "MATERIALIZED_GROUPS_PATH=${MATERIALIZED_GROUPS_PATH}"
echo "STORAGE_AUDIT_JSON=${STORAGE_AUDIT_JSON}"
echo "SUBSET_STORAGE_MAX_RESOLVED_GB=${SUBSET_STORAGE_MAX_RESOLVED_GB}"
echo "SUBSET_TASKS=${SUBSET_TASKS}"
echo "SUBSET_MAX_GROUPS=${SUBSET_MAX_GROUPS}"
echo "SUBSET_SAMPLES_PER_REWARD=${SUBSET_SAMPLES_PER_REWARD}"
echo "SUBSET_MAX_ARTIFACTS_PER_SAMPLE=${SUBSET_MAX_ARTIFACTS_PER_SAMPLE}"
echo "SUBSET_MAX_REPLAY_CONTEXT_GB=${SUBSET_MAX_REPLAY_CONTEXT_GB}"
echo "MATERIALIZE_LINK_MODE=${MATERIALIZE_LINK_MODE}"
echo "MATERIALIZE_INCLUDE_REPLAY_CONTEXT=${MATERIALIZE_INCLUDE_REPLAY_CONTEXT}"
echo "VALIDATE_INSPECT_ARTIFACTS=${VALIDATE_INSPECT_ARTIFACTS}"

container_exec_cpu <<'CONTAINER'
set -euo pipefail

cd "${REPO_ROOT}"

if [ ! -x "${WAN_VA_VENV}/bin/python" ]; then
    echo "Missing venv: ${WAN_VA_VENV}" >&2
    echo "Run jobs/myriad/00_install_container_env.sh first." >&2
    exit 1
fi

source "${WAN_VA_VENV}/bin/activate"

if [ ! -f "${SOURCE_GROUPS_PATH}" ]; then
    echo "Missing source groups file: ${SOURCE_GROUPS_PATH}" >&2
    exit 2
fi

mkdir -p "${SUBSET_ROOT}" "$(dirname "${RAW_SUBSET_JSONL}")" "$(dirname "${MATERIALIZED_GROUPS_PATH}")"

TASK_ARGS=()
if [ -n "${SUBSET_TASKS}" ]; then
    # shellcheck disable=SC2206
    TASK_ITEMS=(${SUBSET_TASKS})
    TASK_ARGS=(--tasks "${TASK_ITEMS[@]}")
fi

SUBSET_ARGS=()
if [ -n "${SUBSET_MAX_GROUPS}" ]; then
    SUBSET_ARGS+=(--max-groups "${SUBSET_MAX_GROUPS}")
fi
if [ -n "${SUBSET_SAMPLES_PER_REWARD}" ]; then
    SUBSET_ARGS+=(--samples-per-reward "${SUBSET_SAMPLES_PER_REWARD}")
fi
if [ -n "${SUBSET_MAX_ARTIFACTS_PER_SAMPLE}" ]; then
    SUBSET_ARGS+=(--max-artifacts-per-sample "${SUBSET_MAX_ARTIFACTS_PER_SAMPLE}")
fi
if [ -n "${SUBSET_MAX_REPLAY_CONTEXT_GB}" ]; then
    SUBSET_ARGS+=(--max-replay-context-gb "${SUBSET_MAX_REPLAY_CONTEXT_GB}")
fi
if [ "${SUBSET_REQUIRE_ARTIFACTS}" = "true" ]; then
    SUBSET_ARGS+=(--require-artifacts)
fi
if [ "${SUBSET_PRESERVE_ADVANTAGES}" = "true" ]; then
    SUBSET_ARGS+=(--preserve-advantages)
fi
if [ "${SUBSET_PRESERVE_GROUP_ID}" = "true" ]; then
    SUBSET_ARGS+=(--preserve-group-id)
fi

python tools/subset_grpo_groups.py \
    "${SOURCE_GROUPS_PATH}" \
    "${TASK_ARGS[@]}" \
    "${SUBSET_ARGS[@]}" \
    --group-id-suffix "${SUBSET_GROUP_ID_SUFFIX}" \
    --out-jsonl "${RAW_SUBSET_JSONL}" \
    --out-manifest "${RAW_SUBSET_MANIFEST}"

MATERIALIZE_ARGS=()
if [ "${MATERIALIZE_INCLUDE_REPLAY_CONTEXT}" = "true" ]; then
    MATERIALIZE_ARGS+=(--include-replay-context)
fi
if [ "${MATERIALIZE_OVERWRITE}" = "true" ]; then
    MATERIALIZE_ARGS+=(--overwrite)
fi

python tools/materialize_grpo_artifacts.py \
    "${RAW_SUBSET_JSONL}" \
    --out-root "${SUBSET_ROOT}" \
    --out-jsonl "${MATERIALIZED_GROUPS_PATH}" \
    --out-manifest "${MATERIALIZED_MANIFEST}" \
    --link-mode "${MATERIALIZE_LINK_MODE}" \
    "${MATERIALIZE_ARGS[@]}"

VALIDATE_ARGS=()
if [ "${VALIDATE_INSPECT_ARTIFACTS}" = "true" ]; then
    VALIDATE_ARGS+=(--inspect-artifacts --require-replay-context)
fi

python tools/validate_grpo_dataset.py \
    "${MATERIALIZED_GROUPS_PATH}" \
    "${VALIDATE_ARGS[@]}" \
    --out-summary "${VALIDATION_JSON}" \
    --fail-on-error

STORAGE_BUDGET_ARGS=()
if [ -n "${SUBSET_STORAGE_MAX_RESOLVED_GB}" ]; then
    STORAGE_BUDGET_ARGS+=(--max-resolved-gb "${SUBSET_STORAGE_MAX_RESOLVED_GB}")
fi

python tools/audit_grpo_artifact_storage.py \
    "${MATERIALIZED_GROUPS_PATH}" \
    --materialize-manifest "${MATERIALIZED_MANIFEST}" \
    --out-json "${STORAGE_AUDIT_JSON}" \
    "${STORAGE_BUDGET_ARGS[@]}" \
    --fail-on-missing

du -sh "${SUBSET_ROOT}" || true
find "${SUBSET_ROOT}" -type l | wc -l || true
echo "Actor replay subset preparation complete: ${SUBSET_ROOT}"
CONTAINER
