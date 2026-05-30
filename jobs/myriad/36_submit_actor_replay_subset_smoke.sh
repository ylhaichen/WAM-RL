#!/bin/bash

# Submit a low-resource real actor replay smoke job for a materialized subset.
# The actual training logic stays in 34_train_actor_replay_grpo_robotwin.sh.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: jobs/myriad/36_submit_actor_replay_subset_smoke.sh [--dry-run]

Submit a low-resource real actor replay smoke job for a materialized subset.
Set SUBSET_ROOT or GRPO_GROUPS_PATH before running.

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

RUN_ID="${RUN_ID:-grpo_actor_subset_smoke_$(date +%Y%m%d_%H%M%S)}"
JOB_NAME="${JOB_NAME:-wam_grpo_actor_subset}"

if [ -z "${GRPO_GROUPS_PATH:-}" ]; then
    if [ -n "${SUBSET_ROOT:-}" ]; then
        GRPO_GROUPS_PATH="${SUBSET_ROOT}/groups/grpo_groups.jsonl"
    else
        echo "Set GRPO_GROUPS_PATH or SUBSET_ROOT before submitting actor replay subset smoke." >&2
        exit 2
    fi
fi

WAM_ROOT="${WAM_ROOT:-/home/zcably0/Scratch/wam-rl}"
GRPO_OUTPUT_DIR="${GRPO_OUTPUT_DIR:-${WAM_ROOT}/results_grpo_actor_replay/${RUN_ID}}"
GRPO_STEPS="${GRPO_STEPS:-1}"
GRPO_LR="${GRPO_LR:-0.0000001}"
GRPO_ACTION_NUM_INFERENCE_STEPS="${GRPO_ACTION_NUM_INFERENCE_STEPS:-10}"
GRPO_LOGPROB_REDUCTION="${GRPO_LOGPROB_REDUCTION:-mean}"
GRPO_LOGPROB_STD_FLOOR="${GRPO_LOGPROB_STD_FLOOR:-0.1}"
GRPO_PROGRESS_EVERY="${GRPO_PROGRESS_EVERY:-1}"
GRPO_TRAINABLE_MODE="${GRPO_TRAINABLE_MODE:-action_heads}"
GRPO_MAX_RESOLVED_GB="${GRPO_MAX_RESOLVED_GB:-40}"
PRECHECK_SUBSET_AUDIT="${PRECHECK_SUBSET_AUDIT:-true}"
SUBSET_STORAGE_AUDIT_JSON="${SUBSET_STORAGE_AUDIT_JSON:-}"
if [ -z "${SUBSET_STORAGE_AUDIT_JSON}" ] && [ -n "${SUBSET_ROOT:-}" ]; then
    SUBSET_STORAGE_AUDIT_JSON="${SUBSET_ROOT}/storage_audit.json"
fi

QSUB_H_RT="${QSUB_H_RT:-2:00:00}"
QSUB_MEM="${QSUB_MEM:-8G}"
QSUB_SLOTS="${QSUB_SLOTS:-4}"
QSUB_TMPFS="${QSUB_TMPFS:-40G}"
QSUB_GPU="${QSUB_GPU:-1}"
DRY_RUN="${DRY_RUN:-0}"

export REPO_ROOT RUN_ID GRPO_GROUPS_PATH GRPO_OUTPUT_DIR
export GRPO_STEPS GRPO_LR GRPO_ACTION_NUM_INFERENCE_STEPS
export GRPO_LOGPROB_REDUCTION GRPO_LOGPROB_STD_FLOOR GRPO_PROGRESS_EVERY
export GRPO_TRAINABLE_MODE GRPO_MAX_RESOLVED_GB

JOB_SCRIPT="${REPO_ROOT}/jobs/myriad/34_train_actor_replay_grpo_robotwin.sh"
if [ ! -f "${JOB_SCRIPT}" ]; then
    echo "Missing actor replay trainer job script: ${JOB_SCRIPT}" >&2
    exit 2
fi
if [ ! -f "${GRPO_GROUPS_PATH}" ]; then
    echo "Missing GRPO groups file: ${GRPO_GROUPS_PATH}" >&2
    exit 2
fi

case "${PRECHECK_SUBSET_AUDIT}" in
    true|True|1|yes|YES|on|ON)
        if [ -n "${SUBSET_STORAGE_AUDIT_JSON}" ]; then
            if [ -f "${SUBSET_STORAGE_AUDIT_JSON}" ]; then
                python - "${SUBSET_STORAGE_AUDIT_JSON}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
budget = data.get("storage_budget")
if isinstance(budget, dict) and budget.get("ok") is False:
    print(f"Subset storage audit budget failed: {path}", file=sys.stderr)
    print(json.dumps(budget, indent=2), file=sys.stderr)
    raise SystemExit(3)
if isinstance(budget, dict):
    print(f"Subset storage audit precheck ok: {path}")
else:
    print(f"Subset storage audit has no storage_budget field: {path}")
PY
            else
                echo "Warning: subset storage audit not found: ${SUBSET_STORAGE_AUDIT_JSON}" >&2
            fi
        fi
        ;;
esac

echo "Submitting actor replay subset smoke job"
echo "  JOB_NAME=${JOB_NAME}"
echo "  RUN_ID=${RUN_ID}"
echo "  REPO_ROOT=${REPO_ROOT}"
echo "  GRPO_GROUPS_PATH=${GRPO_GROUPS_PATH}"
echo "  GRPO_OUTPUT_DIR=${GRPO_OUTPUT_DIR}"
echo "  GRPO_STEPS=${GRPO_STEPS}"
echo "  GRPO_LR=${GRPO_LR}"
echo "  GRPO_ACTION_NUM_INFERENCE_STEPS=${GRPO_ACTION_NUM_INFERENCE_STEPS}"
echo "  GRPO_LOGPROB_REDUCTION=${GRPO_LOGPROB_REDUCTION}"
echo "  GRPO_LOGPROB_STD_FLOOR=${GRPO_LOGPROB_STD_FLOOR}"
echo "  GRPO_MAX_RESOLVED_GB=${GRPO_MAX_RESOLVED_GB}"
echo "  PRECHECK_SUBSET_AUDIT=${PRECHECK_SUBSET_AUDIT}"
echo "  SUBSET_STORAGE_AUDIT_JSON=${SUBSET_STORAGE_AUDIT_JSON}"
echo "  QSUB_H_RT=${QSUB_H_RT}"
echo "  QSUB_MEM=${QSUB_MEM}"
echo "  QSUB_SLOTS=${QSUB_SLOTS}"
echo "  QSUB_TMPFS=${QSUB_TMPFS}"
echo "  QSUB_GPU=${QSUB_GPU}"

QSUB_ARGS=(
    -V
    -N "${JOB_NAME}"
    -l "h_rt=${QSUB_H_RT}"
    -l "mem=${QSUB_MEM}"
    -pe smp "${QSUB_SLOTS}"
    -l "tmpfs=${QSUB_TMPFS}"
)
if [ -n "${QSUB_GPU}" ] && [ "${QSUB_GPU}" != "0" ]; then
    QSUB_ARGS+=(-l "gpu=${QSUB_GPU}")
fi

if [ "${DRY_RUN}" = "1" ]; then
    printf 'qsub'
    printf ' %q' "${QSUB_ARGS[@]}" "${JOB_SCRIPT}"
    printf '\n'
    exit 0
fi

qsub "${QSUB_ARGS[@]}" "${JOB_SCRIPT}"
