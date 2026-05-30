#!/bin/bash -l

# Legacy sequential native/offline RL iteration:
#   1. collect grouped RoboTwin rollouts;
#   2. build/validate GRPO groups;
#   3. run strict-artifact offline GRPO smoke training.
#
# This is an offline contract/debug fallback, not the current real actor replay
# loop. Use jobs/myriad/35_prepare_actor_replay_subset.sh plus
# jobs/myriad/36_submit_actor_replay_subset_smoke.sh for real LingBot-VA actor
# replay smoke work. Set ALLOW_LEGACY_OFFLINE_ITERATION=1 only when deliberately
# running this older fallback.

#$ -S /bin/bash
#$ -N wam_rl_iter
#$ -cwd
#$ -j y
#$ -o logs/jobs
#$ -l h_rt=48:00:00
#$ -l mem=4G
#$ -pe smp 32
#$ -l tmpfs=200G
#$ -l gpu=4
#$ -ac allow=L

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

ITERATION="${ITERATION:-0}"
RUN_ID="${RUN_ID:-rl_iter_${ITERATION}_$(date +%Y%m%d_%H%M%S)}"
ITERATION_ROOT="${ITERATION_ROOT:-${WAM_ROOT}/runs/${RUN_ID}}"
RESULTS_ROOT="${ITERATION_ROOT}/rollout_collection"
GRPO_GROUPS_PATH="${RESULTS_ROOT}/groups/grpo_groups.jsonl"
GRPO_OUTPUT_DIR="${ITERATION_ROOT}/train"
GROUP_SIZE="${GROUP_SIZE:-4}"
GROUPS_PER_TASK="${GROUPS_PER_TASK:-20}"
TASK_NAMES="${TASK_NAMES:-hanging_mug open_microwave turn_switch move_stapler_pad}"
GRPO_STEPS="${GRPO_STEPS:-20}"
ALLOW_LEGACY_OFFLINE_ITERATION="${ALLOW_LEGACY_OFFLINE_ITERATION:-0}"

export REPO_ROOT RUN_ID ITERATION ITERATION_ROOT RESULTS_ROOT
export GRPO_GROUPS_PATH GRPO_OUTPUT_DIR GROUP_SIZE GROUPS_PER_TASK TASK_NAMES GRPO_STEPS

print_job_context
echo "WARNING: jobs/myriad/40_rl_iteration_robotwin.sh is an offline smoke fallback, not the real actor replay loop."
echo "ALLOW_LEGACY_OFFLINE_ITERATION=${ALLOW_LEGACY_OFFLINE_ITERATION}"
echo "ITERATION=${ITERATION}"
echo "RUN_ID=${RUN_ID}"
echo "ITERATION_ROOT=${ITERATION_ROOT}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "GRPO_GROUPS_PATH=${GRPO_GROUPS_PATH}"
echo "GRPO_OUTPUT_DIR=${GRPO_OUTPUT_DIR}"
echo "GROUP_SIZE=${GROUP_SIZE}"
echo "GROUPS_PER_TASK=${GROUPS_PER_TASK}"
echo "TASK_NAMES=${TASK_NAMES}"

if [ "${ALLOW_LEGACY_OFFLINE_ITERATION}" != "1" ]; then
    echo "Refusing to run legacy offline iteration without ALLOW_LEGACY_OFFLINE_ITERATION=1." >&2
    echo "Use jobs/myriad/35_submit_prepare_actor_replay_subset.sh and jobs/myriad/36_submit_actor_replay_subset_smoke.sh for current actor replay work." >&2
    exit 2
fi

mkdir -p "${ITERATION_ROOT}/reports"

bash "${REPO_ROOT}/jobs/myriad/30_collect_grouped_rollouts_4gpu.sh"

bash "${REPO_ROOT}/jobs/myriad/31_train_denoising_grpo_robotwin.sh"

cat > "${ITERATION_ROOT}/reports/iteration_summary.txt" <<EOF
RUN_ID=${RUN_ID}
ITERATION=${ITERATION}
RESULTS_ROOT=${RESULTS_ROOT}
GRPO_GROUPS_PATH=${GRPO_GROUPS_PATH}
GRPO_OUTPUT_DIR=${GRPO_OUTPUT_DIR}
EOF

echo "Sequential RL iteration complete: ${ITERATION_ROOT}"
