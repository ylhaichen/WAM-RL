# WAM-RL Actor Replay Runbook

**Purpose:** repeatable operational workflow for real LingBot-VA actor replay
GRPO on Myriad.

This is an engineering runbook. It records what to run, what to inspect, and
what not to claim. Use `docs/WAM_RL_CURRENT_PROJECT_STATUS.md` for project
stage and claims boundaries, and `docs/WAM_RL_MYRIAD_STORAGE_POLICY.md` for
storage and cleanup rules.

## Current Stage

The project is between Stage 1 and Stage 2:

```text
Stage 1 native offline pipeline: mostly complete
Stage 2 native real actor replay: implemented, smoke-scale validation in progress
Stage 3 veRL scale-up: intentionally deferred
```

Do not treat a one-step or tiny-subset actor replay run as benchmark
improvement. Its job is to verify that replay-context artifacts can update the
real LingBot-VA actor path and write a usable checkpoint.

## 0. Preflight

On Myriad:

```bash
cd /home/zcably0/Scratch/WAM-RL
git status --short
git log -1 --oneline
qstat
df -h /home/zcably0/Scratch
```

Expected repository noise is limited to known untracked local files such as
`outputs/` and `tasks.txt` on the server, or local user archives. Do not remove
them unless the user explicitly asks.

## 1. Validate The Source Replay Dataset

For a replay-context source run:

```bash
SOURCE=/home/zcably0/Scratch/wam-rl/results_grouped_rollouts/<run-id>
export REPO_ROOT=/home/zcably0/Scratch/WAM-RL
source jobs/myriad/common.sh

container_exec_gpu <<'CONTAINER'
set -euo pipefail
cd "$REPO_ROOT"
source "$WAN_VA_VENV/bin/activate"

python tools/validate_grpo_dataset.py \
  "$SOURCE/groups/grpo_groups.jsonl" \
  --inspect-artifacts \
  --require-replay-context \
  --out-summary "$SOURCE/groups/grpo_dataset_validation_actor_replay.json" \
  --fail-on-error

python tools/summarize_grpo_groups.py \
  "$SOURCE/groups/grpo_groups.jsonl" \
  --inspect-artifacts \
  --out-json "$SOURCE/groups/grpo_group_summary_actor_replay.json" \
  --out-csv "$SOURCE/groups/grpo_group_summary_actor_replay.csv" \
  --out-markdown "$SOURCE/groups/grpo_group_summary_actor_replay.md"
CONTAINER
```

For storage-limited replay-context collection, limit capture at collection
time instead of collecting every action chunk:

```bash
STRICT_GRPO_SAVE_REPLAY_CONTEXT=true
STRICT_GRPO_CAPTURE_CHUNK_STRIDE=2   # keep every second action chunk
STRICT_GRPO_CAPTURE_MAX_CHUNKS=4     # or keep only the first four chunks
```

Defaults preserve previous behavior: stride `1` and max chunks `0` means no
chunk-level filtering. Filtering reduces the number of trainable replay
transitions, so use it first for smoke/debug or when Scratch pressure would
otherwise make the run fail.

Promotion criteria for a source dataset:

- validation `ok=true`;
- `transition_count > 0`;
- at least one mixed group;
- non-empty success and failure samples;
- source `server_vis/` is still reachable.

## 2. Prepare A Lightweight Materialized Subset

Do not start real actor replay training from a full k=8 replay-context run by
default. First create a small subset with one success and one failure sample.

```bash
SOURCE_GROUPS_PATH=/home/zcably0/Scratch/wam-rl/results_grouped_rollouts/<run-id>/groups/grpo_groups.jsonl \
SUBSET_ROOT=/home/zcably0/Scratch/wam-rl/results_grpo_actor_replay_subsets/<subset-name> \
SUBSET_TASKS="move_stapler_pad" \
SUBSET_MAX_GROUPS=1 \
SUBSET_SAMPLES_PER_REWARD=1 \
SUBSET_MAX_ARTIFACTS_PER_SAMPLE=2 \
MATERIALIZE_LINK_MODE=symlink \
MATERIALIZE_INCLUDE_REPLAY_CONTEXT=true \
qsub -V -N wam_grpo_subset jobs/myriad/35_prepare_actor_replay_subset.sh
```

For quick direct execution on a login node or already allocated node, run the
same script with the same environment variables through `bash` instead of
`qsub`; it only rewrites JSON and creates symlinks/copies.

The subset preparation job writes:

```text
groups/grpo_groups.jsonl
manifest.json
validation_path_only.json
storage_audit.json
```

## 3. Audit Subset Storage Dependencies

For symlink materialized subsets, the subset directory can be tiny while still
depending on large original replay-context files. Audit before cleanup:

```bash
SUBSET=/home/zcably0/Scratch/wam-rl/results_grpo_actor_replay_subsets/<subset-name>

python tools/audit_grpo_artifact_storage.py \
  "$SUBSET/groups/grpo_groups.jsonl" \
  --materialize-manifest "$SUBSET/manifest.json" \
  --out-json "$SUBSET/storage_audit.json" \
  --fail-on-missing
```

Interpretation:

- `apparent_bytes`: bytes consumed by the listed files or symlinks themselves;
- `resolved_bytes`: bytes after following symlinks to original targets;
- `missing_count > 0`: do not train; restore or rebuild the subset.

If `materialized_replay_contexts.resolved_bytes` is large, the source
`server_vis/` is still required even if the subset directory is small.

## 4. Train One-Step Actor Replay Smoke

Submit a low-resource smoke job from the materialized subset:

```bash
SUBSET_ROOT=/home/zcably0/Scratch/wam-rl/results_grpo_actor_replay_subsets/<subset-name> \
RUN_ID=grpo_actor_subset_smoke_<date> \
jobs/myriad/36_submit_actor_replay_subset_smoke.sh
```

Defaults:

```text
GRPO_STEPS=1
GRPO_LR=1e-7
GRPO_ACTION_NUM_INFERENCE_STEPS=10
GRPO_LOGPROB_REDUCTION=mean
GRPO_LOGPROB_STD_FLOOR=0.1
QSUB_H_RT=2:00:00
QSUB_SLOTS=4
QSUB_TMPFS=40G
QSUB_GPU=1
```

Only raise `GRPO_STEPS`, `GRPO_LR`, or queue resources after a one-step smoke
passes.

## 5. Inspect Actor Replay Output

After the job finishes:

```bash
LOG=$(find logs/jobs -maxdepth 1 -type f -name 'wam_grpo_actor_subset.o*' | sort | tail -1)
grep -E "JOB_ID=|GRPO_GROUPS_PATH=|GRPO_OUTPUT_DIR=|transition_count|final_loss|checkpoint_path|Actor replay GRPO training complete|Traceback|ERROR|CUDA out of memory|Disk quota exceeded" "$LOG" | tail -200

OUT=$(grep '^GRPO_OUTPUT_DIR=' "$LOG" | tail -1 | cut -d= -f2-)
cat "$OUT/input_dataset_validation.json"
cat "$OUT/metrics.json" | tail -120
ls -lh "$OUT"

python tools/summarize_actor_replay_training.py \
  "$OUT" \
  --out-json "$OUT/summary.json" \
  --out-markdown "$OUT/summary.md"

python tools/inspect_actor_replay_checkpoint.py \
  "$OUT/checkpoint.pt" \
  --out-json "$OUT/checkpoint_inspection.json" \
  --out-markdown "$OUT/checkpoint_inspection.md"
```

Pass criteria:

- `input_dataset_validation.json` has `ok=true`;
- a `checkpoint.pt` exists;
- `metrics.json` exists and has finite loss/ratio diagnostics;
- no `Traceback`, `CUDA out of memory`, or `Disk quota exceeded`.

For no-op controls or regression debugging, compare checkpoints without loading
the full LingBot model:

```bash
python tools/inspect_actor_replay_checkpoint.py \
  /path/to/candidate/checkpoint.pt \
  --reference /path/to/lr0_or_previous/checkpoint.pt \
  --out-json /path/to/checkpoint_compare.json \
  --out-markdown /path/to/checkpoint_compare.md
```

Interpret `delta_l2_norm`, `relative_delta_l2`, and `delta_max_abs` before
claiming that a closed-loop eval difference came from a meaningful policy
update.

## 6. Evaluate And Compare

Use fixed prompt and sampling controls for paired smoke evaluation:

```bash
PROMPT_INDEX=0
SEED=0
SAMPLING_SEED=12345
SAMPLING_SEED_PER_ENV=true
ACTION_NUM_INFERENCE_STEPS=10
SAVE_SERVER_DEBUG_TENSORS=false
```

`SEED` controls RoboTwin's starting env seed through
`st_seed = 10000 * (1 + SEED)`. Keep it explicit in paired comparisons; do not
depend on `qsub -V` inheriting whatever `SEED` happened to be in the shell.

Submit baseline and actor checkpoint evals on the same task, prompt, env seeds,
and sampling seed policy:

```bash
ACTOR_REPLAY_CHECKPOINT_PATH=/path/to/actor/checkpoint.pt \
RUN_ID=actor_eval_pair_<date> \
TASK_NAME=move_stapler_pad \
TEST_NUM=2 \
SEED=0 \
jobs/myriad/37_submit_actor_eval_pair_smoke.sh
```

After both jobs finish, summarize aggregate results, per-episode exports, and
matched comparisons in one pass:

```bash
python tools/summarize_actor_eval_pair.py \
  --baseline /path/to/baseline_eval \
  --actor /path/to/actor_eval \
  --out-root /path/to/comparison_dir
```

The summary command fails by default if zero matched episodes are found. Treat
that as a control mismatch, usually `SEED`, `PROMPT_INDEX`, or
`SAMPLING_SEED`. Use `--min-matched-episodes 0` only when intentionally doing
aggregate-only inspection.

Treat `n <= 5` as smoke only. Because RoboTwin closed-loop execution can diverge
even after identical first actions, use matched per-episode comparisons before
interpreting aggregate success rates.

For baseline repeatability controls across two or more repeated eval roots:

```bash
RUN_ID=baseline_repeatability_<date> \
TASK_NAME=move_stapler_pad \
TEST_NUM=10 \
SEED=0 \
jobs/myriad/38_submit_eval_repeatability_pair.sh

python tools/summarize_robotwin_repeatability.py \
  --run baseline_a=/path/to/baseline_eval_a \
  --run baseline_b=/path/to/baseline_eval_b \
  --out-json /path/to/repeatability.json \
  --out-csv /path/to/repeatability.csv \
  --out-markdown /path/to/repeatability.md
```

Use `flipped_count` and `flip_rate` to decide whether a policy delta is larger
than the closed-loop repeat noise on the same matched episode keys.

Before promoting an actor checkpoint beyond smoke status, gate the paired eval
against the baseline repeatability control:

```bash
python tools/gate_actor_eval_promotion.py \
  --comparison /path/to/comparison_dir/comparison.json \
  --baseline-repeatability /path/to/repeatability.json \
  --out-json /path/to/comparison_dir/promotion_gate.json \
  --out-markdown /path/to/comparison_dir/promotion_gate.md
```

The default gate requires at least 10 matched eval episodes, at least 10 matched
baseline-repeatability episodes, baseline `flip_rate <= 0.1`, and a candidate
net improvement rate above the observed baseline flip rate. Lower thresholds
are acceptable only for debugging; do not use them for improvement claims.

## 7. Cleanup Boundary

Before deleting anything:

```bash
qstat
python tools/audit_grpo_artifact_storage.py \
  "$SUBSET/groups/grpo_groups.jsonl" \
  --materialize-manifest "$SUBSET/manifest.json" \
  --fail-on-missing
```

Do not delete a source run's `server_vis/` while:

- a non-empty `grpo_groups.jsonl` references it;
- a symlink materialized subset depends on it;
- an active queued/running job may write to or read from it.

Safe cleanup candidates are failed replay-context attempts, all-success or
all-failure replay-context runs with empty `grpo_groups.jsonl`, and old debug
directories after preserving `groups/`, `attempts/`, and job logs.
