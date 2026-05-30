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

python tools/summarize_grpo_replay_contexts.py \
  "$SOURCE/groups/grpo_groups.jsonl" \
  --inspect-artifacts \
  --print-summary \
  --out-json "$SOURCE/groups/grpo_replay_context_summary.json" \
  --out-csv "$SOURCE/groups/grpo_replay_context_summary.csv" \
  --out-markdown "$SOURCE/groups/grpo_replay_context_summary.md"
CONTAINER
```

This command only stats replay-context files by default. Add
`--inspect-context-tensors` only for small or carefully bounded sources; even
metadata-only `torch.load` can cause heavy filesystem IO on multi-GB
replay-context files.

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

For the common single-task actor-replay data collection case, prefer the
bounded wrapper and dry-run it before submission:

```bash
DRY_RUN=1 \
TASK_NAMES="move_stapler_pad" \
GROUP_SIZE=8 \
GROUPS_PER_TASK=1 \
bash jobs/myriad/39_submit_grpo_replayctx_bounded_4gpu.sh
```

Its defaults use `ACTION_NUM_INFERENCE_STEPS=10`,
`STRICT_GRPO_CAPTURE_MAX_CHUNKS=1`, `STRICT_GRPO_SAVE_REPLAY_CONTEXT=true`, and
`SAVE_SERVER_DEBUG_TENSORS=false`. This preserves a real replay-context smoke
contract while avoiding full-trajectory, all-debug-tensor collection by
default. The wrapper also prints a replay-context storage estimate. The default
estimate is 4.0GB/context for new action-scale-one collections, where replay
context capture prunes the unused CFG negative action branch. Override
`REPLAY_CONTEXT_ESTIMATE_GB` upward for action-guided (`action_guidance_scale >
1`) or unpruned legacy collections.

The storage gate defaults to `STORAGE_BUDGET_MODE=attempt`, so it budgets for
the configured seed-search attempt budget, not only the final accepted group.
This is intentionally conservative because discarded failed attempts can still
leave large `server_vis/` replay-context files behind. Use
`STORAGE_BUDGET_MODE=accepted` only after explicit review.
The same estimate can be run without the submit wrapper:

```bash
python tools/plan_replay_context_collection.py \
  --task-names "move_stapler_pad" \
  --group-size 4 \
  --groups-per-task 1 \
  --group-max-attempts 1 \
  --capture-max-chunks 1 \
  --save-replay-context true \
  --replay-context-estimate-gb 4 \
  --storage-budget-mode attempt \
  --check-scratch-headroom true \
  --scratch-path /home/zcably0/Scratch \
  --min-scratch-headroom-gb 50 \
  --dry-run true \
  --format shell
```

The bounded wrapper also passes `STRICT_GRPO_REPLAY_CONTEXT_MAX_GB=5.0` by
default. The server checks the tensor bytes of each replay context before
calling `torch.save`; if a context exceeds that per-file budget, the attempt
fails early with a clear error instead of filling Scratch until `Disk quota
exceeded`. Increase this only after reviewing
`tools/inspect_grpo_replay_context.py` output.

Before choosing `GROUP_SIZE`, estimate the probability of getting a mixed
success/failure group from recent per-task success rates:

```bash
python tools/estimate_group_mixing.py \
  --summary /path/to/grpo_group_summary_actor_replay.json \
  --group-sizes 4 8
```

This estimate is not a simulator guarantee; it is a planning check for the
storage-vs-mixed-group tradeoff.

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
SUBSET_MAX_REPLAY_CONTEXT_GB=30 \
SUBSET_STORAGE_MAX_RESOLVED_GB=40 \
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
validation_actor_replay.json
storage_audit.json
```

`manifest.json` includes `selection_details`, a compact per-group/per-sample
record of retained `sample_idx`, reward, advantage, seeds, and artifact counts.
Use it to confirm the subset kept both success and failure samples before
spending GPU time on actor replay training.

`SUBSET_MAX_REPLAY_CONTEXT_GB` is a resolved replay-context footprint budget
for the selected subset. It trims artifact references round-robin across the
selected success/failure samples, which keeps a tiny actor replay smoke useful
without accidentally depending on hundreds of GB of KV-cache files.
`SUBSET_STORAGE_MAX_RESOLVED_GB` is a second guard on the final materialized
artifact plus replay-context dependency footprint.
By default the subset job validates strict artifacts with
`--require-replay-context`; external replay contexts are checked with
metadata-only loading so the validation count reflects expanded transitions
without allocating the full KV-cache tensors.

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
jobs/myriad/36_submit_actor_replay_subset_smoke.sh --dry-run
```

Review the printed `qsub` command and rerun without `--dry-run` only after the
subset storage audit and queue request look correct.

The submitter checks that `groups/grpo_groups.jsonl` exists before `qsub`. If
`${SUBSET_ROOT}/storage_audit.json` exists, it also checks `storage_budget.ok`
before submission; set `PRECHECK_SUBSET_AUDIT=false` only for manual debugging.

Defaults:

```text
GRPO_STEPS=1
GRPO_LR=1e-7
GRPO_ACTION_NUM_INFERENCE_STEPS=10
GRPO_LOGPROB_REDUCTION=mean
GRPO_LOGPROB_STD_FLOOR=0.1
GRPO_MAX_RESOLVED_GB=40
QSUB_H_RT=4:00:00
QSUB_SLOTS=4
QSUB_MEM=16G
QSUB_TMPFS=60G
QSUB_GPU=1
```

Only raise `GRPO_STEPS`, `GRPO_LR`, or queue resources after a one-step smoke
passes. The trainer job writes `input_storage_audit.json` before training and
uses `GRPO_MAX_RESOLVED_GB` to fail fast when strict artifacts plus
replay-context files exceed the intended input budget.

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
  --out-csv "$OUT/summary.csv" \
  --out-markdown "$OUT/summary.md"

python tools/inspect_actor_replay_checkpoint.py \
  "$OUT/checkpoint.pt" \
  --out-json "$OUT/checkpoint_inspection.json" \
  --out-markdown "$OUT/checkpoint_inspection.md"
```

For a sweep table over recent actor replay runs:

```bash
python tools/summarize_actor_replay_training.py \
  --discover-root /home/zcably0/Scratch/wam-rl/results_grpo_actor_replay \
  --latest 20 \
  --print-format table \
  --out-json /home/zcably0/Scratch/wam-rl/results_grpo_actor_replay/recent_summary.json \
  --out-csv /home/zcably0/Scratch/wam-rl/results_grpo_actor_replay/recent_summary.csv \
  --out-markdown /home/zcably0/Scratch/wam-rl/results_grpo_actor_replay/recent_summary.md
```

`summary.json` / `summary.csv` record the training config and provenance saved
in `metrics.json` for new runs (`model_path`, `config_name`, `git_commit`,
`learning_rate`, `action_num_inference_steps`, `logprob_reduction`,
`logprob_std_floor`, and `trainable_mode`). If an older run's `metrics.json`
predates config logging, the summary tool falls back to the lightweight
`checkpoint.pt` config and marks `config_source=checkpoint`.

The checkpoint inspection report includes trainable tensor statistics and, for
new checkpoints, the saved actor replay training config/provenance
(`model_path`, `config_name`, `git_commit`, `learning_rate`,
`action_num_inference_steps`, `logprob_reduction`, and `trainable_mode`).

Pass criteria:

- `input_dataset_validation.json` has `ok=true`;
- a `checkpoint.pt` exists;
- `metrics.json` exists and has finite loss/ratio diagnostics;
- `summary.json` has `parameter_update_detected=true` for nonzero-learning-rate
  smoke runs; `ok=true` alone only means the output/validation/checkpoint chain
  completed;
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
jobs/myriad/37_submit_actor_eval_pair_smoke.sh --dry-run
```

Review the two printed `qsub` commands and rerun without `--dry-run` when the
ports, output roots, seed controls, and checkpoint path are correct.

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
jobs/myriad/38_submit_eval_repeatability_pair.sh --dry-run

python tools/summarize_robotwin_repeatability.py \
  --run baseline_a=/path/to/baseline_eval_a \
  --run baseline_b=/path/to/baseline_eval_b \
  --out-json /path/to/repeatability.json \
  --out-csv /path/to/repeatability.csv \
  --out-markdown /path/to/repeatability.md
```

Review the printed `qsub` commands and rerun the submitter without `--dry-run`
when the ports, output roots, and seed controls are correct.

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
  --inspect-replay-contexts \
  --print-summary \
  --fail-on-missing
```

For a pre-run storage budget check, add `--max-resolved-gb <GB>`. With
`--inspect-replay-contexts`, the budget covers both directly referenced strict
artifacts and replay-context files referenced from inside those artifacts. Use
`--omit-replay-context-mapping` when writing a long-lived JSON report where the
per-artifact mapping is not needed.

To inspect why a replay context is large, run:

```bash
python tools/inspect_grpo_replay_context.py \
  /path/to/strict_grpo_replay_context_0.pt \
  --metadata-only \
  --print-summary \
  --out-json /path/to/replay_context_inspection.json \
  --out-markdown /path/to/replay_context_inspection.md
```

Use the top-level tensor-byte breakdown before changing capture format or
compression policy. For large context files, keep `--metadata-only` enabled so
the inspection does not allocate the full KV-cache tensors on CPU.
The compact `--print-summary` output includes scalar fields, top-level tensor
GiB, KV-cache batch sizes, and a conditional-only branch estimate; if
`action_guidance_scale<=1`, future collection should use the conditional-only
replay context path rather than storing both CFG branches.

Do not delete a source run's `server_vis/` while:

- a non-empty `grpo_groups*.jsonl` references it;
- a symlink materialized subset depends on it;
- an active queued/running job may write to or read from it.

Safe cleanup candidates are failed replay-context attempts, all-success or
all-failure replay-context runs with empty `grpo_groups*.jsonl`, and old debug
directories after preserving `groups/`, `attempts/`, and job logs.

Before proposing cleanup, run the non-destructive planner:

```bash
python tools/plan_myriad_storage_cleanup.py \
  /home/zcably0/Scratch/wam-rl/results_grouped_rollouts \
  --min-candidate-gb 1 \
  --large-run-gb 10 \
  --print-summary
```

If this reports no candidates, do not delete large source runs just because they
look old. They may be protected by `grpo_groups_partial.jsonl` or
`grpo_groups_accepted*.jsonl` files that are still part of curated datasets.
