# WAM-RL Myriad Storage Policy

This document defines how WAM-RL data should be stored on Myriad while the
project is in the grouped-rollout and actor-replay GRPO phase.

## Storage Tiers

Use Myriad Scratch for active jobs:

```text
/home/zcably0/Scratch/WAM-RL
/home/zcably0/Scratch/wam-rl
```

Use RDSS only as cold archive:

```text
/rdss/rd00/<project-path>/wam-rl
```

Do not run rollout collection, `torch.save`-heavy capture, or training directly
against RDSS. Active jobs should write to Scratch first. After a job completes,
archive selected outputs to RDSS with `rsync`.

## Why This Matters

Replay-context GRPO artifacts can be extremely large. A small
`STRICT_GRPO_SAVE_REPLAY_CONTEXT=true` smoke run with 10 action denoising steps
produced tens of GB because each saved replay context includes a transformer KV
cache. Full multi-task or 50-step replay-context runs can exceed Scratch quota
or hit `Disk quota exceeded` during `torch.save`.

The storage policy is therefore part of the experiment design, not a cosmetic
cleanup step.

## Keep On Scratch

Keep these on Scratch while they are actively used by training or validation:

- current `groups/grpo_groups.jsonl` inputs;
- strict artifact files referenced by a current `grpo_groups.jsonl`;
- replay-context files referenced by current actor-replay artifacts;
- current actor replay smoke outputs;
- current job logs needed to debug a running or just-failed job.

Do not move a trainable dataset to RDSS unless either:

- every referenced artifact path remains valid; or
- the dataset is copied as a whole and paths are rewritten or resolved by a
  tool that understands the new location.

`grpo_groups.jsonl` stores artifact paths. Moving only `groups/` is enough for
summary/audit records but not enough for training.

## Archive To RDSS

Archive these to RDSS when they are no longer active:

- `results_grpo_datasets/`;
- `results_grpo_train/`;
- `results_grpo_actor_replay/`;
- `logs/jobs/`;
- `groups/`, `attempts/`, summaries, manifests, and validation JSON files from
  completed rollout collections.

Example:

```bash
RDSS_ROOT=/rdss/rd00/<project-path>/wam-rl
mkdir -p "$RDSS_ROOT"

rsync --safe-links -a \
  /home/zcably0/Scratch/wam-rl/results_grpo_datasets/ \
  "$RDSS_ROOT/results_grpo_datasets/"

rsync --safe-links -a \
  /home/zcably0/Scratch/wam-rl/results_grpo_train/ \
  "$RDSS_ROOT/results_grpo_train/"

rsync --safe-links -a \
  /home/zcably0/Scratch/WAM-RL/logs/jobs/ \
  "$RDSS_ROOT/logs/jobs/"
```

For a completed grouped rollout where only paper/audit metadata is needed:

```bash
SRC=/home/zcably0/Scratch/wam-rl/results_grouped_rollouts/<run-id>
DST="$RDSS_ROOT/results_grouped_rollouts/$(basename "$SRC")"

mkdir -p "$DST"
rsync --safe-links -a "$SRC/groups/" "$DST/groups/"
rsync --safe-links -a "$SRC/attempts/" "$DST/attempts/"
```

## Safe Cleanup Rules

Before deleting anything, check `qstat` to ensure no running job is writing to
the target directory.

Usually safe to delete:

- `server_vis/` from failed replay-context runs after saving job logs and
  `groups/failed_attempt_roots.txt`;
- `server_vis/` from all-success or all-failure replay-context runs when
  every `groups/grpo_groups*.jsonl` file has zero lines;
- old debug/sanity rollout directories that are not used by the current
  combined dataset.

Do not delete without explicit review:

- A/B/M/N grouped rollout datasets used to build the current combined GRPO
  dataset;
- any `server_vis/` directory referenced by a non-empty current
  `grpo_groups*.jsonl`;
- `results_grpo_datasets/a_b_m_n_*` and other curated combined datasets;
- actor replay outputs that contain a real checkpoint to evaluate.

Before manual cleanup, generate a non-destructive candidate report. The planner
checks all `groups/grpo_groups*.jsonl` files, so partial or accepted group files
still protect their source artifacts:

```bash
python tools/plan_myriad_storage_cleanup.py \
  /home/zcably0/Scratch/wam-rl/results_grouped_rollouts \
  --min-candidate-gb 1 \
  --large-run-gb 10 \
  --out-json /home/zcably0/Scratch/wam-rl/debug_logs/storage_cleanup_plan.json \
  --out-markdown /home/zcably0/Scratch/wam-rl/debug_logs/storage_cleanup_plan.md \
  --print-summary
```

Treat this as a review artifact, not permission to delete. A run with a
non-empty `grpo_groups_partial.jsonl` or `grpo_groups_accepted*.jsonl` can still
be part of a trainable combined dataset even when canonical `grpo_groups.jsonl`
is empty or missing.
The compact `--print-summary` output includes both `top_candidates` and
`top_protected_runs`; a large protected run needs explicit archive/delete review
because it may still back an active trainable dataset.

Targeted cleanup pattern:

```bash
cd /home/zcably0/Scratch/WAM-RL
qstat

RUN=/home/zcably0/Scratch/wam-rl/results_grouped_rollouts/<run-id>
mkdir -p /home/zcably0/Scratch/wam-rl/debug_logs/<run-id>
cp -r "$RUN/groups" /home/zcably0/Scratch/wam-rl/debug_logs/<run-id>/groups 2>/dev/null || true
cp logs/jobs/<job-log> /home/zcably0/Scratch/wam-rl/debug_logs/<run-id>/ 2>/dev/null || true

rm -rf "$RUN/server_vis"
```

## Lightweight Actor-Replay Subsets

For real actor replay debugging, do not train directly from a huge k=8
replay-context run unless the full dataset is intentionally needed. First build
a small JSON-only subset, then materialize only the referenced files.

For repeatable Myriad runs, use:

```bash
SOURCE_GROUPS_PATH=/path/to/grpo_groups.jsonl \
SUBSET_ROOT=/home/zcably0/Scratch/wam-rl/results_grpo_actor_replay_subsets/<name> \
SUBSET_TASKS="move_stapler_pad" \
SUBSET_MAX_GROUPS=1 \
SUBSET_SAMPLES_PER_REWARD=1 \
SUBSET_MAX_ARTIFACTS_PER_SAMPLE=2 \
SUBSET_MAX_REPLAY_CONTEXT_GB=30 \
SUBSET_STORAGE_MAX_RESOLVED_GB=40 \
jobs/myriad/35_submit_prepare_actor_replay_subset.sh --dry-run
```

Review the printed `qsub` command and rerun without `--dry-run` after checking
the explicit `-v` variable list. Keep `QSUB_EXPORT_CURRENT_ENV=0` unless you
deliberately need `qsub -V` for manual debugging.

This job writes the rewritten groups file, materialization manifest,
actor-replay validation summary, and `storage_audit.json` under `SUBSET_ROOT`.
Keep `SUBSET_MAX_REPLAY_CONTEXT_GB` below the training-side
`GRPO_MAX_RESOLVED_GB` budget; the submitter defaults are currently 30GB for
subset preparation and 40GB for the actor replay smoke trainer.
`SUBSET_STORAGE_MAX_RESOLVED_GB` enforces the same kind of budget during subset
materialization by counting strict artifacts plus materialized replay-context
symlink targets.

### Self-Contained Subset Before Source Cleanup

The default subset flow uses `MATERIALIZE_LINK_MODE=symlink`, so the subset can
be only a few KB while still depending on large source `server_vis/` replay
contexts. Do not delete the source run while any active subset or checkpoint
training record still points through those symlinks.

If the goal is to eventually reclaim a large source run, first prepare a
self-contained copy subset with explicit storage headroom:

Before submitting the subset job, you can also run the materializer itself in
`--dry-run` mode. This resolves the artifact and replay-context mapping and
prints the planned output paths without creating `SUBSET_ROOT` or copying
files. In `--link-mode copy`, check `planned_copy_gb` before submitting the
real copy job:

```bash
python tools/materialize_grpo_artifacts.py \
  /path/to/source/groups/grpo_groups.jsonl \
  --out-root /home/zcably0/Scratch/wam-rl/results_grpo_actor_replay_subsets/<name>_copy \
  --link-mode copy \
  --include-replay-context \
  --dry-run
```

```bash
SOURCE_GROUPS_PATH=/path/to/source/groups/grpo_groups.jsonl \
SUBSET_ROOT=/home/zcably0/Scratch/wam-rl/results_grpo_actor_replay_subsets/<name>_copy \
SUBSET_TASKS="move_stapler_pad" \
SUBSET_MAX_GROUPS=1 \
SUBSET_SAMPLES_PER_REWARD=1 \
SUBSET_MAX_ARTIFACTS_PER_SAMPLE=2 \
SUBSET_MAX_REPLAY_CONTEXT_GB=30 \
SUBSET_STORAGE_MAX_RESOLVED_GB=40 \
MATERIALIZE_LINK_MODE=copy \
jobs/myriad/35_submit_prepare_actor_replay_subset.sh --dry-run
```

Only rerun without `--dry-run` after checking `df -h /home/zcably0/Scratch`.
For the current staplerpad two-sample/two-artifact subset, the resolved replay
context footprint is about 29GB; copying it is useful only if it enables a later
reviewed cleanup of the much larger source `server_vis/`.

After a copy subset finishes, validate that it is self-contained:

```bash
python tools/audit_grpo_artifact_storage.py \
  "$SUBSET_ROOT/groups/grpo_groups.jsonl" \
  --materialize-manifest "$SUBSET_ROOT/manifest.json" \
  --out-json "$SUBSET_ROOT/storage_audit.json" \
  --fail-on-missing \
  --max-resolved-gb 40
```

Then re-run `tools/plan_myriad_storage_cleanup.py`. Treat any source cleanup as
a separate reviewed action; the planner deliberately does not emit destructive
commands.

Then submit a low-resource one-step actor replay smoke from the materialized
subset:

```bash
SUBSET_ROOT=/home/zcably0/Scratch/wam-rl/results_grpo_actor_replay_subsets/<name> \
RUN_ID=grpo_actor_subset_smoke_<name> \
jobs/myriad/36_submit_actor_replay_subset_smoke.sh --dry-run
```

The submitter checks the subset groups file and, when available, the
`storage_audit.json` `storage_budget.ok` flag before `qsub`.
Review the printed command and rerun without `--dry-run` only after the storage
audit and queue request are acceptable.
The printed `qsub` command uses explicit `-v` variables by default instead of
`qsub -V`; do not enable `QSUB_EXPORT_CURRENT_ENV=1` unless you intentionally
want to inherit the submit shell. This prevents stale `GRPO_*` or debug
settings from changing a smoke run.

`36_submit_actor_replay_subset_smoke.sh` only wraps `qsub`; the real training
path remains `jobs/myriad/34_train_actor_replay_grpo_robotwin.sh`. Override
`GRPO_STEPS`, `GRPO_LR`, `QSUB_H_RT`, `QSUB_SLOTS`, or `QSUB_TMPFS` only when a
smoke run has already passed and the next experiment needs more work.

Example for one success and one failure sample, with two artifact refs per
sample:

```bash
RUN=/home/zcably0/Scratch/wam-rl/results_grouped_rollouts/<replayctx-run>
SUBSET=/home/zcably0/Scratch/wam-rl/results_grpo_actor_replay_subsets/<name>

python tools/subset_grpo_groups.py \
  "$RUN/groups/grpo_groups.jsonl" \
  --tasks move_stapler_pad \
  --max-groups 1 \
  --samples-per-reward 1 \
  --max-artifacts-per-sample 2 \
  --require-artifacts \
  --out-jsonl "$RUN/groups/grpo_groups_actor_subset_2samples_2artifacts.jsonl" \
  --out-manifest "$RUN/groups/grpo_groups_actor_subset_2samples_2artifacts_manifest.json"

python tools/materialize_grpo_artifacts.py \
  "$RUN/groups/grpo_groups_actor_subset_2samples_2artifacts.jsonl" \
  --out-root "$SUBSET" \
  --include-replay-context \
  --link-mode symlink \
  --overwrite

python tools/validate_grpo_dataset.py \
  "$SUBSET/groups/grpo_groups.jsonl" \
  --out-summary "$SUBSET/validation_path_only.json" \
  --fail-on-error
```

Use `--link-mode symlink` for Scratch debug runs. This keeps disk usage tiny
because the subset references the original strict artifacts and replay-context
files. Use `--link-mode copy` only when deliberately creating an archive or
portable package, because copy mode can duplicate many GB.

`--include-replay-context` needs `torch` to read each strict artifact's
`replay_context_path` metadata. Run it in the WAM-RL container or another Python
environment with `torch`.

For future replay-context collection, reduce storage at the source when the goal
is smoke/debug or a bounded-size actor replay dataset:

```bash
STRICT_GRPO_SAVE_REPLAY_CONTEXT=true \
STRICT_GRPO_CAPTURE_CHUNK_STRIDE=2 \
STRICT_GRPO_CAPTURE_MAX_CHUNKS=4 \
bash jobs/myriad/32_submit_grpo_scale_8tasks_4gpu.sh
```

`STRICT_GRPO_CAPTURE_CHUNK_STRIDE=1` and `STRICT_GRPO_CAPTURE_MAX_CHUNKS=0`
preserve previous behavior. Increasing the stride or setting a max chunk count
reduces replay-context files and trainable transitions; it should be treated as
a storage/compute tradeoff, not as an algorithmic improvement.

For actor-replay collection smoke runs, prefer:

```bash
DRY_RUN=1 bash jobs/myriad/39_submit_grpo_replayctx_bounded_4gpu.sh
```

Then review the printed command and rerun with `DRY_RUN=0` only after checking
queue and Scratch headroom. The wrapper defaults to one task, one group, k=8,
10 action steps, one captured chunk per rollout, replay-context capture enabled,
and server debug tensors disabled. It prints accepted-run and attempt-budget
storage estimates before submission and fails fast on non-dry-run submission if
the accepted-run estimate plus the configured Scratch headroom does not fit.
It also sets `STRICT_GRPO_REPLAY_CONTEXT_MAX_GB=5.0` by default so the server
checks each replay context before `torch.save`; if the live tensor footprint is
larger than the reviewed per-file budget, the attempt fails before filling
Scratch.

Actor replay training jobs also audit their input storage before loading the
model. Set `GRPO_MAX_RESOLVED_GB=<GB>` for a hard budget; the subset smoke
submitter defaults this to `40` GiB. Leave it at `0` only for an explicitly
reviewed full-source training run.

For pure RoboTwin eval jobs, keep `SAVE_SERVER_DEBUG_TENSORS=false` unless you
need per-chunk `latents_*.pt`, `actions_*.pt`, or `obs_data_*.pt` for a
specific diagnosis. Episode JSON, executed actions, metrics, and videos are
still written by the eval client.

Do not delete the source run's `server_vis/` while a symlink materialized subset
is still active. The symlink package depends on the original artifact files.

## Monitoring Commands

Check available Scratch space:

```bash
df -h /home/zcably0/Scratch /myriadfs/home/zcably0/Scratch 2>/dev/null || true
```

Find large grouped rollout directories:

```bash
du -sh /home/zcably0/Scratch/wam-rl/results_grouped_rollouts/* 2>/dev/null \
  | sort -h | tail -40
```

Inspect replay-context usage:

```bash
du -sh /home/zcably0/Scratch/wam-rl/results_grouped_rollouts/*replayctx* 2>/dev/null \
  | sort -h
```

Plan cleanup candidates without deleting files:

```bash
python tools/plan_myriad_storage_cleanup.py \
  /home/zcably0/Scratch/wam-rl/results_grouped_rollouts \
  --min-candidate-gb 1 \
  --large-run-gb 10 \
  --print-summary
```

In the summary, `cleanup_candidate_count=0` does not mean Scratch is healthy.
Check `top_protected_runs` as well; those are the large directories that are
protected by non-empty group files or curated-source naming.

For future replay-context collection, the bounded submitter uses
`STORAGE_BUDGET_MODE=attempt` by default. This budgets for seed-search attempts,
not only accepted groups, because failed/discarded attempts can still leave
large `server_vis/` files. Switching to `STORAGE_BUDGET_MODE=accepted` is a
manual risk decision, not a default.

List largest files without breaking paths containing spaces:

```bash
python - <<'PY'
from pathlib import Path

root = Path("/home/zcably0/Scratch/wam-rl/results_grouped_rollouts")
for path in sorted(root.rglob("*.pt"), key=lambda p: p.stat().st_size, reverse=True)[:30]:
    print(f"{path.stat().st_size / 1024**3:.2f} GB  {path}")
PY
```

Audit a GRPO groups file or materialized subset without loading `.pt` files:

```bash
python tools/audit_grpo_artifact_storage.py \
  /path/to/groups/grpo_groups.jsonl \
  --out-json /path/to/storage_audit.json \
  --fail-on-missing
```

For materialized actor replay subsets, include the manifest to report both
strict artifacts and replay-context dependencies:

```bash
python tools/audit_grpo_artifact_storage.py \
  "$SUBSET/groups/grpo_groups.jsonl" \
  --materialize-manifest "$SUBSET/manifest.json" \
  --out-json "$SUBSET/storage_audit.json" \
  --fail-on-missing
```

For real actor replay data, inspect replay-context references stored inside the
strict artifacts and keep stdout compact:

```bash
python tools/audit_grpo_artifact_storage.py \
  /path/to/groups/grpo_groups.jsonl \
  --inspect-replay-contexts \
  --omit-replay-context-mapping \
  --print-summary \
  --out-json /path/to/storage_audit_with_replay_contexts.json \
  --max-resolved-gb 500 \
  --fail-on-missing
```

`--max-resolved-gb` exits non-zero if the resolved artifact footprint exceeds
the budget. With `--inspect-replay-contexts`, the budget covers strict artifacts
plus their replay-context files.

Inspect one replay-context file before changing capture/compression policy:

```bash
python tools/inspect_grpo_replay_context.py \
  /path/to/strict_grpo_replay_context_0.pt \
  --metadata-only \
  --print-summary \
  --out-json /path/to/replay_context_inspection.json \
  --out-markdown /path/to/replay_context_inspection.md
```

Use `--metadata-only` on large replay-context files. It loads tensors on the
`meta` device and reports shapes, dtypes, and byte counts without allocating the
full tensor storage on CPU. On the staplerpad k=8 replay-context run, a
representative context file was about 7.23GB, with about 7.22GB in
`transformer_cache` tensors. Use `--print-summary` for a compact stdout view of
file GiB, tensor GiB, scalar config fields, top-level tensor GiB, KV-cache batch
sizes, and the conditional-only branch estimate. If the KV batch size is 2 but
`action_guidance_scale<=1`, future collection should use the pruned conditional
branch instead of storing both CFG branches.

Summarize all replay-context files referenced by a groups file without loading
their tensor metadata:

```bash
python tools/summarize_grpo_replay_contexts.py \
  /path/to/groups/grpo_groups.jsonl \
  --inspect-artifacts \
  --print-summary \
  --out-json /path/to/groups/grpo_replay_context_summary.json \
  --out-csv /path/to/groups/grpo_replay_context_summary.csv \
  --out-markdown /path/to/groups/grpo_replay_context_summary.md
```

This command stats replay-context files by default and is suitable for large
legacy sources. Add `--inspect-context-tensors` only for small or explicitly
bounded sources. On multi-GB replay-context files, metadata-only `torch.load`
can still create heavy filesystem IO even though it avoids CPU tensor
allocation.

In the report, `apparent_bytes` is the bytes consumed by the listed files or
symlinks themselves, while `resolved_bytes` follows symlinks to the target
files. A symlink subset can have tiny `apparent_bytes` but large
`resolved_bytes`, meaning the original source `server_vis/` is still required.
If `broken_symlink_count` is non-zero, the subset is no longer trainable until
the missing source artifacts are restored or the subset is rebuilt.

## Current Practical Guidance

Until KV-cache storage is further reduced, replay-context collection should be
small and monitored:

- prefer single-task actor-replay smoke runs;
- prefer `ACTION_NUM_INFERENCE_STEPS=10` for pipeline validation;
- avoid multi-task replay-context collection;
- avoid 50-step replay-context collection unless enough Scratch headroom is
  available and the run is explicitly justified;
- kill jobs early if `server_vis` grows faster than expected or `df -h` shows
  less than the required safety margin.

RDSS helps preserve completed results and free Scratch, but it does not solve
the in-job memory or high-frequency `torch.save` cost of replay-context capture.
The scalable long-term solution is artifact slimming and selective replay.
