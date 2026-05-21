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
  `groups/grpo_groups.jsonl` has zero lines;
- old debug/sanity rollout directories that are not used by the current
  combined dataset.

Do not delete without explicit review:

- A/B/M/N grouped rollout datasets used to build the current combined GRPO
  dataset;
- any `server_vis/` directory referenced by a non-empty current
  `grpo_groups.jsonl`;
- `results_grpo_datasets/a_b_m_n_*` and other curated combined datasets;
- actor replay outputs that contain a real checkpoint to evaluate.

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

List largest files without breaking paths containing spaces:

```bash
python - <<'PY'
from pathlib import Path

root = Path("/home/zcably0/Scratch/wam-rl/results_grouped_rollouts")
for path in sorted(root.rglob("*.pt"), key=lambda p: p.stat().st_size, reverse=True)[:30]:
    print(f"{path.stat().st_size / 1024**3:.2f} GB  {path}")
PY
```

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
