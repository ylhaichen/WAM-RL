# WAM-RL Agent Operating Rules

This repository is a research codebase for applying SimpleVLA-style GRPO to
LingBot-VA / WAM-RL on RoboTwin. Agents working here must follow the project
plan and the current implementation state instead of inventing a new direction.

## Project Route

Use the current status document as the source of truth. Treat older planning and
paper documents as historical context unless they explicitly say they are
current.

- current status and claims boundary:
  `docs/WAM_RL_CURRENT_PROJECT_STATUS.md`
- active Myriad actor replay operations:
  `docs/WAM_RL_ACTOR_REPLAY_RUNBOOK.md`
- storage, RDSS, and cleanup policy:
  `docs/WAM_RL_MYRIAD_STORAGE_POLICY.md`
- strict GRPO artifact schema:
  `docs/GRPO_STRICT_ARTIFACT_SCHEMA.md`
- historical implementation plan:
  `docs/WAM_RL_RESEARCH_IMPLEMENTATION_PLAN_FRAMEWORK_FINAL.md`
- historical early-stage paper snapshot:
  `docs/WAM_RL_EARLY_STAGE_WORK_SUMMARY.md`
- paper-data planning:
  `docs/WAM_RL_PAPER_DATA_PLAN.md`

The current route is:

```text
RoboTwin baseline and task selection
-> stochastic action-denoising instrumentation
-> grouped rollout collection
-> strict GRPO artifact capture and validation
-> offline GRPO smoke training
-> real LingBot-VA actor replay trainer
-> iterative online GRPO
-> optional video-action consistency
-> optional veRL scale-up
```

Do not skip directly to paper claims, veRL migration, reward shaping, or
video-action consistency before the real actor replay trainer is working.

For current stage, completion, blockers, and claims boundaries, read
`docs/WAM_RL_CURRENT_PROJECT_STATUS.md` first. The older early-stage summary is
historical and may describe limitations that have since been implemented.

## Truthfulness

- Do not fabricate results, benchmark numbers, success rates, or claimed
  improvements.
- If a command was not run, say it was not run.
- If a test cannot be run locally because `torch`, `pytest`, RoboTwin, Myriad, or
  containers are unavailable, say so explicitly.
- Do not describe smoke training as real policy improvement.
- Do not claim GRPO policy learning until a real LingBot-VA actor checkpoint is
  trained and evaluated.

## No Hardcoded Success

Do not implement code that merely appears to work by hardcoding expected output,
special-casing a known test fixture, bypassing the model, or replacing a real
calculation with a constant.

Examples of unacceptable shortcuts:

- returning stored `transition_mean` as the current actor output;
- treating `StrictArtifactScalarPolicy` smoke training as actor training;
- accepting invalid artifact schemas to make validation pass;
- silently ignoring missing replay context;
- generating fake metrics or fake checkpoints.

Fail fast with a clear error when required data or model context is missing.

## Engineering Discipline

- Do not treat a script that only runs the happy path as a complete deliverable.
- Every new tool should have unit tests, negative tests, or a reproducible
  command with raw logs.
- Every validator should be tested on valid and intentionally corrupted inputs
  when feasible.
- Every cache-related change must prove the relevant contract: before/after
  equality when behavior should be identical, or explicit fail-fast behavior
  when the cache state is invalid or over budget.
- Every stochastic component should accept an explicit seed or document why it
  is deterministic.
- Every research/engineering report should include commit hash, command lines,
  raw log paths, artifact paths, and non-claims.
- Do not hide failures with broad `try/except`, silent fallbacks, or `|| true`
  unless the fallback is explicitly documented and tested.
- Do not claim progress from documentation-only changes unless the task is
  explicitly documentation.

## Debugging Standard

Debug from the root cause. Avoid superficial patches that only hide a symptom.

Expected workflow:

1. Reproduce or inspect the failure.
2. Identify the real failing contract or state assumption.
3. Fix the contract or implementation at the source.
4. Add or update tests around the failure mode.
5. Run the most relevant local checks.
6. State remaining uncertainty and server-side checks still needed.

## Decision-Making

When requirements are ambiguous, important context is missing, or multiple
valid implementation paths exist, ask the user before making a major decision.

In particular, ask before:

- changing the research objective;
- changing task selection policy;
- changing training/evaluation claims;
- deleting data or logs;
- broad refactors across unrelated modules;
- replacing native WAM-RL implementation with another framework.

## Current Workflow

The project workflow is:

```text
local machine:
  code edits, lightweight static checks, docs, git commits

Myriad/server:
  containers, RoboTwin rollout collection, GPU tests, training smoke, real actor training
```

Local checks are useful but incomplete. Anything involving `torch` GPU,
RoboTwin, SAPIEN, Apptainer, or large checkpoint loading must be verified on the
server.

## Myriad Remote Operation Rules

Canonical remote target:

```text
ssh alias/host: myriad / myriad.rc.ucl.ac.uk
user: zcably0
project root: /home/zcably0/Scratch/WAM-RL
data root: /home/zcably0/Scratch/wam-rl
```

Allowed without extra confirmation when the user has asked for Myriad
development:

- inspect files, logs, queue state, disk usage, and git state;
- run lightweight read-only commands such as `pwd`, `ls`, `find`, `rg`, `sed`,
  `tail`, `qstat`, and `git status`;
- summarize GRPO job/result state with `tools/report_grpo_run_status.py`
  before hand-writing long `grep` pipelines;
- for queued jobs without logs, use
  `tools/report_grpo_run_status.py --qstat-job-id <job-id>` to inspect
  scheduler metadata and exported job variables; qstat reporting handles
  wrapped SGE `env_list` output and can infer `RESULTS_ROOT`,
  `GRPO_GROUPS_PATH`, and `GRPO_OUTPUT_DIR` when those variables were exported
  at submission time;
- avoid unbounded `qacct` calls during interactive debugging. Prefer `qstat`,
  job logs, and the status reporter; if accounting is needed, query a specific
  finished job with a shell timeout;
- run containerized unit/tool tests that do not submit scheduler jobs;
- edit docs/code, commit, and push when the current task clearly requires it.

Require explicit user review before:

- submitting `qsub` jobs or starting long `qrsh` sessions;
- running `qdel`;
- deleting, moving, archiving, or compressing rollout data, checkpoints,
  artifacts, logs, model paths, dataset paths, or `/home/zcably0/Scratch/WAM-RL`;
- changing branch strategy or pushing to a non-obvious remote/branch.

Before any `qsub`, state the exact command/script, resource request, expected
runtime class, output/log paths, and git commit being used. If the user has
already granted a bounded development session, keep them updated at important
milestones and still avoid destructive file operations without explicit review.

## Myriad Storage Rules

Treat storage as part of the experiment design. Replay-context artifacts can
consume tens or hundreds of GB because they include transformer KV-cache state.

- Active rollout collection and training must write to Myriad Scratch, not
  directly to RDSS.
- RDSS is a cold archive for completed results, logs, summaries, metrics, and
  non-active datasets.
- Do not assume moving only `groups/grpo_groups.jsonl` is enough for training:
  the file contains artifact paths, so referenced strict artifacts and replay
  context files must remain reachable.
- Prefer archiving `groups/`, `attempts/`, job logs, manifests, summaries, and
  validation JSON files before deleting large unusable `server_vis/` trees.
- Failed replay-context runs and all-success/all-failure replay-context runs
  with empty `grpo_groups*.jsonl` are cleanup candidates after preserving small
  debug evidence.
- Do not delete A/B/M/N source datasets, curated combined datasets, or any
  `server_vis/` referenced by a non-empty current `grpo_groups*.jsonl` without
  explicit user approval.
- Check `qstat` before deleting data so no running job is writing to the target
  directory.
- For actor replay debugging, prefer a lightweight workflow:
  `tools/subset_grpo_groups.py` to select a small mixed subset, then
  `tools/materialize_grpo_artifacts.py --link-mode symlink
  --include-replay-context` to create a rewritten groups file plus symlinked
  strict artifacts/replay contexts. Do not delete the source `server_vis/` while
  such a symlink materialized subset is active.
- For queued actor replay subset preparation, prefer
  `jobs/myriad/35_submit_prepare_actor_replay_subset.sh` and review its
  `--dry-run` output. It submits the existing preparation job with explicit
  `qsub -v` variables rather than inheriting the whole interactive shell.
- When building actor replay subsets, set `--max-replay-context-gb` (or
  `SUBSET_MAX_REPLAY_CONTEXT_GB` in the Myriad subset job) so selected artifact
  refs are bounded by actual resolved replay-context footprint, not only by raw
  artifact count. Keep `SUBSET_STORAGE_MAX_RESOLVED_GB` enabled in the subset
  job so materialized strict-artifact plus replay-context dependencies are
  audited before training. The subset job writes `materialize_plan.json` before
  creating symlinks or copies; use it to inspect `planned_copy_gb` and resolved
  dependency footprint before approving copy-mode self-contained subsets.
- When combining multiple bounded `grpo_groups.jsonl` sources, use
  `tools/merge_grpo_groups.py` instead of manual `cat`; it writes a manifest and
  rejects duplicate `group_id` values by default.
- For storage-limited replay-context collection, set
  `STRICT_GRPO_CAPTURE_CHUNK_STRIDE` or `STRICT_GRPO_CAPTURE_MAX_CHUNKS` at
  collection time to save fewer action chunks. Defaults are stride `1` and max
  chunks `0`, preserving full capture.
- For new actor-replay collection smoke runs, prefer
  `jobs/myriad/39_submit_grpo_replayctx_bounded_4gpu.sh` and review its
  `DRY_RUN=1` output before submission. It defaults to one captured chunk per
  rollout and disables server debug tensors. New action-scale-one collections
  use conditional-branch-only replay-context k/v storage, so the wrapper's
  default estimate is 4GB/context. Override `REPLAY_CONTEXT_ESTIMATE_GB` upward
  for action-guided (`action_guidance_scale > 1`) or legacy unpruned data.
  Keep `STORAGE_BUDGET_MODE=attempt` unless the user explicitly accepts the
  risk of failed attempts leaving large replay-context files behind. Set
  `PLAN_JSON=/home/zcably0/Scratch/wam-rl/debug_logs/storage_audits/<run>.json`
  when you want the dry-run storage plan persisted for later audit.
- Keep `STRICT_GRPO_REPLAY_CONTEXT_MAX_GB` enabled for bounded replay-context
  collection. It is a per-context server-side guard checked before `torch.save`;
  if the live KV-cache tensor footprint exceeds the reviewed budget, the
  attempt should fail early instead of filling Scratch.
- For one-step actor replay trainer smoke on a materialized subset, prefer
  `jobs/myriad/36_submit_actor_replay_subset_smoke.sh`. It wraps the real
  trainer job with lower default queue resources and keeps the training logic in
  `jobs/myriad/34_train_actor_replay_grpo_robotwin.sh`. The wrapper uses an
  explicit `qsub -v` variable whitelist by default; keep
  `QSUB_EXPORT_CURRENT_ENV=0` unless deliberately debugging inherited shell
  environment behavior.
- Use `tools/audit_grpo_artifact_storage.py` before cleanup decisions when a
  `grpo_groups.jsonl` or materialized subset might still reference large
  strict artifacts or replay-context symlink targets. Add
  `--inspect-replay-contexts` when auditing real actor replay data, because
  replay-context files are usually referenced from inside each strict artifact
  rather than directly from the JSONL. Use `--print-summary` for logs and
  `--omit-replay-context-mapping` when the full per-artifact mapping would make
  reports too large.
- Use `tools/inspect_grpo_replay_context.py` on one representative
  `strict_grpo_replay_context_*.pt` before designing storage slimming changes;
  it reports tensor bytes by top-level key and largest nested tensors.
- Use `tools/summarize_grpo_replay_contexts.py` on `grpo_groups.jsonl` for
  low-IO aggregate replay-context file footprint. It only stats context files
  by default; add `--inspect-context-tensors` only for small or explicitly
  bounded sources because even metadata-only `torch.load` can create heavy
  filesystem IO on multi-GB context files.
- Use `tools/plan_myriad_storage_cleanup.py` to produce non-destructive cleanup
  candidate reports before proposing deletion. It checks all
  `grpo_groups*.jsonl` files so partial or accepted group files still protect
  their source artifacts.

For concrete commands and RDSS archival patterns, use
`docs/WAM_RL_MYRIAD_STORAGE_POLICY.md`.

## New Agent Onboarding

Before changing code, a new agent must read enough context to understand the
current system:

1. Read this file.
2. Read the core planning docs listed above.
3. Inspect `wan_va/wan_va_server.py` for action denoising and artifact capture.
4. Inspect `wan_va/rl/dataset.py`, `wan_va/rl/denoising_replay.py`,
   `wan_va/rl/grpo_loss.py`, and `wan_va/rl/trainer.py`.
5. Inspect `tools/collect_robotwin_rollouts.py`,
   `tools/build_grpo_groups.py`, and `tools/validate_grpo_dataset.py`.
6. Inspect `jobs/myriad/30_collect_grouped_rollouts_4gpu.sh` and
   `jobs/myriad/31_train_denoising_grpo_robotwin.sh`.
7. Check `git status` and avoid touching unrelated local files.

## Current Technical Truth

- v1 strict artifacts capture one first action denoising transition.
- v2 strict artifacts capture full action denoising trajectories per action
  chunk.
- Existing offline GRPO training is a smoke adapter unless explicitly using the
  real actor replay trainer.
- Real actor replay requires saved replay context, including transformer
  conditioning and KV cache state. Old artifacts without replay context are not
  sufficient for real actor updates.
- When global video CFG is enabled but `action_guidance_scale <= 1`, new replay
  contexts store only the conditional action k/v cache branch and clone only
  that branch during cache snapshot. This preserves the action replay mean
  because the negative action branch is not used, and it avoids roughly half of
  the replay-context k/v storage for common smoke collections.
- For real actor replay, use `GRPO_ACTION_NUM_INFERENCE_STEPS` matching the
  collection run, `GRPO_LOGPROB_REDUCTION=mean`, and a conservative
  `GRPO_LOGPROB_STD_FLOOR` such as `0.1`. The collected diffusion transition
  std can be as small as `0.01`, so tiny replay/numerical mean differences can
  otherwise saturate the clipped GRPO ratio before any useful update.
- For one-GPU baseline vs actor eval comparisons, set `PROMPT_INDEX` explicitly
  so both policies see the same deterministic instruction variant for each
  environment seed. Otherwise prompt sampling can change across runs and make
  seed-level comparisons noisy.
- Also set `SAMPLING_SEED` for eval comparisons. Use
  `SAMPLING_SEED_PER_ENV=true` when evaluating multiple env seeds so each
  episode gets a deterministic but distinct server sampling seed
  (`SAMPLING_SEED + env_seed`). This controls LingBot-VA action sampling but is
  not a full simulator determinism guarantee; repeated RoboTwin closed-loop
  runs can still diverge after the first action chunk, so interpret small
  `n=5` differences as smoke signals rather than policy improvement claims.
- Set `SEED` explicitly for eval jobs. RoboTwin starts from
  `st_seed = 10000 * (1 + SEED)`, so leaving `SEED` to `qsub -V` inheritance can
  silently change the matched episode set.
- Eval episode JSON should carry `run_id`. The eval client now falls back to
  the `save_root` basename when no explicit `RUN_ID`/`--run_id` is provided,
  but paired eval jobs should still pass `RUN_ID` explicitly through the
  submitter.
- Compare baseline and actor evals with
  `tools/compare_robotwin_eval_episodes.py` on matched episode keys before
  interpreting aggregate success-rate differences.
- Long actor replay runs should set `GRPO_PROGRESS_EVERY` so the job log is not
  silent while replaying hundreds of denoising transitions.
- After actor replay training finishes, use
  `tools/summarize_actor_replay_training.py` to inspect validation status,
  final metrics, checkpoint presence, provenance (`model_path`, `config_name`,
  `git_commit`), and failure diagnostics before planning evaluation.
- Use `tools/inspect_actor_replay_checkpoint.py` to inspect actor replay
  checkpoint tensor stats or compare a candidate checkpoint against an `lr=0`
  no-op/reference checkpoint before interpreting tiny closed-loop eval deltas.
- For baseline-vs-actor eval smoke, prefer
  `jobs/myriad/37_submit_actor_eval_pair_smoke.sh` so both runs share task,
  prompt, env seed, sampling seed, and action-step settings while using
  separate ports.
- For baseline repeatability controls, prefer
  `jobs/myriad/38_submit_eval_repeatability_pair.sh` so repeated baseline runs
  use the same eval controls and produce a ready-to-run repeatability summary
  command.
- Keep `SAVE_SERVER_DEBUG_TENSORS=false` for routine eval jobs so the server
  does not write per-chunk latent/action/observation tensors unless a diagnosis
  needs them.
- After paired eval finishes, use `tools/summarize_actor_eval_pair.py` to
  write aggregate summaries, per-episode exports, and matched comparison files
  before interpreting any success-rate difference. Check its
  `provenance_warnings`; missing `run_id`, checkpoint, action-step, prompt, or
  sampling provenance means the run is still useful for debugging but should
  not be treated as a complete experiment record.
- Before promoting a candidate actor checkpoint beyond smoke status, run
  `tools/gate_actor_eval_promotion.py` with a paired comparison JSON and a
  baseline repeatability JSON. The gate is intentionally conservative: tiny
  evals or deltas no larger than baseline closed-loop flip noise stay blocked.
- The first real update surface should be action-specific modules:

```text
action_embedder
condition_embedder_action
action_proj_out
```
