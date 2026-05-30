# WAM-RL Agent Operating Rules

This repository is a research codebase for applying SimpleVLA-style GRPO to
LingBot-VA / WAM-RL on RoboTwin. Agents working here must follow the project
plan and the current implementation state instead of inventing a new direction.

## Project Route

Use the planning documents as the source of truth:

- `docs/WAM_RL_RESEARCH_IMPLEMENTATION_PLAN_FRAMEWORK_FINAL.md`
- `docs/WAM_RL_CURRENT_PROJECT_STATUS.md`
- `docs/WAM_RL_EARLY_STAGE_WORK_SUMMARY.md`
- `docs/GRPO_STRICT_ARTIFACT_SCHEMA.md`
- `docs/WAM_RL_MYRIAD_STORAGE_POLICY.md`
- `docs/WAM_RL_ACTOR_REPLAY_RUNBOOK.md`
- `docs/WAM_RL_PAPER_DATA_PLAN.md`

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
  with empty `grpo_groups.jsonl` are cleanup candidates after preserving small
  debug evidence.
- Do not delete A/B/M/N source datasets, curated combined datasets, or any
  `server_vis/` referenced by a non-empty current `grpo_groups.jsonl` without
  explicit user approval.
- Check `qstat` before deleting data so no running job is writing to the target
  directory.
- For actor replay debugging, prefer a lightweight workflow:
  `tools/subset_grpo_groups.py` to select a small mixed subset, then
  `tools/materialize_grpo_artifacts.py --link-mode symlink
  --include-replay-context` to create a rewritten groups file plus symlinked
  strict artifacts/replay contexts. Do not delete the source `server_vis/` while
  such a symlink materialized subset is active.
- For storage-limited replay-context collection, set
  `STRICT_GRPO_CAPTURE_CHUNK_STRIDE` or `STRICT_GRPO_CAPTURE_MAX_CHUNKS` at
  collection time to save fewer action chunks. Defaults are stride `1` and max
  chunks `0`, preserving full capture.
- For one-step actor replay trainer smoke on a materialized subset, prefer
  `jobs/myriad/36_submit_actor_replay_subset_smoke.sh`. It wraps the real
  trainer job with lower default queue resources and keeps the training logic in
  `jobs/myriad/34_train_actor_replay_grpo_robotwin.sh`.
- Use `tools/audit_grpo_artifact_storage.py` before cleanup decisions when a
  `grpo_groups.jsonl` or materialized subset might still reference large
  strict artifacts or replay-context symlink targets.

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
- Compare baseline and actor evals with
  `tools/compare_robotwin_eval_episodes.py` on matched episode keys before
  interpreting aggregate success-rate differences.
- Long actor replay runs should set `GRPO_PROGRESS_EVERY` so the job log is not
  silent while replaying hundreds of denoising transitions.
- After actor replay training finishes, use
  `tools/summarize_actor_replay_training.py` to inspect validation status,
  final metrics, checkpoint presence, and failure diagnostics before planning
  evaluation.
- For baseline-vs-actor eval smoke, prefer
  `jobs/myriad/37_submit_actor_eval_pair_smoke.sh` so both runs share task,
  prompt, env seed, sampling seed, and action-step settings while using
  separate ports.
- Keep `SAVE_SERVER_DEBUG_TENSORS=false` for routine eval jobs so the server
  does not write per-chunk latent/action/observation tensors unless a diagnosis
  needs them.
- After paired eval finishes, use `tools/summarize_actor_eval_pair.py` to
  write aggregate summaries, per-episode exports, and matched comparison files
  before interpreting any success-rate difference.
- The first real update surface should be action-specific modules:

```text
action_embedder
condition_embedder_action
action_proj_out
```
