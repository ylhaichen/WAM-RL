# WAM-RL Agent Operating Rules

This repository is a research codebase for applying SimpleVLA-style GRPO to
LingBot-VA / WAM-RL on RoboTwin. Agents working here must follow the project
plan and the current implementation state instead of inventing a new direction.

## Project Route

Use the planning documents as the source of truth:

- `docs/WAM_RL_RESEARCH_IMPLEMENTATION_PLAN_FRAMEWORK_FINAL.md`
- `docs/WAM_RL_EARLY_STAGE_WORK_SUMMARY.md`
- `docs/GRPO_STRICT_ARTIFACT_SCHEMA.md`
- `docs/WAM_RL_MYRIAD_STORAGE_POLICY.md`
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
- The first real update surface should be action-specific modules:

```text
action_embedder
condition_embedder_action
action_proj_out
```
