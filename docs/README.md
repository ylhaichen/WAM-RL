# WAM-RL Documentation Index

Use this file to choose the right document before running experiments or making
claims. The repository contains both current operating docs and historical
planning/paper docs.

## Current Source Of Truth

- `WAM_RL_CURRENT_PROJECT_STATUS.md`: current stage, implemented pieces,
  validated datasets, blockers, and claims boundaries.
- `WAM_RL_ACTOR_REPLAY_RUNBOOK.md`: step-by-step Myriad workflow for real
  LingBot-VA actor replay smoke runs.
- `WAM_RL_MYRIAD_STORAGE_POLICY.md`: Scratch/RDSS usage, cleanup guardrails,
  and storage-audit commands.
- `GRPO_STRICT_ARTIFACT_SCHEMA.md`: strict artifact and replay-context schema.

Read these before submitting jobs, deleting data, interpreting metrics, or
claiming project progress.

## Setup And Environment

- `PORTABLE_ENV_SETUP.md`: portable environment setup notes.
- `SERVER_SETUP.md`: server-side setup notes.

## Historical Context

- `WAM_RL_RESEARCH_IMPLEMENTATION_PLAN_FRAMEWORK_FINAL.md`: original
  implementation plan. Useful for intent, not the current execution state.
- `WAM_RL_EARLY_STAGE_WORK_SUMMARY.md`: historical early-stage paper snapshot.
  Some limitations described there have since been implemented.
- `RL_FRAMEWORK_SYSTEM_REVIEW.md`: implementation review updated with the
  current actor replay boundary, but still framed as a system review.
- `WAM_RL_PAPER_DATA_PLAN.md`: paper-data planning. Do not use it to claim
  results before the current status document says the evidence exists.

## Operational Priority

1. Check `AGENT.md` and `WAM_RL_CURRENT_PROJECT_STATUS.md`.
2. For actor replay work, follow `WAM_RL_ACTOR_REPLAY_RUNBOOK.md`.
3. Before any replay-context collection or cleanup, follow
   `WAM_RL_MYRIAD_STORAGE_POLICY.md`.
4. Treat historical docs as context only when they conflict with current status.
