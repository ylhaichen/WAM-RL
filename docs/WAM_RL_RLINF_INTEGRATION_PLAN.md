# WAM-RL RLinf Integration Plan

**Date:** 2026-05-30
**Purpose:** plan the transition from the current native WAM-RL actor-replay
pipeline to an RLinf-first training workflow.
**Scope:** LingBot-VA / RoboTwin / GRPO engineering migration, not paper claims.

This document is a migration plan. The current source of truth for project
status remains `docs/WAM_RL_CURRENT_PROJECT_STATUS.md`; the current Myriad
operational workflow remains `docs/WAM_RL_ACTOR_REPLAY_RUNBOOK.md`; storage
rules remain `docs/WAM_RL_MYRIAD_STORAGE_POLICY.md`.

## Executive Summary

RLinf is a plausible mainline infrastructure target for WAM-RL because current
RLinf documentation explicitly covers `LingBot-VLA`, `RoboTwin 2.0`, and `GRPO`.
It also offers native LingBot-VLA integration inside RLinf's Python memory space,
instead of relying only on an external WebSocket-style model server. That is
directly relevant to WAM-RL's current bottleneck: actor replay currently depends
on very large replay-context KV-cache files.

The migration should not start by replacing the current working code path. The
lowest-risk route is:

```text
native WAM-RL remains the reference implementation
-> RLinf compatibility audit and config mapping
-> RLinf environment smoke
-> eval parity against native WAM-RL
-> small GRPO smoke in RLinf
-> RLinf becomes the mainline scale-up path
```

On current Myriad, the realistic goal is integration readiness and smoke-scale
validation. Large-scale RLinf training should wait for larger active storage and
GPU resources.

## External RLinf Facts To Verify

The following facts were checked against public RLinf docs on 2026-05-30:

- RLinf describes itself as reinforcement-learning infrastructure for embodied
  and agentic AI, with support for workflows including `PPO`, `GRPO`, and `SAC`,
  and backend integrations including `FSDP + HuggingFace/SGLang/vLLM` and
  `Megatron + SGLang/vLLM`.
  Source: https://github.com/RLinf/RLinf
- RLinf has a `LingBot-VLA` + `RoboTwin 2.0` example. The docs describe native
  plugin integration and note that it embeds LingBot-VLA into RLinf's Python
  memory space, unlike traditional WebSocket communication.
  Source: https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/lingbotvla.html
- RLinf LingBot-VLA docs list `GRPO` and `Flow-SDE` action denoising /
  generation, with configurable `noise_method`, `noise_level`, and `num_steps`.
  Source: https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/lingbotvla.html
- RLinf RoboTwin docs describe VLA fine-tuning in RoboTwin with `PPO` and
  `GRPO`, provide example YAML configs, and use YAML `runner.only_eval` for
  evaluation mode.
  Source: https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html
- RLinf quickstart documents Docker and custom-environment install paths and
  includes embodied VLA training and multi-node training entry points.
  Source: https://rlinf.readthedocs.io/en/latest/rst_source/start/index.html

These facts are not a guarantee that RLinf's default LingBot-VLA GRPO objective
matches WAM-RL's current denoising-transition GRPO contract. That equivalence
must be tested explicitly.

## Current WAM-RL Baseline To Preserve

Current WAM-RL has already validated:

- grouped RoboTwin rollout collection;
- strict GRPO artifact validation;
- offline strict GRPO smoke training;
- real LingBot-VA actor replay trainer on a tiny replay-context subset;
- actor replay checkpoint loading for eval;
- paired baseline-vs-actor eval smoke and baseline repeatability tooling;
- Myriad storage audit, cleanup planning, and provenance tracking.

The latest known good commit before this plan was:

```text
863fa18 Propagate actor replay job provenance
```

The latest full Myriad container test result recorded in handoff context was:

```text
225 passed, 1 skipped
```

Do not discard this native path until RLinf can reproduce the same basic
contracts.

## Migration Goals

### Primary Goals

1. Make RLinf the future large-scale training path for WAM-RL.
2. Preserve native WAM-RL as a reference path and regression oracle.
3. Reduce dependence on huge persisted replay-context KV-cache artifacts by
   moving toward an online RLinf rollout/actor/learner flow.
4. Keep WAM-RL's current validation discipline: strict data contracts,
   provenance, storage accounting, paired eval comparison, and promotion gates.

### Non-Goals For The First Agent

- Do not run large-scale RLinf training on current Myriad.
- Do not delete, move, archive, or compress Myriad datasets.
- Do not replace the native actor replay trainer before RLinf smoke parity is
  established.
- Do not claim policy improvement from RLinf smoke runs.
- Do not submit `qsub` jobs unless explicitly approved by the user.
- Do not touch local untracked `WAM-RL_RSS.zip` or `WAM-RL_RSS/`.

## Key Compatibility Questions

The first RLinf agent should answer these before writing large amounts of code:

1. Environment compatibility:
   - Can RLinf's LingBot-VLA + RoboTwin environment be installed or containerized
     on the target system?
   - On Myriad, is Docker unavailable and Apptainer required?
   - Does RLinf's official Docker image need conversion to `.sif`?

2. Model compatibility:
   - Does RLinf's `LingBot-VLA` plugin load the same base checkpoint used by
     WAM-RL?
   - Are tokenizer/backbone paths equivalent to WAM-RL's current
     `lingbot-va-posttrain-robotwin` setup?
   - Can actor-replay checkpoints from WAM-RL be inspected or loaded by the
     RLinf path, or are they only useful as native-reference outputs?

3. Task/eval compatibility:
   - Do RLinf task names and RoboTwin task configs match WAM-RL task names such
     as `move_stapler_pad`, `turn_switch`, and `open_microwave`?
   - Can RLinf expose equivalent controls for `PROMPT_INDEX`, `SEED`,
     `SAMPLING_SEED`, `SAMPLING_SEED_PER_ENV`, and
     `ACTION_NUM_INFERENCE_STEPS`?
   - Can RLinf evaluation output enough per-episode metadata for
     `tools/compare_robotwin_eval_episodes.py` or a successor comparator?

4. GRPO objective compatibility:
   - Does RLinf's LingBot-VLA GRPO compute policy ratios over action denoising
     transitions, action chunks, complete episodes, or another unit?
   - Does it support group-relative rewards with mixed success/failure groups
     in the same way WAM-RL expects?
   - Does it expose policy clipping, KL, logprob reduction, and denoising
     step-count settings sufficiently to map WAM-RL's current trainer config?

5. Storage compatibility:
   - Does RLinf avoid writing full replay-context KV-cache payloads to disk?
   - If it writes rollout buffers, what is the per-episode and per-group
     footprint?
   - Can storage be audited before long jobs, as WAM-RL does with
     `tools/plan_replay_context_collection.py` and
     `tools/audit_grpo_artifact_storage.py`?

## Proposed Repository Structure

Keep the first integration small and additive:

```text
docs/WAM_RL_RLINF_INTEGRATION_PLAN.md
configs/rlinf/
  README.md
  robotwin_move_stapler_pad_grpo_lingbotvla_wamrl.yaml
tools/
  inspect_rlinf_migration_readiness.py
  export_rlinf_eval_manifest.py
```

Only add Python code after the config and compatibility questions are clear.
Prefer read-only inspectors before converters that rewrite data.

## Phased Plan

### Phase 0: Context And Safety

Inputs:

- `AGENT.md`
- `docs/WAM_RL_CURRENT_PROJECT_STATUS.md`
- `docs/WAM_RL_ACTOR_REPLAY_RUNBOOK.md`
- `docs/WAM_RL_MYRIAD_STORAGE_POLICY.md`
- this document

Actions:

1. Check local `git status --short --branch`.
2. Confirm current `HEAD`.
3. If using Myriad, check remote `git log -1 --oneline` and `qstat`.
4. Confirm no `qsub` will be submitted without user approval.

Exit criteria:

- The agent can explain the current native WAM-RL actor replay loop and why
  current Myriad storage blocks scale.

### Phase 1: RLinf Environment Feasibility

Actions:

1. Inspect RLinf official install paths and identify whether Myriad needs
   Docker, Apptainer conversion, or a custom Python environment.
2. Record required RLinf version/commit and dependency stack.
3. Produce a setup note before installing anything large.

Myriad constraint:

- Do not download large Docker images, HuggingFace models, or RoboTwin assets
  without explicit user approval.

Exit criteria:

- A concrete environment plan exists for either Myriad smoke or the future
  larger server.

### Phase 2: Config Mapping

Actions:

1. Create a minimal RLinf YAML skeleton for:

```text
task: move_stapler_pad
algorithm: GRPO
model: LingBot-VLA
action_num_inference_steps: 10
prompt_index: 0
group_size: 4 or 8
eval seeds: explicit
```

2. Map WAM-RL settings to RLinf fields. Mark every uncertain mapping with
   `TODO(verify_rlinf_field)`, not a guessed value.
3. Document unsupported or unknown fields in `configs/rlinf/README.md`.

Exit criteria:

- The config is honest: runnable only where verified, and uncertain fields are
  not hidden.

### Phase 3: Eval Parity Smoke

Goal:

Run evaluation only, not training, and compare RLinf output against native
WAM-RL expectations.

Actions:

1. Use `move_stapler_pad`, `PROMPT_INDEX=0`,
   `ACTION_NUM_INFERENCE_STEPS=10`, and explicit seeds.
2. Ensure output records:
   - task;
   - env seed;
   - prompt index;
   - sampling seed or RLinf equivalent;
   - policy checkpoint;
   - reference checkpoint if applicable;
   - action step count;
   - success/failure;
   - action count.
3. If RLinf output format differs, write an adapter manifest rather than
   modifying native WAM-RL eval tools immediately.

Exit criteria:

- RLinf eval output can be summarized and compared at the episode level, or the
  missing metadata is explicitly listed.

### Phase 4: RLinf GRPO Smoke

Goal:

Run the smallest RLinf GRPO smoke that proves the training loop can update the
LingBot-VLA actor path.

Recommended first task:

```text
move_stapler_pad
group_size: 4 before 8 on current storage
groups_per_task: 1
max accepted or attempted groups: tiny
action_num_inference_steps: 10
```

Exit criteria:

- RLinf writes a checkpoint.
- Training metrics are finite.
- The checkpoint shows nonzero parameter movement in the intended action path.
- Eval smoke can load the checkpoint.
- No policy-improvement claim is made.

### Phase 5: Mainline Transition

Once RLinf smoke parity passes:

1. Treat RLinf as the preferred scale-up path.
2. Keep native WAM-RL as:
   - artifact schema reference;
   - validation reference;
   - eval comparison reference;
   - fallback trainer for small actor-replay debugging.
3. Update `docs/WAM_RL_CURRENT_PROJECT_STATUS.md` to say RLinf migration has
   become active, but only after a real smoke result exists.

## Verification Checklist

Before marking RLinf integration as ready:

- [ ] RLinf version/commit recorded.
- [ ] Environment path documented.
- [ ] Base checkpoint path recorded.
- [ ] RoboTwin assets path recorded.
- [ ] Task config mapped for at least `move_stapler_pad`.
- [ ] Eval-only run works or a concrete blocker is recorded.
- [ ] GRPO smoke run works or a concrete blocker is recorded.
- [ ] Output provenance includes git commit, model path, config path, task,
      seed controls, and checkpoint path.
- [ ] No large Myriad data was moved/deleted.
- [ ] No large `qsub` job was submitted without user approval.

## Risks And Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| RLinf GRPO objective does not match WAM-RL denoising-transition GRPO | Results not comparable | Start with objective audit and eval parity before training |
| RLinf environment conflicts with existing WAM-RL container | Wasted setup time | Keep RLinf env isolated; prefer separate container/env |
| Myriad Docker limitations | RLinf official image may not run directly | Plan Apptainer conversion or defer heavy setup to new server |
| Replay-context storage remains large | Scale still blocked | Prefer online RLinf flow that avoids full KV-cache persistence |
| Eval nondeterminism persists | Small deltas remain non-actionable | Keep paired eval, repeatability controls, and promotion gate |
| New agent rewrites native path too early | Loss of working baseline | Add only `configs/rlinf/` and read-only tooling first |

## Resource Implications

Current Myriad is enough for:

- static integration work;
- config mapping;
- environment planning;
- maybe small eval smoke;
- maybe tiny GRPO smoke after explicit approval.

Current Myriad is not enough for:

- multi-task RLinf GRPO;
- large multi-group collection;
- long actor replay or full denoising trajectory capture;
- benchmark-scale eval sweeps.

For the future server request, RLinf mainline training should still target:

```text
minimum comfortable:
  4 x A100/H100 80GB
  10TB active NVMe/Scratch
  50TB archive
  512GB CPU RAM

project-grade:
  8 x A100/H100 80GB or 4-8 x H200 141GB
  20-30TB active NVMe/Scratch
  100TB archive
  1TB+ CPU RAM
```

## First Agent Task List

Recommended first assignment for the RLinf integration agent:

1. Read the required WAM-RL docs and this plan.
2. Inspect RLinf's `robotwin_click_bell_grpo_lingbotvla.yaml` and related
   LingBot-VLA code paths.
3. Produce a field-by-field mapping from WAM-RL settings to RLinf config keys.
4. Add a minimal `configs/rlinf/README.md` and YAML skeleton with explicit
   `TODO(verify_rlinf_field)` markers.
5. Do not run large jobs. Ask before installing RLinf or downloading large
   assets.

