# WAM-RL Early-Stage Work Summary

**Date:** 2026-05-18
**Purpose:** early-stage paper planning and project-state record
**Project:** SimpleVLA-style GRPO for LingBot-VA / WAM-RL on RoboTwin
**Primary model target:** LingBot-VA / WAM-RL Video-Action VLA
**Primary environment:** RoboTwin / RoboTwin2.0
**Primary implementation stance:** native LingBot-VA/WAM-RL rollout and trainer first; veRL scale-up later

This document summarizes the real current state of the WAM-RL project for writing an early-stage work paper. It separates implemented and validated components from planned work, known limitations, and claims that should not yet be made.

---

## 1. Executive Summary

The current project direction is to adapt SimpleVLA-RL-style grouped outcome RL to LingBot-VA, whose action generation is FlowMatch-based and continuous rather than discrete token-based. This creates a central policy-interface mismatch: SimpleVLA-RL can optimize categorical action-token log probabilities, while LingBot-VA produces continuous action chunks through denoising transitions and does not naturally expose token-level action log probabilities.

The core technical idea is therefore to treat each action denoising transition as a stochastic policy step:

```text
pi_theta(x_{k-1} | x_k, obs, prompt, timestep)
```

instead of treating the policy as:

```text
pi_theta(action_token | obs, prompt)
```

The implemented system already supports grouped RoboTwin rollout collection, binary success/failure reward extraction, strict GRPO artifact capture, dynamic group construction, dataset validation, and offline GRPO smoke training on real rollout artifacts. The most important current limitation is that the trainer path is still a strict-artifact smoke trainer. It validates the data contract, logprob math, GRPO loss, optimizer, and checkpoint writing, but it does not yet update the real LingBot-VA actor weights.

The safest early-stage paper framing is:

> We identify and implement the first native data and optimization pipeline for applying grouped outcome RL to a FlowMatch-based continuous-action VLA, validate the rollout/artifact/data/training contract on RoboTwin, and characterize the task-selection and rollout-stability challenges that must be solved before full actor updates and benchmark gains can be claimed.

The paper should not yet claim policy improvement over LingBot-VA baseline, full online GRPO, or full denoising-trajectory actor replay.

---

## 2. Research Motivation

### 2.1 Why This Project Exists

Recent VLA RL systems such as SimpleVLA-RL show that closed-loop robot performance can improve when the policy is optimized with grouped online rollouts and binary task-success rewards. The key recipe is:

1. sample multiple rollouts from the same task context;
2. execute them in simulation;
3. assign binary success rewards;
4. keep only groups that contain both success and failure;
5. compute group-relative advantages;
6. apply GRPO or PPO-style clipped policy optimization.

This is a natural fit for token-action VLAs because action generation exposes discrete action-token probabilities. LingBot-VA has a different interface: it generates continuous action chunks through FlowMatch denoising. Therefore the project asks whether the same grouped outcome-RL idea can be transferred to a continuous denoising policy.

### 2.2 What Makes LingBot-VA Different

LingBot-VA is not a standard token-action VLA. It combines video-action modeling and action generation, with action-specific modules such as:

- `action_embedder`
- `condition_embedder_action`
- `action_proj_out`

Its action generation loop uses FlowMatch-style denoising, where the current implementation was originally deterministic. To support GRPO, WAM-RL introduces stochasticity into action denoising:

```text
x_{k-1} = mean_theta(x_k, obs, prompt, t) + sigma_rl(t) * epsilon
```

This gives a Gaussian transition distribution:

```text
p_theta(x_{k-1} | x_k, obs, prompt, t)
```

and a log probability that can be used in clipped GRPO.

### 2.3 Core Research Question

The current research question is:

> Can SimpleVLA-RL-style grouped outcome RL be transferred from token-based VLA policies to FlowMatch-based continuous action VLA policies by treating action denoising transitions as stochastic policy steps?

More specific sub-questions:

1. Can binary RoboTwin success/failure rewards produce useful group-relative advantages for LingBot-VA?
2. Is stochastic action denoising a practical replacement for token-level rollout temperature?
3. Which tasks provide enough mixed success/failure signal for GRPO?
4. What artifact schema is sufficient for reproducible rollout validation and actor replay?
5. Which actor parameters can be updated safely before moving to larger-scale distributed training?

---

## 3. Positioning Relative To Prior Systems

This project should be positioned carefully.

### 3.1 Relationship To SimpleVLA-RL

SimpleVLA-RL is the closest practical reference. The shared ingredients are:

- online or offline grouped rollout collection;
- binary task-success reward;
- group-relative advantage;
- dynamic sampling that discards all-success and all-failure groups;
- clipped policy optimization.

The main difference is the action policy interface:

```text
SimpleVLA-RL:
  policy object = discrete action-token distribution

WAM-RL:
  policy object = continuous FlowMatch denoising transition distribution
```

Therefore WAM-RL cannot simply reuse a token-response RL abstraction. It needs direct instrumentation of the LingBot-VA action denoising loop.

### 3.2 Relationship To VAMPO

VAMPO is conceptually useful because it treats diffusion denoising as a sequential decision process. However, the current project is not primarily VAMPO-style video world-model RL. The current objective is RoboTwin task success through action-policy optimization.

The useful VAMPO-like idea is:

```text
denoising trajectory can be treated as a policy trajectory
```

The current WAM-RL target is:

```text
binary outcome RL over executable robot action chunks
```

not:

```text
latent video-dynamics reward optimization
```

### 3.3 Relationship To veRL

veRL remains a possible later scale-up framework, but it is not the first implementation target. The reason is that SimpleVLA-RL's veRL extension is naturally aligned with token-response actors, while LingBot-VA needs continuous denoising-transition logprob replay.

The current stance is:

```text
Stage 1: native offline pipeline
Stage 2: native iterative online pipeline
Stage 3: veRL adapter after FlowMatch actor replay is validated
```

---

## 4. System Plan

The research plan is organized into three stages.

### 4.1 Stage 1: Native + Offline

Goal: validate the data contract and GRPO objective before updating the real actor.

Implemented and partially validated:

1. collect grouped RoboTwin rollouts;
2. save strict action-denoising artifacts;
3. flatten rollout records;
4. build GRPO groups;
5. validate all artifact references;
6. run offline GRPO smoke training.

This stage is designed to catch schema, logprob, grouping, and filesystem issues before expensive actor-training work.

### 4.2 Stage 2: Native + Iterative Online

Goal: run repeated collect-train-eval-promote loops with a real actor update.

Planned loop:

```text
actor_t
  -> grouped RoboTwin rollout collection
  -> strict artifact validation
  -> GRPO group construction
  -> actor replay and update
  -> evaluation on held-out seeds/tasks
  -> checkpoint promotion gate
  -> actor_{t+1}
```

The repository already contains framework-neutral loop pieces, but checkpoint promotion should remain disabled until real actor replay exists.

### 4.3 Stage 3: veRL Scale-Up

Goal: scale actor/rollout/reference computation after the native actor replay path is correct.

veRL should only be introduced after:

- the denoising-transition policy is validated;
- real actor logprob replay works;
- gradients are finite and non-zero on a tiny dataset;
- a trained checkpoint can be evaluated end to end.

---

## 5. Implemented Repository Components

### 5.1 Myriad / Workstation Jobs

The current job layer supports evaluation, grouped rollout collection, smoke training, and iteration wiring.

Key scripts:

- `jobs/myriad/10_eval_smoke_1gpu.sh`
- `jobs/myriad/11_eval_pilot_4gpu.sh`
- `jobs/myriad/12_eval_baseline_sweep_4gpu.sh`
- `jobs/myriad/13_eval_selected_tasks_4gpu.sh`
- `jobs/myriad/30_collect_grouped_rollouts_4gpu.sh`
- `jobs/myriad/30_collect_grouped_rollouts_1gpu.sh`
- `jobs/myriad/31_train_denoising_grpo_robotwin.sh`
- `jobs/myriad/32_submit_grpo_scale_8tasks_4gpu.sh`
- `jobs/myriad/33_submit_grpo_next_round_4gpu.sh`
- `jobs/myriad/40_rl_iteration_robotwin.sh`

The scripts support:

- 4-GPU Myriad collection;
- 1-GPU workstation collection;
- Apptainer container execution;
- `REPO_ROOT` override;
- `GROUP_SIZE` and `GROUPS_PER_TASK`;
- strict GRPO artifact capture;
- seed search and retry loops;
- successful and failed attempt manifests.

### 5.2 Rollout and Dataset Tools

The main Python tools are:

- `tools/collect_robotwin_rollouts.py`
- `tools/build_grpo_groups.py`
- `tools/validate_grpo_dataset.py`
- `tools/summarize_robotwin_results.py`
- `tools/train_offline_grpo_smoke.py`

Their current roles:

- collect rollout JSON records into flat JSONL/CSV summaries;
- group records by task, seed, prompt, and group id;
- filter all-success and all-failure groups;
- require complete group size and sample index coverage;
- require strict artifact references;
- optionally inspect artifact tensors;
- run offline GRPO smoke training.

### 5.3 RL Modules

The `wan_va/rl` package contains the native RL data contract and smoke-training path.

Implemented modules include:

- `wan_va/rl/scheduler_logprob.py`
- `wan_va/rl/dataset.py`
- `wan_va/rl/denoising_replay.py`
- `wan_va/rl/group_builder.py`
- `wan_va/rl/grpo_loss.py`
- `wan_va/rl/trainer.py`
- `wan_va/rl/checkpoint_gate.py`
- `wan_va/rl/iteration_controller.py`
- `wan_va/rl/trajectory_schema.py`
- `wan_va/rl/manifest.py`
- `wan_va/rl/validation.py`
- `wan_va/rl/reward.py`
- `wan_va/rl/evaluator.py`
- `wan_va/rl/rollout_worker.py`

The current trainer validates the strict-artifact GRPO path. It is not yet a real LingBot-VA actor trainer.

### 5.4 Server-Side Instrumentation

`wan_va/wan_va_server.py` now supports strict GRPO capture during action generation. For each captured transition, it saves tensors and metadata such as:

- `scope`
- `sampling_seed`
- `timestep`
- `frame_st_id`
- `action_xt`
- `action_xt_next`
- `transition_mean`
- `transition_std`
- `old_logprob_sum`
- `old_logprob_mean`
- `old_logprob_count`
- `logprob_mask`

The current artifact scope is:

```text
first_action_denoising_step
```

This is enough for strict artifact validation and smoke training, but not enough for full denoising-trajectory replay or full actor optimization.

---

## 6. Engineering Fixes Completed

Several practical bugs were found and fixed during rollout development.

### 6.1 Job and Environment Robustness

Completed fixes:

- fixed `REPO_ROOT` handling for script execution from interactive sessions;
- avoided stale `RESULTS_ROOT` inheritance in submit wrappers;
- avoided stale `STABLE_SEED_CACHE_DIR` inheritance;
- added Myriad and workstation-compatible collection scripts;
- added stricter checks for model paths, RoboTwin root, and `attn_mode`.

### 6.2 Client Argument Propagation

Early failures showed missing fields such as:

```text
KeyError: 'save_root'
KeyError: 'port'
```

The client launch path was fixed so required rollout arguments are propagated into `eval_polict_client_openpi.py`.

### 6.3 Group Identity and Validation

Early grouped rollout artifacts had group-id fragmentation due to prompt-hash suffixes. The grouping path was hardened with:

- canonical legacy group-id support;
- expected group size enforcement;
- expected `sample_idx` coverage;
- strict artifact existence checks;
- artifact shape/value checks;
- dataset validation summaries.

### 6.4 Retry and Seed Search

RoboTwin tasks can fail before policy evaluation because the expert precheck or environment setup is unstable for some seeds. The collection scripts now support:

- `GROUP_SEED_SEARCH=true`;
- `GROUP_SEED_SEARCH_MAX_ATTEMPTS`;
- `GROUP_MAX_ATTEMPTS`;
- `successful_attempt_roots.txt`;
- `failed_attempt_roots.txt`.

This lets failed group attempts be discarded while preserving logs for debugging.

### 6.5 Deterministic Seeding

The rollout path was adjusted to make seeds explicit at the process level before RoboTwin setup, so that grouped rollouts are easier to reproduce and diagnose.

---

## 7. Baseline Sweep Findings

A 50-task baseline sweep was summarized with 20 trials per task. The overall baseline result was:

```text
overall: 919 / 1000 = 91.9%
```

This creates an important task-selection issue: many tasks are too easy for useful GRPO because they produce all-success groups.

### 7.1 Hard Tasks

```text
hanging_mug      6 / 20   30.0%
turn_switch      9 / 20   45.0%
```

These tasks are high-value for GRPO because they are likely to produce mixed success/failure groups, but they are also more operationally unstable.

### 7.2 Medium Tasks

```text
open_microwave        11 / 20   55.0%
put_bottles_dustbin   14 / 20   70.0%
move_stapler_pad      16 / 20   80.0%
press_stapler         16 / 20   80.0%
place_dual_shoes      17 / 20   85.0%
place_fan             17 / 20   85.0%
put_object_cabinet    17 / 20   85.0%
stack_bowls_three     17 / 20   85.0%
```

These are the most relevant near-term collection targets because they can provide signal without being as unstable as the hardest tasks.

### 7.3 Easy Tasks

Many tasks are 95% to 100% successful under the baseline. Examples include:

```text
adjust_bottle
click_bell
pick_dual_bottles
place_mouse_pad
place_bread_basket
place_object_scale
rotate_qrcode
```

These tasks are useful for sanity checks and regression tests, but they are not efficient for GRPO collection because most groups are all-success and get discarded by dynamic sampling.

---

## 8. Collected Data And Smoke Results

The following results were observed from Myriad runs and subsequent validation commands. These are useful for paper-progress reporting, but should be framed as pipeline validation rather than final performance results.

### 8.1 B Collection: Easy-Heavy Tasks

Run path:

```text
/home/zcably0/Scratch/wam-rl/results_grouped_rollouts/grpo_scale_tasks_b_k8_g8_seedsearch_20260514_210921
```

Tasks:

```text
place_mouse_pad
adjust_bottle
pick_dual_bottles
click_bell
```

Rollout summary:

```text
adjust_bottle       62 / 64   96.9%
click_bell          64 / 64  100.0%
pick_dual_bottles   64 / 64  100.0%
place_mouse_pad     61 / 64   95.3%
overall            251 / 256  98.0%
```

GRPO group summary:

```text
total_groups: 32
mixed_groups: 4
skipped_all_success: 28
skipped_all_failure: 0
skipped_incomplete: 0
skipped_missing_artifacts: 0
mixed_group_ratio: 0.125
```

Dataset validation:

```text
transition_count: 199
error_count: 0
ok: true
```

Smoke training:

- 50-step CPU smoke passed;
- 300-step CPU smoke passed;
- 300-step final loss observed: `0.4946063160896301`;
- checkpoint and metrics were written.

Interpretation:

The B collection validates the pipeline, but most tasks are too easy to produce much GRPO learning signal.

### 8.2 A Partial Collection: Hard/Medium Mixed Tasks

Run path:

```text
/home/zcably0/Scratch/wam-rl/results_grouped_rollouts/grpo_scale_tasks_a_k8_g8_retry_20260515_000825
```

Tasks:

```text
hanging_mug
move_stapler_pad
open_microwave
turn_switch
```

One accepted group produced:

```text
hanging_mug        4 / 8   50.0%
move_stapler_pad   5 / 8   62.5%
open_microwave     5 / 8   62.5%
turn_switch        2 / 8   25.0%
overall           16 / 32  50.0%
```

GRPO group summary:

```text
total_groups: 4
mixed_groups: 4
skipped_all_success: 0
skipped_all_failure: 0
skipped_incomplete: 0
skipped_missing_artifacts: 0
mixed_group_ratio: 1.0
```

Dataset validation:

```text
transition_count: 528
error_count: 0
ok: true
```

Smoke training:

- 300-step CPU smoke passed;
- 300-step final loss observed: `0.6019473671913147`;
- checkpoint and metrics were written.

Interpretation:

This partial collection is more valuable for GRPO because every group is mixed, but it is small and expensive to collect because hard tasks create many failed group attempts.

### 8.3 Combined A Partial + B Dataset

Combined dataset path:

```text
/home/zcably0/Scratch/wam-rl/results_grpo_datasets/a_partial_plus_b_20260518_012501
```

Construction:

```text
A partial grpo_groups_partial.jsonl
+ B grpo_groups.jsonl
-> combined grpo_groups.jsonl
```

Validation:

```text
transition_count: 727
error_count: 0
ok: true
```

Smoke training:

- 300-step CPU smoke passed;
- 300-step final loss observed: `0.5725651979446411`;
- checkpoint and metrics were written.

Interpretation:

The combined dataset confirms that independent GRPO group JSONL files can be merged and trained through the smoke path. It remains a pipeline-validation artifact rather than a true actor-training dataset.

### 8.4 Current Follow-Up Collection Jobs

Recent submitted jobs split unstable and easier targets more carefully:

```text
core_no_mw:
  hanging_mug
  turn_switch
  put_bottles_dustbin
  move_stapler_pad

open_microwave:
  isolated open_microwave collection
```

Last observed status:

```text
core_no_mw:      2 accepted / 0 failed
open_microwave: 3 accepted / 3 failed
```

The split is justified because `open_microwave` was responsible for many failed mixed-task group attempts.

---

## 9. Open Microwave Instability

`open_microwave` is currently the clearest environment-level instability.

Observed failure pattern:

```text
open_microwave.py -> play_once()
  -> grasp_actor(... contact_point_id=1)
  -> pre_grasp_pose=None
  -> Action(... target_pose=None)
  -> AssertionError: target_pose cannot be None for move action
```

In grouped rollout jobs, this failure happens during expert precheck or task setup, before useful policy rollout data can be produced. It is not primarily a GRPO loss issue.

Current mitigation:

- isolate `open_microwave` into its own job;
- use seed search and retries;
- keep failed attempt logs for root-cause analysis;
- avoid letting `open_microwave` block unrelated hard/medium tasks.

Potential future fixes:

1. inspect RoboTwin `open_microwave.py` and `_base_task.py` contact point generation;
2. identify whether `contact_point_id=1` is invalid for some object initializations;
3. add a precheck seed filter specific to `open_microwave`;
4. patch RoboTwin task sampling only if the fix is clearly environment-level and does not bias policy evaluation.

Until this is fixed, `open_microwave` should be treated as a useful but unstable target.

---

## 10. What Is Validated

The following claims are currently supported by implementation and observed runs:

1. The repository can launch LingBot-VA RoboTwin grouped rollout collection on Myriad.
2. The server can save strict GRPO denoising-transition artifacts during action generation.
3. Rollout JSON records can reference executed actions, visualizations, and strict artifacts.
4. The grouping tool can construct mixed-success GRPO groups and reject invalid or incomplete groups.
5. Dataset validation can check references and inspect strict artifact tensors in the container environment.
6. Offline smoke training can load real strict artifacts, compute Gaussian transition logprobs, apply clipped GRPO loss, step an optimizer, and write checkpoints.
7. Baseline sweep results can guide task selection.
8. Easy tasks produce high success rates but low mixed-group ratios.
9. Hard/medium tasks produce more useful mixed groups but have higher rollout instability.
10. `open_microwave` is a special unstable case and should be isolated.

---

## 11. What Is Not Yet Validated

The following should not be claimed as completed:

1. Real LingBot-VA actor update.
2. Performance improvement over the LingBot-VA baseline.
3. Full online GRPO training.
4. Checkpoint promotion based on improved evaluation.
5. Full denoising-trajectory capture.
6. Full denoising-trajectory replay.
7. veRL integration.
8. Video-action consistency reward.
9. Generalization across unseen RoboTwin tasks.
10. Robust handling of `open_microwave` environment instability.

---

## 12. Current Main Technical Blocker

The main blocker is actor replay instrumentation.

Current strict artifacts contain enough information to validate a saved Gaussian transition:

```text
action_xt
action_xt_next
transition_mean
transition_std
old_logprob_sum
old_logprob_count
logprob_mask
```

But they do not yet contain enough context to recompute the current actor's `transition_mean_theta` from the original observation, prompt, timestep, and model state.

For real actor training, the trainer must be able to:

1. load a saved transition;
2. reconstruct the exact conditioning context;
3. run the current LingBot-VA actor forward pass;
4. recompute the current transition mean;
5. compute new logprob;
6. compare it to old logprob;
7. backpropagate through selected actor modules.

The missing pieces likely include:

- observation/video/text conditioning state;
- deterministic reconstruction of transformer cache or equivalent context;
- exact action denoising timestep state;
- CFG-conditioned action input structure;
- policy worker adapter that maps a saved artifact to a current actor replay call.

This is the highest-priority implementation milestone before claiming real GRPO.

---

## 13. Safe Claims For An Early-Stage Paper

The following claims are currently safe:

1. WAM-RL identifies a concrete mismatch between token-based VLA RL and FlowMatch continuous-action VLA RL.
2. WAM-RL formulates LingBot-VA action generation as a denoising-transition policy.
3. WAM-RL implements a native RoboTwin grouped rollout collection pipeline.
4. WAM-RL records strict denoising-transition artifacts during rollout.
5. WAM-RL builds group-relative GRPO datasets from binary task outcomes.
6. WAM-RL validates artifact references and tensor integrity.
7. WAM-RL demonstrates offline GRPO smoke training on real RoboTwin rollout artifacts.
8. WAM-RL empirically characterizes task difficulty and mixed-group availability on RoboTwin.
9. WAM-RL identifies rollout instability as a first-class systems issue for online VLA RL.
10. WAM-RL provides a staged path from offline validation to native online actor updates and later veRL scale-up.

---

## 14. Unsafe Claims To Avoid

The paper should not claim:

1. WAM-RL improves LingBot-VA benchmark performance.
2. WAM-RL completes online GRPO for LingBot-VA.
3. WAM-RL has a fully working real actor trainer.
4. WAM-RL performs full trajectory-level denoising optimization.
5. WAM-RL solves `open_microwave` instability.
6. WAM-RL is already equivalent to SimpleVLA-RL scale or maturity.
7. WAM-RL validates video-action consistency preservation.
8. WAM-RL demonstrates robust generalization beyond the tested RoboTwin tasks.

If these topics are discussed, they should be framed as future work or ongoing implementation.

---

## 15. Suggested Paper Framing

### 15.1 Possible Title Direction

Possible working titles:

```text
Towards Outcome RL for FlowMatch-Based Video-Action Models
```

```text
WAM-RL: A Native Pipeline for Grouped Outcome RL in Continuous-Action Video-Action Models
```

```text
Denoising-Step GRPO for FlowMatch-Based Robotic Action Generation
```

The safest title should use "Towards" or "Pipeline" unless real actor-update results are added before submission.

### 15.2 Abstract-Level Framing

A truthful abstract should emphasize:

- token-based VLA RL is not directly applicable to FlowMatch action generation;
- WAM-RL defines a denoising-transition policy;
- the system captures rollout artifacts and builds GRPO groups from RoboTwin outcomes;
- smoke training validates the optimization contract;
- remaining work is real actor replay and closed-loop improvement.

It should not present smoke loss curves as policy improvement.

### 15.3 Contributions

Possible contribution list:

1. **Problem formulation:** define grouped outcome RL for FlowMatch continuous-action VLA policies through denoising-transition logprob.
2. **System pipeline:** implement native RoboTwin rollout collection, strict artifact capture, dynamic GRPO grouping, validation, and smoke training.
3. **Empirical characterization:** analyze RoboTwin task difficulty, mixed-group yield, and rollout instability for online VLA RL.
4. **Roadmap and failure analysis:** identify actor replay and environment instability as the key blockers for full online GRPO.

### 15.4 Recommended Figure

A useful early paper figure would show:

```text
RoboTwin rollout group
  -> binary success/failure
  -> strict denoising artifacts
  -> group-relative advantage
  -> denoising-step GRPO objective
  -> future actor replay update
```

The figure should visually separate:

- implemented path: rollout, artifacts, grouping, validation, smoke trainer;
- future path: real actor replay and checkpoint promotion.

This separation matters because it prevents overclaiming.

---

## 16. Recommended Experimental Story

The current experimental story should be framed as pipeline and feasibility evaluation.

### 16.1 Baseline Task Difficulty

Report the 50-task baseline sweep:

- overall baseline success: `91.9%`;
- hard tasks: `hanging_mug`, `turn_switch`;
- medium tasks: `open_microwave`, `put_bottles_dustbin`, `move_stapler_pad`, `press_stapler`, `place_dual_shoes`, `place_fan`, `put_object_cabinet`, `stack_bowls_three`;
- many easy tasks are unsuitable for GRPO because they produce all-success groups.

Main takeaway:

> Task selection is not a minor detail. For group-relative RL, the best tasks are not necessarily the hardest or easiest tasks, but tasks that reliably produce mixed outcomes.

### 16.2 Mixed-Group Yield

Compare:

```text
B easy-heavy collection:
  4 mixed groups / 32 total groups = 12.5%

A hard/medium partial:
  4 mixed groups / 4 total groups = 100%
```

Main takeaway:

> Easy tasks provide stable rollouts but weak GRPO signal, while hard/medium tasks provide stronger signal but higher collection instability.

### 16.3 Artifact Validation

Report that validation passed for:

```text
B collection:          199 strict transitions
A partial collection:  528 strict transitions
combined dataset:      727 strict transitions
```

Main takeaway:

> The strict artifact schema is sufficient for reproducible offline GRPO smoke training.

### 16.4 Smoke Training

Report smoke training only as optimization-contract validation:

```text
B 300-step smoke:
  final_loss = 0.4946063160896301

A partial 300-step smoke:
  final_loss = 0.6019473671913147

A partial + B 300-step smoke:
  final_loss = 0.5725651979446411
```

Do not interpret these as policy performance metrics.

---

## 17. Near-Term Roadmap

### 17.1 Data Collection

Immediate data tasks:

1. finish current `core_no_mw` collection;
2. finish or stop `open_microwave` isolated collection based on accepted/failed attempt ratio;
3. merge accepted groups into a combined training JSONL;
4. compute per-task dataset accounting:
   - rollout count;
   - success count;
   - failure count;
   - mixed group count;
   - strict transition count;
   - artifact validation status.

The next collection strategy should prioritize medium/hard tasks with useful mixed-group yield:

```text
high priority:
  hanging_mug
  turn_switch
  put_bottles_dustbin
  move_stapler_pad
  press_stapler

isolate:
  open_microwave

secondary:
  place_dual_shoes
  place_fan
  put_object_cabinet
  stack_bowls_three
```

### 17.2 Actor Replay

The next major engineering milestone is real actor replay. A minimal first version should:

1. capture enough conditioning context for one denoising step;
2. recompute `transition_mean_theta` under the current model;
3. compute new Gaussian transition logprob;
4. backpropagate into a small action-specific parameter set;
5. verify non-zero finite gradients;
6. run a tiny train step on 1 to 2 groups;
7. save a real actor checkpoint.

The first update surface should likely be action-specific modules:

```text
action_embedder
condition_embedder_action
action_proj_out
```

This is safer than full-model fine-tuning.

### 17.3 Evaluation

After a real actor checkpoint exists, evaluation should include:

1. trained checkpoint vs base checkpoint;
2. same tasks as training;
3. held-out seeds for each task;
4. at least one easy sanity task to detect collapse;
5. strict logging of success rate confidence intervals;
6. checkpoint promotion only if improvement passes a gate.

### 17.4 Full Denoising Trajectory

Current artifacts capture `first_action_denoising_step`. Later work should capture more or all action denoising steps. This would make the objective closer to the intended full action-chunk probability:

```text
log pi_theta(action_chunk | obs, prompt)
  ~= sum_k log pi_theta(x_{k-1} | x_k, obs, prompt, t_k)
```

### 17.5 Video-Action Consistency

Video-action consistency should remain future work until action-only GRPO is stable. Potential directions:

- preserve imagined video-action alignment;
- penalize action improvements that degrade future visual coherence;
- use WAM-predicted futures for reward shaping or rejection;
- connect to skill arbitration in later work.

---

## 18. Risks And Mitigations

### 18.1 Risk: Too Many All-Success Groups

Easy tasks waste rollout budget because dynamic sampling discards all-success groups.

Mitigation:

- prioritize hard/medium tasks;
- track mixed-group ratio during collection;
- use easy tasks primarily as regression tests.

### 18.2 Risk: Environment-Level Failures

Some tasks fail during expert precheck before policy evaluation.

Mitigation:

- seed search;
- retry loops;
- isolate unstable tasks;
- keep failed attempt logs;
- consider task-specific seed filters.

### 18.3 Risk: Smoke Trainer Gives False Confidence

The current trainer can pass even though it does not update the real actor.

Mitigation:

- label it clearly as smoke training;
- do not report it as policy learning;
- prioritize actor replay instrumentation.

### 18.4 Risk: Full Actor Replay Is Expensive

Reconstructing the LingBot-VA conditioning context may be difficult.

Mitigation:

- start with one denoising step;
- capture minimal replay context;
- train only action-specific modules first;
- add full denoising trajectory later.

### 18.5 Risk: Reward Hacking Or Action Collapse

Binary reward alone may encourage brittle actions or damage video-action coherence.

Mitigation:

- evaluate on held-out seeds;
- include easy sanity tasks;
- add checkpoint gates;
- later add video-action consistency constraints.

---

## 19. Recommended Next Milestone Checklist

Before claiming a real RL result, complete:

- [ ] finish current `core_no_mw` and `open_microwave` collection accounting;
- [ ] merge all valid groups into a single versioned dataset;
- [ ] produce per-task group and transition statistics;
- [ ] add real actor replay context capture;
- [ ] recompute current actor logprob from saved artifacts;
- [ ] verify finite non-zero gradients on action-specific modules;
- [ ] train a tiny actor checkpoint;
- [ ] evaluate checkpoint vs base on held-out seeds;
- [ ] add checkpoint promotion gate;
- [ ] document exact result paths and command lines.

For an early-stage paper draft, the minimum useful artifact set is:

- [ ] baseline sweep table;
- [ ] mixed-group yield table;
- [ ] strict artifact validation table;
- [ ] smoke-training contract table;
- [ ] failure-mode analysis table;
- [ ] roadmap figure separating implemented and future components.

---

## 20. Current Bottom Line

The project is in a credible early-stage systems-and-methods phase. The main value already created is not benchmark improvement yet, but a concrete and tested bridge from token-style VLA RL ideas to FlowMatch continuous-action VLA infrastructure:

```text
RoboTwin grouped rollouts
  -> binary outcome rewards
  -> strict action-denoising artifacts
  -> dynamic GRPO groups
  -> validated offline denoising-step GRPO smoke trainer
```

The next decisive step is real LingBot-VA actor replay and update:

```text
saved denoising transition
  -> reconstruct conditioning context
  -> recompute current actor transition_mean_theta
  -> compute new logprob
  -> apply GRPO update
  -> evaluate checkpoint
```

Until that step is complete, the honest paper framing is "towards" or "pipeline and feasibility" rather than a full RL-improvement paper.
