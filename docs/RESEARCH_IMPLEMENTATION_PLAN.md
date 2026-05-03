# Research Implementation Plan

This plan translates `Liuhaichen_Yang_research_proposal.pdf` into concrete
repo milestones for LingBot-VA/RoboTwin work.

## Core Research Question

Can outcome-based RL post-training improve a pretrained World Action Model
(WAM) beyond demonstration-only adaptation while preserving video-action
coherence?

The proposal has three implementation axes:

1. Conservative outcome-based post-training, inspired by grouped trajectory
   comparison methods such as GRPO.
2. Parameter-efficient adaptation, rather than broad end-to-end updates.
3. Video-action consistency regularization between imagined futures and
   observed rollout futures.

## Current Status

- Apptainer/PyTorch runtime works on Myriad A100 nodes.
- LingBot-VA server inference works with the RoboTwin checkpoint.
- RoboTwin/SAPIEN/CuRobo evaluation works from SGE jobs.
- One-GPU smoke evaluation passed.
- Four-GPU pilot evaluation passed: `38/40 = 95%`.
- Baseline sweep workflow is now available for task selection.

## Phase 1: Baseline Sweep And Task Selection

Goal: identify tasks with enough headroom to show method improvement without
falling into floor effects.

Run a full baseline sweep:

```bash
qsub -v TEST_NUM=20 jobs/myriad/12_eval_baseline_sweep_4gpu.sh
```

Selection rule:

- `easy`: success rate >= 90%. Keep 1-2 tasks as sanity checks.
- `medium`: 50% <= success rate < 90%. Use as primary evaluation tasks.
- `hard`: 10% <= success rate < 50%. Use as challenge tasks.
- `too_hard`: success rate < 10%. Keep for failure analysis unless later
  reward shaping makes progress measurable.

The selected set should be fixed before method tuning to avoid cherry-picking.

## Phase 2: Formal Selected-Task Evaluation

Use the selected-task job to run a fixed task set with larger `TEST_NUM`:

```bash
qsub -v \
  EVAL_NAME=baseline_selected,\
TASK_NAMES="task_a task_b task_c task_d",\
TEST_NUM=50 \
  jobs/myriad/13_eval_selected_tasks_4gpu.sh
```

For paired comparison, reuse the same `TASK_NAMES`, `TEST_NUM`, and `SEED` for
baseline and method checkpoints:

```bash
qsub -v \
  EVAL_NAME=method_selected,\
WAN_VA_MODEL_PATH="$HOME/Scratch/wam-rl/checkpoints/METHOD_CHECKPOINT",\
TASK_NAMES="task_a task_b task_c task_d",\
TEST_NUM=50 \
  jobs/myriad/13_eval_selected_tasks_4gpu.sh
```

Primary metrics:

- per-task success rate
- aggregate success rate
- Wilson confidence interval from `tools/summarize_robotwin_results.py`
- rollout videos and failure traces for qualitative analysis

## Phase 3: Conservative Adaptation Baseline

Before implementing RL, establish the smallest update surface that can train
and evaluate end-to-end.

Initial target:

- freeze most pretrained model parameters
- update only action-related modules or a small adapter/LoRA-style subset
- keep the current supervised diffusion/flow matching loss as a wiring check

This phase should answer whether parameter-efficient updates are viable on
4xA100 40GB before adding RL instability.

Implementation scaffold now exists:

- `robotwin_peft_train` config uses `trainable_mode=action_heads`.
- `WAN_VA_TRAINABLE_MODE` can select `full`, `action_heads`, `patterns`, or
  `frozen`.
- `jobs/myriad/21_train_peft_tiny_4gpu.sh` runs a short 4-GPU PEFT wiring test.
- `tools/collect_robotwin_rollouts.py` converts RoboTwin rollout JSON records
  into a flat reward dataset for later offline/grouped RL experiments.

## Phase 4: Outcome-Based Post-Training

Use the selected medium/hard tasks to collect grouped rollouts from the current
policy. Start conservatively:

- sample multiple action chunks per prompt/observation
- score rollouts using binary success or task progress
- compare grouped samples within the same task/seed context
- regularize updates toward the reference model using KL-style or
  behavior-cloning penalties where tractable

Exact diffusion-policy likelihoods may be impractical. The first practical
implementation can use surrogate losses over sampled action chunks and only move
to a stricter GRPO-style objective after the rollout/reward/update loop is
stable.

## Phase 5: Video-Action Consistency

After outcome optimization works, add a bounded auxiliary consistency term:

- compare predicted future observations against observed rollout observations
- keep the term secondary to task reward
- ablate reward-only vs reward-plus-consistency

The key risk is reward hacking or over-penalizing visual prediction errors that
do not matter for task success, so this term should be introduced only after a
stable outcome baseline exists.

## Near-Term Checklist

1. Finish `TEST_NUM=20` baseline sweep.
2. Select fixed easy/medium/hard task set from `summary.csv`.
3. Run `13_eval_selected_tasks_4gpu.sh` for the baseline checkpoint.
4. Inspect training code and decide the first parameter-efficient update target.
5. Implement a minimal adapter/freeze configuration.
6. Run tiny training smoke, then selected-task evaluation on the adapted model.
