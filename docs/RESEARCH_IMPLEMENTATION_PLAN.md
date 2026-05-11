# Research Implementation Plan

This document turns `Liuhaichen_Yang_research_proposal.pdf` into an
implementation-grade plan for outcome-based RL post-training on LingBot-VA /
RoboTwin. It also records the SimpleVLA-RL design lessons that are useful for
this repository, while keeping the objective honest about the differences
between token-policy VLA models and LingBot-VA's diffusion / flow-matching
action model.

## Core Research Question

Can grouped outcome-based RL post-training improve a pretrained World Action
Model (WAM) / LingBot-VA policy beyond demonstration-only adaptation, using
task success rewards, while preserving action quality and video-action
coherence?

The project has four research axes:

1. Conservative outcome-based post-training, inspired by grouped trajectory
   comparison methods such as GRPO.
2. Parameter-efficient adaptation, rather than broad end-to-end updates.
3. VLA-specific grouped rollout collection on RoboTwin.
4. Video-action consistency regularization between imagined futures and
   observed rollout futures, introduced only after reward-only RL is stable.

## External Reference: SimpleVLA-RL

SimpleVLA-RL is the closest practical reference for this project:

- Repository: <https://github.com/PRIME-RL/SimpleVLA-RL>
- Paper: <https://arxiv.org/abs/2509.09674>
- RoboTwin2 example config:
  <https://raw.githubusercontent.com/PRIME-RL/SimpleVLA-RL/main/examples/run_openvla_oft_rl_twin2.sh>

Useful design lessons:

- Build an end-to-end RL loop with separate rollout, reward, advantage,
  training, evaluation, and checkpointing components.
- Use binary `0/1` outcome rewards first. Avoid task-specific reward shaping
  until the minimal reward setting is established.
- Filter tasks by baseline success rate. SimpleVLA-RL uses a practical range
  such as `0.1 <= success_rate <= 0.9`; this matches our hard/medium task
  selection rule.
- Sample multiple rollouts per context. The RoboTwin example uses
  `data.n_samples=8`.
- Use exploration controls during rollout. The OpenVLA-OFT example uses
  rollout temperature tuning; for LingBot-VA the corresponding controls are
  sampling seed, action/video guidance scale, action denoising stochasticity,
  and action inference settings.
- Use conservative clipping and small actor learning rates. The example uses
  GRPO advantage estimation and asymmetric PPO-style clip values.
- Evaluate frequently and gate checkpoints. RL improvements are only useful if
  they survive fixed-seed selected-task evaluation and do not regress easy
  sanity tasks.

Important mismatch:

- SimpleVLA-RL's OpenVLA / OpenVLA-OFT path can compute action-token log
  probabilities. LingBot-VA generates action chunks through diffusion /
  flow-matching denoising, so exact `log pi(a | s)` is not immediately
  available.
- Therefore the first LingBot-VA implementation should be described as
  `GRPO-style outcome RL` or `WAM-GRPO surrogate`, not as strict GRPO, until a
  diffusion-policy likelihood or equivalent trajectory probability estimator is
  implemented.

## Current Status

- Apptainer/PyTorch runtime works on Myriad A100 nodes.
- LingBot-VA server inference works with the RoboTwin checkpoint.
- RoboTwin/SAPIEN/CuRobo evaluation works from SGE jobs.
- One-GPU smoke evaluation passed.
- Four-GPU pilot evaluation passed: `38/40 = 95%`.
- Baseline sweep workflow exists and job `272608` completed.
- The first sweep result is usable for task selection, but not for RL training
  data, because the rollout directory for that run did not contain saved
  per-rollout action records.
- Selected-task evaluation job exists.
- Rollout JSON / action logging scaffold exists in
  `evaluation/robotwin/eval_polict_client_openpi.py`.
- Flat rollout collection scaffold exists in `tools/collect_robotwin_rollouts.py`.
- PEFT scaffold exists:
  - `robotwin_peft_train` uses `trainable_mode=action_heads`.
  - `WAN_VA_TRAINABLE_MODE` supports `full`, `action_heads`, `patterns`, and
    `frozen`.
  - `action_heads` trains `action_embedder`, `condition_embedder_action`, and
    `action_proj_out`.
  - `jobs/myriad/21_train_peft_tiny_4gpu.sh` runs a tiny PEFT training smoke.

## Fixed Task Policy

The selected task set must be fixed before method tuning. Do not change the
task list based on later RL results.

Initial selected tasks:

```text
hanging_mug
turn_switch
open_microwave
put_bottles_dustbin
move_stapler_pad
press_stapler
blocks_ranking_rgb
place_dual_shoes
place_fan
put_object_cabinet
stack_bowls_three
adjust_bottle
click_bell
```

Roles:

- `hard`: baseline success roughly `10%-50%`; challenge tasks.
- `medium`: baseline success roughly `50%-90%`; primary RL targets.
- `easy`: baseline success `>=90%`; sanity tasks for regression detection.
- `too_hard`: baseline success `<10%`; keep for diagnostics unless later
  progress rewards make improvement measurable.

Future high-compute setting:

- `train_tasks`: 30-60 tasks with baseline success in `10%-90%`.
- `easy_sanity_tasks`: 5-10 tasks with high baseline success.
- `heldout_tasks`: 10-20 tasks never used for RL training.
- `heldout_seeds`: environment seeds not used during rollout collection.

## Phase 1: Baseline And Selected-Task Evaluation

Goal: establish fixed, reproducible baselines before any RL update.

Baseline sweep:

```bash
qsub -v TEST_NUM=20 jobs/myriad/12_eval_baseline_sweep_4gpu.sh
```

Selected-task baseline:

```bash
qsub -v \
EVAL_NAME=baseline_selected_seed1_50,\
SEED=1,\
TASK_NAMES="hanging_mug turn_switch open_microwave put_bottles_dustbin move_stapler_pad press_stapler blocks_ranking_rgb place_dual_shoes place_fan put_object_cabinet stack_bowls_three adjust_bottle click_bell",\
TEST_NUM=50 \
jobs/myriad/13_eval_selected_tasks_4gpu.sh
```

Required baseline artifacts:

- `summary.csv` or summary text from `tools/summarize_robotwin_results.py`.
- Per-task success rates and Wilson confidence intervals.
- Rollout videos for representative success and failure cases.
- Saved rollout JSON and action tensors for any run intended for RL data.

If `rollouts/` is empty, the run is evaluation-only and cannot be used for RL
training.

## Phase 2: Conservative PEFT / SFT Wiring Baseline

Goal: prove that the selected update surface can train, save, reload, and
evaluate before adding RL instability.

Initial training mode:

```text
trainable_mode = action_heads
trainable modules =
  action_embedder
  condition_embedder_action
  action_proj_out
```

Requirements:

- Run `jobs/myriad/21_train_peft_tiny_4gpu.sh`.
- Confirm trainable parameter summary is correct.
- Confirm FSDP can handle mixed frozen/trainable parameters.
- Confirm checkpoint save and reload work.
- Evaluate the PEFT checkpoint on the selected task set.

Promotion gate:

- The tiny training job completes without FSDP, optimizer, or checkpoint errors.
- The checkpoint can be served by the existing LingBot-VA inference path.
- Selected-task evaluation does not collapse easy sanity tasks.

## Phase 3: RL Rollout Data Standard

GRPO requires grouped samples from the same context. For robotics, a context is:

```text
context = (task_name, env_seed, prompt, initial_observation)
```

Within a context, sample `K` rollouts by varying only policy sampling noise or
explicitly recorded exploration settings.

Recommended group sizes:

- `K=4`: debugging.
- `K=8`: first serious RL run, matching the SimpleVLA-RL example scale.
- `K=16`: high-compute setting for better reward variance.
- Adaptive `K`: future setting, more samples for high-variance tasks and fewer
  for solved tasks.

Every rollout JSON must include:

```json
{
  "run_id": "rl_iter_0000",
  "policy_checkpoint": "/path/to/actor",
  "reference_checkpoint": "/path/to/reference",
  "task_name": "open_microwave",
  "group_id": "open_microwave_seed10042_prompt0",
  "sample_idx": 3,
  "group_size": 8,
  "env_seed": 10042,
  "sampling_seed": 730003,
  "prompt": "...",
  "success": true,
  "reward": 1.0,
  "actions_path": ".../executed_actions.npy",
  "server_action_paths": [".../actions_0.pt", ".../actions_4.pt"],
  "server_latent_paths": [".../latents_0.pt"],
  "initial_obs_path": "...",
  "visualization_path": "...",
  "take_action_cnt": 112,
  "step_lim": 200,
  "video_guidance_scale": 1.0,
  "action_guidance_scale": 1.0,
  "action_num_inference_steps": 50
}
```

Data correctness checks:

- Every selected task has the expected number of groups.
- Every group has exactly `K` rollout records unless explicitly marked
  incomplete.
- Every record has existing action tensor paths.
- Every mixed group has at least one success and one failure.
- All rollout metadata records the actor checkpoint and sampling settings.
- Server-side action tensors are preferred for RL training because executed
  env actions may be postprocessed and may not match the model action space.

## Phase 4: Group Builder And Advantage Computation

Create `tools/build_grpo_groups.py`.

Inputs:

- One or more RoboTwin result roots.
- Optional selected task filter.
- Optional minimum group size.
- Optional policy checkpoint filter.

Outputs:

- `grpo_groups.jsonl`
- `grpo_groups_summary.csv`
- validation report printed to stdout.

Each training row should include:

```json
{
  "group_id": "open_microwave_seed10042_prompt0",
  "task": "open_microwave",
  "reward": 1.0,
  "advantage": 0.935,
  "group_reward_mean": 0.375,
  "group_reward_std": 0.484,
  "sample_idx": 3,
  "actions_path": "...",
  "server_action_paths": ["..."],
  "record_path": "...",
  "policy_checkpoint": "...",
  "reference_checkpoint": "..."
}
```

Advantage rule:

```text
A_i = (r_i - mean(r_group)) / (std(r_group) + eps)
```

Rules:

- Skip groups with no reward variance for the first RL implementation.
- Clip advantages to a bounded interval, initially `[-2, 2]`.
- Track the fraction of effective mixed groups.
- Balance tasks so high-throughput tasks do not dominate the RL dataset.

## Phase 5: RL Objective Roadmap

### Phase 5A: Reward-Weighted Action Flow Matching

This is the first stable RL baseline. It is not strict GRPO, but it is the
right first implementation for LingBot-VA.

Base action loss:

```text
loss_action = flow_matching_mse(action_pred, action_target)
```

Reward/advantage weighting:

```text
weight_i = clamp(1 + alpha * A_i, 0, w_max)
loss = weight_i * loss_action
```

Alternative positive-only variant:

```text
weight_i = max(A_i, 0)
```

Use this variant if negative advantages destabilize early training.

Required regularizers:

- Reference action loss against the frozen SFT/pretrained checkpoint.
- Optional demonstration BC anchor from the original LeRobot latent dataset.
- Gradient clipping.
- Small learning rate.

### Phase 5B: WAM-GRPO v0 Pseudo-Ratio

Once Phase 5A works, introduce a GRPO-style clipped surrogate using a
flow-loss-derived pseudo-logprob:

```text
pseudo_logp_theta = -loss_action_per_sample / tau
rho = exp(pseudo_logp_theta - pseudo_logp_old)
```

Clipped objective:

```text
loss_grpo = -min(
  rho * A,
  clip(rho, 1 - eps_low, 1 + eps_high) * A
)
```

Initial settings:

```text
eps_low = 0.2
eps_high = 0.28
tau = tuned on a small heldout rollout batch
advantage_clip = [-2, 2]
```

Important claim boundary:

- This is `GRPO-style`, because `pseudo_logp` is derived from denoising /
  flow-matching reconstruction quality, not exact action-token likelihood.
- Do not present this as strict GRPO until Phase 5C exists.

### Phase 5C: Diffusion / Flow-Matching GRPO

Strict GRPO requires a more faithful probability estimate for generated action
chunks.

Additional rollout data to save:

- Initial action noise.
- Denoising timestep sequence.
- Noisy action state at each timestep.
- Model prediction at each timestep.
- Scheduler transition outputs.
- Final normalized action tensor.

Then estimate:

```text
log p_theta(action | obs, prompt)
  ~= sum_t log p_theta(x_{t-1} | x_t, obs, prompt)
```

This phase is a research contribution by itself. Implement it only after the
surrogate RL loop is validated.

### Phase 5D: Online Actor-Reference RL

Final online setup:

```text
actor_model      = checkpoint being trained
old_policy       = frozen actor snapshot used to collect the current rollout batch
reference_model  = original SFT/pretrained LingBot-VA checkpoint
```

Full loss:

```text
loss =
  clipped_grouped_outcome_loss(actor, old_policy)
  + beta_ref * reference_action_loss(actor, reference_model)
  + beta_bc * demonstration_anchor_loss
  + beta_video * video_action_consistency_loss
```

Keep `beta_video = 0` until reward-only RL is stable.

## Phase 6: Scalable Online RL System

Ignoring current compute constraints, the target production-grade loop should
have six components:

1. `TaskSampler`: samples tasks, seeds, prompts, and initial states.
2. `RolloutWorker`: launches LingBot-VA server(s), runs RoboTwin environments,
   and saves full rollout artifacts.
3. `RewardWorker`: verifies success, computes binary reward, and validates
   records.
4. `GroupBuilder`: constructs groups and advantages.
5. `RLTrainer`: trains actor checkpoints using Phase 5 objectives.
6. `Evaluator`: runs fixed selected-task and heldout evaluation and gates
   checkpoint promotion.

Online iteration:

```text
for iteration t:
  1. freeze actor theta_t as old_policy
  2. collect grouped rollouts with theta_t
  3. build advantage dataset
  4. train theta_{t+1}
  5. evaluate theta_{t+1}
  6. promote theta_{t+1} only if gates pass
```

Checkpoint promotion gates:

- Aggregate selected-task success improves over the previous promoted
  checkpoint.
- Wilson confidence interval is reported for all headline comparisons.
- No severe regression on easy sanity tasks.
- At least a target number of hard/medium tasks improve.
- Action magnitude, action saturation, and episode length remain in a sane
  range.
- Qualitative videos do not show obvious reward hacking or incoherent
  video-action behavior.

## Phase 7: Video-Action Consistency

Introduce this only after outcome optimization works.

Goal:

- Prevent action-only reward optimization from breaking WAM video-action
  coherence.

Candidate term:

```text
loss_video_consistency =
  distance(imagined_future_latents, observed_rollout_future_latents)
```

Rules:

- Keep this term secondary to reward optimization.
- Use a small coefficient.
- Report reward-only vs reward-plus-consistency as an ablation.
- Do not over-penalize visual prediction errors that do not affect task
  success.

## Evaluation Protocol

Every meaningful checkpoint must be evaluated on:

- Selected train-task seeds.
- Selected heldout seeds.
- Heldout tasks.
- Easy sanity tasks.
- Qualitative rollout videos.

Primary metrics:

- Per-task success rate.
- Aggregate success rate.
- Wilson confidence interval from `tools/summarize_robotwin_results.py`.
- Delta over pretrained LingBot-VA baseline.
- Delta over PEFT-SFT baseline.
- Number of tasks improved, unchanged, and regressed.
- Effective mixed-group ratio during rollout collection.
- Episode length and early success rate.
- Action magnitude, action saturation, and invalid action statistics.
- Failure type taxonomy from videos.

Minimum reporting table:

```text
method
checkpoint
trainable_mode
rollout_K
objective
selected_success
hard_success
medium_success
easy_sanity_success
heldout_success
num_improved_tasks
num_regressed_tasks
```

## Ablation Matrix

Core method ablations:

- Pretrained LingBot-VA.
- PEFT-SFT with `action_heads`.
- Reward-weighted action flow matching.
- WAM-GRPO v0 pseudo-ratio.
- WAM-GRPO v0 plus reference regularization.
- WAM-GRPO v0 plus demonstration BC anchor.
- WAM-GRPO v0 plus video-action consistency.

Exploration ablations:

- `K=4`, `K=8`, `K=16`.
- Sampling seed only.
- Sampling seed plus action guidance scale sweep.
- Sampling seed plus action denoising stochasticity.
- Fixed sampling vs adaptive sampling.

Task ablations:

- Medium-only.
- Hard plus medium.
- Hard plus medium plus easy sanity.
- All filtered tasks.
- Heldout task generalization.

Update-surface ablations:

- `action_heads`.
- `action_heads` plus action-specific norms/time embeddings.
- LoRA/adapters on selected transformer blocks.
- Full fine-tuning.

Objective ablations:

- Positive-only reward weighting.
- Signed advantage weighting.
- Pseudo-ratio clipping on/off.
- Reference loss on/off.
- Demonstration anchor on/off.
- Advantage normalization on/off.

## Implementation Milestones

### Milestone 1: Rollout Data Correctness

Implement:

- `group_id`, `sample_idx`, `group_size`, `run_id`, and checkpoint metadata in
  rollout JSON.
- Reliable `server_action_paths` and `server_latent_paths`.
- Initial observation save path.
- Sampling metadata.
- A grouped rollout job for selected tasks.
- A validation script that checks group completeness and path existence.

Exit criteria:

- A debug run produces complete groups.
- `find "$RESULTS/rollouts" -name "*.json"` matches expected count.
- Every JSON points to existing action tensors.

### Milestone 2: Group Builder

Implement:

- `tools/build_grpo_groups.py`.
- `grpo_groups.jsonl`.
- `grpo_groups_summary.csv`.
- Advantage computation and group filtering.

Exit criteria:

- The script reports task-level group counts, mixed-group ratio, and reward
  stats.
- The output is deterministic for the same inputs.

### Milestone 3: Offline RL Dataset

Implement:

- `wan_va/rl/dataset.py`.
- Loader for `grpo_groups.jsonl`.
- Action tensor loading from server-side action paths.
- Reward and advantage fields.
- Shape validation against LingBot-VA action config.

Exit criteria:

- A standalone dataloader smoke test can iterate batches.
- The dataset rejects missing or malformed action paths.

### Milestone 4: Reward-Weighted Trainer

Implement:

- `wan_va/rl_train.py` or a clearly separated RL trainer path.
- `wan_va/configs/va_robotwin_grpo_train_cfg.py`.
- `jobs/myriad/30_train_reward_weighted_4gpu.sh`.
- `trainable_mode=action_heads` by default.

Exit criteria:

- A tiny RL training smoke completes.
- Loss, advantage, reward, and gradient stats are logged.
- Checkpoint save/reload works.

### Milestone 5: WAM-GRPO v0

Implement:

- Per-sample action flow loss.
- Pseudo-logprob computation.
- Old pseudo-logprob cache.
- Ratio clipping.
- Reference action loss.
- Advantage clipping.

Exit criteria:

- Pseudo-ratio values are numerically stable.
- Training does not collapse easy sanity tasks.
- Selected-task evaluation improves at least one hard/medium task without
  severe aggregate regression.

### Milestone 6: Online RL Loop

Implement:

- `jobs/myriad/40_rl_iteration.sh` or equivalent orchestration.
- Iteration directories:
  - `rollouts/`
  - `groups/`
  - `checkpoints/`
  - `eval/`
  - `reports/`
- Checkpoint promotion logic.

Exit criteria:

- One full iteration can collect rollouts, train, evaluate, and decide
  promote/reject automatically.

### Milestone 7: Strict Diffusion GRPO

Implement only after Milestone 6:

- Denoising trajectory logging.
- Diffusion transition likelihood or defensible surrogate.
- Comparison between reward-weighted, pseudo-ratio, and diffusion-ratio
  objectives.

Exit criteria:

- The strict likelihood objective is numerically stable.
- It beats or explains the pseudo-ratio baseline.

### Milestone 8: Video-Action Consistency

Implement only after reward-only RL is reliable:

- Observed future latent extraction.
- Imagined-vs-observed consistency term.
- Reward-only vs reward-plus-consistency ablation.

Exit criteria:

- The consistency term improves qualitative coherence or reduces regressions
  without hiding reward degradation.

## Immediate Next Steps

1. Confirm the selected baseline job saved rollout JSON and action tensors:

```bash
SELECTED=/home/zcably0/Scratch/wam-rl/results_selected_eval/baseline_selected_seed1_50/JOBID
find "$SELECTED/rollouts" -type f -name "*.json" | wc -l
find "$SELECTED/rollouts" -type f -name "*.npy" | wc -l
find "$SELECTED/rollouts" -type f -name "*.json" | head
```

Expected count for the 13-task selected set with `TEST_NUM=50` is `650` JSON
records. If this count is zero, fix rollout logging before any RL work.

2. Run the PEFT tiny training smoke:

```bash
cd ~/Scratch/WAM-RL
git pull
qsub jobs/myriad/21_train_peft_tiny_4gpu.sh
tail -f "$(ls -t logs/jobs/wam_peft_tiny.o* | head -1)"
```

3. Implement Milestone 1:

- Extend rollout JSON schema.
- Preserve server-side action and latent paths.
- Add grouped rollout metadata.
- Add validation for complete groups.

4. Implement Milestone 2:

- Add `tools/build_grpo_groups.py`.
- Generate `grpo_groups.jsonl`.
- Verify mixed-group ratio and reward statistics.

5. Only after Milestone 1 and Milestone 2 pass, implement the first RL trainer:

```text
objective = reward_weighted_action_flow_matching
trainable_mode = action_heads
reward = binary success
group_size = 4 for debug, then 8
```
