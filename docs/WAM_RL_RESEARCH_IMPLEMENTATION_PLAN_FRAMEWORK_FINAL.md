# WAM-RL Research Implementation Plan

**Project:** SimpleVLA-style online GRPO for LingBot-VA / WAM-RL on RoboTwin  
**Model target:** [Robbyant/LingBot-VA](https://github.com/robbyant/lingbot-va)  
**Closest reference:** [PRIME-RL/SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL)  
**Primary environment:** RoboTwin / RoboTwin2.0  
**Primary policy formulation:** Denoising-step policy for FlowMatch action generation  
**Primary implementation stance:** primary native LingBot-VA/WAM-RL stack; veRL is reserved for later scale-up; offline grouped rollout training is the debug fallback.

> Current status note (2026-05-30): this is the original implementation plan,
> not the current execution record. Real LingBot-VA actor replay, external
> replay-context artifacts, bounded subset preparation, checkpoint loading, and
> eval plumbing now exist. Use `docs/WAM_RL_CURRENT_PROJECT_STATUS.md` for the
> current project state and `docs/WAM_RL_ACTOR_REPLAY_RUNBOOK.md` for active
> Myriad operations. `jobs/myriad/40_rl_iteration_robotwin.sh` is currently a
> legacy offline fallback, not the real actor replay loop.

---

## 1. Updated Core Direction

The previous plan treated the project broadly as outcome-based RL post-training for a World Action Model. After comparing VAMPO, SimpleVLA-RL, and LingBot-VA, the direction should be narrowed and reframed:

> The project is not primarily VAMPO-style RL post-training for a video world model. It is SimpleVLA-RL-style online GRPO for a FlowMatch-based Video-Action VLA model, using LingBot-VA as the base model.

The most important difference is the policy representation:

- SimpleVLA-RL uses OpenVLA / OpenVLA-OFT, where actions are represented as discrete action tokens. This naturally provides categorical action-token log probabilities.
- LingBot-VA generates continuous action chunks through FlowMatch / denoising. It does not directly expose categorical action-token log probabilities.
- Therefore WAM-RL must define the policy over action denoising transitions:

```text
pi_theta(x_{k-1} | x_k, obs, prompt, timestep)
```

rather than over discrete action tokens:

```text
pi_theta(action_token | obs, prompt)
```

This makes the central research contribution more precise:

> Extend SimpleVLA-RL-style grouped outcome RL from token-based VLA policies to FlowMatch-based continuous action VLA policies by formulating each action denoising transition as a stochastic policy step.

---

## 2. Detailed Summary of the Previous Discussion

### 2.1 What SimpleVLA-RL contributes

SimpleVLA-RL is the most relevant practical reference for this project. Its core idea is simple but effective:

1. Start from an SFT VLA model.
2. For each task context, sample multiple closed-loop robot trajectories.
3. Execute generated action chunks in the simulator.
4. Assign binary trajectory-level reward:

```text
success = 1
failure = 0
```

5. Compute group-relative advantages across multiple rollouts from the same initial context.
6. Use GRPO / PPO-style clipped policy optimization to update the VLA policy.

The critical insight is that VLA RL should be treated as interactive closed-loop trajectory optimization, not as single-shot text generation. Unlike LLM rollout, a VLA policy must repeatedly observe the environment, generate an action chunk, execute it, receive updated observations, and continue until success or timeout.

SimpleVLA-RL uses three practical exploration and stability tricks:

- **Dynamic Sampling:** discard groups where all rollouts succeed or all fail, because their group-relative advantage becomes zero.
- **Clip Higher:** use asymmetric clipping such as `[0.8, 1.28]` to allow more probability increase for useful low-probability actions.
- **Higher Rollout Temperature:** increase sampling temperature during rollout to improve action diversity.

These lessons transfer directly to WAM-RL, but their implementation differs because LingBot-VA uses continuous FlowMatch action generation rather than token sampling.

### 2.2 Why VAMPO is no longer the main template

VAMPO is useful conceptually because it formulates diffusion denoising as a sequential decision process. However, its target is mainly video dynamics improvement. It optimizes a video action model by rewarding predicted visual dynamics in latent space.

That is not the main goal here.

The current project aims to improve the robot policy through environment outcome feedback, as SimpleVLA-RL does. Therefore:

```text
VAMPO-style: improve visual dynamics through latent reward.
SimpleVLA-style: improve action policy through online task success reward.
WAM-RL target: improve LingBot-VA action generation through online RoboTwin success reward, while optionally preserving video-action consistency later.
```

VAMPO remains useful for one technical idea: a denoising trajectory can be treated as a policy trajectory if we can compute or approximate transition log probabilities.

### 2.3 What changes for LingBot-VA

LingBot-VA has a different architecture from OpenVLA-OFT:

- It is an autoregressive Video-Action model.
- It generates imagined video latents and action chunks.
- It uses a shared transformer with action-specific modules such as `action_embedder`, `condition_embedder_action`, and `action_proj_out`.
- Its inference loop has a video denoising loop and an action denoising loop.
- The action generation loop uses FlowMatch-style denoising and `FlowMatchScheduler.step()`.

The current `FlowMatchScheduler.step()` is deterministic. To apply strict GRPO, we need to modify it into a stochastic policy transition:

```text
x_{k-1} = mean_theta(x_k, obs, prompt, t) + sigma_rl(t) * epsilon
```

Then compute:

```text
log p_theta(x_{k-1} | x_k, obs, prompt, t)
```

This is the chosen policy formulation for the updated project.

---

## 3. Updated Research Question

Can SimpleVLA-RL-style grouped online outcome RL improve LingBot-VA on RoboTwin by optimizing its FlowMatch action denoising policy with binary task-success rewards, while preserving executable action quality and avoiding collapse of video-action coherence?

More specific sub-questions:

1. Can Denoising-step GRPO improve LingBot-VA over its SFT / post-trained baseline on RoboTwin tasks?
2. Does binary outcome reward alone provide enough signal for FlowMatch action generation?
3. Is action-denoising stochasticity a sufficient replacement for token-level temperature sampling?
4. Which update surface is stable: action heads, last-N transformer blocks, LoRA/adapters, or full fine-tuning?
5. Does adding video-action consistency later reduce reward hacking or long-horizon drift?
6. Can the system discover new action strategies beyond SFT, similar to SimpleVLA-RL's “pushcut” phenomenon?

---

## 4. Fixed Design Decisions

These are now fixed unless experiments show a hard failure.

### 4.1 Primary benchmark: RoboTwin first

The first main benchmark should be RoboTwin / RoboTwin2.0, not LIBERO.

Reasons:

- The previous implementation work and infrastructure already focus on LingBot-VA + RoboTwin.
- The project goal is closer to dual-arm long-horizon manipulation.
- SimpleVLA-RL reports substantial gains on RoboTwin2.0, making it a strong comparison target.
- RoboTwin better exposes the value of action-chunk RL, recovery behavior, and long-horizon exploration.

LIBERO can remain a later sanity benchmark or secondary generalization benchmark, but it should not drive the first implementation.

### 4.2 Primary reward: binary outcome reward

Use the SimpleVLA-RL reward first:

```text
R(tau) = 1 if the task succeeds, else 0
```

Do not start with dense shaping. Dense reward can be added only after the binary reward pipeline is stable.

### 4.3 Primary policy: Denoising-step policy

Use **方案 1: Denoising-step policy**.

Each action denoising transition is treated as one policy step:

```text
state_k  = (noisy_action_state x_k, visual observation context, language prompt, timestep k)
action_k = next denoised action state x_{k-1}
```

Policy distribution:

```text
pi_theta(x_{k-1} | x_k, obs, prompt, t_k)
```

The trajectory probability is approximated by summing transition log probabilities:

```text
log pi_theta(action_chunk | obs, prompt)
  ~= sum_k log pi_theta(x_{k-1} | x_k, obs, prompt, t_k)
```

This is more faithful than final-action Gaussian policy and more aligned with FlowMatch generation than action-token discretization.

### 4.4 Rollout worker / RL framework final choice

The rollout worker and RL framework decision is now fixed as a staged design.

```text
Primary framework:
Native LingBot-VA/WAM-RL rollout system with Ray Core or Python multiprocessing,
plus a custom PyTorch/FSDP GRPO trainer.

Backup / scale-up framework:
veRL adapter after the denoising-step policy and grouped RoboTwin rollout are validated.

Debug fallback:
Offline grouped rollout collector plus offline GRPO trainer.
```

The main reason is framework-model mismatch. SimpleVLA-RL extends veRL for token-policy VLA models, where the actor can compute categorical action-token log probabilities. LingBot-VA instead generates continuous action chunks through FlowMatch denoising, so the main algorithmic object is not an action token but a denoising transition:

```text
pi_theta(x_{k-1} | x_k, obs, prompt, t_k)
```

Therefore, the first version must directly instrument LingBot-VA's action denoising loop, save denoising-step trajectories, compute FlowMatch transition log probabilities, and replay those transitions during actor update. This is easier to validate in a native implementation than by immediately adapting veRL's token-response abstraction.

Framework selection:

| Component | Selected choice | Role |
|---|---|---|
| Rollout orchestration | Ray Core or Python multiprocessing | Parallel RoboTwin grouped rollout |
| Policy worker | Modified LingBot-VA / WAM-RL server-client path | Action chunk generation and denoising trajectory logging |
| Trainer | Custom PyTorch/FSDP GRPO trainer | Denoising-step logprob replay and clipped GRPO update |
| Reward manager | Native binary outcome reward module | `success -> 1`, `failure -> 0` |
| Group builder | Native dynamic sampling + advantage builder | Keep mixed groups and compute GRPO advantage |
| Backup framework | veRL adapter | Large-scale distributed actor/rollout/ref worker after native validation |
| Debug fallback | Offline grouped rollout collector + offline GRPO trainer | Reproducible algorithm debugging |

Rejected as first-stage frameworks:

| Framework | Reason not selected as primary |
|---|---|
| veRL | Powerful but assumes token-response style data flow in SimpleVLA-RL; migrate only after FlowMatch logprob works |
| RLlib | Too generic and does not naturally match large-model VLA server-client rollout plus denoising trajectory replay |
| TorchRL | Useful for dataset ideas but not sufficient for LingBot-VA FSDP + RoboTwin rollout + FlowMatch GRPO |
| CleanRL | Useful for PPO reference code, not a scalable VLA/RoboTwin framework |

The rollout worker interface should remain framework-neutral so that the same saved trajectory schema can later be wrapped into a veRL `DataProto` if scale becomes the bottleneck.

---

## 5. Updated System Architecture

### 5.1 Three-stage framework roadmap

The system will be implemented in three stages.

#### Stage 1: Native + Offline

This is the first implementation target.

```text
1. collect grouped RoboTwin rollouts
2. save denoising-step trajectories
3. compute binary success reward
4. build GRPO groups
5. train offline GRPO
6. evaluate selected RoboTwin tasks
```

Purpose:

- validate denoising-step policy formulation;
- verify FlowMatch transition logprob;
- inspect reward, advantage, ratio, and gradient statistics;
- avoid online actor-learner complexity before the algorithm works.

This stage produces frozen rollout datasets. Each rollout batch is collected by an old actor checkpoint and then used for one or more controlled offline GRPO updates.

#### Stage 2: Native + Iterative Online

Once Stage 1 is stable, move to an iterative online loop.

```text
for iteration t:
  1. freeze actor_t as old_policy
  2. actor_t collects grouped RoboTwin rollouts
  3. build dynamic-sampling GRPO groups
  4. train actor_{t+1}
  5. evaluate actor_{t+1}
  6. promote checkpoint only if evaluation gates pass
  7. repeat
```

This is the real SimpleVLA-RL-style training loop for LingBot-VA, but still implemented natively.

#### Stage 3: veRL scale-up

Only after the native Denoising-step GRPO loop is validated, consider a veRL migration.

```text
replace native controller with veRL/RayTrainer-style architecture
keep custom LingBot rollout worker
keep custom FlowMatch logprob actor
scale to more tasks, more contexts, and more GPUs
```

The veRL migration should be a scaling step, not the first research step.

### 5.2 Main online RL loop

The final online version should follow:

```text
for iteration t:
  1. Freeze actor theta_t as old_policy.
  2. Sample RoboTwin tasks, seeds, prompts, and initial states.
  3. For each context, run K stochastic LingBot-VA rollouts.
  4. Save denoising-step action trajectories, generated actions, observations, success flags, and metadata.
  5. Filter groups using Dynamic Sampling: keep only groups with mixed success/failure.
  6. Compute group-relative advantages.
  7. Recompute logprob under current actor on stored denoising transitions.
  8. Apply clipped Denoising-step GRPO update.
  9. Evaluate on fixed RoboTwin selected tasks and heldout seeds.
  10. Promote checkpoint only if it improves without severe regression.
```

### 5.3 Module layout

Recommended project structure:

```text
wan_va/rl/
  __init__.py
  scheduler_logprob.py          # stochastic FlowMatch transition + logprob
  trajectory_schema.py          # typed schemas for rollout and group records
  rollout_worker.py             # RoboTwin rollout with LingBot-VA server/client
  policy_worker.py              # LingBot-VA action sampling with denoising logs
  reward.py                     # binary outcome reward manager
  group_builder.py              # dynamic sampling + GRPO advantage
  denoising_replay.py           # recompute logprob from saved transitions
  grpo_loss.py                  # clipped denoising-step GRPO objective
  trainer.py                    # native PyTorch/FSDP trainer
  evaluator.py                  # fixed RoboTwin evaluation gate

tools/
  collect_robotwin_grpo_rollouts.py
  build_grpo_groups.py
  validate_denoising_rollouts.py
  summarize_rl_iterations.py

jobs/myriad/
  30_collect_grpo_rollouts_robotwin.sh
  31_train_denoising_grpo_robotwin.sh
  32_eval_grpo_checkpoint_robotwin.sh
  40_rl_iteration_robotwin.sh
```

### 5.4 Actor / old policy / reference policy

Use three model roles:

```text
actor_model      = model being updated
old_policy       = frozen snapshot used to collect the current rollout batch
reference_model  = original LingBot-VA SFT/post-train checkpoint
```

Stage 1 can use actor and old_policy only. Add reference regularization if instability appears:

```text
loss =
  L_denoising_grpo(actor, old_policy)
  + beta_ref * L_reference_action(actor, reference_model)
  + beta_bc * L_demonstration_anchor
```

Keep `beta_ref = 0` and `beta_bc = 0` for the first logprob smoke test, then enable them if action quality collapses.

---

## 6. RoboTwin Task Policy

The selected task set should remain fixed before RL tuning.

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

Task roles:

```text
hard:      baseline success roughly 10%-50%
medium:    baseline success roughly 50%-90%
easy:      baseline success >=90%, sanity/regression tasks
too_hard:  baseline success <10%, diagnostics only unless progress reward is later added
```

The first online GRPO target should focus on `hard + medium` tasks. Easy tasks should be included in evaluation but not overrepresented in training, because all-success groups produce zero GRPO signal.

For high-compute future setting:

```text
train_tasks:        30-60 RoboTwin tasks with baseline success in 10%-90%
easy_sanity_tasks:  5-10 high-success tasks
heldout_tasks:      10-20 tasks never used in RL rollout collection
heldout_seeds:      seeds never used during training rollout
```

---

## 7. Denoising-Step Policy Formulation

### 7.1 FlowMatch transition as policy step

Current deterministic FlowMatch step:

```text
x_{k-1} = x_k + v_theta(x_k, t_k, c) * (sigma_{k-1} - sigma_k)
```

Convert it into stochastic policy transition:

```text
mu_theta = x_k + v_theta(x_k, t_k, c) * delta_sigma
x_{k-1} = mu_theta + eta(t_k) * epsilon
```

where:

```text
c = visual observation context + language prompt + KV cache context
eta(t_k) = action denoising exploration scale
epsilon ~ N(0, I)
```

Log probability:

```text
log p_theta(x_{k-1} | x_k, c, t_k)
  = -0.5 * ||epsilon||^2 - log eta(t_k) - 0.5 * log(2pi)
```

The action chunk log probability is:

```text
log p_theta(action_chunk | obs, prompt)
  = sum_k log p_theta(x_{k-1} | x_k, obs, prompt, t_k)
```

### 7.2 What must be saved during rollout

Strict Denoising-step GRPO requires saving the full action denoising trajectory, not only the final executed action.

Each action chunk record must include:

```json
{
  "chunk_id": 0,
  "frame_st_id": 0,
  "action_timesteps_path": ".../action_timesteps_000.pt",
  "action_states_path": ".../action_states_000.pt",
  "action_next_states_path": ".../action_next_states_000.pt",
  "old_logprobs_path": ".../old_action_logprobs_000.pt",
  "action_std_path": ".../action_std_000.pt",
  "final_normalized_action_path": ".../actions_000.pt",
  "executed_action_path": ".../executed_actions_000.npy"
}
```

Top-level rollout JSON should include:

```json
{
  "run_id": "rl_iter_0000",
  "policy_checkpoint": "/path/to/actor_theta_t",
  "old_policy_checkpoint": "/path/to/actor_theta_t",
  "reference_checkpoint": "/path/to/lingbot_va_base_or_posttrain",
  "task_name": "open_microwave",
  "group_id": "open_microwave_seed10042_prompt0",
  "sample_idx": 3,
  "group_size": 8,
  "env_seed": 10042,
  "sampling_seed": 730003,
  "prompt": "...",
  "success": true,
  "reward": 1.0,
  "finish_step": 112,
  "step_lim": 200,
  "video_guidance_scale": 5.0,
  "action_guidance_scale": 1.0,
  "action_num_inference_steps": 50,
  "action_rl_eta": 0.05,
  "chunks": [
    {
      "chunk_id": 0,
      "action_states_path": "...",
      "action_next_states_path": "...",
      "old_logprobs_path": "..."
    }
  ],
  "initial_obs_path": "...",
  "visualization_path": "..."
}
```

### 7.3 Recompute logprob during update

During training, do not backprop through rollout. Instead:

1. Load saved action denoising states.
2. Feed the same context and same `x_k` into current actor.
3. Recompute `mu_theta` and `log p_theta(x_{k-1}|x_k,c,t)`.
4. Compare against saved old logprob:

```text
rho = exp(logp_new - logp_old)
```

5. Apply clipped GRPO loss.

---

## 8. GRPO Objective

### 8.1 Binary outcome reward

Trajectory reward:

```text
R_i = 1 if rollout i succeeds else 0
```

### 8.2 Dynamic Sampling

For each context group with `K` rollouts, keep only mixed groups:

```text
0 < number_of_successes < K
```

Skip all-success and all-failure groups in the first implementation.

### 8.3 Group-relative advantage

```text
A_i = (R_i - mean(R_group)) / (std(R_group) + eps)
```

Initial settings:

```text
K = 4 for debug
K = 8 for first serious run
advantage_clip = [-2, 2]
eps = 1e-6
```

### 8.4 Denoising-step clipped objective

For each saved denoising transition:

```text
ratio_{i,k} = exp(logp_theta(i,k) - logp_old(i,k))
```

Loss:

```text
L_grpo = -mean_k min(
  ratio_{i,k} * A_i,
  clip(ratio_{i,k}, 1 - eps_low, 1 + eps_high) * A_i
)
```

Initial clipping:

```text
eps_low = 0.2
eps_high = 0.28
```

This follows SimpleVLA-RL's clip-higher intuition but applies it to FlowMatch denoising transitions rather than action tokens.

### 8.5 Optional regularization

Start with:

```text
L = L_grpo
```

If unstable, add:

```text
L = L_grpo
  + beta_bc * L_action_flowmatch_BC
  + beta_ref * L_reference_action
  + beta_entropy * L_entropy_or_std_control
```

Recommended initial values:

```text
beta_bc = 0.01 to 0.1
beta_ref = 0.0 to 0.01
beta_entropy = 0.0 initially
```

Do not add video-action consistency until action-only GRPO shows stable improvement.

---

## 9. Rollout Worker / RL Framework Design

### 9.1 Final framework selection

The selected framework strategy is:

```text
Primary framework:
Native LingBot-VA/WAM-RL rollout system with Ray Core or Python multiprocessing,
plus a custom PyTorch/FSDP GRPO trainer.

Backup / scale-up framework:
veRL adapter after the denoising-step policy and grouped RoboTwin rollout are validated.

Debug fallback:
Offline grouped rollout collector plus offline GRPO trainer.
```

This section defines how the three options will be used.

### 9.2 Primary rollout worker: native LingBot-VA/WAM-RL

The primary rollout worker should be implemented natively inside WAM-RL / LingBot-VA:

```text
RolloutWorker = RoboTwin env manager + LingBot-VA policy worker + denoising trajectory logger
```

Responsibilities:

1. Start or connect to a LingBot-VA policy worker.
2. Create a RoboTwin / RoboTwin2.0 task environment.
3. Reset the environment with a fixed task seed.
4. Send observation and language prompt to the policy worker.
5. Sample an action chunk using stochastic action denoising.
6. Save action denoising trajectory tensors:
   - noisy action state `x_k`;
   - next denoised action state `x_{k-1}`;
   - timestep `t_k`;
   - old transition logprob;
   - scheduler standard deviation / exploration scale;
   - final normalized action chunk.
7. Execute the postprocessed action chunk in RoboTwin.
8. Send executed action and new observation back to LingBot-VA for KV-cache update.
9. Repeat until success or timeout.
10. Save rollout JSON and tensor paths.
11. Return success/failure, finish step, and metadata.

### 9.3 Stage 1: Native + Offline

The first working version should not do streaming actor-learner updates. It should use frozen actor checkpoints to collect reproducible rollout datasets.

```text
collect grouped RoboTwin rollouts
save denoising-step trajectories
compute binary success reward
build GRPO groups
train offline GRPO
evaluate selected RoboTwin tasks
```

Recommended debug settings:

```text
tasks = 1-2 RoboTwin tasks
contexts_per_task = 2-4
K = 4
action_rl_eta = 0.02-0.05
video generation = deterministic or fixed seed
action denoising = stochastic
```

Serious first setting:

```text
tasks = selected RoboTwin hard/medium tasks
contexts_per_task = 8-16
K = 8
dynamic_sampling = keep mixed success/failure groups only
```

Why this stage matters:

- It makes rollout data inspectable.
- It allows exact replay of old denoising transitions.
- It separates algorithm debugging from online infrastructure debugging.
- It avoids corrupting the actor before logprob and ratio statistics are validated.

### 9.4 Stage 2: Native + Iterative Online

After offline GRPO works, move to a native online iteration:

```text
actor_t collect rollouts
train actor_{t+1}
evaluate
promote checkpoint
repeat
```

Each iteration should create:

```text
runs/rl_iter_0000/
  actor_old/
  rollouts/
  groups/
  train/
  checkpoints/
  eval_selected/
  eval_heldout/
  reports/
```

Promotion rules:

- selected-task aggregate success improves or remains stable with clear hard/medium gains;
- easy sanity tasks do not collapse;
- ratio, KL-like drift, action magnitude, and action saturation remain within sane ranges;
- qualitative videos do not show obvious reward hacking.

### 9.5 Stage 3: veRL scale-up

veRL should be used only after the native implementation proves the method.

Migration target:

```text
verl/workers/rollout/lingbot_rollout.py
verl/workers/actor/dp_lingbot_flow.py
verl/utils/vla_utils/lingbot/
```

The custom veRL actor must support FlowMatch denoising-step replay, not categorical action-token logprob.

Expected veRL data fields:

```text
batch keys:
  action_states
  action_next_states
  action_timesteps
  old_logprobs
  rewards
  advantages
  finish_step
  obs_context_paths
  prompt
  task_id
  group_id
  sample_idx
```

Do not migrate to veRL until these native components are validated:

- stochastic `step_with_logprob()`;
- grouped RoboTwin rollout;
- binary reward and dynamic sampling;
- denoising trajectory replay;
- clipped GRPO loss;
- stable selected-task evaluation.

### 9.6 Parallelism plan

Initial parallelism:

```text
Debug:
  1 LingBot-VA policy worker
  1 RoboTwin env worker

Small:
  1 LingBot-VA policy worker
  2-4 RoboTwin env workers

Serious:
  4 LingBot-VA policy workers
  8-16 RoboTwin env workers

High scale:
  Ray Core worker pool or veRL-style worker groups
```

The policy worker should be stateful because LingBot-VA uses KV cache. Environment workers should be restartable because RoboTwin/SAPIEN can fail or leak memory during long rollout collection.

---

## 10. Training Surface

Start conservatively.

### 10.1 Stage A: action-specific modules only

Train:

```text
action_embedder
condition_embedder_action
action_proj_out
```

Freeze:

```text
video branch
most shared transformer blocks
VAE
text encoder
```

This is the safest first setting.

### 10.2 Stage B: action heads + last-N transformer blocks

If Stage A underfits, unfreeze:

```text
last 2-4 transformer blocks
```

### 10.3 Stage C: LoRA/adapters

If full action-head updates are insufficient, add LoRA/adapters to:

```text
to_q, to_k, to_v, to_out
FFN projections
possibly action-specific condition pathways
```

### 10.4 Stage D: full fine-tuning

Only for high-compute ablation. Not recommended as the first method.

---

## 11. Implementation Phases

The implementation phases now follow the selected framework roadmap.

### Phase 1: RoboTwin baseline and task selection

Goal: fixed, reproducible RoboTwin baselines before RL.

Tasks:

- Run selected-task baseline.
- Save per-task success rates.
- Save representative videos.
- Verify rollout JSON and action logging.
- Fix task list before method tuning.

Exit criteria:

- Selected tasks have baseline success estimates.
- Hard/medium/easy categories are assigned.
- Easy sanity tasks are fixed.

### Phase 2: Stochastic action denoising instrumentation

Goal: make LingBot-VA action generation into a policy with transition logprob.

Implement:

```text
wan_va/rl/scheduler_logprob.py
```

Add:

- `step_with_logprob()` for FlowMatch action scheduler.
- stochastic denoising noise controlled by `action_rl_eta`.
- transition logprob computation.
- deterministic mode for evaluation.
- logging of `action_states`, `action_next_states`, `action_timesteps`, and `old_logprobs`.

Exit criteria:

- Same seed produces reproducible denoising trajectory.
- Different sampling seeds produce diverse action chunks.
- Logprob tensors have expected shape and finite values.

### Phase 3: Stage 1 framework implementation: Native + Offline

Goal: implement the debug fallback as the first real system.

Implement:

```text
tools/collect_robotwin_grpo_rollouts.py
tools/build_grpo_groups.py
wan_va/rl/trajectory_schema.py
wan_va/rl/rollout_worker.py
wan_va/rl/group_builder.py
wan_va/rl/dataset.py
```

Pipeline:

```text
collect grouped RoboTwin rollouts
save denoising-step trajectories
compute binary success reward
build GRPO groups
train offline GRPO
evaluate selected RoboTwin tasks
```

Required rollout outputs:

- rollout JSON per sampled trajectory.
- denoising trajectory tensors per action chunk.
- final normalized action tensors.
- executed env action tensors.
- success/failure flag.
- finish step.
- task/prompt/seed/group metadata.

Exit criteria:

- `K=4` debug groups are complete.
- Groups contain correct `group_id` and `sample_idx`.
- All tensor paths exist.
- Mixed-group ratio is reported.
- Offline dataloader can replay denoising transitions.

### Phase 4: Denoising-step GRPO trainer

Goal: train LingBot-VA action policy using strict denoising transition logprob.

Implement:

```text
wan_va/rl/denoising_replay.py
wan_va/rl/grpo_loss.py
wan_va/rl/trainer.py
wan_va/configs/va_robotwin_grpo_train_cfg.py
jobs/myriad/31_train_denoising_grpo_robotwin.sh
```

Core training logic:

1. Load saved denoising transitions.
2. Recompute current logprob under actor.
3. Use saved old logprob from rollout actor.
4. Compute ratio.
5. Apply clipped GRPO objective.
6. Backprop through selected actor modules.

Exit criteria:

- Tiny trainer runs on a small saved rollout dataset.
- Ratio distribution is finite and not exploding.
- Gradients are non-zero for selected action modules.
- Checkpoint save/reload works.

### Phase 5: Stage 2 framework implementation: Native + Iterative Online

Goal: run the full collect-train-evaluate loop without veRL.

Implement:

```text
jobs/myriad/40_rl_iteration_robotwin.sh
wan_va/rl/iteration_controller.py
wan_va/rl/checkpoint_gate.py
```

Loop:

```text
actor_t collect rollouts
train actor_{t+1}
evaluate
promote checkpoint
repeat
```

Directory structure:

```text
runs/rl_iter_0000/
  actor_old/
  rollouts/
  groups/
  train/
  checkpoints/
  eval_selected/
  eval_heldout/
  reports/
```

Exit criteria:

- One full iteration can collect rollouts, train, evaluate, and decide promote/reject.
- Easy sanity tasks do not collapse.
- At least one hard/medium task improves.

### Phase 6: Video-action consistency ablation

Only after action-only GRPO is stable.

Add:

```text
r_or_loss_consistency = distance(imagined_future_latents, observed_future_latents)
```

or supervised auxiliary loss:

```text
L_video_consistency = ||z_imagined_next - z_observed_next||
```

Compare:

```text
Action-only GRPO
Action GRPO + video-action consistency
```

Exit criteria:

- Consistency improves qualitative coherence or reduces regressions.
- It does not hide reward degradation.

### Phase 7: Stage 3 framework implementation: veRL scale-up

Only after native Denoising-step GRPO works and scale becomes the bottleneck.

Migration target:

```text
verl/workers/rollout/lingbot_rollout.py
verl/workers/actor/dp_lingbot_flow.py
verl/utils/vla_utils/lingbot/
```

The veRL port should keep:

- custom LingBot rollout worker;
- custom FlowMatch logprob actor;
- denoising trajectory replay;
- RoboTwin grouped rollout and binary reward;
- dynamic sampling.

Exit criteria:

- veRL version matches native results on a small controlled setting.
- Scaling improves rollout throughput or training throughput.
- No change in the mathematical policy definition.

---

## 12. Evaluation Protocol

Every meaningful checkpoint must be evaluated on:

- selected RoboTwin train tasks;
- heldout seeds for selected tasks;
- heldout RoboTwin tasks;
- easy sanity tasks;
- qualitative videos;
- action statistics.

Primary metrics:

```text
per-task success rate
aggregate success rate
Wilson confidence interval
delta over LingBot-VA baseline
delta over PEFT-SFT baseline
number of improved tasks
number of regressed tasks
mixed-group ratio
episode length
early success rate
action magnitude
action saturation ratio
invalid action statistics
```

Minimum reporting table:

```text
method
checkpoint
RL framework / rollout backend
policy formulation
trainable_mode
rollout_K
action_rl_eta
clip_low
clip_high
selected_success
hard_success
medium_success
easy_sanity_success
heldout_success
num_improved_tasks
num_regressed_tasks
```

---

## 13. Ablation Matrix

### 13.1 Method ablations

```text
LingBot-VA baseline
PEFT-SFT action_heads
Denoising-step GRPO, action_heads
Denoising-step GRPO, action_heads + last-N blocks
Denoising-step GRPO + BC anchor
Denoising-step GRPO + reference regularization
Denoising-step GRPO + video-action consistency
```

### 13.2 Exploration ablations

```text
K = 4, 8, 16
action_rl_eta = 0.01, 0.03, 0.05, 0.08
action_guidance_scale = 1, 2, 3
video_guidance_scale fixed vs varied
fixed sampling vs dynamic sampling
```

### 13.3 Clipping ablations

```text
symmetric clip: [0.8, 1.2]
clip higher:    [0.8, 1.28]
more aggressive:[0.8, 1.35]
```

### 13.4 Update-surface ablations

```text
action_heads
last 2 transformer blocks
last 4 transformer blocks
LoRA on attention
LoRA on attention + FFN
full fine-tuning
```

### 13.5 Reward ablations

```text
binary success only
binary success + BC anchor
binary success + action smoothness penalty
binary success + video-action consistency
binary success + progress reward, later only
```

---

## 14. Risk Analysis

### Risk 1: stochastic FlowMatch destabilizes actions

Mitigation:

- start with small `action_rl_eta`;
- keep deterministic evaluation;
- log action magnitude and saturation;
- optionally add BC anchor.

### Risk 2: all groups are all-failure or all-success

Mitigation:

- select tasks with baseline success in `10%-90%`;
- use Dynamic Sampling;
- increase `K`;
- increase action stochasticity gradually.

### Risk 3: ratio explosion

Mitigation:

- clip log-ratio;
- use asymmetric PPO clipping;
- advantage clip;
- gradient clip;
- small learning rate.

### Risk 4: action-only RL breaks video-action coherence

Mitigation:

- freeze video branch first;
- evaluate imagined video qualitatively;
- add video-action consistency only after reward-only improvement.

### Risk 5: native stack does not scale

Mitigation:

- keep rollout data schema framework-neutral;
- use Ray Core for rollout scaling;
- migrate to veRL later if necessary.

---

## 15. Immediate Next Steps

### Step 1: confirm RoboTwin baseline and rollout logging

```bash
SELECTED=/home/zcably0/Scratch/wam-rl/results_selected_eval/baseline_selected_seed1_50/JOBID
find "$SELECTED/rollouts" -type f -name "*.json" | wc -l
find "$SELECTED/rollouts" -type f -name "*.npy" | wc -l
find "$SELECTED/rollouts" -type f -name "*.json" | head
```

Expected for 13 tasks with `TEST_NUM=50`:

```text
650 rollout JSON records
```

If this count is zero, fix rollout logging before any RL work.

### Step 2: implement stochastic FlowMatch scheduler

Implement:

```text
wan_va/rl/scheduler_logprob.py
```

Add unit tests for:

```text
finite logprob
shape correctness
seed reproducibility
stochastic diversity
deterministic eval mode
```

### Step 3: implement grouped RoboTwin rollout worker

Implement:

```text
tools/collect_robotwin_grpo_rollouts.py
wan_va/rl/rollout_worker.py
```

First debug setting:

```text
selected_tasks = 1-2 tasks
K = 4
action_rl_eta = 0.03
num_contexts_per_task = 2
```

### Step 4: build dynamic sampling group builder

Implement:

```text
tools/build_grpo_groups.py
```

Check:

```text
mixed group ratio
per-task balance
advantage distribution
missing tensor paths
```

### Step 5: train tiny Denoising-step GRPO

Implement:

```text
wan_va/rl/trainer.py
jobs/myriad/31_train_denoising_grpo_robotwin.sh
```

Debug setting:

```text
trainable_mode = action_heads
K = 4
batch_size = small
epochs = 1
lr = 1e-6
grad_clip = 0.5
```

### Step 6: evaluate and gate

Run selected task evaluation. Promote checkpoint only if:

```text
no easy sanity collapse
at least one hard/medium task improves
aggregate success does not severely regress
action statistics remain sane
```

---

## 16. Updated Claim Boundaries

The project can safely claim:

```text
We extend SimpleVLA-style online outcome RL to a FlowMatch-based Video-Action VLA by formulating action denoising transitions as stochastic policy steps and optimizing them with grouped binary task-success rewards.
```

Do not claim:

```text
We simply apply SimpleVLA-RL to LingBot-VA.
```

because LingBot-VA requires a different policy likelihood construction.

Do not claim:

```text
We improve world-model visual dynamics through RL.
```

unless video-action consistency or video latent reward is actually implemented and evaluated.

Do not claim strict GRPO until:

```text
old_logprob and new_logprob are computed over saved denoising transitions,
ratio is computed from these logprobs,
and the clipped objective is applied to those ratios.
```

---

## 17. Short Project Pitch

WAM-RL studies how to apply SimpleVLA-style online GRPO to LingBot-VA, a FlowMatch-based Video-Action model. Unlike token-based VLA models such as OpenVLA-OFT, LingBot-VA generates continuous action chunks through denoising, so action-token log probabilities are unavailable. We therefore formulate each action denoising transition as a stochastic policy step, collect grouped RoboTwin rollouts under binary success rewards, compute group-relative advantages, and optimize the action denoising policy with a clipped GRPO objective. The first implementation targets RoboTwin using action-only Denoising-step GRPO, with video-action consistency introduced later as an auxiliary constraint.
