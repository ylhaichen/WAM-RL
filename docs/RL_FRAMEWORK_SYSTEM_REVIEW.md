# RL Framework System Review

Date: 2026-05-30

## Scope

This review tracks the WAM-RL implementation against
`docs/WAM_RL_RESEARCH_IMPLEMENTATION_PLAN_FRAMEWORK_FINAL.md`.

The current priority is the native actor-replay path, while keeping the
native/offline path as the regression baseline:

1. collect grouped RoboTwin rollouts;
2. save strict denoising-step artifacts;
3. build dynamic-sampling GRPO groups;
4. validate artifacts and dataset references;
5. run an offline GRPO loss/trainer smoke path;
6. capture replay context for real LingBot-VA actor replay;
7. prepare storage-bounded actor-replay smoke runs.

## Current Status

### Phase 1-3: Rollout Collection And Offline Dataset

Implemented:

- `jobs/myriad/30_collect_grouped_rollouts_4gpu.sh`
- `jobs/myriad/30_collect_grouped_rollouts_1gpu.sh`
- `tools/collect_robotwin_rollouts.py`
- `tools/build_grpo_groups.py`
- `tools/validate_grpo_dataset.py`
- `wan_va/rl/trajectory_schema.py`
- `wan_va/rl/group_builder.py`
- `wan_va/rl/validation.py`
- `wan_va/rl/manifest.py`
- `wan_va/rl/dataset.py`
- strict artifact saving in `wan_va/wan_va_server.py`
- async-save flushing in `wan_va/utils/utils.py`

Important server-side fixes already in `main`:

- RoboTwin client CLI propagation includes `port`.
- Myriad job scripts derive `REPO_ROOT` from script location unless explicitly overridden.
- Group builder can fail on validation errors and require strict artifact files to exist.

### Phase 4: Denoising-Step GRPO Trainer

Implemented in this pass:

- `wan_va/rl/scheduler_logprob.py`
- `wan_va/rl/denoising_replay.py`
- `wan_va/rl/grpo_loss.py`
- `wan_va/rl/trainer.py`
- `tools/train_offline_grpo_smoke.py`
- `wan_va/configs/va_robotwin_grpo_train_cfg.py`
- `jobs/myriad/31_train_denoising_grpo_robotwin.sh`

The trainer added here is a strict-artifact smoke trainer. It validates:

- loading `groups/grpo_groups.jsonl`;
- flattening strict artifact references;
- Gaussian transition logprob replay from saved tensors;
- asymmetric clipped GRPO loss;
- optimizer step;
- finite metrics;
- checkpoint and metrics writes.

It intentionally does not claim to update LingBot-VA actor weights.

### Phase 5: Native Actor Replay And Iterative Loop

Implemented framework-neutral pieces:

- `wan_va/rl/iteration_controller.py`
- `wan_va/rl/checkpoint_gate.py`
- `wan_va/rl/evaluator.py`
- `wan_va/rl/reward.py`
- `jobs/myriad/40_rl_iteration_robotwin.sh`

These pieces define iteration paths, binary reward summaries, and promotion
rules. They are enough to keep future collect-train-eval-promote jobs from
hardcoding directory conventions and checkpoint acceptance rules.

`jobs/myriad/40_rl_iteration_robotwin.sh` currently runs a sequential
collect-then-smoke-train iteration. It is now a legacy offline fallback and
deliberately does not promote checkpoints. For real actor replay smoke work,
use `jobs/myriad/35_prepare_actor_replay_subset.sh` followed by
`jobs/myriad/36_submit_actor_replay_subset_smoke.sh`.

### Phase 6: Video-Action Consistency

Not implemented. The research plan explicitly says this should wait until
action-only GRPO is stable.

## Main Remaining Blocker

Real actor replay is implemented, but not yet validated as a reliable
improvement loop. The main blocker has moved from missing replay
instrumentation to scalable, repeatable replay-context storage and evaluation.

The older strict artifacts are enough for dataset validation and loss smoke
tests. Real actor replay additionally requires exact action denoising context
for each saved transition. Strict transition artifacts contain:

- `action_xt`
- `action_xt_next`
- `transition_mean`
- `transition_std`
- `old_logprob_sum`
- `logprob_mask`

Actor replay datasets now store the missing forward-pass context externally,
including:

- the observation/video/text conditioning state needed by the transformer;
- the cache state or a deterministic way to rebuild it;
- the exact CFG-conditioned action input structure;
- the policy worker adapter that maps saved rollout context to current
  `transition_mean_theta`.

Therefore the next real actor-training milestone is not another loss function.
It is a storage-bounded train/eval loop that can run repeatedly without filling
Scratch or relying on full 400GB+ replay-context directories.

## Recommended Next Milestones

1. Keep the next rollout collection storage-bounded. Before submitting, run
   the bounded replay-context wrapper in dry-run mode and check Scratch
   headroom:

   ```bash
   DRY_RUN=1 \
   TASK_NAMES=move_stapler_pad \
   GROUP_SIZE=4 \
   GROUPS_PER_TASK=1 \
   GROUP_MAX_ATTEMPTS=1 \
   STRICT_GRPO_CAPTURE_MAX_CHUNKS=1 \
   ACTION_NUM_INFERENCE_STEPS=10 \
   bash jobs/myriad/39_submit_grpo_replayctx_bounded_4gpu.sh
   ```

2. After any accepted replay-context collection, validate it with replay-context
   requirements enabled:

   ```bash
   python tools/validate_grpo_dataset.py \
       "$RESULTS/groups/grpo_groups.jsonl" \
       --inspect-artifacts \
       --require-replay-context \
       --out-summary "$RESULTS/groups/grpo_dataset_validation.json" \
       --fail-on-error
   ```

3. Inspect at least one replay-context file with compact metadata-only output
   before training:

   ```bash
   python tools/inspect_grpo_replay_context.py \
     /path/to/strict_grpo_replay_context_0.pt \
     --metadata-only \
     --print-summary
   ```

4. Prepare a storage-bounded actor replay subset:

   ```bash
   SUBSET_SOURCE_GROUPS="$RESULTS/groups/grpo_groups.jsonl" \
   SUBSET_ROOT="/home/zcably0/Scratch/wam-rl/results_grpo_actor_replay_subset/<run>" \
   SUBSET_MAX_REPLAY_CONTEXT_GB=30 \
   SUBSET_STORAGE_MAX_RESOLVED_GB=40 \
   bash jobs/myriad/35_prepare_actor_replay_subset.sh
   ```

5. Submit the actor replay subset smoke only after the storage audit passes and
   after reviewing the dry-run command:

   ```bash
   SUBSET_ROOT="/home/zcably0/Scratch/wam-rl/results_grpo_actor_replay_subset/<run>" \
   bash jobs/myriad/36_submit_actor_replay_subset_smoke.sh --dry-run
   ```

6. Inspect the training output with
   `tools/summarize_actor_replay_training.py`; require finite ratios and
   `parameter_update_detected=true` for nonzero-learning-rate smoke runs.
7. Only after actor replay produces finite ratios, non-zero gradients/nonzero
   parameter updates, and a reproducible paired eval signal on a bounded
   dataset, replace the offline fallback in
   `jobs/myriad/40_rl_iteration_robotwin.sh`.

## Review Conclusion

The repo now has a complete native/offline data contract, a tested Phase 4
loss/trainer smoke path, and an implemented real actor replay path. The
remaining high-risk component is turning tiny actor replay smoke runs into a
storage-aware, eval-aware improvement loop.
