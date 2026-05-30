# GRPO Strict Artifact Schema

This document defines the saved denoising-transition artifact contract used by
grouped RoboTwin rollout collection and offline GRPO replay.

## Goals

- Preserve compatibility with existing v1 artifacts.
- Capture full action denoising trajectories for new rollout jobs.
- Keep one artifact file per action chunk to avoid excessive small-file I/O on
  Myriad.
- Make validation and replay count true denoising transitions, not just artifact
  paths.
- Optionally capture enough actor replay context to recompute current LingBot-VA
  transition log probabilities for real actor updates.
- Keep storage decisions explicit. See
  `docs/WAM_RL_MYRIAD_STORAGE_POLICY.md` before scaling replay-context runs or
  archiving/deleting rollout artifacts.

## Rollout Metadata Contract

Each rollout JSON record stores:

```json
{
  "strict_grpo_ready": true,
  "strict_grpo_scope": "action_denoising_trajectory",
  "strict_grpo_artifact_count": 3,
  "strict_grpo_artifact_paths": [
    ".../strict_grpo_0.pt",
    ".../strict_grpo_8.pt",
    ".../strict_grpo_16.pt"
  ],
  "strict_grpo_replay_context_count": 3,
  "strict_grpo_replay_context_paths": [
    ".../strict_grpo_replay_context_0.pt",
    ".../strict_grpo_replay_context_8.pt",
    ".../strict_grpo_replay_context_16.pt"
  ],
  "strict_grpo_replay_context_tensor_bytes": [3618508800, 3618508800, 3618508800],
  "strict_grpo_replay_context_total_tensor_bytes": 10855526400,
  "strict_grpo_replay_context_max_gb": 5.0,
  "strict_grpo_capture_chunk_indices": [0, 1, 2],
  "strict_grpo_capture_chunk_stride": 1,
  "strict_grpo_capture_max_chunks": 3
}
```

`strict_grpo_artifact_paths` remains the trainer-facing reference list. A path
can point to either a v1 single-transition artifact or a v2 trajectory artifact.
The replay-context fields are optional but should be present for new actor-replay
collection. They make it possible to audit whether bounded capture settings
actually took effect without loading every multi-GiB context file.

## v1 Artifact

v1 artifacts are kept for backward compatibility:

```python
{
    "schema_version": 1,
    "scope": "first_action_denoising_step",
    "sampling_seed": int | None,
    "frame_st_id": int,
    "timestep": Tensor,
    "action_xt": Tensor,
    "action_xt_next": Tensor,
    "transition_mean": Tensor,
    "transition_std": Tensor,
    "old_logprob_sum": Tensor,
    "old_logprob_mean": Tensor,
    "old_logprob_count": Tensor,
    "logprob_mask": Tensor,
}
```

Each v1 artifact represents one denoising transition.

## v2 Artifact

New collection jobs default to v2:

```python
{
    "schema_version": 2,
    "scope": "action_denoising_trajectory",
    "sampling_seed": int | None,
    "frame_st_id": int,
    "num_transitions": int,
    "transitions": [
        {
            "denoising_step_index": int,
            "timestep": Tensor,
            "action_xt": Tensor,
            "action_xt_next": Tensor,
            "transition_mean": Tensor,
            "transition_std": Tensor,
            "old_logprob_sum": Tensor,
            "old_logprob_mean": Tensor,
            "old_logprob_count": Tensor,
            "logprob_mask": Tensor,
            # Present only when STRICT_GRPO_SAVE_REPLAY_CONTEXT=true:
            "replay_input": {
                "timesteps": Tensor,
                # Older artifacts may also contain noisy_latents. New artifacts
                # omit it because it is exactly the same denoising state already
                # stored as transition action_xt.
                "noisy_latents": Tensor,  # optional legacy field
            },
        }
    ],
    # Present only when STRICT_GRPO_SAVE_REPLAY_CONTEXT=true. New artifacts
    # prefer replay_context_path to keep large KV-cache payloads out of the
    # strict transition artifact. Older artifacts may still contain an inline
    # replay_context dict with the same fields.
    "replay_context_path": "strict_grpo_replay_context_0.pt",
}
```

Each v2 artifact represents one action chunk and stores every captured
non-terminal action denoising transition for that chunk.

The referenced replay-context file is a `torch.save` dict:

```python
{
    "schema_version": 1,
    "cache_name": str,
    "transformer_cache": list[dict[str, Tensor]],
    "grid_id": Tensor,
    "text_emb": Tensor,
    "negative_text_emb": Tensor | None,
    "use_cfg": bool,
    "cfg_pruned_to_conditional": bool,  # optional; true when action CFG is unused
    "action_guidance_scale": float,
    "action_num_inference_steps": int,
    "frame_chunk_size": int,
}
```

`replay_context_path` may be absolute or relative to the strict artifact file.
This is a lossless storage optimization: the replay trainer resolves the path
and receives the same replay-context dict that older inline artifacts stored
under `replay_context`.

When global video CFG is enabled but `action_guidance_scale <= 1`, the action
denoising replay only uses the conditional branch. New replay contexts may
therefore store only the conditional `transformer_cache` k/v branch, set
`use_cfg=false`, omit `negative_text_emb`, and set
`cfg_pruned_to_conditional=true`. This preserves the action replay mean while
roughly halving replay-context k/v storage for the common action-scale-one
collection setting. The cache snapshot path should clone only the kept k/v
branch instead of first copying the full CFG cache to CPU.

For bounded collection jobs, `STRICT_GRPO_REPLAY_CONTEXT_MAX_GB` is a
server-side per-context tensor budget. It is checked before `torch.save`; an
oversized context fails early instead of filling Scratch.

Replay context and per-transition `replay_input` are not required for dataset
validation or scalar smoke training. They are required for the real actor replay
trainer because that trainer recomputes current transition log probabilities by
running the saved denoising state through the current LingBot-VA transformer.
Old artifacts without these fields must fail fast in the actor trainer.

## Validation

Use path-only validation when working outside the container:

```bash
python tools/validate_grpo_dataset.py groups/grpo_groups.jsonl --fail-on-error
```

Use artifact inspection inside the container or any Python environment with
`torch`:

```bash
python tools/validate_grpo_dataset.py \
  groups/grpo_groups.jsonl \
  --inspect-artifacts \
  --out-summary groups/grpo_dataset_validation.json \
  --fail-on-error
```

With `--inspect-artifacts`, `transition_count` is the true denoising transition
count after expanding v2 trajectory artifacts.

For actor replay datasets, require replay fields explicitly:

```bash
python tools/validate_grpo_dataset.py \
  groups/grpo_groups.jsonl \
  --inspect-artifacts \
  --require-replay-context \
  --out-summary groups/grpo_dataset_validation_actor_replay.json \
  --fail-on-error
```

## Summary

For paper/audit tables, use:

```bash
python tools/summarize_grpo_groups.py \
  groups/grpo_groups.jsonl \
  --inspect-artifacts \
  --out-json groups/grpo_group_summary_inspected.json \
  --out-csv groups/grpo_group_summary_inspected.csv \
  --out-markdown groups/grpo_group_summary_inspected.md
```

Without `--inspect-artifacts`, transition counts are artifact path counts. This
is useful on login nodes without `torch`, but it undercounts v2 full-trajectory
data.

## Job Defaults

The grouped rollout jobs now default to:

```bash
STRICT_GRPO_CAPTURE=true
STRICT_GRPO_TRANSITION_STD=0.01
STRICT_GRPO_CAPTURE_SCOPE=action_denoising_trajectory
ACTION_NUM_INFERENCE_STEPS=50
STRICT_GRPO_SAVE_REPLAY_CONTEXT=false
STRICT_GRPO_REPLAY_CONTEXT_MAX_GB=0
```

To reproduce old first-step capture:

```bash
STRICT_GRPO_CAPTURE_SCOPE=first_action_denoising_step \
bash jobs/myriad/30_collect_grouped_rollouts_4gpu.sh
```

To collect data for real actor replay training:

```bash
STRICT_GRPO_CAPTURE_SCOPE=action_denoising_trajectory \
STRICT_GRPO_SAVE_REPLAY_CONTEXT=true \
STRICT_GRPO_REPLAY_CONTEXT_MAX_GB=5.0 \
bash jobs/myriad/30_collect_grouped_rollouts_4gpu.sh
```

## Replay

`wan_va.rl.denoising_replay.load_transition_batch()` expands both schemas into a
single `TransitionBatch`. v1 contributes one transition per artifact path; v2
contributes `num_transitions` transitions per artifact path.

`wan_va.rl.actor_replay.ActorReplayGrpoTrainer` is the real actor replay path.
It requires v2 artifacts with `replay_context` and `replay_input`, restores the
saved transformer KV cache, reruns the current actor on the saved denoising
state, and updates trainable actor parameters such as `action_embedder`,
`condition_embedder_action`, and `action_proj_out`.

For high-dimensional action chunks, use mean-reduced transition log-probabilities
for real actor replay (`--logprob-reduction mean` or
`GRPO_LOGPROB_REDUCTION=mean`). The real actor replay trainer defaults to
`mean` for both the Python CLI and Myriad job path. Sum-reduced chunk
log-probabilities are still available for compatibility, but can saturate the
clipped GRPO ratio and produce zero gradients when replayed means differ
slightly from the behavior artifact. For real actor replay training, also use a
conservative training-time std floor (`--logprob-std-floor` or
`GRPO_LOGPROB_STD_FLOOR`). The direct Python trainer and Myriad trainer job both
default this floor to `0.1`; pass `--logprob-std-floor 0` to the Python tool, or
`GRPO_LOGPROB_STD_FLOOR=0` through the job wrapper, only for a reviewed
diagnostic run that intentionally disables the floor. Strict artifacts can store
diffusion transition std values around `0.01`; with such a narrow Gaussian,
harmless replay/numerical mean drift can make the PPO/GRPO ratio unusably
saturated even when stored behavior logprobs validate correctly.

Trainer metrics include pre-step replay statistics (`ratio_mean`,
`clip_fraction`, `logratio_*`) and post-step update statistics
(`param_update_norm`, `param_update_max`). When comparing learning rates, use
the update metrics or online eval results; pre-step replay statistics are
expected to be identical for runs that start from the same base model.
