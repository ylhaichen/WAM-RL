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
  ]
}
```

`strict_grpo_artifact_paths` remains the trainer-facing reference list. A path
can point to either a v1 single-transition artifact or a v2 trajectory artifact.

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
                "noisy_latents": Tensor,
                "timesteps": Tensor,
            },
        }
    ],
    # Present only when STRICT_GRPO_SAVE_REPLAY_CONTEXT=true:
    "replay_context": {
        "schema_version": 1,
        "cache_name": str,
        "transformer_cache": list[dict[str, Tensor]],
        "grid_id": Tensor,
        "text_emb": Tensor,
        "negative_text_emb": Tensor | None,
        "use_cfg": bool,
        "action_guidance_scale": float,
        "action_num_inference_steps": int,
        "frame_chunk_size": int,
    },
}
```

Each v2 artifact represents one action chunk and stores every captured
non-terminal action denoising transition for that chunk.

`replay_context` and per-transition `replay_input` are not required for dataset
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
