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
        }
    ],
}
```

Each v2 artifact represents one action chunk and stores every captured
non-terminal action denoising transition for that chunk.

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
```

To reproduce old first-step capture:

```bash
STRICT_GRPO_CAPTURE_SCOPE=first_action_denoising_step \
bash jobs/myriad/30_collect_grouped_rollouts_4gpu.sh
```

## Replay

`wan_va.rl.denoising_replay.load_transition_batch()` expands both schemas into a
single `TransitionBatch`. v1 contributes one transition per artifact path; v2
contributes `num_transitions` transitions per artifact path.

