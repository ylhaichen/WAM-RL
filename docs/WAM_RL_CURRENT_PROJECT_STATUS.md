# WAM-RL Current Project Status

**Date:** 2026-05-30
**Purpose:** current engineering status, project alignment check, and near-term execution plan
**Scope:** WAM-RL native SimpleVLA-style GRPO for LingBot-VA on RoboTwin

This document is the current status record. Older planning and paper-data
documents remain useful for historical context, but this file should be checked
first when deciding what to implement or claim.

## Executive Summary

The project is not off-track. The current work is exactly at the boundary
between Stage 1 and Stage 2 of the original plan:

```text
Stage 1: native offline pipeline        mostly complete
Stage 2: native iterative actor update  started, not validated for gains
Stage 3: veRL scale-up                  not started and should wait
```

The central direction is still:

```text
SimpleVLA-style grouped outcome RL
-> LingBot-VA continuous FlowMatch action denoising
-> denoising-transition GRPO
-> RoboTwin closed-loop evaluation
```

The repository now has enough infrastructure to collect grouped RoboTwin
rollouts, build GRPO groups, validate strict action-denoising artifacts, run
offline smoke training, train a real LingBot-VA actor replay checkpoint on a
small replay-context dataset, and load that checkpoint for evaluation.

The project should not yet claim policy improvement. Current actor replay evals
are smoke checks only because the real replay dataset is tiny and RoboTwin
closed-loop evaluation is not fully deterministic even when LingBot-VA sampling
is seeded.

## Completion By Stage

| Stage | Status | Completion | Evidence | Main Missing Piece |
|---|---|---:|---|---|
| Stage 1: Native + Offline | usable | 80-90% | grouped datasets, strict artifact validation, offline GRPO smoke training | refresh docs and keep curated dataset paths clean |
| Stage 2: Native + Iterative Online | in progress | 25-35% | real actor replay trainer, replay-context validation, checkpoint loading, eval plumbing | reliable improvement loop and scalable replay-context storage |
| Stage 3: veRL Scale-Up | not started | 0% | intentionally deferred | native actor replay must first show a signal |

## Implemented And Validated

### Grouped Rollout Data Path

Implemented:

- grouped RoboTwin rollout collection;
- seed search and dynamic sampling;
- strict GRPO artifact references;
- `successful_attempt_roots.txt` and `failed_attempt_roots.txt`;
- validation with optional artifact inspection;
- per-task and per-group summaries.

Current high-value combined dataset:

```text
/home/zcably0/Scratch/wam-rl/results_grpo_datasets/a_b_m_n_20260519_164542/grpo_groups.jsonl
```

Validated stats:

```text
groups: 35
samples: 280
success: 161
failure: 119
success_rate: 0.575
strict transitions: 4644
validation: ok=true, error_count=0
tasks: adjust_bottle, hanging_mug, move_stapler_pad, open_microwave,
       place_mouse_pad, put_bottles_dustbin, turn_switch
```

### Offline Strict GRPO Smoke Training

The offline scalar-policy smoke path validates the dataset, GRPO loss,
optimizer, metrics, and checkpoint-writing contract. It is not a real
LingBot-VA actor update.

Latest closeout:

```text
input:
  /home/zcably0/Scratch/wam-rl/results_grpo_datasets/a_b_m_n_20260519_164542/grpo_groups.jsonl
output:
  /home/zcably0/Scratch/wam-rl/results_grpo_train/grpo_combined_offline_closeout_50_20260519_175844
transition_count: 4644
steps: 50
final_loss: 0.5320302248001099
checkpoint: checkpoint.pt
```

Interpretation:

- useful as a contract and regression test;
- not evidence that LingBot-VA improves;
- still part of Stage 1, not Stage 2.

### Real Actor Replay

Implemented:

- v2 strict artifacts with full action denoising trajectory capture;
- replay context capture and external per-chunk replay-context files;
- safe JSON-only GRPO group subsetting with `tools/subset_grpo_groups.py`;
- artifact materialization with `tools/materialize_grpo_artifacts.py`, using
  symlink mode by default and optional replay-context materialization;
- collection-time strict artifact chunk filtering with
  `STRICT_GRPO_CAPTURE_CHUNK_STRIDE` and `STRICT_GRPO_CAPTURE_MAX_CHUNKS`;
- low-resource subset smoke submission with
  `jobs/myriad/36_submit_actor_replay_subset_smoke.sh`;
- storage footprint auditing with `tools/audit_grpo_artifact_storage.py`;
- operational runbook in `docs/WAM_RL_ACTOR_REPLAY_RUNBOOK.md`;
- actor replay training output summaries with
  `tools/summarize_actor_replay_training.py`;
- actor replay checkpoint tensor inspection/comparison with
  `tools/inspect_actor_replay_checkpoint.py`;
- real actor replay trainer over LingBot-VA transformer parameters;
- checkpoint loading into the inference server via
  `actor_replay_checkpoint_path`;
- `GRPO_ACTION_NUM_INFERENCE_STEPS` support in the trainer job;
- `GRPO_LOGPROB_REDUCTION=mean` and `GRPO_LOGPROB_STD_FLOOR` for more stable
  replay ratios;
- progress logging and failure diagnostics;
- `tools/diagnose_actor_replay.py` for replay-vs-stored logprob diagnosis.

Latest validated real actor replay dataset:

```text
/home/zcably0/Scratch/wam-rl/results_grouped_rollouts/grpo_replayctx_staplerpad_k8_g1_s10_20260521_161720/groups/grpo_groups.jsonl
```

Validated stats:

```text
task: move_stapler_pad
groups: 1
samples: 8
success: 5
failure: 3
transition_count: 640
validation with --require-replay-context: ok=true, error_count=0
disk usage before cleanup/archive: about 432G
```

Interpretation:

- enough to prove the real actor replay data contract and trainer path;
- too small to support any benchmark-improvement claim;
- too storage-heavy to scale without further artifact slimming or selective
  replay.

### Evaluation Protocol

Implemented:

- `PROMPT_INDEX` support for fixed prompt variants;
- explicit `SEED` logging/control for RoboTwin env seed selection;
- `SAMPLING_SEED` support for deterministic LingBot-VA action sampling;
- `SAMPLING_SEED_PER_ENV=true` for deterministic but distinct per-env sampling;
- per-episode export in `tools/summarize_robotwin_results.py`;
- matched per-episode eval comparison with
  `tools/compare_robotwin_eval_episodes.py`;
- paired baseline-vs-actor eval smoke submission with
  `jobs/myriad/37_submit_actor_eval_pair_smoke.sh`;
- paired eval summary/export with `tools/summarize_actor_eval_pair.py`;
- zero-match guard in paired eval summaries to catch seed/prompt/sampling
  control mismatches;
- repeatability summaries across repeated eval roots with
  `tools/summarize_robotwin_repeatability.py`;
- conservative actor eval promotion gating with
  `tools/gate_actor_eval_promotion.py`;
- one-GPU eval job support for actor replay checkpoint loading.

Observed behavior:

- the first generated action chunk can be bit-exact when server sampling is
  fixed;
- RoboTwin closed-loop runs can still diverge after executing the same first
  action chunk;
- small `n=5` eval differences are therefore smoke signals, not robust policy
  improvement evidence.

Known smoke evals:

```text
baseline move_stapler_pad prompt0 s10 n5: 4/5
baseline repeat prompt0 s10 n5:          3/5
actor lr=1e-7 smoke:                     3/5
actor lr=1e-6 smoke:                     3/5
```

Latest explicit-seed matched controls on `move_stapler_pad`, prompt0,
`ACTION_NUM_INFERENCE_STEPS=10`, `SEED=10000`,
`SAMPLING_SEED=12345`, `SAMPLING_SEED_PER_ENV=true`, `n=2`:

```text
baseline previous:       2/2
baseline repeat:         1/2
actor lr=0 no-op:        1/2
actor one-step subset:   0/2
comparison output:
  /home/zcably0/Scratch/wam-rl/results_actor_eval/seed10000_controls_20260530_0422/four_way_comparison.json
baseline repeatability output:
  /home/zcably0/Scratch/wam-rl/results_actor_eval/seed10000_controls_20260530_0422/baseline_repeatability_summary.json
promotion gate outputs:
  /home/zcably0/Scratch/wam-rl/results_actor_eval/seed10000_controls_20260530_0422/lr0_promotion_gate.json
  /home/zcably0/Scratch/wam-rl/results_actor_eval/seed10000_controls_20260530_0422/actor_step_promotion_gate.json
```

The baseline repeat and `lr=0` no-op checkpoint had the same success pattern on
the two matched episodes: seed `100010000` succeeded with 144 actions, while
seed `100010001` reached the 400-step limit and failed. This makes the previous
baseline-vs-`lr=0` delta primarily an eval repeatability issue, not direct
evidence that actor checkpoint loading is broken.

The two baseline runs alone had one stable-success key and one flipped key,
giving a baseline repeatability flip rate of `0.5` on this tiny control set.
This is diagnostic, not a benchmark estimate, but it is enough to make `n=2`
policy deltas non-actionable.

`tools/gate_actor_eval_promotion.py` blocks both the `lr=0` control and the
one-step actor subset under the default promotion thresholds because the matched
eval and baseline-repeatability episode counts are below 10, the baseline
repeatability `flip_rate` is `0.5`, and neither candidate has positive net
matched improvement.

The one-step actor subset checkpoint was compared against the `lr=0` no-op
checkpoint at the tensor level:

```text
trainable tensors: 14
trainable params: 89,084,958
relative_delta_l2: 3.600973986046038e-07
delta_max_abs: 1.1920928955078125e-07
report:
  /home/zcably0/Scratch/wam-rl/results_grpo_actor_replay/grpo_actor_subset_2samples_2artifacts_1step_20260530_020837/checkpoint_vs_lr0/report.json
```

Interpretation:

- evaluation plumbing works;
- current data do not show reliable improvement;
- tiny `n=2` actor-vs-baseline differences are confounded by baseline
  repeatability;
- checkpoint loading is not ruled out as a possible source of small numerical
  differences, but the current evidence points more strongly to RoboTwin
  closed-loop repeat instability;
- future eval should use fixed `PROMPT_INDEX`, fixed sampling seed policy, and
  larger held-out seed sets before promotion.

## Current Blockers

### 1. Replay-Context Storage

Replay context includes transformer KV cache state. Even with 10 action
denoising steps, a single k=8 group can require hundreds of GB. This is the
largest practical blocker for scaling real actor replay.

Current policy:

- run active jobs on Scratch;
- archive completed metadata and selected datasets to RDSS;
- do not write active `torch.save`-heavy jobs directly to RDSS;
- do not delete a non-empty training dataset's referenced `server_vis/` unless
  the dataset is intentionally being retired.

Next engineering target:

- selective replay artifact construction;
- bounded replay-context capture by chunk stride or max chunk count;
- per-chunk replay-context deduplication beyond the current external-context
  split;
- `tools/subset_grpo_groups.py` for building small train/debug subsets that
  keep only selected success/failure samples and artifact references without
  copying or deleting large `.pt` files.
- `tools/materialize_grpo_artifacts.py` for turning such subsets into a small
  rewritten groups file plus symlinked or copied artifact tree. Use symlink mode
  for Scratch debug runs; use copy mode only when intentionally creating an
  archive/package.

### 2. Closed-Loop Evaluation Nondeterminism

LingBot-VA sampling can be seeded, but RoboTwin/SAPIEN closed-loop execution can
still diverge after identical early actions. This limits the meaning of tiny
before/after evals.

Next engineering target:

- always export per-episode seed, prompt, sampling seed, action count, and
  success;
- compare policies by paired seed sets with
  `tools/compare_robotwin_eval_episodes.py`;
- treat `n <= 5` as smoke only;
- use larger `n` only after storage and actor training are stable enough.

### 3. Mixed-Group Yield

Group size 8 is better than group size 4 for low/medium success-rate tasks, but
storage makes k=8 replay-context runs expensive. Task selection should continue
to target medium/hard tasks with expected mixed groups.

Recommended near-term task order:

```text
move_stapler_pad -> turn_switch -> open_microwave -> put_bottles_dustbin
```

Avoid making `hanging_mug` the first actor-replay scaling task unless the goal
is specifically to study low-success hard-task collection; its low success rate
means k=4 is unlikely to produce useful mixed groups reliably.

## Claims Boundary

Safe claims:

- WAM-RL defines and implements a denoising-transition GRPO data contract for
  FlowMatch action generation.
- The native RoboTwin grouped-rollout pipeline can produce mixed binary-reward
  GRPO groups.
- Strict artifacts can be validated and used for offline GRPO smoke training.
- Real actor replay is implemented end to end on a tiny replay-context dataset.
- Replay-context storage and RoboTwin eval nondeterminism are current scaling
  blockers.

Unsafe claims:

- LingBot-VA policy performance improves over baseline.
- Online GRPO is complete.
- veRL integration is needed now.
- 1-group or `n=5` smoke evals prove learning.
- Actor replay storage is solved.

## Near-Term Engineering Plan

### Immediate

1. Keep docs and agent instructions aligned with the current Stage 1 -> Stage 2
   state.
2. Preserve the current curated datasets and avoid deleting referenced
   artifacts.
3. Use Myriad container tests as the source of truth for actor replay and
   RoboTwin-adjacent code.
4. Keep actor replay training defaults conservative:

```text
GRPO_ACTION_NUM_INFERENCE_STEPS=<collection steps>
GRPO_LOGPROB_REDUCTION=mean
GRPO_LOGPROB_STD_FLOOR=0.1
GRPO_PROGRESS_EVERY=50
```

### Next Technical Milestones

1. Use `tools/subset_grpo_groups.py` to create small actor replay train/debug
   subsets before running expensive jobs.
2. Use `jobs/myriad/37_submit_actor_eval_pair_smoke.sh` and
   `tools/summarize_actor_eval_pair.py` for baseline-vs-actor eval comparisons
   before interpreting aggregate success rates.
3. Gate any candidate checkpoint with `tools/gate_actor_eval_promotion.py`
   against baseline repeatability before treating it as more than smoke.
4. Run a controlled actor replay training/eval loop on `move_stapler_pad` with
   more than one mixed group.
5. Only after a reliable signal appears, broaden to `turn_switch` and
   `open_microwave`.

## Bottom Line

The project should stay project-first, not paper-first. The right next step is
not writing more paper text or starting veRL. The right next step is making the
real actor replay loop storage-aware, eval-aware, and repeatable enough that a
small improvement signal would be credible if it appears.
