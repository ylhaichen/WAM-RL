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

### Plan Alignment Check

Current work is aligned with the original implementation plan, but the active
frontier has moved. The missing piece is no longer the GRPO loss, grouped
rollout data contract, or basic actor replay trainer. The missing piece is a
repeatable Stage 2 loop:

```text
bounded replay-context collection
-> storage-audited subset
-> actor replay update with measured parameter movement
-> paired baseline-vs-actor eval
-> promotion gate against baseline repeatability
```

Work that is currently off-scope: paper writing, veRL migration, video-action
consistency, and large unbounded replay-context collections. These should wait
until the bounded native actor replay loop shows a credible signal.

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
- conditional-branch-only replay-context k/v storage when global video CFG is
  enabled but `action_guidance_scale <= 1`, which preserves action replay while
  avoiding an otherwise unused negative action branch and clones only the kept
  branch during cache snapshot;
- safe JSON-only GRPO group subsetting with `tools/subset_grpo_groups.py`;
- safe GRPO group curation/merge checks with `tools/merge_grpo_groups.py`,
  which writes a manifest and rejects duplicate `group_id` values by default;
- replay-context-footprint-aware subsetting with
  `--max-replay-context-gb`, so actor replay smoke subsets are bounded by
  actual resolved KV-cache storage rather than artifact count alone;
- artifact materialization with `tools/materialize_grpo_artifacts.py`, using
  symlink mode by default and optional replay-context materialization;
- materialization `--dry-run` planning, including source artifact/replay-context
  size summaries and `planned_copy_gb`, so copy-mode subsets can be checked
  before writing large replay-context files;
- actor subset preparation now writes `materialize_plan.json` and checks the
  dry-run resolved footprint against `SUBSET_STORAGE_MAX_RESOLVED_GB` before
  creating symlinks or copies;
- metadata-only external replay-context validation for actor replay dataset
  checks, so `--require-replay-context` no longer allocates full KV-cache
  tensors just to inspect keys;
- materialized subset storage-budget enforcement through
  `SUBSET_STORAGE_MAX_RESOLVED_GB`;
- collection-time strict artifact chunk filtering with
  `STRICT_GRPO_CAPTURE_CHUNK_STRIDE` and `STRICT_GRPO_CAPTURE_MAX_CHUNKS`;
- server-side per-replay-context tensor budget checks with
  `STRICT_GRPO_REPLAY_CONTEXT_MAX_GB`, so oversized contexts fail before
  `torch.save` fills Scratch;
- storage auditing with replay-context inspection and optional resolved-size
  budgets via `tools/audit_grpo_artifact_storage.py`;
- per-file replay-context tensor storage inspection with
  `tools/inspect_grpo_replay_context.py`, including scalar config fields,
  top-level tensor GiB, KV-cache batch sizes, compact `--print-summary`
  output, and conditional-only branch savings estimates;
- replay-context aggregate footprint summaries with
  `tools/summarize_grpo_replay_contexts.py`, which reports total
  replay-context file GiB by default and can optionally inspect context tensor
  metadata by config with `--inspect-context-tensors` on bounded sources;
- non-destructive Myriad storage cleanup planning with
  `tools/plan_myriad_storage_cleanup.py`, including protection for any
  non-empty `grpo_groups*.jsonl` source file;
- low-resource subset smoke submission with
  `jobs/myriad/36_submit_actor_replay_subset_smoke.sh`, including submit-time
  groups-file and storage-audit prechecks;
- storage-bounded replay-context collection submission with
  `jobs/myriad/39_submit_grpo_replayctx_bounded_4gpu.sh`, including
  attempt-budget-based Scratch headroom checks;
- standalone replay-context collection storage planning with
  `tools/plan_replay_context_collection.py`;
- operational runbook in `docs/WAM_RL_ACTOR_REPLAY_RUNBOOK.md`;
- actor replay training output summaries with
  `tools/summarize_actor_replay_training.py`, including explicit
  `parameter_update_detected` and update-norm warnings so one-step smoke runs
  are not mistaken for policy-improvement evidence, plus config recovery from
  `metrics.json` or older runs' lightweight `checkpoint.pt`, training
  provenance (`model_path`, `config_name`, `git_commit`), and `--discover-root`
  / `--latest` support for recent-run sweep tables;
- actor replay checkpoint tensor inspection/comparison with
  `tools/inspect_actor_replay_checkpoint.py`;
- real actor replay trainer over LingBot-VA transformer parameters;
- actor replay input storage audit and optional resolved-size training budget
  through `GRPO_MAX_RESOLVED_GB`;
- checkpoint loading into the inference server via
  `actor_replay_checkpoint_path`;
- `GRPO_ACTION_NUM_INFERENCE_STEPS` support in the trainer job;
- `GRPO_LOGPROB_REDUCTION=mean` and `GRPO_LOGPROB_STD_FLOOR` for more stable
  replay ratios;
- progress logging and failure diagnostics;
- `tools/diagnose_actor_replay.py` for replay-vs-stored logprob diagnosis.
- `tools/summarize_grpo_groups.py --inspect-artifacts` now reports
  replay-context count and resolved file GiB without loading replay-context
  tensor payloads.

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
storage audit with replay contexts:
  /home/zcably0/Scratch/wam-rl/results_grouped_rollouts/grpo_replayctx_staplerpad_k8_g1_s10_20260521_161720/groups/storage_audit_with_replay_contexts.json
  /home/zcably0/Scratch/wam-rl/results_grouped_rollouts/grpo_replayctx_staplerpad_k8_g1_s10_20260521_161720/groups/storage_audit_summary_mode.json
  /home/zcably0/Scratch/wam-rl/results_grouped_rollouts/grpo_replayctx_staplerpad_k8_g1_s10_20260521_161720/groups/grpo_group_summary_replay_context_audit.json
  /home/zcably0/Scratch/wam-rl/results_grouped_rollouts/grpo_replayctx_staplerpad_k8_g1_s10_20260521_161720/groups/grpo_replay_context_summary.json
  strict artifacts: 64 unique, 0.006 GiB resolved
  replay contexts: 64 unique
  artifacts + replay contexts: 430.996 GiB resolved
  summary replay-context file footprint: 430.990 GiB
  aggregate replay-context summary: 64 contexts, 430.990 GiB file footprint,
    context_tensor_inspected=false
representative replay-context metadata-only inspection:
  file_bytes: 7,230,819,664
  tensor_bytes: 7,230,772,480
  transformer_cache tensor bytes: 7,222,383,360
  text_emb tensor bytes: 4,194,304
  negative_text_emb tensor bytes: 4,194,304
enhanced inspector interpretation:
  action_guidance_scale: 1.0
  kv_batch_sizes: [2]
  conditional-only estimated tensor bytes: about 3.37 GiB
  conditional-only estimated savings: about 3.36 GiB per context
```

Latest budget subset selection/audit check on the same source, using one
success and one failure sample with a 30GiB replay-context selection budget and
a 40GiB resolved input budget:

```text
/home/zcably0/Scratch/wam-rl/debug_logs/subset_selection_checks/staplerpad_budget30_20260530_100715
```

It produced:

```text
samples: 2
artifact_refs: 4
expanded transitions: 40
validation_actor_replay: ok=true, error_count=0
combined resolved dependency bytes: 28,923,655,328
resolved replay contexts: 4 unique, 26.937 GiB
storage_budget.ok: true
selection_details:
  sample_idx=0, reward=1.0, artifact_ref_count=2
  sample_idx=1, reward=0.0, artifact_ref_count=2
```

Interpretation:

- enough to prove the real actor replay data contract and trainer path;
- too small to support any benchmark-improvement claim;
- too storage-heavy to scale without further artifact slimming or selective
  replay.
- per-file storage is dominated by duplicated transformer KV-cache state, not
  strict transition tensors.
- this run predates the current per-rollout capture metadata, so use it as a
  legacy source dataset only; new bounded collection should show
  `strict_grpo_capture_max_chunks`, capture chunk indices, and replay-context
  tensor bytes directly in rollout/group metadata.

Latest materialization planning check on the same two-sample subset:

```text
dry-run copy plan:
  unique strict artifacts: 4
  unique replay contexts: 4
  planned_copy_gb: 26.937
  source_replay_contexts_gb: 26.937
  missing_contexts: 0
  output root was not created
budget fail-fast smoke:
  /home/zcably0/Scratch/wam-rl/debug_logs/preflight_copy_budget_fail_smoke_20260530
  SUBSET_STORAGE_MAX_RESOLVED_GB=1
  prepare_exit: 3
  materialized groups created: false
  artifacts dir created: false
```

Interpretation: copy-mode self-contained subsets are now quantifiable before
copying, and the prepare job fails before writing large files when the selected
subset exceeds the resolved storage budget.

Latest one-step actor replay smoke on the two-sample/two-artifact staplerpad
subset:

```text
/home/zcably0/Scratch/wam-rl/results_grpo_actor_replay/grpo_actor_subset_2samples_2artifacts_1step_20260530_020837
```

Training output:

```text
transition_count: 40
steps: 1
final_loss: 0.02019871324300766
final_ratio_mean: 0.9099791586399079
final_grad_norm: 0.5571673512458801
final_param_update_norm: 5.789788701804355e-05
final_param_update_max: 1.1920928955078125e-07
parameter_update_detected: true
checkpoint tensors: 14
checkpoint params: 89,084,958
summary:
  /home/zcably0/Scratch/wam-rl/results_grpo_actor_replay/grpo_actor_subset_2samples_2artifacts_1step_20260530_020837/summary.json
checkpoint inspection:
  /home/zcably0/Scratch/wam-rl/results_grpo_actor_replay/grpo_actor_subset_2samples_2artifacts_1step_20260530_020837/checkpoint_inspection.json
job accounting:
  jobname=wam_grpo_actor_subset, slots=4, exit_status=0,
  ru_wallclock=176s, maxvmem=10.284GB
```

This checkpoint predates training-provenance logging, so its summary falls back
to `config_source=checkpoint` and does not include `model_path`, `config_name`,
or `git_commit`. Treat it as a valid smoke checkpoint, not as a fully
provenance-complete experiment record.

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
- paired baseline repeatability submission with
  `jobs/myriad/38_submit_eval_repeatability_pair.sh`;
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

Latest corrected provenance eval pair after removing accidental `qsub -V`
environment leakage from the paired eval submitter:

```text
run_id:
  actor_eval_pair_provenance_fixed_20260530_104124
baseline root:
  /home/zcably0/Scratch/wam-rl/results_actor_eval/baseline_move_stapler_pad_actor_eval_pair_provenance_fixed_20260530_104124
actor root:
  /home/zcably0/Scratch/wam-rl/results_actor_eval/actor_move_stapler_pad_actor_eval_pair_provenance_fixed_20260530_104124
comparison:
  /home/zcably0/Scratch/wam-rl/results_actor_eval/actor_eval_pair_provenance_fixed_20260530_104124_comparison
baseline policy_checkpoint:
  /home/zcably0/Scratch/wam-rl/checkpoints/lingbot-va-posttrain-robotwin
actor policy_checkpoint:
  /home/zcably0/Scratch/wam-rl/results_grpo_actor_replay/grpo_actor_subset_2samples_2artifacts_1step_20260530_020837/checkpoint.pt
reference_checkpoint:
  /home/zcably0/Scratch/wam-rl/checkpoints/lingbot-va-posttrain-robotwin
matched episodes: 2
baseline success: 2/2
actor success: 2/2
actor vs baseline: improved=0, regressed=0, same_success=2, same_failure=0
```

Interpretation:

- this is a wiring/provenance smoke check, not an improvement claim;
- both episode exports contain explicit `policy_checkpoint`,
  `reference_checkpoint`, and `action_num_inference_steps`;
- the previous pre-fix pair jobs were cancelled because `qsub -V` let the
  baseline job inherit the actor checkpoint path from the submit shell.

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

Available mitigations:

- bounded replay-context collection planning with
  `tools/plan_replay_context_collection.py`;
- `jobs/myriad/39_submit_grpo_replayctx_bounded_4gpu.sh`, which dry-runs
  storage estimates before submission and defaults to max-one-chunk capture;
- capture filtering by `STRICT_GRPO_CAPTURE_CHUNK_STRIDE` and
  `STRICT_GRPO_CAPTURE_MAX_CHUNKS`;
- per-file `STRICT_GRPO_REPLAY_CONTEXT_MAX_GB` checks before `torch.save`;
- `tools/subset_grpo_groups.py` for building small train/debug subsets that
  keep only selected success/failure samples and artifact references without
  copying or deleting large `.pt` files;
- `tools/materialize_grpo_artifacts.py` for turning such subsets into a small
  rewritten groups file plus symlinked or copied artifact tree. Use symlink mode
  for Scratch debug runs; use copy mode only when intentionally creating an
  archive/package. Use `--dry-run` or the prepare job's `materialize_plan.json`
  to inspect `planned_copy_gb` before running copy mode.

Next engineering target:

- quantify replay-context footprint across task, prompt, and action-guidance
  settings;
- reduce context payload further, beyond the current external-context split and
  conditional-branch pruning;
- separate short-lived Scratch working data from RDSS archival metadata and
  selected train/eval artifacts.

### 2. Closed-Loop Evaluation Nondeterminism

LingBot-VA sampling can be seeded, but RoboTwin/SAPIEN closed-loop execution can
still diverge after identical early actions. This limits the meaning of tiny
before/after evals.

Next engineering target:

- always export per-episode seed, prompt, sampling seed, action count, and
  success;
- always record per-episode policy/reference checkpoint provenance and
  `ACTION_NUM_INFERENCE_STEPS` for baseline-vs-actor evals;
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

Latest storage-vs-mixing check on 2026-05-30 with about 77.8 GiB free Scratch
and 50 GiB reserved headroom:

```text
move_stapler_pad success_rate: 0.625
k=4 mixed probability: 0.828, expected attempts per mixed group: 1.21
k=8 mixed probability: 0.976, expected attempts per mixed group: 1.02
k=4/g1/max-1-chunk with GROUP_MAX_ATTEMPTS=1:
  16 GiB + 50 GiB headroom -> ok
k=4/g1/max-1-chunk with default GROUP_RETRY_MULTIPLIER=3:
  48 GiB + 50 GiB headroom -> not ok
k=8/g1/max-1-chunk with GROUP_MAX_ATTEMPTS=1:
  32 GiB + 50 GiB headroom -> not ok
planning outputs:
  /home/zcably0/Scratch/wam-rl/debug_logs/storage_audits/next_collection_plan_20260530_094128_mixing.json
  /home/zcably0/Scratch/wam-rl/debug_logs/storage_audits/next_collection_plan_20260530_094128_k4_storage.json
  /home/zcably0/Scratch/wam-rl/debug_logs/storage_audits/next_collection_plan_20260530_094128_k4_attempts3_storage.json
  /home/zcably0/Scratch/wam-rl/debug_logs/storage_audits/next_collection_plan_20260530_094128_k8_storage.json
```

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
- Storage cleanup is now reviewable through a non-destructive planner, but
  cleanup should not be treated as automatic deletion permission.

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
   artifacts. Run `tools/plan_myriad_storage_cleanup.py` before proposing any
   large cleanup.
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
   subsets before running expensive jobs. For queued Myriad subset preparation,
   use `jobs/myriad/35_submit_prepare_actor_replay_subset.sh --dry-run` and
   review the explicit `qsub -v` variables before submission.
   When combining multiple bounded mixed-group sources, use
   `tools/merge_grpo_groups.py` instead of manual `cat` so duplicate group ids
   are caught before validation/training.
2. Use `jobs/myriad/39_submit_grpo_replayctx_bounded_4gpu.sh` for any new
   replay-context collection smoke, with dry-run review before submission.
3. Use `jobs/myriad/37_submit_actor_eval_pair_smoke.sh` and
   `tools/summarize_actor_eval_pair.py` for baseline-vs-actor eval comparisons
   before interpreting aggregate success rates.
4. Gate any candidate checkpoint with `tools/gate_actor_eval_promotion.py`
   against baseline repeatability before treating it as more than smoke.
5. Run a controlled actor replay training/eval loop on `move_stapler_pad` with
   more than one mixed group.
6. Only after a reliable signal appears, broaden to `turn_switch` and
   `open_microwave`.

## Bottom Line

The project should stay project-first, not paper-first. The right next step is
not writing more paper text or starting veRL. The right next step is making the
real actor replay loop storage-aware, eval-aware, and repeatable enough that a
small improvement signal would be credible if it appears.
