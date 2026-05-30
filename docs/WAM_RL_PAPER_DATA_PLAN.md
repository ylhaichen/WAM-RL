# WAM-RL Paper Data Collection Plan

**Date:** 2026-05-18
**Purpose:** local record for paper writing, experiment accounting, and future data collection
**Scope:** early-stage WAM-RL paper around Denoising-step GRPO for LingBot-VA on RoboTwin

> Current status note (2026-05-30): this is a paper-data planning record, not
> the active project execution plan. The project is currently prioritizing real
> actor replay engineering, storage control, and evaluation repeatability over
> paper writing. Use `docs/WAM_RL_CURRENT_PROJECT_STATUS.md` for current
> completion, blockers, and safe claims.

This file records what data the paper needs, what has already been collected, what still needs to be exported, and which code paths should be used to avoid manual statistics errors.

---

## 1. Paper Data Goals

The current paper should be written as an early-stage methods and systems paper, not as a final benchmark-improvement paper. The data should support four claims:

1. **Motivation:** token-action VLA RL does not directly transfer to FlowMatch continuous-action LingBot-VA.
2. **Pipeline feasibility:** grouped RoboTwin rollouts can be converted into strict denoising-transition GRPO datasets.
3. **Data-quality insight:** task difficulty and mixed-group yield are central bottlenecks for online VLA RL.
4. **Optimization-contract validation:** strict artifacts can be loaded, validated, and used in offline GRPO smoke training.

The paper should not claim real LingBot-VA actor improvement until actor replay and checkpoint evaluation are implemented.

---

## 2. Required Paper Tables

### Table A: Baseline RoboTwin Task Difficulty

Purpose: justify task selection.

Required fields:

- `task`
- `success`
- `total`
- `success_rate`
- `95% CI`
- `difficulty_band`

Already available from the 50-task sweep:

```text
overall: 919 / 1000 = 91.9%
hard: hanging_mug, turn_switch
medium: open_microwave, put_bottles_dustbin, move_stapler_pad, press_stapler, place_dual_shoes, place_fan, put_object_cabinet, stack_bowls_three
```

Export command:

```bash
python tools/summarize_robotwin_results.py "$SWEEP" \
  --csv "$SWEEP/summary.csv" \
  --sort rate
```

Files to preserve:

- `$SWEEP/summary.csv`
- all `$SWEEP/**/metrics/*/res.json`
- job log containing checkpoint/config information

### Table B: Grouped Rollout Dataset Quality

Purpose: show how many useful GRPO groups were obtained and where the signal comes from.

Required fields:

- dataset name
- tasks
- accepted attempts
- failed attempts
- total groups
- mixed groups
- mixed-group ratio
- skipped all-success groups
- skipped all-failure groups
- strict transition count
- validation status

Key current datasets:

| dataset | path | status |
|---|---|---|
| A partial hard/medium | `/home/zcably0/Scratch/wam-rl/results_grouped_rollouts/grpo_scale_tasks_a_k8_g8_retry_20260515_000825` | usable partial |
| B easy-heavy | `/home/zcably0/Scratch/wam-rl/results_grouped_rollouts/grpo_scale_tasks_b_k8_g8_seedsearch_20260514_210921` | complete |
| M open_microwave isolated | `/home/zcably0/Scratch/wam-rl/results_grouped_rollouts/grpo_open_microwave_k8_g4_seedsearch_20260517_003049` | complete, artifact-inspected |
| N no_mw accepted4 | `/home/zcably0/Scratch/wam-rl/results_grouped_rollouts/grpo_core_no_mw_k8_g8_seedsearch_20260517_003044` | usable accepted4 partial |

Files to preserve from each dataset:

- `groups/rollouts_flat*.csv`
- `groups/rollouts_flat*.jsonl`
- `groups/grpo_groups*.jsonl`
- `groups/grpo_summary*.json`
- `groups/grpo_manifest*.json`
- `groups/grpo_dataset_validation*.json`
- `groups/successful_attempt_roots.txt`
- `groups/failed_attempt_roots.txt`
- job stdout log under `logs/jobs`

### Table C: Combined GRPO Training Dataset

Purpose: provide the main current dataset used by smoke training and paper audit.

Current main combined dataset:

```text
/home/zcably0/Scratch/wam-rl/results_grpo_datasets/a_partial_b_mw_nomw_accepted4_20260518_164531/grpo_groups.jsonl
```

Known stats:

```text
groups: 23
samples: 184
strict transitions: 3077
validation: ok=true, error_count=0
tasks: 7
```

Per-task current stats:

| task | groups | samples | success | failure | transitions |
|---|---:|---:|---:|---:|---:|
| adjust_bottle | 2 | 16 | 14 | 2 | 90 |
| hanging_mug | 2 | 16 | 5 | 11 | 370 |
| move_stapler_pad | 5 | 40 | 29 | 11 | 308 |
| open_microwave | 5 | 40 | 20 | 20 | 1251 |
| place_mouse_pad | 2 | 16 | 13 | 3 | 109 |
| put_bottles_dustbin | 2 | 16 | 7 | 9 | 634 |
| turn_switch | 5 | 40 | 22 | 18 | 315 |

Use this local tool for reproducible summaries:

```bash
python tools/summarize_grpo_groups.py \
  /home/zcably0/Scratch/wam-rl/results_grpo_datasets/a_partial_b_mw_nomw_accepted4_20260518_164531/grpo_groups.jsonl \
  --out-json /home/zcably0/Scratch/wam-rl/results_grpo_datasets/a_partial_b_mw_nomw_accepted4_20260518_164531/summary.json \
  --out-csv /home/zcably0/Scratch/wam-rl/results_grpo_datasets/a_partial_b_mw_nomw_accepted4_20260518_164531/summary_by_task.csv \
  --out-markdown /home/zcably0/Scratch/wam-rl/results_grpo_datasets/a_partial_b_mw_nomw_accepted4_20260518_164531/summary.md
```

### Table D: Strict Artifact Validation

Purpose: show that the dataset is not just JSON references; the tensor artifacts exist and load correctly.

Required fields:

- dataset
- transition count
- artifact inspection status
- schema version
- action tensor shape summary
- old logprob finite rate

Currently validated:

```text
M open_microwave:
  transition_count = 1031
  inspect-artifacts ok

combined A+B+M+N:
  transition_count = 3077
  JSON/path validation ok
```

Artifact inspection requires `torch`, so run it in the container or a compute node:

```bash
export RESULTS="$M_RESULTS"
export REPO_ROOT="$PWD"
source jobs/myriad/common.sh

container_exec_gpu <<'CONTAINER'
set -euo pipefail
cd "$REPO_ROOT"
source "$WAN_VA_VENV/bin/activate"

python tools/validate_grpo_dataset.py \
  "$RESULTS/groups/grpo_groups.jsonl" \
  --inspect-artifacts \
  --out-summary "$RESULTS/groups/grpo_dataset_validation_inspected.json" \
  --fail-on-error
CONTAINER
```

Do not run large artifact inspection or smoke training on a login node during penalty windows.

### Table E: Offline GRPO Smoke Training

Purpose: validate the data/loss/optimizer/checkpoint contract.

Required fields:

- input dataset path
- transition count
- steps
- learning rate
- clip range
- device
- final loss
- ratio statistics
- checkpoint path
- metrics path

Current known smoke result:

```text
dataset:
  /home/zcably0/Scratch/wam-rl/results_grpo_datasets/a_partial_plus_b_20260518_012501/grpo_groups.jsonl

output:
  /home/zcably0/Scratch/wam-rl/results_grpo_train/grpo_combined_smoke_300_20260518_012851

steps: 300
learning_rate: 0.001
clip_low: 0.2
clip_high: 0.28
device: cpu
transition_count: 727
final_loss: 0.5725651979446411
```

Important interpretation:

This is `StrictArtifactScalarPolicy` smoke training. It is useful for validating the pipeline, but it is not a real LingBot-VA actor update.

Future smoke runs should be submitted with `qsub` or run on an interactive compute node, not on a login node.

---

## 3. Data Still Needed

### 3.1 Full Denoising-Step Capture Data

The current strict artifacts mainly represent the first action denoising transition. The final method needs full or multi-step denoising capture.

After implementing full capture, export:

- transitions per rollout;
- chunks per rollout;
- denoising steps per chunk;
- tensor shape summary;
- old logprob finite rate;
- artifact count per task;
- disk size per rollout/task;
- validation runtime.

Target table:

| task | rollouts | chunks | denoising transitions | transitions/rollout | artifact GB | validation ok |
|---|---:|---:|---:|---:|---:|---|

### 3.2 Real Actor Replay Diagnostics

Once real actor replay is implemented, collect:

- new logprob finite rate;
- old/new logprob difference distribution;
- ratio mean/min/max;
- clip fraction;
- gradient norm by module;
- parameter update norm by module;
- GPU memory;
- train step time.

Target modules for first real update:

- `action_embedder`
- `condition_embedder_action`
- `action_proj_out`

### 3.3 Base vs GRPO Checkpoint Evaluation

This is required before claiming policy improvement.

Minimum evaluation plan:

- base checkpoint vs GRPO checkpoint;
- 4-8 selected tasks;
- held-out seeds;
- confidence intervals;
- easy-task regression checks.

Recommended task set:

```text
core signal:
  open_microwave
  turn_switch
  put_bottles_dustbin
  move_stapler_pad

hard stress:
  hanging_mug

easy regression:
  adjust_bottle
  place_mouse_pad
```

---

## 4. Local Code Support

Current paper-data tools:

- `tools/summarize_robotwin_results.py`: baseline eval `res.json` summary.
- `tools/collect_robotwin_rollouts.py`: flat rollout records.
- `tools/build_grpo_groups.py`: mixed GRPO group construction.
- `tools/validate_grpo_dataset.py`: path and artifact validation.
- `tools/summarize_grpo_groups.py`: paper-facing GRPO group summary.

Recommended local paper-data workflow:

```bash
# 1. Baseline table
python tools/summarize_robotwin_results.py "$SWEEP" \
  --csv "$SWEEP/summary.csv" \
  --sort rate

# 2. Rollout flat records
python tools/collect_robotwin_rollouts.py "${ROOTS[@]}" \
  --out-jsonl "$RESULTS/groups/rollouts_flat.jsonl" \
  --out-csv "$RESULTS/groups/rollouts_flat.csv"

# 3. GRPO groups
python tools/build_grpo_groups.py "${ROOTS[@]}" \
  --canonicalize-legacy-group-ids \
  --expected-group-size 8 \
  --require-strict-artifacts \
  --require-existing-artifacts \
  --out-jsonl "$RESULTS/groups/grpo_groups.jsonl" \
  --out-summary "$RESULTS/groups/grpo_summary.json" \
  --out-manifest "$RESULTS/groups/grpo_manifest.json"

# 4. Dataset validation
python tools/validate_grpo_dataset.py \
  "$RESULTS/groups/grpo_groups.jsonl" \
  --out-summary "$RESULTS/groups/grpo_dataset_validation.json" \
  --fail-on-error

# 5. Paper-facing group summary
python tools/summarize_grpo_groups.py \
  "$RESULTS/groups/grpo_groups.jsonl" \
  --out-json "$RESULTS/groups/grpo_paper_summary.json" \
  --out-csv "$RESULTS/groups/grpo_paper_summary_by_task.csv" \
  --out-markdown "$RESULTS/groups/grpo_paper_summary.md"
```

---

## 5. Paper Claim Boundary

### Safe Current Claims

- The project defines a denoising-transition policy interface for FlowMatch VLA RL.
- The grouped rollout pipeline has produced validated mixed GRPO datasets on RoboTwin.
- The current combined dataset contains `23` groups, `184` samples, and `3077` strict transitions across `7` tasks.
- Task selection strongly affects mixed-group yield.
- Isolating `open_microwave` turns an unstable task into a usable high-signal dataset.
- Offline smoke training validates the strict-artifact GRPO loss and checkpoint path.

### Claims To Avoid Until More Work Is Done

- WAM-RL improves LingBot-VA performance.
- The current trainer updates the real LingBot-VA actor.
- Full online GRPO is complete.
- Full denoising trajectory replay is complete.
- Video-action consistency is validated.

---

## 6. Immediate Next Actions

1. Run `tools/summarize_grpo_groups.py` on the combined dataset and preserve JSON/CSV/Markdown outputs.
2. Record artifact-inspected validation for the combined dataset from a compute node or container.
3. Submit smoke training for the combined dataset via `qsub`, not on the login node.
4. Implement full denoising-step capture with backward-compatible artifact schema.
5. Add a schema-summary tool for full denoising artifacts.
6. Update this document after every major dataset or checkpoint.
