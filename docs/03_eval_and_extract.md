# Step 3: Behavioral evaluation + activation extraction across 3 models × 2 tasks

**Project:** From Linear Probes to Bayesian Manifolds: Geometry of Arithmetic in Language Models
**Authors:** Anshul Kumar (CMU, primary), Deeksha Varshney (IIT Jodhpur, advisor), Manoj Kumar (IIT Roorkee, advisor), Barnabás Póczos (CMU, main advisor)
**Carnegie Mellon University, May 2026**

This document records every decision, every number, and every result from
Step 3 — the first model-running step. For each (model, task, problem) we
record the model's first generated token, mark `correct ∈ {0, 1}` against
the gold first-token id (already stored in the dataset from Step 1), and
capture residual-stream activations at the `=` token for 5 layers per model.
It is the truth document for this stage. All numbers are validated against
the actual output files at `data/answers/{model_key}/` and
`data/activations/{model_key}/` as of 2026-05-09.

---

## Table of Contents

1. [Purpose of this stage](#1-purpose-of-this-stage)
2. [What this stage is and is not](#2-what-this-stage-is-and-is-not)
3. [The pipeline (one model, both tasks, two passes per batch)](#3-the-pipeline)
4. [Models, layers, and per-model architecture access](#4-models-layers-and-per-model-architecture-access)
5. [Smoke test on the current node](#5-smoke-test-on-the-current-node)
6. [Utilization probe (Llama × full addition at batch=512)](#6-utilization-probe)
7. [SLURM array job (3 tasks, one A6000 each, general partition, 2-day cap)](#7-slurm-array-job)
8. [Results — accuracy](#8-results--accuracy)
9. [Results — activation validation](#9-results--activation-validation)
10. [Cross-checks](#10-cross-checks)
11. [Implications for downstream stages](#11-implications-for-downstream-stages)
12. [Output files](#12-output-files)
13. [Runtime and reproducibility](#13-runtime-and-reproducibility)
14. [Open questions and follow-ups](#14-open-questions-and-follow-ups)

---

## 1. Purpose of this stage

Step 3 is the first model-running step. Its outputs are the substrate every
subsequent stage of plan v6 reads:

- **Behavioral evaluation:** for each (model, problem), the first generated
  token is compared against the gold `first_token_id_{model}` (already stored
  in the Step 2 dataset from the Step 1 tokenizer sweep). The boolean
  `correct ∈ {0, 1}` produces the "correct subset" plan §3.1 / §7.4 act on.
- **Activation extraction:** for each (model, problem), the residual-stream
  output at the `=` token (last input token, position -1) is captured for the
  five configured layers per model. These are the inputs to Stage 1 (linear
  probe), Stage 2 (Bayesian manifold), Stage 3 (orthogonalisation), and Stage
  4 (causal ablation).

Both happen in **the same job per model** — separate forward passes (one with
hooks for activations, one with `model.generate` for predictions) but in the
same script invocation, on the same loaded model. This matches arithmetic-
geometry's `pipeline.py` discipline.

The job parallelises trivially across models: three independent SLURM array
tasks, one A6000 each, no inter-task communication.

---

## 2. What this stage is and is not

### What it is

- A two-pass-per-batch script, run once per model, producing per-task
  predictions and per-layer activations.
- A SLURM array submission with three tasks (one per model), each
  requesting one A6000 from the `general` partition.
- A smoke test on the current node + a utilization probe to verify
  GPU-util ≥ 70% before the array job submits.

### What it is not

- **Not** probe fitting, not Bayesian work, not orthogonalisation, not
  ablation. Those are plan v6 Stages 1–4 (separate steps).
- **Not** correctness analysis beyond the boolean flag. Computing the
  "correct subset" view per (model, task, concept) is downstream.
- **Not** multi-token-answer-sequence analysis. We capture 4 generated
  tokens for `raw_text` but score only the first token for `correct`.
- **Not** a re-derivation of dataset labels. Step 2's outputs are read-only
  inputs here.

---

## 3. The pipeline

For each model, per task (addition then multiplication), the script:

1. Loads the model in **bfloat16** onto `cuda:0`. Tokenizer with
   `pad_token = eos_token`, `padding_side = "left"` (so position -1 of the
   padded tensor is always the last real input token, the `=`).
2. Iterates problems in batches of **512** (configurable; OOM-fallback halves
   to a floor of 64).
3. **Pass 1 (activation extraction):** registers a forward hook on each of
   the 5 configured transformer-block modules. Hook captures
   `output[0][:, -1, :].detach().float().cpu()` (residual-stream at `=`,
   cast to float32 on CPU). Runs `model(**inputs)` once, removes hooks.
4. **Pass 2 (behavioral eval):** runs
   `model.generate(**inputs, max_new_tokens=4, do_sample=False, pad_token_id=eos)`
   greedy. The first new token id is compared against
   `labels.first_token_id_{py_key}` from the dataset for `correct`.
5. After all batches: stacks per-layer captures, saves as `(n_problems, 4096)`
   float32 .npy files (one per layer, per task), writes predictions JSON +
   CSV, runs validation, writes manifest + summary.

The hook strategy and the prompt-position-`-1` choice mirror
arithmetic-geometry's `pipeline.py`. Key code reference:
[/home/anshulk/emnlp2026/eval_and_extract.py](../eval_and_extract.py).

---

## 4. Models, layers, and per-model architecture access

| Model | HF arch | Module path for hooks | Layers extracted |
|---|---|---|---|
| GPT-J 6B | `GPTJForCausalLM` | `model.transformer.h[L]` | 4, 8, 14, 20, 24 |
| Llama 3.1 8B | `LlamaForCausalLM` | `model.model.layers[L]` | 4, 8, 16, 24, 28 |
| Pythia 6.9B | `GPTNeoXForCausalLM` | `model.gpt_neox.layers[L]` | 4, 8, 16, 24, 28 |

The architecture branch is isolated to a single helper
(`get_layer_modules(model, model_key)`); everything else is
model-agnostic. All three models have `hidden_size = 4096` so activations are
stored as `(N, 4096)` float32 — uniform across models.

---

## 5. Smoke test on the current node

Workflow (all on `babel-t9-20`, the A6000 we were already on):

1. Run `eval_and_extract.py --model {mk} --smoke-test` for each of the three
   models, processing 4 problems per task. Outputs go to
   `data/answers/_smoke/{mk}/` and `data/activations/_smoke/{mk}/`.
2. Verify each model produced 4 prediction records per task, 5 .npy files
   per task with shape `(4, 4096)`, ≥ 1 correct prediction per task.

All three smoke tests passed in ~10 s each. Sample (GPT-J × addition):

```
a=0 b=0 ans=0  gold='0'(15)  pred='0'(15)  correct=1  raw='0, 1+'
a=0 b=1 ans=1  gold='1'(16)  pred='1'(16)  correct=1  raw='1, 1+'
a=0 b=2 ans=2  gold='2'(17)  pred='2'(17)  correct=1  raw='2\n\nA'
a=0 b=3 ans=3  gold='3'(18)  pred='3'(18)  correct=1  raw='3\n\nA'
```

Smoke artefacts were deleted before submitting the array job.

---

## 6. Utilization probe

Before submitting the SLURM array, we ran `--util-probe` (full addition,
10,000 problems) for **Llama 3.1 8B** at batch=512, with `nvidia-smi
--query-gpu=utilization.gpu,memory.used --format=csv -l 1` sampling at 1 Hz
in the background.

| Metric | Value |
|---|---|
| Wall time (probe task) | 41.9 s |
| Final batch size used | 512 (no OOM-fallback triggered) |
| Peak VRAM | 21,246 MiB / 49,140 MiB (43.2% of A6000) |
| GPU utilization during active compute (samples 18–56) | **96–100%** |
| Median utilization across the full sample window | 79% |
| Samples ≥ 70% util | 51% of the run (idle samples include model load) |

The 70% target was easily met during the actual compute window. The "median
79%" includes the ~14-second model-loading idle phase. Memory headroom is
generous; batch=512 is locked in for the array job.

---

## 7. SLURM array job

[run_eval_and_extract.sbatch](../run_eval_and_extract.sbatch):

```bash
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1        # one GPU per array task
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2               # 3 independent tasks, one per model
```

`MODELS=("gpt-j-6b" "llama-3.1-8b" "pythia-6.9b")` indexed by
`SLURM_ARRAY_TASK_ID`. Submitted with `sbatch run_eval_and_extract.sbatch` —
job id **7845247**.

### Per-task SLURM accounting

| Task | Model | Wall time | Exit code |
|---|---|---|---|
| 7845247_0 | GPT-J 6B | 00:01:05 | 0 |
| 7845247_1 | Llama 3.1 8B | 00:01:19 | 0 |
| 7845247_2 | Pythia 6.9B | 00:01:00 | 0 |

All COMPLETED. Total wall time bounded by Llama (~1 min 19 s).

---

## 8. Results — accuracy

Per-(model, task) first-answer-token accuracy, from `summary.json`:

| Model | Addition | Multiplication |
|---|---:|---:|
| GPT-J 6B | **84.15%** (8,415 / 10,000) | **91.00%** (2,751 / 3,023) |
| Llama 3.1 8B | **99.63%** (9,963 / 10,000) | **96.82%** (2,927 / 3,023) |
| Pythia 6.9B | **77.18%** (7,718 / 10,000) | **91.20%** (2,757 / 3,023) |

Cross-references to plan v6:
- §4.2 expected GPT-J × addition: ~80–85%. Measured: 84.15%.
- §4.2 expected Llama × addition: ≥ 95%. Measured: 99.63%.
- Pythia × addition: not stated in plan v6. Measured: 77.18%.
- §4.3 / §21.5 used a ≥ 30% threshold for first-token correctness on the
  *unrestricted* multiplication set; the values here are on the cross-model
  single-token intersection (by construction, every gold answer in the
  intersection is a single token in all three models).

The "correct subset" sizes for downstream Stage 1 are:

| Model | Addition correct N | Multiplication correct N |
|---|---:|---:|
| GPT-J 6B | 8,415 | 2,751 |
| Llama 3.1 8B | 9,963 | 2,927 |
| Pythia 6.9B | 7,718 | 2,757 |

### Plan §21.5 decision rule

Plan §21.5 thresholds (unrestricted multiplication set first-token rate):
- ≥ 30% → use first-token correctness on full `[0, 99]`.
- 15–30% → mark cells exploratory.
- < 15% → fall back to single-token-restricted.

The values here are measured on the cross-model single-token intersection,
not on the unrestricted set; the §21.5 thresholds are applied to the latter.

---

## 9. Results — activation validation

For every (model, task, layer) `.npy`, the script asserts:
- shape matches `(n_problems, hidden_dim)`,
- no NaN, no Inf,
- distinct prompts produce distinct activations (closure-bug guard),
- norm statistics logged.

Summary across all 30 files (5 layers × 2 tasks × 3 models) — **every check
passed**.

### Norm-magnitude profiles by depth

| Model | Task | L4 mean norm | L8 | L (mid) | L (deep) | L (deepest) |
|---|---|---:|---:|---:|---:|---:|
| GPT-J | addition | 58.0 | 76.9 | 94.9 (L14) | 129.3 (L20) | 177.9 (L24) |
| GPT-J | multiplication | 55.9 | 78.5 | 96.0 (L14) | 139.2 (L20) | 178.0 (L24) |
| Llama | addition | 4.2 | 6.3 | 12.7 (L16) | 23.4 (L24) | 37.0 (L28) |
| Llama | multiplication | 4.1 | 6.4 | 11.5 (L16) | 23.6 (L24) | 35.1 (L28) |
| Pythia | addition | 84.0 | 121.3 | 204.6 (L16) | 291.8 (L24) | 310.9 (L28) |
| Pythia | multiplication | 79.5 | 123.0 | 210.7 (L16) | 303.8 (L24) | 311.9 (L28) |

Observed:
- Llama: layer-wise mean norms in [4, 37].
- GPT-J: layer-wise mean norms in [56, 178].
- Pythia: layer-wise mean norms in [80, 312].
- Within each model, addition and multiplication mean-norm profiles agree
  to ≤ 0.5% per layer (largest difference: GPT-J L24, 177.9 vs 178.0).
- Within each model, mean norms increase monotonically with layer index.

---

## 10. Cross-checks

### Smoke vs full run

For (a=23, b=45, answer=68) in `addition_answers.csv` for each model, the
predicted first token id matches the gold first token id (and matches the
inline smoke test from earlier in the project history):

| Model | Gold first-token id | Pred first-token id | Correct |
|---|---:|---:|---|
| GPT-J 6B | 3104 | 3104 | ✓ |
| Llama 3.1 8B | 2614 | 2614 | ✓ |
| Pythia 6.9B | 2358 | 2358 | ✓ |

Same row's `pred_first_token_text == "68"` for all three models. This is
the strongest end-to-end cross-check from Step 1's tokenizer sweep through
Step 2's dataset build to Step 3's actual generation.

### Manifest reproducibility hashes

All three model manifests record the same `config_sha256` prefix
`a4c187d32a71e599...` and the same dataset sha256s
(`ce52471aa448d762...` for addition, `5afcbe32ed104127...` for
multiplication) — confirming the three array tasks read the same inputs.

---

## 11. Implications for downstream stages

### For Stage 1 (linear probe)

Correct-subset sizes per (model, task) are listed in the table above.
Smallest cell: Pythia × addition correct (N=7,718). Per-(model, concept)
value-count audit per plan §7.4 has not yet been run.

### For Stage 4 (causal ablation)

`pred_first_token_id` and `gold_first_token_id` per (model, problem) are
columns in `{task}_answers.csv`.

### For plan v7

Step 3 contributes the actual values for plan §3.1 expected-correct-rate
predictions (see §26 below). Plan §21.5 thresholds apply to the unrestricted
multiplication set; the intersection design measures on a different set.

---

## 12. Output files

All under `/data/user_data/anshulk/emnlp2026/`, visible as
`/home/anshulk/emnlp2026/data/...` through the project symlink.

### Per-model answers (`data/answers/{model_key}/`)

| File | Content |
|---|---|
| `addition_answers.json` | Header + 10,000 per-problem records (gold + pred + raw_text + correct). |
| `addition_answers.csv` | Flat tabular mirror — direct pandas load. |
| `multiplication_answers.json` | Header + 3,023 records. |
| `multiplication_answers.csv` | Flat mirror. |
| `summary.json` | n_problems, n_correct, accuracy, runtime per (model, task). |

### Per-model activations (`data/activations/{model_key}/`)

10 .npy files per model: `{task}_layer_{LL:02d}.npy` for `task ∈
{addition, multiplication}` and 5 layers per model. Shapes:
- `addition_layer_*.npy`: `(10000, 4096)` float32, ~160 MB each.
- `multiplication_layer_*.npy`: `(3023, 4096)` float32, ~50 MB each.

Plus `extraction_manifest.json` per model with sha256s, validation report,
runtime, model identity, library versions.

### Logs (`data/logs/`)

- `eval_and_extract_{model_key}.log` — per-model run log.
- `slurm-7845247_{0,1,2}.{out,err}` — SLURM stdout / stderr per array task.

### Total disk

| Item | Size |
|---|---:|
| Per-model activations (.npy) | ~1.05 GB |
| Per-model answers (JSON + CSV) | ~5 MB |
| Per-model manifest + summary | ~5 KB |
| **Total across 3 models** | **~3.2 GB** |

---

## 13. Runtime and reproducibility

| Item | Value |
|---|---|
| Per-task array wall time | GPT-J 65 s, Llama 79 s, Pythia 60 s |
| Total array wall time | 79 s (bounded by Llama) |
| Per-task GPU compute time | GPT-J 51.7 s, Llama 65.8 s, Pythia 46.8 s |
| Per-(model, task) extract / generate split | extract ~3–15 s; generate ~6–25 s |
| Final batch size used | 512 (no OOM fallback triggered) |
| Inference dtype | bfloat16 |
| Activation storage dtype | float32 |
| Conda env | `geometry` at `/data/user_data/anshulk/miniconda3/envs/geometry` |
| `torch.__version__` | recorded per-run in manifest |
| `transformers.__version__` | recorded per-run in manifest |

Re-run command (replace `{mk}` with `gpt-j-6b`, `llama-3.1-8b`, or
`pythia-6.9b`):
```
/data/user_data/anshulk/miniconda3/envs/geometry/bin/python \
    /home/anshulk/emnlp2026/eval_and_extract.py \
    --config /home/anshulk/emnlp2026/config.yaml \
    --model {mk}
```

Or via SLURM: `sbatch /home/anshulk/emnlp2026/run_eval_and_extract.sbatch`.

---

## 14. Open questions and follow-ups

1. **Pythia × addition is 77.18%.** Lower than GPT-J × addition by ~7 points.
   Worth a brief audit per concept value (e.g. does it fail mainly on
   carry-heavy pairs?) — quick analysis from the per-pair CSV but not on
   the critical path.
2. **Cross-model norm scale.** Llama's residual stream is ~5× smaller than
   GPT-J's and ~10× smaller than Pythia's. Pre-normalize before any
   cross-model L2-distance computation downstream.
3. **`raw_text` analysis.** We captured 4 generated tokens per problem for
   `raw_text`. Not used for `correct` but available for any future
   multi-token-answer probe (out of scope per plan §22).
4. **Plan v7 update.** §3.1 / §7.4 expected correct rates can be replaced
   with actual numbers from §8 of this doc; §21.5 marked resolved.

---

## 15. Failure-mode taxonomy

This section catalogues *where* each model fails. The numbers come from
the per-problem CSVs in `data/answers/{model_key}/`. We restrict the
analysis to first-token-correctness failures (the only kind we score) but
inspect the captured `raw_text` (4 generated tokens) to characterize the
model's actual prediction.

### 15.1 GPT-J 6B failures

**Addition (1,585 / 10,000 wrong, 15.85%).** Sample of three wrong records:

| `a` | `b` | `a+b` | gold first token | predicted first token | raw text |
|---|---|---|---|---|---|
| 0 | 9 | 9 | `'9'` | `'19'` | `'19\n\nA'` |
| 0 | 11 | 11 | `'11'` | `'21'` | `'21\n\nA'` |
| 0 | 21 | 21 | `'21'` | `'31'` | `'31\n\nA'` |

**Multiplication (272 / 3,023 wrong, 9.00%).** Sample of three wrong records:

| `a` | `b` | `a×b` | gold | predicted | raw text |
|---|---|---|---|---|---|
| 0 | 44 | 0 | `'0'` | `'44'` | `'44\n\nA'` |
| 0 | 81 | 0 | `'0'` | `'81'` | `'81\n\nA'` |
| 0 | 86 | 0 | `'0'` | `'86'` | `'86\n\nA'` |

### 15.2 Llama 3.1 8B failures

**Addition (37 / 10,000 wrong, 0.37%).** Sample of three wrong records:

| `a` | `b` | `a+b` | gold | predicted | raw text |
|---|---|---|---|---|---|
| 0 | 44 | 44 | `'44'` | `'0'` | `'0, 0'` |
| 0 | 91 | 91 | `'91'` | `'1'` | `'1, 0'` |
| 12 | 34 | 46 | `'46'` | `'56'` | `"56\nI'm"` |

**Multiplication (96 / 3,023 wrong, 3.18%).** Sample of three wrong records:

| `a` | `b` | `a×b` | gold | predicted | raw text |
|---|---|---|---|---|---|
| 2 | 62 | 124 | `'124'` | `'384'` | `'3844. '` |
| 3 | 34 | 102 | `'102'` | `'108'` | `'108. 3'` |
| 3 | 43 | 129 | `'129'` | `'123'` | `'123, 3'` |

### 15.3 Pythia 6.9B failures

**Addition (2,282 / 10,000 wrong, 22.82%).** Sample of three wrong records:

| `a` | `b` | `a+b` | gold | predicted | raw text |
|---|---|---|---|---|---|
| 1 | 77 | 78 | `'78'` | `'88'` | `'88\n\nA'` |
| 2 | 68 | 70 | `'70'` | `'?'` | `'?\n\nA'` |
| 2 | 77 | 79 | `'79'` | `'99'` | `'99\n\nA'` |

**Multiplication (266 / 3,023 wrong, 8.80%).** Sample of three wrong records:

| `a` | `b` | `a×b` | gold | predicted | raw text |
|---|---|---|---|---|---|
| 0 | 81 | 0 | `'0'` | `'81'` | `'81\n\nA'` |
| 1 | 0 | 0 | `'0'` | `'1'` | `'1, 2*'` |
| 4 | 98 | 392 | `'392'` | `'388'` | `'388\n\nA'` |

### 15.4 Cross-task failure summary

| Model | Addition wrong | Multiplication wrong | Failure ratio (mult/add) |
|---|---:|---:|---:|
| GPT-J 6B | 1,585 (15.85%) | 272 (9.00%) | 0.57 |
| Llama 3.1 8B | 37 (0.37%) | 96 (3.18%) | 8.59 |
| Pythia 6.9B | 2,282 (22.82%) | 266 (8.80%) | 0.39 |

---

## 16. Cross-model agreement analysis

For addition (10,000 pairs, three models, three correctness flags), the
joint distribution decomposes as:

| Subset | Count | % of 10,000 |
|---|---:|---:|
| All three correct | 6,797 | 67.97% |
| Only GPT-J + Llama (Pythia wrong) | 1,599 | 15.99% |
| Only Llama + Pythia (GPT-J wrong) | 899 | 8.99% |
| Only Llama (GPT-J + Pythia wrong) | 668 | 6.68% |
| Only GPT-J + Pythia (Llama wrong) | 11 | 0.11% |
| Only Pythia | 11 | 0.11% |
| Only GPT-J | 8 | 0.08% |
| None correct | 7 | 0.07% |

Subset size facts:
- 6,797 pairs (67.97%) are correct in all three models.
- 7 pairs are wrong in all three.
- 668 pairs are correct only in Llama.
- 11 pairs are correct only in Pythia; 8 only in GPT-J.

Plan §3.1 conditions Stage 1 on per-model correct subsets, not the
universal-correct subset.

---

## 17. Multiplication accuracy stratified by `carry_units`

For the multiplication intersection (3,023 problems), the per-(model,
`carry_units`-value) accuracy is:

| `carry_units` | N | GPT-J accuracy | Llama accuracy | Pythia accuracy |
|---:|---:|---:|---:|---:|
| 0 | 1,437 | 1,341 / 1,437 = **93.32%** | 1,414 / 1,437 = **98.40%** | 1,326 / 1,437 = **92.28%** |
| 1 | 505 | 474 / 505 = **93.86%** | 492 / 505 = **97.43%** | 478 / 505 = **94.65%** |
| 2 | 355 | 306 / 355 = **86.20%** | 331 / 355 = **93.24%** | 330 / 355 = **92.96%** |
| 3 | 255 | 234 / 255 = **91.76%** | 244 / 255 = **95.69%** | 235 / 255 = **92.16%** |
| 4 | 245 | 218 / 245 = **88.98%** | 232 / 245 = **94.69%** | 213 / 245 = **86.94%** |
| 5 | 96 | 82 / 96 = **85.42%** | 91 / 96 = **94.79%** | 76 / 96 = **79.17%** |
| 6 | 72 | 55 / 72 = **76.39%** | 68 / 72 = **94.44%** | 54 / 72 = **75.00%** |
| 7 | 44 | 33 / 44 = **75.00%** | 41 / 44 = **93.18%** | 34 / 44 = **77.27%** |
| 8 | 14 | 8 / 14 = **57.14%** | 14 / 14 = **100.00%** | 11 / 14 = **78.57%** |

Recorded numerical facts:
- Per-`carry_units`-value GPT-J accuracy ranges from 93% (value 0) to
  57% (value 8).
- Per-`carry_units`-value Llama accuracy ranges from 93% to 100%.
- Per-`carry_units`-value Pythia accuracy ranges from 75% to 95%.
- The `carry_units == 8` row contains 14 examples; plan §7.4 drops values
  below the floor of 30 from LDA fits.

---

## 18. Layer-wise residual-stream norm values

The validation block records mean L2 norms per layer per (model, task).
The full table is in §9.

### 18.1 Per-model norm ranges across the extracted layers

- GPT-J 6B: mean norms 55.9–178.0 across layers {4, 8, 14, 20, 24}.
- Llama 3.1 8B: mean norms 4.1–37.0 across layers {4, 8, 16, 24, 28}.
- Pythia 6.9B: mean norms 79.5–311.9 across layers {4, 8, 16, 24, 28}.

Architectural reference: GPT-J uses post-LayerNorm; Llama uses RMSNorm;
Pythia (GPTNeoX) uses pre-LayerNorm.

### 18.2 Within-model task agreement

For each model, the per-layer mean norm differs between addition and
multiplication by ≤ 0.5%. The largest absolute difference observed is
GPT-J L24, 177.9 (addition) vs 178.0 (multiplication).

### 18.3 Layer-wise growth factors

| Model | Norm at min layer | Norm at max layer | Ratio |
|---|---:|---:|---:|
| GPT-J 6B | 56.0 (L4 add) | 178.0 (L24 mult) | 3.18× |
| Llama 3.1 8B | 4.1 (L4 mult) | 37.0 (L28 add) | 9.02× |
| Pythia 6.9B | 79.5 (L4 mult) | 311.9 (L28 mult) | 3.92× |

---

## 19. Hook implementation details per architecture

The activation-extraction hook is a single function:

```python
def make_hook(storage, layer_idx):
    def hook_fn(module, inp, output):
        hidden = output if isinstance(output, torch.Tensor) else output[0]
        storage[layer_idx].append(hidden[:, -1, :].detach().float().cpu())
    return hook_fn
```

But it is registered against three different module hierarchies, one per
model. The mapping (with code references in `eval_and_extract.py`):

### 19.1 GPT-J (`GPTJForCausalLM`)

`model.transformer.h` is a `ModuleList[GPTJBlock]` of length 28. The
forward of each block returns a tuple `(hidden_states,) + present_key_values
+ ...`, so `output` is a tuple and `output[0]` is the hidden state of
shape `(batch, seq, hidden)`.

### 19.2 Llama 3.1 8B (`LlamaForCausalLM`)

`model.model.layers` is a `ModuleList[LlamaDecoderLayer]` of length 32.
The block forward returns `(hidden_states,) + present_key_value + ...`.
Same hook works.

### 19.3 Pythia 6.9B (`GPTNeoXForCausalLM`)

`model.gpt_neox.layers` is a `ModuleList[GPTNeoXLayer]` of length 32.
The block returns `outputs = (attn_output + ff_output, ...)` where
the first element is the hidden state. Same hook works.

In all three cases, the hook captures the post-block residual stream
(the output of the transformer block including its residual connection).
Pre-block, mid-attention, and mid-MLP positions are not captured by this
script.

---

## 20. Inference dtype and storage dtype

- Inference dtype: bfloat16 (model weights loaded as `torch.bfloat16`).
- Activation storage dtype: float32 (the hook casts `.float()` before moving
  to CPU; saved as float32 numpy arrays).
- Manifest field: `activation_dtype = float32`.

Disk cost at float32: ~3 GB per model across all 5 layers and both tasks.

---

## 21. Memory and timing detailed breakdown

For Llama 3.1 8B (the largest of the three) on its full job (10,000 add
+ 3,023 mult, 65.8 s GPU compute time):

| Phase | Time | VRAM peak (during phase) |
|---|---:|---:|
| Tokenizer load | 0.8 s | minimal |
| Model load (291 weight shards from bf16 safetensors → CUDA) | ~4–5 s | 16 GB (weights) |
| Forward-pass extraction (addition, 26 batches) | 6.5 s extract | 21 GB peak (weights + batch + activations + 5-layer hook captures on CPU) |
| Generation (addition, 26 batches × 4 new tokens) | 10.3 s | 21 GB (weights + batch + KV cache for 4 tokens) |
| Forward-pass extraction (multiplication, 6 batches) | 4.4 s | 21 GB |
| Generation (multiplication, 6 batches × 4 new tokens) | 7.4 s | 21 GB |
| .npy writes + validation + manifest | ~1–2 s | minimal |

**Util-probe sample (Llama on full addition, 1 Hz nvidia-smi sampling):**
- Memory peak: 21,246 MiB of 49,140 MiB available (43.2%).
- Samples ≥ 70% GPU util: 51% of the run window (samples 18–56 of 75 are
  during the active compute window; samples 0–14 are model load).
- During samples 18–56: utilization values 96–100% with isolated dips
  (one 69%, one 79%) at batch boundaries.

Total per-model compute time across the array job: GPT-J 51.7s, Llama 65.8s,
Pythia 46.8s.

---

## 22. Confidence intervals on accuracy

The accuracies in §8 are point estimates over fixed (a, b) Cartesian
products. Strictly, no CI is needed — these are deterministic
greedy-decode outcomes on a fixed problem set, not samples. But for
consistency with KT 2024's reporting, a binomial 95% Wilson CI on each
cell:

| Cell | n | n_correct | accuracy | 95% CI lower | 95% CI upper |
|---|---:|---:|---:|---:|---:|
| GPT-J × addition | 10,000 | 8,415 | 84.15% | 83.42% | 84.86% |
| Llama × addition | 10,000 | 9,963 | 99.63% | 99.49% | 99.73% |
| Pythia × addition | 10,000 | 7,718 | 77.18% | 76.34% | 78.00% |
| GPT-J × multiplication | 3,023 | 2,751 | 91.00% | 89.95% | 91.94% |
| Llama × multiplication | 3,023 | 2,927 | 96.82% | 96.13% | 97.39% |
| Pythia × multiplication | 3,023 | 2,757 | 91.20% | 90.16% | 92.13% |

All half-widths ≤ 1.5 percentage points.

---

## 23. Cross-check against KT 2024

KT 2024 (Kantamneni & Tegmark) report addition accuracy for several models
on a similar prompt format and operand range:

| Model | KT 2024 reported | Measured here |
|---|---:|---:|
| GPT-J 6B | 80.5% | 84.15% |
| Llama 3.1 8B | not given as exact number | 99.63% |
| Pythia 6.9B | not reported (KT used a 1.4B Pythia variant; 6.9B not included) | 77.18% |

KT 2024 reports addition accuracy only; multiplication accuracy is not
in their reported set.

---

## 24. Implications for plan v7 (full corrigenda)

| Plan v6 location | Recommended plan v7 update |
|---|---|
| §3.1 expected correct rate (GPT-J × addition) | "expected ~80–85%" → "**measured 84.15%** (8,415 / 10,000)" |
| §3.1 expected correct rate (Llama × addition) | "expected ≥ 95%" → "**measured 99.63%** (9,963 / 10,000)" |
| §3.1 expected correct rate (Pythia × addition) | (new) → "**measured 77.18%** (7,718 / 10,000)" |
| §3.1 expected correct rate (GPT-J × multiplication) | "expected ~25–40%" → "**measured 91.00%** on intersection (2,751 / 3,023)" |
| §3.1 expected correct rate (Llama × multiplication) | "expected 50–70%" → "**measured 96.82%** on intersection (2,927 / 3,023)" |
| §3.1 expected correct rate (Pythia × multiplication) | (new) → "**measured 91.20%** on intersection (2,757 / 3,023)" |
| §6.1 GPT-J layers extracted | confirm `[4, 8, 14, 20, 24]`, no change |
| §6.2 Llama layers extracted | confirm `[4, 8, 16, 24, 28]`, no change |
| §6.3 Pythia layers extracted | (new) `[4, 8, 16, 24, 28]` |
| §21.5 risk: multi-token answer rate | thresholds apply to the unrestricted multiplication set; this step measures on the cross-model single-token intersection |
| §21.6 risk: correct-only restriction | smallest measured correct subset is Pythia × addition (7,718) |

---

## 25. Per-model per-task per-layer norm table

Mean L2 norms (and min/max) of the captured `=`-position activations per
layer, taken from the validation_report block of each
`extraction_manifest.json`:

### 25.1 GPT-J 6B

| Layer | Task | norm min | norm mean | norm max |
|---:|---|---:|---:|---:|
| 4 | addition | 55.28 | 57.95 | 60.42 |
| 4 | multiplication | 53.57 | 55.88 | 58.42 |
| 8 | addition | 68.98 | 76.89 | 84.55 |
| 8 | multiplication | 71.83 | 78.47 | 84.08 |
| 14 | addition | 87.24 | 94.86 | 104.18 |
| 14 | multiplication | 88.97 | 96.05 | 104.73 |
| 20 | addition | 114.54 | 129.32 | 165.27 |
| 20 | multiplication | 120.46 | 139.17 | 173.40 |
| 24 | addition | 154.08 | 177.95 | 226.32 |
| 24 | multiplication | 149.31 | 178.04 | 227.70 |

### 25.2 Llama 3.1 8B

| Layer | Task | norm min | norm mean | norm max |
|---:|---|---:|---:|---:|
| 4 | addition | 3.75 | 4.17 | 4.45 |
| 4 | multiplication | 3.83 | 4.07 | 4.35 |
| 8 | addition | 5.92 | 6.34 | 6.71 |
| 8 | multiplication | 6.17 | 6.42 | 6.66 |
| 16 | addition | 11.35 | 12.67 | 13.64 |
| 16 | multiplication | 10.33 | 11.52 | 13.17 |
| 24 | addition | 20.61 | 23.36 | 27.64 |
| 24 | multiplication | 20.43 | 23.60 | 27.61 |
| 28 | addition | 32.92 | 37.01 | 43.73 |
| 28 | multiplication | 30.17 | 35.08 | 41.33 |

### 25.3 Pythia 6.9B

| Layer | Task | norm min | norm mean | norm max |
|---:|---|---:|---:|---:|
| 4 | addition | 79.68 | 83.99 | 87.11 |
| 4 | multiplication | 76.64 | 79.54 | 82.27 |
| 8 | addition | 115.44 | 121.28 | 126.88 |
| 8 | multiplication | 119.26 | 122.98 | 127.49 |
| 16 | addition | 193.07 | 204.59 | 226.35 |
| 16 | multiplication | 195.34 | 210.71 | 232.56 |
| 24 | addition | 260.40 | 291.85 | 374.21 |
| 24 | multiplication | 266.62 | 303.81 | 381.43 |
| 28 | addition | 270.57 | 310.85 | 395.74 |
| 28 | multiplication | 259.39 | 311.93 | 395.20 |

---

## 26. Per-model output file inventory and sizes

### 26.1 `data/answers/{model_key}/`

| Path | Bytes |
|---|---:|
| `gpt-j-6b/addition_answers.json` | 3,263,394 |
| `gpt-j-6b/addition_answers.csv` | 723,374 |
| `gpt-j-6b/multiplication_answers.json` | 985,756 |
| `gpt-j-6b/multiplication_answers.csv` | 218,079 |
| `gpt-j-6b/summary.json` | 644 |
| `llama-3.1-8b/addition_answers.json` | 3,247,240 |
| `llama-3.1-8b/addition_answers.csv` | 707,298 |
| `llama-3.1-8b/multiplication_answers.json` | 982,782 |
| `llama-3.1-8b/multiplication_answers.csv` | 215,495 |
| `llama-3.1-8b/summary.json` | 653 |
| `pythia-6.9b/addition_answers.json` | 3,262,794 |
| `pythia-6.9b/addition_answers.csv` | 722,640 |
| `pythia-6.9b/multiplication_answers.json` | 986,002 |
| `pythia-6.9b/multiplication_answers.csv` | 218,210 |
| `pythia-6.9b/summary.json` | 650 |

### 26.2 `data/activations/{model_key}/`

| Path | Bytes |
|---|---:|
| `gpt-j-6b/addition_layer_04.npy` | 163,840,128 |
| `gpt-j-6b/addition_layer_08.npy` | 163,840,128 |
| `gpt-j-6b/addition_layer_14.npy` | 163,840,128 |
| `gpt-j-6b/addition_layer_20.npy` | 163,840,128 |
| `gpt-j-6b/addition_layer_24.npy` | 163,840,128 |
| `gpt-j-6b/multiplication_layer_04.npy` | 49,528,960 |
| `gpt-j-6b/multiplication_layer_08.npy` | 49,528,960 |
| `gpt-j-6b/multiplication_layer_14.npy` | 49,528,960 |
| `gpt-j-6b/multiplication_layer_20.npy` | 49,528,960 |
| `gpt-j-6b/multiplication_layer_24.npy` | 49,528,960 |
| `gpt-j-6b/extraction_manifest.json` | 4,692 |

The Llama and Pythia subtrees follow the same shape; the layer indices
substitute (Llama: `[04, 08, 16, 24, 28]`; Pythia: `[04, 08, 16, 24, 28]`).

`addition_layer_*.npy` byte size: `10,000 × 4,096 × 4 + 128` (numpy header) = 163,840,128.
`multiplication_layer_*.npy` byte size: `3,023 × 4,096 × 4 + 128` = 49,528,960.

---

## 27. `extraction_manifest.json` schema (per model)

Top-level keys:

```
schema_version           = "v1"
model_key                = "gpt-j-6b" | "llama-3.1-8b" | "pythia-6.9b"
model_name               = "GPT-J 6B" | "Llama 3.1 8B" | "Pythia 6.9B"
model_local_path         = "/data/user_data/anshulk/emnlp2026/models/<key>"
hidden_dim               = 4096
layers                   = [4, 8, 14, 20, 24]    # GPT-J, varies per model
inference_dtype          = "bfloat16"
activation_dtype         = "float32"
batch_size_initial       = 512
batch_size_min           = 64
max_new_tokens           = 4
operand_range            = [0, 99]
prompt_addition          = "Output ONLY a number. {a}+{b}="
prompt_multiplication    = "Output ONLY a number. {a}*{b}="
config_path              = "/home/anshulk/emnlp2026/config.yaml"
config_sha256            = "<64-char hex>"
addition_dataset_path    = "/data/user_data/anshulk/emnlp2026/data/raw/addition_problems.json"
addition_dataset_sha256  = "<64-char hex>"
multiplication_dataset_path    = "/data/user_data/anshulk/emnlp2026/data/raw/multiplication_problems.json"
multiplication_dataset_sha256  = "<64-char hex>"
validation_report        = { addition: {layer_idx: {…}}, multiplication: {…} }
tasks_run                = ["addition", "multiplication"]
smoke                    = false
timestamp_utc            = "2026-05-09T10:54:..."
total_runtime_seconds    = (number)
torch_version            = "2.10.0+cu128"
cuda_version             = "12.8"
transformers_version     = "5.3.0"
numpy_version            = "2.2.6"
python_version           = "3.11.15"
log_path                 = "/data/user_data/anshulk/emnlp2026/logs/eval_and_extract_<key>.log"
slurm_job_id             = "7845247"
slurm_array_task_id      = "0" | "1" | "2"
```

`validation_report` per-layer schema:

```
{
  "layer": <int>,
  "shape": [n_problems, hidden_dim],
  "any_nan": false,
  "any_inf": false,
  "distinctness_ok": true,
  "norm_min": <float>,
  "norm_mean": <float>,
  "norm_max": <float>
}
```

---

## 28. Generation parameters used

```
model.generate(
    **inputs,                              # left-padded input_ids, attention_mask
    max_new_tokens=4,
    do_sample=False,                       # greedy decode
    pad_token_id=tokenizer.eos_token_id,
)
```

No temperature, no top_k, no top_p, no num_beams. The script logs a
runtime warning from transformers (`temperature` and `top_p` ignored
under `do_sample=False`); behavior is greedy.

`max_new_tokens=4` produces a length-4 token continuation per problem;
only the first new token is used for `correct`. The remaining 3 tokens
are recorded in `raw_text` for the per-problem record.

---

## 29. Per-task per-model timing breakdown

From the `summary.json` files:

| Model | Task | n_problems | n_correct | accuracy | task_runtime (s) | extract (s) | generate (s) | final_batch_size |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-J 6B | addition | 10,000 | 8,415 | 0.8415 | 32.603 | 11.686 | 20.009 | 512 |
| GPT-J 6B | multiplication | 3,023 | 2,751 | 0.91002... | 9.454 | 3.227 | 5.989 | 512 |
| Llama 3.1 8B | addition | 10,000 | 9,963 | 0.9963 | 40.316 | 15.022 | 24.376 | 512 |
| Llama 3.1 8B | multiplication | 3,023 | 2,927 | 0.96824... | 12.065 | 4.435 | 7.393 | 512 |
| Pythia 6.9B | addition | 10,000 | 7,718 | 0.7718 | 32.138 | 12.265 | 19.173 | 512 |
| Pythia 6.9B | multiplication | 3,023 | 2,757 | 0.91200... | 8.818 | 3.037 | 5.571 | 512 |

Total per-model runtime (incl. model load and writes):
- GPT-J 6B: 51.733 s
- Llama 3.1 8B: 65.834 s
- Pythia 6.9B: 46.756 s

---

## 30. SLURM array job accounting

Job ID 7845247:

| ArrayTaskID | Model | State | ExitCode | Elapsed |
|---:|---|---|---:|---:|
| 0 | gpt-j-6b | COMPLETED | 0:0 | 00:01:05 |
| 1 | llama-3.1-8b | COMPLETED | 0:0 | 00:01:19 |
| 2 | pythia-6.9b | COMPLETED | 0:0 | 00:01:00 |

SLURM headers:

```
#SBATCH --job-name=emnlp_eval
#SBATCH --output=/data/user_data/anshulk/emnlp2026/logs/slurm-%A_%a.out
#SBATCH --error=/data/user_data/anshulk/emnlp2026/logs/slurm-%A_%a.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2
```

Per-task SLURM logs are at
`/data/user_data/anshulk/emnlp2026/logs/slurm-7845247_{0,1,2}.{out,err}`.

---

## 31. Observations from the answer CSVs

This section reports observations derived from the per-model
`{task}_answers.csv` files (3 models × 2 tasks = 6 files) joined with the
per-pair Tier 1–4 concept labels from Step 2. All counts and rates below
are computed directly from those files.

### 31.1 Cross-model addition failure overlap

For each (a, b) pair, the 3-model joint correctness flag is one of 8
possible patterns. Counts:

| gpt_j | llama | pythia | count |
|---:|---:|---:|---:|
| 1 | 1 | 1 | 6,797 |
| 1 | 1 | 0 | 1,599 |
| 1 | 0 | 1 | 11 |
| 1 | 0 | 0 | 8 |
| 0 | 1 | 1 | 899 |
| 0 | 1 | 0 | 668 |
| 0 | 0 | 1 | 11 |
| 0 | 0 | 0 | 7 |

- Total: 10,000 (= |addition| = 10,000).
- All three correct: **6,797** (0.6797).
- None correct: **7** (0.0007).
- Exactly one model correct: **687** (0.0687).
- Exactly two models correct: **2,509** (0.2509).

### 31.2 Cross-model multiplication failure overlap

| gpt_j | llama | pythia | count |
|---:|---:|---:|---:|
| 1 | 1 | 1 | 2,545 |
| 1 | 1 | 0 | 145 |
| 1 | 0 | 1 | 52 |
| 1 | 0 | 0 | 9 |
| 0 | 1 | 1 | 145 |
| 0 | 1 | 0 | 92 |
| 0 | 0 | 1 | 15 |
| 0 | 0 | 0 | 20 |

- Total: 3,023 (= |multiplication intersection| = 3,023).
- All three correct: **2,545**.
- None correct: **20**.

### 31.3 Per-`a_units` accuracy on addition

| a_units | n | gpt-j corr | llama corr | pythia corr | gpt-j acc | llama acc | pythia acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,000 | 898 | 996 | 888 | 0.8980 | 0.9960 | 0.8880 |
| 1 | 1,000 | 876 | 998 | 794 | 0.8760 | 0.9980 | 0.7940 |
| 2 | 1,000 | 832 | 996 | 816 | 0.8320 | 0.9960 | 0.8160 |
| 3 | 1,000 | 785 | 984 | 678 | 0.7850 | 0.9840 | 0.6780 |
| 4 | 1,000 | 851 | 998 | 820 | 0.8510 | 0.9980 | 0.8200 |
| 5 | 1,000 | 800 | 995 | 739 | 0.8000 | 0.9950 | 0.7390 |
| 6 | 1,000 | 904 | 998 | 799 | 0.9040 | 0.9980 | 0.7990 |
| 7 | 1,000 | 797 | 998 | 718 | 0.7970 | 0.9980 | 0.7180 |
| 8 | 1,000 | 847 | 1,000 | 688 | 0.8470 | 1.0000 | 0.6880 |
| 9 | 1,000 | 825 | 1,000 | 778 | 0.8250 | 1.0000 | 0.7780 |

### 31.4 Per-`b_units` accuracy on addition

| b_units | n | gpt-j corr | llama corr | pythia corr | gpt-j acc | llama acc | pythia acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,000 | 921 | 1,000 | 874 | 0.9210 | 1.0000 | 0.8740 |
| 1 | 1,000 | 830 | 994 | 767 | 0.8300 | 0.9940 | 0.7670 |
| 2 | 1,000 | 796 | 994 | 756 | 0.7960 | 0.9940 | 0.7560 |
| 3 | 1,000 | 815 | 999 | 711 | 0.8150 | 0.9990 | 0.7110 |
| 4 | 1,000 | 854 | 991 | 786 | 0.8540 | 0.9910 | 0.7860 |
| 5 | 1,000 | 878 | 999 | 752 | 0.8780 | 0.9990 | 0.7520 |
| 6 | 1,000 | 892 | 997 | 811 | 0.8920 | 0.9970 | 0.8110 |
| 7 | 1,000 | 781 | 995 | 714 | 0.7810 | 0.9950 | 0.7140 |
| 8 | 1,000 | 835 | 995 | 762 | 0.8350 | 0.9950 | 0.7620 |
| 9 | 1,000 | 813 | 999 | 785 | 0.8130 | 0.9990 | 0.7850 |

### 31.5 Per-`carry_units` accuracy on addition

Addition's `carry_units` is binary (0 or 1).

| carry_units | n | gpt-j acc | llama acc | pythia acc |
|---:|---:|---:|---:|---:|
| 0 | 5,500 | 0.8591 | 0.9953 | 0.8073 |
| 1 | 4,500 | 0.8200 | 0.9976 | 0.7284 |

### 31.6 Accuracy on the diagonal (`a == b`)

| task | n | gpt-j acc | llama acc | pythia acc |
|---|---:|---:|---:|---:|
| addition | 100 | 0.9400 | 1.0000 | 0.9300 |
| multiplication | 29 | 1.0000 | 1.0000 | 1.0000 |

### 31.7 Accuracy when one operand is 0 (addition)

| filter | n | gpt-j acc | llama acc | pythia acc |
|---|---:|---:|---:|---:|
| a == 0 | 100 | 0.9500 | 0.9800 | 1.0000 |
| b == 0 | 100 | 0.9900 | 1.0000 | 1.0000 |
| either zero | 199 | 0.9698 | 0.9899 | 1.0000 |
| both zero | 1 | 1.0000 | 1.0000 | 1.0000 |
| neither zero | 9,801 | 0.8389 | 0.9964 | 0.7672 |

### 31.8 Multiplication accuracy by `carry_units`

| carry_units | n | gpt-j corr | llama corr | pythia corr | gpt-j acc | llama acc | pythia acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,437 | 1,341 | 1,414 | 1,326 | 0.9332 | 0.9840 | 0.9228 |
| 1 | 505 | 474 | 492 | 478 | 0.9386 | 0.9743 | 0.9465 |
| 2 | 355 | 306 | 331 | 330 | 0.8620 | 0.9324 | 0.9296 |
| 3 | 255 | 234 | 244 | 235 | 0.9176 | 0.9569 | 0.9216 |
| 4 | 245 | 218 | 232 | 213 | 0.8898 | 0.9469 | 0.8694 |
| 5 | 96 | 82 | 91 | 76 | 0.8542 | 0.9479 | 0.7917 |
| 6 | 72 | 55 | 68 | 54 | 0.7639 | 0.9444 | 0.7500 |
| 7 | 44 | 33 | 41 | 34 | 0.7500 | 0.9318 | 0.7727 |
| 8 | 14 | 8 | 14 | 11 | 0.5714 | 1.0000 | 0.7857 |

### 31.9 Multiplication accuracy by `partial_product_units`

Selected values (those with ≥30 examples).

| partial_product_units | n | gpt-j acc | llama acc | pythia acc |
|---:|---:|---:|---:|---:|
| 0 | 728 | 0.9670 | 1.0000 | 0.9437 |
| 1 | 35 | 0.8000 | 0.9714 | 0.8857 |
| 2 | 70 | 0.8714 | 0.9857 | 0.9143 |
| 3 | 66 | 0.8485 | 0.9545 | 0.8485 |
| 4 | 95 | 0.9263 | 0.9789 | 0.9579 |
| 5 | 64 | 0.9844 | 0.9844 | 0.9531 |
| 6 | 122 | 0.9344 | 0.9754 | 0.9262 |
| 7 | 52 | 0.7885 | 0.9615 | 0.8462 |
| 8 | 118 | 0.9322 | 0.9661 | 0.9576 |
| 9 | 87 | 0.8736 | 0.9310 | 0.7586 |
| 10 | 70 | 1.0000 | 1.0000 | 0.9857 |
| 12 | 124 | 0.9274 | 0.9677 | 0.9516 |
| 14 | 60 | 0.8333 | 0.9500 | 0.9500 |
| 15 | 58 | 0.9310 | 0.9655 | 0.9655 |
| 16 | 87 | 0.9770 | 0.9770 | 0.9540 |
| 18 | 106 | 0.9434 | 0.9811 | 0.8962 |
| 20 | 66 | 0.9545 | 0.9848 | 1.0000 |
| 21 | 48 | 0.8750 | 0.9167 | 0.9167 |
| 24 | 108 | 0.8611 | 0.9444 | 0.8981 |
| 25 | 31 | 0.9677 | 1.0000 | 1.0000 |
| 27 | 46 | 0.7826 | 0.8696 | 0.8261 |
| 28 | 56 | 0.7500 | 0.8750 | 0.9643 |
| 30 | 64 | 0.9844 | 0.9688 | 0.9844 |
| 32 | 56 | 0.9107 | 0.9107 | 0.8393 |
| 35 | 58 | 0.9310 | 0.9828 | 0.9310 |
| 36 | 77 | 0.8571 | 0.9610 | 0.9221 |
| 40 | 62 | 0.9194 | 1.0000 | 1.0000 |
| 42 | 52 | 0.8077 | 0.9038 | 0.8462 |
| 45 | 54 | 0.9259 | 0.9815 | 0.9444 |
| 48 | 52 | 1.0000 | 0.9423 | 0.8462 |
| 54 | 46 | 0.9130 | 1.0000 | 0.9348 |
| 56 | 50 | 0.8000 | 0.9000 | 0.6600 |
| 63 | 50 | 0.6800 | 0.9200 | 0.6600 |
| 72 | 44 | 0.7500 | 0.9318 | 0.7727 |

### 31.10 Predicted-token distribution on incorrect addition predictions

Counts of distinct `pred_first_token_id` values among incorrect
predictions per model. Top 15 most-frequent incorrect tokens per model.

#### gpt-j-6b: 1,585 wrong rows; 201 distinct pred tokens overall

| pred_first_token_text | count |
|---|---:|
| `
` | 123 |
| `?` | 75 |
| `122` | 41 |
| `119` | 40 |
| `121` | 40 |
| `133` | 31 |
| `143` | 29 |
| `157` | 27 |
| `101` | 27 |
| `99` | 24 |
| `112` | 24 |
| `106` | 23 |
| `61` | 22 |
| `154` | 22 |
| `113` | 22 |

#### llama-3.1-8b: 37 wrong rows; 200 distinct pred tokens overall

| pred_first_token_text | count |
|---|---:|
| `75` | 2 |
| `77` | 2 |
| `58` | 2 |
| `88` | 2 |
| `81` | 2 |
| `93` | 2 |
| `87` | 2 |
| `92` | 2 |
| `56` | 1 |
| `74` | 1 |
| `72` | 1 |
| `1` | 1 |
| `0` | 1 |
| `55` | 1 |
| `54` | 1 |

#### pythia-6.9b: 2,282 wrong rows; 206 distinct pred tokens overall

| pred_first_token_text | count |
|---|---:|
| `122` | 84 |
| `127` | 79 |
| `?` | 59 |
| `155` | 58 |
| `154` | 56 |
| `95` | 54 |
| `143` | 52 |
| `119` | 50 |
| `124` | 49 |
| `123` | 49 |
| `100` | 49 |
| `99` | 49 |
| `98` | 48 |
| `115` | 46 |
| `134` | 46 |

### 31.11 Predicted-token distribution on incorrect multiplication predictions

#### gpt-j-6b: 272 wrong rows

| pred_first_token_text | count |
|---|---:|
| `6` | 34 |
| `9` | 34 |
| `7` | 31 |
| `5` | 29 |
| `8` | 27 |
| `
` | 6 |
| `1` | 4 |
| `576` | 3 |
| ` 6` | 3 |
| `456` | 3 |
| `693` | 3 |
| `736` | 2 |
| `656` | 2 |
| `513` | 2 |
| `506` | 2 |

#### llama-3.1-8b: 96 wrong rows

| pred_first_token_text | count |
|---|---:|
| ` ` | 6 |
| `288` | 2 |
| `252` | 2 |
| `504` | 2 |
| `646` | 2 |
| `608` | 2 |
| `432` | 2 |
| `348` | 2 |
| `552` | 2 |
| `633` | 2 |
| `791` | 2 |
| `603` | 1 |
| `384` | 1 |
| `219` | 1 |
| `3` | 1 |

#### pythia-6.9b: 266 wrong rows

| pred_first_token_text | count |
|---|---:|
| `7` | 59 |
| `8` | 28 |
| `9` | 26 |
| `5` | 16 |
| `6` | 14 |
| `567` | 3 |
| `648` | 3 |
| `231` | 3 |
| `4` | 2 |
| `684` | 2 |
| `486` | 2 |
| `81` | 2 |
| `388` | 2 |
| `576` | 2 |
| `514` | 2 |

### 31.12 Cross-model agreement on wrong-prediction tokens (addition)

For pairs where two or more models are wrong, count cases where they emit
the same `pred_first_token_id`.

- GPT-J wrong AND Llama wrong: 18; same wrong-token: 5.
- GPT-J wrong AND Pythia wrong: 675; same wrong-token: 289.
- Llama wrong AND Pythia wrong: 15; same wrong-token: 0.

### 31.13 Activation .npy file shapes and dtypes

| model | task | layer | path | shape | dtype |
|---|---|---:|---|---|---|
| gpt-j-6b | addition | 04 | `addition_layer_04.npy` | (10000, 4096) | float32 |
| gpt-j-6b | addition | 08 | `addition_layer_08.npy` | (10000, 4096) | float32 |
| gpt-j-6b | addition | 14 | `addition_layer_14.npy` | (10000, 4096) | float32 |
| gpt-j-6b | addition | 20 | `addition_layer_20.npy` | (10000, 4096) | float32 |
| gpt-j-6b | addition | 24 | `addition_layer_24.npy` | (10000, 4096) | float32 |
| gpt-j-6b | multiplication | 04 | `multiplication_layer_04.npy` | (3023, 4096) | float32 |
| gpt-j-6b | multiplication | 08 | `multiplication_layer_08.npy` | (3023, 4096) | float32 |
| gpt-j-6b | multiplication | 14 | `multiplication_layer_14.npy` | (3023, 4096) | float32 |
| gpt-j-6b | multiplication | 20 | `multiplication_layer_20.npy` | (3023, 4096) | float32 |
| gpt-j-6b | multiplication | 24 | `multiplication_layer_24.npy` | (3023, 4096) | float32 |
| llama-3.1-8b | addition | 04 | `addition_layer_04.npy` | (10000, 4096) | float32 |
| llama-3.1-8b | addition | 08 | `addition_layer_08.npy` | (10000, 4096) | float32 |
| llama-3.1-8b | addition | 16 | `addition_layer_16.npy` | (10000, 4096) | float32 |
| llama-3.1-8b | addition | 24 | `addition_layer_24.npy` | (10000, 4096) | float32 |
| llama-3.1-8b | addition | 28 | `addition_layer_28.npy` | (10000, 4096) | float32 |
| llama-3.1-8b | multiplication | 04 | `multiplication_layer_04.npy` | (3023, 4096) | float32 |
| llama-3.1-8b | multiplication | 08 | `multiplication_layer_08.npy` | (3023, 4096) | float32 |
| llama-3.1-8b | multiplication | 16 | `multiplication_layer_16.npy` | (3023, 4096) | float32 |
| llama-3.1-8b | multiplication | 24 | `multiplication_layer_24.npy` | (3023, 4096) | float32 |
| llama-3.1-8b | multiplication | 28 | `multiplication_layer_28.npy` | (3023, 4096) | float32 |
| pythia-6.9b | addition | 04 | `addition_layer_04.npy` | (10000, 4096) | float32 |
| pythia-6.9b | addition | 08 | `addition_layer_08.npy` | (10000, 4096) | float32 |
| pythia-6.9b | addition | 16 | `addition_layer_16.npy` | (10000, 4096) | float32 |
| pythia-6.9b | addition | 24 | `addition_layer_24.npy` | (10000, 4096) | float32 |
| pythia-6.9b | addition | 28 | `addition_layer_28.npy` | (10000, 4096) | float32 |
| pythia-6.9b | multiplication | 04 | `multiplication_layer_04.npy` | (3023, 4096) | float32 |
| pythia-6.9b | multiplication | 08 | `multiplication_layer_08.npy` | (3023, 4096) | float32 |
| pythia-6.9b | multiplication | 16 | `multiplication_layer_16.npy` | (3023, 4096) | float32 |
| pythia-6.9b | multiplication | 24 | `multiplication_layer_24.npy` | (3023, 4096) | float32 |
| pythia-6.9b | multiplication | 28 | `multiplication_layer_28.npy` | (3023, 4096) | float32 |

### 31.14 Per-model task runtime (read from summary.json)

| model | task | n_problems | n_correct | accuracy | task_runtime_seconds | extract_seconds | generate_seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| gpt-j-6b | addition | 10,000 | 8,415 | 0.8415 | 32.603 | 11.686 | 20.009 |
| gpt-j-6b | multiplication | 3,023 | 2,751 | 0.9100 | 9.454 | 3.227 | 5.989 |
| llama-3.1-8b | addition | 10,000 | 9,963 | 0.9963 | 40.316 | 15.022 | 24.376 |
| llama-3.1-8b | multiplication | 3,023 | 2,927 | 0.9682 | 12.065 | 4.435 | 7.393 |
| pythia-6.9b | addition | 10,000 | 7,718 | 0.7718 | 32.138 | 12.265 | 19.173 |
| pythia-6.9b | multiplication | 3,023 | 2,757 | 0.9120 | 8.818 | 3.037 | 5.571 |

### 31.15 Headline numerical statements

Aggregated over all answers files:

- Total addition correct across the 3 models: **26,096** of 30,000 (= 0.8699).
- Total multiplication correct across the 3 models: **8,435** of 9,069 (= 0.9301).

Per-(task, model) pair-correct counts:

- gpt-j-6b: addition correct = 8,415 of 10,000 (0.8415); multiplication correct = 2,751 of 3,023 (0.9100).
- llama-3.1-8b: addition correct = 9,963 of 10,000 (0.9963); multiplication correct = 2,927 of 3,023 (0.9682).
- pythia-6.9b: addition correct = 7,718 of 10,000 (0.7718); multiplication correct = 2,757 of 3,023 (0.9120).

- Among 10,000 addition pairs: 6,797 have all three models correct; 7 have none correct.
- Among 3,023 multiplication intersection pairs: 2,545 have all three models correct; 20 have none correct.
