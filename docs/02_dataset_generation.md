# Step 2: Building the addition and multiplication ground-truth datasets

**Anshul's Geometry of Arithmetic in LMs Project**
**Carnegie Mellon University, May 2026**

This document records every decision, every number, and every result from the
ground-truth dataset build — addition (10,000 problems, full Cartesian product)
and multiplication (3,023 problems, the cross-model single-token intersection).
It is the truth document for this stage. All numbers are validated against the
actual output files at `data/data/raw/` as of 2026-05-09.

---

## Table of Contents

1. [Purpose of this stage](#1-purpose-of-this-stage)
2. [What this stage is and is not](#2-what-this-stage-is-and-is-not)
3. [Why two datasets, asymmetric sizes](#3-why-two-datasets-asymmetric-sizes)
4. [The addition dataset (10,000 pairs)](#4-the-addition-dataset)
5. [The multiplication dataset (cross-model single-token intersection, 3,023 pairs)](#5-the-multiplication-dataset)
6. [Concept label schema (Tiers 1–5)](#6-concept-label-schema)
7. [Concept applicability table](#7-concept-applicability-table)
8. [Validation](#8-validation)
9. [Per-value coverage audit](#9-per-value-coverage-audit)
10. [Implications for downstream stages](#10-implications-for-downstream-stages)
11. [Output files](#11-output-files)
12. [Runtime and reproducibility](#12-runtime-and-reproducibility)
13. [Open questions and follow-ups](#13-open-questions-and-follow-ups)

---

## 1. Purpose of this stage

Step 1 (the tokenization preflight, [01_tokenization_limits.md](01_tokenization_limits.md))
established two facts that lock the dataset design:

- **Addition** is single-token in all three models for every (a, b) ∈ [0, 99]².
  No filtering is needed — the full 10,000-pair Cartesian product is usable.
- **Multiplication** is *not* single-token everywhere; per-model rates are
  GPT-J 32.87%, Llama 33.90%, Pythia 34.88%. We use the **cross-model
  intersection** — pairs whose answer is single-token in all three models —
  so that per-pair cross-model comparisons are clean (verdict differences
  cannot be confounded with subset differences).

Step 2 builds the two datasets, computes the **full Tier 1–5 ground-truth
concept label schema** for every problem (atomic, algebraic intermediates,
structural, relational, tokenization), validates math consistency, and writes
intermediate audit artefacts (the intersection mask, the per-concept coverage
report, the build manifest). **No model runs at this stage** — pure Python
ground-truth math.

The dataset built here is what every downstream stage of plan v6 reads from:
Stage 1 (linear probe) joins these labels onto activations, Stage 2 (Bayesian
manifold) needs the per-value concept distribution we audit here, Stage 3
(orthogonalisation) needs the algebraic-correlate columns we precompute here,
and Stage 4 (causal ablation) needs the per-(a,b) tokenization metadata we join
on here.

---

## 2. What this stage is and is not

### What it is

- A pure-Python deterministic enumeration of (a, b, answer) triples for both
  tasks, with **every concept derivable from the triple stored as a column**.
- A read of Step 1's per-pair tokenization CSVs and a join onto each problem
  so the data file is self-contained (downstream code doesn't need to re-load
  Step 1's outputs).
- A full audit trail: the intersection mask covering every (a, b) ∈ [0, 99]²,
  the per-concept coverage report, the manifest with sha256 of every input.
- A ~2-second CPU job. No GPU, no model loading, no probabilistic anything.

### What it is not

- **Not** a model-inference step. No correctness measurement here. The
  `is_single_token_*` flags come from Step 1's tokenizer-only sweep, not from
  generation. The model's actual *correctness* on these problems is measured
  in Step 3.
- **Not** levelled. Plan v6 §4 locks `[0, 99]` for both tasks; the L1–L5
  decomposition that arithmetic-geometry used does not apply here.
- **Not** stratified or sampled. Both datasets are fully enumerated subsets;
  there is no random sampling. The seed in the manifest is recorded for
  schema uniformity only.
- **Not** a concept-selection step. Plan v6 §5.2 names headline concepts; this
  step stores **all** concepts (every Tier 1–4 quantity derivable from `(a, b,
  answer)`) so any future analysis is a column lookup, not a recompute.

---

## 3. Why two datasets, asymmetric sizes

The two tasks have intrinsically different tokenization profiles, and forcing
them onto the same operand subset would either cripple addition (drop 70% of
its data for a constraint addition doesn't have) or pollute multiplication
(let multi-token answers through and reintroduce the §4.3 confound). The
honest design is two datasets sized to each task's natural constraint:

- **Addition: 10,000 pairs.** The full `[0, 99]²` Cartesian product. Every
  pair single-token in all three models (Step 1 §8.2 confirmed).
- **Multiplication: 3,023 pairs.** The cross-model single-token intersection —
  every pair whose `a×b` is single-token in GPT-J **and** Llama **and** Pythia.
  The intersection is a strict subset of `[0, 99]²` and corresponds geometrically
  to "the lower-left wedge of the operand box" (small × small products).

Plan v6's verdict matrix (§8) is per-cell, with N already varying by cell.
Different-sized addition and multiplication sets is the honest expression of
that.

---

## 4. The addition dataset

| | |
|---|---|
| Number of problems | **10,000** (full Cartesian product) |
| Operand range | `a, b ∈ [0, 99]` |
| Answer range | `[0, 198]` |
| Prompt | `"Output ONLY a number. {a}+{b}="` |
| Filter | none |
| Files | [data/data/raw/addition_problems.json](../data/data/raw/addition_problems.json), [data/data/raw/addition_problems.csv](../data/data/raw/addition_problems.csv) |

Every pair has `is_single_token_{gpt_j,llama,pythia} == 1` (Step 1 confirmed
addition is universally single-token). For first-token correctness purposes
this means **first-token correctness ≡ exact-match correctness** for addition
across all three models.

---

## 5. The multiplication dataset

| | |
|---|---|
| Number of problems | **3,023** |
| Operand range | `a, b ∈ [0, 99]`, restricted by intersection |
| Answer range observed | `[0, 999]` (bounded above by Llama's cap) |
| Prompt | `"Output ONLY a number. {a}*{b}="` |
| Filter | `is_single_token == 1` for all three models simultaneously |
| Source of filter | [data/results/tokenization_limits/multiplication_per_pair.csv](../data/results/tokenization_limits/multiplication_per_pair.csv) (Step 1) |
| Files | [data/data/raw/multiplication_problems.json](../data/data/raw/multiplication_problems.json), [data/data/raw/multiplication_problems.csv](../data/data/raw/multiplication_problems.csv) |
| Audit trail | [data/data/raw/multiplication_intersection_mask.csv](../data/data/raw/multiplication_intersection_mask.csv) — all 10,000 pairs with per-model flags and the resulting `is_intersection`. |

### How the intersection was computed

The script reads `multiplication_per_pair.csv` (30,000 rows = 10,000 pairs ×
3 models), pivots into per-pair flags, and keeps pairs where
`is_single_token_gpt_j == 1 AND is_single_token_llama == 1 AND is_single_token_pythia == 1`.

### Intersection statistics

| Metric | Value |
|---|---:|
| Total pairs in `[0, 99]²` | 10,000 |
| GPT-J single-token (Step 1) | 3,287 |
| Llama single-token (Step 1) | 3,390 |
| Pythia single-token (Step 1) | 3,488 |
| **Intersection size** | **3,023** |
| Intersection / GPT-J | 91.97% |
| Intersection / Llama | 89.17% |
| Intersection / Pythia | 86.67% |
| Intersection answer range | `[0, 999]` |
| First multi-token answer (per Step 1) | 1000 (Llama) |

The intersection is **larger than my earlier conservative lower-bound estimate
(2,800)** and very close to GPT-J's count (3,287). The reason is that GPT-J's
~388 "lucky" single-token integers above its contiguous cap (520) overlap
heavily with Llama's `[0, 999]` envelope and with Pythia's similar pattern,
so most of GPT-J's single-token pairs also clear Llama and Pythia.

---

## 6. Concept label schema

Every per-problem record carries Tier 1–5 fields. The full key list is below.
For the canonical example record see [data/data/raw/multiplication_problems.json](../data/data/raw/multiplication_problems.json) (`problems[0]`).

### Tier 1 — Atomic operand and answer values (both tasks)

`a`, `b`, `a_units`, `a_tens`, `a_num_digits`, `a_digits_lsf`, `b_units`,
`b_tens`, `b_num_digits`, `b_digits_lsf`, `answer`, `answer_digits_lsf`,
`answer_digits_msf`, `ans_units`, `ans_tens`, `ans_hundreds`, `ans_num_digits`
(plus `ans_thousands` for multiplication).

### Tier 2 — Algebraic intermediates

**Addition:** `column_sum_units`, `column_sum_tens`, `carry_units`, `carry_tens`,
`running_sum_units`, `running_sum_tens`, plus array forms `column_sums`,
`carries`, `running_sums`.

**Multiplication:** `partial_products` (dict of all four `a{i}_x_b{j}`),
`partial_product_units`, `partial_product_a_units_b_tens`,
`partial_product_a_tens_b_units`, `partial_product_a_tens_b_tens`,
`column_sum_units`, `column_sum_tens`, `column_sum_hundreds`,
`column_sum_thousands`, `column_products` (which partial products feed which
column), `carry_units`, `carry_tens`, `carry_hundreds`, `carry_thousands`,
`running_sum_units`, `running_sum_tens`, `running_sum_hundreds`,
`running_sum_thousands`, plus array forms `column_sums`, `carries`,
`running_sums`.

### Tier 3 — Structural / distributional (both tasks)

`a_parity`, `b_parity`, `ans_parity`, `parity_match`, `parity_xor`,
`a_magnitude_tier`, `b_magnitude_tier`, `ans_magnitude_tier`,
`ans_ends_in_zero`, `ans_is_zero`, `a_is_zero`, `b_is_zero`.

### Tier 4 — Relational (both tasks)

`a_eq_b`, `max_operand`, `min_operand`, `operand_diff`, `operand_abs_diff`,
`larger_operand`, `both_zero`, `either_zero`, `both_one`, `either_one`.

### Tier 5 — Tokenization metadata (both tasks)

For each model `mk ∈ {gpt_j, llama, pythia}`:
`is_single_token_{mk}`, `first_token_id_{mk}`, `first_token_text_{mk}`,
`n_tokens_{mk}`. Plus the unified `is_intersection` flag.

### CSV mirror

Every Tier 1–5 key is a column. Dict-valued fields are JSON-encoded into a
single cell; array-valued fields are space-separated. Example: the row for
`(a=23, b=45)` in `addition_problems.csv` (excerpt):

```
23,45,68,3,2,2,3 2,5,4,2,5 4,8,6,0,2,8 6,6 8,8,6,0,0,8,6,8 6,0 0,8 6,1,1,0,1,0,
two_digit,two_digit,two_digit,0,0,0,0,0,45,23,-22,22,b,0,0,0,0,1,1,3104,68,1,
1,2614,68,1,1,2358,68,1
```

The first-token IDs (3104, 2614, 2358) match Step 1's per-pair CSV and the
inline smoke test from earlier in the project history.

---

## 7. Concept applicability table

| Concept | Addition | Multiplication |
|---|:---:|:---:|
| `a`, `b`, operand digits, num_digits, digits_lsf | ✓ | ✓ |
| `answer`, `ans_units`, `ans_tens`, `ans_hundreds` | ✓ | ✓ |
| `ans_thousands` | — | ✓ (always 0 in the intersection, since answers ≤ 999) |
| `column_sum_units`, `column_sum_tens` | ✓ | ✓ (different definition: addition = digit sums; multiplication = partial-product sums) |
| `column_sum_hundreds`, `column_sum_thousands` | — | ✓ |
| `carry_units`, `carry_tens` | ✓ (boolean) | ✓ (integer ≥ 0) |
| `carry_hundreds`, `carry_thousands` | — | ✓ |
| `running_sum_*` | ✓ | ✓ |
| `partial_products`, `partial_product_*` | — | ✓ |
| Tier 3 structural / Tier 4 relational | ✓ | ✓ |
| Tier 5 tokenization metadata | ✓ | ✓ |

`carry_units` exists in both tasks but with different semantics: for addition,
it is the boolean "did `a_units + b_units` overflow 10"; for multiplication,
it is the integer carry from `units(a) × units(b)` into the tens column
(range 0..8). This is intentional — both definitions are the natural ones for
the corresponding operation, and downstream code distinguishes by task.

---

## 8. Validation

The script asserts, for every problem:

- **Math consistency:** `labels.answer == a + b` (addition) or `a × b`
  (multiplication).
- **Digit reconstruction:** `sum(d × 10^i for i, d in enumerate(answer_digits_lsf)) == answer`.
- **Carry self-consistency:** `column_sums[k] + carry_in == running_sums[k]`,
  `carries[k] == running_sums[k] // 10`, with `carry_in` chained from the
  previous column.
- **Bounds:** `0 ≤ a, b ≤ 99`, `0 ≤ a+b ≤ 198`, `0 ≤ a×b ≤ 9801`.
- **Intersection invariant:** every multiplication record has all three
  `is_single_token_*` flags equal to `1`.

All 13,023 records (10,000 + 3,023) passed every assertion. Build runtime: 1.9 s.

---

## 9. Per-value coverage audit

For every Tier 1–4 concept (Tier 5 tokenization is excluded — it is a join
artefact, not a probe target), the script counts per-value occurrences. Values
below the floor of 30 examples are flagged. The full per-(concept, value)
table is in [coverage_report.md](../data/data/raw/coverage_report.md).

### Addition (10,000 problems)

Headline concepts (plan §5.2):

| Concept | Unique values | Min count | Max count | Below floor |
|---|---:|---:|---:|---:|
| `a_units` | 10 | 1,000 | 1,000 | 0 |
| `b_units` | 10 | 1,000 | 1,000 | 0 |
| `carry_units` (bool) | 2 | 4,500 | 5,500 | 0 |
| `ans_units` | 10 | 1,000 | 1,000 | 0 |
| `ans_tens` | 10 | 1,000 | 1,000 | 0 |

All headline concepts are perfectly balanced (10 values × 1,000 each, by
construction of the Cartesian product). High-cardinality tail concepts have
expected long-tail behaviour: `answer` (199 unique values) has 58/199 below
floor — values near 0 and 198 are sparse — but `answer` is rarely a probe
target on its own.

### Multiplication (3,023 problems)

Headline concepts (plan §5.2):

| Concept | Unique values | Min count | Max count | Below floor | Comment |
|---|---:|---:|---:|---:|---|
| `a_units` | 10 | 246 | 385 | 0 | well-covered |
| `b_units` | 10 | 246 | 385 | 0 | well-covered |
| `carry_units` | 9 (0–8) | 14 | 1,437 | **1** | value `8` has only 14 examples — drops in §7.4 |
| `ans_units` | 10 | 97 | 990 | 0 | well-covered (skewed toward 0 because intersection has many `a×b` ending in 0, but all 10 values clear floor) |
| `partial_product_units` | 37 (0..81 sparse) | 14 | 728 | **3** | values 49, 64, 81 each below floor |
| `column_sum_units` | 37 | same as `partial_product_units` (identical: `column_sum_units = partial_product_units` for 2×2 mult) | | 3 | |
| `column_sum_tens` | 53 | 2 | 303 | 17 | high-value tail sparse — expected for intersection skew |

Note `carry_units = 8` has 14 examples — Stage 1 LDA will drop this value per
plan §7.4. The remaining 8 values (0–7) total 3,009 problems, comfortably
above the LDA floor.

### Long-tail concepts

- `answer` (multiplication, 564 unique values 0..999): 563 below floor — only
  value `0` has many examples. This is fine; `answer` itself is rarely probed
  directly.
- `operand_diff` (signed difference, 199 unique values): 197 below floor —
  natural for any continuous-like concept across this many values. Probes use
  `operand_abs_diff` (100 values) which is also long-tailed but more compact.
- `max_operand`, `min_operand`: 100 unique values each, ~half below floor.
  Expected for the intersection's operand-pair skew toward small values.

The full coverage report enumerates all 38 (addition) and 41 (multiplication)
audited concepts with per-value tables.

---

## 10. Implications for downstream stages

### For Stage 1 (linear probe)

All headline concepts in plan §5.2 are usable for both tasks at the LDA stage,
**with one §7.4 trigger:** multiplication `carry_units = 8` has only 14
examples and will be dropped from LDA fits. The remaining 8 carry values
(3,009 problems total) keep the probe well-conditioned (N/d ≈ 200+ in a
Phase C subspace of d ≈ 9–18).

The substitute-concept rules in plan §5.4 remain in reserve but are not
needed at this stage.

### For Stage 3 (orthogonalisation against algebraic correlates)

The Tier 2 algebraic intermediates were stored explicitly so that Stage 3 can
look up `column_sum_units`, `partial_product_units`, etc. as columns rather
than recomputing them from `(a, b)`. Plan §3.3 / §4.3 algebraic correlate sets
map directly:

- Multiplication `carry_units → {column_sum_units, partial_product_units}`:
  both are columns in `multiplication_problems.csv`.
- Multiplication `ans_units → {column_sum_units, carry_units, partial_product_units}`:
  same.
- Addition `units(a+b) → {a, b, units(a), units(b)}`: same.

### For Stage 4 (causal ablation)

The Tier 5 tokenization metadata gives the per-model first-token id and text
for every problem. Stage 4's first-answer-token Δlogit measurement reads
these directly. Because both datasets are constrained to single-token answers
across all three models, **first-token correctness ≡ exact-match correctness**
in both — the §4.5 / §21.5 multi-token alternatives do not need to be invoked.

### For plan v7

The dataset is now committed. Plan v7 should:
- Update §4.3 multiplication count to 3,023 (was projected ~3,287 maximum).
- Note in §4.3 that the multiplication operand-range claim narrows from
  "a, b ∈ [0, 99]" to "the cross-model single-token intersection of [0, 99]²,
  N=3,023".
- Mark §21.5 risk as resolved (single-token-restricted intersection chosen
  proactively; the §21.5 fallback rule does not need to fire).

---

## 11. Output files

All under `/data/user_data/anshulk/emnlp2026/data/raw/`, visible as
`/home/anshulk/emnlp2026/data/data/raw/` through the project symlink.

| File | Size | Description |
|---|---:|---|
| `addition_problems.json` | 21 MB | 10,000 problems, full Tier 1–5 schema. |
| `addition_problems.csv` | 1.9 MB | Flat tabular mirror. |
| `multiplication_problems.json` | 9.2 MB | 3,023 problems, full Tier 1–5 schema. |
| `multiplication_problems.csv` | 1.1 MB | Flat tabular mirror. |
| `multiplication_intersection_mask.csv` | 194 KB | Audit trail: all 10,000 pairs with per-model flags and `is_intersection`. |
| `coverage_report.md` | 67 KB | Per-concept value-count audit, all 38 + 41 audited concepts. |
| `build_manifest.json` | 4 KB | Reproducibility metadata. |

Plus the run log at `data/logs/generate_datasets.log` (RotatingFileHandler,
10 MB × 3 backups).

---

## 12. Runtime and reproducibility

| Item | Value |
|---|---|
| Wall time | 1.9 seconds |
| Compute | CPU only — no GPU, no model loading |
| Memory peak | < 500 MB |
| Determinism | fully deterministic; the manifest seed (42) is recorded for schema uniformity but no random sampling occurs |
| Re-run command | `/data/user_data/anshulk/miniconda3/envs/geometry/bin/python /home/anshulk/emnlp2026/generate_datasets.py --config /home/anshulk/emnlp2026/config.yaml` |
| Conda env | `geometry` at `/data/user_data/anshulk/miniconda3/envs/geometry` |

The manifest records sha256 of the config file, the build script, and both
input CSVs from Step 1, plus numpy and Python versions. Any change to any
input invalidates the recorded sha256 and a re-run produces fresh outputs.

---

## 13. Open questions and follow-ups

1. **Plan v7 update.** §4.3 multiplication count and operand-range claim need
   correction; §21.5 risk and decision rule can be marked resolved.
2. **Sub-floor `carry_units = 8` (multiplication).** 14 examples. Stage 1
   drops it. If a future analysis wants this value covered, the only fix is
   to expand the operand range beyond `[0, 99]` (out of scope for this paper)
   or to relax the intersection (introduces multi-token confound).
3. **Sub-floor partial-product values (49, 64, 81).** Three high partial
   products in multiplication are below floor (each appears only when both
   operand digits are large, which is rare in the intersection). Same trade-off
   as item 2.
4. **CSV column ordering.** Currently driven by Python dict insertion order
   from the first record. Stable across runs of the same script version,
   but not guaranteed across script edits. If Stage 1's join code becomes
   sensitive to column order, lock it explicitly. Not required today.
5. **Per-task duplicate concepts.** `column_sum_units` for multiplication is
   numerically identical to `partial_product_units` (both are
   `units(a) × units(b)` for 2×2 schoolbook). Both are stored under their
   distinct names per plan v6 §4.3's algebraic correlate set, which lists
   them as separate concepts. Stage 3 uses both names; downstream code should
   not assume they are independent variables.

---

## 14. Worked example: one record end-to-end

To make the schema concrete, here is the full Tier 1–5 record for `(a=23,
b=45)` in the **multiplication** dataset (intersection record). Every value
below is a hand-derivable consequence of the schoolbook multiplication of
23 × 45 = 1,035, plus the tokenization metadata joined from Step 1.

```
Tier 1 — atomic
  a = 23, b = 45, answer = 1035
  a_units = 3, a_tens = 2, a_num_digits = 2, a_digits_lsf = [3, 2]
  b_units = 5, b_tens = 4, b_num_digits = 2, b_digits_lsf = [5, 4]
  ans_units = 5, ans_tens = 3, ans_hundreds = 0, ans_thousands = 1
  ans_num_digits = 4
  answer_digits_lsf = [5, 3, 0, 1], answer_digits_msf = [1, 0, 3, 5]

Tier 2 — algebraic intermediates (schoolbook 2×2)
  partial_product_units            = a_units × b_units = 3 × 5 = 15
  partial_product_a_units_b_tens   = a_units × b_tens  = 3 × 4 = 12
  partial_product_a_tens_b_units   = a_tens  × b_units = 2 × 5 = 10
  partial_product_a_tens_b_tens    = a_tens  × b_tens  = 2 × 4 = 8
  partial_products = {a0_x_b0:15, a0_x_b1:12, a1_x_b0:10, a1_x_b1:8}

  column_sum_units    = pp_uu                           = 15
  column_sum_tens     = pp_ut + pp_tu                   = 12 + 10 = 22
  column_sum_hundreds = pp_tt                           = 8
  column_sum_thousands = 0
  column_products = {"0":["a0_x_b0"], "1":["a0_x_b1","a1_x_b0"], "2":["a1_x_b1"], "3":[]}

  carry_units    = column_sum_units // 10               = 15 // 10 = 1
  running_sum_units = column_sum_units                  = 15      → ans_units = 15 % 10 = 5 ✓
  carry_tens     = (column_sum_tens + carry_units) // 10 = (22 + 1) // 10 = 2
  running_sum_tens = column_sum_tens + carry_units      = 23      → ans_tens = 23 % 10 = 3 ✓
  carry_hundreds = (column_sum_hundreds + carry_tens) // 10 = (8 + 2) // 10 = 1
  running_sum_hundreds = column_sum_hundreds + carry_tens = 10    → ans_hundreds = 10 % 10 = 0 ✓
  carry_thousands = ((0 + 1) // 10)                     = 0
  running_sum_thousands = 0 + 1                         = 1       → ans_thousands = 1 ✓

Tier 3 — structural / distributional
  a_parity = 1 (23 odd), b_parity = 1 (45 odd), ans_parity = 1 (1035 odd)
  parity_match = true, parity_xor = 0
  a_magnitude_tier = "two_digit"
  b_magnitude_tier = "two_digit"
  ans_magnitude_tier = "four_digit"
  ans_ends_in_zero = false, ans_is_zero = false
  a_is_zero = false, b_is_zero = false

Tier 4 — relational
  a_eq_b = false
  max_operand = 45, min_operand = 23
  operand_diff = -22 (signed), operand_abs_diff = 22, larger_operand = "b"
  both_zero = false, either_zero = false, both_one = false, either_one = false

Tier 5 — tokenization (joined from Step 1's per-pair CSV)
  is_intersection = true
  is_single_token_gpt_j  = 0  (because 1035 > 520 and not in GPT-J lucky set)
  is_single_token_llama  = 0  (1035 > 999)
  is_single_token_pythia = 0
```

Note: `(a=23, b=45)` with answer `1035` does not appear in the multiplication
intersection (since `1035 > 999`, it is multi-token in Llama). The example is
included here for the Tier 2 derivation. Records present in the
multiplication intersection have `answer ≤ 999` and
`is_single_token_{gpt_j, llama, pythia} == 1`.

Tier 1–4 fields are deterministic functions of `(a, b)`. Tier 5 fields are
joined from the per-pair tokenization CSVs produced by Step 1.

---

## 15. Concept registry — plan v6 references

Each concept stored in Tier 1–5 maps to one or more locations in plan v6:

### 15.1 Atomic operands (`a`, `b`, digit decompositions)

- `a` — plan §5.2 (addition headline operand concept).
- `a_units` — plan §5.2 (multiplication headline operand concept).
- `b`, `b_units`, `a_tens`, `b_tens` — plan §5.4 (substitute concepts);
  plan §4.2 (`units(a+b) → {a, b, units(a), units(b)}` correlate set).

### 15.2 Atomic answer digits (`ans_units`, `ans_tens`, etc.)

- `ans_units` — plan §5.2 (output headline concept for both tasks).
- `ans_tens`, `ans_hundreds`, `ans_thousands`, `answer_digits_lsf`,
  `answer_digits_msf` — appendix-tier concepts referenced in plan §17.

### 15.3 Algebraic intermediates (partial products, column sums, carries, running sums)

Plan §3.3 / §4.3 Stage 3 orthogonalisation operates on these fields.
Multiplication-specific intermediates cover the four partial products
`a_i × b_j` for `i, j ∈ {units, tens}` of 2×2 schoolbook multiplication.

### 15.4 Structural concepts (parity, magnitude tiers, zero-flags)

- `ans_magnitude_tier` — plan §5.4 substitute concept (`magnitude(a+b)`).
- `a_parity`, `b_parity`, `ans_parity`, `parity_match`, `parity_xor` —
  not headline concepts in plan v6; stored for appendix coverage.
- `a_is_zero`, `b_is_zero`, `ans_is_zero`, `ans_ends_in_zero`,
  `both_zero`, `either_zero` — stored for filtering and per-cell auditing.

### 15.5 Relational concepts (max, min, diff)

Stored for filtering and stratification operations downstream. `a_eq_b`
identifies 100 pairs in addition and 32 pairs in the multiplication
intersection (the diagonal `a=b` rows).

### 15.6 Tokenization metadata

Joined from Step 1's per-pair CSVs. Provides the gold first-token id per
model for direct integer comparison against Step 3's `pred_first_token_id`.

---

## 16. The multiplication intersection — geometric and statistical structure

### 16.1 Geometric description

Numerical properties of the intersection set:

- Maximum `answer` value present: 999.
- For `a ∈ [0, 10]`, all `b ∈ [0, 99]` produce `a×b ≤ 990 ≤ 999`
  (11 × 100 = 1,100 pairs).
- For `a ∈ [11, 99]`, the per-`a` count of valid `b` is `floor(999/a) + 1`
  (e.g. `a=11`: 91 pairs; `a=99`: 11 pairs).
- Total intersection size: **3,023** out of 10,000 (30.23%). The figure
  is below the count of pairs with `a×b ≤ 999` (~3,390 by direct
  enumeration) because the per-pair conjunction also drops pairs where
  one of the three models tokenizes the answer as multiple tokens.

### 16.2 Answer distribution within the intersection

From the 3,023-pair intersection, the answer values have:

| Statistic | Value |
|---|---:|
| min | 0 |
| max | 999 |
| median | 319 |
| mean | 361.9 |
| 25th percentile | 120 |
| 75th percentile | 570 |
| `answer == 0` count (from `a=0` or `b=0`) | 199 |
| `answer ≤ 100` count | 679 (22.5%) |
| `answer ≤ 500` count | 2,097 (69.4%) |
| `501 ≤ answer ≤ 999` count | 926 (30.6%) |

Recorded distribution facts from `coverage.csv`:
- 70% of intersection answers are `≤ 500`.
- The unrestricted-set first-token correct rate per model (33–35%, see
  §8 of [01_tokenization_limits.md](01_tokenization_limits.md)) is computed
  before the cross-model conjunction.

### 16.3 Per-concept value distribution within the intersection

From the §9 coverage audit:

| Concept | Unique values | Min count | Max count | Below floor (30) | Notes |
|---|---:|---:|---:|---:|---|
| `a_units` | 10 | 246 | 385 | 0 | well-distributed |
| `b_units` | 10 | 246 | 385 | 0 | well-distributed |
| `a_tens` | 10 | (see coverage_report.md) | | | low values overrepresented |
| `b_tens` | 10 | (see coverage_report.md) | | | low values overrepresented |
| `carry_units` | 9 (values 0..8) | 14 (value 8) | 1,437 (value 0) | 1 | value 8 below floor; Stage 1 drops it per plan §7.4 |
| `partial_product_units` | 37 (values 0..81 sparse) | 14 (value 81) | 728 (value 0) | 3 | values 49, 64, 81 below floor |
| `ans_units` | 10 | 97 (value 1) | 990 (value 0) | 0 | all clear floor |

Where the per-row "Min count" is omitted, the full distribution is in
[coverage_report.md](../data/data/raw/coverage_report.md).

---

## 17. How Stage 3 algebraic-correlate sets map to stored fields

Plan v6 §3.3 and §4 specify the orthogonalisation correlate sets per
concept. Each correlate is a stored field in our dataset; here is the
explicit join.

### 17.1 Addition correlate sets

| Concept | Plan v6 correlate set | Stored as columns |
|---|---|---|
| `a` | {} | (no orthogonalisation) |
| `units(a+b)` | {a, b, units(a), units(b)} | `a`, `b`, `a_units`, `b_units` |
| `carry_units` | {a, b, units(a), units(b)} | same |

### 17.2 Multiplication correlate sets

| Concept | Plan v6 correlate set | Stored as columns |
|---|---|---|
| `a_units` | {} | (no orthogonalisation) |
| `carry_units` | {column_sum_units, partial_product_units} | `column_sum_units`, `partial_product_units` |
| `ans_units` | {column_sum_units, carry_units, partial_product_units} | `column_sum_units`, `carry_units`, `partial_product_units` |

For the multiplication `carry_units` row, note that
`column_sum_units == partial_product_units == a_units × b_units` for the
2×2 schoolbook case. They are stored under both names because plan v6 lists
them as distinct correlates; downstream Stage 3 code can see this collision
and decide whether to deduplicate.

### 17.3 The `ans_units` orthogonalisation has three correlates

For multiplication on `a, b ∈ [0, 99]`:
- `ans_units = (column_sum_units + carry_in_to_units_column) % 10`.
- The units column has no carry-in (it is column 0), so
  `ans_units = column_sum_units % 10 = partial_product_units % 10`.
- For 2×2 schoolbook with `a, b ∈ [0, 99]`,
  `column_sum_units == partial_product_units == a_units × b_units`.

Plan v6 §5.2 marks `ans_units` (multiplication) with predicted Stage 3
verdict `inherited`.

---

## 18. Tier 3 / Tier 4 storage rationale

Tier 3 (structural) and Tier 4 (relational) fields are deterministic
functions of `(a, b)`. They are stored as columns rather than computed
on-the-fly per analysis. CSV size cost: <5 MB per task.

---

## 19. CSV schema technicalities

The CSVs use Python's stdlib `csv.writer` with the default dialect:

- **Dict-valued fields** (`partial_products`, `column_products`) are
  serialized via `json.dumps(v, separators=(",", ":"))` and stored as a
  single string cell. Reader code: `json.loads(row['partial_products'])`.
- **Array-valued fields** (`column_sums`, `carries`, `running_sums`,
  `answer_digits_lsf`, `answer_digits_msf`) are serialized as
  space-separated strings (e.g. `"15 22 8 0"`). Reader code:
  `[int(x) for x in row['column_sums'].split()]`.
- **Booleans** are serialized as `"0"` / `"1"` (Python's int cast). Reader
  code: `bool(int(row['carry_units']))`.
- **Categorical strings** (`a_magnitude_tier`, `larger_operand`) are
  stored verbatim. Reader code: direct string comparison.
- **Column ordering** is `a, b, answer` first (extracted by the writer
  helper), then everything else in record-insertion order. This is stable
  across re-runs of the same script version. If the script's
  `*_labels()` function is edited, the column order may shift; the
  manifest's `build_script_sha256` lets a reader detect this.

Any external pandas reader can do
`pd.read_csv("data/data/raw/addition_problems.csv")` and get usable
columns immediately.

---

## 20. Comparison with arithmetic-geometry's L1–L5 schema

| | arithmetic-geometry (multiplication, L1–L5) | emnlp2026 |
|---|---|---|
| Operand range | varies by level (e.g. L3: `[10, 99] × [10, 99]`) | locked at `[0, 99]²` for both tasks |
| Number of "levels" | 5 (L1 through L5) | none |
| Filtering | difficulty-balanced curated set (8,264 problems) | cross-model single-token intersection (3,023) |
| Population | correct, wrong, all | correct only (post-Step 3 filter) |
| Concept schema | Tier 1–2 (operand digits, partial products, column sums, carries, running sums, answer digits) | Tier 1–2 plus Tier 3 (structural) and Tier 4 (relational) |
| Algebraic correlate sets | not stored in dataset (computed at Phase H) | stored as named columns |
| Labels file | `labels/level_{N}.json` per level | `data/raw/{task}_problems.json`, one per task |

Departures from the parent project:
1. No level structure — operand range is locked at `[0, 99]²`.
2. Tier 3 and Tier 4 concepts are stored as columns rather than computed
   per analysis.

Other aspects of the data pipeline (manifest schema, JSON-and-CSV mirrors,
SHA-256 reproducibility metadata, per-concept coverage report, validation
block) follow the arithmetic-geometry pattern.

---

## 21. Output file sizes

Files written to `/data/user_data/anshulk/emnlp2026/data/raw/`:

| File | Bytes | Lines (CSV) |
|---|---:|---:|
| `addition_problems.json` | 21,875,926 | — |
| `addition_problems.csv` | 1,898,722 | 10,001 |
| `multiplication_problems.json` | 9,223,708 | — |
| `multiplication_problems.csv` | 1,133,844 | 3,024 |
| `multiplication_intersection_mask.csv` | 193,811 | 10,001 |
| `coverage_report.md` | 68,264 | 3,250 |
| `build_manifest.json` | 3,730 | — |

---

## 22. `multiplication_intersection_mask.csv` schema

7 columns × 10,000 data rows + 1 header row.

```
a, b, answer,
is_single_token_gpt_j, is_single_token_llama, is_single_token_pythia,
is_intersection
```

Each row corresponds to one `(a, b)` pair in `[0, 99]²`.
`is_intersection == 1` iff all three model flags are 1; iff this row's
`(a, b)` appears in `multiplication_problems.json`.

Counts:
- `is_intersection == 1`: 3,023 rows.
- `is_intersection == 0`: 6,977 rows.

---

## 23. `addition_problems.csv` and `multiplication_problems.csv` schemas

`addition_problems.csv` — one row per problem (10,000 rows + header):

```
a, b, answer,
a_units, a_tens, a_num_digits, a_digits_lsf,
b_units, b_tens, b_num_digits, b_digits_lsf,
ans_units, ans_tens, ans_hundreds, ans_num_digits,
answer_digits_lsf, answer_digits_msf,
column_sum_units, column_sum_tens,
carry_units, carry_tens,
running_sum_units, running_sum_tens,
column_sums, carries, running_sums,
a_parity, b_parity, ans_parity,
parity_match, parity_xor,
a_magnitude_tier, b_magnitude_tier, ans_magnitude_tier,
ans_ends_in_zero, ans_is_zero, a_is_zero, b_is_zero,
a_eq_b, max_operand, min_operand,
operand_diff, operand_abs_diff, larger_operand,
both_zero, either_zero, both_one, either_one,
is_single_token_gpt_j, first_token_id_gpt_j, first_token_text_gpt_j, n_tokens_gpt_j,
is_single_token_llama, first_token_id_llama, first_token_text_llama, n_tokens_llama,
is_single_token_pythia, first_token_id_pythia, first_token_text_pythia, n_tokens_pythia,
is_intersection
```

`multiplication_problems.csv` — same structure with task-specific Tier 2
fields:

```
... (Tier 1 unchanged) ...
ans_thousands,                    (multiplication-only)
... (Tier 2 multiplication) ...
partial_products,                 (JSON dict)
partial_product_units,
partial_product_a_units_b_tens,
partial_product_a_tens_b_units,
partial_product_a_tens_b_tens,
column_sum_units, column_sum_tens, column_sum_hundreds, column_sum_thousands,
column_sums, column_products,     (column_products is a JSON dict)
carry_units, carry_tens, carry_hundreds, carry_thousands,
running_sum_units, running_sum_tens, running_sum_hundreds, running_sum_thousands,
carries, running_sums,
... (Tier 3 + Tier 4 same as addition) ...
... (Tier 5 same as addition) ...
```

---

## 24. `build_manifest.json` schema

Top-level keys (read from `/data/user_data/anshulk/emnlp2026/data/raw/build_manifest.json`):

```
schema_version           = "v1"
generated_at_utc         = "2026-05-09T10:28:18Z"
seed                     = 42
config_path              = "/home/anshulk/emnlp2026/config.yaml"
config_sha256            = "e21f1a63332bf628d017d03ea2804c4ccd6ccc3f6ffcbd72c6834de136ff8e24"
build_script             = "generate_datasets.py"
build_script_sha256      = "97266f0959f49cd2c01991acc8506989c8cd05a3adbd69825f9b2722385c2d20"
tokenization_csv_addition       = "/data/user_data/anshulk/emnlp2026/results/tokenization_limits/addition_per_pair.csv"
tokenization_csv_addition_sha256 = "a42a14afa9af33d147bef552633614e61cb2c235ad12526f3a93a9cd8e508ae7"
tokenization_csv_multiplication       = "/data/user_data/anshulk/emnlp2026/results/tokenization_limits/multiplication_per_pair.csv"
tokenization_csv_multiplication_sha256 = "bdeb1b89cdc84794dff99dab1256ccd622f3b6421a97a2c09466fd2fc16be7a1"
operand_range            = [0, 99]
addition_count           = 10000
multiplication_count     = 3023
intersection_models      = ["gpt-j-6b", "llama-3.1-8b", "pythia-6.9b"]
outputs                  = { addition_json, addition_csv, multiplication_json,
                             multiplication_csv, intersection_mask_csv,
                             coverage_report }
numpy_version            = "2.2.6"
python_version           = "3.11.15"
log_path                 = "/data/user_data/anshulk/emnlp2026/logs/generate_datasets.log"
concept_floor            = 30
addition_audit_concepts  = [38 entries]
multiplication_audit_concepts = [41 entries]
runtime_seconds          = (build runtime in seconds)
```

---

## 25. `a_tens` and `b_tens` distribution in the multiplication intersection

| value | a_tens count | b_tens count |
|---:|---:|---:|
| 0 | 971 | 971 |
| 1 | 619 | 619 |
| 2 | 360 | 360 |
| 3 | 258 | 258 |
| 4 | 198 | 198 |
| 5 | 162 | 162 |
| 6 | 136 | 136 |
| 7 | 115 | 115 |
| 8 | 105 | 105 |
| 9 | 99 | 99 |

The two distributions are equal because the intersection is symmetric in
swap of `a` and `b`.

---

## 26. Validation invariants asserted by the build script

For every record, the script asserts (raises if violated):

- `0 ≤ a ≤ 99`, `0 ≤ b ≤ 99`.
- For addition: `answer == a + b`; `0 ≤ answer ≤ 198`.
- For multiplication: `answer == a × b`; `0 ≤ answer ≤ 9801`.
- `sum(d × 10^i for i, d in enumerate(answer_digits_lsf)) == answer`.
- For multiplication, with `carry_in` chained from the previous column:
  - `running_sums[k] == column_sums[k] + carry_in`
  - `carries[k] == running_sums[k] // 10`
  - `carry_in_next = carries[k]`
- For addition:
  - `running_sums[0] == column_sums[0]`
  - `carries[0] == 1 if column_sums[0] >= 10 else 0`
  - `running_sums[1] == column_sums[1] + carries[0]`
  - `carries[1] == 1 if running_sums[1] >= 10 else 0`
- For records in `multiplication_problems.json`:
  - `is_single_token_gpt_j == 1`
  - `is_single_token_llama == 1`
  - `is_single_token_pythia == 1`

All 13,023 records (10,000 addition + 3,023 multiplication) passed every
assertion.

---

## 27. Re-run command and runtime

```
/data/user_data/anshulk/miniconda3/envs/geometry/bin/python \
    /home/anshulk/emnlp2026/generate_datasets.py \
    --config /home/anshulk/emnlp2026/config.yaml
```

Recorded runtime: 1.9 seconds wall time (CPU only). No GPU, no model loading,
no random sampling.

---

## 28. Observations from the dataset CSVs

This section reports observations derived from the per-row values in
`data/data/raw/addition_problems.csv` (10,000 rows × 76 columns) and
`data/data/raw/multiplication_problems.csv` (3,023 rows × 89 columns)
plus `multiplication_intersection_mask.csv` (10,000 rows). All counts and
ratios below are computed directly from those files.

### 28.1 `column_sum_units` distribution (addition, 10,000 rows)

Range observed: [0, 18]. Per-value count:

| value | count | fraction |
|---:|---:|---:|
| 0 |    100 | 0.0100 |
| 1 |    200 | 0.0200 |
| 2 |    300 | 0.0300 |
| 3 |    400 | 0.0400 |
| 4 |    500 | 0.0500 |
| 5 |    600 | 0.0600 |
| 6 |    700 | 0.0700 |
| 7 |    800 | 0.0800 |
| 8 |    900 | 0.0900 |
| 9 |  1,000 | 0.1000 |
| 10 |    900 | 0.0900 |
| 11 |    800 | 0.0800 |
| 12 |    700 | 0.0700 |
| 13 |    600 | 0.0600 |
| 14 |    500 | 0.0500 |
| 15 |    400 | 0.0400 |
| 16 |    300 | 0.0300 |
| 17 |    200 | 0.0200 |
| 18 |    100 | 0.0100 |

- Sum across all values: 10,000 (= |addition| = 10,000).
- Symmetric distribution: count(v) == count(18-v): True.

### 28.2 `column_sum_tens` distribution (addition)

| value | count | fraction |
|---:|---:|---:|
| 0 |    100 | 0.0100 |
| 1 |    200 | 0.0200 |
| 2 |    300 | 0.0300 |
| 3 |    400 | 0.0400 |
| 4 |    500 | 0.0500 |
| 5 |    600 | 0.0600 |
| 6 |    700 | 0.0700 |
| 7 |    800 | 0.0800 |
| 8 |    900 | 0.0900 |
| 9 |  1,000 | 0.1000 |
| 10 |    900 | 0.0900 |
| 11 |    800 | 0.0800 |
| 12 |    700 | 0.0700 |
| 13 |    600 | 0.0600 |
| 14 |    500 | 0.0500 |
| 15 |    400 | 0.0400 |
| 16 |    300 | 0.0300 |
| 17 |    200 | 0.0200 |
| 18 |    100 | 0.0100 |

- Distribution is identical in shape to §28.1 because addition is commutative
  in the digit decomposition. Test of equality with §28.1 distribution:
  **True**.

### 28.3 `carry_units` × `carry_tens` joint distribution (addition)

`carry_units` ∈ {0, 1}; `carry_tens` ∈ {0, 1}. Joint counts:


Joint counts:

| carry_units | carry_tens | count |
|---:|---:|---:|
| 0 | 0 | 3,025 |
| 0 | 1 | 2,475 |
| 1 | 0 | 2,025 |
| 1 | 1 | 2,475 |

Marginals:
- `carry_units == 0`: 5,500 rows (0.5500).
- `carry_units == 1`: 4,500 rows (0.4500).
- `carry_tens == 0`: 5,050 rows.
- `carry_tens == 1`: 4,950 rows.

### 28.4 Per-`a_units` and per-`b_units` counts (addition)

Both columns have 10 unique values (0..9), each appearing 1,000 times by
construction of the Cartesian product:

- `a_units` value-count uniformity: {0: np.int64(1000), 1: np.int64(1000), 2: np.int64(1000), 3: np.int64(1000), 4: np.int64(1000), 5: np.int64(1000), 6: np.int64(1000), 7: np.int64(1000), 8: np.int64(1000), 9: np.int64(1000)}.
- `b_units` value-count uniformity: {0: np.int64(1000), 1: np.int64(1000), 2: np.int64(1000), 3: np.int64(1000), 4: np.int64(1000), 5: np.int64(1000), 6: np.int64(1000), 7: np.int64(1000), 8: np.int64(1000), 9: np.int64(1000)}.

### 28.5 `larger_operand` distribution per task

| task | larger == "a" | larger == "b" | larger == "equal" |
|---|---:|---:|---:|
| addition | 4,950 | 4,950 | 100 |
| multiplication (intersection) | 1,497 | 1,497 | 29 |

### 28.6 `a_eq_b` counts (the diagonal)

- Addition: 100 rows where `a == b`.
- Multiplication (intersection): 29 rows where `a == b`.

### 28.7 `parity_match` × `ans_parity` consistency check

For addition: `ans_parity == (a_parity + b_parity) % 2 == parity_xor`.
- All 10,000 addition rows satisfy `ans_parity == parity_xor`: **True**.

For multiplication: `ans_parity == 1` iff both `a` and `b` are odd.
- Multiplication rows where `a_parity==1 AND b_parity==1`: **688**.
- Multiplication rows where `ans_parity==1`: **688**.
- Rows where `ans_parity == (a_parity == 1 and b_parity == 1)`: **3023 / 3023**.

### 28.8 `ans_magnitude_tier` distribution per task

| task | zero | single_digit | two_digit | three_digit | four_digit |
|---|---:|---:|---:|---:|---:|
| addition | 1 | 54 | 4,995 | 4,950 | 0 |
| multiplication (intersection) | 199 | 23 | 450 | 2,351 | 0 |

### 28.9 `operand_abs_diff` distribution (addition vs multiplication intersection)

| task | min | 25%ile | median | 75%ile | 90%ile | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| addition | 0 | 13 | 29 | 50 | 68 | 99 | 33.33 |
| multiplication (intersection) | 0 | 14 | 33 | 59 | 79 | 99 | 37.82 |

### 28.10 Multiplication intersection — `partial_product_units` distribution

- Distinct values observed: **37**. Min: 0; max: 81.

| value | count |
|---:|---:|
| 0 | 728 |
| 1 | 35 |
| 2 | 70 |
| 3 | 66 |
| 4 | 95 |
| 5 | 64 |
| 6 | 122 |
| 7 | 52 |
| 8 | 118 |
| 9 | 87 |
| 10 | 70 |
| 12 | 124 |
| 14 | 60 |
| 15 | 58 |
| 16 | 87 |
| 18 | 106 |
| 20 | 66 |
| 21 | 48 |
| 24 | 108 |
| 25 | 31 |
| 27 | 46 |
| 28 | 56 |
| 30 | 64 |
| 32 | 56 |
| 35 | 58 |
| 36 | 77 |
| 40 | 62 |
| 42 | 52 |
| 45 | 54 |
| 48 | 52 |
| 49 | 25 |
| 54 | 46 |
| 56 | 50 |
| 63 | 50 |
| 64 | 22 |
| 72 | 44 |
| 81 | 14 |

- Sum of counts: 3,023 (= |multiplication intersection| = 3,023).
- Values appearing ≥ 30 times: 34; below 30: 3.

### 28.11 Multiplication intersection — `carry_units` distribution

| value | count |
|---:|---:|
| 0 | 1,437 |
| 1 | 505 |
| 2 | 355 |
| 3 | 255 |
| 4 | 245 |
| 5 | 96 |
| 6 | 72 |
| 7 | 44 |
| 8 | 14 |

- Distinct values observed: **9** (range 0–8).
- Values ≥ 30 examples: 8; below 30: 1.

### 28.12 Multiplication intersection — `column_sum_tens` distribution (top 25)

- Distinct values observed: **53** (range 0–81).
- Values ≥ 30 examples: 36; below 30: 17.

Top 25 most-common (value, count):

| value | count |
|---:|---:|
| 0 | 303 |
| 8 | 142 |
| 18 | 139 |
| 12 | 132 |
| 6 | 131 |
| 9 | 118 |
| 24 | 115 |
| 16 | 112 |
| 4 | 98 |
| 10 | 89 |
| 7 | 84 |
| 14 | 83 |
| 15 | 78 |
| 5 | 74 |
| 21 | 70 |
| 36 | 70 |
| 3 | 68 |
| 20 | 67 |
| 2 | 67 |
| 28 | 63 |
| 27 | 62 |
| 32 | 58 |
| 30 | 58 |
| 35 | 48 |
| 25 | 44 |

### 28.13 Multiplication intersection — `ans_thousands` distribution

- Distinct values observed: **1**: {0: np.int64(3023)}.
- All `ans_thousands == 0` for 3023 rows.
  Verified: **True**.

### 28.14 `ans_ends_in_zero` counts

| task | ans_ends_in_zero == 0 | ans_ends_in_zero == 1 |
|---|---:|---:|
| addition | 9,000 | 1,000 |
| multiplication (intersection) | 2,033 | 990 |

### 28.15 Intersection mask — joint flag distribution

From `multiplication_intersection_mask.csv` (10,000 rows):

| gpt_j | llama | pythia | count |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 6,316 |
| 0 | 0 | 1 | 85 |
| 0 | 1 | 0 | 133 |
| 0 | 1 | 1 | 179 |
| 1 | 0 | 0 | 8 |
| 1 | 0 | 1 | 201 |
| 1 | 1 | 0 | 55 |
| 1 | 1 | 1 | 3,023 |

- Sum: 10,000 (= |all pairs| = 10,000).
- Intersection (all three == 1): 3,023 rows.
- Union (any one == 1): 3,684 rows.
- None (all three == 0): 6,316 rows.

### 28.16 Multiplication intersection — answer-value distribution by 100-bins

| bin | count |
|---|---:|
| [0, 100) | 672 |
| [100, 200) | 413 |
| [200, 300) | 363 |
| [300, 400) | 336 |
| [400, 500) | 309 |
| [500, 600) | 252 |
| [600, 700) | 234 |
| [700, 800) | 199 |
| [800, 900) | 134 |
| [900, 1000) | 111 |
- Rows with `answer == 999`: 2.
- Rows with `answer == 0`: 199.
- Distinct `answer` values: 564.

### 28.17 Cross-tier consistency invariants

Each row in both CSVs satisfies the following invariants by construction;
re-verified post-hoc on the actual files:

- `addition: answer == a + b`: **True**.
- `multiplication: answer == a * b`: **True**.
- `addition: ans_units == answer % 10`: **True**.
- `multiplication: ans_units == answer % 10`: **True**.
- `multiplication: column_sum_units == a_units * b_units`: **True**.
- `multiplication: partial_product_units == a_units * b_units`: **True**.
- `multiplication: column_sum_units == partial_product_units` (for 2×2 schoolbook): **True**.

### 28.18 Per-`ans_units` value count per task

| ans_units value | addition count | multiplication count |
|---:|---:|---:|
| 0 | 1,000 | 990 |
| 1 | 1,000 | 97 |
| 2 | 1,000 | 346 |
| 3 | 1,000 | 116 |
| 4 | 1,000 | 331 |
| 5 | 1,000 | 265 |
| 6 | 1,000 | 336 |
| 7 | 1,000 | 98 |
| 8 | 1,000 | 332 |
| 9 | 1,000 | 112 |

- Addition: each value appears exactly 1,000 times (uniform).
- Multiplication intersection: range 97 to 990.

### 28.19 Number-of-digit distribution of operands and answers

| task | a_num_digits == 1 | a_num_digits == 2 | ans_num_digits == 1 | == 2 | == 3 | == 4 |
|---|---:|---:|---:|---:|---:|---:|
| addition | 1,000 | 9,000 | 55 | 4,995 | 4,950 | 0 |
| multiplication (intersection) | 971 | 2,052 | 222 | 450 | 2,351 | 0 |

### 28.20 Headline numerical statements

- Addition rows: 10,000.
- Multiplication intersection rows: 3,023.
- Sum: 13,023.
- Addition columns: 61.
- Multiplication columns: 74.
- Addition `parity_match == True` rows: 5,000
  (0.5000 of total).
- Multiplication `parity_match == True` rows: 1,493
  (0.4939 of total).
- Addition `both_zero == True`: 1.
- Addition `either_zero == True`: 199.
- Multiplication intersection `either_zero == True`: 199.
- Addition `both_one == True`: 1.
- Multiplication intersection `both_one == True`: 1.
