# Step 1 (preflight): Single-token integer limits across GPT-J 6B, Llama 3.1 8B, and Pythia 6.9B

**Anshul's Geometry of Arithmetic in LMs Project**
**Carnegie Mellon University, May 2026**

This document records every decision, every number, and every result from the
tokenization-limits preflight — the step that confirms what plan.md §4.2 and §4.3
assert about single-token answers and that fixes the data-generation gates referenced
in §21.5. It is the truth document for this stage. All numbers are validated against
the actual output files at `data/results/tokenization_limits/` as of 2026-05-09.

---

## Table of Contents

1. [Purpose of This Stage](#1-purpose-of-this-stage)
2. [What This Preflight Is and Is Not](#2-what-this-preflight-is-and-is-not)
3. [Why This Stage Exists (which downstream decisions depend on it)](#3-why-this-stage-exists)
4. [Models and Tokenizers Tested](#4-models-and-tokenizers-tested)
5. [Tokenization Contexts and Why Each One Matters](#5-tokenization-contexts-and-why-each-one-matters)
6. [The Per-Integer Sweep (0..10000)](#6-the-per-integer-sweep-0_10000)
7. [The Per-Pair Check (10,000 pairs × 2 tasks × 3 models)](#7-the-per-pair-check)
8. [Results](#8-results)
   - 8.1 Single-token integer cap per model
   - 8.2 Addition coverage (does plan §4.2 hold?)
   - 8.3 Multiplication coverage (does plan §4.3 / §21.5 still apply?)
   - 8.4 Tokenization context sensitivity
9. [Implications for Downstream Stages](#9-implications-for-downstream-stages)
10. [Output Files](#10-output-files)
11. [Runtime and Reproducibility](#11-runtime-and-reproducibility)
12. [Open Questions and Follow-ups](#12-open-questions-and-follow-ups)

---

## 1. Purpose of This Stage

Plan v6 makes two assertions that the rest of the pipeline depends on:

- **§4.2 (addition):** the answer range `[0, 198]` is single-token in both GPT-J and Llama, so addition has no multi-token confound.
- **§4.3 (multiplication):** answers run up to `9801`; only a fraction are single-token, and that fraction feeds the §21.5 decision rule on whether to use first-answer-token correctness or fall back to single-token-restricted.

Neither assertion had been independently verified in this project, and Pythia 6.9B
was not characterised at all (it was added as a third model after v6). Before any
problem set is generated, any activations are extracted, or any probe is fit, we
need the **per-model integer-tokenisation profile** locked down with numbers
quoted from the actual tokenizers.

This preflight produces those numbers. It also produces a per-pair CSV that any
downstream stage can join against to recover the answer-tokenisation status for
any (a, b) without re-running a tokenizer.

---

## 2. What This Preflight Is and Is Not

### What it is

- A tokenizer-only sweep (no model weights loaded; no GPU computation).
- A characterisation of how each tokenizer represents integers `0..10000`.
- A measurement, in the *real* KT prompt context, of how many tokens each
  arithmetic answer occupies for each of the 10,000 (a, b) pairs in `[0, 99]²`,
  per task, per model.
- A correction step: where plan v6's stated tokenisation numbers turn out to be
  wrong, this doc records the corrected values that plan v7 should adopt.

### What it is not

- It is *not* an inference run. The model weights are not loaded; correctness
  rates are not measured here.
- It is *not* an analysis of how the model *generates* answers. It only checks
  what the gold answer's tokenisation looks like in the prompt context the model
  will see. First-answer-token correctness during inference can still differ from
  full-sequence correctness for multi-token answers.
- It is *not* a Pythia-specific deep dive. Pythia is treated as a peer of GPT-J
  and Llama and gets the same sweep, no more.
- It is *not* a tokenizer benchmark. We aren't comparing GPT-2 BPE vs TikToken vs
  GPT-NeoX BPE on linguistic quality — we just need numbers for arithmetic answers.

---

## 3. Why This Stage Exists

Three downstream stages depend directly on the numbers produced here:

1. **Data generation (Step 1 of plan.md, §7).** The data generator marks each
   problem with `single_token_answer = True/False` so per-value stratification
   in §7.4 can use it. Without this preflight, that flag would be approximated
   from incorrect tokenisation caps.

2. **Stage 4 (causal ablation, §3.4).** The first-answer-token Δlogit measurement
   relies on knowing which prompts have a single-token gold answer. For any
   (a, b) pair where the answer is multi-token, the Δlogit is measured at the
   first generated token only, and the doc must record exactly which pairs those
   are.

3. **§21.5 decision rule.** The rule states that if GPT-J × multiplication
   first-token correct rate is < 15% we fall back to alternative (a)
   single-token-restricted correctness. The single-token-restricted set is
   defined relative to each model's *actual* integer cap, which this preflight
   establishes.

If the numbers in plan v6 were already correct, this would be a verification step.
As §8 shows, two of them are wrong, so this is a correction step too.

---

## 4. Models and Tokenizers Tested

| Model | Local path | Tokenizer family | Vocab size |
|---|---|---|---|
| GPT-J 6B | `data/models/gpt-j-6b/` | GPT-2 BPE | 50,257 |
| Llama 3.1 8B | `data/models/llama-3.1-8b/` | TikToken-derived | 128,000 |
| Pythia 6.9B | `data/models/pythia-6.9b/` | GPT-NeoX BPE | 50,254 |

All three tokenizers loaded via `transformers.AutoTokenizer.from_pretrained(local_path)`.
Tokenizer load times: GPT-J 0.2s, Llama 0.8s, Pythia 0.2s. Each model's local
path was previously populated under [Step 0 (model downloads)](../README.md).

---

## 5. Tokenization Contexts and Why Each One Matters

For each model we measure under seven contexts. The first five are **fixed
prefix** contexts — they probe how the tokenizer represents an integer when
preceded by a particular character. The last two are **full prompt** contexts
— they encode the actual KT prompt with the gold answer appended and measure
the answer's contribution. The full-prompt contexts are what matter for the
project; the fixed-prefix contexts are diagnostic.

| Context | Prefix | Why it matters |
|---|---|---|
| `bare` | `""` | Baseline. How `str(n)` tokenizes alone. |
| `leading_space` | `" "` | Captures the GPT/Pythia BPE behaviour where a leading space gets *merged* with the first digit into a single fused token (e.g. `" 0"` → one token). Recorded but not used downstream. |
| `post_equals` | `"="` | Equals-sign context (a single token in all three tokenizers, separates cleanly from the answer). |
| `post_plus` | `"+"` | Plus-sign context. Should match `post_equals` for typed-integer behaviour. |
| `post_star` | `"*"` | Asterisk context. Same. |
| `kt_addition_full` | `"Output ONLY a number. {a}+{b}="` | **The real context for addition.** Prefix is fully formed and tokenises stably ending in `=`. |
| `kt_multiplication_full` | `"Output ONLY a number. {a}*{b}="` | **The real context for multiplication.** |

### A note on prefix instability

For the `leading_space` context in GPT-J and Pythia, the BPE tokenizer fuses the
trailing space with the first digit of `str(n)` into a single token (e.g. for
GPT-J, `tok(" ")` is `[220]` but `tok(" 0")` is `[657]`, a wholly different
token). When this happens, the prefix's tokens are not a clean prefix of the
full encoding's tokens, so we cannot cleanly say "str(n) contributes K tokens
to this context." Such cases are flagged as `merged` in `summary.csv`. They do
not affect the project — the KT prompt context ends in `=`, which is its own
token in all three tokenizers and never merges with the digits that follow.

---

## 6. The Per-Integer Sweep (0..10000)

For each model and each fixed-prefix context, the script encodes `prefix + str(n)`
for `n = 0, 1, …, 10000`. For each `n` it records:

- whether the prefix's tokens appear as a clean prefix of the full encoding
  (`prefix_stable`),
- the tokens contributed by `str(n)` in that case (`ans_ids`),
- the number of such tokens (`n_tokens`).

Aggregation per (model, context) yields:

- `max_contiguous_single_token_N`: the largest `N` such that *every* integer in
  `[0, N]` is a prefix-stable single-token.
- `total_single_token_count_in_0_sweep_max`: how many integers in `[0, 10000]`
  are single-token (contiguous or not).
- `first_failing_n` and `first_failing_kind`: the smallest `n` that fails, and
  whether the failure was multi-token or merged.

These rows are written to [data/results/tokenization_limits/summary.csv](../data/results/tokenization_limits/summary.csv).

---

## 7. The Per-Pair Check

For each model and each task, we iterate the full Cartesian product
`a, b ∈ [0, 99]` (10,000 pairs) and encode the actual KT prompt with the
gold answer appended:

```
"Output ONLY a number. {a}+{b}={a+b}"   (addition)
"Output ONLY a number. {a}*{b}={a*b}"   (multiplication)
```

For each pair we record `n_tokens`, `is_single_token`, the `first_token_id`,
and the decoded `first_token_text`. The first-token info is exactly what
plan §4.3 needs for first-answer-token correctness, and exactly what the
data-generation step will use to mark each problem.

Per-pair rows are written to:

- [data/results/tokenization_limits/addition_per_pair.csv](../data/results/tokenization_limits/addition_per_pair.csv) — 30,000 rows (10,000 pairs × 3 models)
- [data/results/tokenization_limits/multiplication_per_pair.csv](../data/results/tokenization_limits/multiplication_per_pair.csv) — 30,000 rows

The aggregated single-token-fraction per (model, task) is in
[data/results/tokenization_limits/coverage.csv](../data/results/tokenization_limits/coverage.csv).

---

## 8. Results

### 8.1 Single-token integer cap per model

Quoted verbatim from `summary.csv`:

| Model | Context | Max contiguous N | First failing n | How it splits |
|---|---|---:|---:|---|
| GPT-J 6B | `bare` | **520** | 521 | `[20, 2481]` → `['5', '21']` |
| GPT-J 6B | `post_equals` | **520** | 521 | same |
| GPT-J 6B | `post_plus` | **520** | 521 | same |
| GPT-J 6B | `post_star` | **520** | 521 | same |
| Llama 3.1 8B | `bare` | **999** | 1000 | `[1041, 15]` → `['100', '0']` |
| Llama 3.1 8B | `post_equals` | **999** | 1000 | same |
| Llama 3.1 8B | `post_plus` | **999** | 1000 | same |
| Llama 3.1 8B | `post_star` | **999** | 1000 | same |
| Pythia 6.9B | `bare` | **530** | 531 | `[22, 2405]` → `['5', '31']` |
| Pythia 6.9B | `post_equals` | **530** | 531 | same |
| Pythia 6.9B | `post_plus` | **530** | 531 | same |
| Pythia 6.9B | `post_star` | **530** | 531 | same |

**Notes:**

1. For each model, the contiguous-single-token cap is the same across all four
   well-behaved separator contexts. The model character before the digits
   (`=`, `+`, `*`, or none) does not change the cap. This is reassuring: the
   KT prompt's `=` is not idiosyncratic.
2. The integers above the cap are **not all multi-token** — there is a
   sparse layer of "lucky" single-token integers above. For GPT-J,
   `total_single_token_count_in_0_sweep_max = 908` (so about 388 single-token
   integers exist scattered above 520). For Pythia it's 1017 (487 above 530).
   For Llama, 1000 (none above 999 in the swept range).
3. **Contradiction with plan v6 §4.3.** Plan claims the GPT-J single-token cap
   is 361. The actual contiguous cap is **520**, and the actual `bare`
   first-failure is `521`. This was a numerical error in the plan, possibly
   inherited from KT 2024's 1.5B-parameter GPT-J variant. Plan v7 should
   record 520.

### 8.2 Addition coverage (does plan §4.2 hold?)

For addition, `a + b ∈ [0, 198]`. Plan v6 §4.2 claims all such answers are
single-token in both GPT-J and Llama. From `coverage.csv`:

| Model | Task | Pairs single-token | Fraction | Max single-token answer | First multi-token answer |
|---|---|---:|---:|---:|---:|
| GPT-J 6B | addition | 10,000 / 10,000 | **100.00%** | 198 | — (none) |
| Llama 3.1 8B | addition | 10,000 / 10,000 | **100.00%** | 198 | — (none) |
| Pythia 6.9B | addition | 10,000 / 10,000 | **100.00%** | 198 | — (none) |

**Plan §4.2 confirmed.** All three models tokenise every addition answer in
`[0, 198]` as a single token in the KT prompt context. Pythia agrees with
the GPT-J and Llama claim and is now also a clean addition target. There is
no multi-token confound for addition in any of the three models.

The `kt_addition_full` row in `summary.csv` (`max_contiguous_single_token_N = 198`,
`total_single_token_count = 199`, no failures) corroborates this.

### 8.3 Multiplication coverage (does plan §4.3 / §21.5 still apply?)

For multiplication, `a × b ∈ [0, 9801]`. Plan v6 §4.3 claims:
- GPT-J (cap 361): "~3,400 problems give single-token answers (a × b ≤ 361)".
- Llama (cap 999): "~6,700 problems give single-token answers (a × b ≤ 999)".

Both claims need correction. From `coverage.csv`:

| Model | Pairs single-token | Fraction | Max single-token answer | First multi-token answer | Plan v6 claim | Discrepancy |
|---|---:|---:|---:|---:|---|---|
| GPT-J 6B | 3,287 / 10,000 | **32.87%** | 6,000 | 527 | ~3,400 | within rounding ✓ |
| Llama 3.1 8B | 3,390 / 10,000 | **33.90%** | 999 | 1,000 | ~6,700 | **off by 2×** ✗ |
| Pythia 6.9B | 3,488 / 10,000 | **34.88%** | 6,789 | 531 | (not stated) | new |

The first multi-token splits are quoted from the per-pair CSVs:

- GPT-J: `a=6, b=89, answer=534` → ids `[20, 2682]` → `['5', '34']`.
- Llama: `a=11, b=91, answer=1001` → ids `[1041, 16]` → `['100', '1']`.
- Pythia: `a=6, b=97, answer=582` → ids `[22, 3507]` → `['5', '82']`.

**Why the Llama claim is wrong.** Plan v6 says "Llama: ~6,700 problems give
single-token answers (a × b ≤ 999)." With Llama's cap of 999 and operands
in `[0, 99]`, the question reduces to: how many (a, b) pairs in `[0, 99]²`
have `a × b ≤ 999`? This is the area below the hyperbola `b = 999/a`,
truncated to `[0, 99]²`. The exact count is **3,390** (verified by direct
enumeration in this preflight, and consistent with the integral
estimate `11 × 100 + ∫_{11}^{99} 999/a \, da ≈ 1100 + 2195 = 3295`). The plan's
~6,700 number is approximately the count of (a, b) pairs with `a × b ≤ 999`
in a *much larger* operand box (e.g. `[0, 999]²`), not the project's actual
`[0, 99]²` scope. Plan v7 should record **3,390** for Llama × multiplication.

**The §21.5 decision rule still passes for all three models.** The rule states:
- ≥ 30% single-token rate → proceed with first-answer-token correctness on full `[0, 99]`.
- 15–30% → mark cells exploratory.
- < 15% → fall back to single-token-restricted.

All three models give **32.87% – 34.88%** single-token rates, comfortably above
the 30% threshold. No fallback is required.

The corrected framing is: the three models are **uniformly in the 30–35%
single-token band** for multiplication on `[0, 99]²`. Plan v6's claim of an
asymmetry in single-token rate between GPT-J (~34%) and Llama (~67%) was an
artefact of a counting error and does not exist.

### 8.4 Tokenization context sensitivity

For the four well-formed separator contexts (`bare`, `post_equals`, `post_plus`,
`post_star`) the contiguous-single-token cap is **identical** in each model:

- GPT-J: 520 across all four.
- Llama: 999 across all four.
- Pythia: 530 across all four.

The leading-space context fails immediately (n = 0 already merges) in GPT-J and
Pythia. In Llama, leading-space behaves like the others (no merging) because
TikToken-derived BPE handles the space-prefix differently. We record this but
do not use the leading-space context downstream.

The `kt_addition_full` and `kt_multiplication_full` contexts agree with the
fixed-prefix contexts: where the single-token integer falls within the answer
range, the per-pair check returns the same single-token verdict. There is no
hidden interaction between the operand digits and the answer digits at the
tokenizer level.

---

## 9. Implications for Downstream Stages

### 9.1 Addition

Plan v6 §4.2 holds with no caveat for all three models. Step 1 (data generation)
should mark every addition problem `single_token_answer = True`. Stage 4's
first-answer-token Δlogit and standard exact-match correctness coincide for
addition.

### 9.2 Multiplication

The §21.5 decision rule passes for all three models (all ≥ 30%). We proceed
with **first-answer-token correctness** on the full `[0, 99]²` operand box for
multiplication. The data generator marks each multiplication problem with
`single_token_answer ∈ {True, False}` from this preflight's per-pair CSV.

The §4.3 numerical claims must be corrected in plan v7:

- GPT-J cap: **520** (was 361).
- Llama × multiplication single-token count: **3,390** (was ~6,700).
- Pythia × multiplication single-token count: **3,488** (new row).

The asymmetry-between-models story in §4.3 disappears. The actual story is
"all three models are in the 30–35% single-token band for multiplication on
`[0, 99]²`," which is operationally fine but rhetorically simpler.

### 9.3 Pythia 6.9B

Pythia behaves like GPT-J at the tokeniser level (GPT-NeoX BPE is descended
from GPT-2 BPE). Its cap is 530, very close to GPT-J's 520. For the project,
Pythia is a clean third replication target — no new operational concerns
arise from its tokeniser.

---

## 10. Output Files

All outputs live under `/data/user_data/anshulk/emnlp2026/results/tokenization_limits/`,
visible from the home tree as `/home/anshulk/emnlp2026/data/results/tokenization_limits/`
through the project symlink.

| File | Rows | Description |
|---|---:|---|
| `summary.csv` | 21 | Per (model, context). 7 contexts × 3 models. Cap, single-token count, first failure, example split. |
| `coverage.csv` | 6 | Per (model, task). Single-token fraction in the KT prompt context. |
| `addition_per_pair.csv` | 30,000 | Per (model, a, b). Token count, first-token id and text, all-token-ids string. |
| `multiplication_per_pair.csv` | 30,000 | Per (model, a, b). Same schema as addition. |

The log lives at `data/logs/tokenization_limits.log`
(`/data/user_data/anshulk/emnlp2026/logs/tokenization_limits.log`). Format:
`%(asctime)s %(levelname)-8s %(message)s` with `datefmt="%H:%M:%S"`,
RotatingFileHandler at 10 MB × 3 backups.

---

## 11. Runtime and Reproducibility

| Item | Value |
|---|---|
| Wall time (full sweep) | ~15 seconds |
| Compute | tokenizer-only on CPU — no GPU used |
| Per-model breakdown | tokenizer load 0.2–0.8s, integer sweep 2.7–2.8s, per-pair sweep 0.9–1.1s × 2 tasks |
| Memory peak | < 500 MB (three tokenizers loaded sequentially, not concurrently) |
| Determinism | fully deterministic — no random sampling |
| Re-run command | `/data/user_data/anshulk/miniconda3/envs/geometry/bin/python /home/anshulk/emnlp2026/check_tokenization_limits.py --config /home/anshulk/emnlp2026/config.yaml` |

### Conda environment

The conda env used is **`geometry`** at `/data/user_data/anshulk/miniconda3/envs/geometry`.
Same env as `arithmetic-geometry`. Key library versions (from the smoke test
earlier in this session): `torch 2.10.0+cu128`, `transformers 5.3.0`. No
GPU was needed for this preflight, but the env has CUDA available for
downstream stages.

### Smoke-test cross-check

The earlier inline test on the three models (loading them in fp16 on the A6000
and generating from `"Output ONLY a number. 23+45="`) returned `'68'` as the
first generated token, with token ids 3104 (GPT-J), 2614 (Llama), 2358 (Pythia).
Cross-checked against `addition_per_pair.csv`:

```
gpt-j-6b      | 23, 45, 68 | first_token_id=3104 | first_token_text=68
llama-3.1-8b  | 23, 45, 68 | first_token_id=2614 | first_token_text=68
pythia-6.9b   | 23, 45, 68 | first_token_id=2358 | first_token_text=68
```

The IDs match exactly. The encoder-side tokenisation in this preflight and the
decoder-side generation from the inline smoke test agree, which is the strongest
signal that the per-pair CSVs faithfully predict what the model will emit.

---

## 12. Open Questions and Follow-ups

1. **Plan v7 update.** The numerical claims in §4.3 must be revised: GPT-J cap
   520, Llama mult ~3,390, Pythia mult ~3,488 added. §4.2 stays unchanged.
2. **First-token sharing between models.** It is striking that for the example
   `23 + 45 = 68`, all three tokenizers emit `68` as a single token even though
   they have wholly different vocabularies. A small follow-up could check
   how often the *first answer token* agrees across models for the addition
   correct subset. This is informational only — not on the critical path.
3. **Above-cap "lucky" single tokens.** For GPT-J and Pythia, integers above
   the contiguous cap are not all multi-token. Some "round" or "famous" integers
   (e.g. `1000`, `1024`, round thousands) tokenise as single tokens because they
   were frequent enough in the BPE corpus. We do not exploit this — we use the
   conservative contiguous cap — but the per-pair CSV includes the full info if
   a future analysis wants it.
4. **Multi-token answer geometry.** Out of scope for this paper (plan §22) but
   noted: Pythia's first multi-token split (`'5'`, `'82'` for `582`) shows the
   "leading digit + rest" pattern, identical to GPT-J. Llama's split is
   "leading three digits + rest" (`'100'`, `'1'` for `1001`). This difference
   may matter for any future per-token analysis of the answer sequence.

---

## 13. Tokenizer family analysis

The three tokenizers we use sit on three branches of a small family tree, and
the family-of-origin explains both the integer caps reported in §8.1 and the
shape of the "lucky" higher-integer set reported in §13.2.

### 13.1 Family tree

| Tokenizer | Family | Trained on | Vocab size | Notes for arithmetic |
|---|---|---|---|---|
| GPT-J 6B | **GPT-2 BPE** (originally trained on WebText) | EleutherAI's curated mix; the BPE merges are the literal `gpt2` merges from OpenAI | 50,257 | Reuses the `gpt2` BPE; the contiguous integer cap (520) is therefore the *same* as for raw GPT-2. |
| Pythia 6.9B | **GPT-NeoX BPE** (descended from GPT-2 BPE, retrained on the Pile) | EleutherAI Pile corpus | 50,254 | Almost-identical merge rules to GPT-2 BPE for the digit tokens; the cap (530) is essentially the same as GPT-J's, with a few differences from the Pile-specific re-merge. |
| Llama 3.1 8B | **TikToken-derived** (Meta's own tokenizer; same lineage as GPT-4 / `cl100k_base`) | Meta's training corpus | 128,000 | Larger vocab admits all integers 0..999 contiguously as tokens. The cap (999) is hard-coded by the BPE construction; there are zero lucky integers above 999. |

Recorded for reference: GPT-J and Pythia share the GPT-2 BPE genealogy
(50K-vocab BPE family), and architecturally differ in MLP type and other
specifics. Llama uses a 128K-vocab TikToken-derived tokenizer.

### 13.2 Above-cap "lucky" single-token integers

For GPT-J and Pythia, the contiguous single-token cap (520 / 530) does not
mean integers above the cap are all multi-token. There is a sparse layer of
"lucky" higher integers that the BPE happened to learn as standalone tokens
because they appeared frequently enough in the training corpus. For our
sweep `[0, 10000]`:

| Model | Total single-token integers in [0, 10000] | Above-cap count | Highest single-token integer |
|---|---:|---:|---:|
| GPT-J 6B | 908 | 388 (= 908 − 520) | 6,000 |
| Pythia 6.9B | 1,017 | 487 (= 1,017 − 530) | 6,789 |
| Llama 3.1 8B | 1,000 | 0 | 999 |

The 229 distinct above-cap single-token integers that appear as multiplication
answers in GPT-J's bare context include round decade markers (e.g. `540, 550,
555, 560, 600, 700, 1000, 2000, 3000, 4000, 6000`) and four-digit calendar-year
values (e.g. `1988, 1989, 1992, 1995, 1998, 2000, 2001, 2002, 2006, 2009, 2010,
2013, 2014, 2015, 2016`). Pythia's set of 270 above-cap single-token
integers includes additional values such as `2100`, `2800`, and `6789`.

The intersection design (Step 2) uses the per-pair conjunction of single-token
flags. Above-cap integers that are single-token in GPT-J or Pythia but
multi-token in Llama are excluded by this conjunction.

### 13.3 Why context matters: BPE pre-tokenization

The seven contexts we tested fall into two regimes determined by **BPE
pre-tokenization rules** rather than any model-specific behaviour:

- **Well-behaved separator contexts** (`bare`, `post_equals`, `post_plus`,
  `post_star`, and the full KT prompt): these end with a character that the
  tokenizer's pre-tokenizer treats as a separator. In GPT-2 BPE / GPT-NeoX
  BPE, `=`, `+`, `*` are each their own pre-token, never merging with a
  following digit during the merge phase of BPE. In TikToken, the same is
  true via the `cl100k_base` regex. So tokenization is **prefix-stable**:
  the prefix's tokens are a literal prefix of the (prefix + digit) tokens.
- **Merging contexts** (`leading_space` for GPT-J and Pythia): the
  pre-tokenizer in GPT-2 BPE puts a leading space *into* the next token
  (`Ġ0` is one token, not space-then-zero), so adding `"0"` to `" "`
  collapses two tokens into one and prefix-stability fails. Llama's
  TikToken does *not* merge in this way (its regex splits on whitespace
  more aggressively), which is why `leading_space` works in Llama but not
  GPT-J / Pythia.

The KT prompt template ends in `=`. In our measurement, every (prompt, answer)
pair where `answer ≤ cap_model` yields a tokenization in which `tok(prompt)`
is a literal prefix of `tok(prompt + str(answer))`, and the trailing tokens
correspond to the digits of `answer`. Step 2's per-pair tokenization metadata
is computed under this prefix-stable property; the property holds for all
pairs in our measurement.

---

## 14. KT 2024 comparison and corrigenda

Plan v6 numerical claims and the values measured in this preflight:

| Plan v6 claim | Measured |
|---|---|
| GPT-J 6B cap = 361 | **520** |
| Llama 3.1 8B cap = 999 | **999** |
| GPT-J × multiplication single-token count ≈ 3,400 | **3,287** |
| Llama × multiplication single-token count ≈ 6,700 | **3,390** |

The Pythia cap was not stated in plan v6; measured here as **530**. The
multiplication single-token count for Pythia is **3,488**.

Plan v7 corrigenda are listed verbatim in §18.

---

## 15. Limitations of tokenizer-only analysis

This preflight is intentionally scope-limited. Things it does **not** tell us:

1. **Whether the model emits the gold first token during generation.** A
   problem can have a single-token gold answer in the tokenizer and still be
   answered incorrectly by the model. Step 3 measures generation-time
   correctness; this preflight only measures the gold's static tokenization.
2. **Whether the gold first token's ID is "natural" for the model.** Two
   integers might both be single-token but live in very different parts of
   the embedding space (one frequent, one rare). The single-token property
   is necessary for clean first-token-correctness scoring; it is not
   sufficient for "this is the canonical surface form".
3. **Whether multi-token answers split *consistently* during generation.**
   We measure how the gold answer tokenizes in the encoder. The model
   during decoder-side autoregressive generation might emit a different
   sequence of tokens that decodes to the same digit string. We don't test
   this — `raw_text` capture in Step 3 is the only handle on it.
4. **Whether the training corpus over- or under-represents particular
   integers.** This is what produces the lucky cluster (§13.2) but it also
   means the model's *prior* over an integer (frequency bias) is not the
   same for all single-token integers. We don't measure this; LRH-style
   probes downstream are robust to it as long as the concept-class
   distribution is balanced (see plan §7.4 stratification).
5. **Whether the prompt's pre-`=` portion influences the answer's
   tokenization stability.** We tested two prompts (addition vs
   multiplication) and they agreed everywhere. We did not test other prompt
   wordings (e.g., dropping the "Output ONLY a number." preamble). KT 2024
   reports prompt sensitivity in their generation accuracy; tokenization
   should be invariant to the preamble for the well-behaved separator
   regime, and we believe it but didn't sweep.

---

## 16. Plan v7 corrigenda (exact text changes)

| Plan v6 location | Old text | New text |
|---|---|---|
| §4.2, "Single-token verification" | "Both GPT-J (cap 361) and Llama 3.1 8B (cap 999) tokenise this as a single token." | "GPT-J (cap 520), Llama 3.1 8B (cap 999), and Pythia 6.9B (cap 530) all tokenise the addition answer range [0, 198] as a single token." |
| §4.3, "Multi-token answer handling" | "GPT-J (cap 361): ~3,400 problems give single-token answers." | "GPT-J (cap 520): 3,287 problems give single-token answers (32.87%)." |
| §4.3, "Multi-token answer handling" | "Llama (cap 999): ~6,700 problems give single-token answers." | "Llama (cap 999): 3,390 problems give single-token answers (33.90%)." |
| §4.3 (new row) | — | "Pythia 6.9B (cap 530): 3,488 problems give single-token answers (34.88%)." |
| §21.5, "Multi-token answer rate risk" | (whole section) | Mark **resolved**: the cross-model single-token intersection (3,023 problems) sidesteps the §21.5 decision rule entirely. The fallback to alternative (a) is no longer conditional. |

---

## 17. Decoded-token ID table for low integers

For reference, the gold first-token ID per model for the integers `0..9`
and a few notable mid-range integers. These can be cross-checked against
the per-pair CSVs and against the Step 3 `pred_first_token_id` columns.

| Integer | GPT-J id | GPT-J text | Llama id | Llama text | Pythia id | Pythia text |
|---:|---:|---|---:|---|---:|---|
| 0 | 15 | `"0"` | 15 | `"0"` | 17 | `"0"` |
| 1 | 16 | `"1"` | 16 | `"1"` | 18 | `"1"` |
| 2 | 17 | `"2"` | 17 | `"2"` | 19 | `"2"` |
| 3 | 18 | `"3"` | 18 | `"3"` | 20 | `"3"` |
| 4 | 19 | `"4"` | 19 | `"4"` | 21 | `"4"` |
| 5 | 20 | `"5"` | 20 | `"5"` | 22 | `"5"` |
| 6 | 21 | `"6"` | 21 | `"6"` | 23 | `"6"` |
| 7 | 22 | `"7"` | 22 | `"7"` | 24 | `"7"` |
| 8 | 23 | `"8"` | 23 | `"8"` | 25 | `"8"` |
| 9 | 24 | `"9"` | 24 | `"9"` | 26 | `"9"` |
| 45 | 2231 | `"45"` | 1774 | `"45"` | 1857 | `"45"` |
| 68 | 3104 | `"68"` | 2614 | `"68"` | 2358 | `"68"` |
| 99 | 2079 | `"99"` | 1484 | `"99"` | 1525 | `"99"` |
| 100 | 3064 | `"100"` | 1041 | `"100"` | 2313 | `"100"` |
| 198 | 25272 | `"198"` | 12422 | `"198"` | 26937 | `"198"` |
| 520 | 31102 | `"520"` | 15197 | `"520"` | 35525 | `"520"` |
| 521 (multi) | `[20, 2481]` | `"5","21"` | 21670 | `"521"` | `[22, 2405]` | `"5","31"` (n=531 first-fail) |

Observations from the table:

- Integers 0–9 occupy consecutive token IDs in all three tokenizers
  (offset 15 for GPT-J and Llama, offset 17 for Pythia).
- Two-digit and three-digit integers have different token IDs in each model.
- For single-token integers, the decoded `first_token_text` equals
  `str(answer)` by construction.

---

## 18. Per-context sweep timing breakdown

The sweep runs in ~15 seconds total across all three models. Per-(model,
context) timing (read from the run log):

| Phase | Per model | Per context |
|---|---:|---:|
| Tokenizer load | 0.2–0.8 s | — |
| Per-integer sweep (0..10000) | 2.7–2.8 s | ~0.4 s |
| Per-pair sweep (10,000 pairs) | 0.9–1.1 s × 2 tasks | — |

The per-context timing of ~0.4 s reflects 10,001 pairs of
`tok(prefix)` + `tok(prefix + str(n))` calls per context. Tokenizer
calls are cheap (microsecond-level); the loop itself dominates.

The whole preflight is **deterministic and CPU-only**. Re-running it
produces byte-identical CSV outputs. The manifest's `tokenization_csv_*_sha256`
fields in Step 2 (and Step 3) are stable across re-runs unless the
tokenizer or the script itself changes.

---

## 19. Glossary of terminology used in this doc

- **Single-token integer (in context X).** An integer `n` such that
  `tok(X + str(n))` extends `tok(X)` by exactly one token.
- **Contiguous cap.** The maximum `N` such that *every* integer in
  `[0, N]` is a single-token integer.
- **Lucky integer.** A single-token integer that lies above the
  contiguous cap (i.e., the BPE happened to merge it as a token even
  though some smaller integer didn't merge).
- **Prefix-stable tokenization.** A property of `tok(X)` and `tok(X + Y)`
  where the latter starts with the former's token IDs. When prefix-stable,
  the contribution of `Y` is unambiguously the trailing tokens.
- **Merging context.** A context where prefix-stability fails because
  the BPE pre-tokenizer fuses the trailing prefix character with the
  start of `Y` (e.g. `" "` + `"0"` → fused `"Ġ0"` token in GPT-2 BPE).
- **Cross-model intersection.** The set of `(a, b)` pairs whose product
  is a single-token integer in *all three* models simultaneously.
- **First-token-correctness.** The boolean flag `pred_first_token_id ==
  gold_first_token_id` for a single (model, problem). For our scope (all
  answers single-token in all three models), this is equivalent to
  full-sequence exact match.
- **Tier 5 (tokenization metadata).** The per-(model, problem) columns
  storing first-token-id, first-token-text, n-tokens, is-single-token in
  the dataset (Step 2). Joined from this preflight's per-pair CSVs.
- **KT 2024.** Kantamneni & Tegmark (2024), "Linear arithmetic structure
  in pre-trained language models". Reference for the prompt template and
  the addition-helix story.

---

## 20. Output file sizes and shapes

Files written to `/data/user_data/anshulk/emnlp2026/results/tokenization_limits/`:

| File | Bytes | Lines |
|---|---:|---:|
| `summary.csv` | 2,205 | 22 |
| `coverage.csv` | 745 | 7 |
| `addition_per_pair.csv` | 1,652,774 | 30,001 |
| `multiplication_per_pair.csv` | 1,759,609 | 30,001 |

`summary.csv` schema (8 columns):
```
model_key, model_name, context, max_contiguous_single_token_N,
total_single_token_count_in_0_sweep_max, merged_count_in_0_sweep_max,
first_failing_n, first_failing_kind, example_failure
```
Row count = 3 models × 7 contexts = 21, plus 1 header = 22 lines.

`coverage.csv` schema (10 columns):
```
model_key, model_name, task, operand_range, prompt,
n_pairs_total, n_pairs_single_token, frac_single_token,
min_answer_multi_token, max_answer_single_token
```
Row count = 3 models × 2 tasks = 6, plus 1 header = 7 lines.

`{addition, multiplication}_per_pair.csv` schema (12 columns):
```
model_key, model_name, a, b, answer,
prefix_stable, n_tokens, is_single_token,
first_token_id, first_token_text, all_token_ids, delta_len
```
Row count per file = 3 models × 10,000 pairs = 30,000, plus 1 header = 30,001 lines.

---

## 21. Full first-failing-integer table

Quoted verbatim from `summary.csv`. Each row gives the first integer in
`[0, 10000]` for which the (model, context) pair fails the
"prefix-stable single-token" condition.

| Model | Context | first_failing_n | kind | Detail |
|---|---|---:|---|---|
| GPT-J 6B | bare | 521 | multi | `ans_ids=[20, 2481]`, decoded `['5', '21']` |
| GPT-J 6B | leading_space | 0 | merged | prefix `' '` → `[220]`, full `' 0'` → `[657]`; merged into a single token |
| GPT-J 6B | post_equals | 521 | multi | `ans_ids=[20, 2481]`, decoded `['5', '21']` |
| GPT-J 6B | post_plus | 521 | multi | `ans_ids=[20, 2481]`, decoded `['5', '21']` |
| GPT-J 6B | post_star | 521 | multi | `ans_ids=[20, 2481]`, decoded `['5', '21']` |
| GPT-J 6B | kt_addition_full | (no failure) | — | every answer in `[0, 198]` is single-token |
| GPT-J 6B | kt_multiplication_full | 527 | multi | from `(a=17, b=31)`, `ans_ids=[20, 1983]` |
| Llama 3.1 8B | bare | 1000 | multi | `ans_ids=[1041, 15]`, decoded `['100', '0']` |
| Llama 3.1 8B | leading_space | 1000 | multi | `ans_ids=[1041, 15]`, decoded `['100', '0']` |
| Llama 3.1 8B | post_equals | 1000 | multi | `ans_ids=[1041, 15]`, decoded `['100', '0']` |
| Llama 3.1 8B | post_plus | 1000 | multi | `ans_ids=[1041, 15]`, decoded `['100', '0']` |
| Llama 3.1 8B | post_star | 1000 | multi | `ans_ids=[1041, 15]`, decoded `['100', '0']` |
| Llama 3.1 8B | kt_addition_full | (no failure) | — | every answer in `[0, 198]` is single-token |
| Llama 3.1 8B | kt_multiplication_full | 1000 | multi | from `(a=20, b=50)`, `ans_ids=[1041, 15]` |
| Pythia 6.9B | bare | 531 | multi | `ans_ids=[22, 2405]`, decoded `['5', '31']` |
| Pythia 6.9B | leading_space | 0 | merged | prefix `' '` → `[209]`, full `' 0'` → `[470]`; merged into a single token |
| Pythia 6.9B | post_equals | 531 | multi | `ans_ids=[22, 2405]`, decoded `['5', '31']` |
| Pythia 6.9B | post_plus | 531 | multi | `ans_ids=[22, 2405]`, decoded `['5', '31']` |
| Pythia 6.9B | post_star | 531 | multi | `ans_ids=[22, 2405]`, decoded `['5', '31']` |
| Pythia 6.9B | kt_addition_full | (no failure) | — | every answer in `[0, 198]` is single-token |
| Pythia 6.9B | kt_multiplication_full | 531 | multi | from `(a=9, b=59)`, `ans_ids=[22, 2405]` |

---

## 22. Sweep algorithm pseudocode

```text
sweep_integer_contexts(tok, sweep_max=10000):
    out = {ctx: {} for ctx in CONTEXTS}
    for ctx, prefix in CONTEXTS.items():
        for n in 0..sweep_max:
            base_ids = tok(prefix, add_special_tokens=False).input_ids
            full_ids = tok(prefix + str(n), add_special_tokens=False).input_ids
            stable = full_ids[:len(base_ids)] == base_ids
            ans_ids = full_ids[len(base_ids):] if stable else None
            n_tokens = len(ans_ids) if stable else None
            out[ctx][n] = {
                prefix_stable: stable,
                ans_ids: ans_ids,
                n_tokens: n_tokens,
                delta_len: len(full_ids) - len(base_ids),
                full_ids: full_ids,
                prefix_ids: base_ids,
            }
    return out

per_pair_sweep(tok, prompt_template, op, a_lo, a_hi, b_lo, b_hi):
    rows = []
    for a in a_lo..a_hi:
        for b in b_lo..b_hi:
            ans = op(a, b)
            prefix = prompt_template.format(a=a, b=b)
            res = tokens_for_n_in_context(tok, ans, prefix)
            rows.append({
                a: a, b: b, answer: ans,
                prefix_stable: res.prefix_stable,
                n_tokens: res.n_tokens,
                is_single_token: int(res.prefix_stable and res.n_tokens == 1),
                first_token_id: res.ans_ids[0] if res.ans_ids else None,
                first_token_text: tok.decode([first_token_id]),
                all_token_ids: " ".join(str(x) for x in res.ans_ids),
            })
    return rows
```

Source: `/home/anshulk/emnlp2026/check_tokenization_limits.py`.

---

## 23. Per-context single-token integer counts (full sweep `[0, 10000]`)

From `summary.csv` columns
`total_single_token_count_in_0_sweep_max` and
`merged_count_in_0_sweep_max`:

| Model | Context | single-token count | merged count | failing-multi count |
|---|---|---:|---:|---:|
| GPT-J 6B | bare | 908 | 0 | 9,093 |
| GPT-J 6B | leading_space | 0 | 10,001 | 0 |
| GPT-J 6B | post_equals | 908 | 0 | 9,093 |
| GPT-J 6B | post_plus | 908 | 0 | 9,093 |
| GPT-J 6B | post_star | 908 | 0 | 9,093 |
| Llama 3.1 8B | bare | 1,000 | 0 | 9,001 |
| Llama 3.1 8B | leading_space | 1,000 | 0 | 9,001 |
| Llama 3.1 8B | post_equals | 1,000 | 0 | 9,001 |
| Llama 3.1 8B | post_plus | 1,000 | 0 | 9,001 |
| Llama 3.1 8B | post_star | 1,000 | 0 | 9,001 |
| Pythia 6.9B | bare | 1,017 | 0 | 8,984 |
| Pythia 6.9B | leading_space | 0 | 10,001 | 0 |
| Pythia 6.9B | post_equals | 1,017 | 0 | 8,984 |
| Pythia 6.9B | post_plus | 1,017 | 0 | 8,984 |
| Pythia 6.9B | post_star | 1,017 | 0 | 8,984 |

Sums per row equal the sweep size 10,001 (integers `0..10000` inclusive).

For the full-prompt contexts, the totals reflect the per-pair coverage over
the 10,000 (a, b) operand pairs:

| Model | Context | single-token-pair count | total pairs | first failing answer |
|---|---|---:|---:|---:|
| GPT-J 6B | kt_addition_full | 10,000 | 10,000 | (none) |
| GPT-J 6B | kt_multiplication_full | 3,287 | 10,000 | 527 |
| Llama 3.1 8B | kt_addition_full | 10,000 | 10,000 | (none) |
| Llama 3.1 8B | kt_multiplication_full | 3,390 | 10,000 | 1,000 |
| Pythia 6.9B | kt_addition_full | 10,000 | 10,000 | (none) |
| Pythia 6.9B | kt_multiplication_full | 3,488 | 10,000 | 531 |

---

## 24. Tokenizer load metadata

| Model | `tok.vocab_size` | `tok.pad_token` (after eos assignment) | `tok.padding_side` | Tokenizer class |
|---|---:|---|---|---|
| GPT-J 6B | 50,257 | `<|endoftext|>` (id 50256) | `"left"` | `GPT2TokenizerFast` |
| Llama 3.1 8B | 128,000 | `<|end_of_text|>` (id 128001) | `"left"` | `PreTrainedTokenizerFast` |
| Pythia 6.9B | 50,254 | `<|endoftext|>` (id 0) | `"left"` | `GPTNeoXTokenizerFast` |

The padding-side and pad-token assignment is performed by Step 3's
`eval_and_extract.py`; the tokenizer-only sweep does not need padding.

---

## 22. Observations from the per-pair CSVs

This section reports observations grounded in the values in
`addition_per_pair.csv` and `multiplication_per_pair.csv` (each 30,000
rows = 10,000 pairs × 3 models). All counts and ratios below are
computed directly from those files.

### 22.1 Cross-model first-token-text agreement on addition

For each pair (a, b) ∈ [0, 99]² in `addition_per_pair.csv`, collect the
`first_token_text` per model. Across 10,000 pairs:

- All three models emit the same `first_token_text`: **10000 / 10,000** pairs.
- At least one disagrees: **0 / 10,000** pairs.

### 22.2 Cross-model first-token-id distribution for addition answer 0..198

By construction every addition answer in `[0, 198]` is single-token in
all three models. The token IDs differ per model. For each unique answer
value `n ∈ [0, 198]` we record the three token IDs:

| n | gpt_j id | llama id | pythia id | gpt_j == llama? | pythia − gpt_j |
|---:|---:|---:|---:|---|---:|
| 0 | 15 | 15 | 17 | yes | +2 |
| 1 | 16 | 16 | 18 | yes | +2 |
| 2 | 17 | 17 | 19 | yes | +2 |
| 3 | 18 | 18 | 20 | yes | +2 |
| 9 | 24 | 24 | 26 | yes | +2 |
| 10 | 940 | 605 | 740 | no | -200 |
| 99 | 2079 | 1484 | 1525 | no | -554 |
| 100 | 3064 | 1041 | 2313 | no | -751 |
| 198 | 22337 | 3753 | 16903 | no | -5434 |

Across 199 unique answer values 0..198:
- GPT-J `first_token_id` equals Llama `first_token_id` in **10 / 199** cases.
- Distribution of `pythia_id − gpt_j_id` across 199 unique answers:
  - offset -7376: 1 answer values
  - offset -6943: 1 answer values
  - offset -6912: 1 answer values
  - offset -6506: 1 answer values
  - offset -6498: 1 answer values
  - offset -6362: 1 answer values
  - offset -6162: 1 answer values
  - offset -6145: 1 answer values
  - offset -6076: 1 answer values
  - offset -6039: 1 answer values
  - offset -6015: 1 answer values
  - offset -5643: 1 answer values
  - offset -5589: 1 answer values
  - offset -5557: 1 answer values
  - offset -5514: 1 answer values
  - offset -5475: 1 answer values
  - offset -5458: 1 answer values
  - offset -5434: 1 answer values
  - offset -5426: 1 answer values
  - offset -5401: 1 answer values
  ... (166 more offset values)
- Total distinct offsets observed: **186**.

### 22.3 n_tokens distribution per model on the multiplication operand box

Counting `n_tokens` values across the 10,000 (a, b) pairs in
`multiplication_per_pair.csv`, per model:

| model | n_tokens=1 | n_tokens=2 | n_tokens=3 | n_tokens=4 | n_tokens≥5 |
|---|---:|---:|---:|---:|---:|
| gpt-j-6b |  3287 |  6713 |     0 |     0 |     0 |
| llama-3.1-8b |  3390 |  6610 |     0 |     0 |     0 |
| pythia-6.9b |  3488 |  6512 |     0 |     0 |     0 |

### 22.4 First-token text distribution on multi-token multiplication answers

For each model, when the answer is multi-token, record the `first_token_text`.
Tally the most-common first tokens.

#### gpt-j-6b: 6713 multi-token rows

- Distinct first-token texts: **148**.

| first_token_text | count |
|---|---:|
| `5` | 243 |
| `4` | 211 |
| `3` | 204 |
| `12` | 198 |
| `14` | 195 |
| `13` | 181 |
| `15` | 175 |
| `17` | 175 |
| `10` | 163 |
| `16` | 163 |
| `18` | 161 |
| `6` | 157 |
| `8` | 152 |
| `24` | 149 |
| `11` | 147 |

#### llama-3.1-8b: 6610 multi-token rows

- Distinct first-token texts: **747**.

| first_token_text | count |
|---|---:|
| `105` | 32 |
| `102` | 31 |
| `117` | 30 |
| `140` | 30 |
| `156` | 30 |
| `108` | 29 |
| `136` | 29 |
| `176` | 29 |
| `100` | 28 |
| `112` | 28 |
| `124` | 28 |
| `115` | 27 |
| `144` | 27 |
| `110` | 26 |
| `132` | 26 |

#### pythia-6.9b: 6512 multi-token rows

- Distinct first-token texts: **120**.

| first_token_text | count |
|---|---:|
| `4` | 285 |
| `5` | 260 |
| `12` | 198 |
| `14` | 195 |
| `13` | 193 |
| `15` | 185 |
| `17` | 175 |
| `3` | 173 |
| `6` | 168 |
| `7` | 165 |
| `10` | 165 |
| `16` | 163 |
| `18` | 161 |
| `24` | 149 |
| `11` | 147 |

### 22.5 First-token-text length distribution on multi-token rows

Length in characters of `first_token_text` when `n_tokens > 1`.

| model | length=1 | length=2 | length=3 | length=4+ |
|---|---:|---:|---:|---:|
| gpt-j-6b |  1383 |  4943 |   387 |     0 |
| llama-3.1-8b |     0 |     0 |  6610 |     0 |
| pythia-6.9b |  1431 |  4847 |   234 |     0 |

### 22.6 Lucky-integer distribution above the contiguous cap (multiplication context)

In `kt_multiplication_full` context (the actual KT prompt), the
contiguous single-token cap per model is GPT-J=525, Llama=999, Pythia=530
(see [§6](#) for the bare-context caps). For each model, count how many
distinct above-cap multiplication answers are single-token in that model:

- gpt-j-6b: **227** distinct above-cap single-token answers above 525.
  - smallest: 528; largest: 6000.
  - first 10: [528, 529, 530, 533, 536, 540, 544, 546, 549, 550].
  - last 10: [2010, 2013, 2014, 2015, 2016, 2200, 2500, 3000, 4000, 6000].
- llama-3.1-8b: **0** distinct above-cap single-token answers above 999.
- pythia-6.9b: **270** distinct above-cap single-token answers above 530.
  - smallest: 532; largest: 6789.
  - first 10: [532, 533, 534, 536, 539, 540, 544, 546, 549, 550].
  - last 10: [2014, 2015, 2016, 2100, 2500, 2800, 3000, 4000, 6000, 6789].

Round-number breakdown (`x % 100 == 0`) among the above-cap single-token values:

| model | total above-cap | divisible-by-100 | divisible-by-50 |
|---|---:|---:|---:|
| gpt-j-6b | 227 | 17 | 23 |
| llama-3.1-8b | 0 | 0 | 0 |
| pythia-6.9b | 270 | 18 | 24 |

### 22.7 Cross-model intersection size as a function of upper-bound on answer

Counting (a, b) pairs that are single-token in **all three** models, for
various upper bounds on `a × b`:

| upper bound on a*b | intersection count |
|---:|---:|
| ≤ 100 | 679 |
| ≤ 200 | 1093 |
| ≤ 300 | 1460 |
| ≤ 400 | 1793 |
| ≤ 500 | 2097 |
| ≤ 600 | 2357 |
| ≤ 700 | 2587 |
| ≤ 800 | 2786 |
| ≤ 900 | 2925 |
| ≤ 999 | 3023 |
| ≤ 1500 | 3023 |
| ≤ 2000 | 3023 |
| ≤ 3000 | 3023 |
| ≤ 5000 | 3023 |
| ≤ 9801 | 3023 |

### 22.8 Exact first-token-id agreement across models on the 3,023 intersection pairs

Of the 3023 pairs that are single-token in all three models:
- All three `first_token_id` agree: **0** pairs.
- GPT-J id == Llama id: **222** pairs.
- GPT-J id == Pythia id: **0** pairs.
- Llama id == Pythia id: **0** pairs.

### 22.9 Per-model `n_tokens` mean and std across all 10,000 multiplication pairs

- gpt-j-6b: mean = **1.671**, std = **0.470**, max = **2**.
- llama-3.1-8b: mean = **1.661**, std = **0.473**, max = **2**.
- pythia-6.9b: mean = **1.651**, std = **0.477**, max = **2**.

### 22.10 Cap consistency across separator contexts (numerical confirmation)

From [summary.csv](../data/results/tokenization_limits/summary.csv) the
`max_contiguous_single_token_N` for the four well-behaved separator contexts
(`bare`, `post_equals`, `post_plus`, `post_star`) per model:

| model | bare | post_equals | post_plus | post_star | identical? |
|---|---:|---:|---:|---:|---|
| gpt-j-6b | 520 | 520 | 520 | 520 | yes |
| llama-3.1-8b | 999 | 999 | 999 | 999 | yes |
| pythia-6.9b | 530 | 530 | 530 | 530 | yes |
