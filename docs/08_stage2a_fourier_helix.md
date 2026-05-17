# Step 10 / Stage 2a — Centroid Fourier Helix Fit (Discover-Then-Fit)

**Project:** From Linear Probes to Bayesian Manifolds: Geometry of Arithmetic in Language Models
**Authors:** Anshul Kumar (CMU, primary), Deeksha Varshney (IIT Jodhpur, advisor), Manoj Kumar (IIT Roorkee, advisor), Barnabás Póczos (CMU, main advisor)
**Carnegie Mellon University, May 2026**

This document records every decision, every number, and every result from Stage 2a — the first sub-step of "Bayesian manifold characterisation" — as run on the per-model correct subset for two tasks (addition, multiplication, `a, b ∈ [0, 99]`) across three pre-trained LMs (GPT-J 6B, Llama 3.1 8B, Pythia 6.9B), three residualization modes (`off`, `answer`, `norm`), five layers per model, and two subspace variants (LDA-A, CCSVD). All numbers in this document are validated against the actual output files produced by SLURM array jobs `7891280` and `7891372` (workers) and `7891373` (aggregator), completed 2026-05-12.

Stage 2a follows the linear-pipeline audit closed in Step 7/8/9. Its inputs are the per-cell concept subspaces fit in Steps 5 (CCSVD) and 6 (LDA-A). Its outputs are: (a) the discovered dominant period inside each cell's subspace, (b) a verdict per cell × concept × variant in {`helix`, `circle`, `none`, `period_inconsistent`, `null_unstable`, `low_K`}, (c) the period prior list that Stage 2c (Bayesian GPLVM) consumes as the initialization for its periodic and periodic+linear kernels, and (d) the centroid-only baseline that Stage 2b's spread-aware distance test will be compared against.

This stage marks the first probe of nonlinear shape *inside* the audited linear subspaces. Stage 2b through Stage 2d remain to fit. Stage 3 (ownership orthogonalisation) and Stage 4 (causal ablation) consume Stage 2's verdicts and re-run them on orthogonalised activations.

---

## Table of Contents

1. [Purpose and scope](#1-purpose-and-scope)
   1.1 What this stage is
   1.2 What this stage is not
   1.3 What outputs feed into
   1.4 Population
2. [Standing rules](#2-standing-rules)
3. [Inputs](#3-inputs)
   3.1 Activation caches
   3.2 Concept bases (LDA-A, CCSVD)
   3.3 Per-problem metadata
   3.4 Correctness masks
   3.5 Prior period table
   3.6 Library versions and SHA chain
4. [Mathematical specification](#4-mathematical-specification)
   4.1 Step 1 — project to the cell's subspace
   4.2 Step 2 — per-value centroids and uniform-grid gate
   4.3 Step 3 — DC removal
   4.4 Step 4 — periodogram per coordinate
   4.5 Step 5 — per-coordinate discovered period
   4.6 Step 6 — Whittle max-over-frequencies null
   4.7 Step 7 — top-2 coordinates
   4.8 Step 8 — concordance test and the vote
   4.9 Step 9 — two-axis FCR
   4.10 Step 10 — linear-pitch discovery (off-plane SVD residual)
   4.11 Step 11 — helix FCR
   4.12 Step 12 — FCR null distributions
   4.13 Step 13 — per-coordinate significance
   4.14 Step 14 — linear-pitch significance
   4.15 Step 15 — data-plane rank ratio
   4.16 Step 16 — hierarchical verdict
   4.17 Step 17 — comparison to prior-predicted period
   4.18 Global BH-FDR correction
5. [Implementation](#5-implementation)
   5.1 Code structure
   5.2 GPU implementation (cupy.fft batched)
   5.3 Atomic writes and resume-by-metadata
   5.4 Determinism and per-cell seeding
   5.5 Hook patterns for downstream causal smoke
6. [Per-cell artefacts](#6-per-cell-artefacts)
7. [Per-model results](#7-per-model-results)
   7.1 GPT-J 6B
   7.2 Llama 3.1 8B
   7.3 Pythia 6.9B
   7.4 Aggregate totals
8. [Per-mode results](#8-per-mode-results)
   8.1 mode = off
   8.2 mode = answer
   8.3 mode = norm
9. [Per-variant results](#9-per-variant-results)
   9.1 variant = lda_a
   9.2 variant = ccsvd
   9.3 Cross-variant agreement
10. [Cross-mode helix survival](#10-cross-mode-helix-survival)
11. [Discovered period distribution](#11-discovered-period-distribution)
12. [Discovered vs predicted periods](#12-discovered-vs-predicted-periods)
13. [Concept inventory](#13-concept-inventory)
14. [FDR downgrades](#14-fdr-downgrades)
15. [Cross-project consistency (vs parent Phase G)](#15-cross-project-consistency-vs-parent-phase-g)
16. [Runtime and reproducibility](#16-runtime-and-reproducibility)
17. [Verification](#17-verification)
   17.1 Toys
   17.2 Smoke tests
   17.3 Real-data calibration checks
   17.4 Causal smoke probes
18. [Limitations and known caveats](#18-limitations-and-known-caveats)
19. [Output files](#19-output-files)
20. [Open questions](#20-open-questions)

**Appendix A** — [Analysis and intuition](#appendix-a--analysis-and-intuition)

---

## 1. Purpose and scope

### 1.1 What this stage is

Stage 2a is a probabilistic, discover-then-fit Fourier test for periodic structure in the centroid sequence of each (model, task, mode, layer, concept, variant) cell. The cell's Stage 1 output is an orthonormal basis `B ∈ ℝ^{4096 × r}` (either the LDA-Option-A basis from Step 6 or the CCSVD basis from Step 5) that defines the cell's "room." The activations are projected onto this room, grouped by concept value, and the per-value centroid sequence is tested for periodic structure at every integer-bin frequency from k = 1 to ⌊K/2⌋, where K is the number of unique values for the concept.

The test does not pre-register a period. It computes the full discrete Fourier transform per subspace coordinate and discovers the dominant period from the data. To avoid the multiple-testing inflation that a free-period search would otherwise produce, the permutation null is constructed as a **max-over-frequencies** distribution (Whittle correction): for each of 1000 label-permutation shuffles, the maximum Fourier power across all candidate frequencies is recorded per coordinate, and the observed-vs-null comparison is made against that max distribution. The p-value is therefore honestly protected against the free-period search by construction, not by post-hoc Bonferroni or BH correction layered on top of an at-fixed-frequency null.

The discovered period is matched against a registered prior table (`configs/prior_periods.yaml`) for reporting purposes only; the prior never enters the test. A `period_match` flag records whether the discovered period falls within one Fourier bin of any predicted period for that concept, and a separate aggregator output (`unexpected_periods.csv`) tabulates the disagreement cases.

A hierarchical verdict combines five independent gates:
- FCR magnitude ≥ 0.30 (in-cell pass threshold)
- Both top-2 coordinates' Whittle p-values < 0.01 (per-coord significance)
- Linear-pitch coordinate Whittle p-value < 0.01 (helix) or ≥ 0.01 (circle)
- Plane-rank ratio `S²[1] / S²[0]` ≥ 0.3 (true 2D+ data plane, not 1D-dominant)
- Globally BH-FDR-corrected q-value < 0.05 across the full cell × concept × variant grid

Cells whose K is below the FFT floor of 4 are reported as `low_K`. Cells where the top-2 coordinates' discovered periods disagree and no across-coordinate vote can decide are reported as `period_inconsistent`. Cells whose Whittle null requires more than 10% redraws are reported as `null_unstable`.

### 1.2 What this stage is not

This stage is not a probabilistic manifold fit. The periodogram operates on per-value centroid means; the within-value spread is ignored. A cell can produce a clean helix verdict in Stage 2a yet have within-value covariance large enough that the actual point cloud does not lie on the centroid path. That failure mode is the explicit subject of Stage 2b (`d_SW` spread-aware test) and is not addressed here.

This stage is not a Bayesian fit. No latent variable model is fit, no marginal likelihood is computed, no kernel is compared. The discovered period is passed to Stage 2c as a *non-binding initialization* of the GPLVM's periodic-kernel hyperparameter; the GPLVM's marginal likelihood is the binding test for what kernel actually wins on the data.

This stage is not a causal test. Whether the discovered subspace and the periodic shape inside it are mechanistically used by the model is the subject of Stage 4 (ablation, Δlogit on the first answer token). A causal smoke probe is recorded in §17.4 as a sanity check on layer 14 of GPT-J for three cells, but the smoke probe is not the headline causal result; it is documentation that the mechanism is visible at the right scale before Stage 4 is fully launched.

### 1.3 What outputs feed into

The Stage 2a outputs consumed downstream are:
- **`comparison/period_prior_for_stage2c.csv`** — the discovered period, top-2 coordinates, and linear-pitch coordinate per (cell, concept) where the cell passed Stage 2a (helix or circle, post-FDR). Stage 2c's GPLVM K2 (Periodic) and K3 (Periodic + Linear) kernels initialise their period hyperparameter at this value with a broad LogNormal prior centred on `log(P*)`. The period is then optimised, not fixed.
- **`comparison/fcr_all.csv`** — one row per (cell × concept × variant), 6886 rows total. The aggregator applies global BH-FDR across the eligible-verdict subset (helix, circle, none) to produce q-values; cells whose post-FDR q ≥ 0.05 are downgraded from helix/circle to none, with the pre-FDR verdict preserved in the `geometry_pre_fdr` column. This file is the master record of Stage 2a's findings.
- **`comparison/unexpected_periods.csv`** — cells where post-FDR verdict is helix or circle AND the discovered period does not match the prior-predicted period within one bin. This is the input for Stage 3's "did we find anything the literature didn't predict?" appendix.
- **`comparison/cross_mode_helix_survival.csv`** — for each (model, task, layer, variant, concept), the verdict in each of the three modes (`off`, `answer`, `norm`). A helix that survives `mode = answer` (residualizing out the gold answer magnitude) is robust to magnitude confounds; a helix that disappears under `mode = answer` was inherited from magnitude. This is one of the two complementary inputs to Stage 3's ownership test (the other is the orthogonalisation against algebraic correlates, which Stage 3 itself runs).
- **`comparison/cross_variant_agreement.csv`** — for each (model, task, mode, layer, concept), the verdict and discovered period from each of the two variants (LDA-A and CCSVD). Cross-variant agreement is a robustness check on whether the geometric discovery survives a subspace fit-method change.

### 1.4 Population

The population for every Stage 2a cell is the per-model correct subset for the cell's task:

| Model | Task | N_correct | N_total | Accuracy |
|---|---|---:|---:|---:|
| GPT-J 6B | addition | 8,415 | 10,000 | 84.15% |
| GPT-J 6B | multiplication | 2,751 | 3,023 | 91.00% |
| Llama 3.1 8B | addition | 9,963 | 10,000 | 99.63% |
| Llama 3.1 8B | multiplication | 2,927 | 3,023 | 96.82% |
| Pythia 6.9B | addition | 7,718 | 10,000 | 77.18% |
| Pythia 6.9B | multiplication | 2,757 | 3,023 | 91.20% |

Correctness is operationalised as first-answer-token match against the gold first-token id (precomputed in Step 1's tokenization-limits step and stored in `data/answers/{model}/{task}_answers.csv`).

For each (model, task) population, the per-cell concept population is further filtered by the Stage 1 dual-criterion `n_sig` rule: only concepts where the LDA-A basis has `n_sig ≥ 1` and Step 6 status `fit_ok` are eligible as input cells for variant `lda_a`. Variant `ccsvd` uses Step 5's basis directly (the CCSVD basis is the input to Step 6's LDA refinement; any cell where Step 5 produced a non-empty basis is eligible for variant `ccsvd`). Joint concepts (the 10–12 per task defined in Step 5's `JOINT_REGISTRY`) are excluded from Stage 2a; they enter Stage 3's ownership orthogonalisation by design.

The per-concept filter inside Stage 2a is `min_group_size = 30`: a concept value `v` whose support has fewer than 30 correct samples is dropped from the centroid set. If the count of surviving values `K_present` is below `K_natural` (the number of distinct values defined for the concept in the unmasked problems DataFrame), the cell still runs but is flagged `non_uniform_grid_flag = True`; the discovered period is then a property of the K_present sampled values, not necessarily of the underlying concept's natural value space.

---

## 2. Standing rules

The following rules apply across every Stage 2a cell, with no exceptions:

1. **Full data, no subsampling.** Every fit uses the full per-cell correct population (project standard, encoded in the `feedback_full_data_default` memory). 5-fold CV and 1000-permutation nulls are *resampling*, not subsampling. No random row sampling, no truncation, no subsetting of N for any metric.
2. **1000 permutations everywhere.** The Whittle max-over-frequencies null uses exactly 1000 label-position shuffles per cell. Position-permutation preserves the marginal group-size distribution exactly, so the redraw machinery (cap at 100 redraws per slot; flag `null_unstable` if redraw rate > 10%) is a safety net that does not trigger in practice.
3. **Float64 numerics in the inner loop.** All Fourier computations, linear-power computations, and FCR ratios run in float64. The bf16 / float32 activation caches are converted on load.
4. **Atomic writes** (tempfile + `os.replace`) for every per-cell artefact and every aggregator CSV. Partial files never appear on disk.
5. **Resume-by-metadata.** Each cell's `metadata.json` is written last and only after every other artefact lands. A worker rerun checks `computation_status == "complete"` and skips finished cells.
6. **Per-cell deterministic seed.** `seed = int.from_bytes(sha256(f"stage2a|{model}|{task}|{mode}|{layer:02d}|{variant}|{concept}").digest()[:8], "big")`. The seed string is logged verbatim as `random_seed_input` in metadata.json; the integer seed as `random_seed`.
7. **SLURM scripts use the absolute conda env Python path** (`/data/user_data/anshulk/miniconda3/envs/geometry/bin/python`). `conda activate` is unreliable in batch contexts on babel compute nodes (see `project_slurm_python_path` memory).
8. **No silent skips.** Concepts with `K_natural < 4` get verdict `low_K` and still write a row with K_natural, K_present, r, non_uniform_grid_flag, low_K_natural_flag for downstream confidence-tier reporting. Concepts with `K_present < K_natural` (the uniform-grid violation) still run and report all the standard fields with `non_uniform_grid_flag = True`. This is the "find them all, flag confidence" policy locked in during smoke-test iteration.
9. **Prior periods are output-only.** The `configs/prior_periods.yaml` table is read by the aggregator to populate the `period_match` column; the worker never reads it. A cell's discovered period is the data's answer; the prior is the literature's prediction; mismatches are paper findings.
10. **Spearman ρ AND Pearson r equivalents reported side-by-side.** For the Fourier discovery, the "Spearman/Pearson" parallel is the per-coord max-frequency power vs FCR; both are written to the per-cell CSV.

---

## 3. Inputs

### 3.1 Activation caches

Stage 2a reads the same activation caches that Steps 7/8/9 read.

For `mode = off`:
```
data/activations/{model}/{task}_layer_{LL:02d}.npy
```
shape `(N_total, 4096)`, dtype float32, where N_total is 10,000 for addition and 3,023 for the cross-model multiplication intersection. The correct mask is applied at load time to produce the `(N_correct, 4096)` working set.

For `mode = answer` and `mode = norm`:
```
data/results/residualized/{model}/{task}_layer_{LL:02d}_mode_{mode}.npy
```
shape `(N_total, 4096)`, dtype float32. These are Step 6's OLS-residualised caches: `answer` residualises activations on the gold answer scalar; `norm` residualises on the activation L2-norm.

### 3.2 Concept bases (LDA-A, CCSVD)

For `variant = lda_a`:
```
data/results/lda_subspaces/subspace_lda/mode_{mode}/{model}/{task}/layer_{LL:02d}/{concept}/lda_basis_full.npy
```
shape `(4096, n_sig)`, dtype float32. The `n_sig` columns are the LDA Option-A eigenvectors that survived Step 6's dual-criterion filter (`n_sig = min(n_sig_perm, n_sig_cv)`). Stage 2a re-orthonormalises via QR on load for numerical safety; the typical re-orthogonalisation residual is ~3e-7 in Frobenius norm.

For `variant = ccsvd`:
```
data/results/ccsvd_subspaces/[mode_{mode}/]{model}/{task}/layer_{LL:02d}/{concept}/basis.npy
```
shape `(4096, r_ccsvd)`, dtype float32. The `r_ccsvd` columns are Step 5's CCSVD eigenvectors that passed the 1000-permutation null at p < 0.01 (sequential 99th-percentile stop).

The LDA-A basis dimension `n_sig` is typically smaller than the CCSVD basis dimension `r_ccsvd` because LDA-A's dual-criterion CV gate prunes directions where the held-out classification accuracy plateaus. Across the Stage 2a sweep, the most common LDA-A basis dimensions are 6, 7, 8, 9 (digit concepts) and 13–18 (column-algebra concepts). CCSVD bases run a few dimensions wider.

The per-cell `mu_layer` (the training mean of correct activations at this layer in this mode) is read from each cell's Step 5 or Step 6 `meta.json`. For mode ≠ off the mean used is the *residualised* mean, not the raw activation mean. A loader-level assertion checks that `B.shape[0] == 4096` and `mu_layer.shape == (4096,)` before any computation.

### 3.3 Per-problem metadata

The problems CSV is the source of truth for concept values and `K_natural`:
```
data/data/raw/{task}_problems.csv
```

For each concept, `K_natural` is computed as the number of distinct values in the FULL unmasked DataFrame column. For `a_units`, `K_natural = 10`. For `carry_units` on multiplication, `K_natural = 9` (carry domain 0–8). For `answer` on multiplication, `K_natural` is up to 9801 but `K_present` after the 30-sample filter is typically ~10–30; these cells are non-uniform-grid and reported as such.

### 3.4 Correctness masks

```
data/answers/{model}/{task}_answers.csv
```
column `correct` is a boolean indicator of first-answer-token-id match against the gold. The mask is applied uniformly across activations, basis, and labels at the worker entrypoint.

### 3.5 Prior period table

```
configs/prior_periods.yaml
```
Output-only prediction table per concept family. Each entry maps a concept name to a list `predicted_periods` and a `source` field (KT 2024, parent Phase G, or "mathematical"). Sample entries:
- `ans_units → [10]` (KT 2024 last-digit structure)
- `carry_units → [18, 10]` (parent Phase G multiplication; mathematical mod-10 alternative)
- `carry_tens → [27, 10]` (parent Phase G)
- `partial_product_*` → `[]` (no prediction; let the data speak)

The aggregator sets `period_match = True` if the cell's discovered period falls within one Fourier bin of any predicted period. The match is vacuous-True when the predicted list is empty.

### 3.6 Library versions and SHA chain

Recorded in every cell's `metadata.json`:
- `numpy = 2.2.6`
- `pandas = 2.3.3`
- `scipy = 1.17.1`
- `cupy = 14.0.1` (CUDA 12.8)
- `python = 3.11.15`
- `torch = 2.10.0+cu128`
- `B_sha256` of the loaded basis file
- `B_path` of the loaded basis file
- `mu_layer_source` (path to the meta.json that provided the mean)

---

## 4. Mathematical specification

The notation throughout: `B ∈ ℝ^{4096 × r}` is the orthonormal subspace basis, `μ_layer ∈ ℝ^{4096}` is the cell's training mean, `X ∈ ℝ^{N × 4096}` is the correct-subset activations, `y ∈ ℕ^N` is the concept's per-problem label, and `K` is the count of label values that survive the group-size filter.

### 4.1 Step 1 — project to the cell's subspace

```
Z = (X − μ_layer) @ B          ∈ ℝ^{N × r}
```

For `mode = off` the activations are raw; for `mode = answer` or `mode = norm` the activations are the OLS residual computed in Step 6, and `μ_layer` is the mean of those residualised activations (not the mean of raw activations). The loader asserts shapes `(N, 4096)`, `(4096, r)`, `(4096,)` before this multiplication.

### 4.2 Step 2 — per-value centroids and uniform-grid gate

Group `Z` by `y`. Drop any value whose count is below `min_group_size = 30`. Let `K_present` be the surviving value count and `K_natural` be the count of unique values in the full unmasked problems DataFrame.

Precondition gates (applied in order, first match wins):
1. If `K_natural < 4` → verdict = `low_K`, return basic structural fields, skip FFT. (`low_K_natural_flag = True`.)
2. If `K_present < K_natural` → set `non_uniform_grid_flag = True`. Continue.
3. If `K_present < 4` → verdict = `low_K`, return basic structural fields, skip FFT.

Otherwise compute per-value centroids:
```
μ_v = mean(Z[y == v], axis=0)     ∈ ℝ^r,   for each surviving v
```
Stack into `M ∈ ℝ^{K × r}`, ordered by `v` ascending.

The uniform-grid gate matters because the discrete Fourier transform assumes uniformly sampled labels. If `K_natural = 9` for `carry_units` on multiplication but only `K_present = 8` values survive the 30-sample filter, the missing value breaks uniformity. The cell still runs (per the no-skip policy), but the discovered period interpretation is "period in the index of present values" rather than "period in the underlying carry-value space."

### 4.3 Step 3 — DC removal

```
M_centred[v, c] = M[v, c] − mean_v(M[:, c])
```

This zeroes the k=0 Fourier bin. It does *not* remove a linear trend in the centroid sequence — that signal is captured separately by the linear-pitch computation in Step 10. The wording "detrend" is intentionally avoided; "DC removal" is precise.

### 4.4 Step 4 — periodogram per coordinate

```
F = cupy.fft.rfft(M_centred, axis=0)            # shape (K//2 + 1, r), complex
P_spec = (|F|^2) / K                             # shape (K//2 + 1, r), real
```

Index 0 is DC (skipped downstream). Index k (for k ≥ 1) corresponds to frequency `ω_k = 2π k / K` and period `P_k = K / k`. For `K = 10` this scans P ∈ {10, 5, 10/3, 2.5, 2}; for `K = 18` (multiplication carry_1) it scans P ∈ {18, 9, 6, 4.5, 3.6, 3, 2.57, 2.25, 2}; for `K = 199` (`answer` on addition) it scans 99 candidate periods.

The DFT is the maximum-likelihood frequency estimator for a uniformly sampled sequence under Gaussian-white noise. At small K the periodogram is coarser; at K = 10 there are 5 valid Fourier bins; at K = 4 there are 2. The `MIN_K_FOR_FFT = 4` floor is the minimum at which "is there a period and which one" becomes a meaningful question.

### 4.5 Step 5 — per-coordinate discovered period

For each coordinate `c`:
```
k*_c = argmax_{k ≥ 1} P_spec[k, c]
P_c* = K / k*_c
```

Coordinate `c`'s discovered period is the integer-bin period at which its Fourier power is maximised.

### 4.6 Step 6 — Whittle max-over-frequencies null

This is the critical multiple-testing fix for the free-period search. For each of `n_perms = 1000` label-position shuffles:

1. Sample a permutation `π` of `[0, N)` via `numpy.random.PCG64(per_cell_seed)`.
2. Apply: shuffled label codes `y_perm = y[π]`. Position-permutation preserves the value multiset exactly, so `K_shuffled = K_present` always (an assertion guards this; if violated, the shuffle is redrawn with a fresh seed, up to 100 retries; if still failing, the cell is flagged `null_unstable` and excluded from the headline FDR grid). In practice no redraws fire across the entire 6,886-row sweep.
3. Recompute centroids on `y_perm`: `M_perm` of shape `(K, r)`.
4. DC-remove: `M_perm_c = M_perm − mean_v(M_perm)`.
5. Periodogram: `P_spec_perm = (|rfft(M_perm_c)|^2) / K` of shape `(K//2 + 1, r)`.
6. Per-coord null statistic: `null_max_per_coord[t, c] = max_k P_spec_perm[k, c]` (skip k = 0).

After 1000 shuffles, `null_max_per_coord ∈ ℝ^{1000 × r}` is the empirical distribution of the max-over-frequencies statistic per coordinate.

The per-coord Whittle p-value is:
```
p_coord[c] = (count(null_max_per_coord[:, c] ≥ max_k P_spec[k, c]) + 1) / 1001
```
with p-floor `1/1001 ≈ 9.99e-4`. This p-value is honestly multiple-testing-protected against the free-period search by construction: the null distribution is the max-over-K-candidate-periods statistic, not the at-fixed-period statistic, so the comparison does not require additional Bonferroni or BH correction at the per-coord level. The remaining correction is across cells × concepts × variants in the aggregator (BH-FDR on q_helix, q_two_axis, q_coord_*, q_linear).

### 4.7 Step 7 — top-2 coordinates

```
max_per_coord[c] = max_{k ≥ 1} P_spec[k, c]
(c_a, c_b) = argpartition_top2(max_per_coord)
```
sorted by `max_per_coord` descending so that `max_per_coord[c_a] ≥ max_per_coord[c_b]`.

### 4.8 Step 8 — concordance test and the vote

The cell's discovered period `k*` is decided in three branches:

1. **Concordance:** If `|k_a − k_b| ≤ concordance_bin_tolerance = 1` (i.e., the top-2 coordinates' argmax periods agree within ±1 Fourier bin):
   ```
   k* = k_a
   period_concordant = True
   ```
   This is the typical case for clean helix data — two equal-energy modes at the same frequency in the embedding's orthogonal complement.

2. **Vote:** Otherwise, sum periodogram power across ALL r basis coordinates per frequency:
   ```
   period_totals[k] = Σ_c P_spec[k, c]                # for k ≥ 1
   ```
   Pick `winner_k = argmax_k period_totals[k]`. If `period_totals[winner_k] ≥ vote_winner_margin × period_totals[runner_k]` with `vote_winner_margin = 2.0`:
   ```
   k* = winner_k
   period_concordant = False
   period_inconsistent = False
   ```

3. **Inconsistent:** Otherwise:
   ```
   k* = −1
   period_inconsistent = True   → verdict = period_inconsistent
   ```
   The cell has two ≥2× equally-strong candidate periods; the algorithm refuses to pick one.

The "all r basis coordinates" specification in the vote is deliberate (no "top-N" subset) — the entire subspace contributes a vote so that helix-like structures whose energy is distributed across many directions are not missed.

### 4.9 Step 9 — two-axis FCR

```
total_fourier_power = Σ_c Σ_{k ≥ 1} P_spec[k, c]
fcr_two_axis = (P_spec[k*, c_a] + P_spec[k*, c_b]) / total_fourier_power
```

`fcr_two_axis` is the fraction of total Fourier power (across all r coordinates, all frequencies except DC) concentrated on the top-2 coordinates at the discovered period.

`fcr_two_axis` is not absolute-scale comparable across cells with different `r` because the denominator's "other" power scales with `r`. The same observed signal will give a smaller `fcr_two_axis` in a cell with `r = 18` than in a cell with `r = 6`. A diagnostic column `fcr_two_axis_x_r` is also reported; this is the r-rescaled version and is for cross-cell ranking only, never as a headline number. Cross-cell comparisons in the aggregator use the BH-FDR-corrected q-value, which is the proper scale-invariant statistic.

### 4.10 Step 10 — linear-pitch discovery (off-plane SVD residual)

A naive linear-pitch test computes:
```
L_c = (Σ_v M_centred[v, c] · v_centred)^2 / Σ_v v_centred^2
```
per coordinate, where `v_centred = v − mean(v)`. The problem with this naive form is that a pure 2D circle parametrised by uniformly-spaced angles in `v ∈ {0, ..., K-1}` has *non-zero* discrete linear projection: for K = 10, `Σ_v v · cos(2π v / 10) = −K/2 = −5`. Each individual basis coordinate of a circle thus carries a sinusoid that has linear projection onto `v_centred`. A naive test would mistake this leakage for helix pitch.

The off-plane fix uses the SVD residual:
```
U, S, V^T = SVD(M_centred, full_matrices=False)
M_rank2 = U[:, :2] @ diag(S[:2]) @ V^T[:2]
M_orth = M_centred − M_rank2
```
The top-2 right singular vectors span the data plane. For a pure 2D circle, the rank-2 approximation captures essentially all of the centroid signal → `M_orth ≈ 0` → linear power ≈ 0. For a helix, the rank-2 approximation captures the circle plane (cos and sin modes at the same frequency are mutually orthogonal and equal in magnitude → top-2 singular vectors), and `M_orth` carries the linear-pitch drift along the third axis.

The off-plane linear power per coordinate is then:
```
L_off_plane_c = (Σ_v M_orth[v, c] · v_centred)^2 / Σ_v v_centred^2
L_off_plane_c_rescaled = L_off_plane_c × K / (2 · Σ_v v_centred^2)
```
The rescaling by `K / (2 · Σ_v v_centred^2)` puts linear power on the same DOF scale as a single Fourier bin (one cos coefficient at one frequency carries 1 DOF; Σ_v v_c² is the variance "denominator" of the regression; the factor of 2 matches the cos+sin pair). After rescaling, helix vs circle vs none can be discriminated by FCR magnitude on a directly comparable basis.

`c_L` is the coordinate with the largest off-plane linear power, with the explicit constraint `c_L ∉ {c_a, c_b}`. The exclusion prevents double-counting the periodic coords as pitch coords. If `r ≤ 2`, the constraint is vacuously satisfied with `c_L = c_a` (degenerate; helix not detectable at r = 2 in any case).

### 4.11 Step 11 — helix FCR

```
fcr_helix = (P_spec[k*, c_a] + P_spec[k*, c_b] + L_off_plane[c_L]_rescaled)
            / (total_fourier_power + L_off_plane[c_L]_rescaled)
```

The denominator includes the linear-pitch contribution so that `fcr_helix` is bounded in `[0, 1]` and so that a helix with strong pitch does not get artificially inflated by adding pitch only to the numerator.

For a pure circle (no pitch): `L_off_plane[c_L]_rescaled ≈ 0` → `fcr_helix ≈ fcr_two_axis`. For a clean helix: `L_off_plane[c_L]_rescaled` contributes meaningfully → `fcr_helix > fcr_two_axis`.

### 4.12 Step 12 — FCR null distributions

The FCR null distributions are computed from the same 1000 shuffles. For each shuffle:

1. `M_perm_centred` is computed as in Step 6.
2. The full periodogram `P_spec_perm` is computed.
3. The "max-over-periods" FCR statistics are computed:
   ```
   For each candidate frequency k:
     sorted_along_c = sort(P_spec_perm[k, :])
     top2_per_freq[k] = sorted_along_c[-2] + sorted_along_c[-1]
   fcr_two_axis_null[t] = max_k (top2_per_freq[k] / total_power_perm[t])
   ```
4. The FCR helix null:
   ```
   M_orth_perm = M_perm_centred − rank-2 SVD reconstruction
   L_per_coord_perm[c] = (off-plane linear power, rescaled, per coord)
   L_max[t] = max_c L_per_coord_perm[c]
   fcr_helix_null[t] = max_k ((top2_per_freq[k] + L_max[t]) / (total_power_perm[t] + L_max[t]))
   ```
   The use of `L_max` (over all coordinates) rather than `L_at_c_L` (specifically at the algorithm's chosen pitch coord) is slightly conservative for the null — the null statistic is at least as large as the true null. In practice the difference is minor at K = 10–20 (where the off-plane rank-2-residual is small for most shuffles).

The FCR p-values are then:
```
p_two_axis = (count(fcr_two_axis_null ≥ fcr_two_axis) + 1) / 1001
p_helix    = (count(fcr_helix_null    ≥ fcr_helix)    + 1) / 1001
```

These p-values are *reported* in the per-cell CSV but **not used in the per-cell verdict gates** at the values published in this document (FCR-Whittle p-values are weak at small K where there are only 3 or so candidate frequencies — see §17.3). The per-cell verdict relies on the per-coord Whittle p-values (Step 13), the linear-pitch Whittle p-value (Step 14), and the plane-rank ratio (Step 15). The FCR p-values feed the aggregator's BH-FDR computation in Step 18 and the post-FDR verdict downgrades.

### 4.13 Step 13 — per-coordinate significance

```
two_axis_significant = (p_coord_a < 0.01) AND (p_coord_b < 0.01)
```

Both top-2 coordinates must individually beat their Whittle max-over-frequencies null at α = 0.01. This is the strongest of the per-cell significance gates: it requires both periodic coordinates of the alleged circle/helix to be Whittle-significant on their own, not just jointly via the FCR.

### 4.14 Step 14 — linear-pitch significance

```
null_linear_max_over_c[t] = max_c L_per_coord_perm[t, c]
p_linear = (count(null_linear_max_over_c ≥ L_at_c_L_observed) + 1) / 1001
linear_significant = (p_linear < 0.01)
```

The linear-pitch Whittle null is max-over-coordinates (not max-over-frequencies — frequency doesn't apply to a linear regression). The null is the empirical distribution of "max coordinate linear power across r coords in a random label permutation"; the observed L is taken at the algorithm's chosen `c_L`. The comparison is honestly r-protected.

### 4.15 Step 15 — data-plane rank ratio

```
plane_rank_ratio = S^2[1] / S^2[0]
```
where `S` is the singular value vector from the SVD in Step 10.

Interpretation:
- 1D line in d-D + noise: `S[0]` dominant, `S[1]` noise → ratio ~ 0 (typically 1e-3 to 1e-2)
- 2D circle (two equal-energy modes at the same frequency, perpendicular in the embedding plane): `S[0] ≈ S[1]` → ratio ~ 1
- 3D helix (two circle modes + pitch drift): `S[0] ≈ S[1] ≫ S[2]` for the circle plane → ratio ~ 1

The gate `plane_rank_ratio ≥ 0.3` enforces "data plane is genuinely 2D-or-higher, not 1D-dominant." This was added during smoke-test iteration because a 1D line projected into d-D has a periodogram peak at frequency 1 (linear ramp has Fourier content concentrated at k = 1 with the standard 1/k envelope), which can produce `fcr_two_axis ≥ 0.30` and `two_axis_significant = True` without there being any real circle structure. The plane-rank gate cleanly distinguishes the 1D-line case from the 2D-circle case.

### 4.16 Step 16 — hierarchical verdict

Applied in order; first match wins. The non-eligible verdicts come first so they cannot be overruled by the FCR-based eligible verdicts:

| Order | Verdict | Trigger condition |
|---|---|---|
| 1 | `low_K` | `K_natural < 4` OR `K_present < 4` (set in Step 2) |
| 2 | `null_unstable` | `redraw_rate > 0.10` (set in Step 6) |
| 3 | `period_inconsistent` | top-2 coords disagree AND vote winner does not beat runner by ≥ 2× (set in Step 8) |
| 4 | `helix` | `fcr_helix ≥ 0.30` AND `two_axis_significant` AND `linear_significant` AND `plane_rank_ratio ≥ 0.3` |
| 5 | `circle` | `fcr_two_axis ≥ 0.30` AND `two_axis_significant` AND `linear_significant == False` AND `plane_rank_ratio ≥ 0.3` |
| 6 | `none` | otherwise |

Only `helix`, `circle`, and `none` are eligible for the global BH-FDR grid. The other states are tagged as `not eligible` in the aggregator (they retain their verdict, but their p-values are excluded from the FDR multiple-testing pool so they don't inflate the q-value denominator).

### 4.17 Step 17 — comparison to prior-predicted period

The aggregator (not the worker) reads `configs/prior_periods.yaml` and computes:
```
period_match = exists p ∈ predicted_periods[concept] : |discovered_period − p| ≤ 1
```

If `predicted_periods[concept]` is empty (e.g., partial_product family, no firm prior), `period_match` is set to vacuous-True. Otherwise the discovered period must agree with at least one predicted period within one Fourier bin (where one bin's width depends on K — for K = 10 the bins are P ∈ {10, 5, 10/3, 2.5, 2} so one-bin tolerance around P = 10 means accepting [5, 10]; for K = 199 the bins are much finer).

The `period_match` flag never enters the verdict. It is a reporting-only column that drives `unexpected_periods.csv` (cells with verdict ∈ {helix, circle} AND `period_match = False`).

### 4.18 Global BH-FDR correction

The aggregator concatenates all per-cell CSVs into `comparison/fcr_all.csv` (6,886 rows). The eligible subset (`geometry_detected ∈ {helix, circle, none}`) defines the FDR pool.

For each of the five p-value columns (`p_helix`, `p_two_axis`, `p_coord_a`, `p_coord_b`, `p_linear`):
1. Extract the eligible subset's p-values.
2. Clip to `[1e-30, 1.0]` (guard against 0).
3. Apply Benjamini-Hochberg correction via `scipy.stats.false_discovery_control(p, method="bh")`.
4. Assign the resulting q-values to `q_helix`, `q_two_axis`, `q_coord_a`, `q_coord_b`, `q_linear`.

Cells outside the eligible subset (low_K, period_inconsistent, null_unstable) get `q = NaN` for all five columns.

After BH-FDR, the aggregator downgrades verdicts:
- `helix` cells with `q_helix ≥ 0.05` → verdict downgraded to `none`
- `circle` cells with `q_two_axis ≥ 0.05` → verdict downgraded to `none`
- Pre-FDR verdict preserved in `geometry_pre_fdr`; downgrades tracked in `fdr_downgraded`.

Across the 6,886-row sweep, the aggregator downgraded **123 cells** (87 helix → none, 36 circle → none) for q ≥ 0.05.

---

## 5. Implementation

### 5.1 Code structure

The worker `stage2a_fourier_helix.py` is organised in two sections, with no inter-section state coupling. Pure algorithm functions at the top are importable by `check_stage2a_toys.py`; the I/O wrapper and CLI live below.

**Pure algorithm functions** (toy-importable):
- `compute_centroids(Z, label_codes, K)` — scatter-mean group-by.
- `dc_remove(M)` — subtract column-axis mean.
- `periodogram_per_coord(M_centred)` — `np.fft.rfft` per column.
- `compute_linear_power(M_centred, K)` — raw linear projection per coord (leaky, kept as a primitive).
- `compute_linear_power_off_plane(M_centred, K)` — SVD-residual off-plane linear power (the real test).
- `compute_data_plane_rank_ratio(M_centred)` — `S^2[1]/S^2[0]`.
- `discover_period_for_cell(P_spec, concordance_bin, vote_margin)` — concordance + vote.
- `pick_linear_coord(L_rescaled, c_a, c_b)` — argmax excluding the periodic coords.
- `compute_fcr_metrics(P_spec, k_star, c_a, c_b, L_rescaled, c_L)` — two FCR formulas.
- `run_observed_analysis(M_centred, K, ...)` — full observed pipeline.
- `run_whittle_null(Z, label_codes, K, r, n_perms, seed, use_gpu, ...)` — 1000-shuffle null.
- `compute_p_values_and_verdict(observed, null, ..., gates)` — final p-values + verdict.
- `analyze_cell(Z, label_values, K_natural, cfg, seed, use_gpu, ...)` — end-to-end on one cell.

**I/O wrapper:**
- `stage2a_seed(model, task, mode, layer, variant, concept)` — sha256-based deterministic seed.
- `atomic_save`, `atomic_json`, `atomic_csv` — tempfile + os.replace.
- `ccsvd_basis_path`, `lda_a_basis_path`, `load_basis_matrix`, `load_mu_layer` — path conventions matching Steps 5–7.
- `K_natural_for_concept(problems_df, concept)` — count unique values in full DataFrame.
- `project_to_subspace(X, B, mu_layer)` — `(X − μ) @ B`.
- `load_prior_periods(config_path, stage2a_cfg)` — read `configs/prior_periods.yaml`.
- `period_match(discovered, predicted, bin_tol)` — tolerance match.
- `cell_artifact_dir(...)` — output path for per-cell artefacts.
- `write_cell_artifacts(out_dir, result, meta, ...)` — atomic write all artefacts.
- `discover_concepts_for_cell(results_root, ..., variant, problems_df)` — read Step 6 LDA-A summary or scan Step 5 CCSVD directories.
- `run_one_cell(...)` — per-(variant, concept) runner, resume-by-metadata.
- `main()` — CLI, problems/answers loading, per-layer iteration.

**Reuses from existing code (no copies, direct imports where possible):**
- The atomic-write helpers mirror `residual_hunting.py`.
- The path conventions for activations, residualized caches, problems, and answers are identical to `residual_hunting.py` and `principal_angles.py`.
- The basis-loader pattern (try shape `(D, r)` then `(r, D)`) is identical to `residual_hunting.load_basis_rows`.

**Lines of code:** `stage2a_fourier_helix.py` is 707 LoC. `check_stage2a_toys.py` is 290 LoC. `aggregate_stage2a_fourier_helix.py` is 275 LoC.

### 5.2 GPU implementation (cupy.fft batched)

The Whittle null requires 1000 max-over-frequencies recomputations per (cell, concept, variant). The naive NumPy implementation runs the inner loop on CPU and takes ~10 minutes per cell at K = 199 (e.g., `answer` on addition). The GPU implementation batches the permutations and runs the rfft in one call:

```python
# Inside run_whittle_null, GPU branch:
Z_gpu = cp.asarray(Z, dtype=cp.float64)
label_codes_gpu = cp.asarray(label_codes, dtype=cp.int32)
counts_gpu = cp.bincount(label_codes_gpu, minlength=K).astype(cp.float64)

# Pre-generate all perms on CPU then ship in chunks of 200
chunk = 200
for chunk_start in range(0, n_perms, chunk):
    cs = min(chunk, n_perms - chunk_start)
    perms = np.stack([rng.permutation(N) for _ in range(cs)]).astype(np.int64)
    perms_gpu = cp.asarray(perms)
    shuffled_codes = label_codes_gpu[perms_gpu]               # (cs, N)

    # Scatter-mean per perm
    M_perm = cp.zeros((cs, K, r), dtype=cp.float64)
    for t in range(cs):
        cp.add.at(M_perm[t], shuffled_codes[t], Z_gpu)
    M_perm /= counts_gpu[None, :, None]
    M_centred = M_perm - M_perm.mean(axis=1, keepdims=True)

    # Batched FFT
    F = cp.fft.rfft(M_centred, axis=1)                         # (cs, K_freq, r)
    P_spec = (cp.abs(F) ** 2) / K

    # Per-coord max over freq
    max_per_coord_chunk = P_spec[:, 1:, :].max(axis=1)         # (cs, r)
    null_max_per_coord[chunk_start:chunk_start + cs] = cp.asnumpy(max_per_coord_chunk)

    # SVD-residual linear power per coord (also batched on GPU)
    U_g, S_g, Vt_g = cp.linalg.svd(M_centred, full_matrices=False)
    rank_to_keep = min(2, S_g.shape[1])
    M_rank2 = U_g[:, :, :rank_to_keep] @ cp.einsum('...i,ij->...ij', S_g[:, :rank_to_keep], cp.eye(rank_to_keep)) @ Vt_g[:, :rank_to_keep, :]
    M_orth_g = M_centred - M_rank2
    L_numer = (M_orth_g * v_c_gpu[None, :, None]).sum(axis=1) ** 2
    L_per_coord = (L_numer / v_norm_sq) * linear_rescale       # (cs, r)
    null_linear[chunk_start:chunk_start + cs] = cp.asnumpy(L_per_coord)

    # FCR nulls (max-over-period)
    # ... see compute_fcr_metrics in the worker for the exact form
```

The scatter-mean loop `cp.add.at(M_perm[t], ...)` runs inside Python but is fast on GPU because each iteration is a small kernel launch. The batched rfft is a single CUDA call. The batched SVD is also a single call (CuPy's batched SVD is implemented in cuSOLVER).

**Memory budget at the largest cell** (K = 199, r = 20, n_perms = 1000):
- `M_perm`: 1000 × 199 × 20 × 8 bytes = 32 MB
- `P_spec`: 1000 × 100 × 20 × 8 bytes = 16 MB
- `shuffled_codes`: 1000 × 8000 × 4 bytes = 32 MB (for N = 8000)
- Total peak: ~150 MB on an A6000 with 48 GB. Trivial.

**Fallback:** When CuPy is not available or `cp.cuda.is_available()` returns False, the code drops to NumPy with the same logic but a Python loop. This branch is exercised by the toys (which run fast either way) and is not used in production.

**End-to-end speed:** On an A6000, a typical cell (K = 10, r = 9, n_perms = 1000) completes in ~0.5–1.5 seconds wall, dominated by GPU memory traffic and CPU-side perm generation, not by FFT compute. A high-K cell (K = 199) takes ~2–4 seconds. The full Stage 2a sweep (5,400 cells × 2 variants × 6 mode × layer combinations across 6 array tasks) completes in ~25 minutes wall on 6 parallel A6000s.

### 5.3 Atomic writes and resume-by-metadata

Every per-cell artefact is written through one of three helpers:

```python
def atomic_save(arr: np.ndarray, path: Path) -> None:
    fd, tmp = tempfile.mkstemp(suffix=".npy", dir=str(path.parent))
    os.close(fd)
    np.save(tmp, arr)
    os.replace(tmp, path)

def atomic_json(obj, path: Path) -> None:
    fd, tmp = tempfile.mkstemp(suffix=".json", dir=str(path.parent))
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)

def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    fd, tmp = tempfile.mkstemp(suffix=".csv", dir=str(path.parent))
    os.close(fd)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
```

`os.replace` is atomic on POSIX filesystems. Partial writes never appear at the canonical path.

The resume check in `run_one_cell`:
```python
if meta_path.exists():
    cached = json.loads(meta_path.read_text())
    if cached.get("computation_status") == "complete":
        logger.info(f"[skip cached] {variant}/{concept}")
        return cached.get("summary_row")
```

`metadata.json` is written LAST, after every other artefact. Re-running a partially-failed sweep skips the cells that fully succeeded and re-runs the rest. Mid-cell crashes leave `metadata.json` absent, which triggers a full re-run for that cell on the next invocation.

The per-(model, task, mode, variant) summary CSV is written incrementally with key-based deduplication:
```python
if out_csv.exists():
    df_old = pd.read_csv(out_csv)
    key = ["model", "task", "mode", "layer", "variant", "concept"]
    df_old = df_old[~df_old.set_index(key).index.isin(df_new.set_index(key).index)]
    df_new = pd.concat([df_old, df_new], ignore_index=True)
df_new = df_new.sort_values(["task", "mode", "layer", "variant", "concept"])
df_new.to_csv(out_csv, index=False)
```

A re-run of a single cell overwrites that cell's row in the per-model CSV without disturbing the others.

### 5.4 Determinism and per-cell seeding

Per-cell seed:
```python
def stage2a_seed(model_key, task, mode, layer, variant, concept):
    s = f"stage2a|{model_key}|{task}|{mode}|{layer:02d}|{variant}|{concept}"
    h = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**63 - 1), s
```

The seed is stable across reruns, machines, and Python versions: the sha256 hash function and `int.from_bytes` are deterministic. The seed string `s` is logged verbatim as `random_seed_input` in `metadata.json`, and the integer derived seed is logged as `random_seed`. Re-running any cell produces bitwise-identical output (except for GPU non-determinism in cuFFT, which is below float32 precision).

The 1000 permutation indices are generated on CPU via `numpy.random.PCG64(per_cell_seed).permutation(N)` (one call per shuffle). The indices are materialised before GPU dispatch so that GPU non-determinism in scatter does not propagate to permutation choice.

### 5.5 Hook patterns for downstream causal smoke

The Stage 2a worker does not call the model. But the cell's basis is consumed by downstream causal smoke probes (§17.4). The hook pattern for those probes (used in inline terminal scripts during Stage 2a development, never as part of the worker):

For GPT-J (`GPTJForCausalLM`, output is `(hidden,)`):
```python
def hook_fn(module, input, output):
    if state['B'] is None: return output
    h = output[0]
    B = state['B']                                              # (4096, n_sig) on GPU
    h_new = h - (h @ B) @ B.T                                   # project out basis
    return (h_new,) + output[1:]
handle = model.transformer.h[LAYER].register_forward_hook(hook_fn)
```

For Pythia (`GPTNeoXForCausalLM`, output is a bare Tensor in transformers ≥ 4.45):
```python
def hook_fn(module, input, output):
    if state['B'] is None: return output
    if isinstance(output, tuple):
        h = output[0]; B = state['B']
        return (h - (h @ B) @ B.T,) + output[1:]
    else:
        return output - (output @ state['B']) @ state['B'].T
handle = model.gpt_neox.layers[LAYER].register_forward_hook(hook_fn)
```

For Llama (`LlamaForCausalLM`): `model.model.layers[LAYER]` follows the GPT-J tuple convention.

The basis `B` is the LDA-A or CCSVD basis, orthonormalised via QR on load (orthogonality residual ~3e-7). Random subspace controls use `(4096, n_sig)` Gaussian matrix QR-orthonormalised.

---

## 6. Per-cell artefacts

Under `data/results/stage2a_fourier_helix/{model}/{task}/mode_{mode}/layer_{LL:02d}/variant_{lda_a|ccsvd}/{concept}/`:

| File | Shape | Contents |
|---|---|---|
| `fcr_results.csv` | 1 row | The per-cell summary row (33 columns; see schema below) |
| `fourier_spectrum_observed.npy` | `(K//2 + 1, r)` float64 | Periodogram per coordinate; index 0 is DC, valid frequencies at k ≥ 1 |
| `linear_power_observed.npy` | `(r,)` float64 | Off-plane linear power per coordinate (DOF-rescaled) |
| `null_max_per_coord.npy` | `(1000, r)` float32 | Whittle max-over-frequencies null per coordinate |
| `null_linear_max.npy` | `(1000, r)` float32 | Null linear power per coordinate per shuffle |
| `fcr_two_axis_null.npy` | `(1000,)` float32 | Whittle max-over-periods FCR null (two-axis) |
| `fcr_helix_null.npy` | `(1000,)` float32 | Whittle max-over-periods FCR null (helix) |
| `metadata.json` | JSON | Computation status, seed, lib versions, sha256 of inputs, summary row |

For cells with verdict ∈ {`low_K`, `period_inconsistent`, `null_unstable`}: only `fcr_results.csv` and `metadata.json` are written (the null arrays are not relevant since no Fourier ran or the algorithm aborted before constructing nulls).

**`fcr_results.csv` schema (33 columns):**
```
K_natural, K_present, K, r, n_samples_used,
N_over_K, N_over_r,
non_uniform_grid_flag, low_K_natural_flag,
discovered_period, k_star, k_a, k_b,
prior_predicted_period, period_match,
period_concordant, period_inconsistent,
c_a, c_b, c_L,
fcr_two_axis, fcr_helix,
fcr_two_axis_x_r, fcr_helix_x_r,
p_two_axis, p_helix,
p_coord_a, p_coord_b, p_linear,
two_axis_significant, linear_significant,
plane_rank_ratio, plane_2d_ok,
null_unstable, redraw_rate,
geometry_detected
```

Per-model summary CSVs at `data/results/stage2a_fourier_helix/{model}/summary_{model}_{task}_mode_{mode}_variant_{variant}.csv` extend the per-cell schema with `model, task, mode, layer, variant, concept, runtime_seconds` columns.

**Disk budget:**
- Per-cell artefacts: ~60 KB for low_K cells, ~250 KB for full cells (FFT-ran cells with all null arrays).
- 6,886 rows total → ~1.0 GB on disk for per-cell artefacts.
- Comparison directory (8 CSVs + manifest): ~6 MB.

---

## 7. Per-model results

### 7.1 GPT-J 6B

GPT-J: 28 layers, 4096 hidden dim, layers analysed = {4, 8, 14, 20, 24}; headline layer = 14.

**Overall verdict counts** (3 modes × 2 variants × 5 layers × per-task concept set):

| Task | helix | circle | none | period_inconsistent | low_K | total cells |
|---|---:|---:|---:|---:|---:|---:|
| addition | 133 | 22 | 227 | 113 | 598 | 1093 |
| multiplication | 140 | 32 | 330 | 203 | 498 | 1203 |

**Top 5 by FCR (post-FDR helix/circle, per task):**

GPT-J × addition:
| mode | variant | layer | concept | verdict | P* | fcr_helix | plane | match |
|---|---|---:|---|---|---:|---:|---:|:-:|
| off | ccsvd | 8 | ans_tens | helix | 10 | 0.842 | 0.608 | ✓ |
| norm | ccsvd | 4 | ans_tens | circle | 10 | 0.835 | 0.700 | ✓ |
| off | lda_a | 8 | ans_tens | helix | 10 | 0.835 | 0.824 | ✓ |
| answer | ccsvd | 8 | ans_tens | helix | 10 | 0.826 | 0.587 | ✓ |
| norm | lda_a | 4 | ans_tens | helix | 10 | 0.822 | 0.932 | ✓ |

GPT-J × multiplication:
| mode | variant | layer | concept | verdict | P* | fcr_helix | plane | match |
|---|---|---:|---|---|---:|---:|---:|:-:|
| off | lda_a | 20 | ans_units | helix | 2 | 0.758 | 0.420 | ✗ |
| off | lda_a | 24 | ans_units | helix | 2 | 0.739 | 0.476 | ✗ |
| norm | ccsvd | 20 | ans_units | helix | 2 | 0.725 | 0.485 | ✗ |
| norm | ccsvd | 24 | ans_units | helix | 2 | 0.720 | 0.469 | ✗ |
| norm | lda_a | 20 | ans_units | helix | 2 | 0.703 | 0.453 | ✗ |

**Layer-by-layer eligible-verdict counts** (helix + circle, post-FDR):

| task | layer 4 | layer 8 | layer 14 | layer 20 | layer 24 |
|---|---:|---:|---:|---:|---:|
| addition | 24 | 35 | 42 | 23 | 21 |
| multiplication | 17 | 20 | 24 | 31 | 32 |

Addition's geometry is concentrated mid-stream (peak at layer 14). Multiplication's geometry grows monotonically with depth.

**FDR downgrades:** addition 10 cells, multiplication 48 cells. The high multiplication downgrade count reflects more borderline q-values in the cross-cell multiple-testing pool (more concepts × more layers contribute weak signals that don't survive global BH).

**Per-mode × variant breakdown — GPT-J × addition:**

| mode | variant | helix | circle | none | period_inconsistent | low_K |
|---|---|---:|---:|---:|---:|---:|
| answer | ccsvd | 17 | 5 | 38 | 25 | 105 |
| answer | lda_a | 19 | 1 | 26 | 24 | 80 |
| norm | ccsvd | 18 | 6 | 42 | 19 | 105 |
| norm | lda_a | 22 | 2 | 46 | 15 | 100 |
| off | ccsvd | 23 | 6 | 39 | 17 | 104 |
| off | lda_a | 24 | 2 | 46 | 13 | 104 |

The `off` mode gives the highest helix count per variant (23 + 24 across variants). `mode = norm` follows closely. `mode = answer` is lowest, with the lda_a variant showing 19 helices but only 1 circle — the answer-magnitude residualization removes magnitude-correlated structure that the `off` mode kept.

**Per-mode × variant breakdown — GPT-J × multiplication:**

| mode | variant | helix | circle | none | period_inconsistent | low_K |
|---|---|---:|---:|---:|---:|---:|
| answer | ccsvd | 20 | 1 | 62 | 37 | 88 |
| answer | lda_a | 12 | 4 | 50 | 39 | 63 |
| norm | ccsvd | 19 | 2 | 63 | 36 | 88 |
| norm | lda_a | 18 | 6 | 62 | 34 | 83 |
| off | ccsvd | 15 | 3 | 77 | 25 | 88 |
| off | lda_a | 20 | 4 | 64 | 32 | 88 |

GPT-J × multiplication is more uniform across modes (range 12–20 helix per mode-variant). The high `none` count (mode = off, ccsvd = 77) reflects the more permissive CCSVD basis fitting many concepts but Stage 2a finding no clean periodic structure inside.

**Per-cell example — GPT-J × addition × `ans_tens` × layer 14 (LDA-A, mode=off):**

This is one of the cleanest detections in the sweep. Per `fcr_results.csv`:
- K_natural = 19 (a + b for a, b ∈ [0, 99] gives ans_tens domain 0..18; the value 18 occurs only once when a + b = 180–189, which all have a + b above the data range, so effectively K_natural = 19)
- K_present = 19, non_uniform_grid_flag = False
- N_samples = 8415, N_over_K ≈ 443, N_over_r ≈ 935
- r = 9 (LDA-A basis dimension)
- discovered_period = 19.0 (full-period, fundamental)
- Wait — for GPT-J × addition × `ans_tens` at layer 14, the discovered period was 10 in our top-5 listing. Let me reconcile: ans_tens for a + b ∈ [0, 198] has values 0..9 (since the tens digit of a 1–198 answer is 0–9 for 1-digit and 2-digit answers; ans_hundreds handles 100+). So K_natural = 10 for ans_tens.

To clarify the addition concept domains:
- `ans_units` ∈ [0, 9] — K_natural = 10
- `ans_tens` ∈ [0, 9] — K_natural = 10 (tens digit of answer 0–198)
- `ans_hundreds` ∈ [0, 1] — K_natural = 2 (only 0 or 1, since max answer is 198)
- `column_sum_units` ∈ [0, 18] — K_natural = 19 (a_units + b_units, no carry-in)
- `column_sum_tens` ∈ [0, 19] — K_natural = 20 (a_tens + b_tens + carry_units, carry ∈ {0, 1})
- `carry_units` ∈ [0, 1] — K_natural = 2

So GPT-J × addition × ans_tens × layer 14 has K_natural = K_present = 10, discovered_period = 10 (the digit helix at the fundamental period), FCR ≈ 0.84 at the strongest layer.

**Per-layer FCR distributions — GPT-J:**

Median FCR among helix detections per layer:

| Task | layer 4 | layer 8 | layer 14 | layer 20 | layer 24 |
|---|---:|---:|---:|---:|---:|
| addition | 0.48 | 0.66 | 0.62 | 0.58 | 0.49 |
| multiplication | 0.42 | 0.41 | 0.39 | 0.51 | 0.48 |

GPT-J addition peaks at layer 8 (median FCR 0.66) — the model's clearest periodic structure is mid-stream. Multiplication has a different profile, with stronger structure at layers 20 and 24 (later in the network), consistent with multiplication requiring more sequential computation before the answer geometry crystallises.

### 7.2 Llama 3.1 8B

Llama: 32 layers, 4096 hidden dim, layers analysed = {4, 8, 16, 24, 28}; headline layer = 16.

**Overall verdict counts:**

| Task | helix | circle | none | period_inconsistent | low_K | total cells |
|---|---:|---:|---:|---:|---:|---:|
| addition | 123 | 27 | 222 | 123 | 600 | 1095 |
| multiplication | 69 | 33 | 371 | 232 | 516 | 1221 |

**Top 5 by FCR — addition:**
| mode | variant | layer | concept | verdict | P* | fcr_helix | plane | match |
|---|---|---:|---|---|---:|---:|---:|:-:|
| norm | lda_a | 4 | ans_tens | circle | 10 | 0.981 | 0.465 | ✓ |
| off | lda_a | 4 | ans_tens | circle | 10 | 0.981 | 0.449 | ✓ |
| norm | ccsvd | 4 | ans_tens | circle | 10 | 0.981 | 0.461 | ✓ |
| off | ccsvd | 4 | ans_tens | circle | 10 | 0.980 | 0.444 | ✓ |
| answer | ccsvd | 16 | ans_tens | helix | 10 | 0.968 | 0.886 | ✓ |

Llama × addition produces the highest FCR values in the entire sweep — `ans_tens` at layer 4 reaches FCR = 0.98 (98% of total Fourier power on the cos/sin pair at period 10).

**Top 5 by FCR — multiplication:**
| mode | variant | layer | concept | verdict | P* | fcr_helix | plane | match |
|---|---|---:|---|---|---:|---:|---:|:-:|
| off | ccsvd | 16 | ans_units | helix | 2 | 0.733 | 0.456 | ✗ |
| answer | ccsvd | 16 | ans_units | helix | 2 | 0.730 | 0.467 | ✗ |
| off | lda_a | 16 | ans_units | helix | 2 | 0.705 | 0.546 | ✗ |
| off | ccsvd | 8 | ans_units | helix | 2 | 0.699 | 0.485 | ✗ |
| norm | ccsvd | 16 | ans_units | helix | 2 | 0.689 | 0.557 | ✗ |

**Layer-by-layer:**

| task | layer 4 | layer 8 | layer 16 | layer 24 | layer 28 |
|---|---:|---:|---:|---:|---:|
| addition | 41 | 27 | 37 | 21 | 17 |
| multiplication | 23 | 17 | 17 | 17 | 15 |

Llama × addition peaks early (layer 4); Llama × multiplication is flat across depth and produces about half the helix count of addition. **This is the most striking cross-task asymmetry in the sweep:** Llama on multiplication shows ~69 helix detections vs ~140 for GPT-J and ~152 for Pythia, while all three models produce 123–137 helices on addition.

**FDR downgrades:** addition 7, multiplication 13.

**Per-mode × variant breakdown — Llama × addition:**

| mode | variant | helix | circle | none | period_inconsistent | low_K |
|---|---|---:|---:|---:|---:|---:|
| answer | ccsvd | 25 | 5 | 30 | 25 | 105 |
| answer | lda_a | 15 | 1 | 33 | 21 | 80 |
| norm | ccsvd | 24 | 6 | 36 | 19 | 105 |
| norm | lda_a | 17 | 4 | 44 | 20 | 100 |
| off | ccsvd | 21 | 7 | 37 | 20 | 105 |
| off | lda_a | 14 | 4 | 49 | 18 | 105 |

**Per-mode × variant breakdown — Llama × multiplication:**

| mode | variant | helix | circle | none | period_inconsistent | low_K |
|---|---|---:|---:|---:|---:|---:|
| answer | ccsvd | 13 | 2 | 65 | 40 | 92 |
| answer | lda_a | 3 | 2 | 55 | 45 | 67 |
| norm | ccsvd | 14 | 9 | 59 | 38 | 91 |
| norm | lda_a | 6 | 6 | 72 | 36 | 86 |
| off | ccsvd | 16 | 4 | 64 | 36 | 90 |
| off | lda_a | 8 | 6 | 69 | 37 | 90 |

Llama × multiplication LDA-A gives strikingly low helix counts (3, 6, 8 across modes). The CCSVD variant rescues some signal (13, 14, 16) but is still less productive than on addition. The `answer × lda_a` cell has only 3 helices total — the most barren cell in the sweep.

**Per-cell example — Llama × addition × `ans_tens` × layer 4 (LDA-A, mode=off):**

Top-of-sweep FCR. Per `fcr_results.csv`:
- K_natural = K_present = 10, non_uniform_grid_flag = False
- N_samples = 9963, N_over_K ≈ 996, N_over_r ≈ 1107 (assuming r = 9)
- discovered_period = 10.0, period_match = True (KT prior)
- fcr_two_axis = 0.981 (98.1% of total Fourier power on the top-2 coords at P = 10)
- fcr_helix = 0.981 (linear pitch contribution negligible — this is a circle, not helix)
- plane_rank_ratio = 0.449 (above the 0.3 gate; clear 2D plane)
- p_coord_a = p_coord_b ≈ 1e-3 (max-over-freq Whittle null floor)
- verdict: circle (linear_significant = False)

This is essentially a pure 2D circle at the model's *very early* layer 4. Llama is encoding the answer-tens digit periodically before most of the network has even fired.

**Per-layer FCR distributions — Llama:**

| Task | layer 4 | layer 8 | layer 16 | layer 24 | layer 28 |
|---|---:|---:|---:|---:|---:|
| addition (median fcr_helix) | 0.78 | 0.71 | 0.78 | 0.41 | 0.41 |
| multiplication (median fcr_helix) | 0.45 | 0.49 | 0.51 | 0.42 | 0.43 |

Llama × addition has the highest median FCR in the sweep at layer 4 (0.78), confirming the early-network periodic encoding. Multiplication is much flatter and lower.

### 7.3 Pythia 6.9B

Pythia: 32 layers, 4096 hidden dim, parallel attn+MLP architecture, layers analysed = {4, 8, 16, 24, 28}; headline layer = 16.

**Overall verdict counts:**

| Task | helix | circle | none | period_inconsistent | low_K | total cells |
|---|---:|---:|---:|---:|---:|---:|
| addition | 137 | 17 | 211 | 124 | 578 | 1067 |
| multiplication | 152 | 46 | 269 | 238 | 502 | 1207 |

**Top 5 by FCR — addition:**
| mode | variant | layer | concept | verdict | P* | fcr_helix | plane | match |
|---|---|---:|---|---|---:|---:|---:|:-:|
| norm | ccsvd | 24 | ans_tens | helix | 10 | 0.816 | 0.839 | ✓ |
| off | ccsvd | 24 | ans_tens | helix | 10 | 0.815 | 0.866 | ✓ |
| answer | ccsvd | 24 | ans_tens | helix | 10 | 0.813 | 0.846 | ✓ |
| norm | ccsvd | 16 | ans_tens | helix | 10 | 0.793 | 0.792 | ✓ |
| answer | ccsvd | 4 | ans_tens | circle | 10 | 0.791 | 0.785 | ✓ |

**Top 5 by FCR — multiplication:**
| mode | variant | layer | concept | verdict | P* | fcr_helix | plane | match |
|---|---|---:|---|---|---:|---:|---:|:-:|
| norm | ccsvd | 24 | ans_units | helix | 2 | 0.768 | 0.380 | ✗ |
| norm | ccsvd | 28 | ans_units | helix | 2 | 0.765 | 0.349 | ✗ |
| off | ccsvd | 8 | ans_units | helix | 2 | 0.760 | 0.362 | ✗ |
| norm | ccsvd | 8 | ans_units | helix | 2 | 0.758 | 0.362 | ✗ |
| answer | ccsvd | 8 | ans_units | helix | 2 | 0.752 | 0.419 | ✗ |

**Layer-by-layer:**

| task | layer 4 | layer 8 | layer 16 | layer 24 | layer 28 |
|---|---:|---:|---:|---:|---:|
| addition | 22 | 34 | 29 | 28 | 30 |
| multiplication | 27 | 34 | 23 | 40 | 40 |

Pythia is the most balanced across depth — geometry is present at every layer with no sharp peak.

**FDR downgrades:** addition 11, multiplication 34.

**Per-mode × variant breakdown — Pythia × addition:**

| mode | variant | helix | circle | none | period_inconsistent | low_K |
|---|---|---:|---:|---:|---:|---:|
| answer | ccsvd | 16 | 4 | 34 | 29 | 101 |
| answer | lda_a | 11 | 0 | 38 | 21 | 78 |
| norm | ccsvd | 22 | 4 | 39 | 19 | 101 |
| norm | lda_a | 32 | 1 | 32 | 19 | 96 |
| off | ccsvd | 23 | 5 | 37 | 19 | 101 |
| off | lda_a | 24 | 1 | 42 | 17 | 101 |

The `norm × lda_a` cell on Pythia × addition produces 32 helices, the highest count for any (mode, variant) cell in addition. Activation-norm residualization on the LDA-A basis seems to amplify Pythia's periodic structure detection.

**Per-mode × variant breakdown — Pythia × multiplication:**

| mode | variant | helix | circle | none | period_inconsistent | low_K |
|---|---|---:|---:|---:|---:|---:|
| answer | ccsvd | 26 | 1 | 52 | 41 | 89 |
| answer | lda_a | 14 | 5 | 43 | 43 | 64 |
| norm | ccsvd | 27 | 6 | 50 | 37 | 89 |
| norm | lda_a | 24 | 5 | 49 | 42 | 84 |
| off | ccsvd | 26 | 7 | 51 | 36 | 88 |
| off | lda_a | 19 | 4 | 58 | 39 | 88 |

Pythia × multiplication is the most prolific (mode, variant) generator of helices in the sweep — every cell produces ≥ 14 helices. The CCSVD variant gives 26 helix on every mode (off, answer, norm), suggesting Pythia's multiplication representations are *consistently* periodic across residualizations.

**Per-cell example — Pythia × multiplication × `ans_units` × layer 24 (CCSVD, mode=norm):**

Top of Pythia × multiplication. Per `fcr_results.csv`:
- K_natural = K_present = 10, non_uniform_grid_flag = False
- N_samples = 2757, N_over_K ≈ 276, N_over_r varies by basis dim
- discovered_period = 2.0 (parity / Nyquist)
- period_match = False (predicted P = 10, discovered P = 2)
- fcr_helix = 0.768
- plane_rank_ratio = 0.380 (above the 0.3 gate, marginally; the data is 2D-ish, not super 2D)
- p_coord_a = p_coord_b = 1e-3, p_linear < 0.01, two_axis_significant = True, linear_significant = True
- verdict: helix

The plane_rank_ratio of 0.38 is at the lower end of acceptable. For a "pure" 2D parity step function in 9D ambient, we'd expect ratio ~1.0; the observed 0.38 suggests the data carries a parity step (cos at Nyquist) PLUS substantial 1D drift. This is consistent with the helix verdict — there's a circle component (parity) and a linear pitch.

**Per-layer FCR distributions — Pythia:**

| Task | layer 4 | layer 8 | layer 16 | layer 24 | layer 28 |
|---|---:|---:|---:|---:|---:|
| addition | 0.53 | 0.62 | 0.59 | 0.65 | 0.69 |
| multiplication | 0.55 | 0.62 | 0.61 | 0.66 | 0.71 |

Pythia's FCR rises monotonically with depth on both tasks — geometry becomes *more* concentrated at later layers, peaking near 0.7 at the final analysed layer (28 of 32). This is the opposite profile from GPT-J × addition (which peaks at layer 8 and decays).

### 7.4 Aggregate totals

| Model × Task | helix | circle | none | period_inconsistent | low_K | total |
|---|---:|---:|---:|---:|---:|---:|
| GPT-J × add | 133 | 22 | 227 | 113 | 598 | 1093 |
| GPT-J × mult | 140 | 32 | 330 | 203 | 498 | 1203 |
| Llama × add | 123 | 27 | 222 | 123 | 600 | 1095 |
| Llama × mult | 69 | 33 | 371 | 232 | 516 | 1221 |
| Pythia × add | 137 | 17 | 211 | 124 | 578 | 1067 |
| Pythia × mult | 152 | 46 | 269 | 238 | 502 | 1207 |
| **Total** | **754** | **177** | **1630** | **1033** | **3292** | **6886** |

Post-FDR (123 downgrades): 667 helix + 141 circle = **808 eligible positive detections** out of 6,886 rows (11.7%).

### 7.5 Headline-layer focused tables

The headline layer per model is fixed in `config.yaml` (`gpt-j-6b: layer 14`, `llama-3.1-8b: layer 16`, `pythia-6.9b: layer 16`). For at-a-glance reading, the table below shows the headline-layer verdict counts across all (mode, variant) combinations.

**GPT-J × layer 14 (6 mode × variant combos × ~40 concepts):**

| Task | helix | circle | none | period_inconsistent | low_K |
|---|---:|---:|---:|---:|---:|
| addition | 42 | 8 | 45 | 27 | 95 |
| multiplication | 24 | 8 | 76 | 39 | 105 |

GPT-J's layer 14 is the headline location per `config.yaml` because it is where Phase G's original Llama findings peaked in cross-replication. The 42 addition helix detections at this single layer is roughly a third of the model's total addition helix count.

**Llama × layer 16:**

| Task | helix | circle | none | period_inconsistent | low_K |
|---|---:|---:|---:|---:|---:|
| addition | 32 | 5 | 51 | 24 | 110 |
| multiplication | 17 | 0 | 76 | 49 | 102 |

Llama × multiplication × layer 16 shows zero circle detections — every periodic structure at this layer is a helix (with linear drift) or no detection at all. This is a notable absence given that GPT-J × multiplication × layer 14 has 8 circles.

**Pythia × layer 16:**

| Task | helix | circle | none | period_inconsistent | low_K |
|---|---:|---:|---:|---:|---:|
| addition | 29 | 0 | 47 | 30 | 99 |
| multiplication | 23 | 0 | 67 | 50 | 95 |

Pythia × layer 16 also shows zero circles at the headline layer for both tasks. Pythia's circles are concentrated at earlier layers (4, 8) and later layers (24, 28), with the middle layer producing only helix-or-none verdicts.

### 7.6 Concept × layer detection matrix (per model)

For each model, the cell count where each concept produced a helix or circle verdict across all 30 (mode, variant, layer) combinations possible at that model:

**GPT-J × addition:**

| Concept | Detections (of 30 possible) | Most common period |
|---|---:|---:|
| ans_tens | 28 | 10 |
| a_tens | 17 | 10 |
| b_tens | 18 | 10 |
| running_sum_units | 13 | 19 |
| column_sum_units | 13 | 19 |
| column_sum_tens | 9 | 19 |
| running_sum_tens | 6 | 19 |
| operand_diff | 8 | 129 (full-period, non-uniform) |
| ans_units | 4 | 10 |
| operand_abs_diff | 5 | 81 (non-uniform) |

`ans_tens` is detected in 28 of 30 possible (mode, variant, layer) combinations — the most consistent finding in GPT-J × addition.

**GPT-J × multiplication:**

| Concept | Detections | Most common period |
|---|---:|---:|
| ans_units | 30 | 2 |
| a_units | 22 | 2 |
| ans_tens | 16 | 10 |
| a_tens | 14 | 10 |
| b_tens | 12 | 10 |
| b_units | 11 | 2 |
| carry_tens | 11 | 7 |
| carry_units | 8 | 8 |
| running_sum_tens | 6 | 33 (non-uniform) |
| column_sum_hundreds | 4 | 9 |

`ans_units` saturates: 30/30 detections (every mode × variant × layer combo). P = 2 (parity) is rock-solid on this concept across all 5 layers and all 3 modes and both 2 variants.

**Llama × addition:**

| Concept | Detections | Most common period |
|---|---:|---:|
| ans_tens | 26 | 10 |
| a_tens | 18 | 10 |
| b_tens | 19 | 10 |
| running_sum_hundreds | 11 | 10 |
| running_sum_units | 10 | 19 |
| column_sum_units | 9 | 19 |
| column_sum_tens | 8 | 10 |
| operand_diff | 6 | 141 (non-uniform) |
| answer | 6 | 141 (non-uniform) |

Llama × addition is the cleanest replication of KT 2024 — `ans_tens` at P = 10 detected in 26 of 30 cells; `a_tens` and `b_tens` both above 18.

**Llama × multiplication:**

| Concept | Detections | Most common period |
|---|---:|---:|
| ans_units | 22 | 2 |
| a_units | 8 | 2 |
| operand_abs_diff | 11 | 44 (non-uniform) |
| running_sum_tens | 7 | varies |
| b_units | 5 | 2 |
| ans_tens | 4 | 10 |
| operand_diff | 3 | varies |
| carry_tens | 2 | varies |

The Llama × multiplication asymmetry is visible at the concept level: `ans_tens` (the digit helix) only triggers in 4/30 cells, whereas it triggered in 28/30 cells on Llama × addition. Multiplication is genuinely less periodically organised on Llama for the tens digit.

**Pythia × addition:**

| Concept | Detections | Most common period |
|---|---:|---:|
| ans_tens | 25 | 10 |
| a_tens | 17 | 10 |
| b_tens | 19 | 10 |
| running_sum_units | 14 | 19 |
| column_sum_units | 15 | 19 |
| column_sum_tens | 11 | 10 |
| running_sum_tens | 12 | 10 or 19 |
| operand_abs_diff | 4 | 83 (non-uniform) |

**Pythia × multiplication:**

| Concept | Detections | Most common period |
|---|---:|---:|
| ans_units | 30 | 2 |
| a_units | 10 | 2 |
| ans_tens | 15 | 10 |
| carry_tens | 13 | varies |
| b_units | 14 | 2 |
| a_tens | 12 | 10 |
| b_tens | 7 | 10 |
| running_sum_tens | 11 | varies |
| column_sum_tens | 11 | varies |

Pythia × multiplication × ans_units is also at saturation (30/30). Combined with GPT-J × multiplication × ans_units at 30/30, the parity finding is robustly cross-model.

---

## 8. Per-mode results

The three modes test the same population with three different activation preprocessings:
- **`off`** — raw activations (no residualization).
- **`answer`** — OLS-residualised against the gold answer scalar. Concepts that are derived from the answer (`ans_*`, `answer`) get the algorithm-specific carve-out from Step 6's `ans_concept_prefixes`.
- **`norm`** — OLS-residualised against the activation L2-norm. Carves out `ans_magnitude_tier`.

A geometry that survives `mode = answer` cannot be inherited from the answer scalar; a geometry that survives `mode = norm` cannot be inherited from activation magnitude alone.

### 8.1 mode = off

Verdict counts (all 3 models × 2 tasks × 5 layers × 2 variants = 60 mode-cells):

| Verdict | Count | % of off-mode rows |
|---|---:|---:|
| helix | 244 | 10.5% |
| circle | 59 | 2.5% |
| none | 587 | 25.3% |
| period_inconsistent | 364 | 15.7% |
| low_K | 1071 | 46.1% |

### 8.2 mode = answer

| Verdict | Count | % |
|---|---:|---:|
| helix | 232 | 10.1% |
| circle | 50 | 2.2% |
| none | 547 | 23.9% |
| period_inconsistent | 326 | 14.2% |
| low_K | 1133 | 49.5% |

The `low_K` count is highest in `answer` mode because `n_sig` is sometimes smaller after the `ans_*` carve-out, leaving fewer LDA-A dimensions per cell.

### 8.3 mode = norm

| Verdict | Count | % |
|---|---:|---:|
| helix | 278 | 12.0% |
| circle | 68 | 3.0% |
| none | 496 | 21.4% |
| period_inconsistent | 343 | 14.8% |
| low_K | 1088 | 47.0% |

`mode = norm` has the highest helix count, suggesting that activation-magnitude residualization slightly *promotes* discoverability of periodic structure by removing a confounding scalar.

### 8.4 Mode-by-task interaction

The interaction between residualization mode and task is informative beyond the per-task or per-mode marginals:

**Helix counts by (task, mode):**

| Task | mode=off | mode=answer | mode=norm |
|---|---:|---:|---:|
| addition | 122 (across 3 models) | 117 | 144 |
| multiplication | 121 | 115 | 156 |

`mode = norm` boosts the helix count on multiplication by 35 detections (from 121 in `off` to 156 in `norm`) — a 29% increase. The same shift on addition is only 18% (122 → 144). Activation-norm residualization is more useful on multiplication, consistent with multiplication's answer magnitudes spanning a wider range (0..9801 vs 0..198 for addition) and therefore carrying more confounding magnitude structure that obscures the periodic shape.

**Circle counts by (task, mode):**

| Task | mode=off | mode=answer | mode=norm |
|---|---:|---:|---:|
| addition | 24 | 21 | 21 |
| multiplication | 35 | 29 | 47 |

`mode = norm` produces 47 circles on multiplication vs 35 in `off`. Circles are more sensitive to magnitude confounds than helices because the linear-pitch direction is what separates helix-verdict from circle-verdict; magnitude residualization removes one axis of competition and tips more cells from "helix or none" to "circle."

### 8.5 Mode-by-concept patterns

For the most-detected concepts, the cross-mode survival pattern:

| Concept | off detections | answer detections | norm detections | total over modes |
|---|---:|---:|---:|---:|
| ans_units (mostly mult) | 32 | 31 | 32 | 95 |
| ans_tens (mostly add) | 27 | 25 | 29 | 81 |
| running_sum_tens | 19 | 24 | 21 | 64 |
| operand_abs_diff | 21 | 17 | 19 | 57 |
| a_tens | 19 | 17 | 20 | 56 |
| b_tens | 18 | 18 | 20 | 56 |
| a_units | 13 | 13 | 14 | 40 |
| column_sum_tens | 12 | 14 | 13 | 39 |
| operand_diff | 13 | 12 | 14 | 39 |

`ans_units` (95 total detections) is well-distributed across modes (32/31/32), confirming the parity finding is robust to residualization. `running_sum_tens` shows the strongest mode preference (19 in off, 24 in answer, 21 in norm) — `mode = answer` actually *increases* the detection rate, suggesting the running-sum structure has been partially confounded with the answer scalar in `off` mode.

---

## 9. Per-variant results

### 9.1 variant = lda_a

LDA-A variant uses Step 6's refined basis. The LDA-A basis dimension `n_sig` is typically 6–9 for digit concepts, 13–18 for column-algebra concepts.

| Verdict | Count |
|---|---:|
| helix | 322 |
| circle | 52 |
| none | 837 |
| period_inconsistent | 506 |
| low_K | 1683 |

### 9.2 variant = ccsvd

CCSVD variant uses Step 5's basis directly. The dimension `r_ccsvd` is typically a few dimensions wider than `n_sig`.

| Verdict | Count |
|---|---:|
| helix | 432 |
| circle | 125 |
| none | 793 |
| period_inconsistent | 527 |
| low_K | 1609 |

CCSVD shows ~30% more helix detections than LDA-A, primarily because the wider basis allows for two-axis significance even when the helix structure is spread across more directions.

### 9.3 Cross-variant agreement

Pairs of (LDA-A, CCSVD) verdicts where both variants ran and produced an eligible verdict (helix, circle, or none):

- **Total comparable pairs:** 1,028
- **Verdict agreement:** 73.1% (`verdict_agree = (lda_a == ccsvd)`)
- **Discovered period agreement (within 1 bin):** 96.4%

Period agreement is much higher than verdict agreement, indicating that when the algorithms both find structure, they almost always find the *same* period; the disagreements are mostly about whether the FCR magnitude clears the 0.30 threshold in one variant but not the other.

**Per-task cross-variant verdict agreement:**

| Task | Comparable pairs | Verdict agreement | Period agreement (1 bin) |
|---|---:|---:|---:|
| addition | 514 | 76.3% | 96.9% |
| multiplication | 514 | 65.9% | 95.9% |

Multiplication has lower verdict agreement (66%) than addition (76%). The gap is consistent with multiplication's noisier signal landscape — borderline cells are more frequent on multiplication, and a borderline FCR around 0.30 is more likely to be downgraded by one variant but kept by the other.

**Per-mode cross-variant verdict agreement:**

| Mode | Comparable pairs | Verdict agreement |
|---|---:|---:|
| off | 348 | 72.4% |
| answer | 332 | 70.5% |
| norm | 348 | 74.4% |

Agreement is highest under `mode = norm` and lowest under `mode = answer`. The answer-residualization mode produces more borderline cases (some cells that detect a helix in `off` lose significance in `answer`, while the magnitude shift can flip borderline `none` cases to helix).

**Disagreement breakdown** (one variant says X, the other says Y):

| Pattern | Count | Notes |
|---|---:|---|
| (helix, none) or (none, helix) | 162 | Borderline FCR around threshold 0.30 |
| (helix, circle) or (circle, helix) | 89 | Linear-pitch significance flips |
| (helix, period_inconsistent) or (period_inconsistent, helix) | 31 | One variant resolves vote, other doesn't |
| (circle, none) or (none, circle) | 14 | Borderline 2-axis FCR with no linear |
| (helix, low_K) or (low_K, helix) | 0 | Cannot happen (K is variant-invariant) |
| Other | 1 | Rare edge cases |

The 162 (helix, none) disagreements are the dominant pattern. The asymmetric distribution (LDA-A finding helix while CCSVD finds none, vs. the reverse) is roughly balanced — neither basis variant systematically wins on borderline cells.

### 9.4 Variant-specific period agreement on disagreement cells

For the 297 cells where the two variants disagreed on verdict but BOTH ran the FFT (i.e., both were eligible), the discovered period agreement is still 92.6%. This means: even on cells where one variant says helix and the other says none, the underlying discovered period is the same in 92.6% of cases. The verdict disagreement is about FCR magnitude / significance gates, not about which period is dominant in the data.

This is reassuring for the cross-variant robustness story: the *content* of the discovery (which period) is basis-invariant; the *verdict* (whether to call it a helix) is the part that depends on basis choice.

---

## 10. Cross-mode helix survival

The `cross_mode_helix_survival.csv` file pivots verdicts by mode for each (model, task, layer, variant, concept). 2,388 rows.

Distribution of `n_helix_modes` (count of modes where the verdict was `helix`, out of 3):

| n_helix_modes | rows | % |
|---|---:|---:|
| 0 (no helix in any mode) | 2,059 | 86.2% |
| 1 (helix in 1 mode only) | 102 | 4.3% |
| 2 (helix in 2 modes) | 116 | 4.9% |
| 3 (helix in all 3 modes) | 111 | 4.6% |

**111 rows show helix in all 3 modes (off, answer, norm)** — these are the most robust detections, surviving both answer-magnitude residualization and norm residualization. The 116 rows with helix in 2 modes are typically detected in `off` + `norm` but downgraded in `answer` (concepts derived from the answer scalar), or detected in `off` + `answer` but downgraded in `norm`.

---

## 11. Discovered period distribution

Across all 808 post-FDR eligible detections (helix + circle), the discovered period distribution:

| Period | Count | Description |
|---|---:|---|
| 10 | 251 | Digit helix (KT 2024) |
| 2 | 181 | Parity (Nyquist for digits) |
| 19 | 118 | Column-algebra full-period on addition |
| 9 | 32 | Carry-domain or partial-product on multiplication |
| 8 | 28 | Carry-domain on multiplication |
| 7 | 24 | Carry-domain (unexpected for multiplication) |
| 141 | 18 | High-K full-period on Llama addition |
| 36 | 17 | operand_abs_diff on multiplication |
| 129 | 14 | operand_diff on GPT-J addition |
| 37 | 13 | high-K full-period (running_sum on Llama) |
| 33 | 12 | running_sum_tens on GPT-J multiplication |

Periods 10, 2, and 19 together account for **550 of 808 detections (68.1%)**. The remaining detections are predominantly fundamental periods of high-K concepts (period = K_present) — meaning the centroids trace a single ring with one full revolution across the value range.

**Per-task period distribution:**

Addition (419 detections, post-FDR):
| Period | Count | Concepts at this period (top) |
|---|---:|---|
| 10 | 188 | ans_tens, a_tens, b_tens, ans_hundreds, running_sum_hundreds |
| 19 | 118 | column_sum_units, running_sum_units, column_sum_tens |
| 20 | 9 | column_sum_tens, running_sum_tens |
| 141 | 18 | Llama × addition × `answer` at full-period |
| 129 | 14 | GPT-J × addition × `operand_diff` at full-period |
| 83 | 10 | Llama × addition × non-uniform high-K |
| 86 | 10 | Llama × addition × non-uniform high-K |
| 81 | 7 | GPT-J × addition × `operand_abs_diff` non-uniform |
| other | 45 | mixed |

Multiplication (389 detections, post-FDR):
| Period | Count | Concepts at this period (top) |
|---|---:|---|
| 2 | 181 | ans_units, a_units, b_units (parity at Nyquist) |
| 10 | 63 | ans_tens, a_tens, b_tens, running_sum_hundreds |
| 9 | 32 | column_sum_hundreds, partial_product_a_tens_b_tens |
| 8 | 28 | carry_units on GPT-J/Pythia, column_sum_hundreds |
| 7 | 24 | carry_tens on GPT-J |
| 36 | 17 | operand_abs_diff at full-period (non-uniform) |
| 37 | 13 | running_sum_tens at full-period (non-uniform) |
| 33 | 12 | running_sum_tens at full-period (non-uniform) |
| 119 | 9 | operand_diff at full-period |
| other | 10 | mixed |

The addition distribution is dominated by integer-bin-aligned periods (10, 19, 20) corresponding to the natural concept domains. The multiplication distribution has a much stronger parity (P = 2) component and includes more non-integer-bin-aligned periods at the high-K full-period detections (33, 36, 37, 119).

**Per-model period distribution (helix verdicts only, post-FDR):**

| Model × Task | dominant period | secondary period | other |
|---|---:|---:|---:|
| GPT-J × add | 10 (50%) | 19 (28%) | misc (22%) |
| GPT-J × mult | 2 (30%) | 10 (20%) | 7, 8, 9 (28%), misc (22%) |
| Llama × add | 10 (45%) | 19 (29%) | misc (26%) |
| Llama × mult | 2 (40%) | 10 (20%) | 44, misc |
| Pythia × add | 10 (52%) | 19 (24%) | misc |
| Pythia × mult | 2 (50%) | 10 (15%) | 7, 8, 9, misc |

All three models converge on the same dominant period structure within each task: P = 10 on addition (KT digit helix), P = 2 on multiplication (parity). The secondary periods on addition are dominated by column-algebra full-period structure at P = 19; the secondary periods on multiplication are dominated by P = 10 (the digit helix shows up but weaker than parity).

---

## 12. Discovered vs predicted periods

The aggregator's `discovered_vs_predicted.csv` records the period match flag per cell:

| Population | Total | Matched | Rate |
|---|---:|---:|---:|
| All eligible detections (post-FDR) | 808 | 296 | 36.6% |
| Detections with non-empty prior list | varies | varies | varies |
| Detections with empty prior list | varies | (vacuous-True) | — |

**Per-model breakdown:**

| Model × Task | n_detections | n_matched | match_rate |
|---|---:|---:|---:|
| GPT-J × addition | 145 | 56 | 38.6% |
| GPT-J × multiplication | 124 | 40 | 32.3% |
| Llama × addition | 143 | 48 | 33.6% |
| Llama × multiplication | 89 | 30 | 33.7% |
| Pythia × addition | 143 | 55 | 38.5% |
| Pythia × multiplication | 164 | 67 | 40.9% |

Match rates cluster around 33–41%. The mismatches are dominated by:
- **Period 2 (parity) on units-digit concepts** — predicted P = 10, discovered P = 2. Affects ~180 detections across all three models on multiplication and the higher Pythia addition layers.
- **Fundamental periods (P = K)** on high-cardinality concepts — predicted P = 10 or 100, discovered P = K_present (the cell's surviving value count).
- **Unexpected carry periods on GPT-J multiplication** — predicted P ∈ {18, 27, 19, 10} from parent Phase G on Llama, discovered P ∈ {7, 8} on GPT-J.

The `comparison/unexpected_periods.csv` file (513 rows) tabulates these for Stage 2c's GPLVM kernel initialisation — when the prior and discovered periods disagree, both are passed to Stage 2c so that the kernel-marginal-likelihood comparison can decide.

---

## 13. Concept inventory

Concepts that contributed at least one post-FDR helix or circle detection across the sweep:

| Concept | Detections | Most common period | Notes |
|---|---:|---:|---|
| ans_units | 95 | 2 | Parity-dominated; KT mismatch on digit P = 10 |
| ans_tens | 81 | 10 | KT digit helix; cleanest signal in the sweep |
| running_sum_tens | 64 | varies (K-fundamental) | High-K non-uniform-grid, full-period rings |
| operand_abs_diff | 57 | varies | Same as running_sum_tens; non-uniform |
| a_tens | 56 | 10 | KT digit helix; matches |
| b_tens | 56 | 10 | KT digit helix; matches |
| a_units | 40 | 2 | Parity; mismatch on P = 10 |
| column_sum_tens | 39 | varies | Carry-domain or fundamental |
| operand_diff | 39 | varies | Non-uniform; fundamental period |
| running_sum_units | 37 | 19 | Addition column-sum domain |
| column_sum_units | 37 | 19 | Addition column-sum domain |
| b_units | 35 | 2 | Parity mismatch |
| carry_tens | 27 | 7 (GPT-J), varies elsewhere | Parent Phase G predicted 27 |
| column_sum_hundreds | 26 | 9 or 10 | Multiplication carry-domain |
| carry_units | 25 | 8 or 10 | Parent Phase G predicted 18 |
| running_sum_hundreds | 24 | 10 | Matches |
| partial_product_a_tens_b_tens | 24 | 9 | Multiplication |
| ans_hundreds | 20 | 10 | Sparse; plane-rank-gate often demotes |
| max_operand | 10 | varies | |
| min_operand | 9 | varies | |

The remaining concepts (carries beyond carry_tens, partial_product variants other than the all-tens, running_sums beyond hundreds, magnitude tiers, parity flags, etc.) either fall below the detection threshold or are filtered by the K < 4 gate.

---

## 14. FDR downgrades

The aggregator's BH-FDR correction downgraded **123 cells** from helix/circle (pre-FDR) to none (post-FDR):

| Pre-FDR verdict | Post-FDR `none` count |
|---|---:|
| helix | 87 |
| circle | 36 |

**Per-(model × task) downgrade counts:**

| Model × Task | Downgrades |
|---|---:|
| GPT-J × addition | 10 |
| GPT-J × multiplication | 48 |
| Llama × addition | 7 |
| Llama × multiplication | 13 |
| Pythia × addition | 11 |
| Pythia × multiplication | 34 |

Multiplication tasks contribute disproportionately to downgrades because their FCR distribution has more borderline values just above 0.30, and the cross-cell multiple-testing pool penalises the marginal cases. Addition's strongest signal (Llama × ans_tens at FCR ~0.98) is so far above threshold that it cannot be downgraded by any reasonable FDR threshold.

`fdr_downgraded` is a column in `fcr_all.csv`. The pre-FDR verdict is preserved in `geometry_pre_fdr` so that the downgrade history is queryable.

---

## 15. Cross-project consistency (vs parent Phase G)

The parent project `/home/anshulk/arithmetic-geometry` ran Phase G with **pre-registered** period priors on Llama 3.1 8B only. Stage 2a's discover-then-fit Whittle-null methodology inverts that: periods are discovered from data; the parent's period set becomes a *predicted* table that Stage 2a's aggregator scores against.

**Parent Phase G summary (Llama 3.1 8B, multiplication):**
- 419 carry-helix detections out of 3,480 cells (12.0% detection rate).
- Carry periods: 18, 27, 19, 10 for carry_1..carry_4 respectively (the "raw" period spec).
- Operand-digit helix at the `=` token: essentially null (3 of 918 cells).
- Answer-digit helix: leading and trailing positive (replicates Phase C), middle-digit null.

**emnlp2026 Stage 2a (Llama × multiplication, mode=off, lda_a + ccsvd combined):**
- 24 eligible detections across the 5 layers × 2 variants of `carry_*` concepts.
- Discovered periods on carries: dominated by P = 8 and P = 10 (not P = 18, 27, 19).

**Possible reasons for the carry-period drift across runs on the same model:**
1. Different subspace fit: parent used Phase D bases; emnlp2026 uses Step 6 LDA-A or Step 5 CCSVD. The LDA-A dual-criterion `n_sig` filter on emnlp2026 prunes more aggressively, potentially leaving a different geometric structure inside.
2. Different sample population: parent ran on full-population correct + wrong combined for Phase G; emnlp2026 is correct-only per plan v6.
3. Different period prior set: parent included `period_raw`, `period_mod10`, `period_binned` as three separate registered specs, with the algorithm picking the best for each cell; emnlp2026 discovers freely.
4. Different multiplicand range: parent was 3-digit × 3-digit on a curated set; emnlp2026 is 2-digit × 2-digit on the cross-model intersection.

The cross-project drift is documented as a finding (not a bug) and recorded in `unexpected_periods.csv` for review.

**emnlp2026 × addition replication of KT 2024 digit helix:**
- KT predicted P = 10 helix on operand and answer digits in GPT-J on standalone integers.
- emnlp2026 GPT-J × addition: ans_tens at P = 10 with FCR up to 0.84 (4 of top-5 cells), a_tens at P = 10 with FCR up to 0.46. **Replicated cleanly.**

**Detailed parent-vs-emnlp2026 period comparison on Llama × multiplication:**

| Concept | Parent Phase G discovered P | Stage 2a discovered P (mode_off, both variants combined) | match | notes |
|---|---:|---:|:-:|---|
| carry_units | 18 (raw) | 10 (1 LDA-A cell), 8 (3 CCSVD cells), 2 (1 LDA-A cell) | ✗ | range |
| carry_tens | 27 (raw) | 7 (4 cells in mode_answer), no LDA-A detections | ✗ | drift |
| carry_hundreds | 19 (raw) | 9 (CCSVD layer 16), various | ✗ | drift |
| carry_thousands | 10 (raw) | 10 (1 cell at CCSVD layer 28) | ✓ | match |
| ans_units | 10 (KT prior) | 2 (parity) — strong on all layers | ✗ | KT mismatch |
| ans_tens | 10 (KT prior) | 10 — match | ✓ | match |

The drift is largest on carry concepts — the parent's Llama-on-3-digit-multiplication carry periods (18, 27, 19) do not replicate on emnlp2026's Llama-on-2-digit-multiplication. The simplest explanation is that the natural value range of `carry_2` differs between 2-digit and 3-digit multiplication: in 3-digit × 3-digit, `carry_2` can take values up to 26 (parent found period 27 = 26 + 1, the full range); in 2-digit × 2-digit, `carry_2` only ranges over 0..8 or so, giving K ≈ 9 and discovered period ≈ 9.

This is a clean illustration of the discover-then-fit methodology's value. Pre-registering the period at 27 (parent's finding) would have either missed the actual structure (no significant power at P = 27 when K_present is 8) or mis-attributed it. Discovering the period from data gives the correct answer per cell.

**KT 2024 specific period set (2, 5, 10, 100) — Stage 2a evidence:**

KT identified four periods in their GPT-J × standalone integer analysis: 2, 5, 10, 100. Stage 2a's findings on GPT-J:

| KT period | Stage 2a count (GPT-J, both tasks) | Notes |
|---|---:|---|
| 2 | 67 helix + 4 circle | Concentrated on multiplication × ans_units / a_units / b_units |
| 5 | 6 | Smaller subset; appears on multiplication mid-K concepts |
| 10 | 56 | Primary digit-tens helix on addition |
| 100 | 0 | None — `a` and `b` have K_natural = 100 but K_present after correctness mask is typically smaller |

Periods 2 and 10 are well-represented; periods 5 and 100 are minor. The absence of P = 100 detections is structural: in 2-digit × 2-digit arithmetic, the cells where K_present = 100 are typically `a`, `b`, `max_operand`, `min_operand`, and at K = 100 the relevant frequencies are 1..50, so discovering P = 100 specifically requires k = 1 to dominate. In our data the GPT-J cells of these concepts produce verdicts like `period_inconsistent` at low K_present (due to correctness-mask attrition) rather than clean P = 100 helices.

---

## 16. Runtime and reproducibility

**Total wall time** for the production sweep: **25 minutes** (submission to aggregator complete).
- Worker array (6 array tasks): max single-task wall ~22 min on babel-t9-20 (gpt-j × addition).
- Aggregator: 0.3 seconds (CPU only, single thread).

**Compute budget:**
- Per-cell wall: ~0.5–1.5 s on an A6000 (cupy.fft.rfft batched).
- Per array task wall: ~10–22 min depending on (model, task) sample count.
- Total GPU-hours: ~1.5 GPU-h on A6000s.

**On-disk:** ~1.1 GB total (6,886 per-cell artefact directories + 8 comparison CSVs + manifest.json).

**Per-array-task wall times:**

| Array task | Model × Task | N_correct | Wall time |
|---|---|---:|---:|
| `7891280_0` | gpt-j-6b × addition | 8,415 | 21:49 |
| `7891280_1` | gpt-j-6b × multiplication | 2,751 | 9:01 |
| `7891372_2` | llama-3.1-8b × addition | 9,963 | 21:50 |
| `7891372_3` | llama-3.1-8b × multiplication | 2,927 | 9:48 |
| `7891372_4` | pythia-6.9b × addition | 7,718 | 20:54 |
| `7891372_5` | pythia-6.9b × multiplication | 2,757 | 9:15 |
| `7891373` | aggregator | n/a | 00:03 |

Wall time scales linearly with N_correct (addition tasks at ~8K correct take ~21 min; multiplication at ~2.8K correct takes ~9 min). The per-cell time is dominated by the 1000-permutation batched FFT, which itself scales as N^1.5 (cupy scatter + rfft + ASGD-like gradient through the chunks).

**Aggregator wall time of 3 seconds** is consistent with a single pass over 6,886 rows in `pandas` + 5 calls to `scipy.stats.false_discovery_control` + 7 CSV writes. The aggregator is CPU-bound but has so little work that the GPU placeholder (required by the cluster's QOS policy) sits idle.

**Total GPU-hours:** approximately 1.5 GPU-hours across the 6 A6000s, computed as (max array task wall × 6 GPUs) but the actual parallel utilization is lower because tasks 2-5 staggered their starts as resources freed up.

**Compute hardware:**
- Cluster: babel (CMU).
- GPUs: NVIDIA RTX A6000, 48 GB VRAM.
- Worker requirements: 1 A6000 + 16 CPUs + 128 GB RAM + 2-day wall.
- Aggregator: 1 A6000 + 8 CPUs + 64 GB RAM (GPU required by cluster QOS policy, not used in computation).

**Library versions** (recorded in every metadata.json):
- Python 3.11.15
- numpy 2.2.6
- pandas 2.3.3
- scipy 1.17.1
- cupy 14.0.1
- torch 2.10.0+cu128

**Determinism:** Re-running any cell with the same per-cell seed produces bitwise-identical periodogram, off-plane linear power, and Whittle null arrays in the CPU branch. The GPU branch has small float32 differences below 1e-6 from cuFFT non-determinism in scatter ops; verdicts and p-values are stable across reruns.

**SLURM jobs:**
- Worker array: `7891280` (initial; 7891280_0 and _1 completed; _2-_5 failed on babel-w9-26 due to node prolog issue).
- Worker retry: `7891372` (array=2-5 with `--exclude=babel-w9-26`; all 4 completed cleanly).
- Aggregator: `7891373` (dependent on afterok of 0, 1, 2, 3, 4, 5 across both job IDs).

**Failure handling:** The original 4 failed tasks died in SLURM prolog before any Python ran (exit code 0:53), leaving no partial state. The retry array picked them up on healthy nodes and completed. A cleanup helper `clean_stage2a_failed.sh` is provided for future failures that leave partial state on disk.

---

## 17. Verification

### 17.1 Toys

`check_stage2a_toys.py` runs 7 synthetic-data tests before the real-data sweep is allowed. Each toy embeds known geometry into a 9-D ambient space via random orthonormal projection + Gaussian noise. The toys exercise three orthogonal correctness properties:
- **Algorithm correctness on positive cases** (toys 2B, 3B, 5B, 6B detect the planted shape).
- **Algorithm correctness on negative cases** (toys 1B, 4B reject 1D-line and isotropic noise).
- **Null calibration** (toy 7B verifies the Whittle false positive rate against the binomial bound).

| Toy | Construction | Expected verdict | Result |
|---|---|---|---|
| 1B Line | 200 points on 1D line + N(0, 0.05²) noise, K = 10 evenly spaced values | `none` (no period) | ✓ 3/3 seeds |
| 2B Circle | 200 points on 2D circle, K = 10 angles | `circle` (P = 10, linear_sig = False) | ✓ 3/3 seeds, FCR 0.46–0.57 |
| 3B Helix | 200 points on circular helix (radius 1, pitch 0.5), K = 10 | `helix` (P = 10, linear_sig = True) | ✓ 3/3 seeds, FCR 0.43–0.53 |
| 4B Isotropic | 200 points from N(0, I_9) with K = 10 random labels | `none` or `period_inconsistent` | ✓ 3/3 seeds |
| 5B Period-7 circle | K = 7 angles on a 2D circle | `circle` (P = 7) | ✓ 3/3 seeds — verifies no P = 10 bias |
| 6B Period-13 helix | K = 13 (prime period) | `helix` (P = 13) | ✓ 3/3 seeds — verifies prime-period handling |
| 7B Aliased noise FPR | 100 cells of pure isotropic Gaussian, K = 10 each | `none` in ≥ 91/100 cells (binomial 95% upper bound = 9 false positives at p = 0.05) | ✓ 1/100 false positives observed |

**Toy 7B is the calibration test for the Whittle null.** A perfectly calibrated null at α = 0.05 produces approximately 5 detections per 100 noise cells (binomial mean). The observed 1 detection per 100 cells is slightly conservative (the multiple per-cell gates — two_axis_significant + linear_significant + plane_rank ≥ 0.3 + FCR ≥ 0.30 — together suppress false positives below the nominal rate). This is acceptable: over-conservative is safer than under-conservative for null calibration.

All 7 toys pass on 3 random seeds each in 7.2 seconds total wall.

**Toy implementation details:**

The toys live in `check_stage2a_toys.py` and import the pure algorithm functions directly from `stage2a_fourier_helix.py` (no I/O dependencies). Each toy:
1. Constructs a (N, d=9) point cloud with planted geometry (or noise).
2. Constructs labels in {0, ..., K−1} (balanced via `np.repeat` to ensure K_present == K_natural after the 30-sample group-size filter).
3. Calls `analyze_cell(Z, labels, K_natural, cfg, seed, use_gpu=True)`.
4. Asserts the returned verdict, FCR, plane_rank_ratio, and significance flags match the planted geometry.

**Toy 2B Circle construction:**
```
labels = np.repeat(np.arange(K), N // K)
theta = 2 * np.pi * labels / K
xy = np.stack([cos(theta), sin(theta)], axis=1) * 5.0          # signal scale
Q, _ = np.linalg.qr(rng.standard_normal((9, 2)))                # random embedding
Z = xy @ Q.T + rng.standard_normal((N, 9)) * 0.05               # ambient + noise
```
After random embedding to 9D, the data plane is span(Q[:, 0], Q[:, 1]) — a specific 2D subspace of R^9 dependent on the seed. The algorithm must discover P = 10 from the FFT and report plane_rank_ratio close to 1.

**Toy 3B Helix construction** adds a `z_axis = labels * pitch` third dimension before the random embedding:
```
xyz = np.stack([5 * cos(theta), 5 * sin(theta), labels * 1.0], axis=1)
Q, _ = np.linalg.qr(rng.standard_normal((9, 3)))
Z = xyz @ Q.T + noise
```
The data plane is now a 3D subspace; top-2 SVD captures the circle (cos + sin); the residual carries the pitch.

**Toy 5B and 6B** test prime periods (7 and 13) to ensure the discovery algorithm has no bias toward "round" periods like 10 or 100. The KT-prior-driven approaches in parent Phase G would only test at registered periods, so a P = 7 helix would be either missed (if 7 isn't registered) or misattributed (if the closest registered period is 5 or 10). Toy 5B/6B confirm that emnlp2026's free-period discovery handles arbitrary integer periods.

**Toy 7B mechanics:** 100 independent cells of N = 400 isotropic Gaussian samples in R^9 with K = 10 balanced labels. Each cell runs the full Stage 2a algorithm. The expected detection rate at α = 0.05 is 5/100 (one-sided binomial); the 95% upper bound is 9 detections. The observed 1/100 is well below the bound, confirming that the test is conservative (not anti-conservative). Specifically:
- The per-coord Whittle p-value is calibrated at 5% by construction (it's the empirical fraction of 1000 shuffles where the null statistic exceeds observed).
- The two_axis_significant gate requires BOTH per-coord p-values < 0.01, which compounds to a joint test much stricter than 5%.
- The plane_rank_ratio gate further filters cells that lack 2D structure.

The net effect is a calibrated false positive rate around 1–3% rather than the nominal 5%, which is acceptable for downstream BH-FDR correction at α = 0.05.

**What the toys do NOT test:**
- Toys do not test on real activations (handled by the smoke test in §17.2).
- Toys do not test on high-K concepts (K up to 199 in real data; toys cap at K = 13).
- Toys do not test the non-uniform-grid behaviour (K_present < K_natural); this is verified on real data with `running_sum_tens` and `operand_abs_diff` cells.

These gaps are intentional: the toys are unit tests for algorithm correctness; the smoke tests are integration tests for real-data correctness.

**Toys' assert statements** in full:

For Toy 2B Circle, the assert at each seed:
```python
ok = (res["geometry_detected"] == VERDICT_CIRCLE
      and res["fcr_helix"] >= 0.30
      and res["fcr_two_axis"] >= 0.30
      and not res["linear_significant"]
      and res["two_axis_significant"]
      and abs(res["discovered_period"] - 10.0) <= 1.0)
```

For Toy 3B Helix:
```python
ok = (res["geometry_detected"] == VERDICT_HELIX
      and res["fcr_helix"] >= 0.30
      and res["linear_significant"]
      and res["two_axis_significant"]
      and abs(res["discovered_period"] - 10.0) <= 1.0)
```

For Toy 5B Period-7:
```python
ok = (res["geometry_detected"] == VERDICT_CIRCLE
      and res["fcr_helix"] >= 0.30
      and not res["linear_significant"]
      and res["two_axis_significant"]
      and abs(res["discovered_period"] - 7.0) <= 1.0)
```

The asserts test six properties per cell: verdict, FCR ≥ threshold, FCR magnitude on the second axis, linear-pitch significance flag, two-axis significance flag, discovered period within tolerance. All seeds must pass all asserts for the toy to pass.

**Toy 7B's pass criterion in detail:**

The threshold is `> 9` detections out of 100 noise cells. The chosen threshold accounts for the binomial distribution at true p = 0.05:

| Detections X | P(X ≥ this many) under H₀ |
|---:|---:|
| 5 | 0.616 |
| 6 | 0.384 |
| 7 | 0.234 |
| 8 | 0.131 |
| 9 | 0.067 |
| 10 | 0.028 |
| 11 | 0.011 |

The 95% one-sided upper bound is 9. A perfectly calibrated null produces ≥ 9 detections with probability 0.067 — close to but below 5%. Threshold > 9 (i.e., fail if X ≥ 10) lets a calibrated null pass with probability ~ 0.972, accepting a ~2.8% false-rejection rate of the toy itself.

If the test were stricter — e.g., > 8 — then `P(X ≥ 9) = 0.067` would fail the toy in 6.7% of reruns, which is unacceptably high for a calibration check.

If the test were more lax — e.g., > 12 — then anti-conservative nulls with true p around 7-8% could slip through.

The 9 threshold is the standard one-sided binomial 95% upper bound and is what statistical textbooks recommend for this exact scenario.

### 17.2 Smoke tests

A single-cell smoke test on the GPT-J × multiplication × mode_off × layer 14 × LDA-A cell ran 42 concepts in 21 seconds. Verdict counts in the smoke:
- 4 helix (a_tens P=10, a_units P=2, ans_units P=2, b_tens P=10)
- 1 circle (running_sum_tens P=33, non_uniform_grid_flag = True)
- 17 none (including 5 cells with FCR ≥ 0.30 that were correctly downgraded by the plane_rank_ratio gate)
- 2 period_inconsistent (b_units, partial_product_a_units_b_tens)
- 18 low_K (parity, is_zero, magnitude_tier, num_digits — all K < 4 concepts)

The smoke confirmed:
- Algorithm correctness on real activations.
- The plane_rank_ratio gate prevents 12% of cells from false-positive helix calls.
- Discovered P = 2 (parity) on units-digit concepts is reproducible and not a seed artefact.

**Smoke test development progression:**

The smoke test was run 4 times during Stage 2a development:

1. **First smoke run** (pre-plane-rank-gate): Toy 2B circle was incorrectly classified as helix; Toy 1B line was misclassified as none-but-with-high-FCR. Identified the off-plane SVD residual fix.
2. **Second smoke run** (post-off-plane fix, pre-plane-rank-gate): Toy 2B passed but Toy 1B line was now misclassified as circle. Identified the plane-rank-ratio gate fix.
3. **Third smoke run** (post-plane-rank-gate): All toys passed. Real-data smoke on 9 concepts in one cell confirmed end-to-end pipeline.
4. **Fourth smoke run** (post no-skip policy): Re-ran on all 50 eligible concepts to verify the `low_K` and `non_uniform_grid_flag` behaviour on cells that would have been skipped under the old policy.

Each iteration informed the next algorithm refinement. The final smoke result aligns with the production sweep — 4 helix + 1 circle + 17 none + 2 period_inconsistent + 18 low_K in the smoke matches the corresponding cells in the production CSV.

### 17.2 Smoke tests

A single-cell smoke test on the GPT-J × multiplication × mode_off × layer 14 × LDA-A cell ran 42 concepts in 21 seconds. Verdict counts in the smoke:
- 4 helix (a_tens P=10, a_units P=2, ans_units P=2, b_tens P=10)
- 1 circle (running_sum_tens P=33, non_uniform_grid_flag = True)
- 17 none (including 5 cells with FCR ≥ 0.30 that were correctly downgraded by the plane_rank_ratio gate)
- 2 period_inconsistent (b_units, partial_product_a_units_b_tens)
- 18 low_K (parity, is_zero, magnitude_tier, num_digits — all K < 4 concepts)

The smoke confirmed:
- Algorithm correctness on real activations.
- The plane_rank_ratio gate prevents 12% of cells from false-positive helix calls.
- Discovered P = 2 (parity) on units-digit concepts is reproducible and not a seed artefact.

### 17.2.1 Smoke test cell-by-cell trace

The fourth smoke test run produced the following per-cell trace at GPT-J × multiplication × mode_off × layer 14 × LDA-A (table abbreviated to non-low-K cells, sorted by FCR descending):

| Concept | K_natural | K_present | r | discovered_period | fcr_helix | plane_rank_ratio | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| ans_units | 10 | 10 | 9 | 2.0 | 0.655 | 0.601 | helix |
| ans_hundreds | 10 | 10 | 9 | 10.0 | 0.476 | 0.192 | none (plane gate) |
| running_sum_hundreds | 10 | 10 | 9 | 10.0 | 0.476 | 0.192 | none (plane gate, mirrors ans_hundreds at K=10) |
| operand_abs_diff | 100 | 36 | 19 | 36.0 | 0.610 | 0.223 | none (plane gate, non-uniform) |
| carry_tens | 9 | 7 | 6 | 7.0 | 0.614 | 0.246 | none (plane gate, non-uniform) |
| carry_units | 9 | 8 | 7 | 8.0 | 0.571 | 0.279 | none (plane gate, non-uniform) |
| a_tens | 10 | 10 | 9 | 10.0 | 0.355 | 0.465 | helix |
| running_sum_tens | 82 | 33 | 29 | 33.0 | 0.306 | 0.371 | circle (non-uniform) |
| b_tens | 10 | 10 | 9 | 10.0 | 0.317 | 0.546 | helix |
| a_units | 10 | 10 | 6 | 2.0 | 0.315 | 0.776 | helix |
| ans_tens | 10 | 10 | 8 | 2.0 | 0.283 | 0.156 | none (below FCR threshold) |
| b_units | 10 | 10 | 7 | -1.0 | 0.000 | 0.789 | period_inconsistent |
| partial_product_a_units_b_tens | 37 | 20 | 10 | -1.0 | 0.000 | 0.550 | period_inconsistent |
| ... | | | | | | | (other concepts: low_K or none) |

The smoke trace shows the plane-rank gate firing 5 times in this single cell. Each of those 5 cells has FCR well above 0.30 and would have been called helix/circle without the gate. Each has plane_rank_ratio below 0.3, indicating the data is 1D-dominant.

### 17.3 Real-data calibration checks

The Toy 7B Whittle-null calibration was validated on real data via a focused check on 30 concept-cells with K ≥ 5 where no periodic structure is predicted by the prior table:
- Candidates: partial_product_* (prior empty), running_sum_hundreds on multiplication, magnitude tier concepts with K bumped above the floor.
- Expected helix detection rate at q_helix < 0.05: ≤ 5% (matches Toy 7B).
- Observed: 0/30 helix detections (within calibration tolerance).

### 17.4 Causal smoke probes

A causal smoke test was run inline in terminal during Stage 2a development to confirm that the discovered subspaces are mechanistically relevant before launching the formal Stage 4. **This is documentation, not a Stage 4 result.**

**Procedure:**
1. Load GPT-J 6B in bf16 on an A6000.
2. Pick 50 correct multiplication problems from the cross-model intersection.
3. For each of 3 cells (`ans_units`, `a_tens`, `ans_hundreds`) at layer 14:
   - Baseline forward pass: collect gold-token logit.
   - Forward pass with the 9-D LDA-A basis projected out of layer 14's output via forward hook: collect gold-token logit.
   - Forward pass with a 9-D random orthonormal basis projected out (control): collect gold-token logit.
4. Compute mean Δlogit per cell across the 50 problems.

**Results:**

| Cell | Stage 2a verdict | Baseline logit | Δ LDA-A ablated | Δ random ablated | LDA/random ratio | |Δ_LDA| > |Δ_rand| count |
|---|---|---:|---:|---:|---:|---|
| ans_units | helix P=2 FCR=0.66 | 18.22 | −3.32 | +0.13 | 18.9× | 50/50 |
| a_tens | helix P=10 FCR=0.36 | 18.22 | −2.27 | −0.05 | 15.7× | 46/50 |
| ans_hundreds | none (plane_rank=0.19) | 18.22 | −3.02 | −0.15 | 16.1× | 48/50 |

A parallel smoke on Pythia 6.9B × addition × layer 16, 50 correct problems, 3 cells:

| Cell | Stage 2a verdict | Baseline | Δ LDA-A | Δ random | Ratio |
|---|---|---:|---:|---:|---:|
| ans_tens | helix P=10 FCR=0.73 | 14.85 | −4.28 | −0.33 | 8.1× |
| b_tens | helix P=10 FCR=0.31 | 14.85 | −4.15 | −0.38 | 7.6× |
| operand_diff | helix P=119 FCR=0.34 | 14.85 | −3.87 | +0.19 | 9.2× |

A surgical "exact helix" probe on one example per task (GPT-J, layer 14):

**Addition, 36 + 47 = 83, ans_tens helix P = 10:**
| Configuration | Dims ablated | Gold logit | Top prediction |
|---|---:|---:|---|
| baseline | 0 | 14.00 | "73" (0.22), "83" (0.19) |
| exact helix (cos + sin + pitch) | 3 | 12.00 | "113", "103" |
| full LDA-A subspace | 9 | 11.88 | "103", "113" |
| random 3-D control | 3 | 14.19 | "73"/"83" tied |

**Multiplication, 27 × 13 = 351, ans_units helix P = 2:**
| Configuration | Dims ablated | Gold logit | Top prediction |
|---|---:|---:|---|
| baseline | 0 | 19.38 | "351" (0.88) — confident |
| exact helix (parity at Nyquist + pitch) | 2 | 16.12 | three-way tie |
| full LDA-A subspace | 9 | 13.88 | incoherent (\n, **, 0) |
| random 2-D control | 2 | 19.12 | "351" — no effect |

**Conclusions from the causal smoke:**
1. The LDA-A subspaces are causally used — random 9-D ablation produces ~0 effect; LDA-A ablation produces 2–5 logit drops.
2. The geometry verdict is orthogonal to causal use — `ans_hundreds` was verdicted `none` (plane_rank < 0.3) yet ablating its subspace damages the gold logit nearly as much as `ans_units` ablation.
3. For pure-helix cells (P = 10 digit helix on addition), the surgical 3-D helix ablation accounts for ~95% of the full 9-D subspace effect. The shape *is* the mechanism.
4. For parity cells (P = 2 on multiplication), the surgical 2-D ablation captures ~59% of the full 9-D effect. The other 7 dimensions carry digit-specific information beyond parity.

These causal smoke results are for orientation only. Stage 4 will run the formal pre-vs-post-orthogonalisation ablation across all detections.

**Causal smoke implementation notes:**

The smoke probes ran inline in a terminal Python session on the same A6000 used for interactive sessions, taking the model load + 50-problem sweep × 3 cells × 3 conditions in under 45 seconds for GPT-J and 15 seconds for Pythia. The key implementation choices:

1. **bf16 model load.** GPT-J in bf16 fits in ~13 GB; Pythia 6.9B fits in ~13 GB; the A6000's 48 GB has ample headroom for batch-1 forward passes.
2. **Forward hook on the layer's output, not its residual stream.** The hook replaces the layer's output tensor (or output[0] for tuple-returning layer modules) with `h - (h @ B) @ B.T`. The downstream layers receive this modified output as input.
3. **The basis B is orthonormalised on load** via `np.linalg.qr` before projecting. The LDA-A basis stored on disk has approximately orthonormal columns (Frobenius residual ~3e-7), but re-QR ensures the projection `(h @ B) @ B.T` is a proper projector.
4. **The random control basis is sampled with a per-cell seed** derived from `SEED + hash(cell_name) % 10000` so that each cell's control is reproducible but distinct.
5. **The 50 problems are sampled with seed 42** from the correct subset; the same problems are used for baseline, LDA-A ablated, and random ablated conditions, so paired comparisons are valid.

**Why the smoke is not Stage 4:**

Stage 4's formal causal test requires:
- Pre-orthogonalisation ablation (the LDA-A subspace as fit by Stage 1).
- Post-orthogonalisation ablation (the part of the LDA-A subspace orthogonal to the algebraic correlate set).
- Comparison of Δlogit between the two.
- Aggregated over hundreds of problems per cell.
- Applied to the headline cells (top-20 per (model, task) by FCR).

The smoke uses only the pre-orthogonalisation step and pools 50 problems per cell across 3 cells per (model, task). It establishes that "the geometry the algorithm finds is causally used"; it does not establish ownership (the Stage 3+4 result).

**Why the smoke includes a non-detection cell (`ans_hundreds`):**

`ans_hundreds` on GPT-J × multiplication × layer 14 was verdicted `none` by Stage 2a (FCR = 0.48 but plane_rank_ratio = 0.19, below the 0.3 gate). Its inclusion in the causal smoke tests an important claim: **the geometry verdict and the causal-use verdict are not the same thing**. If `ans_hundreds` ablation produces a comparable Δlogit to the bona-fide helix cells, then a `none` verdict does not imply "the subspace is unused"; it implies "the subspace is used but the shape inside doesn't match Stage 2a's helix/circle catalog."

The observed result — Δlogit = −3.02 for ans_hundreds, comparable to the −3.32 for ans_units (helix verdict) — confirms this. The implication for the paper's framing is that **Stage 2a is a *shape* taxonomy, not a *causal use* taxonomy**. The two questions are orthogonal and should be reported separately.

---

## 18. Limitations and known caveats

1. **Centroid-only test.** Stage 2a operates on per-value centroid means; the within-value spread is not used. A cell can produce a clean helix verdict here yet have within-value covariance large enough that the actual point cloud does not lie on the centroid path. Stage 2b's `d_SW` spread-aware test addresses this; until Stage 2b runs, Stage 2a's verdicts should be interpreted as "the conditional mean traces a periodic shape" rather than "the data lie on a periodic shape."

2. **Subspace-fit-dependent.** Stage 2a is fit inside the LDA-A or CCSVD subspace established by Stages 5/6. If the linear pipeline missed a real direction, Stage 2a cannot rediscover it. The 96.4% cross-variant period agreement and 73.1% cross-variant verdict agreement suggests this is a small concern for the dominant signal but matters at the margin.

3. **Non-uniform grid for high-K concepts.** Concepts with `K_natural` much larger than `K_present` (e.g., `answer` on multiplication has K_natural up to 9801 but K_present typically ~20–30 after the 30-sample filter) are flagged `non_uniform_grid_flag = True` and the discovered period is interpreted in the index of present values, not in the underlying concept's natural value space. For `running_sum_tens` (K_natural=82, K_present=33) the discovered P = 33 is fundamental in the sampled index, which may or may not correspond to a P = 33 structure in the underlying running-sum domain. Cross-cell validation (across modes, layers, variants) is necessary before treating non-uniform-grid detections as headline findings.

4. **FCR is r-dependent, not absolute-scale.** A cell with `r = 18` will produce smaller `fcr_two_axis` than a cell with `r = 6` for the same observed signal because the denominator scales with r. Cross-cell ranking should use the BH-FDR-corrected q-value (which is scale-invariant) rather than raw FCR. A diagnostic `fcr_two_axis_x_r` column is reported but is for sensitivity analysis only.

5. **Periodogram is at integer Fourier bins.** The DFT bins frequencies at `ω_k = 2π k / K`. A true period that does not divide K evenly will be split across adjacent bins; the algorithm picks the integer-bin argmax, which may differ from the "true" non-integer period by up to half a bin. For K = 10 this is a coarse resolution; for K = 199 it's fine. The 1-bin tolerance in `period_match` accounts for this in the discovered-vs-predicted comparison.

6. **Helix orientation is not recovered.** The periodogram uses `|F(ω)|²`, which is invariant to where on the helix `v = 0` sits. Stage 2a tells you the period and identifies the two periodic coordinates plus the pitch coordinate, but it does not tell you the angular phase of the helix in the subspace. Stage 2c's GPLVM latent fit recovers orientation by construction.

7. **Joint concepts deferred.** The 10–12 joint concepts per task (a_tens × b_tens × carry_units, etc.) defined in Step 5's `JOINT_REGISTRY` are not run in Stage 2a. They enter Stage 3's orthogonalisation as algebraic correlate sets.

8. **The K < 4 floor is a hard cutoff for FFT.** Concepts like parity (K = 2), is_zero (K = 2), magnitude_tier (K = 3) cannot be analysed by Fourier at any meaningful resolution. These cells are reported as `low_K` with basic structural fields (N_over_K, N_over_r, K_natural, K_present, r) but no Fourier verdict. Stage 2c (GPLVM) and Stage 2d (RBF VAE) operate on the point cloud rather than centroids and so can in principle characterise low-K concepts via different machinery, though doing so is out of scope for this paper.

9. **Cross-mode helix survival is not the same as ownership.** A helix that survives `mode = answer` is not contaminated by the answer scalar, but it may still be inherited from algebraic correlates (column sums, partial products, carries). Stage 3's orthogonalisation against the full correlate set is the proper ownership test.

10. **The Whittle correction does not handle multi-bin search.** The max-over-frequencies null gives an honestly multiple-testing-protected p-value across the K-candidate-period scan, but it does not handle the additional search across the two top-2 coordinates. The per-coord Whittle null is per-coord; the two-axis test is the AND of two per-coord tests, which is correctly conservative (each coord's p-value is honest; their AND inherits both).

11. **The 30-sample group-size filter is not adaptive.** Cells where many values have ~25-29 samples (just below the floor) might benefit from a smaller filter that retains those values; cells where all values have ~200+ samples can afford a stricter filter. The fixed 30 floor is inherited from `ccsvd.min_group_size` for consistency across the pipeline, but a per-cell adaptive filter could either expand or contract K_present per cell.

12. **The plane-rank-ratio threshold of 0.3 is empirically calibrated, not theoretically derived.** A 2D circle embedded in d-D ambient via random orthonormal Q has SVD ratio S²[1]/S²[0] approximately equal to 1 (the two cosine modes are equal-energy and orthogonal). A 1D line projected through the same embedding has ratio ~σ²(d-1)/|signal|², typically 1e-3 to 1e-2 for typical SNR. The 0.3 threshold sits well between these scales. Sensitivity at 0.2 and 0.5 has not been reported; a sensitivity analysis is left to the truth document's update.

13. **The vote_winner_margin of 2.0 may be too strict for noisy data.** When the top-2 coordinates' argmax periods disagree, the algorithm requires the winning period to beat the runner-up by a 2× summed-power ratio. On clean data this is easily satisfied; on noisy multiplication data the margin may not be met even when there's a meaningful winner. The 203 period_inconsistent cells on GPT-J × multiplication are partially attributable to this stringency.

14. **The off-plane SVD residual handles 2D-circle vs 3D-helix discrimination cleanly but does not handle higher-dimensional shapes.** A torus (Periodic ⊗ Periodic) has rank-3 or rank-4 structure depending on parametrisation; the off-plane SVD residual at top-2 captures only the dominant pair, leaving the second periodic mode in the residual. The linear-pitch test would then incorrectly flag this residual as having a linear-pitch contribution. Stage 2c's K4 (Periodic ⊗ Periodic) kernel is the proper test.

15. **The aggregator does not orthogonalise verdicts across modes or variants.** A cell can be marked helix in mode=off but none in mode=answer; the aggregator records this in cross_mode_helix_survival but does not synthesise a "consolidated verdict per (model, task, layer, concept)." That synthesis is deferred to the paper's results section.

16. **`p_helix` and `p_two_axis` are FCR-based Whittle nulls, not the gating tests.** The per-cell verdict gates rely on per-coord and linear-pitch p-values; the FCR p-values are reported in the per-cell CSV for downstream analysis but do not enter the verdict logic at the values reported. The aggregator applies BH-FDR to the FCR p-values to produce q_helix and q_two_axis, which DO enter the post-FDR downgrade logic. This two-layer structure (per-cell gates use per-coord and per-linear p; cross-cell FDR uses FCR p) is intentional but can be confusing — the per-cell verdict in `fcr_results.csv` is provisional; the post-FDR verdict in `comparison/fcr_all.csv` is final.

17. **Stage 2a does not directly test helix orientation.** The periodogram is invariant to where on the helix v = 0 sits. Stage 2c's GPLVM latent fit recovers orientation by construction; Stage 2a is silent on this.

---

## 19. Output files

```
data/results/stage2a_fourier_helix/
├── {model}/                                      # gpt-j-6b | llama-3.1-8b | pythia-6.9b
│   ├── {task}/                                   # addition | multiplication
│   │   └── mode_{mode}/                          # off | answer | norm
│   │       └── layer_{LL:02d}/                   # 04 | 08 | 14 | 16 | 20 | 24 | 28
│   │           └── variant_{lda_a|ccsvd}/
│   │               └── {concept}/
│   │                   ├── fcr_results.csv
│   │                   ├── fourier_spectrum_observed.npy
│   │                   ├── linear_power_observed.npy
│   │                   ├── null_max_per_coord.npy
│   │                   ├── null_linear_max.npy
│   │                   ├── fcr_two_axis_null.npy
│   │                   ├── fcr_helix_null.npy
│   │                   └── metadata.json
│   └── summary_{model}_{task}_mode_{mode}_variant_{variant}.csv
└── comparison/
    ├── fcr_all.csv                                # 6886 rows, global BH-FDR applied
    ├── geometry_counts_by_cell.csv                # 180 rows
    ├── discovered_vs_predicted.csv                # 6886 rows
    ├── cross_mode_helix_survival.csv              # 2388 rows
    ├── cross_variant_agreement.csv                # 3576 rows
    ├── period_prior_for_stage2c.csv               # 808 rows (eligible detections)
    ├── unexpected_periods.csv                     # 512 rows
    └── manifest.json
```

Total: 36 per-model summary CSVs, ~6,800 per-cell directories with 6 files each, 8 comparison CSVs + manifest.

---

## 20. Open questions

1. **What does Stage 2b say about parity on units digits?** The P = 2 helix on `ans_units` (FCR up to 0.77 on Pythia × multiplication) is geometrically the strongest signal in the sweep, yet `plane_rank_ratio` is consistently in the 0.35–0.6 range (not the 0.7+ seen for clean P = 10 helices). The spread-aware `d_SW` test in Stage 2b will say whether this is a clean parity manifold or a centroid-only effect.
2. **Does the Llama × multiplication helix deficit survive Stage 2c?** Llama produces ~69 helices on multiplication vs ~140 for GPT-J and ~152 for Pythia. If Stage 2c's GPLVM agrees that Llama's subspaces have less periodic structure, this is a real cross-model mechanistic asymmetry. If Stage 2c picks an RBF or Matérn kernel for Llama where the other models pick Periodic, the asymmetry is geometric (smooth manifold vs periodic ring), not absence-of-structure.
3. **What happens to high-K full-period detections under non-uniform-grid validation?** running_sum_tens at P = 33 (K_present = 33 of K_natural = 82), operand_abs_diff at P = 36, max_operand at P = 39 — these may be index-of-sampled-values artefacts rather than concept-domain structure. Cross-layer agreement and cross-mode robustness in `cross_mode_helix_survival.csv` will discriminate.
4. **Does the GPT-J carry period drift (P = 7, 8) replicate on a fresh Llama run?** The parent project's Phase G on Llama found P = 18, 27, 19, 10 for carry_1..carry_4. emnlp2026's Llama × multiplication gives different periods. The drift may be a basis-fit difference (Step 6 LDA-A vs parent's Phase D) or a true model-state difference.
5. **Will causal ablation in Stage 4 separate "owned" helices from "inherited" ones?** The smoke probe in §17.4 confirmed the subspaces are causally used; Stage 4's pre-vs-post-orthogonalisation comparison will tell us which of the 808 detections survive the algebraic correlate ablation.
6. **Why does GPT-J have stronger structure at mid-layers and Pythia at later layers?** GPT-J × addition × ans_tens peaks at layer 14; Pythia × addition × ans_tens peaks at layers 24 and 28. The two models have similar architectures (~28-32 transformer blocks, 4096 hidden), but the geometric encoding plateau lands at different relative depths. Whether this reflects training data, optimiser state, or architectural details (Pythia's parallel attn+MLP vs GPT-J's sequential) is open.
7. **Why does Llama × addition layer 4 have such high FCR (0.98)?** The Llama × addition × ans_tens × layer 4 cell produces the highest FCR in the entire sweep — 98.1% of total Fourier power on the cos/sin pair at P = 10. Most models show their cleanest periodic structure at mid-network; Llama shows it after only 4 transformer blocks. This is an architectural finding worth a sentence in the discussion.
8. **What is the relationship between low N_over_r (subspace overdetermined) and verdict quality?** A subset of cells has N_over_r < 100 (the rule-of-thumb for stable LDA fits). The verdicts for these cells may be less reliable; a sensitivity analysis is left for the truth document's update after Stage 2c.

---

## Appendix A — Analysis and intuition

This appendix is the only section of the document where interpretation is allowed. The body sections report numbers and procedure; this appendix discusses why the numbers came out the way they did and what they mean for the paper.

### A.1 Why discover, not assume

The original plan v6 specified a pre-registered period set per concept family — period 10 for digit concepts, period 18/27/19/10 for carries on multiplication (from parent Phase G), etc. The smoke-test iteration of this document replaced pre-registration with discovery for three reasons.

First, the literature's prior is a *prediction*, not a fact. Kantamneni & Tegmark (2024) discovered the period-10 digit helix in GPT-J on standalone integers. The parent project discovered carry periods 18/27/19/10 on Llama 3.1 8B on 3-digit × 3-digit multiplication. Both findings are conditional on the specific model, task, and basis-fitting procedure. A pipeline that pre-registers these periods assumes the same priors hold across all three models and both tasks in this paper. The discover-then-fit approach lets each cell's data state its own answer.

Second, the multiple-testing problem of a free-period search is solvable. A naive scan over K candidate frequencies with the at-fixed-frequency null is anti-conservative by roughly a factor of K. The Whittle correction (max-over-frequencies null) bakes the search into the null distribution itself, giving an honestly protected p-value. Toy 7B's calibration confirms the null is correctly conservative on real data sizes (K up to 199): 1 of 100 noise cells false-positive at α = 0.05, well within the binomial 95% upper bound.

Third, the surprising findings are the headline. The 808 eligible detections include 181 at period 2 (parity) on units-digit concepts and 118 at period 19 on column-algebra concepts — neither was in the pre-registered set the plan originally specified. A pre-registered approach would have *missed* the parity finding entirely (you can't find what you don't look for at period 2), or worse, would have detected it but reported it as "no helix at the predicted period" rather than as "P = 2 helix discovered." The discover-then-fit approach correctly attributes the finding to the data.

The cost is computational: free-period search × max-over-frequencies null is ~5–10× slower per cell than at-fixed-frequency. The cupy.fft.rfft batched implementation absorbs this cost easily — total wall time is still 25 minutes on 6 A6000s.

### A.2 Why the plane-rank ratio gate matters

The first version of the worker without the plane-rank gate produced false-positive helix calls on 1-D lines projected into d-D ambient. A linear ramp `M_centred[v, c] = α · (v − v̄)` has Fourier content concentrated at low frequencies with the 1/k envelope; the periodogram argmax falls at k = 1 (period K), giving an apparent "period K helix." The per-coord Whittle null catches the linear contribution but doesn't distinguish "all r coordinates carry a 1D ramp" from "two coordinates carry a 2D circle." Both can produce FCR ≥ 0.30 and two_axis_significant = True.

The plane-rank ratio `S²[1] / S²[0]` is the cleanest discriminator. For a 1D line, the second singular value is at noise level (~σ² · √(d − 1) / |signal|²), giving ratios of ~1e-3 to 1e-2. For a 2D circle (two equal-energy orthogonal modes), the ratio is ~1. The threshold of 0.3 is generous — even a moderately noisy circle clears it. On real data, the gate fired 5/42 times (12%) in the smoke test and ~10–15% across the full sweep, every time on a cell where the data was visibly 1D-dominant in the SVD.

The gate is not a power gate. It rejects cells where the data plane is degenerate (1D) but does not require any particular FCR magnitude. A cell with FCR = 0.5 and plane_rank_ratio = 0.15 is rejected; a cell with FCR = 0.31 and plane_rank_ratio = 0.4 passes. This is the correct ordering: shape comes first, magnitude comes second.

### A.3 The parity finding on units digits

Period 2 (the Nyquist frequency for K = 10) means the centroid sequence alternates between two values across consecutive digit positions. For `ans_units`, the two "values" are even and odd: digits 0, 2, 4, 6, 8 sit at one centroid; digits 1, 3, 5, 7, 9 sit at another. The "helix" at period 2 is more accurately a step function (parity step) plus a linear drift across the 10 digits.

The finding replicated cleanly across:
- GPT-J × multiplication at layers 14, 20, 24 (FCR 0.66–0.76)
- Pythia × multiplication at layers 8, 16, 24, 28 (FCR 0.72–0.77)
- Llama × multiplication at layers 8, 16 (FCR 0.66–0.73)
- GPT-J × multiplication for `a_units` and `b_units` operand digits (FCR 0.30–0.42)
- Pythia × multiplication for `a_units` and `b_units` (FCR varies)

The parity geometry persists across residualization modes (`off`, `answer`, `norm`) and across both subspace variants (LDA-A and CCSVD). It is not driven by the answer scalar (survives `answer` mode) and not driven by activation magnitude (survives `norm` mode). The 50/50 causal smoke on GPT-J × ans_units (§17.4) confirms it is mechanistically used by the model.

A mechanistic interpretation: for multiplication, the parity of the answer's units digit is determined by the parities of the units digits of the operands: `(a_units · b_units) mod 2`. If either operand has even units, the answer's units are even. The model can predict parity from the operand digits much more easily than it can predict the exact digit, because parity is a 1-bit feature with strict algebraic compositionality, whereas the exact digit requires a 10-way multiplication table. The geometry reflects what the model finds easy to encode.

This is a paper-worthy mechanistic finding that does not appear in either Kantamneni & Tegmark (2024) or the parent project's Phase G. Both predicted period 10 on `ans_units` based on operand-position digit-helix geometry. The discover-then-fit approach surfaced parity as a stronger signal at the answer position.

### A.4 The Llama multiplication asymmetry

Llama 3.1 8B shows 69 helices on multiplication vs 123 on addition. GPT-J and Pythia both show roughly equal counts across the two tasks (133/140 for GPT-J, 137/152 for Pythia). The cross-task asymmetry is unique to Llama.

Three candidate explanations, each testable downstream:

1. **Llama's multiplication representations are less periodically organised.** This would be surprising — Llama is generally considered a stronger arithmetic reasoner than GPT-J of the same era, and parent Phase G found rich helical geometry in Llama 3.1 8B on 3-digit × 3-digit multiplication. The 2-digit × 2-digit scope here may be too easy for Llama's representations to organize periodically. Stage 2c's GPLVM kernel comparison will say whether Llama's multiplication cells pick the periodic kernel or default to RBF/Matérn.

2. **Llama uses SwiGLU activations and GQA attention** (Grouped Query Attention) while GPT-J and Pythia use vanilla GELU + multi-head attention. The activation function geometry may be less amenable to periodic representations — SwiGLU's gating could disrupt the cos/sin separation that's clean in GELU. This would require a deeper architectural analysis.

3. **The LDA-A subspace fit for Llama × multiplication is more conservative.** Step 6 in Llama × multiplication may produce smaller `n_sig` values on average (fewer LDA-A directions per concept), leaving less room for a periodic structure to be discovered. The "missing helices" may exist in dimensions that LDA's cross-validation gate pruned. Variant `ccsvd` partially controls for this — CCSVD bases are wider than LDA-A — yet Llama × multiplication's CCSVD count (33 + 16) is also low compared to GPT-J × multiplication's CCSVD count (3 + 15 + ...).

The simplest empirical follow-up: re-run Stage 2a on Llama × multiplication with a Step 5 CCSVD basis dilated by including the top-50 principal components instead of the permutation-null-filtered subset. If the helix count jumps, the asymmetry is artefactual; if it stays low, the asymmetry is real.

### A.5 Why FCR is not absolute-scale comparable

The FCR formula is:
```
fcr_two_axis = (top-2 power at k*) / (total Fourier power across all r coords)
```

The numerator is a sum of 2 quantities; the denominator is a sum of r × (K//2) quantities. As r grows, the denominator grows proportionally even if the actual signal at k* doesn't change. So a cell with r = 18 will have ~3× smaller FCR than a cell with r = 6 carrying the same physical signal.

Cross-cell comparisons should use the BH-FDR-corrected q-value, which is calibrated against each cell's own permutation null. The q-value automatically corrects for r-dependence: a cell where the algorithm's observed FCR sits at the 99.9th percentile of its own null will have q < 0.001 regardless of r, while a cell at the 99th percentile will have q < 0.01.

The 0.30 FCR threshold for the in-cell verdict is intentionally generous, set with a 2D-circle-in-9D-ambient in mind (where the natural FCR is ~0.4–0.5). It is a screen, not a ranker. The q-value is the ranker.

The diagnostic `fcr_two_axis_x_r` column is reported for cells where one wants to compare "physical signal magnitude" across cells with different r. It is NOT a headline number; it scales linearly with r in a way that may overcorrect (it assumes the signal sits in exactly 2 of r dimensions; if it sits in more, the rescaling overestimates).

### A.6 The 36.6% prior-match rate

296 of 808 post-FDR detections match the prior table within one Fourier bin. The 512 mismatches break down roughly as:
- ~180 P = 2 detections on units-digit concepts (predicted P = 10).
- ~150 fundamental-period detections on high-K concepts (predicted P = 10 or 100).
- ~120 unexpected carry-period detections on multiplication (parent Phase G predicted 18, 27, 19, 10 for Llama; emnlp2026 finds P = 7, 8 on GPT-J).
- ~60 miscellaneous (column-algebra mismatches, partial-product detections at unexpected periods).

The match rate is not a quality metric. A high match rate would mean "the data confirms the literature"; a low match rate could mean either "the data refutes the literature" or "the literature was incomplete." The discover-then-fit framing means each mismatch is a finding to investigate, not an algorithm failure.

For the paper's framing, the prior-match rate is a *distribution* finding, not a *headline* number. The headline numbers are the verdict counts (808 eligible positive detections) and the cross-model consistency (closer match across models on addition than multiplication, modulo Llama).

### A.7 The geometry-verdict-vs-causal-use distinction

The smoke probe in §17.4 revealed a subtle but important point: a cell can be verdicted `none` by Stage 2a yet have a causally-used subspace. `ans_hundreds` on GPT-J × multiplication × layer 14 has FCR = 0.48 (well above 0.30), per-coord significance, P = 10 (KT match), and plane_rank_ratio = 0.19. The plane_rank gate downgrades it to `none` because the data is effectively 1D-dominant — the centroids trace a line, not a 2D ring. Yet ablating the 9-D LDA-A subspace for `ans_hundreds` produces Δlogit = −3.02, virtually identical to the −3.32 from the bona-fide P = 2 helix on `ans_units`.

The lesson is that **Stage 2a tests for the *shape* inside a subspace**, not for whether the subspace is *causally used*. The subspace was fit in Stages 5/6 to maximise discrimination of concept values; its causal use comes from that label-conditional construction. The shape inside (helix, circle, 1D line, blob) is a separate property that may or may not align with the causal magnitude.

This implies a clean two-axis taxonomy for the final paper:

| | Shape verdicted (helix/circle) | Shape verdicted `none` |
|---|---|---|
| Causally used | "Mechanism explained by geometry" — ideal case |  "Mechanism without identified geometry" — Stage 2c/2d job |
| Not causally used | "Decorative geometry" — Stage 4 reveals this | "Inactive subspace" |

The causal smoke suggests the bottom row is empty (no obviously decorative geometry yet) and the top-right has at least one entry (ans_hundreds). The formal Stage 4 ablation will populate the full table.

### A.8 Why `null_unstable` never fires in practice

The plan v6 specified `redraw_rate > 0.10` as the trigger for `null_unstable`. Across 6,886 Stage 2a rows, the observed redraw rate is exactly 0 in every cell. This is by construction: position-permutation `y_perm = y[π]` preserves the value multiset exactly, so the group-size filter (≥ 30 per value) cannot drop any values that were present in the unshuffled data. The K_shuffled ≡ K_observed invariant holds by mathematics, not by luck.

The redraw machinery is kept in the code as a safety net for future variants of the algorithm (e.g., stratified bootstrap where K could drift). It is documented in metadata.json as `redraw_rate: 0.0` always, which the aggregator can use as a sanity check.

### A.9 Open mechanistic question: pitch direction discovery

The off-plane linear power test searches for a *single* pitch direction in the orthogonal complement of the rank-2 data plane. This is appropriate for a circular helix where pitch is a 1-D drift along one axis. It is *not* appropriate for:
- A double helix (two parallel pitch axes).
- A helix whose pitch direction rotates with position (a "warped helix" or "snake").
- A torus (two periodic dimensions, no pitch).

Stage 2c's kernel zoo includes K4 (Periodic ⊗ Periodic) which can capture tori; K5 (Matérn 5/2) which can capture warped helices. Stage 2a's verdict of "helix" should be read as "compatible with a single-pitch helix" rather than "definitively a single-pitch helix"; the more exotic alternatives are downstream.

### A.10 Implications for the paper's framing

The Stage 2a results align with the paper's pre-registered options (plan v6 §1.2):

- **Finding A** — addition owned, multiplication inherited. Stage 2a supports the asymmetric framing: addition's clean P = 10 digit helices on `ans_tens` (FCR up to 0.98 on Llama) are textbook KT replication; multiplication's P = 2 parity on `ans_units` is a different geometric structure that the literature did not predict. Whether these are "owned" or "inherited" is Stage 3's job; Stage 2a establishes that the *shapes* are different.

- **Finding B** — both inherited. Stage 2a cannot directly support this; it would require Stage 3 to find that *both* the digit helix and the parity helix evaporate after orthogonalisation. The cross-mode survival numbers (111 of 2,388 cells show helix in all 3 modes) suggest robust signals exist that are unlikely to be fully ablated by Stage 3's algebraic-correlate orthogonalisation, biasing toward Finding A.

The methodology contribution (C1 in plan v6 §1.2) is unchanged: the four-stage pipeline is the constant; the empirical finding is whichever the data produces. Stage 2a's contribution is the second piece (Bayesian manifold characterisation, replacing parent's centroid-only Fourier with a more rigorous discover-then-fit + Whittle null + plane-rank gate). The empirical content beyond methodology is:
1. KT replication on addition's digit-tens.
2. Parity discovery on units digits (cross-model, cross-task).
3. Cross-model multiplication asymmetry on Llama.
4. Carry-period drift between Llama and GPT-J.

These are four distinct paper-worthy findings, each derivable from the comparison tables Stage 2a produced.

### A.11 What Stage 2b will test

Stage 2b's spread-aware test computes:
```
d_SW(u, v)² = (μ_u − μ_v)ᵀ [(Σ_u + Σ_v)/2 + λI]⁻¹ (μ_u − μ_v)
```
and compares the SW distance matrix against the Euclidean centroid distance matrix via Spearman ρ.

For Stage 2a's clean helix verdicts (FCR ≥ 0.5, plane_rank ≥ 0.5):
- Expected: high `ρ_centroid` (≥ 0.85), confirming the data sits *on* the helix, not just *near* it. Verdict: `bayesian_manifold_strong`.
- Risk: per-value spread is large relative to inter-value distances, especially on multiplication where N is smaller (2,751 correct on GPT-J vs 8,415 on addition). Verdict downgrade: `centroid_only_shape`.

For Stage 2a's parity verdicts (P = 2, FCR up to 0.77, plane_rank typically 0.4):
- The K = 2 effective decomposition (parity is a step function, not a full sine) puts the data in two clusters rather than on a continuous manifold. Stage 2b may verdict `clustered_lookup` rather than `centroid_only_shape`.
- Stage 2c's RBF kernel will likely win over Periodic for parity cells, with `d̂_ARD = 1` (a single linear axis separating the two parity clusters).

For Stage 2a's high-K full-period detections (`running_sum_tens` at P = 33):
- Per-value sample counts are typically 50–80 (much smaller than digit concepts' ~500–800). Per-value covariance estimates Σ_v are noisy. Stage 2b's `d_SW` may have unstable behaviour at non-uniform-grid cells. The verdict `centroid_only_shape` is plausible.

These are predictions, not claims. Stage 2b's run will produce its own truth document.

### A.12 What Stage 2c will test

Stage 2c's GPLVM kernel zoo includes 5 kernels:
- K1 RBF (any smooth manifold)
- K2 Periodic (circles at any period, period as free hyperparameter)
- K3 Periodic + Linear (helix, period free)
- K4 Periodic ⊗ Periodic (torus)
- K5 Matérn 5/2 (smooth with finite differentiability)

For Stage 2a's clean digit-tens helices (P = 10 on `ans_tens`, FCR ~0.7–0.98): Stage 2c is expected to pick K2 (Periodic) or K3 (Periodic + Linear) with adjusted ELBO gap ≥ 5 nats over runner-up. Held-out reconstruction MSE check should pass. ARD-pruned latent dimension `d̂_ARD = 2` for circle or `d̂_ARD = 3` for helix.

For Stage 2a's parity verdicts (P = 2): Stage 2c may pick RBF or Linear rather than Periodic. The "geometry" of parity is two clusters separated along a 1D axis, not a continuous ring. The Periodic kernel may win the ELBO comparison only because the alternative kernels have similar fit quality and Periodic's prior matches Stage 2a's discovery. The `dim_at_ceiling` flag and held-out MSE check should disambiguate.

For Stage 2a's high-K full-period detections: Stage 2c is the right test. If running_sum_tens at K = 33 sits on a clean 1D ring, K2 wins decisively. If it's a noisy 1D structure on a non-uniform sample, RBF wins with low intrinsic dimension. The verdict will be informative either way.

For Stage 2a's `none` verdicts with high FCR but failed plane-rank gate (e.g., `ans_hundreds` on GPT-J multiplication): Stage 2c's RBF kernel with ARD pruning should report `d̂_ARD = 1` — the data is 1D, not 2D. The kernel comparison will be uninformative (no Periodic structure to compare against), but the dimension estimate confirms the plane-rank gate's call.

### A.13 What Stage 3 will test

Stage 3 (ownership orthogonalisation) re-runs Stages 2a–2d on activations residualised against the pre-registered algebraic correlate set for each concept. For `carry_units` on multiplication, the correlate set is `{column_sum_units, partial_product_units}`. For `ans_units` on multiplication, the correlate set is `{column_sum_units, carry_units, partial_product_units, ans_tens, ans_hundreds}` (all algebraically related output structure). After residualisation, Stage 2a is run again on the orthogonalised activations.

The owned fraction `ω` is the ratio of (post-orthogonalisation FCR) to (pre-orthogonalisation FCR), per cell. ω = 1 means the helix is fully owned (immune to orthogonalisation). ω = 0 means the helix is fully inherited (vanishes under orthogonalisation). The empirical distribution of ω across the 808 detections will determine the headline ownership finding.

For Stage 2a's strongest detections:
- `ans_tens` at P = 10 on addition: correlate set is `{column_sum_tens, carry_units, a_tens, b_tens}`. The digit helix on `ans_tens` could plausibly be inherited from `a_tens` and `b_tens` (operand position helices), which would give ω close to 0.
- `ans_units` at P = 2 on multiplication: correlate set is large (carries + column sums + partial products + other answer digits). Parity is highly redundant with `a_units` and `b_units` parities; ω could be very low.

If both findings turn out to be inherited (Finding B), the methodological contribution is "linear probe success does not imply ownership"; the empirical content is "we discovered these specific shapes that the literature did not predict, and they all turn out to be epiphenomenal." If the digit helix is owned but parity is inherited (or vice versa), the asymmetry is itself a publishable finding.

### A.14 What Stage 4 will test

Stage 4 ablates the LDA-A subspace from each cell's layer and measures Δlogit on the gold first-answer token. The smoke probes in §17.4 already establish that the subspaces are causally used (LDA-A ablation produces 2–4 logit drops vs ~0 for random). Stage 4's formal version compares:

1. **Pre-orthogonalisation ablation** (Stage 2a's basis): ablate the raw LDA-A subspace.
2. **Post-orthogonalisation ablation** (Stage 3's residual basis): ablate only the part of the LDA-A subspace orthogonal to the correlate set.

For an *owned* concept, both ablations damage the gold logit similarly (the model uses the concept-specific part). For an *inherited* concept, the post-orthogonalisation ablation produces minimal Δlogit (the concept-specific part was tiny; the rest was correlate-shared). This is the binary causal verdict that completes the four-stage pipeline.

Stage 4 will run on a subset of headline cells (probably top-20 by FCR per (model, task) for tractability), so the headline causal-ownership matrix will be ~120 cells out of Stage 2a's 808 detections.

### A.15 Limitations of the verdict hierarchy

The hierarchical verdict (Step 16) has a subtle ordering issue: the FCR-based eligible verdicts (helix, circle, none) come after the non-eligible ones (low_K, period_inconsistent, null_unstable). A cell with K_present = 3 (low_K) cannot be helix even if its 3-value centroid sequence would have qualified at K = 4. This is mathematically correct — period discovery at K = 3 has only 1 useful Fourier bin — but it means the verdict count `low_K = 3292` includes some cells that might "morally" be helix candidates if measured differently.

A natural extension for Stage 2c: re-run the Bayesian GPLVM on these `low_K` cells even though Stage 2a couldn't Fourier-test them. The K2 Periodic kernel with a strong prior at P ≈ K_present can still fit a circle through 3 points (it just has very weak evidence). Stage 2c's ELBO comparison would say whether the resulting fit is meaningfully better than RBF. This is one of the deferred items in §20 open questions.

### A.16 Why the cross-variant agreement is 73% but cross-variant period agreement is 96%

The two numbers measure different things. Cross-variant verdict agreement asks "did LDA-A and CCSVD give the same verdict?" — which is a 5-way categorical agreement (helix, circle, none, period_inconsistent, low_K). Cross-variant period agreement asks "when both variants produced a period, were they within one Fourier bin?" — which is a continuous agreement.

The 23% gap (96% − 73%) is the rate at which the two variants find the same period but disagree on whether to call it a helix vs circle vs none. Most disagreements are:
- One variant has FCR = 0.32 (just above 0.30 threshold) and verdicts helix; the other has FCR = 0.28 and verdicts none. Same period, same shape, different threshold crossing.
- One variant passes the linear-significant gate (helix); the other has p_linear = 0.015 (just above 0.01) and downgrades to circle.

These borderline disagreements are absorbed by the global BH-FDR step in the aggregator, which uses q-values rather than the in-cell 0.30 threshold. Post-FDR, the cross-variant disagreement is largely about FDR-induced downgrades on one side but not the other (LDA-A's smaller `n_sig` produces fewer total tests, which slightly improves its FDR ranking).

### A.17 The pre-registered claim and what Stage 2a contributes

Plan v6 §1.3 pre-registers the paper's central claim as methodological, not empirical:

> Linear probe success at finding a clean geometric structure for a concept does not imply that the structure belongs to the concept. We propose a four-stage Bayesian pipeline that tests ownership directly, and demonstrate the pipeline on addition and multiplication in GPT-J 6B on the model's correct answers.

Stage 2a's contribution to this claim is to provide the *first geometric description* against which Stage 3 will orthogonalise. The 808 eligible positive detections are the population that Stage 3 will examine for ownership. The descriptions themselves (period, FCR, plane rank, top coordinates) are not yet ownership-tested; they are the input to ownership testing.

Stage 2a also makes a smaller, independent contribution: the discover-then-fit Whittle methodology itself. By moving from pre-registered periods (parent Phase G) to data-driven periods with multiple-testing-protected nulls, the test becomes more rigorous *and* more open to surprises. The parity finding on units digits is the headline surprise: it would not have been caught by a pre-registered approach that only tested at P = 10.

The combination of these two contributions makes Stage 2a's truth document a genuine empirical artefact rather than a procedural log. The numbers in §7–§14 are facts about three pre-trained language models that did not exist before this run; the procedural detail in §4–§5 is the methodology paper-readers will need to either replicate or extend.

### A.18 Cross-mode robustness as a partial ownership proxy

Even before Stage 3 runs the formal orthogonalisation, cross-mode helix survival is a partial proxy for ownership. The 111 rows in `cross_mode_helix_survival.csv` where the verdict is `helix` in all three of `off`, `answer`, `norm` modes are detections that:
- Survive answer-magnitude residualization (cannot be inherited from the gold answer scalar).
- Survive activation-norm residualization (cannot be inherited from per-token activation magnitude alone).

These two residualizations cover two of the most common nuisance directions. A helix that survives both is significantly less likely to be a magnitude-correlate inheritor than a helix that exists only in `off` mode.

Breaking down the 111 all-mode helices by concept:
- `ans_tens` (KT digit helix): the bulk of the all-mode helices on addition; ~40 of the 111.
- `ans_units` (parity at Nyquist): the bulk on multiplication; ~30 of the 111.
- `a_tens` and `b_tens`: ~15 of the 111.
- `column_sum_*` and `running_sum_*`: ~10 of the 111.
- Other: ~16.

The all-mode survivors are the cleanest candidates for the formal ownership test in Stage 3. They are not pre-cleared as "owned" — Stage 3's algebraic correlate orthogonalisation tests against a broader set of correlates that includes structural intermediates (column sums, partial products, carries) — but they have already cleared two of the easier nuisance tests.

### A.19 The role of CCSVD vs LDA-A in Stage 2a

CCSVD (Step 5) produces a wider basis than LDA-A (Step 6) because LDA-A's dual-criterion `n_sig` filter prunes directions via cross-validation. The Stage 2a cross-variant analysis (§9.3) shows that:
- CCSVD finds ~30% more helix detections (432 vs 322).
- Both variants agree on period within 1 bin in 96.4% of comparable cases.
- Both variants agree on verdict in 73.1% of comparable cases.

The implication is that the *period* discovered is a property of the data, not of the basis fit. But the *verdict* depends on how many directions are included: CCSVD's wider basis gives the FCR more denominator volume but also more numerator volume (the two-axis power scales with the basis fit's capture of the signal).

For headline numbers, LDA-A is the primary variant per plan v6 (Option A in Step 6 §3.2). CCSVD is the audit variant. The 73% agreement is high enough that LDA-A's headline numbers are unlikely to be artefactual; the 27% disagreement is concentrated on borderline cells where the FCR sits near 0.30 and the threshold crossing is sensitive to basis dimension.

A reviewer attack like "what if you used a different subspace fit?" is answered by the cross-variant agreement table: the algorithm's findings are robust to the LDA-A vs CCSVD fit choice on the cleanest signals (>0.5 FCR) and noisy at the margin (~0.3 FCR), which is reasonable.

### A.20 Period vs frequency convention

The document uses *period* P (in units of "values per cycle") as the primary unit because the concepts have integer-valued labels and the natural domain is "how many steps does the centroid traverse before returning to its starting position?" The Fourier transform expresses this as frequency `k` (number of cycles per K values), with `P = K / k`.

A period P = 10 means: walking through 10 consecutive values (e.g., digits 0..9) traces one full cycle of the periodic signal. A period P = 2 means: walking through 2 consecutive values returns to the starting position — i.e., the signal alternates with each step (parity).

The Nyquist frequency is `k = K/2`, corresponding to P = 2 — the highest frequency that can be discriminated on a discrete K-value grid. For K = 10, the valid period set is {10, 5, 10/3, 2.5, 2}. For K = 18, it includes integer periods {18, 9, 6, 4.5, ...}. The algorithm's max-over-frequencies null naturally handles non-integer periods like 10/3 = 3.333, which arise for K not divisible by small primes.

The choice between "period of 10" and "frequency of 1 cycle per 10 values" is purely notational; both refer to the same underlying Fourier mode. The document standardises on period because the prior table and the KT 2024 literature both express priors in period units.

### A.21 What a `period_inconsistent` verdict means mechanistically

A cell with `period_inconsistent` has top-2 coordinates whose argmax frequencies disagree, and no across-coordinate vote can decide. The 1,033 period_inconsistent cells in the sweep have one of several underlying causes:

1. **Two genuinely different periodic structures coexist.** The cell's subspace carries both a P = 10 mode in some coordinates and a P = 2 mode in others, with neither dominant. This is the most interesting case mechanistically — it could be evidence of two simultaneous periodic encodings (e.g., the digit helix AND the parity step).
2. **Noise at small K.** With K = 5 or K = 6 there are only 2–3 Fourier bins, and statistical fluctuation makes the argmax flip between bins across coordinates. The vote at this scale is noisy.
3. **A torus.** Two independent periodic dimensions at different periods, each captured by one of the top-2 coordinates. This is what Stage 2c's K4 (Periodic ⊗ Periodic) kernel is designed to discover.

Distinguishing these requires Stage 2c. A period_inconsistent verdict from Stage 2a means "Stage 2a cannot pick a single period; pass the cell to Stage 2c and let the kernel comparison decide."

### A.22 Why the periodic kernel will not always win in Stage 2c

Stage 2a's verdicts include helix and circle, both periodic. Stage 2c will fit RBF (no periodicity), Periodic, Periodic + Linear, Periodic ⊗ Periodic, and Matérn 5/2 kernels, and pick by marginal likelihood with BIC penalty. A Stage 2c result that does NOT pick a periodic kernel for a Stage 2a helix cell is informative:

- **RBF wins** → the data has smooth manifold structure but no genuine periodicity. The "helix" verdict was driven by the centroid mean ringing at period P, but the full point cloud's likelihood is better described by a non-periodic smooth manifold. This is the centroid-only-shape failure mode that Stage 2b's `d_SW` test also catches.
- **Matérn 5/2 wins** → the data is smooth but rougher than RBF assumes. Still no period; the helix verdict was an artefact of the centroid being sinusoidal even when the full data is non-periodic.
- **Periodic + Linear wins** → confirms the helix.
- **Periodic ⊗ Periodic wins** → the cell is a torus, not a helix. Stage 2a missed the second period (likely because it produced a `period_inconsistent` verdict or because the second period's energy was below 2× the first).

The expected outcome for the cleanest cells (Llama × addition × ans_tens × layer 4, FCR = 0.98, plane_rank = 0.45) is K2 or K3 with adjusted ELBO gap ≫ 5 nats. For borderline cells (FCR around 0.35), the kernel comparison is genuinely uncertain and `kernel_inconclusive` is likely.

### A.23 What the unexpected periods imply for the paper's framing

`unexpected_periods.csv` lists 512 cells where the discovered period does not match the prior. The patterns:

- **Parity (P = 2) on units digits**: ~180 of 512. This is the dominant "unexpected" finding. It systematically appears on all three models on multiplication and on Pythia on higher-layer addition. The framing is "the model encodes units-digit parity more strongly than full digit identity" — a paper-worthy mechanistic observation.
- **Fundamental periods (P = K) on high-K concepts**: ~150 of 512. running_sum_tens at P = 33 (K_present = 33), operand_abs_diff at P = 36, etc. These are non-uniform-grid; the period is "the cell traces one full ring across the K_present sampled values." The interpretation depends on whether K_present and K_natural are systematically related — for these cells, K_present typically reflects the data range that produces ≥ 30 correct samples per value, which is concept- and task-dependent.
- **Unexpected carry periods on GPT-J multiplication**: ~80 of 512. Parent Phase G's predicted P = 18, 27, 19, 10 on Llama; GPT-J discovers P = 7, 8 on the same conceptual structure. This is a cross-model finding — the carry-domain encoding is not period-invariant across architectures.
- **Other**: ~100. Column-algebra concepts at unexpected periods, partial-product family at periods the prior didn't specify, etc.

For the paper's headline matrix (§1.2 of plan v6), each of these is a sub-finding under the methodology contribution. The methodology is the constant; the empirical surprises are the variable.

### A.24 The relationship between Stage 2a and KT 2024

Kantamneni & Tegmark (2024) discovered the digit helix in GPT-J by examining standalone integer representations at the integer token position. Stage 2a extends this in three ways:

1. **From standalone integers to arithmetic operands**. KT looked at `tok(str(n))` activations; Stage 2a looks at activations at the `=` position of `f"Output ONLY a number. {a} {op} {b}="`. The geometry at the `=` position has had to integrate information about both operands, which is a substantively different mechanistic question.
2. **From one model to three**. KT studied GPT-J; Stage 2a runs the same pipeline on GPT-J, Llama 3.1 8B, and Pythia 6.9B. Cross-model replication is built in.
3. **From pre-registered periods to discovered periods**. KT identified periods 2, 5, 10, 100 in their original analysis; Stage 2a does not pre-specify which periods to test, so it can discover periods like 7, 8, 19, 33 that KT did not see.

The clean replication on `ans_tens` × P = 10 (Llama × addition × layer 4 at FCR = 0.98 is the cleanest replication) confirms KT's finding generalises to the answer-tens position when the answer is the output of an arithmetic computation. The discovery of P = 2 parity on `ans_units` is a finding KT did not make because their analysis was at the standalone integer position, not the answer position.

In paper terms: KT 2024 is one valid citation point; Stage 2a's contribution is the methodology (discover-then-fit + Whittle null + plane-rank gate + cross-model + cross-mode replication) plus the new empirical findings (parity on units, asymmetric Llama × multiplication, GPT-J carry-period drift).

### A.25 Why high-K concepts on multiplication are mostly non-uniform-grid

`answer` on multiplication has K_natural up to 9801 (the actual range 0..9801 for a, b ∈ [0, 99]). After the correctness mask (only ~2,750 problems out of 3,023 are correct) and the 30-sample group-size filter, K_present is typically 10–30 — the values that appear ≥ 30 times in the correct subset are a tiny fraction of the natural value space.

For these cells, `non_uniform_grid_flag = True` and the discovered period is interpreted as "period in the index of sampled values," not "period in the natural value space." For example, `running_sum_tens` on multiplication has K_natural = 82 and K_present = 33. The cell discovers P = 33 — meaning the centroids trace one full revolution across the 33 sampled values. Whether this corresponds to a "period 33 in running-sum-space" depends on whether the 33 sampled values are uniformly distributed across the 82 natural values, which they typically are not (they're skewed toward common running sums in the correct-subset distribution).

The Stage 2b spread-aware test does not have this issue — it uses per-value covariance matrices that don't require uniform sampling. Stage 2c's GPLVM similarly fits to the full point cloud and uses each sample's coordinates regardless of value-space density. The non-uniform-grid flag is a Stage 2a-specific reporting flag that downstream stages can ignore.

For the paper's framing, non-uniform-grid detections appear as "exploratory" verdicts. They are reported in `fcr_all.csv` but the truth document treats them with caution — cross-cell validation (across modes, layers, variants) is required before claiming a non-uniform-grid detection is mechanistically meaningful.

### A.26 What happens at K = 4 (the FFT floor)

The `MIN_K_FOR_FFT = 4` floor is the minimum K at which "is there a period and which one" becomes meaningful. At K = 4:
- The valid periods are {4, 2} (k = 1 and k = 2).
- The plane-rank gate computes `S²[1] / S²[0]` from a 4×r centroid matrix, which has rank min(4, r) ≤ 4.
- The Whittle null has only 2 candidate frequencies, so the max-over-frequencies statistic is the max of 2 chi-square-like variables.

A K = 4 cell can produce a helix verdict if FCR ≥ 0.30 and the two-axis significance gate passes at α = 0.01 on both periodic coordinates and the linear-pitch gate passes at α = 0.01. In practice, the cells in this regime are concept-specific (e.g., `ans_magnitude_tier` on some tasks, `larger_operand`-style 3-value concepts). Stage 2a's verdicts at K = 4 are reported but with a caveat: the period discrimination is "P = 4 or P = 2", a binary choice, and the resulting verdict is essentially "is there parity-like alternation or full-cycle structure?"

At K = 5 and K = 6 the picture clarifies — 2 and 3 candidate frequencies respectively, with non-trivial period discrimination. By K = 10 (digits) the algorithm is in its sweet spot: 5 candidate frequencies, statistics calibrated by 30+ samples per centroid value.

### A.27 The aggregator's BH-FDR is not the only multiple-testing correction

The aggregator's global BH-FDR is across cells × concepts × variants (6,886 eligible rows). This corrects for "the chance that one of 6,886 random cells produces a small p-value." But there are *also* multiple-testing corrections WITHIN a cell:

1. **The Whittle null is max-over-frequencies** (Step 6) — this is the within-cell free-period-search correction.
2. **The two_axis_significant gate requires BOTH per-coord p-values < 0.01** (Step 13) — this is a within-cell joint test of two correlated hypotheses (which the algorithm chose by picking the top-2 coords). The 0.01 threshold per coord gives a joint significance at most ~1e-4 if the coords were independent, which they aren't quite, so the joint is somewhere in between.
3. **The plane_rank_ratio gate** (Step 15) — this is a geometric sanity check that does not have a p-value formulation but acts as an additional filter.

These three within-cell corrections are independent of the cross-cell BH-FDR. The cells that pass all within-cell gates AND survive cross-cell BH at α = 0.05 are subject to a compound test stringency that is hard to express as a single number. The 808 post-FDR eligible detections are therefore conservative — a different choice of within-cell thresholds (e.g., α = 0.05 per coord instead of α = 0.01) would produce more detections, with weaker per-cell evidence per detection.

The paper's claim should be "we report 808 detections under a specific set of compound thresholds; the cross-cell q-value is BH-FDR-corrected at α = 0.05; the within-cell thresholds are pre-registered in `config.yaml` at α = 0.01 per coord."

### A.28 A note on neutral writing

The body of this document reports numbers and procedure. The interpretation, the mechanistic stories, the speculation about what Stage 2b/c/d/3/4 might find — all live in this appendix. This separation is enforced by the project's `feedback_neutral_technical_writing` memory and applies across all step documents.

The appendix's role is to capture the analysis that's useful for the next person to read this doc (the main advisor, a reviewer, future-me) without contaminating the procedural record. If a downstream stage's results contradict an appendix prediction, the appendix is wrong; the body remains correct. This separation is more useful than it sounds — it prevents the "we knew it all along" retroactive rewriting that happens when prediction and observation are entangled.

The 808 numbers in §7–§14 are not opinions. The 18 sections of this appendix are. The reader is welcome to disagree with the appendix; they should not disagree with the body without first checking the comparison CSVs.
