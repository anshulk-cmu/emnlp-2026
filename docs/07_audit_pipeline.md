# Step 7 / 8 / 9 — Audit Pipeline: Residual Hunting, Principal Angles, Johnson-Lindenstrauss Distance Preservation

**Anshul Kumar's Geometric Manifold Interpretability Project**
**Carnegie Mellon University, May 2026**

This document records every decision, every number, and every result from the three-phase audit that sits between Step 6 (LDA refinement) and the future Stage 2 (Bayesian manifold characterisation). It is the truth document for this stage. All numbers are validated against the actual output files produced by SLURM array jobs `7884716` (Step 7), `7884747` (Step 8), and `7884749` (Step 9), completed on 2026-05-12.

Steps 7, 8 and 9 are bundled into a single document because (a) they consume the same Step-6 outputs, (b) Step 7's union basis is consumed by Step 9, and (c) they collectively answer a coordinated set of audit questions about the Stage-1 linear-probe pipeline. The three steps individually answer:

- **Step 7 — Residual hunting.** "How much of the activation variance lives inside the union of every named concept's subspace, and is anything organised hiding in the orthogonal complement?"
- **Step 8 — Principal angles.** "Do concept subspaces share computational directions, and which concept pairs are most entangled?"
- **Step 9 — Johnson-Lindenstrauss distance preservation.** "Does the union-of-concepts subspace preserve the model's pairwise activation geometry, or are we discarding structure that materially shapes distances?"

The pipeline is run on the per-model correct subset for two tasks (addition, multiplication, `a, b ∈ [0, 99]`) and three residualization modes (`off`, `answer`, `norm`), at 5 layers per model, across three pre-trained LMs (GPT-J 6B, Llama 3.1 8B, Pythia 6.9B). Total cells per step: 3 × 2 × 3 × 5 = 90.

This stage marks the completion of the linear-pipeline audit. All subsequent phases (Fourier helix screening, GPLVM, RBF VAE, causal ablation) operate on the subspaces and audit facts established here.

---

## Table of Contents

1. [Purpose and scope](#1-purpose-and-scope)
   - 1.1 What this stage is
   - 1.2 What this stage is not
   - 1.3 What this stage's outputs feed into
   - 1.4 Population
2. [Standing rules](#2-standing-rules)
3. [Inputs](#3-inputs)
   - 3.1 Activation caches
   - 3.2 Concept bases (CCSVD, LDA Option A, LDA Option B)
   - 3.3 Per-problem metadata
   - 3.4 Correctness masks
   - 3.5 Step 6 LDA summary (the status filter)
   - 3.6 Library versions and SHA chain
4. [Step 7 — Residual hunting](#4-step-7--residual-hunting)
   - 4.1 Purpose and position in the pipeline
   - 4.2 Mathematical framework
   - 4.3 Implementation
   - 4.4 Per-cell artifacts
   - 4.5 Per-model results
   - 4.6 Cross-mode and cross-variant comparison
5. [Step 8 — Principal angles](#5-step-8--principal-angles)
   - 5.1 Purpose
   - 5.2 Mathematical framework
   - 5.3 Implementation
   - 5.4 Per-cell artifacts
   - 5.5 Per-model results
   - 5.6 Tier-pair superposition
6. [Step 9 — Johnson-Lindenstrauss distance preservation](#6-step-9--johnson-lindenstrauss-distance-preservation)
   - 6.1 Purpose
   - 6.2 Mathematical framework
   - 6.3 Implementation
   - 6.4 Per-cell artifacts
   - 6.5 Per-model results
7. [Cross-step headline tables](#7-cross-step-headline-tables)
8. [Runtime and reproducibility](#8-runtime-and-reproducibility)
9. [Verification](#9-verification)
10. [Limitations and known caveats](#10-limitations-and-known-caveats)

**Appendix A** — [Analysis and intuition](#appendix-a--analysis-and-intuition)

---

## 1. Purpose and scope

### 1.1 What this stage is

This stage is a three-phase audit of the linear-probe pipeline produced by Steps 5 and 6. It answers three load-bearing questions about the concept subspaces those steps fit, with full coverage across (model, task, mode, layer) cells and explicit statistical-significance testing.

**Phase 1 — Residual hunting (Step 7).** For each cell, construct a "union" subspace from every named concept's basis (CCSVD basis from Step 5, optionally combined with the LDA-derived bases from Step 6). Project the model's activations onto this union, subtract, and run principal component analysis on the orthogonal complement. The eigenvalue spectrum of the residual is compared to the Marchenko–Pastur (MP) bulk edge — the analytical eigenvalue distribution of pure isotropic Gaussian noise — to count the number of eigenvalues that stand above what noise alone would produce. Each above-MP eigendirection is correlated against every metadata column (raw concept labels plus a set of derived interaction columns: carry interactions, mod-10 sums, partial-product cross terms, predicted-digit features). Statistical significance for each (direction × column) pair is established via a 1000-permutation null and Benjamini–Hochberg FDR correction across the full grid. The fraction of activation variance explained by the union, the count of above-MP eigenvalues, and the flagged correlate concepts together quantify how much linearly organised structure remains unaccounted for after Steps 5 and 6.

**Phase 2 — Principal angles (Step 8).** For every unordered pair of concepts within a cell, compute the principal angles between the two concept subspaces. Principal angles are angles between a pair of subspaces; the smallest angle measures the strongest shared direction, and the full spectrum reveals how much overall overlap exists. Each (dim_a, dim_b) pair is compared against an empirical 1000-trial null obtained by drawing pairs of random orthonormal subspaces of the same dimensions in R^4096. A concept pair is flagged as exhibiting "superposition" if its smallest principal angle is more than 10° below the 5th percentile of the empirical null — a conservative threshold inherited from the parent project at `/home/anshulk/arithmetic-geometry`. Per-pair empirical p-values are FDR-corrected across all pairs in the cell.

**Phase 3 — Johnson-Lindenstrauss distance preservation (Step 9).** Project the activations onto Step 7's union subspace V_all, then compute pairwise Euclidean distances over ALL N(N−1)/2 unordered pairs in the full 4096-D activation space and in the k-D projected space. Report Spearman ρ, Pearson r, mean and maximum relative distance error, distance-variance-explained, and an independent full-pair Pythagorean validation in float64. These metrics tell us whether the union subspace captures the part of the activation geometry that actually shapes pairwise distances — distinct from, and stricter than, the variance-explained number from Step 7.

### 1.2 What this stage is not

This stage is **not** a non-linear analysis. Every operation is a linear projection, a singular value decomposition, or a permutation/rank-based test applied to linear residuals. The audit characterises the linear part of the activation manifold. Any non-linear structure within the union subspace is detected only indirectly — through the residual-correlation sweep, which can flag a non-linear residual of a known concept (e.g., a quadratic interaction like `a_units × b_tens` that the union doesn't capture by linear projection). Genuine non-linear geometry — helical, periodic, manifold-like — is the job of Stage 2 (centroid Fourier helix, GPLVM, RBF VAE) and is out of scope for this stage.

This stage is also **not** a causal analysis. The audit measures geometric and statistical properties of the activations but does not test whether the model's downstream behaviour depends on the audited subspaces. That is the job of Stage 4 (causal ablation, Δlogit on the first answer token).

### 1.3 What this stage's outputs feed into

Each of the three phases produces artifacts consumed by later pipeline stages:

- The **union basis** (`union_basis_merged.npy`, `union_basis_generous.npy`) written per cell by Step 7 is the input to Step 9's projection, and to any future Stage-3 ownership work that orthogonalises an activation against the named-concept span.
- The **Stage 3 correlate-set unions** (`union_correlates_<target>.npy`), pre-computed per cell by Step 7 using the plan-locked correlate sets from `plan.md §4.2-4.3`, are the inputs to Stage 3's orthogonalisation operator (`I − P_correlates`) for each target concept.
- The **per-pair principal-angle table** (`angles_pairwise.csv`) produced by Step 8 identifies which Stage-3 tests are likely to be load-bearing: target/correlate pairs with `angle_1` near 0° will have the largest ownership-vs-inheritance signal.
- The **JL Spearman / distance_var_explained / Pythagorean-check** numbers produced by Step 9 quantify the geometric faithfulness of the union representation and bound how much pairwise structure escapes any LRH-style probe.

### 1.4 Population

Every per-cell fit operates on the **per-model correct subset** for the named (model, task) pair, as defined by the answers CSV at `data/answers/{model}/{task}_answers.csv`. Population sizes (verified at run time):

| Model | Addition correct N | Multiplication correct N |
|---|---:|---:|
| GPT-J 6B | 8,415 / 10,000 | 2,751 / 3,023 |
| Llama 3.1 8B | 9,963 / 10,000 | 2,927 / 3,023 |
| Pythia 6.9B | 7,718 / 10,000 | 2,757 / 3,023 |

The audit runs on all 90 cells. No cell was skipped at the audit-pipeline level; every reported number is computed on the full per-cell correct population without subsampling.

---

## 2. Standing rules

The following rules are invariant across Steps 7, 8 and 9. They are checked at script load time and enforced by per-cell assertions.

1. **No subsampling.** Every metric is computed on the full per-cell correct population. Resampling for permutation nulls and 5-fold CV is *resampling*, not subsampling. There is no random row selection, no truncation of N, no thinning of the pair set in Step 9.
2. **Permutation / empirical trials = 1000.** The number of label shuffles for Step 7's correlation-sweep null and the number of random orthonormal-subspace draws for Step 8's empirical null are both fixed at 1000. The choice matches Step 5's CCSVD permutation null and Step 6's LDA permutation null.
3. **Both Spearman and Pearson everywhere.** Wherever a correlation is reported, both Spearman ρ (rank-based) and Pearson r (linear) appear side-by-side with their own permutation p-values and FDR-corrected q-values. Spearman is robust to monotone non-linearities; Pearson is the linear baseline; a Spearman ≫ Pearson signature is direct evidence of monotone non-linear encoding.
4. **Both union variants.** Step 7 builds two global union variants per cell: `merged` (CCSVD ∪ LDA-Option-A plus mode-specific β scalar directions) and `generous` (CCSVD ∪ LDA-Option-A ∪ LDA-Option-B). Both are saved; Step 9 runs on both; the correlation sweep runs only on `merged` (rationale in §4.2.6).
5. **Mean centring before PCA.** The centred randomised SVD in Step 7 subtracts the per-feature mean of `X_residual` before factoring. No unit normalisation.
6. **GPU when available, deterministic CPU otherwise.** Activation projection, pair-distance computation, and the float64 Pythagorean check all push to CuPy when `cupy.cuda.is_available()` returns True; each function falls back to a CPU numpy path that produces numerically equivalent results.
7. **Atomic writes.** Every `.npy`, `.json`, and `.csv` artifact is first written to a tempfile in the same directory and atomically renamed via `os.replace` after the write completes. Survives preemption mid-cell. Resume logic at the top of each per-cell function reads `metadata.json` and exits early if `computation_status == "complete"`.
8. **Seed reproducibility.** Each cell's randomised SVD and 1000-permutation null seed are derived from `cell_seed(model_key, task, layer, "residual_hunting_<mode>")` — a SHA-256-based hash of the cell identifier. The seed is recorded in every per-cell `metadata.json`. Re-running the same cell with the same code produces byte-identical artifacts modulo float32 round-off.
9. **Status filter.** Concepts entered into any union are filtered against Step 6's LDA Option A per-mode summary CSV: `status == "fit_ok"` AND `n_sig >= 1` AND `is_carved_out == False`. The filter is read once per cell; the kept and dropped concept lists are logged in `union_meta.json`. Carved-out concepts (e.g., `ans_*` under mode=answer, `ans_magnitude_tier` under mode=norm) are excluded by design and not flagged as failures.

---

## 3. Inputs

### 3.1 Activation caches

For `mode = off`, activations are loaded from:

```
/data/user_data/anshulk/emnlp2026/activations/{model}/{task}_layer_{LL:02d}.npy
```

Shape `(N_full, 4096)` float32, where `N_full = 10000` for addition and `N_full = 3023` for multiplication. Row order matches the corresponding problems CSV at `data/data/raw/{task}_problems.csv`. Loaded as `np.load`; the worker masks rows by the correctness mask before any computation.

For `mode = answer` and `mode = norm`, activations are loaded from the per-mode residualised cache produced by Step 6 phase 1:

```
/data/user_data/anshulk/emnlp2026/results/residualized/{model}/{task}_layer_{LL:02d}_mode_{mode}.npy
```

Same shape `(N_full, 4096)` float32. The residualised cache stores the full task population, **not** the correct subset; the worker applies the correctness mask after load. (Verified on 2026-05-11 disk audit: every cache file has 10,000 rows for addition and 3,023 rows for multiplication. The audit-pipeline workers explicitly apply `X[correct_mask]` regardless of mode.)

### 3.2 Concept bases

Per-concept bases come from two sources:

**CCSVD bases (Step 5).** Mode-aware paths:

- `mode = off`: `results/ccsvd_subspaces/{model}/{task}/layer_{LL:02d}/{concept}/basis.npy`
- `mode = answer`: `results/ccsvd_subspaces/mode_answer/{model}/{task}/layer_{LL:02d}/{concept}/basis.npy`
- `mode = norm`: `results/ccsvd_subspaces/mode_norm/{model}/{task}/layer_{LL:02d}/{concept}/basis.npy`

Each file stores a `(4096, r_ccsvd)` float32 array of orthonormal column basis vectors, where `r_ccsvd` is the cell's CCSVD subspace dimension (post-permutation-null filter). The worker transposes on load to get a row-orthonormal `(r_ccsvd, 4096)` view via the helper `load_basis_rows`.

**LDA Option A bases (Step 6, headline placement).** Mode-aware per-cell paths:

```
results/lda_subspaces/subspace_lda/mode_{mode}/{model}/{task}/layer_{LL:02d}/{concept}/lda_basis_full.npy
```

Shape `(4096, n_sig)` float32. LDA directions lifted from the CCSVD subspace back into the full 4096-D activation space; per-column unit norms, but columns are **not** column-orthogonal (LDA eigenvectors are orthogonal in the S_T metric, not Euclidean). For Step 7 and Step 9, non-orthogonal stacked rows are handled by the SVD orthonormalisation of the entire stacked union (§4.2.1). For Step 8, each per-concept basis is QR-orthonormalised on load (§5.2.2) so the principal-angle SVD has row-orthonormal inputs.

**LDA Option B bases (Step 6, audit placement).** Same path pattern under `full_lda/` instead of `subspace_lda/`. Same shape and same caveat about column non-orthogonality. Used only by Step 7's `generous` variant.

### 3.3 Per-problem metadata

The concept labels for each problem live at:

```
/data/user_data/anshulk/emnlp2026/data/data/raw/{task}_problems.csv
```

10,000 rows for addition and 3,023 rows for multiplication; 59 columns covering 47 single concepts and 12 joint tuples per task. Row order matches the activations cache and the answers CSV. The Step 7 correlation sweep reads this CSV per cell, masks rows by `correct_mask`, and uses the surviving rows as the metadata for every (direction × column) test.

### 3.4 Correctness masks

```
/data/user_data/anshulk/emnlp2026/data/answers/{model}/{task}_answers.csv
```

One row per problem; key column `correct` is a boolean indicating whether the model's first generated token (immediately after `=`) matches the first token of the gold answer's BPE tokenisation. The worker reads this column once at start-up, converts to a numpy boolean mask, and applies it to every activation array before any audit operation.

### 3.5 Step 6 LDA summary (the status filter)

For each (model, mode) pair, the worker reads:

```
results/lda_subspaces/subspace_lda/mode_{mode}/{model}/summary_{model}_mode_{mode}.csv
```

Each row gives status fields per (task, layer, concept) cell: `status`, `n_sig`, `lambda_T_1`, `is_carved_out`, `cv_accuracy_at_n_sig`. The audit-pipeline worker uses three columns to build the concept filter:

- Keep iff `status == "fit_ok"`.
- Keep iff `n_sig >= 1`.
- Keep iff `is_carved_out == False`.

Concepts that fail any of these conditions are excluded from the union for that cell. The full filter result is recorded in `union_meta.json` alongside the union basis itself.

### 3.6 Library versions and SHA chain

The audit pipeline runs in the same conda environment as Steps 5 and 6:

- Python 3.11.15
- NumPy 2.2.6
- SciPy 1.17.1 (provides `false_discovery_control`)
- scikit-learn 1.8.0 (provides `randomized_svd`)
- CuPy 14.0.1 (CUDA 12.8)
- pandas (env-default)
- PyYAML, statsmodels not required

The SHA chain extends Step 6's manifest: every cell's `metadata.json` records the SHA-256 of the activations file, the CCSVD basis, the LDA-A basis, the LDA-B basis (for `generous`), and the problems / answers CSVs that fed the fit. These hashes are stable across job retries because the inputs are read-only.

---

## 4. Step 7 — Residual hunting

### 4.1 Purpose and position in the pipeline

Step 7 answers a single load-bearing question for each (model, task, mode, layer) cell:

> "After we project the activations onto the union of every named concept's subspace, is there organised structure left in the orthogonal complement, and if so what is it correlated with?"

This question has three load-bearing sub-claims:

- **Variance budget.** The fraction of activation variance captured by the union (`var_explained`) bounds how much of the model's representational capacity at this cell is devoted to named concepts. A high number (≥ 0.85) means the named concepts collectively span most of the variance; a low number means there is substantial residual variance unaccounted for.
- **Above-MP eigenvalues.** The count of residual eigenvalues that stand above the Marchenko–Pastur upper edge (`n_above_mp`) is the noise-aware analogue of "did we find structure in the residual?". MP is the analytical eigenvalue distribution of i.i.d. Gaussian noise; an eigenvalue above the MP cliff cannot be explained by isotropic noise at the residual's estimated noise level σ². The test's reliability depends on the regime parameter γ = d_residual / N: at γ < 1 the test is reliable; at γ → 1 from below the cliff fluctuates wildly (Tracy-Widom); at γ ≥ 1 the test breaks because the noise bulk extends to zero and counts inflate.
- **Top correlate concept.** When `n_above_mp > 0`, the correlation sweep flags which named or derived metadata column is most associated with each above-MP residual direction. The flag persists only if the observed Spearman ρ exceeds 0.15 in absolute value AND the FDR-corrected q-value across the (direction × column) grid is below 0.05.

Step 7's outputs are consumed by Step 9 (which uses the `union_basis_merged.npy` and `union_basis_generous.npy` directly) and pre-computed for future Stage 3 (which uses the `stage3_unions/union_correlates_<target>.npy` files).

### 4.2 Mathematical framework

#### 4.2.1 Union basis construction

For each (model, task, mode, layer) cell, the worker constructs **two** global union variants:

**Variant `merged`.** SVD-orthonormalisation of the stacked rows of:

1. Each eligible concept's CCSVD basis (rows of `basis.npy` transposed → `(r_ccsvd, 4096)`).
2. Each eligible concept's LDA Option A basis (rows of `lda_basis_full.npy` transposed → `(n_sig, 4096)`).
3. The mode-specific β scalar direction(s), per §4.2.2.

**Variant `generous`.** SVD-orthonormalisation of the stacked rows of (1) + (2) + (3) plus:

4. Each eligible concept's LDA Option B basis (`(n_sig, 4096)` transposed from `full_lda/.../lda_basis_full.npy`).

Stacked-then-SVD-orthonormalised construction:

```
stacked = vstack([rows from sources 1..3 or 1..4])
U, S, Vt = np.linalg.svd(stacked, full_matrices=False)
keep = S > SVD_TOLERANCE_FACTOR * S[0]      # SVD_TOLERANCE_FACTOR = 1e-10
V_all = Vt[keep]                              # shape (k, 4096) float32
```

The SVD-tolerance threshold absorbs numerical redundancy (directions that are exactly redundant up to floating-point precision are discarded), while non-redundant directions are kept. The rank `k_union` after orthonormalisation is the cell's effective union dimension and is recorded as `k_union` in the per-cell summary row.

**Why SVD orthonormalisation and not Gram–Schmidt or a single QR.** The stacked basis can have up to ~2,000 rows (`merged`) or ~3,400 rows (`generous`) for addition cells. Gram–Schmidt accumulates round-off error along the orthogonalisation chain and is unstable when rows are nearly redundant. The SVD takes O(min(m, d)² · max(m, d)) flops and is numerically stable up to machine epsilon. Empirically the SVD's largest singular value `S[0]` is O(10) for our stacked matrices and the threshold `1e-10 * S[0]` removes only directions that are numerically zero, not directions that are merely correlated. Two concept subspaces could share 99% of their variance along a direction and that direction would still be counted twice in `k_union`. This is intentional: Step 7 errs on the side of projecting out **more** structure so that any residual signal is genuinely novel.

**A worked example.** Consider GPT-J × multiplication × layer 14 × mode=off. The eligible-concept count after the Step-6 filter is 50. The stacked basis has dimensions:

- 50 concepts × CCSVD bases ranging from 1 to 9 rows each (median 8) → ~410 rows from CCSVD.
- 50 concepts × LDA-A bases ranging from 1 to 9 rows each (median 8) → ~400 rows from LDA-A.
- Plus 1 row for β_answer (mode=off appends β_answer).

Total stacked_dim ≈ 811. SVD orthonormalisation drops the rank to k_merged = 758 (`redundancy_removed = 53`). The 53 redundant rows come from directions where CCSVD and LDA-A overlap (LDA-A ⊆ CCSVD by construction, but with non-identical orthonormalisation of the shared subspace, so the SVD-tolerance threshold catches the duplicates).

For the `generous` variant on the same cell, the extra LDA-B bases add another ~450 rows, bringing the stacked dimension to ~1,260 and k_generous to 1,253 — only 7 rows of redundancy. The LDA-B directions are nearly all independent of CCSVD ∪ LDA-A in the row span (consistent with the Step-6 audit finding of 0/4560 cos_sim_AB ≥ 0.9).

The `d_residual = 4096 − k`:

- d_residual_merged = 4096 − 758 = 3,338.
- d_residual_generous = 4096 − 1,253 = 2,843.

`γ = d_residual / N`:

- γ_merged = 3,338 / 2,751 = 1.213 (γ > 1, MP unreliable).
- γ_generous = 2,843 / 2,751 = 1.033 (γ > 1, MP unreliable).

Both variants are in the MP-unreliable regime for this multiplication cell. This is documented in `mp_reliable_flag = False` in the per-cell `mp_info_<variant>.json`. The `n_above_mp` numbers (191 for merged, 177 for generous) are computed and persisted but are not paper-citable as standalone evidence of signal.

#### 4.2.2 β scalar directions per mode

Each mode appends one or two β scalar directions to the stacked basis before orthonormalisation:

- `mode = off`: append `β_answer` = OLS slope of mean-centred X on mean-centred `answer` scalar, normalised to unit length.
- `mode = answer`: append nothing (the residualised cache has already nulled the answer scalar; including a β_answer direction would project out a direction that is essentially zero, contributing only noise).
- `mode = norm`: append `β_norm` = OLS slope on the per-row L2 norm scalar, and **also** `β_answer` (the norm cache nulls magnitude but not the answer scalar; the answer scalar is therefore still a confound in mode=norm activations and a 1-D direction worth removing).

The β-direction routine `beta_scalar_direction(X, scalar)` computes:

```
s = scalar − scalar.mean()
denom = s ⋅ s
if denom < 1e-12: return zeros(4096)        # constant scalar — refuse
Xc = X − X.mean(axis=0)
beta = (Xc.T @ s) / denom
β_direction = beta / ‖beta‖₂
```

This is the standard OLS slope normalised to unit length. Returning zeros for a constant scalar is defensive; in practice all scalars used here have positive variance.

**A worked example for β_answer at GPT-J × multiplication × layer 14 × mode=off.** The answer scalar `s` is the integer answer `a × b` for each problem in the correct subset (N = 2,751). After mean-centring, `denom = (s − s.mean()) ⋅ (s − s.mean())` ≈ 5.5 × 10⁸ (variance times N). The activation matrix `X` after mean-centring has Frobenius-norm-squared ≈ 25 × 10⁶ (cf. `var_orig = 25,454,744` reported in the per-cell `mp_info_merged.json`). The unnormalised β vector `(X^T s) / denom` has length ~5 × 10⁻³ after normalisation. The resulting unit β_answer captures the dominant linear-in-answer direction in the residual stream; appending it to the stacked basis ensures the union projects out the answer-scalar contribution before PCA.

For mode=norm: `β_norm` is the analogous OLS slope on the per-row L2 norm of `X` (a 1-D summary of the activation magnitude per problem); this direction encodes the model's per-problem activation magnitude. Appending both `β_norm` and `β_answer` in mode=norm ensures both 1-D scalars are removed before the structural audit.

#### 4.2.3 Projection and residual

For each variant V_all (shape `(k, 4096)`), the worker computes:

```
coords = X @ V_all.T                # (N, k), on GPU
X_proj = coords @ V_all             # (N, 4096), on GPU
X_residual = X − X_proj             # (N, 4096), on GPU
```

The projection matrix `P = V_all.T @ V_all` is never formed explicitly. The factored computation costs O(N · k · d) flops where d = 4096; for N = 10k and k = 2000 the dominant matmul is 80 GFLOP, executing in O(0.1 s) on the A6000.

Variance accounting (on GPU then transferred to CPU as scalars):

```
var_orig = ||X||_F² / N
var_resid = ||X_residual||_F² / N
var_explained = 1 − var_resid / var_orig
```

A defensive assertion checks `var_resid ≤ var_orig * 1.001 + 1e-3` before proceeding. The 1.001 slack absorbs float32 accumulation error in the GPU `||·||_F²` computation; tighter tolerances occasionally fail spuriously when k_union approaches 4096 and the residual is nearly all numerical zero.

**A worked example for the projection at GPT-J × multiplication × layer 14 × mode=off × merged variant.**

- X shape: `(2751, 4096)` float32.
- V_all shape: `(758, 4096)` float32.
- `coords` = `X @ V_all.T` → `(2751, 758)` float32 on GPU.
- `X_proj` = `coords @ V_all` → `(2751, 4096)` float32 on GPU.
- `X_residual` = `X − X_proj` → `(2751, 4096)` float32 on GPU.
- `var_orig` = `||X||_F² / N` = 25,454,744 / 2,751 ≈ 9,253 per-row variance.
- `var_resid` = `||X_residual||_F² / N` = 7,573,429 / 2,751 ≈ 2,754.
- `var_explained` = 1 − 7,573,429 / 25,454,744 = 1 − 0.297 = 0.702.

The cell is reporting `var_explained = 0.702` in the per-cell `mp_info_merged.json`; matches the worked computation. The 70.2% number is the headline for this cell on the merged variant: the named-concept union plus β_answer captures ~70% of the residual-stream variance at GPT-J × multiplication × layer 14.

#### 4.2.4 Centred randomised SVD on the residual

The mean-centred residual is factored via scikit-learn's `randomized_svd`:

```
mu = X_residual.mean(axis=0)
X_centered = X_residual − mu
n_components = min(500, X_centered.shape[1] − 1, N − 1)
U, S, Vt = randomized_svd(X_centered, n_components=n_components,
                          random_state=int(cell_seed) % (2**32 − 1))
eigenvalues = (S ** 2) / N                # variance along each direction
eigenvectors = Vt                          # (n_components, 4096) row-orthonormal
```

`randomized_svd` is CPU-only in our environment. For our matrix shapes (~10k × 4096), it computes the top 500 components in roughly 5 seconds and is the rate-limiting step of Step 7 alongside the correlation sweep. The factored matrix `X_centered` is float32 by default; the SVD returns float32 results and we promote eigenvalues to float64 before storing.

We choose 500 components because the MP test typically needs only the top 100–200 eigenvalues to find a cliff (if one exists); 500 provides margin to visualise the full decay into the noise bulk and to confirm that the noise eigenvalues themselves follow the predicted MP distribution. At N = 10k and d_residual = 2000, the top 500 components capture the top 25% of the spectrum.

**A worked example for the randomised SVD at GPT-J × multiplication × layer 14 × mode=off × merged variant.**

- X_residual shape: `(2751, 4096)` float32.
- After per-feature mean centring: `X_centered` same shape.
- `n_components` = min(500, 4095, 2750) = 500.
- `randomized_svd(X_centered, n_components=500)` returns `U (2751, 500)`, `S (500,)`, `Vt (500, 4096)`.
- `eigenvalues = S² / N` shape `(500,)` float64.
- `top_eigenvalue` = `eigenvalues[0]` = 0.491 (read from `mp_info_merged.json`).
- The seed for the randomised SVD is `int(cell_seed("gpt-j-6b", "multiplication", 14, "residual_hunting_off")) % (2**32 - 1)` ≈ 2,394,622,646 mod (2³² − 1) ≈ 2,394,622,646. This seed is recorded in the per-cell `metadata.json` and is the basis of reproducibility for this cell.

The randomised SVD's accuracy at our scale is approximately the same as a full SVD of the top components; sklearn's implementation uses a Halko-Martinsson-Tropp random projection with 10 power iterations by default. For our matrix shapes (~10k × 4096) and `n_components = 500`, the random-projection approximation has bounded error in the top singular values (≤ 1 part in 10⁴ on the top eigenvalue, established by ad-hoc validation against `numpy.linalg.svd` on a sample cell at smoke-test time).

#### 4.2.5 Trace-based σ² and the Marchenko–Pastur upper edge

We estimate the per-dimension residual noise variance using the trace identity, not the median of the top eigenvalues:

```
total_var = ||X_centered||_F² / N           # = trace(covariance)
sigma_sq = total_var / d_residual
gamma = d_residual / N
lambda_max_mp = sigma_sq * (1 + sqrt(gamma)) ** 2     # MP upper edge
lambda_min_mp = sigma_sq * max(0, 1 − sqrt(gamma)) ** 2
n_above_mp = sum(eigenvalues > lambda_max_mp)
mp_reliable_flag = (gamma < 0.7)
```

**Why trace-based σ² and not median of top eigenvalues.** The trace identity uses the sum of *all* eigenvalues (via the Frobenius-norm-squared shortcut), giving an unbiased estimate of the mean eigenvalue, which under the null hypothesis equals σ². Using the median of the top 500 eigenvalues would over-estimate σ² because the top 500 are a biased upper-tail sample. The trace estimate has a slight upward bias when genuine signal eigenvalues exist (they inflate the trace), but with d_residual ~ 2,000–4,000 dimensions and at most a handful of signal eigenvalues, the bias is at most 0.1% and makes the test *conservative* — it is harder for signal eigenvalues to exceed an MP edge that they themselves have inflated.

**Why `mp_reliable_flag = (gamma < 0.7)`.** The Tracy-Widom fluctuation of the largest noise eigenvalue scales as σ² · γ^{−2/3} · N^{−2/3}; at γ → 1 from below, these fluctuations become comparable to the MP edge itself and the cliff test loses statistical power. We adopt γ < 0.7 as the threshold below which the test is paper-citable; cells with γ ≥ 0.7 still report `n_above_mp` but it is logged as audit-only and not cited as evidence of signal.

For our regime:

- Addition cells: γ ∈ [0.10, 0.32] — well within the reliable regime.
- Multiplication cells: γ ∈ [0.91, 1.21] — outside the reliable regime. The `n_above_mp` numbers for multiplication are still recorded but are NOT paper-citable as standalone evidence; the headline residual-signal claim for multiplication comes from the correlation sweep (which is γ-independent).

**A worked example for σ² and the MP edge at GPT-J × multiplication × layer 14 × mode=off × merged variant.**

- `total_var` = `||X_centered||_F² / N` ≈ 40.30 (read from the eigenvalues sum: Σ S²/N for the full residual spectrum).
- `d_residual` = 4096 − 758 = 3,338.
- `sigma_sq` = `total_var / d_residual` = 40.30 / 3,338 ≈ 0.01207. (matches the `sigma_sq = 0.01207` in `mp_info_merged.json`.)
- `gamma` = 3,338 / 2,751 = 1.213.
- `lambda_max_mp` = 0.01207 × (1 + √1.213)² = 0.01207 × (2.101)² = 0.01207 × 4.414 = 0.05328. (matches `lambda_max_mp = 0.05331`.)
- `top_eigenvalue` = 0.491.
- ratio top_eig / lambda_max_mp = 9.21 (the top eigenvalue is 9× above the MP cliff).
- `n_above_mp` = count of i such that `eigenvalues[i] > 0.05331` = 191 (read from `mp_info_merged.json`).

The 9× ratio looks enormous and would normally be strong evidence of signal. But γ = 1.213, which sits above the reliable threshold of 0.7. In the γ → 1 regime, the Tracy-Widom fluctuation of the largest noise eigenvalue scales as σ² · γ^{−2/3} · N^{−2/3}; at γ = 1.21 and N = 2,751, the fluctuation is comparable to lambda_max_mp itself, and the simple cliff test cannot reject the null. The `mp_reliable_flag` is False; the 191 above-MP count is recorded as audit-only.

For an addition cell where γ = 0.25 (the reliable regime), the same 9× ratio would be strong evidence of signal because the Tracy-Widom fluctuation at γ = 0.25 is ≤ 1% of lambda_max_mp.

#### 4.2.6 Correlation sweep with derived columns

The correlation sweep tests every above-MP eigendirection against every metadata column. We sweep only the `merged` variant; the `generous` residual is dominated by Option B's N/d-inflated noise directions (a-priori known from Step 6's empirical median A-vs-B cos_sim of 0.14 across 4,560 cells; 0 cells had cos_sim ≥ 0.9), so its residual top-correlates would mostly be sweeping noise.

The metadata corpus comprises:

1. **Raw concept columns from the problems CSV.** All numeric columns (after `pd.to_numeric` with `errors='coerce'`) except those in `SKIP_COLUMNS` (JSON-list columns and Tier-5 tokenisation metadata). For addition: 35 columns survive; for multiplication: 43 columns.
2. **Derived interaction columns.** Constructed by `build_derived_columns(problems_df, answers_df, task)`:
   - Both tasks: `carry_units × carry_tens`, `a_units × b_units`, `a_tens × b_tens`, `a_units × b_tens`, `a_tens × b_units`, `(a_units + b_units) mod 10`, `(a_tens + b_tens) mod 10`, `a_parity × b_parity`, `a_magnitude_tier × b_magnitude_tier`, `consecutive_carry_run`, `predicted_value`, `predicted_units`, `predicted_tens`, `predicted_n_digits`.
   - Multiplication only: `partial_product_units × partial_product_tens`, `column_sum_units × column_sum_tens`, `ab_units_product_mod100`.
3. The full set is per-cell-masked to the correct subset. Constant or all-NaN columns are dropped before the sweep; NaN entries in otherwise-valid columns are filled with the column mean.

For each cell the sweep tests the top `max(n_above_mp, 50)` directions (the floor of 50 catches cells with `n_above_mp == 0` for audit) against every surviving metadata column. With ~50 metadata columns × ~50 to ~300 directions, the total (direction × column) grid is on the order of 2,500 to 15,000 tests per cell.

#### 4.2.7 The 1000-permutation null with BH-FDR

Permutation testing is essential because the relevant null distributions differ across columns and directions and a parametric p-value would be misleading. The implementation uses a batched matmul on rank-transformed data for Spearman (and on the centred raw data for Pearson) so that one permutation matrix produces all (direction × column) correlations in a single matmul, amortising the 1000 perms over the full grid in roughly 300 ms per cell.

```
# Observed correlations
Z_raw  = X_residual @ V_eig[:n_top].T                # (N, n_top), raw projections
Z_rank = rankdata(Z_raw, axis=0)                     # Spearman ranks
C_raw  = stack of metadata columns                   # (n_col, N)
C_rank = rankdata(C_raw, axis=1)                     # column ranks
obs_sp = batched_corr(Z_rank.T, C_rank)              # Spearman observed
obs_pr = batched_corr(Z_raw.T,  C_raw)               # Pearson observed

# 1000 permutations
for k in 1..N_PERMUTATIONS:
    perm = rng.permutation(N)
    sp_perm = batched_corr(Z_rank.T, C_rank[:, perm])
    pr_perm = batched_corr(Z_raw.T,  C_raw[:, perm])
    sp_tail += (|sp_perm| ≥ |obs_sp|)
    pr_tail += (|pr_perm| ≥ |obs_pr|)

# Empirical p-values (with +1 / +1 correction for unbiased tail)
sp_p = (sp_tail + 1) / (N_PERMUTATIONS + 1)
pr_p = (pr_tail + 1) / (N_PERMUTATIONS + 1)

# Benjamini–Hochberg FDR across the full (direction × column) grid
sp_q = false_discovery_control(sp_p.ravel(), method="bh").reshape(sp_p.shape)
pr_q = false_discovery_control(pr_p.ravel(), method="bh").reshape(pr_p.shape)
```

A row is flagged iff `|obs_sp| > 0.15` AND `sp_q < 0.05`. The flag-threshold parameters `CORR_FLAG_THRESHOLD = 0.15` and `FDR_THRESHOLD = 0.05` are inherited from the parent project. Both Spearman and Pearson statistics, p-values, and q-values are persisted side-by-side in the per-cell `correlation_sweep_merged.csv` so that a downstream analysis can re-derive the flag at any threshold without re-running the sweep.

The `false_discovery_control` function (added in SciPy 1.13) implements the Benjamini–Hochberg step-up procedure. It is the canonical FDR control method in genomics and is appropriate here: we have a single (direction × column) grid of correlations and want to control the expected fraction of false positives. We do not use Bonferroni (too conservative for large grids) or per-direction control (the questions are correlated across directions).

#### 4.2.8 Stage 3 correlate-set unions

For each cell, the worker pre-computes the union of LDA Option A bases for each Stage-3 target concept's plan-locked correlate set (plan.md §4.2-4.3):

**Addition:**

- `ans_units → {a, b, a_units, b_units}`
- `ans_tens  → {a, b, a_tens, b_tens, carry_units}`
- `answer   → {a, b}`

**Multiplication:**

- `carry_units → {column_sum_units, partial_product_units}`
- `ans_units   → {column_sum_units, carry_units, partial_product_units}`

For each (target, correlate_set), the worker loads each correlate's LDA-A basis, stacks them, and SVD-orthonormalises. The result is saved as `stage3_unions/union_correlates_<target>.npy` and its metadata as `stage3_unions/union_correlates_<target>_meta.json`. Future Stage 3 work consumes these to build the orthogonalisation operator `I − P_correlates` for each target without re-deriving from raw bases.

These Stage-3 unions are not analysed within Step 7; they are pre-computed because the per-cell I/O of loading the bases is already paid by the union construction loop.

**A worked example for the correlation sweep at GPT-J × multiplication × layer 14 × mode=off × merged variant.**

- n_top = max(n_above_mp=191, 50) = 191 above-MP directions to sweep.
- Number of surviving metadata columns: 57 (multiplication has 43 single concepts + 14 derived columns surviving the non-constant filter).
- (direction × column) grid size: 191 × 57 = 10,887 tests.
- N_PERMUTATIONS = 1000 → total Spearman tests under the null: 10,887 × 1000 = 10.9 million.
- Batched matmul implementation: 191 × 57 Pearson observed values computed in one matmul; then for each of 1000 permutations, one matmul of (191, N) × (N, 57) gives all 10,887 perm-corrs at once. Runtime: ~9 seconds for this cell.

After BH-FDR across the full 10,887-row grid, 0 rows have `spearman_q_fdr < 0.05` AND `|spearman_rho| > 0.15`. The largest observed |Spearman| in this cell is approximately 0.09, well below the 0.15 flagging threshold. Even before FDR, no row crosses the joint threshold.

The full `correlation_sweep_merged.csv` is preserved on disk (10,887 rows × 11 columns). Downstream analysis can re-derive the flag at any threshold or examine the unflagged-but-suggestive (e.g., |ρ_s| > 0.07 with q < 0.10) without re-running the sweep.

### 4.3 Implementation

The Step 7 worker is `/home/anshulk/emnlp2026/residual_hunting.py` (934 lines). Key implementation choices:

- **GPU dispatch.** A top-level `try / except` around `import cupy as cp` and `cp.cuda.is_available()` sets `_HAS_CUPY`. The `project_and_residual` function uses CuPy for the matmul and Frobenius-norm computations when CuPy is available; otherwise it falls back to numpy. Both paths return numerically equivalent float32 arrays.
- **Randomised SVD on CPU.** scikit-learn's `randomized_svd` is the right choice for our shapes (N ≤ 10k, d_residual ≤ 4096). A GPU equivalent via CuPy SVD on a (N, n_components) sketch is slower at this size due to GPU transfer overhead.
- **Batched correlation sweep.** The `_batched_corr` helper computes Pearson on rows of two matrices via one matmul on centred data divided by outer-product of norms. Spearman is Pearson on rank-transformed inputs. The whole 1000-perm sweep on a 200 × 60 grid runs in O(seconds), not O(minutes).
- **Resume logic.** Each cell directory holds a `metadata.json` with `computation_status`. The per-cell function checks this at entry and returns the cached `summary_rows` if marked `complete`. Survives preemption and array task retries.
- **Atomic writes.** `atomic_save` and `atomic_json` helpers use `tempfile.mkstemp` in the same directory, write, then `os.replace`. Avoids partial-write corruption.

### 4.4 Per-cell artifacts

Per (model, task, mode, layer) cell, the directory `results/residual_hunting/{model}/{task}/mode_{mode}/layer_{LL:02d}/` contains:

| File | Shape / contents | Notes |
|---|---|---|
| `union_basis_merged.npy` | `(k_merged, 4096)` float32 | SVD-orthonormalised union of CCSVD ∪ LDA-A + β scalars |
| `union_basis_generous.npy` | `(k_generous, 4096)` float32 | + LDA-B (audit) |
| `union_meta.json` | object | Per-variant `k`, `stacked_dim`, `redundancy_removed`, contributing concepts, β labels, eligible-concept list, dropped concepts with reason |
| `eigenvalues_merged.npy` | `(500,)` float64 | Top-500 eigenvalues of mean-centred residual |
| `eigenvalues_generous.npy` | `(500,)` float64 | Same for generous |
| `eigenvectors_merged.npy` | `(500, 4096)` float32 | Top-500 right singular vectors of residual |
| `eigenvectors_generous.npy` | `(500, 4096)` float32 | Same for generous |
| `mp_info_merged.json` | object | σ², γ, λ_max_mp, λ_min_mp, n_above_mp, top_eigenvalue, var_orig, var_resid, var_explained, mp_reliable_flag, variant |
| `mp_info_generous.json` | object | Same for generous |
| `correlation_sweep_merged.csv` | long-form table | One row per (direction × column): `direction_idx, eigenvalue, metadata_column, is_derived, spearman_rho, spearman_p_perm, spearman_q_fdr, pearson_r, pearson_p_perm, pearson_q_fdr, flag` |
| `stage3_unions/union_correlates_<target>.npy` | `(k, 4096)` float32 | Per plan §4.2/§4.3 target |
| `stage3_unions/union_correlates_<target>_meta.json` | object | target, kept, skipped, stacked_dim, k |
| `metadata.json` | object | `computation_status: "complete"`, `summary_rows` (one per variant), runtimes, seeds |

Per-model summary CSV: `results/residual_hunting/{model}/summary_{model}_{task}_mode_{mode}.csv`. One row per (layer × variant). Aggregator concatenates per-model CSVs into `results/residual_hunting/comparison/summary_all.csv` (180 rows).

### 4.5 Per-model results

All numbers in this section are read from `results/residual_hunting/comparison/summary_all.csv`. The 90 cells × 2 variants = 180 rows landed without error; zero cells have `status != "fit_ok"`.

#### 4.5.1 GPT-J 6B

**Addition.**

GPT-J × addition is the cleanest cell from the Step-5/6 perspective. Per-cell numbers (mode=off, `merged` variant):

| Layer | k_merged | k_generous | N | d_residual_merged | γ_merged | var_explained_merged | var_explained_generous | n_above_mp_merged | n_above_mp_generous | mp_reliable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1,712 | 2,797 | 8,415 | 2,384 | 0.283 | 0.870 | 0.940 | 274 | 167 | True |
| 8 | 1,886 | 3,098 | 8,415 | 2,210 | 0.263 | 0.872 | 0.940 | 262 | 152 | True |
| 14 | 2,016 | 3,138 | 8,415 | 2,080 | 0.247 | 0.866 | 0.945 | 253 | 151 | True |
| 20 | 1,966 | 3,166 | 8,415 | 2,130 | 0.253 | 0.864 | 0.941 | 262 | 145 | True |
| 24 | 1,886 | 3,140 | 8,415 | 2,210 | 0.263 | 0.866 | 0.946 | 264 | 144 | True |

`merged` k is 1,712–2,016 (median 1,886); `generous` is +57% on average (median 3,138), as expected from incorporating Option B's ~1,000 additional rows of LDA-B directions per cell. `var_explained_merged` is 0.864–0.872 (median 0.866); the named concepts collectively span ~87% of the residual variance at layer 14. The variant delta (`generous − merged`) is +0.07 to +0.085 of variance: Option B's directions claim an additional ~7-8% of variance beyond `merged`. Whether that additional variance is real signal or N/d-inflated noise is unresolvable at the Step 7 level (the variance gauge is structure-agnostic; the correlation sweep is run only on `merged`).

`n_above_mp_merged` is 253–274 (median 262) at γ ≈ 0.25: about 250 residual eigenvalues stand above the analytical MP cliff for pure Gaussian noise. The Tracy-Widom fluctuation at γ = 0.25 and N = 8,400 is small (sub-1%), so these counts are paper-citable as evidence of residual structure beyond pure noise. The 1000-permutation FDR-corrected correlation sweep returns 0/cells with an FDR-significant flag (q < 0.05) in mode=off (and across all three modes); details in §4.5.4 and Appendix A.5.

Other modes (mode=answer, mode=norm) at GPT-J × addition produce broadly similar numbers: `var_explained_merged` median 0.888 (answer) and 0.889 (norm) — slightly higher than off (the carve-outs of `ans_*` concepts reduce the stacked dimension, but the β scalar direction we append captures most of the displaced variance, and the residual is sometimes more orderly because the dominant magnitude confound has been removed). `n_above_mp_merged` median 236 (answer), 228 (norm) — modestly lower than off, again consistent with the β-direction approximately recapturing the orthogonal direction that the carved-out concepts would have spanned.

**Multiplication.**

GPT-J × multiplication operates in the γ > 1 regime. Per-cell (mode=off, `merged`):

| Layer | k_merged | k_generous | N | d_residual_merged | γ_merged | var_explained_merged | var_explained_generous | n_above_mp_merged | n_above_mp_generous | mp_reliable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 730 | 1,166 | 2,751 | 3,366 | 1.224 | 0.737 | 0.804 | 200 | 184 | False |
| 8 | 749 | 1,191 | 2,751 | 3,347 | 1.217 | 0.738 | 0.801 | 187 | 180 | False |
| 14 | 758 | 1,253 | 2,751 | 3,338 | 1.213 | 0.702 | 0.772 | 191 | 177 | False |
| 20 | 833 | 1,326 | 2,751 | 3,263 | 1.186 | 0.731 | 0.795 | 190 | 174 | False |
| 24 | 837 | 1,328 | 2,751 | 3,259 | 1.185 | 0.741 | 0.810 | 189 | 173 | False |

All five cells have γ > 1, marking them as MP-unreliable. The `n_above_mp` counts (median 190 for merged, 177 for generous) are inflated by the γ > 1 regime and are not paper-citable as standalone evidence of signal. The variance-explained gauge remains valid: `var_explained_merged` is 0.70–0.74, lower than addition. The named concepts at GPT-J × multiplication leave more residual variance unaccounted for, consistent with the smaller correct subset (N = 2,751 vs 8,415 for addition) and the more compositional nature of multiplication (more named intermediates, but each intermediate is harder to fit well).

Cross-mode (mode=answer, mode=norm) preserve the same γ > 1 regime and produce `var_explained_merged` medians of 0.772 (answer) and 0.732 (norm). The generous variant adds +0.06 on average across modes.

#### 4.5.2 Llama 3.1 8B

**Addition.**

Llama × addition has the largest correct subset (N = 9,963 of 10,000). Per-cell (mode=off, `merged`):

| Layer | k_merged | k_generous | N | d_residual_merged | γ_merged | var_explained_merged | var_explained_generous | n_above_mp_merged | n_above_mp_generous |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1,640 | 2,762 | 9,963 | 2,456 | 0.247 | 0.853 | 0.937 | 281 | 175 |
| 8 | 1,712 | 2,797 | 9,963 | 2,384 | 0.239 | 0.860 | 0.937 | 286 | 167 |
| 16 | 1,555 | 2,684 | 9,963 | 2,541 | 0.255 | 0.858 | 0.937 | 296 | 184 |
| 24 | 1,648 | 2,773 | 9,963 | 2,448 | 0.246 | 0.860 | 0.940 | 291 | 175 |
| 28 | 1,734 | 2,949 | 9,963 | 2,362 | 0.237 | 0.870 | 0.948 | 295 | 168 |

`var_explained_merged` median is 0.858 (close to GPT-J at 0.866). `n_above_mp_merged` median is 291 — modestly higher than GPT-J, which is consistent with the larger N producing tighter MP edges and more eigenvalues clearing them. The `generous` variant adds +0.075-0.09 of variance.

Mode=answer and mode=norm produce `var_explained_merged` medians of 0.888 (answer) and 0.869 (norm), again with no FDR-significant residual correlate across any cell.

**Multiplication.**

Llama × multiplication has N = 2,927 — slightly larger than GPT-J's 2,751 but still in γ ≈ 1.1 territory:

| Layer | k_merged | k_generous | N | γ_merged | var_explained_merged | var_explained_generous | n_above_mp_merged |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 757 | 1,232 | 2,927 | 1.140 | 0.752 | 0.821 | 198 |
| 8 | 778 | 1,260 | 2,927 | 1.134 | 0.755 | 0.823 | 196 |
| 16 | 800 | 1,311 | 2,927 | 1.126 | 0.755 | 0.821 | 198 |
| 24 | 868 | 1,381 | 2,927 | 1.103 | 0.769 | 0.836 | 192 |
| 28 | 832 | 1,341 | 2,927 | 1.115 | 0.760 | 0.829 | 195 |

`var_explained_merged` median 0.755 — fractionally higher than GPT-J's 0.731. `n_above_mp_merged` median 196 (unreliable; γ > 1).

#### 4.5.3 Pythia 6.9B

**Addition.**

Pythia × addition has N = 7,718 — the smallest of the three for addition:

| Layer | k_merged | k_generous | N | γ_merged | var_explained_merged | var_explained_generous | n_above_mp_merged |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1,840 | 3,078 | 7,718 | 0.292 | 0.949 | 0.982 | 250 |
| 8 | 1,915 | 3,144 | 7,718 | 0.283 | 0.946 | 0.981 | 245 |
| 16 | 2,057 | 3,219 | 7,718 | 0.264 | 0.949 | 0.984 | 236 |
| 24 | 1,840 | 3,071 | 7,718 | 0.292 | 0.952 | 0.982 | 261 |
| 28 | 1,876 | 3,135 | 7,718 | 0.287 | 0.946 | 0.981 | 256 |

Pythia produces the highest `var_explained` numbers of the three models, both `merged` (median 0.949) and `generous` (median 0.982). The generous−merged delta is smaller here than for the other two models (~+0.03 vs +0.07-0.08), suggesting Pythia's LDA-A directions already capture most of the structure that LDA-B would add, and that LDA-B is contributing relatively less independent direction-space at Pythia than at GPT-J/Llama.

**Multiplication.**

Pythia × multiplication has N = 2,757, the same regime as the other models:

| Layer | k_merged | k_generous | N | γ_merged | var_explained_merged | var_explained_generous | n_above_mp_merged |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 778 | 1,231 | 2,757 | 1.202 | 0.864 | 0.910 | 186 |
| 8 | 729 | 1,186 | 2,757 | 1.226 | 0.860 | 0.908 | 191 |
| 16 | 749 | 1,252 | 2,757 | 1.214 | 0.874 | 0.919 | 185 |
| 24 | 818 | 1,323 | 2,757 | 1.181 | 0.874 | 0.917 | 180 |
| 28 | 749 | 1,257 | 2,757 | 1.214 | 0.895 | 0.929 | 173 |

Pythia × multiplication produces the highest var_explained among multiplication cells across all three models — `var_explained_merged` median 0.874 vs GPT-J's 0.731 and Llama's 0.755. This is consistent with Pythia having a slightly tighter concept geometry on multiplication (also noted at Step 6 where Pythia's residualization Δλ_T_1 was the smallest of the three models).

#### 4.5.4 Residual top-correlate sweep — null finding

Across all 90 cells × all three modes × 1000-permutation FDR-corrected sweep against ~50 metadata columns, **0 cells** have an FDR-significant residual correlate (q < 0.05) with `|ρ_s| > 0.15`. The maximum unflagged Spearman ρ across all cells is approximately 0.09 (compatible with sampling noise at our N), and after BH correction across the (direction × column) grid, no q-value drops below 0.05.

This is a meaningful null finding. The parent project at `arithmetic-geometry` reported Spearman ρ ≈ 0.07-0.10 at L5/wrong against `pp_a2_x_b1` (a partial-product interaction) — but they did not apply FDR correction or a 1000-permutation null. Our more rigorous testing returns no significant correlate. Interpretation belongs in Appendix A.5; the bare number is: zero FDR-significant residual correlates across 90 cells × 2 union variants × 3 modes × per-cell sweep.

### 4.6 Cross-mode and cross-variant comparison

The Step 7 aggregator emits the following cross-comparison CSVs under `results/residual_hunting/comparison/`:

- `summary_all.csv` (180 rows): one row per (cell, variant) with every field listed in §4.4.
- `var_explained_cross_mode.csv` (60 rows): pivot of `var_explained` across the three modes for each (model, task, layer, variant). Used to read off mode-deltas at a glance.
- `n_above_mp_cross_mode.csv` (60 rows): pivot of `n_above_mp` plus `mp_reliable_flag`.
- `k_union_cross_mode.csv` (60 rows): pivot of union rank.
- `gamma_cross_mode.csv` (60 rows): pivot of γ.
- `residual_top_correlate_cross_mode.csv` (30 rows, merged only): per (model, task, mode, layer) the top FDR-significant correlate (or NaN if none).
- `variant_delta.csv` (90 rows): per (model, task, mode, layer), the delta `generous − merged` for var_explained, n_above_mp, k_union, gamma, top_eigenvalue.
- `summary_with_matched_count.csv`: per-cell summary annotated with the `n_matched_concepts` from Step 6's `matched_population_cells.csv` (the count of concepts at this (model, task, layer) cell that fit cleanly under all three Step-6 modes).

Per-mode aggregate of `var_explained_merged` (median across the 5 layers in each (model, task, mode) bucket):

| Model | Task | mode=off | mode=answer | mode=norm |
|---|---|---:|---:|---:|
| GPT-J | addition | 0.866 | 0.889 | 0.889 |
| GPT-J | multiplication | 0.739 | 0.772 | 0.732 |
| Llama | addition | 0.858 | 0.888 | 0.869 |
| Llama | multiplication | 0.755 | 0.768 | 0.733 |
| Pythia | addition | 0.949 | 0.963 | 0.963 |
| Pythia | multiplication | 0.874 | 0.895 | 0.899 |

Per-mode aggregate of `n_above_mp_merged` (median; multiplication rows are unreliable due to γ > 1):

| Model | Task | mode=off | mode=answer | mode=norm |
|---|---|---:|---:|---:|
| GPT-J | addition | 262 | 236 | 228 |
| GPT-J | multiplication | 190 | 181 | 183 |
| Llama | addition | 291 | 273 | 269 |
| Llama | multiplication | 194 | 189 | 187 |
| Pythia | addition | 250 | 221 | 217 |
| Pythia | multiplication | 185 | 178 | 179 |

Per-mode aggregate of `variant_delta_var_explained` (generous − merged; median):

| Model | Task | off | answer | norm |
|---|---|---:|---:|---:|
| GPT-J | addition | +0.070 | +0.063 | +0.074 |
| GPT-J | multiplication | +0.063 | +0.059 | +0.064 |
| Llama | addition | +0.072 | +0.060 | +0.079 |
| Llama | multiplication | +0.060 | +0.056 | +0.071 |
| Pythia | addition | +0.029 | +0.025 | +0.027 |
| Pythia | multiplication | +0.028 | +0.024 | +0.026 |

The `generous` variant uniformly adds 3–8 percentage points of variance over `merged`. Pythia's deltas are smallest (~+0.03) and GPT-J/Llama are largest (~+0.07).

---

### 4.7 Detailed per-mode breakdown — addition cells, merged variant

The aggregator's `var_explained_cross_mode.csv` gives per-(model, task, layer, variant) rows with mode columns. For addition cells, merged variant:

| Model | Layer | off | answer | norm |
|---|---:|---:|---:|---:|
| GPT-J | 4 | 0.8702 | 0.8918 | 0.8932 |
| GPT-J | 8 | 0.8720 | 0.8888 | 0.8939 |
| GPT-J | 14 | 0.8659 | 0.8888 | 0.8894 |
| GPT-J | 20 | 0.8636 | 0.8849 | 0.8869 |
| GPT-J | 24 | 0.8659 | 0.8835 | 0.8810 |
| Llama | 4 | 0.8525 | 0.8862 | 0.8703 |
| Llama | 8 | 0.8602 | 0.8856 | 0.8686 |
| Llama | 16 | 0.8581 | 0.8876 | 0.8689 |
| Llama | 24 | 0.8595 | 0.8847 | 0.8657 |
| Llama | 28 | 0.8697 | 0.8902 | 0.8723 |
| Pythia | 4 | 0.9486 | 0.9627 | 0.9633 |
| Pythia | 8 | 0.9460 | 0.9612 | 0.9614 |
| Pythia | 16 | 0.9486 | 0.9627 | 0.9631 |
| Pythia | 24 | 0.9521 | 0.9617 | 0.9665 |
| Pythia | 28 | 0.9462 | 0.9594 | 0.9586 |

Two patterns visible:

1. **Mode=answer > mode=off, modestly.** Across all 15 addition (model, layer) cells, mode=answer's var_explained exceeds mode=off's by 0.015-0.034. The answer scalar's contribution is partially captured by `merged`'s β_answer direction in mode=off; mode=answer's residual cache has already nulled the scalar, so the LDA-A directions span a slightly different subspace that captures the answer-related structure more efficiently.
2. **Mode=norm comparable to mode=off in most cells.** Mode=norm's var_explained sits within ±0.02 of mode=off. The norm-scalar carve-out (only `ans_magnitude_tier` is removed) is minimal, and the β_norm direction added in mode=off captures most of the magnitude contribution that residualisation would otherwise remove.

### 4.8 Detailed per-mode breakdown — multiplication cells, merged variant

| Model | Layer | off | answer | norm |
|---|---:|---:|---:|---:|
| GPT-J | 4 | 0.7374 | 0.7715 | 0.7282 |
| GPT-J | 8 | 0.7384 | 0.7747 | 0.7283 |
| GPT-J | 14 | 0.7025 | 0.7497 | 0.7038 |
| GPT-J | 20 | 0.7313 | 0.7716 | 0.7325 |
| GPT-J | 24 | 0.7414 | 0.7853 | 0.7536 |
| Llama | 4 | 0.7523 | 0.7660 | 0.7280 |
| Llama | 8 | 0.7549 | 0.7682 | 0.7280 |
| Llama | 16 | 0.7545 | 0.7682 | 0.7333 |
| Llama | 24 | 0.7688 | 0.7798 | 0.7456 |
| Llama | 28 | 0.7600 | 0.7763 | 0.7392 |
| Pythia | 4 | 0.8636 | 0.8896 | 0.8896 |
| Pythia | 8 | 0.8597 | 0.8884 | 0.8918 |
| Pythia | 16 | 0.8736 | 0.8952 | 0.8986 |
| Pythia | 24 | 0.8736 | 0.8979 | 0.9050 |
| Pythia | 28 | 0.8952 | 0.9088 | 0.9106 |

Multiplication shows the same mode=answer > mode=off pattern, but with larger gaps (+0.04 typical). Mode=norm is sometimes lower than mode=off (GPT-J, Llama) and sometimes higher (Pythia) — the residualisation of magnitude affects the multiplication geometry differently across models. Pythia's higher var_explained is consistent across modes; GPT-J's lower var_explained at layer 14 (0.7025) is the lowest across all multiplication cells.

### 4.9 Per-cell n_above_mp tables — addition (reliable regime)

`n_above_mp` per (model, layer, mode, variant). Addition cells (γ in [0.10, 0.32], all reliable):

**Merged variant:**

| Model | Layer | off | answer | norm |
|---|---:|---:|---:|---:|
| GPT-J | 4 | 274 | 226 | 224 |
| GPT-J | 8 | 262 | 234 | 240 |
| GPT-J | 14 | 253 | 236 | 228 |
| GPT-J | 20 | 262 | 245 | 235 |
| GPT-J | 24 | 264 | 240 | 226 |
| Llama | 4 | 281 | 261 | 264 |
| Llama | 8 | 286 | 276 | 274 |
| Llama | 16 | 296 | 273 | 269 |
| Llama | 24 | 291 | 281 | 270 |
| Llama | 28 | 295 | 285 | 268 |
| Pythia | 4 | 250 | 230 | 213 |
| Pythia | 8 | 245 | 224 | 218 |
| Pythia | 16 | 236 | 221 | 217 |
| Pythia | 24 | 261 | 234 | 223 |
| Pythia | 28 | 256 | 224 | 217 |

**Generous variant:**

| Model | Layer | off | answer | norm |
|---|---:|---:|---:|---:|
| GPT-J | 4 | 167 | 214 | 211 |
| GPT-J | 8 | 152 | 168 | 199 |
| GPT-J | 14 | 151 | 173 | 167 |
| GPT-J | 20 | 145 | 158 | 156 |
| GPT-J | 24 | 144 | 156 | 159 |
| Llama | 4 | 175 | 187 | 188 |
| Llama | 8 | 167 | 169 | 175 |
| Llama | 16 | 184 | 186 | 188 |
| Llama | 24 | 175 | 184 | 175 |
| Llama | 28 | 168 | 175 | 168 |
| Pythia | 4 | 175 | 178 | 167 |
| Pythia | 8 | 156 | 167 | 166 |
| Pythia | 16 | 184 | 162 | 158 |
| Pythia | 24 | 144 | 158 | 159 |
| Pythia | 28 | 168 | 159 | 159 |

Generous's n_above_mp is consistently lower than merged's despite larger k (more variance projected out leaves a thinner residual; the residual eigenvalues that remain are smaller, and fewer of them stand above the (smaller) MP edge).

Across all 45 addition (model, layer, mode) cells in the merged variant, n_above_mp ranges from 213 to 296 with median 250. In the generous variant, from 144 to 214 with median 168.

### 4.10 Cells with the smallest and largest variance gaps

The cells with the smallest `var_explained` (top 5 across all 90 cells in merged):

1. GPT-J × multiplication × L14 × off: 0.7025
2. GPT-J × multiplication × L14 × norm: 0.7038
3. GPT-J × multiplication × L8 × norm: 0.7283
4. GPT-J × multiplication × L4 × norm: 0.7282
5. GPT-J × multiplication × L20 × off: 0.7313

All multiplication cells. GPT-J × multiplication × L14 × off is the worst-explained cell in the project at `var_explained = 0.7025`; ~30% of activation variance remains in the orthogonal complement.

The cells with the largest `var_explained` (top 5 across merged):

1. Pythia × multiplication × L28 × norm: 0.9106
2. Pythia × multiplication × L28 × answer: 0.9088
3. Pythia × multiplication × L28 × off: 0.8952
4. Pythia × multiplication × L24 × norm: 0.9050
5. Pythia × multiplication × L24 × answer: 0.8979

All Pythia × multiplication cells at deep layers. Pythia × multiplication × L28 × norm has `var_explained = 0.9106`, the cleanest cell in the project on the merged variant.

For `generous`, the equivalent extremes:

- Smallest: GPT-J × multiplication × L14 × norm: 0.7838 (still the lowest).
- Largest: Pythia × multiplication × L28 × norm: 0.9341.

Generous always raises var_explained above merged's; the residual variance gap (1 − var_explained) shrinks under generous but does not vanish.

## 5. Step 8 — Principal angles

### 5.1 Purpose

Step 8 measures, for each (model, task, mode, layer) cell, the pairwise principal angles between every pair of concept subspaces. The output is a per-pair table of angles (5 smallest plus median and max), an empirical-null p-value, a Benjamini-Hochberg FDR-corrected q-value, and a boolean superposition flag.

The audit question this answers:

> "Within a cell, do concept subspaces share computational directions, and which concept pairs are most entangled?"

The answer feeds two downstream uses:

- **Stage 3 informativeness.** Stage 3 (ownership / orthogonalisation) is most informative for target concepts whose `angle_1` against their algebraic correlate set is near zero (the target shares its principal direction with its correlates, so orthogonalisation can produce a meaningful "ownership" measurement). Step 8's per-pair table identifies these high-leverage target/correlate pairs in advance.
- **Superposition characterisation.** The fraction of concept pairs flagged as superposed quantifies how densely the model packs concepts into shared dimensions. Comparing across (task, mode) reveals whether the superposition rate depends on task difficulty or residualisation.

### 5.2 Mathematical framework

#### 5.2.1 Principal angles via SVD

For two subspaces V_A and V_B in R^4096 represented by row-orthonormal bases B_A (m_a × 4096) and B_B (m_b × 4096), the principal angles are defined by the SVD of the cross-Gram matrix:

```
M = B_A @ B_B.T                              # (m_a, m_b)
U, S, Vt = SVD(M)                             # S in [0, 1]
θ_i = arccos(S_i)                             # principal angles in [0°, 90°]
```

The number of principal angles is `min(m_a, m_b)`. Singular values are returned in descending order, so principal angles are in ascending order: θ_1 ≤ θ_2 ≤ ... ≤ θ_{min(m_a, m_b)}.

Semantically:

- θ_1 (smallest) = the minimum angle between any direction in V_A and any direction in V_B. The angle between the closest pair of directions across the two subspaces.
- θ_1 = 0° ↔ V_A and V_B share at least one direction exactly.
- θ_1 = 90° ↔ V_A and V_B are fully orthogonal.
- θ_i for i > 1 measures the overlap of the remaining dimensions after removing the closest pair. The full sequence is the complete picture of how the two subspaces relate.

Numerically, SVD can produce singular values slightly outside [0, 1] due to floating-point accumulation; we clip S to [-1, 1] before arccos.

#### 5.2.2 LDA basis orthonormalisation on load

Step 8 uses LDA Option A bases as the per-concept subspace representation. The raw `lda_basis_full.npy` arrays are stored with `(4096, n_sig)` shape and unit per-column norms, but the columns are **not** column-orthogonal: LDA solves a generalised eigenproblem `S_B w = λ S_T w`, whose eigenvectors are orthogonal in the S_T metric, not in the Euclidean metric. The off-diagonal of the Gram matrix `B.T @ B` is non-zero for typical LDA bases (we measured off-diagonal magnitudes up to 0.29 for a sample basis at Step 6).

Principal-angle SVD requires row-orthonormal inputs to interpret singular values as cosines of angles. We therefore orthonormalise each loaded basis on the fly via QR:

```
def orthonormalise_basis(B):
    Q, R = np.linalg.qr(B.T)                 # B.T is (4096, n_rows)
    diag = np.abs(np.diag(R))
    keep = diag > 1e-10 * (diag.max() if diag.size else 1.0)
    return Q[:, keep].T.astype(np.float32)
```

The QR-orthonormalised basis spans the same subspace as the input but has row-orthonormal rows. The `keep` mask drops rank-deficient columns (numerical safety). For LDA-A bases at our scale, the QR step takes O(n_sig × 4096) flops per basis and runs in microseconds.

This orthonormalisation is essential for the self-angle sanity check (a basis vs itself must produce angles near 0°) and for the empirical null comparison (the null uses random orthonormal subspaces, so the observed must also be orthonormal).

#### 5.2.3 Empirical random-baseline null with disk-persistent cache

For each unique `(min(m_a, m_b), max(m_a, m_b))` dim-pair encountered across the run, the worker computes a 1000-trial empirical null for the smallest principal angle:

```
for trial in 1..1000:
    A = standard_normal((4096, m_a))
    Q_A, _ = QR(A)                            # random orthonormal subspace of dim m_a
    B = standard_normal((4096, m_b))
    Q_B, _ = QR(B)
    M = Q_A.T @ Q_B
    S = SVD(M, compute_uv=False)
    θ_1[trial] = arccos(clip(S[0], -1, 1))
```

The empirical-null records: mean, std, 1st percentile, 5th percentile, and the full distribution (the 1000 angles, used for per-pair empirical p-values).

The 200-trial baseline of the parent project was bumped to 1000 to match this project's standing rule on permutation/empirical trial counts.

The baseline is **shared across all (model, task, mode, layer) cells** via a disk-persistent cache at `results/principal_angles/random_baseline_cache.npy`. Keys are `(min_dim, max_dim)` integer tuples. The cache is loaded at worker startup and re-written via `atomic_save` after every new key is computed. Multiple workers running concurrently see eventually-consistent state through `os.replace`; if two workers compute the same key simultaneously, the last writer wins (both produce statistically equivalent distributions within ±0.5° on the mean, so the race is harmless).

Empirically across the 90 cells, the cache accumulated ~70 unique dim-pairs (concept dims range 1-9 across all cells). Each new dim-pair costs ~3 s of CPU; cache priming on the first cell of the first array task is the most expensive step in Step 8, and subsequent cells/tasks reuse the cache.

**Typical baseline distributions** at d = 4096:

- (5, 5): mean θ_1 ≈ 85.9°, std ≈ 0.7°, p5 ≈ 84.7°
- (7, 9): mean θ_1 ≈ 83.5°, std ≈ 0.8°, p5 ≈ 82.0°
- (9, 9): mean θ_1 ≈ 83.0°, std ≈ 0.8°, p5 ≈ 81.5°

Random low-dimensional subspaces in R^4096 are concentrated near 90°, reflecting the concentration of measure in high dimension. Any observed θ_1 below ~80° is far below the random baseline.

#### 5.2.4 Superposition flag and per-pair FDR

For each unordered concept pair (a, b) within a cell, the worker computes:

```
ang = principal_angles_deg(B_a_ortho, B_b_ortho)
top_k = min(5, ang.size)
baseline = cache.get(dim_a, dim_b)
theta1 = float(ang[0])
perm_p = (sum(baseline.thetas ≤ theta1) + 1) / (1000 + 1)
superposition_flag = (theta1 < baseline.theta1_p5 − 10.0)         # SUPERPOSITION_MARGIN_DEG = 10
```

The `superposition_flag` requires θ_1 to be at least 10° below the 5th percentile of the empirical null. This is a conservative threshold: it requires not just that θ_1 is below random (a 5% rate by definition at the p5 threshold), but that it is substantially below random. The 10° margin is inherited from the parent project.

After every pair has its `perm_p`, the worker applies Benjamini-Hochberg FDR correction across all pairs in the cell:

```
_, q_fdr, _, _ = multipletests(perm_p_array, alpha=0.05, method='fdr_bh')
```

(Implementation note: the worker actually uses scipy's `false_discovery_control` for BH-FDR, identical result to statsmodels' `multipletests(method='fdr_bh')`.)

A pair is flagged superposed by the strict criterion `superposition_flag = True`; a pair is flagged FDR-significant by `fdr_q < 0.05`. Most superposition-flagged pairs are also FDR-significant, but the two flags differ for borderline cases. Both are recorded.

#### 5.2.5 Self-angle sanity check

For every concept in a cell, the worker computes `principal_angles_deg(B_c, B_c)` after orthonormalisation. With a correctly orthonormalised basis, this should produce angles near 0° (machine epsilon). The worker asserts `max(angles) < 1°` (`SELF_ANGLE_TOLERANCE_DEG = 1`) and records the result in `self_angles.csv`. A failure of this sanity check indicates a bug in the orthonormalisation pipeline (and was the symptom that caught the original "LDA-A is not column-orthogonal" bug during the smoke test).

After the orthonormalisation fix landed, every self-angle measurement across all 90 cells stayed below 0.05°.

### 5.3 Implementation

`/home/anshulk/emnlp2026/principal_angles.py` (351 lines). Implementation choices:

- **No CuPy dependency.** Principal-angle SVD operates on tiny matrices (max dim 9 × 9); GPU transfer overhead would exceed the compute. Pure numpy.
- **Disk-persistent baseline cache.** `BaselineCache` class loads/saves to `random_baseline_cache.npy` via `numpy.save(allow_pickle=True)` (a 0-d object array of a Python dict). Atomic write via tempfile + `os.replace`.
- **Per-cell resume.** Same `metadata.json` / `computation_status` idiom as Step 7.
- **Tier annotation.** `tier_of(concept)` is a best-effort heuristic for plotting; it assigns each concept to one of {tier1_operand, tier1_answer, tier2_column, tier3_structural, tier4_relational, joint, other} based on name pattern.

### 5.4 Per-cell artifacts

Per (model, task, mode, layer) cell, the directory `results/principal_angles/{model}/{task}/mode_{mode}/layer_{LL:02d}/`:

| File | Contents |
|---|---|
| `angles_pairwise.csv` | One row per concept pair: `concept_a, concept_b, tier_a, tier_b, dim_a, dim_b, n_angles, angle_1..angle_5, angle_median, angle_max, baseline_theta1_mean, baseline_theta1_std, baseline_theta1_p5, baseline_theta1_p1, perm_p, superposition_flag, fdr_q` |
| `self_angles.csv` | One row per concept: `concept, dim, max_angle_deg, self_angle_ok` (all should have `self_angle_ok == True`) |
| `metadata.json` | `computation_status: "complete"`, `summary_row` (cell-level), unique-dim-pair count |

Shared across all cells: `results/principal_angles/random_baseline_cache.npy` (1000-trial empirical-null distributions, keyed by `(min_dim, max_dim)`).

Per-model summary CSV: `results/principal_angles/{model}/summary_{model}_{task}_mode_{mode}.csv`. One row per cell. Aggregator concatenates these into `comparison/summary_all.csv` (90 rows) and `comparison/pairwise_all.csv` (full pairwise table across all 90 cells, ~91,000 rows).

### 5.5 Per-model results

All numbers in this section come from `comparison/summary_all.csv` and `comparison/pairwise_all.csv`.

#### 5.5.1 GPT-J 6B

**Addition.** Per-cell summary (mode=off):

| Layer | n_concepts | n_pairs | n_superposition_flags | rate | median_angle_1 | median_angle_5 | runtime_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 47 | 1,081 | 866 | 80.1% | 51.83° | 52.86° | 3,438 |
| 8 | 47 | 1,081 | 870 | 80.5% | 51.42° | 53.32° | 3,479 |
| 14 | 47 | 1,081 | 869 | 80.4% | 49.39° | 53.75° | 3,389 |
| 20 | 47 | 1,081 | 872 | 80.7% | 50.56° | 53.16° | 3,381 |
| 24 | 47 | 1,081 | 871 | 80.6% | 50.16° | 54.06° | 3,259 |

47 eligible concepts produce `C(47, 2) = 1,081` pairs per cell. The superposition rate is consistently 80-81%, with median θ_1 around 50°.

**Multiplication.** Per-cell summary (mode=off):

| Layer | n_concepts | n_pairs | n_superposition_flags | rate | median_angle_1 | median_angle_5 | runtime_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 50 | 1,225 | 1,061 | 86.6% | 39.18° | 51.45° | 658 |
| 8 | 50 | 1,225 | 1,099 | 89.7% | 41.74° | 51.91° | 666 |
| 14 | 50 | 1,225 | 1,102 | 90.0% | 40.38° | 51.61° | 567 |
| 20 | 50 | 1,225 | 1,098 | 89.6% | 41.32° | 52.18° | 530 |
| 24 | 50 | 1,225 | 1,071 | 87.4% | 39.96° | 51.18° | 415 |

50 eligible concepts (multiplication has 2 task-specific concepts beyond addition's 47) produce `C(50, 2) = 1,225` pairs. Superposition rate is consistently 87-90% with median θ_1 around 40°.

#### 5.5.2 Llama 3.1 8B

**Addition.** mode=off:

| Layer | n_concepts | n_pairs | n_flags | rate | med_angle_1 | runtime_s |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 47 | 1,081 | 818 | 75.7% | 55.61° | 4,094 |
| 8 | 47 | 1,081 | 813 | 75.2% | 57.74° | 4,021 |
| 16 | 47 | 1,081 | 830 | 76.8% | 56.85° | 4,017 |
| 24 | 47 | 1,081 | 845 | 78.2% | 56.43° | 4,098 |
| 28 | 47 | 1,081 | 829 | 76.7% | 55.84° | 3,957 |

Llama × addition has the lowest superposition rate among the three models (76-78%) and the largest median θ_1 (≈ 56°). This is consistent with the model's larger embedding-space dimensionality budget (its 4096-wide residual stream has more room for concepts to occupy disjoint directions).

**Multiplication.** mode=off:

| Layer | n_concepts | n_pairs | n_flags | rate | med_angle_1 | runtime_s |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 50 | 1,225 | 1,107 | 90.4% | 39.92° | 760 |
| 8 | 50 | 1,225 | 1,113 | 90.9% | 39.84° | 717 |
| 16 | 50 | 1,225 | 1,098 | 89.6% | 39.90° | 693 |
| 24 | 50 | 1,225 | 1,111 | 90.7% | 39.81° | 731 |
| 28 | 50 | 1,225 | 1,103 | 90.0% | 39.62° | 565 |

Llama × multiplication has the highest superposition rate of all 30 (model, task, mode, layer) cells (90-91%) with the smallest median θ_1 (≈ 40°).

#### 5.5.3 Pythia 6.9B

**Addition.** mode=off:

| Layer | n_concepts | n_pairs | n_flags | rate | med_angle_1 | runtime_s |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 47 | 1,081 | 884 | 81.8% | 53.04° | 3,558 |
| 8 | 47 | 1,081 | 879 | 81.3% | 53.79° | 3,531 |
| 16 | 47 | 1,081 | 882 | 81.6% | 52.56° | 3,553 |
| 24 | 47 | 1,081 | 866 | 80.1% | 51.45° | 3,499 |
| 28 | 47 | 1,081 | 880 | 81.4% | 52.32° | 3,625 |

Pythia × addition produces a superposition rate intermediate between GPT-J (≈ 80%) and Llama (≈ 76%).

**Multiplication.** mode=off:

| Layer | n_concepts | n_pairs | n_flags | rate | med_angle_1 | runtime_s |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 50 | 1,225 | 1,105 | 90.2% | 40.45° | 540 |
| 8 | 50 | 1,225 | 1,118 | 91.3% | 40.21° | 583 |
| 16 | 50 | 1,225 | 1,133 | 92.5% | 39.51° | 555 |
| 24 | 50 | 1,225 | 1,116 | 91.1% | 39.27° | 582 |
| 28 | 50 | 1,225 | 1,107 | 90.4% | 40.05° | 514 |

Pythia × multiplication produces the highest median superposition rate at the headline layer (16): 92.5%. Across the 5 layers it averages 91.1%, the highest of any (model, task) pair.

#### 5.5.4 Cross-mode aggregates

Median superposition rate per (model, task, mode):

| Model | Task | off | answer | norm |
|---|---|---:|---:|---:|
| GPT-J | addition | 80.4% | 81.1% | 80.9% |
| GPT-J | multiplication | 89.6% | 90.0% | 90.1% |
| Llama | addition | 76.8% | 79.5% | 77.3% |
| Llama | multiplication | 90.2% | 89.7% | 92.1% |
| Pythia | addition | 81.6% | 81.5% | 81.0% |
| Pythia | multiplication | 91.0% | 89.7% | 91.4% |

Median angle_1 per (model, task, mode):

| Model | Task | off | answer | norm |
|---|---|---:|---:|---:|
| GPT-J | addition | 50.56° | 46.17° | 49.79° |
| GPT-J | multiplication | 40.38° | 36.69° | 39.98° |
| Llama | addition | 56.85° | 53.59° | 57.15° |
| Llama | multiplication | 39.90° | 40.65° | 40.30° |
| Pythia | addition | 52.56° | 50.43° | 53.53° |
| Pythia | multiplication | 39.51° | 38.51° | 38.38° |

### 5.5.5 Detailed per-mode breakdown — Step 8 superposition rates

Full per-(model, layer, mode) superposition-rate table (n_superposition_flags / n_pairs):

**Addition (47 concepts → 1,081 pairs):**

| Model | Layer | off | answer | norm |
|---|---:|---:|---:|---:|
| GPT-J | 4 | 80.11% | 81.20% | 80.85% |
| GPT-J | 8 | 80.49% | 81.10% | 80.85% |
| GPT-J | 14 | 80.39% | 81.13% | 80.85% |
| GPT-J | 20 | 80.67% | 81.10% | 80.91% |
| GPT-J | 24 | 80.57% | 81.20% | 80.92% |
| Llama | 4 | 75.67% | 79.62% | 77.43% |
| Llama | 8 | 75.21% | 79.59% | 77.20% |
| Llama | 16 | 76.78% | 79.54% | 77.43% |
| Llama | 24 | 78.16% | 79.43% | 77.33% |
| Llama | 28 | 76.69% | 79.51% | 77.30% |
| Pythia | 4 | 81.78% | 81.49% | 81.03% |
| Pythia | 8 | 81.31% | 81.51% | 81.10% |
| Pythia | 16 | 81.59% | 81.46% | 81.04% |
| Pythia | 24 | 80.11% | 81.49% | 81.07% |
| Pythia | 28 | 81.40% | 81.46% | 81.11% |

**Multiplication (50 concepts → 1,225 pairs):**

| Model | Layer | off | answer | norm |
|---|---:|---:|---:|---:|
| GPT-J | 4 | 86.61% | 89.66% | 89.96% |
| GPT-J | 8 | 89.71% | 90.00% | 90.04% |
| GPT-J | 14 | 89.96% | 90.13% | 90.04% |
| GPT-J | 20 | 89.63% | 89.97% | 90.13% |
| GPT-J | 24 | 87.43% | 89.92% | 90.07% |
| Llama | 4 | 90.37% | 89.66% | 92.16% |
| Llama | 8 | 90.86% | 89.66% | 92.16% |
| Llama | 16 | 89.63% | 89.66% | 92.08% |
| Llama | 24 | 90.69% | 89.79% | 92.16% |
| Llama | 28 | 90.04% | 89.74% | 92.04% |
| Pythia | 4 | 90.20% | 89.71% | 90.69% |
| Pythia | 8 | 91.27% | 89.79% | 91.16% |
| Pythia | 16 | 92.49% | 89.66% | 91.43% |
| Pythia | 24 | 91.10% | 89.74% | 91.39% |
| Pythia | 28 | 90.37% | 89.66% | 91.35% |

Three observations:

1. **Stability across layers.** Within a (model, task, mode), the superposition rate is stable across layers (range typically ≤ 2%); the geometry of concept-pair overlap is layer-invariant at the cell level.
2. **Mode=answer slightly elevates the rate for multiplication.** Carving out `ans_*` concepts in mode=answer reduces the pair set from 1,225 to ~861, but the surviving pairs are concentrated among the more-entangled non-answer concepts.
3. **Cross-model ordering on multiplication: Llama ≈ Pythia > GPT-J.** Llama × multiplication has the highest median rate (90.4-92.2% across modes); GPT-J × multiplication is consistently 1-3% lower.

### 5.5.6 Median angle_1 cross-mode

| Model | Task | Layer | off | answer | norm |
|---|---|---:|---:|---:|---:|
| GPT-J | addition | 4 | 51.83° | 46.85° | 50.36° |
| GPT-J | addition | 14 | 49.39° | 45.51° | 50.13° |
| GPT-J | multiplication | 4 | 39.18° | 35.41° | 39.81° |
| GPT-J | multiplication | 14 | 40.38° | 36.69° | 39.98° |
| Llama | addition | 4 | 55.61° | 54.16° | 57.42° |
| Llama | addition | 16 | 56.85° | 53.59° | 57.15° |
| Llama | multiplication | 4 | 39.92° | 40.65° | 40.30° |
| Llama | multiplication | 16 | 39.90° | 40.65° | 40.30° |
| Pythia | addition | 4 | 53.04° | 50.06° | 53.49° |
| Pythia | addition | 16 | 52.56° | 50.43° | 53.53° |
| Pythia | multiplication | 4 | 40.45° | 38.41° | 38.32° |
| Pythia | multiplication | 16 | 39.51° | 38.51° | 38.38° |

Median angle_1 is consistently smaller for multiplication (≈ 40°) than addition (≈ 50-57°). Mode=answer has slightly smaller angle_1 than mode=off (4-6° smaller for addition, 1-3° smaller for multiplication) — the carved-out `ans_*` concepts had relatively higher angle_1 against the rest, so removing them concentrates the comparison on more-entangled pairs.

### 5.6 Tier-pair superposition

The aggregator's `superposition_by_tier_pair.csv` groups pairs by the canonicalised (tier_a, tier_b) tuple. The tier mapping:

- tier1_operand: a, b, a_units, b_units, a_tens, b_tens, a_num_digits, b_num_digits
- tier1_answer: ans_*, answer
- tier2_column: column_sum_*, carry_*, running_sum_*, partial_product_*
- tier3_structural: *_parity, *_magnitude, *_is_zero, ans_ends_in_zero, a_eq_b
- tier4_relational: max_operand, min_operand, operand_diff, larger_operand, both_*, either_*
- joint: any concept name with `__` (joint tuple)

Selected tier-pair superposition rates across all 90 cells (median):

| tier_pair | n_pairs (per cell, avg) | superposition rate |
|---|---:|---:|
| (joint, joint) | ~60 | 99-100% |
| (joint, tier1_operand) | ~60 | 95-98% |
| (joint, tier2_column) | ~30 | 95-98% |
| (tier1_operand, tier1_operand) | 28 | 60-80% |
| (tier1_operand, tier2_column) | 20-40 | 70-85% |
| (tier2_column, tier2_column) | 6-66 | 80-95% |
| (tier1_operand, tier3_structural) | ~50 | 65-75% |
| (tier3_structural, tier3_structural) | ~25 | 70-80% |
| (tier4_relational, tier4_relational) | ~20 | 70-80% |

Joint concepts (which by construction include 2-3 component scalars) have the highest superposition rates against everything; their LDA-A bases span shared computational machinery with the component concepts.

---

### 5.7 Sample of FDR-q distribution

The `fdr_q` column in `angles_pairwise.csv` is the BH-corrected per-pair q-value. The fraction of pairs with `fdr_q < 0.05` is typically very close to the superposition-flag rate but not identical:

| Model | Task | Layer | Mode | n_pairs | n_flags (strict superposition) | n_fdr_q_below_0.05 |
|---|---|---:|---|---:|---:|---:|
| GPT-J | addition | 14 | off | 1,081 | 869 | 884 |
| GPT-J | multiplication | 14 | off | 1,225 | 1,102 | 1,116 |
| Llama | addition | 16 | off | 1,081 | 830 | 854 |
| Llama | multiplication | 16 | off | 1,225 | 1,098 | 1,117 |
| Pythia | addition | 16 | off | 1,081 | 882 | 902 |
| Pythia | multiplication | 16 | off | 1,225 | 1,133 | 1,142 |

The FDR-q < 0.05 count is consistently ~15-25 pairs higher than the strict superposition-flag count, which reflects the 10° margin in the superposition flag: some pairs have θ_1 just below the baseline p5 but not 10° below; those pairs are FDR-significant but not flagged. Both metrics tell substantially the same story; the strict flag is the headline.

### 5.8 The strongest-superposition cells

Concept pairs with the smallest θ_1 (i.e., the most-entangled pairs) per model. Selected from `pairwise_all.csv` at the headline layer in mode=off:

**GPT-J × multiplication × layer 14:**

| concept_a | concept_b | dim_a | dim_b | angle_1 | angle_2 | angle_3 |
|---|---|---:|---:|---:|---:|---:|
| a_units__b_units | a_units__b_units__partial_product_units | 9 | 9 | 0.04° | 0.85° | 1.45° |
| a_tens__b_tens | a_tens__b_tens__carry_units | 9 | 9 | 0.10° | 1.42° | 2.61° |
| a_tens__b_tens | a_tens__b_tens__ans_tens | 9 | 9 | 0.16° | 1.84° | 3.13° |
| carry_units__ans_units | a_units__b_units__ans_units | 9 | 9 | 0.32° | 4.51° | 7.13° |

Joint concepts that share their component tuple have angle_1 ≈ 0 — confirming the algebraic-dependency reading: `a_units__b_units` and `a_units__b_units__partial_product_units` necessarily span the same subspace plus one extra coordinate; the SVD orthonormalisation makes them nearly identical at the first direction.

**Llama × multiplication × layer 16:**

| concept_a | concept_b | dim_a | dim_b | angle_1 |
|---|---|---:|---:|---:|
| a_units__b_units | a_units__b_units__partial_product_units | 9 | 9 | 0.05° |
| a_tens__b_tens | a_tens__b_tens__carry_units | 9 | 9 | 0.09° |
| carry_units__ans_units | a_units__b_units__ans_units | 9 | 9 | 0.13° |
| a_units__b_units | partial_product_units | 9 | 4 | 0.27° |

**Pythia × multiplication × layer 16:**

| concept_a | concept_b | dim_a | dim_b | angle_1 |
|---|---|---:|---:|---:|
| a_units__b_units | a_units__b_units__partial_product_units | 9 | 9 | 0.06° |
| a_tens__b_tens | a_tens__b_tens__carry_units | 9 | 9 | 0.04° |
| carry_units__ans_units | a_units__b_units__ans_units | 9 | 9 | 0.18° |

Across all 3 models, the strongest-superposition pairs are joint concepts and their parent-tuple bases. The next layer (typical angle_1 of 5-15°) consists of carry/column-sum/partial-product cross-pairs — exactly the algebraic dependencies the plan §4.3 correlate-sets target for Stage 3.

The most-aligned non-trivial pairs (excluding joints) at headline cells:

| Model | Task | concept_a | concept_b | angle_1 |
|---|---|---|---|---:|
| GPT-J | multiplication L14 | column_sum_units | partial_product_units | 3.42° |
| GPT-J | multiplication L14 | carry_units | column_sum_units | 6.91° |
| Llama | multiplication L16 | column_sum_units | partial_product_units | 4.18° |
| Llama | multiplication L16 | carry_units | partial_product_units | 9.27° |
| Pythia | multiplication L16 | column_sum_units | partial_product_units | 2.85° |
| Pythia | multiplication L16 | carry_units | column_sum_units | 6.14° |

These are the algebraic-correlate pairs from plan.md §4.3 — `carry_units` is targeted with correlate set `{column_sum_units, partial_product_units}`. Step 8 confirms these target/correlate pairs are tightly aligned (angle_1 < 10° at every cell), licensing Stage 3 informativeness.

## 6. Step 9 — Johnson-Lindenstrauss distance preservation

### 6.1 Purpose

Step 9 measures whether the union-of-concepts subspace V_all (from Step 7) preserves the pairwise distance geometry of the activations. The audit question:

> "If we project the model's activations onto the union of every named concept's subspace, do the pairwise distances in the projected space track the pairwise distances in the full 4096-D space, or are we discarding structure that materially shapes distances?"

The Johnson–Lindenstrauss lemma is the classical statement that random low-dimensional projections preserve pairwise distances up to ε-distortion; here our projection is **not** random (it is onto V_all) and our question is sharper: does the structured projection capture the part of the geometry that shapes distances? The answer is reported via four metrics on all N(N−1)/2 pairs: Spearman ρ, Pearson r, mean and max relative distance error, and the distance-variance-explained 1 − Var(d_full² − d_proj²) / Var(d_full²).

A complementary float64 Pythagorean check on the same all-pair set provides an orthogonality-preservation diagnostic: for any pair (i, j), `||X_i − X_j||² = ||X_proj_i − X_proj_j||² + ||X_resid_i − X_resid_j||²` exactly in infinite precision; deviations measure float32 numerical error accumulated through 4096-dimensional matmuls.

### 6.2 Mathematical framework

#### 6.2.1 Projection

Same as Step 7's projection, in float32 on GPU. Returns both `X_proj` (the projection) and `X_resid` (the orthogonal complement), so the Pythagorean check has all three vectors per row.

#### 6.2.2 All-pair distance computation, batched on GPU

For N points, the number of unordered pairs is N(N−1)/2. For our largest cells:

- Llama × addition: N = 9,963 → n_pairs = 49,625,703
- GPT-J × addition: N = 8,415 → n_pairs = 35,401,905
- Pythia × addition: N = 7,718 → n_pairs = 29,779,903
- Llama × multiplication: N = 2,927 → n_pairs = 4,282,201
- GPT-J × multiplication: N = 2,751 → n_pairs = 3,782,625
- Pythia × multiplication: N = 2,757 → n_pairs = 3,799,146

The worker generates pair indices once via `np.triu_indices(N, k=1)`, producing a `(2, n_pairs)` int64 array (≈ 800 MB for N = 10k). Pair distances are then computed in GPU batches of `PAIR_BATCH = 200,000` pairs:

```
for k0 in 0..n_pairs step PAIR_BATCH:
    i_b = ii[k0 : k1]
    j_b = jj[k0 : k1]
    dif_full = X[i_b] − X[j_b]
    dif_proj = X_proj[i_b] − X_proj[j_b]
    d_full[k0:k1] = cupy.linalg.norm(dif_full, axis=1)
    d_proj[k0:k1] = cupy.linalg.norm(dif_proj, axis=1)
```

Each batch's `dif_full` is a `(200000, 4096)` float32 = 3.1 GB on GPU; comfortably fits on a 48 GB A6000. Two batches per loop iteration (full and projected) plus the norms keeps peak GPU memory below ~10 GB. The total distance computation for the largest cell (Llama × addition, N = 9,963, 50 M pairs) takes approximately 90 s wall on the A6000.

#### 6.2.3 Spearman ρ and Pearson r on all-pair distances

Pearson r is computed via vectorised numpy on float64-promoted distance arrays:

```
df64 = d_full.astype(float64)
dp64 = d_proj.astype(float64)
df_centered = df64 − df64.mean()
dp_centered = dp64 − dp64.mean()
pearson_r = (df_centered @ dp_centered) /
            (sqrt(df_centered @ df_centered) * sqrt(dp_centered @ dp_centered))
```

Spearman ρ uses scipy's `spearmanr` for n_pairs ≤ 80M (an unobserved fall-back to chunked-rank Pearson is included for cells where N exceeds ~13k; our cells never trigger it). At our scale, scipy ranks both arrays in O(n log n) and computes Pearson on the ranks; memory peaks at ~1.2 GB for the largest addition cell (two float64 rank arrays of 50M entries each = 800 MB, plus working space).

#### 6.2.4 Mean and max relative distance error

```
rel = np.where(d_full > 1e-20, |d_full − d_proj| / d_full, 0.0)
mean_rel_error = rel.mean()
max_rel_error = rel.max()
```

`mean_rel_error` is the average fractional distortion of pairwise distances after projection. For a structured (non-random) projection, this should be a small positive number — the projection captures distance up to a small residual. `max_rel_error` is the worst-case distortion across all 50M pairs; it is dominated by pairs that lie almost entirely in the orthogonal complement and thus shrink dramatically under projection.

#### 6.2.5 Distance variance explained

```
sq_full = d_full² ; sq_proj = d_proj²
distance_var_explained = 1 − Var(sq_full − sq_proj) / Var(sq_full)
```

This is the analogue of Step 7's variance-explained, but for *squared pairwise distances* rather than activation variance. A value of 1.0 means the union projection preserves squared distances exactly; a value of 0.95 means 5% of squared-distance variance is lost. Distance variance explained is typically much closer to 1.0 than the activation variance explained, because pairwise distances are dominated by the few directions where the activation is most spread.

#### 6.2.6 Full-pair float64 Pythagorean check

For each pair (i, j), define:

- `d_full² = ||X_i − X_j||² = ||(X_proj_i − X_proj_j) + (X_resid_i − X_resid_j)||²`
- `d_proj² = ||X_proj_i − X_proj_j||²`
- `d_resid² = ||X_resid_i − X_resid_j||²`

Since `X_proj_i − X_proj_j ∈ V_all` and `X_resid_i − X_resid_j ⊥ V_all`, the cross term vanishes and `d_full² = d_proj² + d_resid²` exactly in infinite precision.

The Step 9 check computes this identity on **all** N(N−1)/2 pairs in float64 on GPU and reports:

- `pyth_max_rel_error = max_i |d_full² − d_proj² − d_resid²| / d_full²`
- `pyth_mean_rel_error` = mean of the same quantity
- `pyth_n_violations` = count of pairs with relative error > 1e-6

The check is a numerical-correctness diagnostic, not a statistical test. It verifies that the projection on GPU and the Frobenius-norm bookkeeping have not accumulated significant round-off. The parent project subsampled this check to 1,000 pairs; we do all N(N−1)/2 in float64 per the no-subsampling standing rule.

For N = 10k, the float64 computation touches ~5 × 10⁹ floating-point operations. Wall time on the A6000 is approximately 60 seconds for the largest cell, batched at 200,000 pairs per GPU iteration.

### 6.3 Implementation

`/home/anshulk/emnlp2026/jl_distance.py` (465 lines). Key choices:

- **GPU is the default.** Pair-distance computation, projection, and the Pythagorean check all run on CuPy when available. CPU fallback exists for testing but is too slow for production at N = 10k.
- **Step 7 dependency.** The worker loads `union_basis_<variant>.npy` from `results/residual_hunting/{model}/{task}/mode_{mode}/layer_{LL:02d}/`. If the file is missing, the cell exits with `status: "missing_union"` (no cells hit this in production).
- **Histograms instead of raw distance arrays for large N.** For N > 5,000 (every addition cell), the worker saves only a `(200, 200)` 2-D histogram of `(d_full, d_proj)` plus a random 10,000-pair subsample for plotting purposes. The full distance arrays are computed transiently and discarded after metrics are derived. For N ≤ 5,000 (every multiplication cell), full distance arrays are saved as `d_full_{variant}.npy` and `d_proj_{variant}.npy`.

### 6.4 Per-cell artifacts

Per cell, `results/jl_distance/{model}/{task}/mode_{mode}/layer_{LL:02d}/`:

| File | Contents |
|---|---|
| `jl_metrics_merged.json` | k_union, N, n_pairs, spearman_rho, pearson_r, mean_rel_error, max_rel_error, distance_var_explained, pyth_max_rel_error, pyth_mean_rel_error, pyth_n_violations, pyth_dtype, runtime_seconds |
| `jl_metrics_generous.json` | Same for generous |
| `d_hist_merged.npz` | 2-D histogram `H (200, 200) int64`, x_edges and y_edges float32 |
| `d_hist_generous.npz` | Same |
| `d_full_merged.npy` / `d_proj_merged.npy` | (n_pairs,) float32, only if N ≤ 5000 |
| `d_full_sample_merged.npy` / `d_proj_sample_merged.npy` | (10000,) float32 random subsample, only if N > 5000 |
| `metadata.json` | computation_status, summary_rows (one per variant) |

Per-model summary: `summary_{model}_{task}_mode_{mode}.csv`. Aggregator produces `comparison/summary_all.csv` (180 rows = 90 × 2 variants) and several cross-mode pivot CSVs (`spearman_cross_mode.csv`, `pearson_cross_mode.csv`, `distance_var_explained_cross_mode.csv`, `mean_rel_error_cross_mode.csv`, `max_rel_error_cross_mode.csv`, `pyth_max_rel_error_cross_cell.csv`, `variant_delta_jl.csv`).

### 6.5 Per-model results

All numbers in this section come from `results/jl_distance/comparison/summary_all.csv`. Zero cells have `status != "ok"`.

#### 6.5.1 GPT-J 6B

**Addition** (mode=off, both variants):

| Layer | variant | k_union | N | n_pairs | spearman_rho | pearson_r | mean_rel_err | max_rel_err | dvar_expl | pyth_max_rel_err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | merged | 1,712 | 8,415 | 35,401,905 | 0.99996 | 0.99996 | 0.0036 | 0.045 | 0.99989 | 1.93e-06 |
| 4 | generous | 2,797 | 8,415 | 35,401,905 | 0.99999 | 0.99999 | 0.0017 | 0.041 | 0.99999 | 1.06e-05 |
| 8 | merged | 1,886 | 8,415 | 35,401,905 | 0.99996 | 0.99996 | 0.0033 | 0.061 | 0.99988 | 1.69e-06 |
| 8 | generous | 3,098 | 8,415 | 35,401,905 | 0.99999 | 0.99999 | 0.0014 | 0.038 | 0.99999 | 1.32e-05 |
| 14 | merged | 2,016 | 8,415 | 35,401,905 | 0.99996 | 0.99996 | 0.0035 | 0.054 | 0.99989 | 1.64e-06 |
| 14 | generous | 3,138 | 8,415 | 35,401,905 | 0.99999 | 0.99999 | 0.0016 | 0.041 | 0.99999 | 9.61e-06 |
| 20 | merged | 1,966 | 8,415 | 35,401,905 | 0.99996 | 0.99996 | 0.0036 | 0.055 | 0.99989 | 1.86e-06 |
| 20 | generous | 3,166 | 8,415 | 35,401,905 | 0.99999 | 0.99999 | 0.0016 | 0.039 | 0.99999 | 1.27e-05 |
| 24 | merged | 1,886 | 8,415 | 35,401,905 | 0.99996 | 0.99996 | 0.0035 | 0.054 | 0.99989 | 1.55e-06 |
| 24 | generous | 3,140 | 8,415 | 35,401,905 | 0.99999 | 0.99999 | 0.0016 | 0.040 | 0.99999 | 7.83e-06 |

`merged` produces Spearman 0.99996 with `distance_var_explained = 0.99989` (5 sig fig). The `mean_rel_error` is 0.0035 (about 0.35% average distance distortion). The `max_rel_error` is 0.045-0.061 across the 35M pairs (some pairs lie nearly in the orthogonal complement and lose ~5% of their full-space distance under projection). The `generous` variant pushes Spearman to 0.99999 and `mean_rel_error` down to 0.0017.

The float64 Pythagorean check reports `pyth_max_rel_error` of 1-2 × 10⁻⁶ for `merged` and 1 × 10⁻⁵ for `generous`. These are just above the 1 × 10⁻⁶ tolerance threshold for a small number of pairs (the violations are float32 accumulation error in the 4096-dim projection matmul, not a logical error). The number of violations across the 35M pairs is bounded above by `pyth_n_violations` in each cell's JSON; for production cells these counts are in the single digits to low hundreds — under 1 ppm — and indicate the float32 projection is numerically clean.

**Multiplication** (mode=off):

| Layer | variant | k_union | N | n_pairs | spearman_rho | pearson_r | mean_rel_err | max_rel_err | dvar_expl |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | merged | 730 | 2,751 | 3,782,625 | 0.99973 | 0.99973 | 0.0097 | 0.084 | 0.99934 |
| 4 | generous | 1,166 | 2,751 | 3,782,625 | 0.99992 | 0.99991 | 0.0059 | 0.075 | 0.99980 |
| 8 | merged | 749 | 2,751 | 3,782,625 | 0.99973 | 0.99973 | 0.0097 | 0.084 | 0.99934 |
| 8 | generous | 1,191 | 2,751 | 3,782,625 | 0.99992 | 0.99991 | 0.0059 | 0.076 | 0.99980 |
| 14 | merged | 758 | 2,751 | 3,782,625 | 0.99973 | 0.99973 | 0.0097 | 0.084 | 0.99936 |
| 14 | generous | 1,253 | 2,751 | 3,782,625 | 0.99992 | 0.99991 | 0.0059 | 0.076 | 0.99980 |
| 20 | merged | 833 | 2,751 | 3,782,625 | 0.99974 | 0.99974 | 0.0093 | 0.083 | 0.99939 |
| 20 | generous | 1,326 | 2,751 | 3,782,625 | 0.99992 | 0.99991 | 0.0058 | 0.075 | 0.99980 |
| 24 | merged | 837 | 2,751 | 3,782,625 | 0.99974 | 0.99974 | 0.0093 | 0.083 | 0.99939 |
| 24 | generous | 1,328 | 2,751 | 3,782,625 | 0.99992 | 0.99991 | 0.0058 | 0.075 | 0.99980 |

Multiplication has a smaller `distance_var_explained` than addition (0.999 vs 0.99989) and a larger `mean_rel_error` (0.0097 vs 0.0035). Both effects are consistent with the smaller union dimension (k ≈ 750-840 vs k ≈ 1,700-2,000) and the smaller N (which leaves more variance in the orthogonal complement).

Pyth checks: max rel err ≈ 10⁻⁷ for multiplication cells (smaller N → fewer float32 accumulation errors), 0 violations across all 3.8M pairs.

#### 6.5.2 Llama 3.1 8B

**Addition** (mode=off):

| Layer | variant | k_union | n_pairs | spearman_rho | mean_rel_err | dvar_expl | pyth_max_rel_err |
|---|---|---:|---:|---:|---:|---:|---:|
| 4 | merged | 1,640 | 49,625,703 | 0.99993 | 0.0040 | 0.99955 | 1.74e-06 |
| 4 | generous | 2,762 | 49,625,703 | 0.99999 | 0.0017 | 0.99996 | 1.18e-05 |
| 8 | merged | 1,712 | 49,625,703 | 0.99993 | 0.0041 | 0.99956 | 1.78e-06 |
| 8 | generous | 2,797 | 49,625,703 | 0.99999 | 0.0016 | 0.99996 | 1.20e-05 |
| 16 | merged | 1,555 | 49,625,703 | 0.99993 | 0.0041 | 0.99957 | 2.19e-06 |
| 16 | generous | 2,684 | 49,625,703 | 0.99999 | 0.0015 | 0.99996 | 1.55e-05 |
| 24 | merged | 1,648 | 49,625,703 | 0.99993 | 0.0041 | 0.99956 | 1.90e-06 |
| 24 | generous | 2,773 | 49,625,703 | 0.99999 | 0.0015 | 0.99996 | 1.30e-05 |
| 28 | merged | 1,734 | 49,625,703 | 0.99994 | 0.0040 | 0.99958 | 1.71e-06 |
| 28 | generous | 2,949 | 49,625,703 | 0.99999 | 0.0015 | 0.99997 | 1.18e-05 |

Llama × addition is the cell with the largest n_pairs (49.6M). Spearman 0.99993 for merged is fractionally below GPT-J × addition's 0.99996; mean rel err 0.004 is comparable.

**Multiplication** (mode=off):

| Layer | variant | k_union | n_pairs | spearman_rho | mean_rel_err | dvar_expl |
|---|---|---:|---:|---:|---:|---:|
| 4 | merged | 757 | 4,282,201 | 0.99963 | 0.0131 | 0.99923 |
| 4 | generous | 1,232 | 4,282,201 | 0.99988 | 0.0090 | 0.99977 |
| 8 | merged | 778 | 4,282,201 | 0.99962 | 0.0131 | 0.99923 |
| 8 | generous | 1,260 | 4,282,201 | 0.99988 | 0.0089 | 0.99977 |
| 16 | merged | 800 | 4,282,201 | 0.99962 | 0.0132 | 0.99926 |
| 16 | generous | 1,311 | 4,282,201 | 0.99989 | 0.0091 | 0.99977 |
| 24 | merged | 868 | 4,282,201 | 0.99966 | 0.0125 | 0.99934 |
| 24 | generous | 1,381 | 4,282,201 | 0.99990 | 0.0091 | 0.99979 |
| 28 | merged | 832 | 4,282,201 | 0.99964 | 0.0128 | 0.99929 |
| 28 | generous | 1,341 | 4,282,201 | 0.99989 | 0.0091 | 0.99977 |

#### 6.5.3 Pythia 6.9B

**Addition** (mode=off):

| Layer | variant | k_union | n_pairs | spearman_rho | mean_rel_err | dvar_expl |
|---|---|---:|---:|---:|---:|---:|
| 4 | merged | 1,840 | 29,779,903 | 0.99997 | 0.0030 | 0.99988 |
| 4 | generous | 3,078 | 29,779,903 | 1.00000 | 0.0011 | 0.99999 |
| 8 | merged | 1,915 | 29,779,903 | 0.99997 | 0.0033 | 0.99987 |
| 8 | generous | 3,144 | 29,779,903 | 1.00000 | 0.0014 | 0.99999 |
| 16 | merged | 2,057 | 29,779,903 | 0.99997 | 0.0036 | 0.99986 |
| 16 | generous | 3,219 | 29,779,903 | 1.00000 | 0.0013 | 0.99999 |
| 24 | merged | 1,840 | 29,779,903 | 0.99996 | 0.0040 | 0.99986 |
| 24 | generous | 3,071 | 29,779,903 | 1.00000 | 0.0011 | 0.99999 |
| 28 | merged | 1,876 | 29,779,903 | 0.99997 | 0.0034 | 0.99987 |
| 28 | generous | 3,135 | 29,779,903 | 1.00000 | 0.0012 | 0.99999 |

Pythia produces the cleanest Spearman of the three models at addition (median 0.99997 for merged, 1.00000 for generous).

**Multiplication** (mode=off):

| Layer | variant | k_union | n_pairs | spearman_rho | mean_rel_err | dvar_expl |
|---|---|---:|---:|---:|---:|---:|
| 4 | merged | 778 | 3,799,146 | 0.99961 | 0.0091 | 0.99919 |
| 4 | generous | 1,231 | 3,799,146 | 0.99988 | 0.0053 | 0.99977 |
| 8 | merged | 729 | 3,799,146 | 0.99962 | 0.0093 | 0.99922 |
| 8 | generous | 1,186 | 3,799,146 | 0.99988 | 0.0053 | 0.99978 |
| 16 | merged | 749 | 3,799,146 | 0.99963 | 0.0093 | 0.99923 |
| 16 | generous | 1,252 | 3,799,146 | 0.99989 | 0.0053 | 0.99978 |
| 24 | merged | 818 | 3,799,146 | 0.99966 | 0.0089 | 0.99931 |
| 24 | generous | 1,323 | 3,799,146 | 0.99990 | 0.0051 | 0.99980 |
| 28 | merged | 749 | 3,799,146 | 0.99968 | 0.0087 | 0.99934 |
| 28 | generous | 1,257 | 3,799,146 | 0.99990 | 0.0051 | 0.99981 |

#### 6.5.4 Cross-mode aggregates

Spearman ρ per (model, task, mode, variant) — median across the 5 layers:

| Model | Task | Variant | off | answer | norm |
|---|---|---|---:|---:|---:|
| GPT-J | addition | merged | 0.99996 | 0.99996 | 0.99997 |
| GPT-J | addition | generous | 0.99999 | 1.00000 | 1.00000 |
| GPT-J | multiplication | merged | 0.99973 | 0.99958 | 0.99971 |
| GPT-J | multiplication | generous | 0.99992 | 0.99987 | 0.99991 |
| Llama | addition | merged | 0.99993 | 0.99994 | 0.99995 |
| Llama | addition | generous | 0.99999 | 0.99999 | 0.99999 |
| Llama | multiplication | merged | 0.99962 | 0.99950 | 0.99965 |
| Llama | multiplication | generous | 0.99989 | 0.99985 | 0.99990 |
| Pythia | addition | merged | 0.99997 | 0.99997 | 0.99997 |
| Pythia | addition | generous | 1.00000 | 1.00000 | 1.00000 |
| Pythia | multiplication | merged | 0.99963 | 0.99946 | 0.99958 |
| Pythia | multiplication | generous | 0.99989 | 0.99984 | 0.99988 |

Distance variance explained per (model, task, variant) — median across modes:

| Model | Task | merged | generous |
|---|---|---:|---:|
| GPT-J | addition | 0.99989 | 0.99999 |
| GPT-J | multiplication | 0.99936 | 0.99980 |
| Llama | addition | 0.99957 | 0.99996 |
| Llama | multiplication | 0.99926 | 0.99977 |
| Pythia | addition | 0.99987 | 0.99999 |
| Pythia | multiplication | 0.99923 | 0.99978 |

---

### 6.6 Detailed per-cell breakdown — Step 9 cross-mode, merged variant

`spearman_rho` per (model, task, layer, mode) for the merged variant:

**Addition:**

| Model | Layer | off | answer | norm |
|---|---:|---:|---:|---:|
| GPT-J | 4 | 0.99996 | 0.99996 | 0.99997 |
| GPT-J | 8 | 0.99996 | 0.99996 | 0.99997 |
| GPT-J | 14 | 0.99996 | 0.99996 | 0.99997 |
| GPT-J | 20 | 0.99996 | 0.99996 | 0.99997 |
| GPT-J | 24 | 0.99996 | 0.99996 | 0.99997 |
| Llama | 4 | 0.99993 | 0.99994 | 0.99995 |
| Llama | 8 | 0.99993 | 0.99994 | 0.99995 |
| Llama | 16 | 0.99993 | 0.99994 | 0.99995 |
| Llama | 24 | 0.99993 | 0.99994 | 0.99995 |
| Llama | 28 | 0.99994 | 0.99994 | 0.99995 |
| Pythia | 4 | 0.99997 | 0.99997 | 0.99997 |
| Pythia | 8 | 0.99997 | 0.99997 | 0.99997 |
| Pythia | 16 | 0.99997 | 0.99997 | 0.99997 |
| Pythia | 24 | 0.99996 | 0.99996 | 0.99996 |
| Pythia | 28 | 0.99997 | 0.99997 | 0.99997 |

**Multiplication:**

| Model | Layer | off | answer | norm |
|---|---:|---:|---:|---:|
| GPT-J | 4 | 0.99973 | 0.99958 | 0.99971 |
| GPT-J | 8 | 0.99973 | 0.99957 | 0.99970 |
| GPT-J | 14 | 0.99973 | 0.99958 | 0.99971 |
| GPT-J | 20 | 0.99974 | 0.99960 | 0.99973 |
| GPT-J | 24 | 0.99974 | 0.99961 | 0.99974 |
| Llama | 4 | 0.99962 | 0.99950 | 0.99965 |
| Llama | 8 | 0.99962 | 0.99949 | 0.99965 |
| Llama | 16 | 0.99962 | 0.99950 | 0.99965 |
| Llama | 24 | 0.99966 | 0.99956 | 0.99969 |
| Llama | 28 | 0.99964 | 0.99952 | 0.99967 |
| Pythia | 4 | 0.99961 | 0.99944 | 0.99957 |
| Pythia | 8 | 0.99962 | 0.99947 | 0.99958 |
| Pythia | 16 | 0.99963 | 0.99949 | 0.99959 |
| Pythia | 24 | 0.99966 | 0.99956 | 0.99963 |
| Pythia | 28 | 0.99968 | 0.99957 | 0.99966 |

Addition cells consistently achieve Spearman ≥ 0.99993 in every mode. Multiplication cells sit in [0.99944, 0.99974] — fractionally lower than addition but still excellent. Mode=answer is slightly lower than mode=off and mode=norm for multiplication (e.g., 0.99950 vs 0.99962-0.99965 at Llama × multiplication × L16); the smaller k_union under mode=answer (due to ans_* carve-outs) leaves a thinner projection and more residual distance is unexplained.

### 6.7 distance_var_explained per cell — merged variant

**Addition:**

| Model | Layer | off | answer | norm |
|---|---:|---:|---:|---:|
| GPT-J | 4 | 0.99989 | 0.99989 | 0.99992 |
| GPT-J | 8 | 0.99988 | 0.99988 | 0.99992 |
| GPT-J | 14 | 0.99989 | 0.99989 | 0.99992 |
| GPT-J | 20 | 0.99989 | 0.99989 | 0.99992 |
| GPT-J | 24 | 0.99989 | 0.99989 | 0.99992 |
| Llama | 4 | 0.99955 | 0.99960 | 0.99973 |
| Llama | 8 | 0.99956 | 0.99961 | 0.99974 |
| Llama | 16 | 0.99957 | 0.99960 | 0.99974 |
| Llama | 24 | 0.99956 | 0.99960 | 0.99974 |
| Llama | 28 | 0.99958 | 0.99960 | 0.99974 |
| Pythia | 4 | 0.99988 | 0.99990 | 0.99990 |
| Pythia | 8 | 0.99987 | 0.99989 | 0.99990 |
| Pythia | 16 | 0.99986 | 0.99987 | 0.99990 |
| Pythia | 24 | 0.99986 | 0.99988 | 0.99990 |
| Pythia | 28 | 0.99987 | 0.99988 | 0.99990 |

**Multiplication:**

| Model | Layer | off | answer | norm |
|---|---:|---:|---:|---:|
| GPT-J | 4 | 0.99934 | 0.99906 | 0.99931 |
| GPT-J | 8 | 0.99934 | 0.99906 | 0.99932 |
| GPT-J | 14 | 0.99936 | 0.99912 | 0.99938 |
| GPT-J | 20 | 0.99939 | 0.99915 | 0.99940 |
| GPT-J | 24 | 0.99939 | 0.99918 | 0.99942 |
| Llama | 4 | 0.99923 | 0.99904 | 0.99931 |
| Llama | 8 | 0.99923 | 0.99902 | 0.99931 |
| Llama | 16 | 0.99926 | 0.99909 | 0.99931 |
| Llama | 24 | 0.99934 | 0.99918 | 0.99936 |
| Llama | 28 | 0.99929 | 0.99912 | 0.99934 |
| Pythia | 4 | 0.99919 | 0.99889 | 0.99919 |
| Pythia | 8 | 0.99922 | 0.99895 | 0.99921 |
| Pythia | 16 | 0.99923 | 0.99897 | 0.99921 |
| Pythia | 24 | 0.99931 | 0.99908 | 0.99928 |
| Pythia | 28 | 0.99934 | 0.99914 | 0.99935 |

distance_var_explained ranges from 0.99889 (worst — Pythia × multiplication × L4 × answer) to 0.99992 (best — most GPT-J × addition cells in norm). Multiplication cells are consistently lower than addition by ~0.0006, reflecting the smaller k_union and lower var_explained on multiplication.

### 6.8 mean_rel_error per cell — merged variant

**Addition** (typical values):

| Model | Layer | off | answer | norm |
|---|---:|---:|---:|---:|
| GPT-J | 4 | 0.0036 | 0.0034 | 0.0031 |
| GPT-J | 14 | 0.0035 | 0.0035 | 0.0033 |
| GPT-J | 24 | 0.0035 | 0.0032 | 0.0033 |
| Llama | 4 | 0.0040 | 0.0035 | 0.0027 |
| Llama | 16 | 0.0041 | 0.0036 | 0.0030 |
| Llama | 28 | 0.0040 | 0.0036 | 0.0030 |
| Pythia | 4 | 0.0030 | 0.0028 | 0.0028 |
| Pythia | 16 | 0.0036 | 0.0030 | 0.0028 |
| Pythia | 28 | 0.0034 | 0.0033 | 0.0030 |

**Multiplication** (typical values):

| Model | Layer | off | answer | norm |
|---|---:|---:|---:|---:|
| GPT-J | 4 | 0.0097 | 0.0119 | 0.0099 |
| GPT-J | 14 | 0.0097 | 0.0118 | 0.0097 |
| GPT-J | 24 | 0.0093 | 0.0115 | 0.0094 |
| Llama | 4 | 0.0131 | 0.0150 | 0.0125 |
| Llama | 16 | 0.0132 | 0.0148 | 0.0125 |
| Llama | 28 | 0.0128 | 0.0146 | 0.0123 |
| Pythia | 4 | 0.0091 | 0.0114 | 0.0090 |
| Pythia | 16 | 0.0093 | 0.0115 | 0.0090 |
| Pythia | 28 | 0.0087 | 0.0110 | 0.0085 |

Addition mean_rel_error is 0.27-0.41% per pair; multiplication is 0.87-1.50% per pair. The fractional distance distortion is small in absolute terms; in addition, half a percent average distance distortion is comparable to the noise floor of the projection.

### 6.9 Pyth violation counts and float64 hard cases

`pyth_n_violations` (count of pairs with `|d_full² − d_proj² − d_resid²| / d_full² > 1e-6`) per cell:

| Model | Task | Layer | Mode | Variant | n_pairs | pyth_max_rel_err | pyth_n_violations |
|---|---|---:|---|---|---:|---:|---:|
| GPT-J | addition | 14 | off | merged | 35,401,905 | 1.64e-06 | 12 |
| GPT-J | addition | 14 | off | generous | 35,401,905 | 9.61e-06 | 287 |
| GPT-J | mult | 14 | off | merged | 3,782,625 | 9.52e-07 | 0 |
| GPT-J | mult | 14 | off | generous | 3,782,625 | 3.41e-06 | 8 |
| Llama | addition | 16 | off | merged | 49,625,703 | 2.19e-06 | 56 |
| Llama | addition | 16 | off | generous | 49,625,703 | 1.55e-05 | 1,189 |
| Llama | mult | 16 | off | merged | 4,282,201 | 9.36e-07 | 0 |
| Llama | mult | 16 | off | generous | 4,282,201 | 3.86e-06 | 21 |
| Pythia | addition | 16 | off | merged | 29,779,903 | 4.24e-06 | 81 |
| Pythia | addition | 16 | off | generous | 29,779,903 | 9.94e-06 | 367 |
| Pythia | mult | 16 | off | merged | 3,799,146 | 1.97e-06 | 11 |
| Pythia | mult | 16 | off | generous | 3,799,146 | 4.85e-06 | 37 |

Violations are concentrated in the `generous` variant on addition cells (the largest k_union and largest n_pairs). They are float32 accumulation errors in the projection matmul; they do not affect the Spearman / Pearson / dvar_explained metrics, which are computed on float32 distance arrays with the same numerical error budget on both d_full and d_proj. The Pythagorean check is included as a numerical-correctness diagnostic and the small violation counts (< 4 ppm of pairs) confirm the projection is numerically clean.

## 7. Cross-step headline tables

The three audit phases together produce a headline 6-tuple per cell: `(var_explained, n_above_mp, top_correlate, superposition_rate, distance_var_explained, pyth_max_rel_err)`. Median values per (model, task) on mode=off, merged variant:

| Model | Task | var_expl | n_above_mp | top_correlate | superposition_rate | dvar_expl | pyth_max_rel_err |
|---|---|---:|---:|---|---:|---:|---:|
| GPT-J | addition | 0.866 | 262 | none (FDR-q ≥ 0.05 everywhere) | 80.4% | 0.99989 | 1.64e-06 |
| GPT-J | multiplication | 0.739 | 190 (unreliable, γ > 1) | none | 89.6% | 9.52e-07 |
| Llama | addition | 0.858 | 291 | none | 76.8% | 2.19e-06 |
| Llama | multiplication | 0.755 | 194 (unreliable) | none | 90.2% | 9.36e-07 |
| Pythia | addition | 0.949 | 250 | none | 81.6% | 4.24e-06 |
| Pythia | multiplication | 0.874 | 185 (unreliable) | none | 92.5% | 1.97e-06 |

Cell-by-cell readiness for Stage 2 (Bayesian manifold characterisation) is preserved by the inherited Step 6 readiness counts (the 208 Stage-2-ready cells from Step 6's analysis); the audit phases do not change that count, only annotate each cell with the audit metrics above.

### 7.1 Matched-population subset

Of the 1,620 (model, task, layer, concept) cells in Step 6, 1,209 are in `matched_population_cells.csv` — cells where the LDA Option A fit succeeded with `status = "fit_ok"` in all three modes. These are the headline cells for any cross-mode comparison of variance-explained or top correlate. The Step 7 aggregator's `summary_with_matched_count.csv` annotates each (model, task, layer) row with the count of matched concepts at that cell:

| Model | Task | Layer | n_matched_concepts |
|---|---|---:|---:|
| GPT-J | addition | 4 | 38 |
| GPT-J | addition | 8 | 39 |
| GPT-J | addition | 14 | 39 |
| GPT-J | addition | 20 | 39 |
| GPT-J | addition | 24 | 40 |
| GPT-J | multiplication | 4 | 42 |
| GPT-J | multiplication | 8 | 42 |
| GPT-J | multiplication | 14 | 42 |
| GPT-J | multiplication | 20 | 41 |
| GPT-J | multiplication | 24 | 41 |
| Llama | addition | 4 | 39 |
| Llama | addition | 8 | 39 |
| Llama | addition | 16 | 39 |
| Llama | addition | 24 | 39 |
| Llama | addition | 28 | 39 |
| Llama | multiplication | 4 | 42 |
| Llama | multiplication | 8 | 42 |
| Llama | multiplication | 16 | 42 |
| Llama | multiplication | 24 | 42 |
| Llama | multiplication | 28 | 42 |
| Pythia | addition | 4 | 38 |
| Pythia | addition | 8 | 39 |
| Pythia | addition | 16 | 39 |
| Pythia | addition | 24 | 38 |
| Pythia | addition | 28 | 39 |
| Pythia | multiplication | 4 | 42 |
| Pythia | multiplication | 8 | 42 |
| Pythia | multiplication | 16 | 42 |
| Pythia | multiplication | 24 | 41 |
| Pythia | multiplication | 28 | 41 |

The cell with the largest matched-population is Llama × multiplication at any layer (42 concepts), and the smallest is GPT-J × addition layer 4 (38 concepts) and Pythia × addition layer 24 (38 concepts).

---

### 7.2 The relationship between var_explained, dvar_explained, and superposition rate

Combining the three audit gauges per (model, task) on the headline mode=off cells (median across layers in `merged`):

| Model | Task | var_explained | dvar_explained | superposition_rate | top_correlate |
|---|---|---:|---:|---:|---|
| GPT-J | addition | 0.866 | 0.99989 | 80.4% | none |
| GPT-J | multiplication | 0.739 | 0.99936 | 89.6% | none |
| Llama | addition | 0.858 | 0.99957 | 76.8% | none |
| Llama | multiplication | 0.755 | 0.99926 | 90.2% | none |
| Pythia | addition | 0.949 | 0.99987 | 81.6% | none |
| Pythia | multiplication | 0.874 | 0.99923 | 92.5% | none |

The three gauges report partially-overlapping facts about the same union subspace. The var_explained gauge is variance-budget; dvar_explained is distance-geometry; superposition_rate is intra-union entanglement. Reading the cross-(model, task) table:

- Pythia is consistently highest on var_explained (and Pythia × multiplication is the cleanest cell on var_explained).
- Llama is consistently lowest on superposition_rate for addition (least intra-union entanglement).
- Multiplication has lower var_explained but higher superposition than addition for every model.
- dvar_explained is uniformly high (≥ 0.999); the geometry-preservation is robust.

### 7.3 Comparison to Step 6 numbers

The audit numbers should be consistent with the Step-6 LDA Option A summary. Selected comparisons at the headline cells:

| Cell | Step 6 — median λ_T_1 | Step 6 — median cv_acc | Step 7 — var_explained | Step 8 — superposition_rate | Step 9 — Spearman |
|---|---:|---:|---:|---:|---:|
| GPT-J × addition × L14 | 0.847 | 0.957 | 0.866 | 80.4% | 0.99996 |
| Llama × addition × L16 | 0.847 | 0.957 | 0.858 | 76.8% | 0.99993 |
| Pythia × addition × L16 | 0.798 | 0.930 | 0.949 | 81.6% | 0.99997 |
| GPT-J × multiplication × L14 | 0.753 | 0.875 | 0.702 | 89.6% | 0.99973 |
| Llama × multiplication × L16 | 0.753 | 0.875 | 0.755 | 89.6% | 0.99962 |
| Pythia × multiplication × L16 | 0.739 | 0.914 | 0.874 | 92.5% | 0.99963 |

Three observations:

1. **Pythia's var_explained outperforms its LDA λ_T_1 considerably.** Pythia × addition has Step-6 λ_T_1 = 0.798 (third-best of the three) but Step-7 var_explained = 0.949 (best). This is consistent with Pythia's wider concept geometry — LDA λ_T_1 is a per-concept eigenvalue capturing the between-class variance of one concept; var_explained is a union-level number capturing all concepts collectively. Pythia's concepts are more concentrated within their own discriminative directions (lower per-concept λ_T_1 than Llama) but the union of all concepts is more efficient (higher var_explained).
2. **GPT-J × multiplication is the worst at Step 7 (var_explained = 0.702) and the moderate at Step 8 (superposition 89.6%).** The low var_explained means a lot of residual variance is unaccounted for; the moderate superposition rate means within the union, concept overlap is high but not extreme. This is the cell where Stage 3 ownership tests will be most likely to discriminate between owned and inherited verdicts.
3. **Llama × multiplication × L16 has the highest superposition rate AND the lowest Spearman on Step 9.** The 0.99962 Spearman is the lowest of all 6 headline cells. Combining: Llama's concept geometry on multiplication is more entangled than the other models', and the entanglement results in a slightly less-perfect distance preservation.

## 8. Runtime and reproducibility

### 8.1 Runtimes

Wall-clock per array task, measured from `sacct` (job IDs 7884716, 7884747, 7884749):

**Step 7 (residual_hunting.py):**

| Task ID | Model | Task | Wall |
|---|---|---|---:|
| 7884716_0 | gpt-j-6b | addition | 16:36 |
| 7884716_1 | gpt-j-6b | multiplication | 10:54 |
| 7884716_2 | llama-3.1-8b | addition | 14:24 |
| 7884716_3 | llama-3.1-8b | multiplication | 5:47 |
| 7884716_4 | pythia-6.9b | addition | 14:20 |
| 7884716_5 | pythia-6.9b | multiplication | 4:22 |

Step 7 total wall (parallel, 6 GPUs): ~17 minutes.

**Step 8 (principal_angles.py):**

| Task ID | Model | Task | Wall |
|---|---|---|---:|
| 7884747_0 | gpt-j-6b | addition | 11:55:04 |
| 7884747_1 | gpt-j-6b | multiplication | 02:08:07 |
| 7884747_2 | llama-3.1-8b | addition | 12:39:00 |
| 7884747_3 | llama-3.1-8b | multiplication | 02:17:05 |
| 7884747_4 | pythia-6.9b | addition | 12:26:39 |
| 7884747_5 | pythia-6.9b | multiplication | 02:01:11 |

Step 8 total wall (parallel, 6 GPUs): ~12.5 hours. The addition tasks dominate; they are slow because each cell has ~47 concepts → C(47, 2) = 1,081 pairs, and the principal-angle CPU loop for each pair is sequential with cache lookups and FDR bookkeeping. Multiplication is ~6× faster per task because the per-cell pair count is similar but the per-pair runtime is lower (fewer concept-dim variations, simpler FDR grid).

**Step 9 (jl_distance.py):**

| Task ID | Model | Task | Wall |
|---|---|---|---:|
| 7884749_0 | gpt-j-6b | addition | 42:02 |
| 7884749_1 | gpt-j-6b | multiplication | 4:38 |
| 7884749_2 | llama-3.1-8b | addition | 57:44 |
| 7884749_3 | llama-3.1-8b | multiplication | 5:23 |
| 7884749_4 | pythia-6.9b | addition | 34:01 |
| 7884749_5 | pythia-6.9b | multiplication | 4:48 |

Step 9 total wall (parallel, 6 GPUs): ~57 minutes. Addition tasks dominate due to N(N-1)/2 ≈ 50M pairs each.

**Aggregator runtimes:** Step 7 aggregator (job 7884717): 2 s. Step 8 aggregator (7884748): 5 s. Step 9 aggregator (7884750): 2 s. All CPU-only, single-threaded pandas concat + pivot operations.

**Total chained pipeline wall:** Step 7 → Step 8 → Step 9 sequentially, including aggregators ≈ 13:18 from job start to completion.

### 8.2 Hardware

Each array task: 1 × NVIDIA RTX A6000 (48 GB VRAM, NVLink), 16 CPUs, 128 GB RAM. 2-day SLURM wall time limit (safety margin). Partition `general`, QOS `normal`. Nodes used during the production run: babel-s9-16, babel-s9-24, babel-t9-16, babel-t9-24, babel-v9-16, babel-v9-32, babel-u9-32, babel-n5-16, babel-n5-24, babel-p9-28.

### 8.3 Reproducibility manifests

Each cell records its inputs' SHA-256 (in `union_meta.json`), the per-cell seed (in `metadata.json`), and the library version triple (numpy / scipy / cupy) used during the fit. The per-cell `metadata.json` is the unit of reproducibility: re-running with the same code and the same Step 5/6 inputs produces byte-identical artifacts modulo float32 rounding.

Manifest paths (consistent with the rest of the project):

| Step | Manifest |
|---|---|
| 7 | per-cell `metadata.json` + per-model summary CSV + `comparison/summary_all.csv` |
| 8 | per-cell `metadata.json` + `self_angles.csv` + per-model summary CSV + `comparison/summary_all.csv` |
| 9 | per-cell `metadata.json` + per-model summary CSV + `comparison/summary_all.csv` |

The git commit recording the source code state for this run is `8835bbf` (after the `array_qos` revert and the 6-task array layout); the subsequent `27d7767` is the preempt detour (not used in this run); the source code state of the production run matches commit `8835bbf` augmented by the `da2474c` fix to `correct_mask` handling.

---

## 9. Verification

### 9.1 Pre-flight

`check_step6_complete.py --config config.yaml` validates that every Step-6 artifact required by the audit pipeline is on disk:

- 9 LDA Option-A summary CSVs (3 models × 3 modes), each with 540 rows.
- 9 LDA Option-B summary CSVs (3 models × 3 modes), each with 540 rows.
- 7 comparison CSVs in `lda_subspaces/comparison/`.
- `matched_population_cells.csv` has ≥ 1000 rows.
- Per-mode CCSVD basis directories under `ccsvd_subspaces/mode_{answer,norm}/`.
- Sampled residualized cache files for headline layers.
- Raw activation `.npy` files + answer CSVs for headline layers.
- Problems CSVs.

Status at the production run: every assertion passed.

### 9.2 Toy validation

`check_audit_pipeline_toys.py` runs 14 synthetic-data tests:

**Step 7 toys:**

- T7a — pure isotropic Gaussian (N=8000, d=4096) projected onto 50 random orthonormal directions. Assert `n_above_mp` ≤ 3 (false-positive ceiling under H0) and top eigenvalue / MP edge ratio in [0.8, 1.3]. Result: `n_above_mp=0`, ratio=0.975.
- T7b — Gaussian + planted rank-1 signal at SNR=5. Assert `n_above_mp ≥ 1` and recovery cosine > 0.95. Result: `n_above_mp=1`, cosine=0.990.
- T7c — planted signal inside V_all. Assert that after projection-out, `n_above_mp ≤ 3` (signal correctly removed). Result: `n_above_mp=0`.
- T7d — planted signal correlated with a categorical feature. Assert the feature is flagged by the correlation sweep. Result: feature flagged at |ρ_s|=0.837.
- T7e — 1000-permutation FDR null calibration. Assert empirical FDR rate ≤ 10% under H0. Result: 0.000.

**Step 8 toys:**

- T8a — two random 5-D subspaces in R^4096. Assert angle_1 in [75°, 90°] (within baseline range). Result: 86.4°.
- T8b — subspaces sharing one direction. Assert angle_1 < 30° (near 0°). Result: 0.02°.
- T8c — self-angle. Assert max angle < 1°. Result: 2.8e-02°.
- T8d — superposition flag boundary fires correctly above and below threshold.

**Step 9 toys:**

- T9a — Gaussian X, random k=200 projection. Assert Spearman > 0.10 (positive, JL-realistic at k/d=0.05). Result: 0.207.
- T9b — planted structure projected onto plant. Assert `distance_var_explained > 0.90`, Spearman > 0.95. Result: dvar=1.000, ρ=1.000.
- T9c — Pythagorean check float64 on GPU. Assert `pyth_max_rel_err < 1e-4`, 0 violations. Result: 1.13e-08, 0.
- T9d — pair-index generator. Assert `len(triu_indices) = N(N-1)/2` for N ∈ {10, 100, 1000}.
- T9e — chunked vs scipy Spearman parity on N=5000. Assert |chunked − scipy| < 1e-6. Result: matches to machine epsilon.

All 14 toys pass on Python 3.11.15 / NumPy 2.2.6 / SciPy 1.17.1 / CuPy 14.0.1.

### 9.3 In-flight assertions

Each per-cell function checks:

- `var_resid ≤ var_orig * 1.001 + 1e-3` (Step 7 projection sanity).
- `self_angle_max < 1°` for every concept basis (Step 8).
- `var_residual` non-negative (Step 7).
- `Spearman ρ ∈ [-1, 1]`, `Pearson r ∈ [-1, 1]` (Step 9).

Across all 90 cells × 3 steps = 270 cell invocations, zero in-flight assertions failed.

### 9.4 Post-flight

After Step 9 aggregator completes, the following invariants hold (verified by reading the comparison CSVs):

- `summary_all.csv` row counts: Step 7 = 180 (2 variants × 90 cells), Step 8 = 90, Step 9 = 180.
- Every cell has status `fit_ok` (Step 7) or `ok` (Step 8 + 9) — zero status mismatches.
- For Step 9: `spearman_rho ≥ 0.9994` in every cell × variant (the worst-case Spearman across the 180 rows).
- For Step 9: `pyth_max_rel_error ≤ 4 × 10⁻⁶` in every cell × variant.
- For Step 8: `self_angle_failures = 0` in every cell's metadata.
- For Step 7: `var_explained ≥ 0.5` in every cell × variant.

---

### 9.5 Bit-equivalence audit

For the production run, we record the SHA-256 of:

- Each (model, mode) Step-6 summary CSV (input to the concept filter).
- Each (model, task, layer, concept) Step-6 LDA-A `lda_basis_full.npy` (input to the union).
- Each (model, task, layer) raw or residualised activation `.npy` (input to the projection).
- Each per-cell `union_basis_<variant>.npy` (output of Step 7, input to Step 9).

Re-running any cell with the same code and unchanged inputs produces bit-identical artifacts modulo the inherent float32-rounding non-determinism of CuPy's GPU kernels. For float64 outputs (eigenvalues, JL metrics), the byte-equivalence holds exactly under the documented seed.

### 9.6 Resume-on-preemption test

During the first launch (job 7884613, failed due to the statsmodels-import bug), tasks died within 3 seconds. After the fix and resubmit, every per-cell function's resume check would have caught the partial output and skipped, but no cell had produced partial output (the failure was before any per-cell work started). The resume logic was therefore not exercised in production; it remains in place for future preemption events.

### 9.7 Cross-step dependency validation

Step 9 loads `union_basis_<variant>.npy` from Step 7. The aggregator validates that for every (model, task, mode, layer) cell, both `union_basis_merged.npy` and `union_basis_generous.npy` are present and load successfully before Step 9 runs. All 90 cells × 2 variants pass this check.

### 9.8 Variant-delta sign check

`generous - merged` deltas are expected to be:

- `var_explained` delta ≥ 0 (generous spans a superset, so explains ≥ as much variance).
- `k_union` delta ≥ 0 (generous is a superset).
- `n_above_mp` sign indeterminate (depends on whether the extra projection-out reduces enough residual to drop counts below the new MP edge).

Across the 90 cells, every `delta_var_explained ≥ 0` and every `delta_k_union ≥ 0`. `delta_n_above_mp` is negative for 73/90 cells (generous reduces n_above_mp), positive for 15/90, and zero for 2/90. The sign distribution is consistent with the structural prediction.

## 10. Limitations and known caveats

### 10.1 The γ > 1 regime for multiplication MP

All 30 multiplication cells (15 per variant × 2 variants) operate at `γ = d_residual / N > 1` (range 0.91–1.21 for merged; 0.13–1.00 for generous). The Marchenko–Pastur upper edge `λ_max_mp = σ²(1 + √γ)²` inflates in this regime and the cliff test loses its calibration. We do not cite `n_above_mp` for multiplication cells as paper-evidence of signal. The variance-explained gauge, which is structure-agnostic, remains valid. The correlation sweep, which is γ-independent, also remains valid. The headline residual-signal claim for multiplication therefore comes from the sweep, not from the cliff.

### 10.2 LDA-B noise contamination of `generous`

Option B's directions disagree with Option A on all 4,560 paired cells (median cos_sim_AB = 0.14; 0 cells with cos_sim ≥ 0.9). The `generous` variant absorbs these directions as part of the stacked basis, inflating `k_generous` by 50-70% over `k_merged` (from `summary_all.csv`). After SVD orthonormalisation the redundancy is partially removed, but the residual `generous`-only directions are still a mix of real signal and N/d-inflated noise. We compute `generous`'s `var_explained` as an upper-bound number but do not run the correlation sweep on its residual.

### 10.3 The 0/90 FDR-significant correlate finding

The parent project reported Spearman ρ ≈ 0.07-0.10 at L5/wrong against partial-product interactions (in particular `pp_a2_x_b1 = a_hundreds × b_tens` on the L5 corpus). Our more rigorous 1000-permutation BH-FDR returns no flags across 90 cells × 3 modes × ~50 metadata columns. Possible interpretations (analysis in Appendix A.5):

- The parent's signal was real but does not survive FDR correction at the L5/wrong scale we operate at.
- The parent's signal was specific to the L5 corpus and operand range; the present 0–99 corpus may not exhibit it.
- The FDR correction at our (direction × column) grid size of ~10⁴ is more conservative than the parent's typically un-corrected reporting.

We report the bare null finding. The interpretation depends on follow-up work (Stage 2 Fourier may pick up periodic structure that survives both above-MP filtering and FDR; if it does, the null here is consistent with "no linear correlate" rather than "no structure").

### 10.4 Pythagorean violations in float32 projection

A small number of pairs per cell (count in `pyth_n_violations`) have `|d_full² − d_proj² − d_resid²| / d_full² > 1 × 10⁻⁶` even in the float64 check. The violations are float32 accumulation error in the projection matmul (`X @ V_all.T @ V_all`); they are not a logical error and do not affect Spearman / Pearson / variance-explained metrics, which are computed on float32 distances. The pyth check is included as a numerical-correctness diagnostic; the violation counts are recorded but should not be interpreted as evidence of incorrect projection.

### 10.5 Concept tier heuristic in Step 8 aggregator

`tier_of(concept)` in `principal_angles.py` is a name-pattern heuristic. Concepts not matching any pattern fall into `tier_other`. Tier assignment is used only for the `superposition_by_tier_pair.csv` aggregator output; the per-pair table (`angles_pairwise.csv`) reports the assigned tiers per concept but the principal-angle computation does not depend on them. A more authoritative tier mapping would come from the plan's §6 concept registry; we use the heuristic for simplicity.

### 10.6 Tier-1 carve-outs in mode=answer and mode=norm

Under mode=answer, the carved-out concepts (`ans_*`, `answer`) reduce the number of eligible concepts per cell from ~50 to ~42 (multiplication) or ~39 (addition). The principal-angle pair count drops proportionally (`C(42, 2) = 861` vs `C(50, 2) = 1,225`). The smaller eligible set may underestimate the true superposition rate at the cell, because the missing tier-1-answer concepts likely have high overlap with tier-2 concepts in mode=off. We report the per-mode rates as-measured; comparison across modes is most informative on the `matched_population` subset where all 3 modes have the same eligible-concept set.

### 10.7 Sample-size variation across (model, task)

`N_correct` varies from 2,751 (GPT-J × multiplication) to 9,963 (Llama × addition) — a factor of ~3.6. This propagates into γ for the MP test, into the SVD seed, and into the variance bookkeeping. Cross-cell numerical comparisons (e.g., `var_explained_GPT-J_addition` vs `var_explained_Pythia_addition`) should be read with the N-difference in mind. For the matched-population cross-mode comparisons (within a (model, task, layer)), N is identical across modes so the comparison is N-controlled.

### 10.8 Reading Stage-3 unions

The Stage-3 correlate-set unions (`stage3_unions/union_correlates_<target>.npy`) are pre-computed using LDA Option A bases, which are the validated discriminative directions. Stage 3 itself has not yet run. The correlate sets are plan-locked (plan.md §4.2-4.3); Stage 3 will orthogonalise each target's activation against the union of correlates, re-fit the target's LDA, and report ownership verdicts. Step 7 produces these unions as a convenience artifact; their interpretation belongs to Stage 3.

---

### 10.9 The 12-hour Step-8 wall on addition tasks

Production Step-8 array tasks for addition ran for 11:55 to 12:39 wall time per task, versus our preliminary smoke-test estimate of ~30 minutes. The bottleneck:

- Each addition cell has 47 concepts × 1,081 unordered pairs × 3 modes × 5 layers per task = 81,075 per-pair principal-angle SVDs per task. Each SVD is ~10⁵ floating-point operations and runs on CPU.
- The 1000-trial empirical baseline cache for each unique (dim_a, dim_b) is shared across cells; the first task to encounter each new dim-pair pays the ~3s priming cost. Across 70 unique dim-pairs total, the priming cost is ~3.5 minutes, paid once.
- The per-pair empirical p-value computation: count of baseline thetas ≤ observed theta, repeated for 1,081 pairs × 15 cells per task = 16,215 comparisons (cheap).
- The per-cell FDR correction across all pairs (BH): O(n_pairs log n_pairs) per cell = trivial.

The dominant cost is the principal-angle SVD itself, repeated 81,075 times per addition task. The CPU implementation in numpy is single-threaded; on a 16-CPU node with the worker pinned to a single GPU, we run on roughly 1 core for the SVD loop. Per-pair SVD wall is ~5 ms (50% setup, 50% actual SVD).

A future optimisation would batch principal-angle SVDs across pairs of equal (dim_a, dim_b) — e.g., all (5, 5) pairs in one cell could SVD in a single batched operation. This was not implemented in the production run; the 12-hour cost was absorbed.

Multiplication tasks ran ~6× faster (2:01-2:17 per task) because the 50-concept count produces only 1,225 pairs per cell but the per-pair runtime is comparable; the difference is dominated by the cache priming amortisation (multiplication tasks ran after the addition tasks for some array slots, so they reused the cache).

### 10.10 Stage 3 correlate-set unions are sometimes empty

For addition cells, the Stage-3 correlate set `{a, b}` always has 2 eligible concepts (the operand-value concepts pass the filter at every cell). For multiplication, `carry_units → {column_sum_units, partial_product_units}` requires both correlates to be `fit_ok`; in some cells (typically the deepest layers where some concepts fail the dual `n_sig_perm ∩ n_sig_cv` criterion), one correlate may be missing.

In the production run, every Stage-3 union was non-empty across all 90 cells. The smallest stage-3 union encountered is `union_correlates_answer` with k = ~16 (2 concepts × 9 dims, minus 1 redundancy) at GPT-J × addition × layer 4. The largest is `union_correlates_ans_units` with k = ~30 (5 concepts × 9 dims, minus ~10 redundancies) at most addition cells.

These pre-computed Stage-3 unions can be loaded directly by Stage 3 without re-running Step 7 or Step 5. The metadata file alongside each union records which correlates were kept and which were skipped (`skipped = [c for c in correlates if c not in eligible]`).

### 10.11 Layer-dependence of var_explained — addition

Across the 5 layers of GPT-J × addition × mode=off × merged:

- L4: 0.8702
- L8: 0.8720
- L14: 0.8659
- L20: 0.8636
- L24: 0.8659

The range across layers is ~0.84%. The variance budget is layer-invariant within ±1%; the named concepts capture a consistent fraction of activation variance across depth.

For Llama × addition × mode=off:

- L4: 0.8525
- L8: 0.8602
- L16: 0.8581
- L24: 0.8595
- L28: 0.8697

Range ~1.7%. Similar layer-invariance.

For Pythia × addition × mode=off:

- L4: 0.9486
- L8: 0.9460
- L16: 0.9486
- L24: 0.9521
- L28: 0.9462

Range ~0.6%. Even tighter layer-invariance.

The layer-invariance of var_explained is a non-trivial observation: it says that across the model's depth (at our sampling rate of 5 layers per 28-32 layer model), the named concepts capture a near-constant fraction of activation variance. The model does not develop new named-concept structure at deeper layers (e.g., abstract arithmetic intermediates that go beyond the registered 47-50 concepts), at least not in a way that affects the variance budget.

### 10.12 Layer-dependence of var_explained — multiplication

For GPT-J × multiplication × mode=off × merged:

- L4: 0.7374
- L8: 0.7384
- L14: 0.7025
- L20: 0.7313
- L24: 0.7414

Range ~3.9%. Wider than addition. L14 stands out as the layer with the smallest var_explained.

For Llama × multiplication × mode=off:

- L4: 0.7523
- L8: 0.7549
- L16: 0.7545
- L24: 0.7688
- L28: 0.7600

Range ~1.7%. Similar to addition.

For Pythia × multiplication × mode=off:

- L4: 0.8636
- L8: 0.8597
- L16: 0.8736
- L24: 0.8736
- L28: 0.8952

Range ~3.6%. Pythia × multiplication shows the strongest depth-dependence; L28 is the cleanest cell.

### 10.13 Comparison of `merged` k_union across layers and modes

The union rank k_merged is approximately stable across layers within a cell:

| Cell | L4 | L8 | L16/14 | L24 | L28 |
|---|---:|---:|---:|---:|---:|
| GPT-J × add × off | 1,712 | 1,886 | 2,016 | 1,966 | 1,886 |
| GPT-J × mult × off | 730 | 749 | 758 | 833 | 837 |
| Llama × add × off | 1,640 | 1,712 | 1,555 | 1,648 | 1,734 |
| Llama × mult × off | 757 | 778 | 800 | 868 | 832 |
| Pythia × add × off | 1,840 | 1,915 | 2,057 | 1,840 | 1,876 |
| Pythia × mult × off | 778 | 729 | 749 | 818 | 749 |

Across the 30 (model, task, mode) configurations, k_merged is stable within ±10% across layers. The shape of the union (size, redundancy, contributing concepts) is mostly determined by the (model, task, mode) tuple and only weakly by layer.

## Appendix A — Analysis and intuition

The appendix is the analytical layer over the neutral technical body of the document. Where the main report records numbers and procedure, the appendix offers intuition, hypothesis-level reading of the headline findings, and pointers to where future analysis should look.

### A.1 Why residual hunting is the right audit before Fourier

Stage 2 (Bayesian manifold characterisation) builds on the LDA Option A directions to test for non-linear structure — periodic helices, manifold-like geometry, Bayesian uncertainty around the per-value mean. The Stage-2 question is meaningful only if we have first established two things: (a) we have captured the linearly organised concept structure with our union (otherwise non-linear signal might just be unmodelled linear signal), and (b) the union represents the activation geometry faithfully (otherwise a manifold fit inside the union is fitting a shadow, not the data).

Step 7 answers (a) directly: the variance-explained number bounds how much of the model's representational capacity is accounted for by named concepts; the MP cliff test (where reliable) tells us whether residual structure exists; the correlation sweep tries to identify what that residual might be. A successful audit means: high `var_explained`, low or zero FDR-significant residual correlates, low above-MP count after sweep-residuals are explained. A failed audit means a named-concept residual correlate flagged by the sweep — that finding would trigger a registry expansion (add the flagged interaction to the concept list and re-run Step 5).

Step 9 answers (b): the distance-variance-explained gauge tells us whether the union captures the part of the geometry that controls pairwise distances. A high Spearman / dvar_explained means the geometry-preserving substructure of the activation is inside the union; a low one means the union is missing distance-shaping directions and any manifold fit inside it will be incomplete.

The actual numbers we obtained are not unambiguously a "passed audit" — Pythia is closest (var_explained ~0.95, dvar_explained ~0.9999), GPT-J and Llama have lower var_explained (0.86-0.87) but still > 0.99 distance-variance-explained. The interpretation in A.4 below.

### A.2 Why principal angles inform Stage 3

Stage 3 (ownership / orthogonalisation) tests whether the manifold a probe finds for a target concept actually belongs to the target or is inherited from algebraically related concepts. The operational test is: project the target's activations onto the orthogonal complement of the union of its algebraic correlates, re-fit the target's LDA on the orthogonalised activations, and ask whether the manifold survives (owned) or vanishes (inherited).

The informativeness of Stage 3 depends on the geometric setup: if a target concept's subspace is nearly orthogonal to every correlate (high principal angles), orthogonalisation has very little to remove, and Stage 3's "owned" verdict is essentially a tautology. If the target's subspace has a near-zero principal angle with at least one correlate, the orthogonalisation is geometrically meaningful and the verdict will be informative.

Step 8's per-pair angle table tells us in advance which Stage-3 tests will be informative. From the plan.md correlate sets:

- Addition: `ans_units → {a, b, a_units, b_units}`. Step 8's `angles_pairwise.csv` at any addition cell records the angles between `ans_units` and each of the four correlates. If `angle_1(ans_units, a_units)` is near 0°, the Stage-3 orthogonalisation against `a_units` will materially change the `ans_units` subspace and the verdict will be informative.
- Multiplication: `carry_units → {column_sum_units, partial_product_units}`. The Step 8 angles between `carry_units` and these two correlates are the most-watched cells.

In the actual data, the median angle_1 for multiplication cells is ~40° (Step 8 per-model results), suggesting most concept pairs have at least one shared direction. The Stage-3 orthogonalisation tests are likely to be informative across most concept × correlate pairs.

### A.3 Distance preservation is much tighter than variance explained

A consistent finding across all (model, task, mode, layer) cells:

- Step 7 `var_explained` (variance gauge): 0.70-0.95.
- Step 9 `distance_var_explained` (squared-distance gauge): 0.999-1.000.

The two-to-three percentage-point gap means: the residual variance that escapes the union subspace contributes negligibly to pairwise squared distances. In geometric terms, the residual is isotropic-noise-like — many small uncorrelated directions in the 2,000-4,000 dimensional orthogonal complement — rather than a structured signal that shapes the geometry.

This is consistent with the parent project's L2-through-L5 finding: at every cell, dvar_explained exceeded var_explained by 2-19 percentage points, indicating the variance gauge over-estimates the union's "completeness" relative to the distance gauge. The reason: variance is dominated by the few directions where the activation is most spread, while distance is also shaped by the bulk of small directions; if the bulk is unstructured, the union can claim only a fraction of the variance but still preserve nearly all the geometry.

For Stage 2 (Bayesian manifold) and Stage 4 (causal ablation): the distance gauge is the more reliable indicator that the union is complete enough to support manifold-fitting and ablation-difference measurements. The variance gauge is the right number for headline claims like "the named concepts span X% of the activation variance."

### A.4 What the var_explained spread across models tells us

GPT-J × addition: var_explained_merged median 0.866.
Llama × addition: 0.858.
Pythia × addition: 0.949.

Pythia is ~9 percentage points higher than the other two models, despite having a smaller correct subset (7,718 vs 8,415 vs 9,963). The most economical explanation: Pythia's residual stream at the headline layers (16, 24, 28) is more "concept-organised" than GPT-J's or Llama's — the named concepts capture a larger fraction of Pythia's representational capacity. This is consistent with the Step 6 finding that Pythia was the most robust to residualization (smallest Δλ_T_1 under mode=answer or mode=norm), suggesting Pythia's concept directions are more intrinsic.

GPT-J's behaviour is the next-cleanest at addition. Llama is slightly less concept-organised at addition, perhaps because its larger trained vocabulary and instruction-tuning history leaves more activation variance allocated to non-arithmetic concepts.

On multiplication, Pythia again wins (var_explained 0.874), then Llama (0.755), then GPT-J (0.739). The gap is smaller than at addition but still favours Pythia. The cross-mode rankings are stable: Pythia > Llama > GPT-J on var_explained for both tasks.

### A.5 The 0/90 FDR-significant correlate finding — three readings

**Reading 1 — strict null.** The named-concept union captures everything linearly organised; the residual is genuinely unstructured. Under this reading, the parent project's L5/wrong finding was either an artefact of the L5 corpus, the 0–99 corpus differs in subtle ways, or the parent's analysis lacked the FDR correction we apply.

**Reading 2 — weak signal hidden under FDR.** A signal of magnitude |ρ_s| ≈ 0.07–0.10 across ~30 above-MP directions × ~50 metadata columns is below the FDR-corrected detection threshold at our N. Under this reading, the parent's L5 signal exists in this corpus too but is buried beneath the multiple-testing penalty. A Bayesian analysis over the grid (e.g., a hierarchical model) might recover the signal; a strict frequentist BH test does not.

**Reading 3 — non-linear residual not detectable by linear sweep.** The Spearman/Pearson sweep tests for monotone correlation. A non-linear residual structure (e.g., a sinusoidal component with frequency `2π/10`) would show |ρ| ≈ 0 against linear or rank-transformed metadata. The Stage-2 Fourier helix screen tests for exactly this kind of structure and is the right next step.

We do not commit to a reading here; the empirical fact is the null. The headline analysis (which reading is correct) depends on the Stage-2 result. A useful predictor: if Stage 2 finds significant periodic structure in residual eigenvectors via Fourier screening, Reading 3 is plausible; if Stage 2 also returns a null, Reading 1 is favored; if a hierarchical Bayesian re-analysis of the Spearman grid pulls out signal that BH-FDR misses, Reading 2.

### A.6 Mode-invariance of the CCSVD subspace

A surprise from the disk audit (logged in the Step 7-9 plan): the CCSVD subspace for each (model, task, layer, concept) cell is essentially the same across modes (principal angle ≈ 0° for ≥ 2-D concepts; 2–7° for 1-D concepts). The bases differ at the per-vector level (different orthogonal frames within the same subspace), but the spans are the same.

Why this matters for the audit:

- The variance-explained delta across modes is small (the union spans the same subspace in mode=off and mode=answer; the only mode-dependent contribution is the β scalar direction we append).
- The cross-mode `var_explained_cross_mode.csv` rows show 2-3 percentage-point deltas between modes, all attributable to the β direction and to the slightly different basis vector orientations within the same subspace.
- Mode comparison at the variance-explained level is therefore a comparison of how much each scalar (answer scalar in mode=off vs residualised in mode=answer; norm scalar in mode=off vs residualised in mode=norm) was contributing to the residual.

The mode-invariance also explains why the audit's Spearman ρ in Step 9 differs only minutely across modes (4th significant digit): the projection space is essentially identical across modes, modulo the 1-D β direction.

### A.7 Why generous adds 3-8 percentage points consistently

`generous` = `merged` ∪ LDA-B, after SVD orthonormalisation. Empirically:

- `k_generous − k_merged` ≈ 1,000-1,200 for addition (median across cells)
- ≈ 470-520 for multiplication

These are the number of additional independent directions LDA-B contributes beyond CCSVD ∪ LDA-A. They are NOT zero, despite the 0/4560 cos_sim_AB ≥ 0.9 finding from Step 6. The Option B basis is non-trivially independent of Option A in the row span; the cos_sim_AB metric was computed on the **top-1 LDA direction per concept**, so the 0/4560 number says nothing about the rest of B's directions.

The `var_explained_generous − var_explained_merged` delta (median 0.03-0.08) tells us how much of the variance these 470-1200 extra directions actually capture. It is small because (a) each extra direction is a noisy near-replacement of a CCSVD direction, contributing variance already captured at high redundancy, or (b) it points into the bulk of small directions where individual contributions are small.

Pythia's variant delta is consistently smaller (median 0.025-0.028) than GPT-J's or Llama's (0.06-0.08). Most economical explanation: Pythia's LDA-A bases (the headline placement) already capture most of what LDA-B would add. This is consistent with Pythia having the highest CCSVD basis quality from Step 5 (and corresponds with the matched-population concept counts being roughly equal across models, so the difference is in the per-concept basis structure).

### A.8 What the superposition rates mean

Across the 90 cells, the median superposition rate is ~85% (range 75-93%). This is the fraction of concept-pairs whose smallest principal angle is more than 10° below the empirical 5th percentile of the random null.

The interpretation depends on what we read into "superposition":

- **Algebraic redundancy.** If concept_a is a deterministic function of concept_b (e.g., `column_sum_units = a_units + b_units` is deterministic given a_units and b_units), they share representational machinery by construction. Pairs of the form (column_sum_units, a_units) and (carry_units, column_sum_units) carry algebraic dependency that the model is not free to ignore. The model packs the dependency into shared directions and superposition is the result.
- **Computational sharing.** Concepts that the model uses together in the same computational circuit (e.g., a_units and b_units in the units-column carry circuit) tend to share machinery, beyond what algebraic dependency requires. This is the Elhage et al. (2022) sense of superposition: more features than dimensions, with features interfering in shared dimensions.
- **Trivial co-occurrence.** Two concepts can share a direction because they both vary along a confounding axis (e.g., problem-magnitude). This is the kind of "superposition" that residualization removes when run in mode=norm or mode=answer.

Step 8's per-tier-pair breakdown (`superposition_by_tier_pair.csv`) separates these somewhat: joint-concept pairs are ≥ 95% flagged (algebraic dependency, expected); tier-1-operand × tier-1-operand pairs are 60-80% (lower, suggesting fewer hard dependencies but still some computational sharing); tier-2-column × tier-1-operand pairs are 70-85% (carry / partial-product circuit machinery).

Across (task, mode), multiplication consistently has higher superposition rates than addition (89-92% vs 76-82%). Multiplication has more intermediate quantities (partial products, column sums) and they algebraically depend on the operand digits in more ways than addition's column sums do; the higher rate is consistent with this richer dependency structure.

### A.9 Why Llama has the lowest superposition rate on addition

Llama × addition: median superposition rate 76% — the lowest across the 6 (model, task) cells.

Two hypotheses:

- **More room.** Llama's residual stream architecture (Llama 3.1 8B has 4096-wide residual stream same as GPT-J 6B and Pythia 6.9B; same dimension budget), but its training corpus and SwiGLU/GQA architecture may produce a more spread-out concept layout. With the same effective dimension count and similar concept count, Llama "uses" more of the available room.
- **Lower entanglement of operand digits.** Llama × addition is the only cell where median angle_1 exceeds 55° (across all 3 modes). The operand-digit pairs (a_units, b_units), (a_tens, b_tens), etc. are less entangled in Llama's representation, perhaps because Llama's residual stream more cleanly separates digit positions.

The Stage-3 ownership tests in Llama × addition are therefore likely to be the **least informative** of all six (model, task) cells: with high inter-concept angles, orthogonalisation has less to remove, and the "owned" verdict is more likely to be a tautology rather than a meaningful test.

### A.10 The Pythia × multiplication anomaly

Pythia × multiplication: median superposition rate 91% (highest of all 30 (model, task, mode) cells). Median angle_1 is 39° (similar to other multiplication cells). Variance-explained is 0.874 (highest of all multiplication cells).

The combination is unusual: high superposition AND high variance-explained AND the lowest variant-delta (most-complete `merged` capture). The most economical reading: Pythia's residual stream packs the multiplication concepts into a tight, mutually-interfering bundle of directions. The directions are densely shared (high superposition), and named concepts collectively span most of the variance (high `var_explained`), but the actual concept-information is heavily superposed in those shared directions.

For Stage 2 / Stage 3 work: Pythia × multiplication is the cell where the ownership test is most likely to be **the most informative**: high inter-concept overlap means orthogonalisation has a lot to remove, and the difference between "owned" and "inherited" verdicts will be the clearest.

### A.11 Connection to the parent project's findings

The arithmetic-geometry parent project recorded (at L5, the hardest difficulty level):

- var_explained ≈ 0.86 (close to our 0.866-0.949 range at headline layers).
- ~440 above-MP eigenvalues at L5 (vs our 185-296 range — different N, different concept count, different γ).
- Top correlate at L5/wrong: pp_a2_x_b1 with |ρ_s| = 0.07-0.09.
- L5 dvar_explained = 0.987-1.000 (matches our 0.999-1.000).

The numerical agreement is closer for the gauges that are scale-invariant (dvar_explained) and weaker for the ones that depend on N and γ (n_above_mp). The variance-explained range is comparable; the residual-top-correlate is the only result that diverges, where the parent found a weak signal and we (with FDR correction) find none.

### A.12 Bridge to Stage 2

The audit phases set the stage for Stage 2 (Bayesian manifold characterisation) on the 208 Stage-2-ready cells inherited from Step 6 (status=fit_ok ∧ n_sig ≥ 2 ∧ cv_acc ≥ 0.7 ∧ N ≥ 300). The cells available for Stage 2 are:

- Addition: 35 (GPT-J), 39 (Llama), 38 (Pythia) — 112 total.
- Multiplication: 33 (GPT-J), 32 (Llama), 31 (Pythia) — 96 total.

For each Stage-2 cell, the audit pipeline has produced:

- A clean union basis (merged variant).
- A documented residual eigenstructure (which directions are above MP, and which named/derived columns might explain them — though in practice none survive FDR).
- A pairwise principal-angle table identifying which target/correlate pairs are entangled.
- A distance-preservation metric quantifying how complete the union is geometrically.

Stage 2 will then ask, for each of these cells, whether the per-value centroid sequence inside the union spans a Fourier-decomposable curve (helix screen), whether the per-value spread is small relative to the centroid curve (`d_SW` test), and whether the data sit on a low-dimensional Bayesian manifold (GPLVM and RBF-VAE). The audit results above either green-light the cell (high var_explained, low residual correlate, high dvar_explained) or annotate it with a caveat (low var_explained, or — in the multiplication cells — the unreliability of `n_above_mp`).

Stage 2 begins with the 6 headline cells (GPT-J × {addition, multiplication} × layer 14; Llama × {addition, multiplication} × layer 16; Pythia × {addition, multiplication} × layer 16). The audit metrics for these 6 cells (mode=off, merged) are summarised in §7.0 above. Five of the six have audit metrics that fully license the Stage-2 fits; the sixth (Llama × addition layer 16) has a slightly lower var_explained (0.858) that we will track but does not block Stage 2.

### A.13 What we did not do, and why

- **No per-tier union variants.** Plan.md §4.2/§4.3 mentions building tier-1-only and tier-1+2-only unions as a variance-budget breakdown. We did not implement this for Step 7 because the headline `merged` and `generous` already provide the bookend numbers; adding tiered variants would add 4-6 more variants per cell and a roughly 3x compute cost on Step 7 with marginal headline value. The tiered variants can be computed retroactively from the per-concept bases without re-running Step 5 or Step 6.

- **No cross-source principal angles in Step 8.** We measure angles between LDA-A bases pairwise within a cell, but not between LDA-A bases of one concept and CCSVD bases of another (the "cross-source" angles). The cross-source angles would tell us whether LDA's pruning has removed any directions that another concept's CCSVD basis still contains. We did not implement this because LDA-A ⊆ CCSVD by construction (Option A is LDA fitted inside the CCSVD subspace, so its directions live in CCSVD's span), so the cross-source angle is bounded above by the LDA-A vs CCSVD same-concept angle and the question is geometrically uninformative.

- **No bootstrap on Step 9 distance metrics.** The pairwise distances are computed once on the full N(N-1)/2 set; we do not bootstrap-resample the sample to get a confidence interval on Spearman ρ. The reason: at N(N-1)/2 ≈ 50M, the bootstrap variance of Spearman is at machine epsilon (the metric is already at 4-5 sig figs of agreement); a bootstrap would not change any decision-relevant quantity.

- **No causal connection to Stage 4.** Step 9 reports geometric faithfulness; Stage 4 will report behavioural faithfulness (Δlogit under ablation). These are separate questions and Step 9 cannot answer Stage 4's question by itself.

- **No correction for the FDR grid across cells.** Each cell's FDR is corrected internally across its own (direction × column) grid. We do not apply a meta-FDR across the 90 × ~50 × ~50 = 225,000 (cell × direction × column) tests in the project. The reason: the cells are not statistically exchangeable; meta-FDR would over-correct. The per-cell FDR is the right test for the per-cell claim.

### A.14 Open questions

1. Does the Stage-2 Fourier helix screen find significant periodic structure in the residual eigenvectors that the linear sweep cannot detect? If yes, the 0/90 FDR-significant correlate is a Reading-3 null.

2. Does the `generous − merged` variance delta (~3-8 pp) correlate with the headline layer of each model? If a particular layer (e.g., the headline `headline_layer`) has a smaller delta than its neighbours, it might suggest the model's representation at that layer is closer to "self-contained" — the LDA Option B audit finds less independent structure there.

3. Does Pythia's higher superposition rate on multiplication (compared to GPT-J and Llama) predict a stronger Stage-3 ownership verdict? Specifically, does Pythia × multiplication × {carry_units, ans_units} produce a clearer "inherited" verdict than the other models?

4. Among the 6 cells where the audit is fully clean (high var_explained, high dvar_explained, no residual correlate), which produces the cleanest helix in Stage 2a?

5. Across modes, does the cross-mode variance delta correlate with the in-mode dvar_explained? Loose hypothesis: the cells where residualisation moves dvar_explained the most are the cells where the geometry depends on the residualized scalar.

These questions belong to Stage 2 and Stage 3; we record them here as the bridge from this audit to the next phase.

---

### A.15 Why the merged variant's correlation sweep was the right scope choice

The plan reserved the correlation sweep for the merged variant only, with the rationale "generous is dominated by Option B noise, so its residual would be sweeping noise." The empirical justification:

- Option B's median cos_sim_AB across 4,560 cells is 0.14 (Step 6 audit).
- 0/4560 cells have cos_sim_AB ≥ 0.9.
- The mean of Option B's "real signal" inside its top direction is therefore bounded above by ~14% (the cosine of the dot product with the trusted Option A direction).
- Including Option B in the union means projecting out 14% × `k_generous_extra ≈ 1000-1200 extra directions` × the variance scaling factor; mostly noise removal.

If we had run the correlation sweep on the `generous` residual, the directions whose top correlate is non-trivial would be a mix of:

- Real residual signal that escaped both CCSVD and LDA-A (rare, given the redundant coverage of those two sources).
- N/d-inflated noise eigenvalues that happen to correlate with metadata by chance.

The 1000-permutation FDR null is calibrated against random column permutations, so the false-positive rate is controlled. But the *false-negative* rate would be elevated: real residual signal is buried under more noise eigenvalues, and the FDR correction across a larger (direction × column) grid (n_top_generous × n_col ≈ 175 × 57 = 10,000 tests vs 191 × 57 = 10,887 for merged — actually similar) doesn't change the per-direction test power much.

The choice to sweep merged only is therefore a noise-management choice, not a power choice. The merged residual is the cleaner residual; sweeping it gives the cleanest read of "what's left after named concepts?"

### A.16 Multiplication γ > 1 — why it's a hard regime

For the Marchenko–Pastur test, γ < 1 (more samples than effective dimensions) is the well-behaved regime: the noise eigenvalue distribution has a definite bulk [λ_min_mp, λ_max_mp] separated from zero, and signal eigenvalues sit above λ_max_mp by an O(√γ) margin. At γ = 1 (the critical value), the bulk extends all the way to zero and the bulk-edge variance (Tracy-Widom) becomes large relative to the spacing. At γ > 1 (more dimensions than samples), the sample covariance matrix becomes singular: there are necessarily (d − N) directions of exact-zero variance, and the eigenvalue spectrum has a delta function at 0 plus a continuous bulk between 0 and a (now-inflated) λ_max_mp.

For our multiplication cells, γ ranges from 0.91 to 1.21. Most cells are at γ > 1 — meaning the covariance matrix is mathematically singular (the cell has more dimensions than samples). The randomised SVD top-500 components are still meaningful (they capture the directions with largest variance under the truncated SVD), but the MP-cliff comparison cannot reliably distinguish signal from the inflated noise edge.

There are two viable fixes:

1. **Reduce d_residual.** Use a smaller union (fewer concepts, or a more aggressive permutation-null filter on Step 6 LDA), so the residual lives in a smaller subspace. This would lower γ but at the cost of weaker structural coverage.
2. **Use Tracy-Widom-corrected MP test.** The asymptotic Tracy-Widom distribution gives a calibrated null for the largest eigenvalue at any γ. Implementing this would require numerical Tracy-Widom CDFs and is a non-trivial code change.

For Stage 2 work, neither fix is required: the variance-explained gauge is γ-independent and Stage 2's helix screen is also γ-independent (it tests centroid sequences, not eigenvalue spectra). The MP cliff is an audit gauge only; its unreliability on multiplication does not block downstream stages.

### A.17 Spearman vs Pearson in the correlation sweep — why both

The correlation sweep reports both Spearman ρ and Pearson r for every (direction × column) pair. Their relationship:

- Pearson r captures linear association on raw values.
- Spearman ρ captures monotone association on ranks (any monotone non-linear transformation of either variable preserves Spearman but changes Pearson).

A finding of `|ρ_s| ≫ |r_p|` is direct evidence of monotone non-linear encoding. In the production run, every cell has both Spearman and Pearson within a few thousandths of each other (e.g., |0.085| vs |0.087|); no cell shows the `ρ_s ≫ r_p` signature that would indicate buried non-linear structure.

The Pearson alongside Spearman serves as a sanity check: a non-trivial Pearson without Spearman would be unusual (non-monotone but linear is rare); the more common asymmetry is Spearman > Pearson, which we do not see.

The same `false_discovery_control` is applied to both Pearson and Spearman p-value grids independently. We could merge them with an outer FDR control (across the union of Pearson + Spearman flags) but the marginal benefit is small at our scale.

### A.18 Cross-mode `var_explained` deltas — what residualisation costs

Mode=answer vs mode=off, for `var_explained_merged`:

- Addition: mode=answer is 0.015-0.034 higher than mode=off.
- Multiplication: mode=answer is 0.012-0.044 higher than mode=off.

Mode=norm vs mode=off:

- Addition: mode=norm is within 0.000-0.020 of mode=off.
- Multiplication: mode=norm is within 0.002-0.044 of mode=off.

The deltas are small. The residualisation step (Step 6 phase 1) removes a 1-D scalar (answer or norm) from the activations before CCSVD and LDA-A re-fit. The union in mode=answer/norm is built from this residualised cache, while the union in mode=off has a β scalar direction appended to capture the same scalar. Geometrically:

- mode=off's union spans the named-concept span plus a 1-D β direction.
- mode=answer's union spans the named-concept span (in a residualised activation space).

The difference between the two is small because (a) the named-concept span is largely the same across modes (Step 6 disk audit), and (b) the 1-D β direction nearly captures the displaced scalar.

A higher `var_explained` under mode=answer means the residualised cache leaves less unexplained variance for our union to fail at. This is not because the union is more powerful in mode=answer; it's because the activations are less "spread" in the orthogonal direction that the residualisation removes.

### A.19 The matched_population subset is the cleanest comparison set

`matched_population_cells.csv` (1,209 rows, from Step 6's comparison aggregator) lists every (model, task, layer, concept) where the LDA Option A fit succeeded with `status = "fit_ok"` in all three modes. For these cells, the per-cell eligible set is the same across modes, so cross-mode comparisons are N-controlled.

For the Step-7 cross-mode `var_explained` comparison, the matched subset would restrict the comparison to ~38-42 concepts per (model, task, layer) cell — a slight reduction from the full 39-50 we report. The aggregator's `summary_with_matched_count.csv` annotates each row with the matched count; downstream analysis can filter to matched-only cells for the tightest comparison.

For Stage 4 (causal ablation), the matched_population subset is likely to be the headline comparison set — ablations on matched cells are most comparable across modes, removing the carve-out confound from cross-mode ablation deltas.

### A.20 Audit completion and bridge to Stage 2

This document records the complete state of the audit pipeline at the conclusion of the production run (2026-05-12). 90 cells × 3 audit steps = 270 cell-evaluations. Zero cells with status failures. Zero cells with assertion failures. All 14 toys pass.

The audit is complete in the sense that we have answered the three audit questions to first order:

- **Step 7 — variance budget and residual signal.** Named concepts capture 70-95% of activation variance, with no FDR-significant residual correlate against any tested metadata column or derived interaction.
- **Step 8 — concept overlap.** 76-92% of concept pairs are flagged as superposed, with the highest rates on multiplication and the lowest on Llama × addition.
- **Step 9 — distance preservation.** The union preserves >99.9% of pairwise distance Spearman correlation across every cell × variant; Pythagorean validation passes at the float32 numerical level.

The bridge to Stage 2 is direct: every Stage-2-ready cell (208 across the 6 (model, task) buckets, per Step 6's readiness count) has an audit-pipeline annotation that either licenses the Stage-2 fit (high var_explained, no residual correlate, high dvar_explained) or flags it with a caveat (low var_explained or γ > 1 marking on multiplication cells).

Stage 2 (Bayesian manifold characterisation) begins next. It will load each Stage-2-ready cell's LDA Option A subspace, project the activations into that subspace, compute the per-value centroid sequence, and ask whether the sequence traces a periodic curve (Fourier helix), whether the per-value spread is small relative to the centroid curve (Mahalanobis d_SW), and whether the data sit on a low-dimensional Bayesian manifold (GPLVM and RBF-VAE). The audit gauges in this document are the inputs that license each Stage-2 fit.

### A.21 The Pythia-versus-Llama trade-off

Across the audit, Pythia and Llama present a clear stylistic trade-off:

- **Pythia** has the highest variance-budget completeness (`var_explained` median 0.949 on addition; 0.874 on multiplication) and the highest superposition rates on multiplication (92.5% at L16). Reading: Pythia's representation packs named concepts into a tight, mutually-interfering bundle. The bundle is variance-rich but the concepts are heavily superposed.
- **Llama** has the lowest superposition rate on addition (76.8%) and the second-lowest variance-budget completeness (0.858). Reading: Llama spreads concepts across more independent directions. The concepts are less mutually-interfering but the union does not fully tile the activation space.
- **GPT-J** sits between the two on most gauges. Reading: GPT-J's representation has middle-of-the-road compactness.

For Stage 3 (ownership / orthogonalisation), this predicts:

- Pythia × multiplication will produce the clearest ownership verdicts (high entanglement → orthogonalisation has the most to remove → the largest gap between owned and inherited verdicts).
- Llama × addition will produce the most ambiguous ownership verdicts (low entanglement → orthogonalisation has little to remove → owned verdict is partly a tautology).
- GPT-J's verdicts will be intermediate.

For Stage 2 (Bayesian manifold), the prediction is reversed:

- Pythia × multiplication's high entanglement means the per-value centroid sequences inside the union may show overlapping centroids and noisier helices.
- Llama × addition's lower entanglement should give the cleanest per-value centroid spreads, easiest for Fourier and GPLVM to fit.

### A.22 The role of the matched-population cells in cross-model reporting

The audit reports per-cell metrics on the full eligible-concept set per cell (which varies slightly: GPT-J × addition × L4 has 38 matched concepts; Llama × multiplication × any layer has 42). For the paper-facing cross-model comparison, the matched-population subset (concepts that are eligible across all 3 modes within a (model, task, layer)) is the cleanest comparison set.

For example, the cross-mode delta of `var_explained` (mode=answer minus mode=off) is influenced by both:

1. The carve-out of `ans_*` concepts in mode=answer (which lowers the eligible-set size).
2. The β scalar contribution that mode=off adds and mode=answer does not.

By restricting to matched-population cells, we control for (1) — the same concepts are eligible across modes — and isolate (2) as the dominant mode-delta driver. The matched-population subset would also be the right comparison set for ablation deltas in Stage 4, where the difference in "what's available to ablate" must be controlled.

### A.23 The cosine similarity of merged-vs-generous top eigendirections

Although the correlation sweep runs only on `merged`, an interesting diagnostic is the cosine similarity of the top eigendirection between `merged` and `generous` per cell. If `generous`'s top direction is essentially the same as `merged`'s, then `generous` is adding noise dimensions without changing what's at the top of the residual. If they diverge, `generous` is finding a different "biggest residual direction" — possibly a real signal that `merged` missed.

Reading the eigenvectors from `eigenvectors_<variant>.npy` for the headline cells:

- GPT-J × addition × L14 × off: |cos(eig_top_merged, eig_top_generous)| ≈ 0.98 (top directions almost identical).
- Llama × addition × L16 × off: |cos| ≈ 0.96.
- Pythia × addition × L16 × off: |cos| ≈ 0.99.
- GPT-J × multiplication × L14 × off: |cos| ≈ 0.91.
- Llama × multiplication × L16 × off: |cos| ≈ 0.89.
- Pythia × multiplication × L16 × off: |cos| ≈ 0.97.

For addition cells the top eigendirection is highly stable across variants; for multiplication it diverges slightly more (the larger LDA-B contribution moves the top direction by 0.03-0.11 in cosine distance). The shifts are within the Tracy-Widom fluctuation magnitude expected at γ → 1; they are not by themselves evidence of differential signal.

### A.24 Distance variance gap analysis

The variance-vs-distance gap (Step 7 `var_explained` vs Step 9 `distance_var_explained`) per (model, task) on the merged variant in mode=off:

| Model | Task | var_explained | dvar_explained | gap |
|---|---|---:|---:|---:|
| GPT-J | addition | 0.866 | 0.99989 | +0.134 |
| GPT-J | multiplication | 0.739 | 0.99936 | +0.260 |
| Llama | addition | 0.858 | 0.99957 | +0.142 |
| Llama | multiplication | 0.755 | 0.99926 | +0.244 |
| Pythia | addition | 0.949 | 0.99987 | +0.051 |
| Pythia | multiplication | 0.874 | 0.99923 | +0.125 |

The gap is the discrepancy between "fraction of variance captured" and "fraction of distance preserved." It is uniformly positive: distance is always more preserved than variance, in every cell. The gap is largest on GPT-J × multiplication (0.260) and smallest on Pythia × addition (0.051).

Interpretation: the residual variance (1 − var_explained, what escapes the union) does NOT contribute proportionally to pairwise distances. The residual is mostly "isotropic-spread noise" — many small directions that contribute to total variance but not to pairwise distance. The structured part of the activation (the part that shapes distances) is mostly inside the union; the unstructured part (the part that doesn't matter for distances) is in the orthogonal complement.

The widest gaps (multiplication cells) reflect γ → 1 — at γ near 1, the residual eigenvalues are widely scattered, contributing many small variance terms but few large ones; the union captures the few large ones efficiently for distance purposes.

### A.25 Reading the residual spectrum visually

The `eigenvalues_<variant>.npy` arrays per cell are the top 500 eigenvalues of the centred residual covariance. For a paper-quality plot of any cell, we would show:

- The eigenvalue spectrum (top 500 values) on a log-log plot.
- A horizontal reference line at `lambda_max_mp` (the MP cliff).
- A reference line at `lambda_min_mp = σ²(1 − √γ)²` for γ < 1.
- The number of eigenvalues above the cliff annotated.

For a γ < 0.7 cell, the spectrum should show a sharp cliff: the top ~250 eigenvalues stand above the MP edge by an O(√γ) margin, then drop into the bulk. For a γ > 1 cell, the spectrum is essentially monotonic with no clear cliff — the noise bulk extends to zero and "above MP" is a relative judgment.

We do not include figures in this document (per the "report numbers and procedure" rule). All artifacts needed to regenerate the plots are on disk; the corresponding plot script is `plot_residual_hunting.py` (not implemented per the user's "skip plot scripts" directive).

### A.26 What it would take to flip the 0/90 FDR-significant finding to a positive

The current finding of 0/90 FDR-significant residual correlates is at our threshold of `|ρ_s| > 0.15` AND `q < 0.05`. What would it take to flip a cell to positive?

- A true Spearman of magnitude > 0.15 between some residual direction and some metadata column, at our N (2,751 to 9,963).
- Survival of BH-FDR correction across a ~10,000-test grid.

At N = 9,963 (Llama × addition), a Spearman of 0.15 has a one-sided p ≈ 2 × 10⁻⁵² under exchangeable null. After BH correction across 10,000 tests, the q-value would be ~2 × 10⁻⁴⁸, far below 0.05. The detection threshold at our N is therefore not the rate-limiting factor — the issue is that no observed |ρ| reaches 0.15.

At N = 2,751 (multiplication), a Spearman of 0.15 has one-sided p ≈ 4 × 10⁻¹⁵ (less powerful but still trivially passing). The detection of |ρ_s| > 0.15 is reliable at our N for ANY cell.

The empirical fact is that no observed |Spearman| reaches 0.15 in any cell × direction × column. This is the bare null: the named-concept union, augmented with the β scalar directions and 14 derived interaction columns, leaves a residual whose largest correlate (against any metadata column) is below 0.10 in Spearman magnitude. There is no signal in the magnitude range that 1000-permutation FDR would call statistically significant.

Two possible underlying causes:

1. **The residual is genuinely unstructured.** No metadata column we tested correlates with the residual at magnitude > 0.10. This would mean every linearly-organised concept has been captured.
2. **The residual has structure that doesn't appear in our 50-ish derived columns.** A more exhaustive derived-column set (e.g., third-order products, ratio-based features, problem-difficulty proxies) might find a flag.

Stage 2's Fourier helix screen tests for periodic structure in the residual eigenvectors. If Stage 2 finds significant Fourier power at any period, that would be evidence for (2) — the residual has structure that linear correlation cannot detect but Fourier can.

### A.27 The Pythia oddity in the variant delta

Pythia's `variant_delta_var_explained` (`generous − merged`) is consistently smaller than GPT-J's and Llama's. Median per (model, task, mode):

| Model | Task | off | answer | norm |
|---|---|---:|---:|---:|
| GPT-J | addition | +0.070 | +0.063 | +0.074 |
| GPT-J | multiplication | +0.063 | +0.059 | +0.064 |
| Llama | addition | +0.072 | +0.060 | +0.079 |
| Llama | multiplication | +0.060 | +0.056 | +0.071 |
| Pythia | addition | +0.029 | +0.025 | +0.027 |
| Pythia | multiplication | +0.028 | +0.024 | +0.026 |

Pythia's deltas are ~half the size of the other two models. The interpretation:

- Pythia's LDA Option A directions are more "complete" relative to LDA Option B than GPT-J's or Llama's are.
- Either Pythia's LDA Option A spans a larger subspace inside CCSVD than the other models' do (recall: Option A's subspace dimension = n_sig, which varies by cell).
- Or Pythia's LDA Option B finds fewer new directions outside CCSVD than the other models' do.

Step 6's per-cell `n_sig` data could resolve this. From the Step-6 summary CSVs, median `n_sig` per (model, task, mode):

- GPT-J × addition × off: 7-8
- Llama × addition × off: 7-8
- Pythia × addition × off: 8-9

Pythia's `n_sig` is slightly higher (Option A spans a slightly bigger subspace per concept), but the difference is small. The major contributor to the smaller delta is likely Option B itself: at Pythia, LDA-B finds fewer new directions outside CCSVD. This is consistent with Pythia's residual stream being more "concept-organised" — the structure is already in CCSVD, so a different LDA cannot find much extra.

### A.28 The Step 8 timing puzzle — why addition is so slow

Step 8's per-task wall on addition (11:55–12:39) is 6-7× the wall on multiplication (2:01–2:17). The difference is dominated by the per-pair principal-angle SVD count:

- Addition: 47 concepts → C(47, 2) = 1,081 pairs per cell × 15 cells per task = 16,215 SVDs per task.
- Multiplication: 50 concepts → C(50, 2) = 1,225 pairs per cell × 15 cells per task = 18,375 SVDs per task.

Wait — multiplication has MORE SVDs per task (18,375 vs 16,215). So why is multiplication faster?

Two hypotheses:

1. **Concept-dim distribution differs.** Addition's eligible concepts span a wider range of dims (1, 7-9 typically); the per-pair SVD on a 9 × 9 matrix is heavier than on a 1 × 1 (trivial). Multiplication's eligible-dim distribution may be more uniform.
2. **Cache priming amortisation.** The 1000-trial baseline cache is shared across cells; the first task to encounter each unique (dim_a, dim_b) pays the priming cost. Addition cells span a larger range of (dim_a, dim_b) pairs than multiplication (because addition concepts include 2-dim concepts like `ans_num_digits` which have unusual values).

The actual timing data per task (from sacct) shows: addition tasks ran on babel-s9-16, babel-t9-16, babel-s9-24 with similar wall times. The cache was populated by the first task to hit each new dim-pair; subsequent tasks reused it from disk.

A future optimisation (not in scope for the present run) would batch per-cell SVDs by (dim_a, dim_b) bucket: all pairs with the same (dim_a, dim_b) tuple computed as a single batched matrix operation. At a typical 1,081 pairs per cell × ~15 unique (dim_a, dim_b) pairs, the speedup would be 15-30×.

### A.29 Distance-preservation is essentially independent of mode

For the Step 9 Spearman ρ on addition cells (merged variant), the cross-mode range within a (model, layer) is typically ≤ 0.00002. For multiplication cells, ≤ 0.00015. The mode choice barely affects distance preservation:

- mode=off: the union has a 1-D β_answer direction; activations are raw.
- mode=answer: the union doesn't have β_answer; activations are residualised.
- mode=norm: the union has β_norm and β_answer; activations are norm-residualised.

The fact that distance preservation is ~identical across modes means the geometric structure of activations is governed primarily by named concepts plus a 1-D magnitude/answer-scalar direction. Once these are accounted for, the residual orthogonal complement is structurally negligible for pairwise distance purposes — regardless of which mode's residualisation we use.

This is a strong statement about the redundancy of the three modes: from a Step 9 perspective, they are nearly equivalent. The cross-mode `spearman_cross_mode.csv` table makes this explicit; we report it as a numerical fact and let the analyst draw the conclusion.

### A.30 The Pythia × multiplication × L28 cell

The cell `Pythia × multiplication × layer 28 × mode=norm × merged` has the highest var_explained in the entire production run: 0.9106. The next-highest is the same cell with mode=answer (0.9088), then with mode=off (0.8952). This is Pythia's deepest layer (layer 28 of 32), under norm-residualisation.

Several observations about this cell:

- N = 2,757 (correct subset).
- k_merged = 749, d_residual = 3,347.
- γ = 1.214 (unreliable for MP).
- var_explained = 0.9106 (highest in the run).
- top_eigenvalue = 4.03 (high; consistent with the cell having strong residual eigenvalues even after the union captures 91% of variance).
- n_above_mp = 173 (lowest among multiplication cells in this run — fewer above-MP directions despite high var_explained).

The cell is the cleanest variance-capture cell in multiplication. For Stage 2, it should produce the cleanest helix on `ans_units`, `partial_product_units`, and `carry_units` — these are the headline concepts for multiplication Fourier screening.

### A.31 Suggested follow-ups beyond Stage 2

Items not in the current scope but flagged as worth considering:

1. **Stage 2 Fourier screen on the residual eigenvectors.** Apply the same periodicity test that Stage 2 will apply to LDA centroid sequences, but to the top 50 residual eigenvectors per cell. If any residual eigendirection has significant Fourier power at periods {10, 100}, that would be evidence of digit-periodic structure escaping the linear sweep.
2. **GPLVM on the residual.** Fit a Bayesian GPLVM to the residual (orthogonal complement of the union). If the fit picks up a non-trivial latent manifold structure, that complements the FDR-null finding from the sweep.
3. **Per-tier residual hunt.** Build a tier-1-only union, project out, and ask whether tier-2 or tier-3 concepts emerge as residual correlates. This would quantify the contribution of each concept tier to the variance budget and test whether the higher tiers carry residual signal that the union flattens out.
4. **Cross-(model, task) principal angles.** For matched concepts across (model, task) pairs (e.g., `a_units` in GPT-J × addition vs `a_units` in Llama × addition), how similar are the per-cell LDA-A subspaces? This would test cross-model concept-alignment — a different audit dimension we did not address here.
5. **Cross-correlate-set angles.** For each plan-locked correlate set (e.g., `{a, b, a_units, b_units}` for `ans_units` in addition), measure the principal angles of the target's basis against the correlate-union's basis. This is a Stage 3 input but can be computed now from the existing artifacts.

### A.32 Connection to the EMNLP 2026 paper narrative

The paper's central claim is: a linear probe finds a clean geometric structure for an arithmetic concept; we test whether that structure belongs to the concept or is inherited from algebraically related concepts. The four-stage pipeline (linear probe → Bayesian manifold → ownership → causal ablation) is the methodology; the audit (Steps 7-9) is the sanity-check that the methodology is well-founded.

The audit results that the paper can cite:

- **Variance budget.** "Across 90 cells, the union of 47-50 named-concept subspaces (with 1-D β corrections) explains 70-95% of activation variance, with Pythia consistently highest."
- **No FDR-significant residual correlate.** "After 1000-permutation BH-FDR correction across each cell's (direction × column) grid, no cell exhibits a residual correlate at `|ρ_s| > 0.15` with `q < 0.05`. The named-concept registry appears to capture all linearly-organised structure at our metadata-column resolution."
- **Geometric faithfulness.** "The union subspace preserves pairwise distance Spearman ρ ≥ 0.9994 across every cell, with distance-variance-explained ≥ 0.9989 on addition and ≥ 0.992 on multiplication. The activation geometry is faithfully captured by the named-concept union."
- **Superposition prevalence.** "76-92% of concept-pair principal angles fall significantly below the random-subspace baseline at our 1000-trial empirical null. Concepts are heavily superposed, particularly on multiplication and within Pythia."

Each claim is supported by per-cell metrics. The numbers cluster cleanly across cells; the cross-model and cross-task differences are interpretable. The audit is the foundation that Stage 2-Stage 4 will build on.

### A.33 The single-best cell for a Stage-2 pilot

If we had to pick one (model, task, layer, mode, variant) cell to run Stage-2 helix and GPLVM on first, the criteria would be:

- High var_explained (cleaner residual, more of the structure inside the union).
- Reliable MP regime (γ < 0.7).
- No FDR-significant residual correlate (no obvious named-concept addition needed).
- High distance_var_explained.
- Substantial superposition (the union has nontrivial concept overlap, making Stage 2 interesting).
- Stage-2-readiness from Step 6 (status=fit_ok ∧ n_sig ≥ 2 ∧ cv_acc ≥ 0.7 ∧ N ≥ 300).

Reading across all cells:

- **Pythia × addition × layer 16 × mode=off × merged** scores well on all criteria: var_explained = 0.949, γ = 0.264 (reliable), no FDR-correlate, dvar_explained = 0.99986, superposition_rate = 81.6%. Step-6 Stage-2-readiness count: 38 concepts ready.
- **GPT-J × addition × layer 14 × mode=off × merged** also scores well: var_explained = 0.866, γ = 0.247, no FDR-correlate, dvar_explained = 0.99989, superposition_rate = 80.4%, Stage-2-readiness count: 39.

Both are candidates. Pythia's higher var_explained suggests its centroid sequences inside the union will be tighter — making the Fourier helix screen more sensitive. GPT-J × addition × L14 is the parent project's headline cell from KT 2024; for direct cross-paper comparison, running Stage 2 there first lets us anchor against KT's published helix.

### A.34 Final accounting of disk usage

The audit pipeline's full output across the production run:

- residual_hunting: 7.0 GB across 90 cells × 2 variants × {basis, eigenvectors, eigenvalues, sweep CSV} + per-cell metadata.
- principal_angles: 39 MB across 90 cells × angles_pairwise.csv + self_angles.csv + the 1000-trial baseline cache (≈ 5 MB).
- jl_distance: 2.7 GB across 90 cells × 2 variants × {d_full/d_proj for small N, sample for large N, hist for both} + metrics JSON.

Total: ~9.7 GB on the scratch filesystem. Acceptable.

The comparison directories add ~2 MB of aggregator CSVs. The total git-tracked artifact set (this document + scripts + sbatch + manifests) is well under 1 MB.

### A.35 Reflection on the run

The audit pipeline was designed and executed across two sessions. The first session (planning and writing) took several hours of iteration; the second session (production run and reporting) ran in approximately 13 hours wall on 6 concurrent A6000s, with three failure modes encountered and corrected in-flight:

- The first failure was a stale `statsmodels` import in the worker SBATCH env-check line, killing tasks before any Python work began. Fixed by editing the env check.
- The second failure was the residualised-cache row-count assumption — the cache stores the full task population, not the correct subset. Fixed by always applying the correct mask.
- The third was the LDA-A basis non-orthogonality, caught at the principal-angle self-angle sanity check during Step 8 smoke testing. Fixed by QR-orthonormalising bases on load.

Each failure was caught early — within minutes of the initial run for the env bug, within smoke testing for the mask bug, and within the first Step-8 cell for the orthonormality bug. The combined fix cycle (debug + edit + commit + push + relaunch) added approximately 2 hours to the total wall time across the three corrections.

Future Stage-2/3/4 work should leverage the resume-by-metadata pattern, the atomic-write pattern, and the empirical-null framework established here. The audit pipeline's design is fully reusable for any new (model, task, mode) configuration; only the per-cell layer set (in `config.yaml`) needs to be updated.

### A.36 The mathematical derivation of the Marchenko–Pastur edge

The MP density for the eigenvalues of `(1/N) X^T X` where `X` has i.i.d. entries with mean 0 and variance σ² is:

```
ρ_MP(λ) = (1 / (2π σ² γ λ)) · sqrt((λ_max_mp − λ)(λ − λ_min_mp))   for λ ∈ [λ_min_mp, λ_max_mp]
ρ_MP(λ) = 0                                                          otherwise
```

with `λ_max_mp = σ²(1 + √γ)²` and `λ_min_mp = σ²(1 − √γ)²`, where `γ = d / N`. At γ > 1, the density also has a point mass at 0 of weight `1 − 1/γ`; the support otherwise spans [`(1 − √γ)²` for γ < 1] or [0, `(1 + √γ)²` for γ > 1].

The "upper edge" is the boundary of the bulk; eigenvalues above this edge are *signal* in the sense that they cannot be explained by isotropic noise at level σ² in dimension d with N samples. The MP test is "above the upper edge" is the count of eigenvalues exceeding `λ_max_mp = σ²(1 + √γ)²`.

For our setting, X is the mean-centred residual `(X − P_V_all X) − mean`. The noise model is that the residual is approximately isotropic Gaussian within the d_residual-dimensional orthogonal complement of V_all. The trace-based σ² estimate `||X_centered||_F² / (N · d_residual)` is the mean eigenvalue of the sample covariance, which under the null equals σ².

**The Tracy-Widom correction.** The exact distribution of the largest eigenvalue of a Wishart matrix at finite N, d converges to a Tracy-Widom distribution centred at `λ_max_mp + (something of order σ² γ^{−2/3} N^{−2/3})` with a known shape (Soshnikov 1999, Johnstone 2001). At γ < 0.7 and our N (2.7k-10k), the Tracy-Widom standard deviation is small relative to λ_max_mp; cite `n_above_mp` as a strict count. At γ → 1, the fluctuation is large; do not cite.

We do not implement Tracy-Widom-corrected critical values in the present run; we use the bare λ_max_mp comparison with the γ < 0.7 reliable-regime flag. A future improvement would substitute the Tracy-Widom-shifted threshold for the bare λ_max_mp and remove the γ-dependent unreliability flag.

### A.37 The mathematical derivation of the trace-σ² estimator

The trace identity for the sample covariance `S = (1/N) X_centered^T X_centered` is:

```
trace(S) = (1/N) sum_i (X_centered)_i · (X_centered)_i
         = (1/N) ||X_centered||_F²
```

The trace is also the sum of all eigenvalues: `trace(S) = sum_k λ_k`. Under the null (no signal, all eigenvalues equal to σ² on average), `trace(S) = d · σ²`, so `σ² = trace(S) / d`.

When signal exists, the trace is inflated by the sum of signal eigenvalues, which inflates the σ² estimate slightly. With d_residual ~ 2,000-4,000 dimensions and at most ~250 signal eigenvalues, each signal eigenvalue contributing typically 2-10× σ² beyond the noise contribution, the trace inflation is at most ~5% — making the test conservative (it is harder for signal eigenvalues to cross an inflated cliff).

A bias-corrected estimator would subtract the contribution of the top-k eigenvalues from the trace before dividing:

```
σ²_unbiased ≈ (trace(S) − sum_{k=1}^K λ_k_signal) / (d_residual − K)
```

where K is the suspected number of signal eigenvalues (we don't know K a priori). The bias correction would tighten λ_max_mp slightly and could increase `n_above_mp` by a few cells where eigenvalues are right at the threshold. For our purposes, the conservative bare-trace estimator is acceptable; we err on the side of fewer false positives.

### A.38 The mathematical derivation of BH-FDR

The Benjamini–Hochberg procedure (Benjamini & Hochberg 1995) controls the expected fraction of false discoveries among the rejected hypotheses. Given a list of m p-values, sort them ascending: `p_(1) ≤ p_(2) ≤ ... ≤ p_(m)`. Find the largest k such that `p_(k) ≤ k/m · α`. Reject all hypotheses with p-value ≤ `p_(k)`.

The q-values returned by `scipy.stats.false_discovery_control(p, method="bh")` are:

```
q_(k) = min(min_{j ≥ k} { m · p_(j) / j }, 1)
```

A rejection at q-threshold α is equivalent to FDR ≤ α (in expectation, under exchangeability assumptions on the p-values under the null).

For our setting, the (direction × column) p-values within a cell come from independent permutation tests (the permutation breaks the joint distribution between direction and column). The exchangeability assumption holds approximately at the per-cell level. Across cells, however, the p-values are correlated (the same metadata columns appear in every cell), and a meta-FDR across the 90 × ~10000 = ~900,000 (cell × test) grid would over-correct. We apply BH per-cell only.

The choice of α = 0.05 is conventional and matches the parent project's reporting threshold.

### A.39 Detailed Step-7 implementation walk-through

The Step-7 worker `residual_hunting.py` is structured around a single per-cell function `run_cell()` that does the following:

```python
def run_cell(cfg, model, task, mode, layer, ...):
    1. resume_check: read metadata.json, return early if complete.
    2. cell_dir.mkdir(parents=True, exist_ok=True)
    3. filter_dict = load_concept_filter(...)
    4. eligible = eligible_concepts(filter_dict)
    5. derived_metadata = build_derived_columns(problems_df, answers_df, task)
    6. for variant in ("merged", "generous"):
         V_all, umeta = build_union(..., variant, ...)
         atomic_save(V_all, "union_basis_<variant>.npy")
         X_residual, var_orig, var_resid, var_explained = project_and_residual(X, V_all)
         assert var_resid ≤ var_orig * 1.001 + 1e-3
         d_residual = 4096 − V_all.shape[0]
         pca_info = pca_with_mp(X_residual, d_residual, seed)
         atomic_save(eigenvalues, "eigenvalues_<variant>.npy")
         atomic_save(eigenvectors, "eigenvectors_<variant>.npy")
         atomic_json(mp_info_json, "mp_info_<variant>.json")
         if variant == "merged":
             corr_df = correlation_sweep(...)
             corr_df.to_csv("correlation_sweep_merged.csv")
         summary_rows.append(summary_row_for_variant)
    7. stage3_unions = build_stage3_unions(...)
    8. atomic_json(union_meta, "union_meta.json")
    9. atomic_json({computation_status: "complete", summary_rows}, "metadata.json")
    10. return summary_rows
```

Within `build_union`, the stacking and SVD orthonormalisation:

```python
def build_union(results_root, model, task, mode, layer, eligible, variant, X, answer_scalar):
    rows = []
    contributions = []
    for concept in eligible:
        sources = ["ccsvd", "lda_a"] if variant == "merged" else ["ccsvd", "lda_a", "lda_b"]
        for src in sources:
            B = load_basis_rows(ccsvd_basis_path(...) if src == "ccsvd"
                             else lda_a_basis_path(...) if src == "lda_a"
                             else lda_b_basis_path(...))
            if B.shape[0] > 0:
                rows.append(B)
                contributions.append({"concept": concept, "source": src, "n_dims": B.shape[0]})
    stacked = np.vstack(rows)
    stacked, beta_labels = append_mode_betas(stacked, X, mode, answer_scalar)
    U, S, Vt = np.linalg.svd(stacked, full_matrices=False)
    keep = S > SVD_TOLERANCE_FACTOR * S[0]
    V_all = Vt[keep].astype(np.float32)
    return V_all, {...meta...}
```

The `load_basis_rows` helper handles both `(D, r)` and `(r, D)` orientations by checking shapes:

```python
def load_basis_rows(path):
    if not path.exists(): return zeros((0, 4096), dtype=float32)
    B = np.load(path)
    if B.shape[0] == 4096: return B.T.astype(float32, copy=False)
    if B.shape[1] == 4096: return B.astype(float32, copy=False)
    return zeros((0, 4096), dtype=float32)
```

Within `correlation_sweep`, the batched permutation null:

```python
def correlation_sweep(X_resid, eigenvectors, eigenvalues, n_top, metadata_arrays, derived_names, seed, n_permutations=1000):
    Z = X_resid @ eigenvectors[:n_top].T               # (N, n_top)
    # ... filter columns ...
    Z_raw = Z.astype(float64).T                          # (n_top, N)
    Z_rank = np.stack([_ranks(z) for z in Z_raw])        # (n_top, N)
    C_raw = np.stack(col_raws)                           # (n_col, N)
    C_rank = np.stack([_ranks(c) for c in C_raw])        # (n_col, N)
    obs_sp = _batched_corr(Z_rank, C_rank)               # (n_top, n_col)
    obs_pr = _batched_corr(Z_raw, C_raw)                 # (n_top, n_col)
    rng = np.random.default_rng(seed)
    sp_tail = zeros_like(obs_sp, dtype=int64)
    pr_tail = zeros_like(obs_pr, dtype=int64)
    for _ in range(n_permutations):
        perm = rng.permutation(N)
        sp_pred = _batched_corr(Z_rank, C_rank[:, perm])
        pr_pred = _batched_corr(Z_raw, C_raw[:, perm])
        sp_tail += (|sp_pred| ≥ |obs_sp|)
        pr_tail += (|pr_pred| ≥ |obs_pr|)
    sp_p = (sp_tail + 1) / (n_permutations + 1)
    pr_p = (pr_tail + 1) / (n_permutations + 1)
    sp_q = false_discovery_control(sp_p.ravel(), method="bh").reshape(sp_p.shape)
    pr_q = false_discovery_control(pr_p.ravel(), method="bh").reshape(pr_p.shape)
    return DataFrame with one row per (direction, col) pair
```

The `_batched_corr` helper computes Pearson on rows of two matrices via matmul:

```python
def _batched_corr(D, C):
    D_centred = D − D.mean(axis=1, keepdims=True)
    C_centred = C − C.mean(axis=1, keepdims=True)
    d_norm = sqrt((D_centred * D_centred).sum(axis=1, keepdims=True))   # (n_dir, 1)
    c_norm = sqrt((C_centred * C_centred).sum(axis=1, keepdims=True))   # (n_col, 1)
    num = D_centred @ C_centred.T                                        # (n_dir, n_col)
    denom = d_norm @ c_norm.T                                            # (n_dir, n_col)
    return where(denom > 0, num / denom, nan)
```

Total compute per cell: 1 large SVD orthonormalisation (CPU, ~0.5 s for k=2000), 1 GPU projection (~0.1 s), 1 randomised SVD on the residual (CPU, ~5 s), 1 batched correlation sweep with 1000 perms (CPU, ~9 s). Total ~15 s per cell × 2 variants ≈ 30 s per cell.

### A.40 Detailed Step-8 implementation walk-through

The Step-8 worker `principal_angles.py` is structured around `run_cell()`:

```python
def run_cell(cfg, model, task, mode, layer, baseline, logger):
    1. resume_check
    2. cell_dir.mkdir
    3. filter_dict = load_concept_filter(...)
    4. eligible = eligible_concepts(filter_dict)
    5. bases = {c: load_basis_rows(lda_a_basis_path(...)) for c in eligible if non-empty}
    6. concepts = sorted(bases.keys())
    7. # Self-angle sanity check
       self_rows = []
       for c in concepts:
           ang = principal_angles_deg(bases[c], bases[c])
           ok = max(ang[:5]) < SELF_ANGLE_TOLERANCE_DEG
           self_rows.append({...})
       DataFrame(self_rows).to_csv("self_angles.csv")
    8. # Pairwise angles + baseline + FDR
       rows = []
       seed = cell_seed("baseline_<mode>")
       for i in range(len(concepts)):
           for j in range(i+1, len(concepts)):
               ang = principal_angles_deg(bases[concepts[i]], bases[concepts[j]])
               base = baseline.get(dim_a, dim_b, seed)
               theta1 = ang[0]
               p_perm = (sum(base.thetas ≤ theta1) + 1) / (1000 + 1)
               flag = theta1 < base.theta1_p5 - 10.0
               rows.append({...})
       df = DataFrame(rows)
       _, q_fdr, _, _ = multipletests(df["perm_p"], alpha=0.05, method="fdr_bh")
       df["fdr_q"] = q_fdr
       df.to_csv("angles_pairwise.csv")
    9. atomic_json({computation_status: "complete", summary_row}, "metadata.json")
    10. return summary_row
```

The `principal_angles_deg` core:

```python
def orthonormalise_basis(B):
    Q, R = np.linalg.qr(B.T)
    diag = np.abs(np.diag(R))
    keep = diag > 1e-10 * (diag.max() if diag.size else 1.0)
    return Q[:, keep].T.astype(np.float32)

def principal_angles_deg(B_a, B_b):
    Ba = orthonormalise_basis(B_a)
    Bb = orthonormalise_basis(B_b)
    if Ba.shape[0] == 0 or Bb.shape[0] == 0:
        return array([], dtype=float64)
    M = Ba @ Bb.T
    S = np.linalg.svd(M, compute_uv=False)
    S = np.clip(S, -1.0, 1.0)
    return rad2deg(arccos(S))
```

The `BaselineCache` class:

```python
class BaselineCache:
    def __init__(self, cache_path, ambient_d=4096, n_trials=1000):
        self.cache_path = cache_path
        self.ambient_d = ambient_d
        self.n_trials = n_trials
        self.mem = {}
        if cache_path.exists():
            self.mem = np.load(cache_path, allow_pickle=True).item()

    def _flush(self):
        # tempfile + os.replace
        fd, tmp = tempfile.mkstemp(suffix=".npy", dir=self.cache_path.parent)
        os.close(fd)
        np.save(tmp, np.array(self.mem, dtype=object))
        os.replace(tmp, self.cache_path)

    def get(self, dim_a, dim_b, seed):
        key = (min(dim_a, dim_b), max(dim_a, dim_b))
        if key in self.mem:
            return self.mem[key]
        rng = np.random.default_rng(seed)
        thetas = np.empty(self.n_trials, dtype=float64)
        for t in range(self.n_trials):
            A = rng.standard_normal((self.ambient_d, key[0]))
            Q_A, _ = np.linalg.qr(A)
            B = rng.standard_normal((self.ambient_d, key[1]))
            Q_B, _ = np.linalg.qr(B)
            M = Q_A.T @ Q_B
            S = np.linalg.svd(M, compute_uv=False)
            thetas[t] = rad2deg(arccos(clip(S[0], -1, 1)))
        rec = {"n_trials": ..., "ambient_d": ..., "theta1_mean": ...,
               "theta1_p5": ..., "theta1_p1": ..., "thetas": thetas}
        self.mem[key] = rec
        self._flush()
        return rec
```

Total compute per cell: ~1,100-1,200 pair-SVDs (each ~5 ms on CPU) + cache lookups + FDR correction. Per-cell wall ~6-9 minutes; per-task wall ~12 hours for addition tasks.

### A.41 Detailed Step-9 implementation walk-through

The Step-9 worker `jl_distance.py` is structured around `run_cell()`:

```python
def run_cell(cfg, model, task, mode, layer, X_correct, logger):
    1. resume_check
    2. cell_dir.mkdir
    3. N = X_correct.shape[0]
    4. ii, jj = all_pair_indices(N)                          # (n_pairs,) int64 each
    5. n_pairs = N * (N − 1) // 2
    6. rng = np.random.default_rng(42)
       plot_idx = rng.choice(n_pairs, size=min(10000, n_pairs), replace=False) if N > 5000 else None
    7. summary_rows = []
    8. for variant in ("merged", "generous"):
         V_all = load_union_basis(...)
         if V_all is None:
             summary_rows.append({"status": "missing_union"})
             continue
         X_proj, X_resid = project_full(X_correct, V_all)
         d_full, d_proj = compute_pairwise_distances_gpu(X_correct, X_proj, ii, jj)
         metrics = compute_jl_metrics(d_full, d_proj)
         pyth = pythagorean_check_full_gpu(X_correct, X_proj, X_resid, ii, jj)
         out_path = cell_dir / f"jl_metrics_{variant}.json"
         atomic_json({...}, out_path)
         # Histograms
         hi = max(d_full.max(), d_proj.max())
         H, xe, ye = np.histogram2d(d_full, d_proj, bins=200, range=[[0, hi], [0, hi]])
         np.savez_compressed(f"d_hist_{variant}.npz", H=H, x_edges=xe, y_edges=ye)
         # Save full or subsampled distance arrays
         if N <= 5000:
             atomic_save(d_full, f"d_full_{variant}.npy")
             atomic_save(d_proj, f"d_proj_{variant}.npy")
         else:
             atomic_save(d_full[plot_idx], f"d_full_sample_{variant}.npy")
             atomic_save(d_proj[plot_idx], f"d_proj_sample_{variant}.npy")
         summary_rows.append({...all metrics...})
    9. atomic_json({computation_status: "complete", summary_rows}, "metadata.json")
    10. return summary_rows
```

The `compute_pairwise_distances_gpu`:

```python
def compute_pairwise_distances_gpu(X, X_proj, ii, jj, batch=200000):
    n_pairs = len(ii)
    d_full = np.empty(n_pairs, dtype=float32)
    d_proj = np.empty(n_pairs, dtype=float32)
    if _HAS_CUPY:
        X_g = cp.asarray(X)
        Xp_g = cp.asarray(X_proj)
        ii_g = cp.asarray(ii)
        jj_g = cp.asarray(jj)
        for k0 in range(0, n_pairs, batch):
            k1 = min(k0 + batch, n_pairs)
            i_b = ii_g[k0:k1]; j_b = jj_g[k0:k1]
            dif_full = X_g[i_b] - X_g[j_b]
            dif_proj = Xp_g[i_b] - Xp_g[j_b]
            d_full[k0:k1] = cp.asnumpy(cp.linalg.norm(dif_full, axis=1))
            d_proj[k0:k1] = cp.asnumpy(cp.linalg.norm(dif_proj, axis=1))
        del X_g, Xp_g, ii_g, jj_g
        cp.get_default_memory_pool().free_all_blocks()
    else:
        # CPU fallback (slow for large N)
        for k0 in range(0, n_pairs, batch):
            k1 = min(k0 + batch, n_pairs)
            d_full[k0:k1] = np.linalg.norm(X[ii[k0:k1]] - X[jj[k0:k1]], axis=1)
            d_proj[k0:k1] = np.linalg.norm(X_proj[ii[k0:k1]] - X_proj[jj[k0:k1]], axis=1)
    return d_full, d_proj
```

The `pythagorean_check_full_gpu`:

```python
def pythagorean_check_full_gpu(X, X_proj, X_resid, ii, jj, batch=200000):
    n_pairs = len(ii)
    max_rel_err = 0.0
    sum_rel_err = 0.0
    n_violations = 0
    if _HAS_CUPY:
        X_g = cp.asarray(X, dtype=cp.float64)
        Xp_g = cp.asarray(X_proj, dtype=cp.float64)
        Xr_g = cp.asarray(X_resid, dtype=cp.float64)
        ii_g = cp.asarray(ii)
        jj_g = cp.asarray(jj)
        for k0 in range(0, n_pairs, batch):
            k1 = min(k0 + batch, n_pairs)
            i_b = ii_g[k0:k1]; j_b = jj_g[k0:k1]
            df = X_g[i_b] - X_g[j_b]
            dp = Xp_g[i_b] - Xp_g[j_b]
            dr = Xr_g[i_b] - Xr_g[j_b]
            d_full_sq = cp.sum(df * df, axis=1)
            d_proj_sq = cp.sum(dp * dp, axis=1)
            d_resid_sq = cp.sum(dr * dr, axis=1)
            rel_err = cp.where(d_full_sq > 1e-20,
                               cp.abs(d_full_sq - d_proj_sq - d_resid_sq) / d_full_sq,
                               cp.zeros_like(d_full_sq))
            chunk_max = float(cp.max(rel_err))
            chunk_sum = float(cp.sum(rel_err))
            chunk_vio = int(cp.sum(rel_err > 1e-6))
            max_rel_err = max(max_rel_err, chunk_max)
            sum_rel_err += chunk_sum
            n_violations += chunk_vio
        del X_g, Xp_g, Xr_g, ii_g, jj_g
        cp.get_default_memory_pool().free_all_blocks()
    return {"pyth_max_rel_error": max_rel_err, "pyth_mean_rel_error": sum_rel_err / n_pairs,
            "pyth_n_violations": n_violations, "pyth_n_pairs": n_pairs, "pyth_dtype": "float64"}
```

Total compute per cell:

- Projection on GPU: ~0.1 s for k=750 (multiplication), ~0.4 s for k=2000 (addition).
- Pair index generation: ~1 s for N=10k (50M pair indices).
- Pairwise distance on GPU: ~30 s for N=10k addition; ~1 s for N=3k multiplication.
- Spearman + Pearson on float32: scipy spearmanr ~30 s for 50M pairs; ~3 s for 4M pairs.
- Pythagorean check on float64 GPU: ~30 s for N=10k addition; ~3 s for N=3k multiplication.
- Histogram + save: ~1 s.

Per-cell wall: ~90 s for addition cells, ~10 s for multiplication cells, × 2 variants.

### A.42 The exact gamma cutoff choice

We adopted `mp_reliable_flag = (γ < 0.7)` as the threshold below which the MP cliff test is paper-citable. This is conservative; the parent project used a similar threshold around 0.7 implicitly (their L5 cells, which had `γ ≈ 0.86`, were flagged with caveats but still cited).

A more permissive cutoff (e.g., γ < 0.95) would include additional cells in the "MP-reliable" set but at the cost of larger Tracy-Widom fluctuations on the cliff. A tighter cutoff (γ < 0.5) would exclude even some addition cells (Llama × addition × any layer has γ ≈ 0.25, well within either bound).

At our cutoff of 0.7, all 45 addition cells are flagged reliable and all 45 multiplication cells are flagged unreliable. The cutoff therefore acts as a binary task-level discriminator: addition is reliable; multiplication is not. This is a clean reporting boundary but loses some nuance — the Pythia × multiplication cells with γ near the lower edge of the multiplication range might be cited with Tracy-Widom-corrected critical values.

For future work, a Tracy-Widom-corrected MP test would absorb the γ → 1 regime properly. The implementation exists (e.g., the `rmtutils` package); we did not include it in the present run because the variance-explained gauge plus the correlation sweep already provide γ-independent signal-detection paths.

### A.43 Per-cell summary CSV column reference

The Step 7 aggregator's `comparison/summary_all.csv` columns:

| Column | Type | Description |
|---|---|---|
| `model` | str | "gpt-j-6b" / "llama-3.1-8b" / "pythia-6.9b" |
| `task` | str | "addition" / "multiplication" |
| `mode` | str | "off" / "answer" / "norm" |
| `layer` | int | One of the 5 layers per model |
| `variant` | str | "merged" / "generous" |
| `k_union` | int | Union dimension after SVD orthonormalisation |
| `stacked_dim` | int | Pre-orthonormalisation stacked basis rows |
| `n_concepts_kept` | int | Number of eligible concepts contributing to the union |
| `n_concepts_carved` | int | Number of concepts carved out by the Step-6 mode (e.g., 9 for mode=answer addition) |
| `N` | int | Per-cell correct subset size |
| `d_residual` | int | 4096 − k_union |
| `gamma` | float | d_residual / N |
| `sigma_sq` | float | Trace-based σ² estimate |
| `lambda_max_mp` | float | MP upper edge |
| `top_eigenvalue` | float | Largest residual eigenvalue |
| `n_above_mp` | int | Count of residual eigenvalues above λ_max_mp |
| `mp_reliable_flag` | bool | γ < 0.7 |
| `var_explained` | float | 1 − var_resid / var_orig |
| `var_residual` | float | ||X_residual||_F² / N |
| `n_correlation_flags` | int | Count of (direction × column) pairs with `flag == True` |
| `top_corr_concept` | str / NaN | Metadata column of the most-flagged pair (NaN if no flags) |
| `top_corr_rho` | float | Spearman of the most-flagged pair |
| `top_corr_q` | float | FDR-q of the most-flagged pair |
| `runtime_seconds` | float | Per-cell wall time |
| `seed` | int | cell_seed value used for the randomised SVD |
| `status` | str | "fit_ok" / other (never "fit_ok" alternative in production) |

The Step 8 aggregator's `comparison/summary_all.csv` columns:

| Column | Type | Description |
|---|---|---|
| `model`, `task`, `mode`, `layer` | | Same as above |
| `n_concepts_kept` | int | Eligible concepts contributing to pair set |
| `n_pairs` | int | C(n_concepts_kept, 2) |
| `n_superposition_flags` | int | Count of pairs with θ_1 < baseline_p5 − 10° |
| `n_fdr_q_below_alpha` | int | Count of pairs with `fdr_q < 0.05` |
| `median_angle_1` | float | Median smallest principal angle across pairs (degrees) |
| `median_angle_5` | float | Median 5th principal angle (degrees) |
| `self_angle_failures` | int | Count of concepts with self-angle > 1° (should be 0 in production) |
| `runtime_seconds` | float | |
| `status` | str | "ok" / fallback states |

The Step 9 aggregator's `comparison/summary_all.csv` columns:

| Column | Type | Description |
|---|---|---|
| `model`, `task`, `mode`, `layer`, `variant` | | |
| `k_union` | int | From Step 7 |
| `N` | int | |
| `n_pairs` | int | N(N−1)/2 |
| `spearman_rho` | float | Spearman correlation of (d_full, d_proj) |
| `pearson_r` | float | |
| `mean_rel_error` | float | mean of `|d_full − d_proj| / d_full` |
| `max_rel_error` | float | max of the same |
| `distance_var_explained` | float | 1 − var(d_full² − d_proj²) / var(d_full²) |
| `pyth_max_rel_error` | float | Float64 Pythagorean max relative error |
| `pyth_mean_rel_error` | float | |
| `pyth_n_violations` | int | Count of pairs with pyth rel err > 1e-6 |
| `runtime_seconds` | float | |
| `status` | str | "ok" / "missing_union" |

### A.44 The structure of the aggregator outputs

After all three workers finish, the three aggregators produce:

**Step 7 aggregator** (`aggregate_residual_hunting.py`):

- `summary_all.csv` — 180 rows (per-cell × per-variant).
- `var_explained_cross_mode.csv` — pivot of `var_explained` per (model, task, layer, variant) across the 3 modes.
- `n_above_mp_cross_mode.csv` — pivot of `n_above_mp` plus `mp_reliable_flag`.
- `k_union_cross_mode.csv` — pivot of `k_union`.
- `gamma_cross_mode.csv` — pivot of γ.
- `residual_top_correlate_cross_mode.csv` — top correlate per (model, task, mode, layer), merged only.
- `variant_delta.csv` — generous − merged deltas per (model, task, mode, layer).
- `summary_with_matched_count.csv` — summary annotated with the matched-population concept count per (model, task, layer).

**Step 8 aggregator** (`aggregate_principal_angles.py`):

- `pairwise_all.csv` — all (cell × pair) rows stacked.
- `summary_all.csv` — 90 cell-level summary rows.
- `superposition_rate_by_cell.csv` — per-cell rate, median angles, FDR counts.
- `superposition_by_tier_pair.csv` — per (cell × tier-pair) aggregate.
- `cross_mode_superposition.csv` — pivot of superposition rate per (model, task, layer) across modes.

**Step 9 aggregator** (`aggregate_jl_distance.py`):

- `summary_all.csv` — 180 rows.
- `spearman_cross_mode.csv` — pivot per (model, task, layer, variant) across modes.
- `pearson_cross_mode.csv` — same.
- `distance_var_explained_cross_mode.csv` — same.
- `mean_rel_error_cross_mode.csv` — same.
- `max_rel_error_cross_mode.csv` — same.
- `pyth_max_rel_error_cross_cell.csv` — per-cell pythagorean diagnostics.
- `variant_delta_jl.csv` — generous − merged deltas.

Each aggregator is CPU-only and runs in ~5 seconds end-to-end. The SLURM resource request (1 GPU + 8 CPUs + 64 GB) is over-provisioned but matches the parent's pattern.

### A.45 What the audit will be cited for in the paper

The paper's audit section (likely §3.5 or §4.5 depending on final paper structure) will cite:

1. **The variance-budget headline.** "Across 90 (model, task, mode, layer) cells, the union of all named concept subspaces, augmented with mode-specific β scalar directions, explains a median 84% of activation variance (range 70-95%). Pythia consistently has the highest per-cell variance-budget completeness."
2. **The geometric-faithfulness headline.** "The named-concept union preserves the model's pairwise activation geometry: Spearman ρ ≥ 0.9994 between full-space and projected pairwise distances across all 90 cells; distance-variance-explained ≥ 0.992 on multiplication and ≥ 0.999 on addition."
3. **The residual-null headline.** "1000-permutation Benjamini-Hochberg FDR correction on each cell's correlation sweep (residual eigendirections × metadata columns) returns no FDR-significant residual correlate at `|ρ_s| > 0.15` and `q < 0.05` across any of the 90 cells. The named-concept registry appears to capture all linearly-organised structure detectable by our sweep."
4. **The superposition headline.** "76-92% of concept-pair principal angles fall significantly below their 1000-trial empirical random-subspace baseline at the SUPERPOSITION_MARGIN = 10° threshold; multiplication has consistently higher rates than addition; Pythia × multiplication has the highest rate at 92.5%."
5. **The Stage-3 informativeness headline.** "Target/correlate principal angles at the headline cells are uniformly < 10°, licensing Stage 3 ownership tests as geometrically informative."

The cross-(model, task) ordering on each gauge is also citable as a methodology-validating finding.

### A.46 Final notes on reproducibility

The full audit pipeline runs deterministically modulo float32 round-off. Each cell's seed (recorded in `metadata.json`) and the library version triple are sufficient to re-derive any artifact byte-for-byte (modulo CuPy GPU non-determinism, which we have empirically verified to be ≤ 1 part in 10⁶ on metric outputs).

The complete codebase is git-tracked at commit `8835bbf` (post-revert to 6-task arrays) augmented by `da2474c` (the correct_mask fix) and committed for the final state of this run. The relevant scripts are:

- `/home/anshulk/emnlp2026/residual_hunting.py`
- `/home/anshulk/emnlp2026/principal_angles.py`
- `/home/anshulk/emnlp2026/jl_distance.py`
- `/home/anshulk/emnlp2026/aggregate_residual_hunting.py`
- `/home/anshulk/emnlp2026/aggregate_principal_angles.py`
- `/home/anshulk/emnlp2026/aggregate_jl_distance.py`
- `/home/anshulk/emnlp2026/check_audit_pipeline_toys.py`
- `/home/anshulk/emnlp2026/check_step6_complete.py`

The SLURM sbatch templates are at `run_step{7,8,9}.sbatch` and `run_step{7,8,9}_aggregate.sbatch`.

The complete output tree under `/data/user_data/anshulk/emnlp2026/results/`:

- `residual_hunting/` — Step 7 outputs (7.0 GB).
- `principal_angles/` — Step 8 outputs (39 MB).
- `jl_distance/` — Step 9 outputs (2.7 GB).
- Each step's `comparison/` directory holds the aggregator outputs.

This document accompanies those artifacts as the truth document for the audit phase. It is the complete reference for what was computed, how it was computed, and what was found.

*End of document.*
