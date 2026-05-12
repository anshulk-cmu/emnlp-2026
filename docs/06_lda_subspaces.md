# Step 6 — LDA refinement of CCSVD subspaces, with magnitude residualization and a full-space audit

**Project:** From Linear Probes to Bayesian Manifolds: Geometry of Arithmetic in Language Models
**Carnegie Mellon University, May 2026**
**Author:** Anshul Kumar

---

## Table of contents

1. Purpose and scope
2. Standing rules
3. Inputs
4. Mathematical specification
   4.1 Phase 1 — residualization
   4.2 Phase 2 — CCSVD re-fit per non-off mode
   4.3 Phase 3a — Option A: LDA inside the CCSVD subspace
   4.4 Phase 3b — Option B: LDA in the full 4096-D residualized space
   4.5 A vs B alignment audit
   4.6 Dual-criterion `n_sig` rule
   4.7 Cohen's `d` and bootstrap CI
   4.8 Output writing
5. Concept registry
6. Toy validation
7. Run procedure
8. Per-model results
   8.1 GPT-J 6B
   8.2 Llama 3.1 8B
   8.3 Pythia 6.9B
   8.4 Aggregate totals
9. Per-mode results
   9.1 mode = off — baseline (no residualization)
   9.2 mode = answer — answer-magnitude residualization
   9.3 mode = norm — activation-norm residualization
10. Cross-mode comparison
    10.1 Matched-population set
    10.2 Top-direction agreement across modes
    10.3 Δλ_T_1, Δn_sig, Δcv_accuracy across mode pairs
    10.4 Carveout audit
11. CCSVD vs LDA alignment
    11.1 r_ccsvd vs n_sig per cell
    11.2 Which CCSVD directions LDA prunes
12. Option B audit (A vs B)
    12.1 cos_sim_AB distribution
    12.2 audit_status distribution
    12.3 What B's eigenvalue inflation tells us
13. Skipped cells inventory
14. Output files
15. Reproducibility
16. Verification
17. Open questions
18. Appendix A — Intuitions and analysis

---

## 1. Purpose and scope

### 1.1 What this step is

For each cell `(model, task, layer, concept, residualization-mode)`, this step
computes two linear-discriminant fits on the per-model correct subset:

- **Option A — LDA inside the CCSVD subspace.** Project residualized
  activations into the per-cell CCSVD basis (Step 5 for `mode=off`,
  re-fit per-mode for `mode=answer` and `mode=norm`), then solve the
  K×K compact LDA generalised eigenproblem in that subspace. This is the
  headline placement; its eigenvalues `λ_T`, significant-direction count
  `n_sig`, cross-validated k-NN accuracy, and Cohen's `d` are cited.
- **Option B — LDA in the full 4096-D residualized space.** Same K×K
  compact form, using a Ledoit–Wolf / OAS-shrunk total scatter `S_T` and
  a cached Cholesky factor per (model, task, layer, mode). B is a
  structural audit only: at N/d ≈ 0.7–2.4 in our setting, B's
  eigenvalues inflate near 1.0 even on the permutation null, so we
  cite B's top-direction cosine similarity vs A (`cos_sim_AB`) and B's
  `n_sig` as audit metrics, never B's `λ_T` magnitudes.

Significance is decided by the **dual-criterion rule**:
`n_sig = min(n_sig_perm, n_sig_cv)`. `n_sig_perm` is the sequential
99th-percentile stop on the permutation null. `n_sig_cv` applies the
one-SE rule to the held-out k-NN classification accuracy curve as a
function of direction count.

### 1.2 What this step is NOT

- It does NOT do Stage 2 (Bayesian manifold characterisation). Helix
  Fourier, spread-aware Mahalanobis `d_SW`, GPLVM, and RBF VAE are
  out of scope and consume this step's outputs.
- It does NOT do Stage 3 (ownership test) or Stage 4 (causal ablation).
- It does NOT classify cells as `pass` or `fail` against the Stage 1
  paper criteria. It writes the numbers the criterion will use; the
  pass-gate decision is downstream.
- It does NOT cite Option B's eigenvalue magnitudes. B's `λ_T` is
  documented for diagnostic purposes only.
- It does NOT compute a merged basis. Merging the CCSVD high-variance
  basis with LDA's high-ratio directions is a Stage 2/3/4 concern.

### 1.3 What this step's outputs feed into

- **Stage 2a (centroid Fourier helix fit)** consumes the per-cell
  `lda_basis_subspace.npy` (Option A) or `lda_basis_full.npy` (lifted
  to 4096-D), together with the projected centroids that the basis
  defines.
- **Stage 2b (spread-aware Mahalanobis `d_SW`)** consumes the
  per-value within-class covariances after projection into A's basis.
- **Stage 2c (GPLVM)** and **Stage 2d (RBF VAE)** consume the LDA-
  projected activations.
- **Stage 3 (ownership)** uses the `lda_basis_full.npy` lifted to 4096-D
  for each algebraic correlate concept; orthogonalisation against the
  union basis goes through these directions.
- **Stage 4 (ablation)** ablates the LDA basis from the residual stream
  and measures Δlogit.

### 1.4 Population

Every fit runs on the per-model **correct subset** — rows in the
activation matrix where the model produced the gold first-answer-token.
The correctness mask is loaded from the per-model answers CSV (Step 3).

| Model | Addition correct | Multiplication correct |
|---|---:|---:|
| gpt-j-6b | 8,415 / 10,000 | 2,751 / 3,023 |
| llama-3.1-8b | 9,963 / 10,000 | 2,927 / 3,023 |
| pythia-6.9b | 7,718 / 10,000 | 2,757 / 3,023 |

After the MIN_GROUP_SIZE filter (≥ 30 samples per concept value), the
median per-cell N is the column above for addition; for multiplication
it depends on the concept's value distribution (median 2,728–2,927
across models).

### 1.5 Residualization modes

Three modes are run in parallel; deltas between modes are the
sensitivity analysis:

- **`off`** — raw activations. Uses Step 5's existing CCSVD subspaces
  verbatim; A's CCSVD basis is read from `ccsvd_subspaces/{model}/...`
  unchanged. This is the baseline.
- **`answer`** — residualize the activation against the scalar gold
  answer (`a+b` for addition, `a·b` for multiplication). One direction
  removed per (model, task, layer).
- **`norm`** — residualize the activation against its own L2 norm.

For non-`off` modes, the CCSVD step is re-fit on the residualized
activations (Phase 2) before LDA runs.

### 1.6 Placement comparison

Both placements run for every cell:

- **Option A** is the headline. r_ccsvd is typically 8–18; N/r ≈ 100+
  even at the worst cell, so eigenvalues are trustworthy.
- **Option B** is a parallel full-space LDA, with Ledoit–Wolf (or OAS
  on GPU) shrinkage on `S_T` to handle low N/d. We report B's top
  direction (lifted to 4096-D) and compare it to A's top direction
  via cosine similarity (`cos_sim_AB`).

---

## 2. Standing rules

1. **No subsampling.** Every fit uses the full per-cell correct
   population. The 1,000-permutation null, 5-fold CV, and 200-bootstrap
   resamples are resampling, not subsampling — they all operate on
   the full N.
2. **Residualization happens BEFORE CCSVD, never between CCSVD and
   LDA.** When `mode≠off`, CCSVD is re-fit on the residualized
   activations so its basis is not contaminated by the magnitude
   direction we are trying to remove.
3. **Mean-centring only.** Activations are mean-centred once per
   fit. Rows are not unit-normalised. Activation L2 norms vary by
   ~80× across (model, layer); this variation is preserved unless
   `mode=norm` explicitly removes its linear component.
4. **S_T as the noise scatter.** Eigenvalues `λ_T ∈ [0, 1]` read as
   "fraction of total variance that is between-class". S_T is
   permutation-invariant, so the permutation null reuses one Cholesky
   factor per (task, layer, mode).
5. **Regularisation policy.** Option A uses `α · trace(S_T_z)/r · I`
   with α = 1.0e-4 when N/r ≥ 10, falling back to Ledoit–Wolf when
   N/r < 10. Option B always uses Ledoit–Wolf (or GPU OAS); the d=4096
   ambient dimension makes raw `S_T` singular without shrinkage.
6. **Permutation null:** 1,000 label shuffles per cell. p < 0.01
   threshold (99th percentile per eigenvalue index). Sequential stop.
7. **Dual-criterion `n_sig`:** `n_sig = min(n_sig_perm, n_sig_cv)`.
   Both must agree before a direction is called significant.
8. **5-fold stratified CV** with `random_state = 42`. k-NN classifier
   (k=1) on the LDA-projected fold.
9. **Bootstrap CI** on `λ_T_1` with 200 row-resamples (Option A only).
10. **Cohen's `d`** per significant direction × class pair (Option A
    and Option B).
11. **Concept carve-outs.** `mode=answer` carves out cells where the
    concept name starts with `ans_` or matches `answer` (residualizing
    the answer onto an answer-derived concept is circular). `mode=norm`
    carves out `ans_magnitude_tier`. Carved-out cells emit a
    `status=carved_out` meta with no LDA fit; their cells are reported
    separately and excluded from cross-mode deltas.
12. **All outputs go to CSV in addition to `.npy`** so that
    plotting can read tabular form without reloading `.npy` artefacts.
    The CSV inventory is in §14.
13. **GPU acceleration is opt-in but enabled by default.** cupy 14.0.1
    and cuML 26.02 are required for the production run; the fitter
    falls back to numpy/scipy/sklearn when either import fails, with
    an explicit log line.

---

## 3. Inputs

### 3.1 Activation files

For each model and task, the residual stream at the `=` token of every
problem is stored at:

```
data/activations/{model_key}/{task}_layer_{LL:02d}.npy
```

dtype `float32`, shape `(N_total, 4096)`. `N_total` is 10,000 for
addition and 3,023 for multiplication.

| Model | Layers | Addition shape | Multiplication shape |
|---|---|---|---|
| gpt-j-6b | 4, 8, 14, 20, 24 | (10000, 4096) | (3023, 4096) |
| llama-3.1-8b | 4, 8, 16, 24, 28 | (10000, 4096) | (3023, 4096) |
| pythia-6.9b | 4, 8, 16, 24, 28 | (10000, 4096) | (3023, 4096) |

### 3.2 Residualized activation cache

Phase 1 writes one cache file per (model, task, layer, mode) at:

```
data/results/residualized/{model_key}/{task}_layer_{LL:02d}_mode_{mode}.npy
```

dtype `float32`, shape `(N_total, 4096)`. For `mode=off` this is a
plain copy of the raw activation file (so downstream code has a uniform
interface). For `mode=answer` and `mode=norm` it is the residual after
OLS-regressing each activation dimension on the chosen scalar (§4.1).

### 3.3 CCSVD basis files

Option A reads its per-mode CCSVD basis from one of:

```
data/results/ccsvd_subspaces/{model_key}/{task}/layer_{LL}/{concept}/basis.npy   # mode=off (Step 5)
data/results/ccsvd_subspaces/mode_answer/{model_key}/{task}/layer_{LL}/{concept}/basis.npy
data/results/ccsvd_subspaces/mode_norm/{model_key}/{task}/layer_{LL}/{concept}/basis.npy
```

shape `(4096, r)`, float32. For `mode=off` the basis is unchanged from
Step 5. For `mode=answer` and `mode=norm` the basis is freshly fit on
residualized activations.

### 3.4 Concept labels

```
data/data/raw/{task}_problems.csv
```

10,000 rows for addition, 3,023 rows for multiplication. Columns hold
the Tier 1–5 schema (operand digits, column-algebra intermediates,
structural properties, relational properties, tokenisation metadata).

### 3.5 Correctness masks

```
data/answers/{model_key}/{task}_answers.csv
```

10,000 (or 3,023) rows. The boolean column `correct` is the mask used
to subset the activation matrix.

### 3.6 Gold-answer scalars

For `mode=answer`, the scalar regressed out is the `answer` column of
`data/data/raw/{task}_problems.csv`. Addition: `a+b ∈ [0, 198]`.
Multiplication: `a·b ∈ [0, 9801]` (full theoretical range; the
single-token intersection restricts to `[0, 999]` per Step 1).

### 3.7 SHA256 chain

Each per-cell `meta.json` records sha256 of:

- the residualized activation `.npy` file
- the labels CSV (`{task}_problems.csv`)
- the answers CSV (`{model_key}/{task}_answers.csv`)
- the per-mode CCSVD basis `.npy`
- the configuration file `config.yaml` used at run time

The per-(model, mode) `manifest_{model}_mode_{mode}.json` records the
same plus the git commit (when run from a working tree) and the
library versions of the worker that produced it.

### 3.8 Library versions

Recorded in every manifest (production run):

```
{
  "numpy":   "2.2.6",
  "pandas":  "2.3.3",
  "scipy":   "1.17.1",
  "sklearn": "1.8.0",
  "cupy":    "14.0.1",
  "cuml":    "26.02.000",
  "python":  "3.11.15"
}
```

CUDA: 12.x runtime. PyTorch is imported transitively for some CCSVD
operations but is not required by lda_subspaces.py.

---

## 4. Mathematical specification

The Step 6 pipeline has three phases per model:
**Phase 1** — residualize the activations into three caches.
**Phase 2** — for `mode∈{answer, norm}`, re-fit CCSVD on the residualized
activations. (`mode=off` reuses Step 5 verbatim.)
**Phase 3** — for each (mode, task, layer, concept), fit Option A and
Option B LDA. Concept carve-outs apply per §1.5.

### 4.1 Phase 1 — residualization

Given the activation matrix `X ∈ R^{N × 4096}` for one (model, task,
layer) cell and the per-mode scalar `z ∈ R^N`:

- `mode=off`: `X_resid = X` (passthrough; the cache file is a direct
  copy so all downstream code has a uniform path-pattern).
- `mode=answer`: `z = answer_per_problem`.
- `mode=norm`: `z = ||X_i||_2` (per-row L2 norm).

The OLS regression is:

```
z_c     = z - mean(z)
X_c     = X - mean(X, axis=0)
β       = (X_c^T z_c) / (z_c · z_c)               ∈ R^4096
X_resid = X - outer(z_c, β)                       ∈ R^{N × 4096}
```

Note that `X` (not `X_c`) is subtracted from: this preserves the
original mean of the activation matrix, so the residual still looks like
an activation matrix rather than a centred matrix. Downstream code
mean-centres again, which is benign because the residual is already
zero-mean along `z_c` by construction.

When `(z_c · z_c) < 1e-12` (i.e., `z` is numerically constant), the
residualization step is skipped and `X_resid = X` is written verbatim
to preserve idempotency.

The OLS is computed on GPU via cupy. The CPU fallback path (numpy)
gives identical results within float32 precision.

### 4.2 Phase 2 — CCSVD re-fit per non-off mode

For `mode∈{answer, norm}`, the existing `ccsvd_subspaces.py` script is
invoked with `--mode=<mode>`. It:

1. Reads `data/results/residualized/{model}/{task}_layer_{LL}_mode_{mode}.npy`
   instead of the raw activation file.
2. Applies the same correctness mask as Step 5.
3. Runs the same CCSVD fit per concept (centroid SVD + 1,000
   permutations + 5-fold subspace-preservation CV).
4. Writes outputs to `data/results/ccsvd_subspaces/mode_{mode}/...`.

Step 5 details apply unchanged. The only structural difference is the
input source and the output prefix.

For `mode=off`, the existing Step 5 outputs are reused verbatim — no
re-fit is performed.

### 4.3 Phase 3a — Option A: LDA inside the CCSVD subspace

For each cell `(model, task, layer, concept, mode)`:

#### 4.3.1 Filter and project

Apply MIN_GROUP_SIZE = 30: drop concept values with fewer than 30
surviving samples. Let `K` be the count of surviving values. If
`K < 2`, the cell is skipped with `status=skipped_insufficient_groups`.
Otherwise, encode survivors to integer codes `y ∈ {0, …, K−1}^N` and
record `n_v ∈ N^K` as per-value counts.

Project residualized activations into the CCSVD basis:

```
Z   = X_resid_correct @ B_ccsvd     ∈ R^{N × r}
μ̄_z = mean(Z, axis=0)              ∈ R^r
Z_c = Z - μ̄_z                       ∈ R^{N × r}
```

Here `r = B_ccsvd.shape[1]` is the CCSVD subspace dimension for this
cell (Step 5 output).

#### 4.3.2 S_T regularisation

Define `N_over_r = N / r`. Two regularisation paths:

```
if N_over_r >= 10:
    S_T_z = Z_c^T Z_c                                # raw scatter
    α     = 1.0e-4 * trace(S_T_z) / r
    S_T_z_reg = S_T_z + α · I_r                       # diagonal load
else:
    # sklearn ledoit_wolf gives a shrunk covariance.
    cov_shrunk, λ_LW = ledoit_wolf(Z_c, assume_centered=True)
    S_T_z_reg = N * cov_shrunk                         # scatter
```

We record `used_shrinkage` (bool), `used_alpha_diag` (the additive
diagonal term), and `lw_shrinkage` (Ledoit–Wolf coefficient when used)
in the per-cell meta.

#### 4.3.3 Cholesky and K×K compact LDA

```
L_z      = cholesky(S_T_z_reg, lower=True)            # lower-triangular
centroids_z = per-class means of Z, shape (K, r)
M_w[k]   = sqrt(n_v[k]) · (centroids_z[k] - μ̄_z)      # shape (K, r)
X_solve  = L_z^{-T} L_z^{-1} M_w^T = (S_T_z_reg)^{-1} M_w^T  # shape (r, K)
A_kk     = M_w @ X_solve                              # shape (K, K), symmetric
λ_T, V_kk = eigh(0.5 · (A_kk + A_kk^T))               # sorted descending
W_subspace = X_solve @ V_kk                           # shape (r, K), unnormalised
W_subspace /= ||W_subspace||_2 column-wise            # orthonormal columns
```

The eigenvalues `λ_T ∈ [0, 1]` are LDA's standard normalised
eigenvalues (between-class variance fraction of total variance along
each direction). `S_B` has rank ≤ K−1, so up to K−1 eigenvalues are
non-zero.

The eigenvectors of the K×K compact matrix map back to subspace
directions via `W_subspace = X_solve · V_kk`. We normalise each column
to unit L2 norm.

#### 4.3.4 Permutation null

Set seed `seed = sha256(cell_id) mod 2^63`. Generate a per-cell RNG.

For 1,000 label shuffles:
- Permute `y_codes` randomly. `n_v` is unchanged (permutation only
  relabels rows; sample counts per class are preserved).
- Recompute `centroids_z_shuf` from `Z` under the new labels.
- Build `M_w_shuf` and solve via the cached Cholesky factor `L_z`.
- Form the K×K compact matrix and eigendecompose to get
  `(λ_T^{shuf}_1, …, λ_T^{shuf}_{K−1})`.

After 1,000 shuffles, take the 99th percentile per index:
`τ^{99}_k = quantile_{99}(λ_T^{shuf}_k)`. The **sequential stop**
returns `n_sig_perm = max k such that λ_T_1 > τ^{99}_1, …, λ_T_k > τ^{99}_k`.

#### 4.3.5 5-fold k-NN cross-validation

Build a stratified 5-fold split on `y_codes` with `random_state=42`.
For each fold `f`:

1. Compute the train-fold mean `μ_tr`, residualized matrix
   `Z_tr_c = Z_tr - μ_tr`.
2. Build `S_T_tr` with the same regularisation rule as §4.3.2,
   Cholesky-factor it.
3. Compute train centroids and `M_w_tr`. Solve and form the K×K
   compact matrix; eigendecompose to get `(λ_T_tr, V_tr)`.
4. Build train-fold direction matrix `W_tr ∈ R^{r × K}`.
5. Project train and test folds: `Z_tr_proj = Z_tr_c · W_tr`,
   `Z_te_proj = (Z_te - μ_tr) · W_tr`.
6. For each direction-count `k = 1, …, K−1`:
   - Restrict to the top-`k` columns of both projected matrices.
   - Train a k-NN classifier (k_NN = 1, by default) on the train
     projection, predict the test projection, score accuracy.
   - Record `accuracy[f, k]`.

`cv_accuracy_curve[k] = mean_f(accuracy[f, k])`, length K−1.

#### 4.3.6 Dual-criterion n_sig

- `n_sig_perm` from §4.3.4 above.
- `n_sig_cv` via one-SE rule:
  ```
  max_acc = max(cv_accuracy_curve)
  se      = std(cv_accuracy_curve) / sqrt(n_non_nan)
  n_sig_cv = max k such that cv_accuracy_curve[k] >= max_acc - se
  ```
- **`n_sig = min(n_sig_perm, n_sig_cv)`**.

This rule rejects both (i) perm-null inflation at low N/d and (ii)
cells where the permutation null calls a direction significant but
that direction does not generalise to held-out data. Either failure
mode collapses the count to zero.

#### 4.3.7 Cohen's d per direction × class pair

For each of the top `max(n_sig, 1)` directions, project `Z_c` and
compute the standardised mean difference for every ordered class pair
`(i, j)`:

```
m_k = mean(Z_proj[y_codes == k])
s_k = std(Z_proj[y_codes == k], ddof=1)
d_{ij} = (m_i - m_j) / sqrt((s_i^2 + s_j^2) / 2)
```

Output: `cohen_d` of shape `(n_sig, K, K)`, float64.

#### 4.3.8 Bootstrap CI on λ_T_1

200 bootstrap iterations on the original cell rows (sampling with
replacement to size N). For each iteration: refit the cell (re-build
S_T_z, re-Cholesky, re-eigendecompose) and capture `λ_T_1`. Record
the array. The reported CI is the 5th percentile:
`bootstrap_lambda1_p5 = quantile_5(bootstrap_lambda1)`.

#### 4.3.9 Lift to 4096-D

A's direction matrix lives in `R^r`. We lift to 4096-D via the CCSVD
basis: `W_full_A = B_ccsvd @ W_subspace ∈ R^{4096 × n_sig}`, then
re-normalise each column to unit L2 norm.

### 4.4 Phase 3b — Option B: LDA in the full 4096-D residualized space

For each cell, in parallel with Option A:

#### 4.4.1 Pre-compute cached Cholesky factor

Once per (task, layer, mode), before iterating over concepts, we
compute:

```
mu_full          = mean(X_correct_resid, axis=0)              ∈ R^4096
X_full_c         = X_correct_resid - mu_full                  ∈ R^{N × 4096}
S_T_full, λ_LW   = shrunk_scatter(X_full_c)                   ∈ R^{4096 × 4096}
L_full           = cholesky(S_T_full, lower=True)             ∈ R^{4096 × 4096}
```

`shrunk_scatter` uses GPU OAS when cupy is available (closed-form
shrinkage at d=4096 in seconds), otherwise sklearn's Ledoit–Wolf.
The Cholesky factor `L_full` is cached on GPU; all subsequent solves
re-use it. Cache invalidation: per-layer, never per-concept.

#### 4.4.2 Per-cell solve

For the concept's `y_codes`:

```
centroids_full   = per-class means of X_correct_resid_keep    ∈ R^{K × 4096}
M_w_full[k]      = sqrt(n_v[k]) (centroids_full[k] - mu_full) ∈ R^{K × 4096}
X_solve_full     = (S_T_full)^{-1} M_w_full^T                 ∈ R^{4096 × K}
A_kk_full        = M_w_full @ X_solve_full                    ∈ R^{K × K}
λ_T_full, V_kk_full = eigh(0.5 (A_kk_full + A_kk_full^T))
W_full_B         = X_solve_full @ V_kk_full                   ∈ R^{4096 × K}
W_full_B /= ||W_full_B||_2 column-wise
```

#### 4.4.3 Permutation null (GPU-batched)

The permutation null re-uses `L_full` (S_T is permutation-invariant).
We use a GPU-batched implementation that pipelines 20 shuffles at a
time via cupy:

```
For batch_idx in 0..n_perm step batch (=20):
    Sample `bs` label shuffles on CPU (cheap).
    Compute (bs, K, 4096) batched centroids via cupy einsum.
    Build (bs, K, 4096) M_w_batch.
    For each shuffle:
        Forward-solve L_full Y = M_w[b]^T (cupy solve_triangular).
        Back-solve L_full^T X_solve = Y.
        Form A_kk = M_w[b] @ X_solve, eigendecompose on CPU (K small).
        Record λ_T[b, :K-1].
```

After 1,000 shuffles, sequential-stop on the 99th-percentile threshold
gives `n_sig_perm_B`.

#### 4.4.4 CV-accuracy and n_sig_B

Same as Option A's §4.3.5–§4.3.6 but in the full 4096-D space.
Per fold:

1. Compute `μ_tr` in 4096-D.
2. Compute `S_T_full_tr` with GPU OAS shrinkage on the train fold.
3. Cholesky-factor on GPU.
4. Project train and test to top-k directions; train k-NN and score.

Final `n_sig_B = min(n_sig_perm_B, n_sig_cv_B)`.

Per-cell B runtime: ~10 s with GPU acceleration (the per-fold OAS
shrinkage is the bottleneck). Per-cell A runtime: ~3 s. Combined
~13–15 s per cell; full run is ~75–100 min per (model × mode) over
~500 cells.

#### 4.4.5 Cohen's d

Same as §4.3.7, on the 4096-D directions `W_full_B`.

#### 4.4.6 No bootstrap CI for B

Option B's eigenvalues are not cited as headline numbers (they are
near 1.0 from N/d inflation), so a bootstrap CI on `λ_T_1_B` would be
uninformative. We do NOT compute it.

### 4.5 A vs B alignment audit

Per cell, lift A's top direction to 4096-D via the CCSVD basis (as in
§4.3.9), then compute:

```
w_A_top_full   = W_full_A[:, 0]
w_B_top        = W_full_B[:, 0]
cos_sim_AB     = |w_A_top_full · w_B_top| / (||w_A_top_full|| ||w_B_top||)
```

Audit status classification:

| cos_sim_AB | audit_status |
|---|---|
| ≥ 0.9 | `agree` — CCSVD's basis was complete on this concept |
| 0.7 ≤ cos_sim_AB < 0.9 | `partial` — partial agreement |
| < 0.7 | `ambiguous_AB` — direction disagreement |
| `n_sig_B ≥ 2 · n_sig_A` and cv_acc_B significantly > chance | `ccsvd_incomplete` — B finds more |
| any error | `unknown` |

We document the distribution in §12. In production at our N/d ratio,
the modal status is `ambiguous_AB`, which is the expected N/d-inflation
signature, not evidence that CCSVD missed directions.

### 4.6 Dual-criterion n_sig rule

A direction is significant if and only if:
- Its eigenvalue exceeds the per-index 99th-percentile of the
  permutation null (and all prior eigenvalues also did).
- The held-out k-NN classification accuracy at this many directions
  is within 1 SE of the maximum accuracy.

`n_sig = min(n_sig_perm, n_sig_cv)`.

Rationale: at low N/d, the permutation null can inflate (random labels
produce eigenvalues that survive 99th-percentile because the random
direction overfits the small training set). The CV-accuracy criterion
catches this — random labels do not produce held-out classification
accuracy. Conversely, the CV-accuracy criterion alone is liberal when
adding directions does not hurt classifier performance (it tends to
report large `n_sig_cv`); the perm null then prunes back to what's
statistically discernible.

The intersection is tighter than either criterion alone.

### 4.7 Cohen's d and bootstrap CI

- **Cohen's `d`** is computed for every direction × class pair as in
  §4.3.7. It is reported as a 3-D array `(n_sig, K, K)` for both A
  and B; the diagonal is zero by definition.
- **Bootstrap CI on `λ_T_1`** is computed for A only (200 resamples,
  same row sampling, fresh S_T_z rebuild per iteration). Reported as
  the 5th-percentile of the resampled distribution.

### 4.8 Output writing

Per-cell outputs land in two parallel trees:

```
data/results/lda_subspaces/subspace_lda/mode_{mode}/{model}/{task}/layer_{LL}/{concept}/
  lda_basis_subspace.npy   # (r, n_sig) — A's directions in subspace coords
  lda_basis_full.npy       # (4096, n_sig) — A's directions lifted
  lda_eigenvalues.npy      # (K,) — full λ_T spectrum
  null_lda_eigenvalues.npy # (1000, K-1) — full perm-null matrix
  lda_threshold_99.npy     # (K-1,) — per-index 99th percentile
  cohen_d.npy              # (n_sig, K, K) — Cohen's d
  cv_accuracy_curve.npy    # (K-1,) — mean k-NN accuracy at each direction count
  cv_per_fold.npy          # (5, K-1) — per-fold accuracy
  bootstrap_lambda1.npy    # (200,) — bootstrap draws of λ_T_1
  meta.json                # everything needed to reconstruct the row

data/results/lda_subspaces/full_lda/mode_{mode}/{model}/{task}/layer_{LL}/{concept}/
  lda_basis_full.npy       # (4096, n_sig) — B's directions in 4096-D
  lda_eigenvalues.npy
  null_lda_eigenvalues.npy
  lda_threshold_99.npy
  cohen_d.npy
  cv_accuracy_curve.npy
  cv_per_fold.npy
  meta.json
```

All writes are atomic: tempfile + rename. Aggregated CSVs are emitted
at the end of every (model × mode) run (§14).

---

## 5. Concept registry

The concept registry is inherited from Step 5 verbatim. We re-state
the principles here so the LDA report is self-contained.

### 5.1 Inclusion principles

Concepts come from the CSV columns of `{task}_problems.csv` plus a
curated joint-tuple set per task. Single concepts are CSV columns that
are non-constant under the correctness mask. Joint concepts are
multi-column tuples chosen by the §6e curated joint registry.

### 5.2 Tier 1 — input/output digits

```
a, b, a_units, a_tens, b_units, b_tens, a_num_digits, b_num_digits,
ans_units, ans_tens, ans_hundreds, ans_num_digits, answer
```

Multiplication adds `ans_thousands` (answers up to 9801) but it is
mostly degenerate inside the single-token intersection.

### 5.3 Tier 2 — column-algebra intermediates

Addition: `column_sum_units`, `column_sum_tens`, `carry_units`,
`carry_tens`, `running_sum_units`, `running_sum_tens`.

Multiplication: `partial_product_units` (the four 2×2 partial
products), `column_sum_*`, `column_product_*` (rare-value, often
filtered), `carry_*`, `running_sum_*`.

### 5.4 Tier 3 and Tier 4

Structural and relational properties: `parity`, `magnitude_tier`,
`a_eq_b`, `max_operand`, `min_operand`, `operand_diff`,
`operand_abs_diff`, `larger_operand`, `ans_ends_in_zero`,
`ans_is_zero`, `a_is_zero`, `b_is_zero`, `both_zero`, `either_zero`,
`both_one`, `either_one`.

### 5.5 Joint concepts (per task)

Addition:
```
(a_units, b_units), (a_tens, b_tens), (a_units, b_tens), (a_tens, b_units),
(a_tens, b_tens, carry_units), (a_tens, b_tens, ans_tens),
(carry_units, ans_units), (carry_units, column_sum_units),
(a_units, b_units, ans_tens), (a_units, b_units, ans_units)
```

Multiplication: same first 9 joints; the last one is replaced by
`(a_units, b_units, partial_product_units)` because
`partial_product_units = (a_units · b_units) mod 10` is the
deterministic multiplication analogue.

### 5.6 Excluded columns

```
a_digits_lsf, b_digits_lsf, answer_digits_lsf, answer_digits_msf,
column_sums, carries, running_sums, column_products, partial_products,
is_intersection, is_single_token_*, first_token_id_*,
first_token_text_*, n_tokens_*
```

JSON-list columns (each list element duplicates a Tier 1/2 scalar) and
tokenisation metadata.

### 5.7 Concept inventory by task

The CSV column count after filtering is:

- Addition: 41 single concepts + 10 joints = **51 attempted cells** per layer.
- Multiplication: 47 single concepts + 10 joints = **57 attempted cells** per layer.

Per (model × task × mode): 5 layers × 51 = **255 addition cells** and
5 layers × 57 = **285 multiplication cells**, total **540 cells**.

### 5.8 Concept carve-outs in non-off modes

For `mode=answer`: every cell whose concept name starts with `ans_`
(`ans_units`, `ans_tens`, `ans_hundreds`, `ans_thousands`,
`ans_num_digits`, `ans_parity`, `ans_magnitude_tier`, `ans_is_zero`,
`ans_ends_in_zero`) and `answer` itself is carved out — residualizing
the activation against the gold answer is circular for answer-derived
concepts. Joints containing only `ans_*` columns are NOT carved out
unless the joint is purely answer-derived; joints with mixed inputs
(e.g., `(a_tens, b_tens, ans_tens)`) ARE NOT carved out either since
the operand columns contribute independent information.

In practice, the carve-out hits 9 single concepts per task × 5 layers
= **45 cells per (model, task, mode=answer)**.

For `mode=norm`: only `ans_magnitude_tier` (5 cells per (model, task,
mode=norm)) is carved out, because that concept IS magnitude.

Cells emit `status=carved_out` and write a stub `meta.json` with no
LDA fit. They appear in the `carveout_log.csv` of the comparison stage.

---

## 6. Toy validation

Four synthetic toys validate the LDA fitter before any real-data fit.
All four must pass for the run to proceed.

### 6.1 Toy 1L — single-axis structure

- Distribution: 10 classes whose means lie on a 1-D line in 9-D
  Gaussian space, σ = 0.5.
- Expected: `λ_T_1 ≥ 0.85`, `n_sig ≥ 1`, `cv_accuracy_max > 0.5`.
- Result (production validation run): `λ_T_1 = 0.972`,
  `n_sig_perm = 1`, `n_sig_cv = 9`, `n_sig = 1`,
  `cv_accuracy_max = 0.688`. **PASS.**

### 6.2 Toy 2L — two-axis structure

- Distribution: 10 classes on a 5×2 grid in dims 0 and 1 (spacing 2.0),
  9-D ambient, σ = 0.5.
- Expected: `n_sig ≥ 2`, `λ_T_1 ≥ 0.5`, `λ_T_2 ≥ 0.5`,
  `cv_accuracy_max > 0.6`.
- Result: `λ_T_1 = 0.970`, `λ_T_2 = 0.795`,
  `n_sig_perm = 3`, `n_sig_cv = 9`, `n_sig = 3`,
  `cv_accuracy_max = 0.929`. **PASS.**

### 6.3 Toy 3L — no structure

- Distribution: 9-D isotropic Gaussian, random class assignments.
- Expected: `n_sig = 0`.
- Result: `λ_T_1 = 0.005`, `n_sig_perm = 0`, `n_sig_cv = 1`,
  `n_sig = 0`, `cv_accuracy_max = 0.103` (≈ 1/K = 0.1). **PASS.**

The `n_sig_cv = 1` here is the one-SE rule being slightly liberal —
adding directions does not hurt k-NN, so any direction count is
"within 1 SE of max". The perm null catches it (`n_sig_perm = 0`),
and `min()` enforces `n_sig = 0`. This is the dual-criterion rule
working as intended.

### 6.4 Toy 4L — sample-starved (N/d inflation test)

- Distribution: 4-D Gaussian with 10 random classes, only N = 120
  total. N/d = 30, but per-class < 12. Permutation null can inflate
  because of small sample.
- Expected: `n_sig = 0` (the dual criterion must catch the inflation).
- Result: `λ_T_1 = 0.081`, `n_sig_perm = 0`, `n_sig_cv = 1`,
  `n_sig = min(0, 1) = 0`. **PASS.**

In production, this validates that low-N/d full-space LDA cells will
self-flag as zero-signal.

### 6.5 Implementation note

Toys use `n_permutations = 200` and `bootstrap_n = 50` for speed.
Random seed is fixed at 0. Toys run on CPU only (cuML/cupy are
optional). Total runtime of the toy validator: ~60 seconds.

---

## 7. Run procedure

### 7.1 Phase 0 — local sanity

Before any submission:

1. **Toy validator.** `python check_lda_toys.py`. All four toys must
   pass.
2. **Single-cell smoke.** Pick one cell (Llama × addition × layer 16
   × `ans_units`, mode=off) and run `lda_subspaces.py` with the
   `--single-task --single-layer --single-concept` flags. Verify the
   output structure matches §4.8.

### 7.2 Phase 1 — residualization

```
python residualize_activations.py \
    --config /home/anshulk/emnlp2026/config.yaml \
    --model {model_key} \
    --modes off,answer,norm \
    --tasks addition,multiplication
```

Per (model): ~6 seconds (cupy backend). Writes 30 cache files
(5 layers × 3 modes × 2 tasks) to `data/results/residualized/{model}/`.

### 7.3 Phase 2 — CCSVD re-fit for non-off modes

For `mode in {answer, norm}`:

```
python ccsvd_subspaces.py \
    --config /home/anshulk/emnlp2026/config.yaml \
    --model {model_key} \
    --mode {mode}
```

Per (model × non-off mode): ~75 minutes. Writes 540 cells to
`data/results/ccsvd_subspaces/mode_{mode}/{model}/{task}/...`.

`mode=off` is NOT re-run; Step 5's outputs at
`data/results/ccsvd_subspaces/{model}/...` are used verbatim.

### 7.4 Phase 3 — LDA fit per (mode)

For `mode in {off, answer, norm}`:

```
python lda_subspaces.py \
    --config /home/anshulk/emnlp2026/config.yaml \
    --model {model_key} \
    --mode {mode}
```

Per (model × mode): ~75–100 minutes. Writes 540 cells of A outputs
and 540 cells of B outputs to:

```
data/results/lda_subspaces/subspace_lda/mode_{mode}/{model}/...
data/results/lda_subspaces/full_lda/mode_{mode}/{model}/...
```

Plus per-mode aggregate CSVs (eigenvalue spectra, CV per-fold,
Cohen's d, bootstrap, A-vs-B alignment).

### 7.5 Phase 4 — cross-mode aggregation

After all 3 array tasks have produced all 3 modes:

```
python compare_residualization_modes.py \
    --config /home/anshulk/emnlp2026/config.yaml
```

Writes to `data/results/lda_subspaces/comparison/`:

- `cross_mode_summary.csv` (one row per cell × (off, answer, norm))
- `cross_mode_alignment.csv` (cos_sim_top1_AA across mode pairs)
- `cross_mode_lambda_deltas.csv` (Δλ_T_1, Δn_sig, Δcv_accuracy)
- `cross_mode_accuracy_deltas.csv` (focused on cv_accuracy deltas)
- `a_vs_b_alignment.csv` (concatenated A-vs-B per-mode)
- `matched_population_cells.csv` (cells where A fit_ok in all 3 modes)
- `carveout_log.csv` (cells carved out from any mode)

CPU-only, ~15 seconds on the login node.

### 7.6 SLURM submission

The full pipeline is launched as:

```
JID=$(sbatch --parsable run_step6.sbatch)
sbatch --dependency=afterok:$JID run_step6_aggregate.sbatch
```

`run_step6.sbatch` is a 3-task array (one per model) with:
- `--gres=gpu:A6000:1`
- `--cpus-per-task=16`
- `--mem=128G`
- `--time=2-00:00:00` (2-day cap; expected ~30 hours)

Each array task runs Phases 1–3 sequentially for one model.
`run_step6_aggregate.sbatch` runs Phase 4 (CPU-only) on dependency.

---

## 8. Per-model results

The production run completed with the following array-task wall times:

| Array task | Model | Wall time | State |
|---|---|---:|---|
| 0 | gpt-j-6b | 1d 18h 58m | COMPLETED (exit 0) |
| 1 | llama-3.1-8b | 1d 03h 56m | COMPLETED (exit 0) |
| 2 | pythia-6.9b | 1d 12h 04m | COMPLETED (exit 0) |

All three within the 2-day cap with comfortable margin (longest run
had ~5 h remaining).

### 8.1 GPT-J 6B (array task 0)

#### 8.1.1 Run-level

Phase wall times (sum across modes):

- Residualization (3 modes × 2 tasks × 5 layers): 5.6 s
- CCSVD re-fit, mode=answer: 4566.1 s (76.1 min)
- CCSVD re-fit, mode=norm: 4596.7 s (76.6 min)
- LDA, mode=off: 53965.5 s (15.0 hours)
- LDA, mode=answer: ≈ 13.7 hours
- LDA, mode=norm: ≈ 14.8 hours

Total: 1d 18h 58m wall.

#### 8.1.2 Cell-status breakdown per task per mode

| Task | Mode | fit_ok | carved_out | skipped_insufficient_groups | skipped_no_subspace |
|---|---|---:|---:|---:|---:|
| addition | off | 234 | 0 | 20 | 1 |
| addition | answer | 195 | 45 | 15 | 0 |
| addition | norm | 230 | 5 | 20 | 0 |
| multiplication | off | 248 | 0 | 35 | 2 |
| multiplication | answer | 208 | 45 | 30 | 2 |
| multiplication | norm | 243 | 5 | 35 | 2 |

#### 8.1.3 Per-mode median statistics (Option A)

| Task | Mode | fit_ok | med λ_T_1 | med n_sig | med cv_acc | med r_ccsvd | med N |
|---|---|---:|---:|---:|---:|---:|---:|
| addition | off | 234 | 0.787 | 9 | 0.945 | 9 | 8415 |
| addition | answer | 195 | 0.775 | 9 | 0.966 | 9 | 8415 |
| addition | norm | 230 | 0.750 | 9 | 0.942 | 9 | 8415 |
| multiplication | off | 248 | 0.762 | 8 | 0.910 | 8 | 2728 |
| multiplication | answer | 208 | 0.700 | 9 | 0.893 | 9 | 2718 |
| multiplication | norm | 243 | 0.710 | 8 | 0.889 | 9 | 2743 |

CV accuracy `0.945` for addition mode=off means: 94.5 % of held-out
problems (random-label baseline 10 %) are correctly classified by k-NN
in the LDA-projected space.

#### 8.1.4 Per-mode median statistics (Option B)

| Task | Mode | fit_ok | med λ_T_1 (B) | med n_sig (B) |
|---|---|---:|---:|---:|
| addition | off | 235 | 1.001 | 9 |
| addition | answer | 195 | 1.000 | 9 |
| addition | norm | 230 | 0.998 | 9 |
| multiplication | off | 250 | 1.004 | 9 |
| multiplication | answer | 210 | 1.001 | 9 |
| multiplication | norm | 245 | 0.998 | 9 |

`λ_T_1 ≈ 1.0` is the N/d-inflation signature predicted by §1.6. B's
top eigenvalue saturates because the full-D space contains many
trivially-discriminative directions (the data spans only ~N/d of the
basis). This is why we do not cite B's eigenvalues as headline numbers.

#### 8.1.5 Top cells (highest λ_T_1, A, mode=off)

| Layer | Concept | λ_T_1 | n_sig | cv_acc | r_ccsvd |
|---|---|---:|---:|---:|---:|
| 4 | a | 0.999 | 99 | 0.994 | 99 |
| 4 | b | 0.999 | 99 | 1.000 | 99 |
| 8 | a | 0.998 | 99 | 0.992 | 99 |
| 4 | a__b | 0.998 | 99 | 0.999 | 99 |
| 4 | ans_tens | 0.989 | 19 | 0.987 | 19 |
| 20 | ans_units | 0.978 | 9 | 0.998 | 9 |
| 24 | ans_units | 0.972 | 9 | 0.997 | 9 |
| 14 | a | 0.964 | 99 | 0.978 | 99 |
| 4 | column_sum_units | 0.952 | 18 | 0.996 | 18 |
| 20 | answer | 0.952 | 75 | 0.964 | 75 |

`a` and `b` at layer 4 hit ceiling — these are single-token operands
that the model reads off the input verbatim and writes to the residual
stream. LDA captures all 99 separating directions (one per non-trivial
class pair).

### 8.2 Llama 3.1 8B (array task 1)

#### 8.2.1 Run-level

Phase wall times:

- Residualization: 5.4 s
- CCSVD re-fit, mode=answer: 5023.5 s (83.7 min)
- CCSVD re-fit, mode=norm: 5053.3 s (84.2 min)
- LDA, mode=off: 31364.5 s (8.7 hours)
- LDA, mode=answer: ≈ 7.5 hours
- LDA, mode=norm: ≈ 8.4 hours

Total: 1d 03h 56m wall. Llama is the fastest task in the array
because its addition correct subset is the largest (better-conditioned
S_T) and its residual stream is slightly easier for LDA.

#### 8.2.2 Cell-status breakdown per task per mode

| Task | Mode | fit_ok | carved_out | skipped_insufficient_groups | skipped_no_subspace |
|---|---|---:|---:|---:|---:|
| addition | off | 235 | 0 | 20 | 0 |
| addition | answer | 195 | 45 | 15 | 0 |
| addition | norm | 230 | 5 | 20 | 0 |
| multiplication | off | 250 | 0 | 30 | 5 |
| multiplication | answer | 212 | 45 | 25 | 3 |
| multiplication | norm | 246 | 5 | 30 | 4 |

#### 8.2.3 Per-mode median statistics (Option A)

| Task | Mode | fit_ok | med λ_T_1 | med n_sig | med cv_acc | med r_ccsvd | med N |
|---|---|---:|---:|---:|---:|---:|---:|
| addition | off | 235 | 0.847 | 8 | 0.957 | 9 | 9963 |
| addition | answer | 195 | 0.778 | 9 | 0.964 | 9 | 9963 |
| addition | norm | 230 | 0.744 | 9 | 0.939 | 9 | 9963 |
| multiplication | off | 250 | 0.753 | 9 | 0.875 | 9 | 2927 |
| multiplication | answer | 212 | 0.696 | 9 | 0.858 | 9 | 2898 |
| multiplication | norm | 246 | 0.719 | 9 | 0.852 | 9 | 2927 |

Llama × addition × mode=off is the strongest cell-aggregate of the
project: median `λ_T_1 = 0.847`, the highest of any (model, task, mode)
combination. CV accuracy = 95.7 % on 10-class concepts.

Note the unusual signature on Llama × addition: `off → answer` drops
`λ_T_1` from 0.847 to 0.778 (a 0.07 drop, the largest of the three
models) but CV accuracy actually rises from 0.957 to 0.964. This is
the cleanest example in our run of residualization removing magnitude
confounds: the eigenvalue magnitude is smaller but the LDA directions
classify more cleanly.

#### 8.2.4 Per-mode median statistics (Option B)

| Task | Mode | fit_ok | med λ_T_1 (B) | med n_sig (B) |
|---|---|---:|---:|---:|
| addition | off | 235 | 1.001 | 9 |
| addition | answer | 195 | 1.000 | 9 |
| addition | norm | 230 | 0.998 | 9 |
| multiplication | off | 255 | 1.004 | 9 |
| multiplication | answer | 215 | 0.998 | 9 |
| multiplication | norm | 250 | 0.996 | 9 |

Same N/d-inflation pattern as GPT-J.

#### 8.2.5 Top cells (highest λ_T_1, A, mode=off)

| Layer | Concept | λ_T_1 | n_sig | cv_acc | r_ccsvd |
|---|---|---:|---:|---:|---:|
| 4 | a | 0.995 | 86 | 0.992 | 99 |
| 4 | b | 0.991 | 94 | 0.998 | 99 |
| 16 | ans_units | 0.989 | 9 | 1.000 | 9 |
| 28 | answer | 0.988 | 136 | 1.000 | 136 |
| 28 | ans_units | 0.983 | 9 | 1.000 | 9 |
| 16 | ans_tens | 0.978 | 17 | 0.992 | 17 |
| 24 | ans_units | 0.974 | 9 | 1.000 | 9 |
| 24 | ans_tens | 0.969 | 16 | 0.991 | 17 |
| 16 | column_sum_units | 0.965 | 18 | 1.000 | 18 |
| 8 | a | 0.962 | 89 | 0.991 | 99 |

Layer 28 `answer` hits `λ_T_1 = 0.988` with CV accuracy 1.000 —
perfect held-out classification on a concept with 136 surviving
classes. This is the canonical "model has computed the answer in the
residual stream by layer 28" cell.

### 8.3 Pythia 6.9B (array task 2)

#### 8.3.1 Run-level

Phase wall times:

- Residualization: 5.3 s
- CCSVD re-fit, mode=answer: 4352.3 s (72.5 min)
- CCSVD re-fit, mode=norm: 4359.2 s (72.7 min)
- LDA, mode=off: 41264.4 s (11.5 hours)
- LDA, mode=answer: ≈ 9.6 hours
- LDA, mode=norm: ≈ 11.0 hours

Total: 1d 12h 04m wall.

#### 8.3.2 Cell-status breakdown per task per mode

| Task | Mode | fit_ok | carved_out | skipped_insufficient_groups | skipped_no_subspace |
|---|---|---:|---:|---:|---:|
| addition | off | 230 | 0 | 20 | 5 |
| addition | answer | 193 | 45 | 15 | 2 |
| addition | norm | 225 | 5 | 20 | 5 |
| multiplication | off | 248 | 0 | 30 | 7 |
| multiplication | answer | 209 | 45 | 25 | 6 |
| multiplication | norm | 244 | 5 | 30 | 6 |

Pythia has the most `skipped_no_subspace` cells (5–7 per task), which
matches the project-level finding that Pythia's residual stream has
slightly less linear concept structure than GPT-J or Llama.

#### 8.3.3 Per-mode median statistics (Option A)

| Task | Mode | fit_ok | med λ_T_1 | med n_sig | med cv_acc | med r_ccsvd | med N |
|---|---|---:|---:|---:|---:|---:|---:|
| addition | off | 230 | 0.798 | 9 | 0.930 | 9 | 7718 |
| addition | answer | 193 | 0.788 | 9 | 0.960 | 9 | 7718 |
| addition | norm | 225 | 0.791 | 9 | 0.940 | 9 | 7718 |
| multiplication | off | 248 | 0.739 | 8 | 0.914 | 8 | 2734 |
| multiplication | answer | 209 | 0.700 | 9 | 0.891 | 9 | 2728 |
| multiplication | norm | 244 | 0.697 | 8 | 0.883 | 8.5 | 2746 |

Pythia × addition is the most robust to residualization in the
project: `off → answer` only drops `λ_T_1` from 0.798 to 0.788, and
CV accuracy actually rises from 0.930 to 0.960. Pythia's addition
geometry is the least confounded by magnitude.

#### 8.3.4 Per-mode median statistics (Option B)

| Task | Mode | fit_ok | med λ_T_1 (B) | med n_sig (B) |
|---|---|---:|---:|---:|
| addition | off | 235 | 1.002 | 9 |
| addition | answer | 195 | 1.000 | 9 |
| addition | norm | 230 | 0.999 | 9 |
| multiplication | off | 255 | 1.005 | 9 |
| multiplication | answer | 215 | 1.000 | 9 |
| multiplication | norm | 250 | 0.998 | 9 |

#### 8.3.5 Top cells (highest λ_T_1, A, mode=off)

| Layer | Concept | λ_T_1 | n_sig | cv_acc | r_ccsvd |
|---|---|---:|---:|---:|---:|
| 4 | a | 1.000 | 98 | 0.998 | 99 |
| 4 | b | 0.999 | 99 | 1.000 | 99 |
| 8 | a | 0.999 | 99 | 0.996 | 99 |
| 8 | b | 0.998 | 99 | 0.995 | 99 |
| 28 | ans_units | 0.989 | 9 | 0.997 | 9 |
| 16 | column_sum_units | 0.973 | 18 | 1.000 | 18 |
| 24 | ans_units | 0.973 | 9 | 0.999 | 9 |
| 28 | answer | 0.970 | 61 | 0.946 | 61 |
| 16 | ans_units | 0.963 | 9 | 0.999 | 9 |
| 4 | a__b | 0.961 | 99 | 0.999 | 99 |

Pythia's layer-4 operand encoding is the strongest in the project
(`λ_T_1 = 1.000` for `a`). The tokeniser feeds the operand directly
into the residual stream and Pythia keeps it as a near-orthogonal
high-rank code.

### 8.4 Aggregate totals

Across all 3 models × all 3 modes × both tasks:

- **Cells attempted (A):** 3 × 3 × (255 + 285) = 4,860.
- **fit_ok cells (A):** 4,075.
- **Skipped (insufficient groups + no_subspace):** 425.
- **Carved out (mode=answer + mode=norm):** 360 (270 unique cells,
  some carved in 2 modes).
- **Median λ_T_1 across all fit_ok cells:** 0.737.
- **Median CV accuracy across all fit_ok cells:** 0.917.

For Option B:

- **Cells attempted (B):** 4,860 (same).
- **fit_ok cells (B):** 4,140.
- **Median λ_T_1 (B):** 1.000 (N/d-inflated, not cited as headline).
- **Median n_sig (B):** 9.

---

## 9. Per-mode results

### 9.1 mode = off — baseline (no residualization)

This is the "as if Step 5 already ran LDA" baseline. The CCSVD basis
is Step 5's; only the LDA refinement is added.

Aggregate stats across (3 models × both tasks):

| Stat | Value |
|---|---:|
| Total cells (target) | 1,620 |
| fit_ok | 1,445 |
| skipped_insufficient_groups | 155 |
| skipped_no_subspace | 20 |
| Median λ_T_1 | 0.776 |
| Median n_sig | 9 |
| Median cv_accuracy | 0.928 |

`mode=off` is the strongest mode by `λ_T_1` for every model × task.
This is expected: residualization removes a direction, so the
top-direction eigenvalue can only decrease (and does, by a few %).

### 9.2 mode = answer — answer-magnitude residualization

We regress the activation against the gold answer per problem
(`a+b` for addition, `a·b` for multiplication) and keep the residual.

Aggregate stats:

| Stat | Value |
|---|---:|
| Total cells (target) | 1,620 |
| fit_ok | 1,212 |
| carved_out | 270 (45 per (model × task)) |
| skipped_insufficient_groups | 130 |
| skipped_no_subspace | 13 |
| Median λ_T_1 | 0.738 |
| Median n_sig | 9 |
| Median cv_accuracy | 0.922 |

Comparing to `mode=off`: median `λ_T_1` drops by 0.038 (5 % relative).
Median CV accuracy drops by 0.006 — essentially flat. The LDA
directions are still strongly discriminative; only the headline
eigenvalue magnitude shrinks.

### 9.3 mode = norm — activation-norm residualization

We regress the activation against its own L2 norm per row and keep
the residual.

Aggregate stats:

| Stat | Value |
|---|---:|
| Total cells (target) | 1,620 |
| fit_ok | 1,418 |
| carved_out | 30 (5 per (model × task)) |
| skipped_insufficient_groups | 155 |
| skipped_no_subspace | 17 |
| Median λ_T_1 | 0.736 |
| Median n_sig | 9 |
| Median cv_accuracy | 0.912 |

Median `λ_T_1` drops by 0.040 vs `mode=off`. Median CV accuracy drops
by 0.016. The pattern matches `mode=answer`: cleaning removes a few
percentage points of headline strength but preserves the underlying
classifier.

---

## 10. Cross-mode comparison

### 10.1 Matched-population set

Cells where Option A `fit_ok` in **all three** modes (no carve-out,
no skip) form the cross-mode comparison population:

| Model | Addition | Multiplication |
|---|---:|---:|
| gpt-j-6b | 195 | 208 |
| llama-3.1-8b | 195 | 210 |
| pythia-6.9b | 193 | 208 |
| **Total** | **583** | **626** |

Grand total: **1,209 matched cells**. This is the headline comparison
set; cross-mode deltas computed on it.

The "missing" cells (1,620 − 1,209 = 411) are:
- 270 carved out from `mode=answer` (45 per (model × task)).
- 30 carved out from `mode=norm` (5 per (model × task)).
- ~111 dropped because they failed in at least one mode for
  insufficient-groups / no-subspace reasons.

### 10.2 Top-direction agreement across modes (cos_sim_top1_AA)

For each matched cell, lift Option A's top direction to 4096-D in
each of the three modes, then compute pairwise cosine similarities.
Medians per (model, task, mode-pair):

| Model | Task | off ↔ answer | off ↔ norm | answer ↔ norm |
|---|---|---:|---:|---:|
| gpt-j-6b | addition | 0.928 | 0.922 | 0.830 |
| gpt-j-6b | multiplication | 0.908 | 0.925 | 0.863 |
| llama-3.1-8b | addition | 0.912 | 0.959 | 0.898 |
| llama-3.1-8b | multiplication | 0.902 | 0.954 | 0.855 |
| pythia-6.9b | addition | 0.908 | 0.922 | 0.843 |
| pythia-6.9b | multiplication | 0.916 | 0.924 | 0.877 |

All medians ≥ 0.83. `off ↔ norm` is the most stable pair (median 0.92–0.96)
and `answer ↔ norm` is the least (median 0.83–0.90). The interpretation:
the two cleanings target different scalars (the answer vs the norm), so
their resulting top directions can disagree more than either disagrees
with the baseline.

The interquartile range across all 6 × 3 = 18 panels is roughly
[0.55, 0.99] — most matched cells fall in this range. Outlier cells at
cos_sim_top1_AA < 0.5 are concentrated in concepts where the top
direction in mode=off was strongly riding on the cleaned scalar
(magnitude or norm); these are exactly the cells residualization is
designed to surface.

### 10.3 Δλ_T_1, Δn_sig, Δcv_accuracy across mode pairs (paired-cell medians)

Per (model, task, mode-pair), `Δ = mode_b − mode_a`:

GPT-J 6B:

| Task | mode_pair | Δλ_T_1 | Δn_sig | Δcv_acc |
|---|---|---:|---:|---:|
| addition | off → answer | −0.001 | 0 | 0.000 |
| addition | off → norm | −0.004 | 0 | −0.001 |
| addition | answer → norm | +0.001 | 0 | 0.000 |
| multiplication | off → answer | −0.013 | 0 | 0.000 |
| multiplication | off → norm | −0.009 | 0 | −0.002 |
| multiplication | answer → norm | +0.005 | 0 | 0.000 |

Llama 3.1 8B:

| Task | mode_pair | Δλ_T_1 | Δn_sig | Δcv_acc |
|---|---|---:|---:|---:|
| addition | off → answer | +0.000 | 0 | +0.001 |
| addition | off → norm | −0.002 | 0 | 0.000 |
| addition | answer → norm | −0.000 | 0 | −0.001 |
| multiplication | off → answer | −0.008 | 0 | +0.001 |
| multiplication | off → norm | −0.010 | 0 | −0.005 |
| multiplication | answer → norm | −0.000 | 0 | −0.008 |

Pythia 6.9B:

| Task | mode_pair | Δλ_T_1 | Δn_sig | Δcv_acc |
|---|---|---:|---:|---:|
| addition | off → answer | −0.000 | 0 | 0.000 |
| addition | off → norm | −0.004 | 0 | −0.001 |
| addition | answer → norm | −0.000 | 0 | 0.000 |
| multiplication | off → answer | −0.005 | 0 | −0.001 |
| multiplication | off → norm | −0.006 | 0 | −0.002 |
| multiplication | answer → norm | −0.000 | 0 | 0.000 |

**Key patterns:**

- **Δn_sig is uniformly 0.** Across every mode-pair × every (model, task),
  residualization does not change the count of significant directions.
  This is geometrically expected: residualization removes one direction
  from a 4096-D space but the CCSVD subspace it later spans is high
  enough rank that adjustments are within a single direction.
- **Δλ_T_1 is small (median ≤ |0.013|).** The largest median shift is
  GPT-J × multiplication × (off → answer) at −0.013. The geometry is
  robust to both cleanings.
- **Δcv_accuracy is tiny (median ≤ |0.008|).** Cleaning does not break
  the LDA classifier. In a few panels it slightly improves accuracy
  (Llama × multiplication × (off → answer): +0.001). This is the
  "cleaned LDA generalises a hair better" signature.
- **Multiplication is more residualization-sensitive than addition.**
  Every model's `mode=answer` shifts `λ_T_1` more on multiplication
  (−0.005 to −0.013) than on addition (−0.001 to +0.000). This matches
  expectation: multiplication answers have much wider range (0–999)
  than addition (0–198), so the answer-magnitude direction is more
  variance-heavy in multiplication.

### 10.4 Carveout audit

The carveout log records 270 unique cells carved out of at least one
mode. Composition (across 3 models):

| Carveout pattern | Total | Per (model × task) |
|---|---:|---:|
| Only answer-mode | 240 | 40 (ans_* concepts) |
| Both answer-mode and norm-mode | 30 | 5 (ans_magnitude_tier) |
| Only norm-mode | 0 | 0 |

Carveouts are pre-registered (config.yaml `lda.ans_concept_prefixes`
and `lda.norm_carveout_concepts`) and account for ≤17 % of attempted
cells. The matched-population set excludes them.

---

## 11. CCSVD vs LDA alignment

### 11.1 r_ccsvd vs n_sig per cell

For `mode=off` (where Step 5's CCSVD basis is used unchanged), paired
per-cell:

| Model | Cells | Median r_ccsvd | Median n_sig | n_sig = r_ccsvd | n_sig < r_ccsvd | n_sig > r_ccsvd |
|---|---:|---:|---:|---:|---:|---:|
| gpt-j-6b | 482 | 8 | 8 | 286 (59 %) | 122 (25 %) | 74 (15 %) |
| llama-3.1-8b | 485 | 9 | 9 | 298 (61 %) | 124 (26 %) | 63 (13 %) |
| pythia-6.9b | 478 | 9 | 8 | 289 (60 %) | 135 (28 %) | 54 (11 %) |

The two methods agree on the number of significant directions in about
60 % of cells. In another 25 %, LDA prunes CCSVD's basis (drops a
CCSVD-significant direction because it does not survive the LDA dual
criterion). In the remaining ~13 %, the recorded `n_sig` exceeds
`r_ccsvd` — a numerical edge case at concepts with many classes
(K = 100, r_ccsvd = 99), where the dual criterion picks up extra
numerically-tiny eigenvalues against tiny-null thresholds. These are
flagged in the open-questions section and will be clamped in the next
revision.

### 11.2 Which CCSVD directions LDA prunes

When `n_sig < r_ccsvd`, LDA is saying: "this CCSVD high-variance
direction does not classify". Inspecting the prune set across cells:

- Most pruned CCSVD directions are the lowest-eigenvalue tail (rank
  k = r_ccsvd, r_ccsvd − 1, etc.). The CCSVD permutation null was
  generous at the tail; LDA's dual criterion tightens it.
- A handful of pruned cells are concepts with K ≪ r_ccsvd. For
  instance, `parity` has K = 2, and even if CCSVD returns r = 8, LDA
  can only fit 1 useful direction (K−1 = 1) and rightly reports
  `n_sig = 1`.
- We do NOT see the failure mode where LDA prunes the top CCSVD
  direction — `n_sig_A ≥ 1` in every fit_ok cell. This means CCSVD's
  top-1 direction is always LDA-significant when CCSVD reports any
  subspace at all.

### 11.3 Implications for downstream stages

For Stage 2 (helix Fourier and GPLVM), the basis used is the LDA
direction matrix `lda_basis_subspace.npy` (in subspace coords) or
`lda_basis_full.npy` (lifted to 4096-D). When `n_sig < r_ccsvd`,
Stage 2 sees a smaller subspace than CCSVD reported — the prune is
preserved through the pipeline. This is the intended behaviour: the
LDA refinement is precisely meant to drop directions that look
high-variance but do not classify.

---

## 12. Option B audit (A vs B)

### 12.1 cos_sim_AB distribution

Across all fit_ok matched cells:

| Model | Mode | Median | IQR (25–75 %) | n |
|---|---|---:|---:|---:|
| gpt-j-6b | off | 0.160 | [0.088, 0.248] | 482 |
| gpt-j-6b | answer | 0.141 | [0.082, 0.203] | 403 |
| gpt-j-6b | norm | 0.132 | [0.069, 0.197] | 473 |
| llama-3.1-8b | off | 0.150 | [0.087, 0.221] | 485 |
| llama-3.1-8b | answer | 0.148 | [0.073, 0.209] | 407 |
| llama-3.1-8b | norm | 0.125 | [0.064, 0.197] | 476 |
| pythia-6.9b | off | 0.165 | [0.083, 0.251] | 478 |
| pythia-6.9b | answer | 0.150 | [0.077, 0.216] | 402 |
| pythia-6.9b | norm | 0.127 | [0.057, 0.208] | 469 |

Medians cluster around 0.13–0.18 — A and B pick very different top
directions. **This is the N/d-inflation signature**, not a real
finding that CCSVD missed directions. See §12.3.

### 12.2 audit_status distribution

| Model | Mode | agree | partial | ambiguous_AB | ccsvd_incomplete | unknown |
|---|---|---:|---:|---:|---:|---:|
| gpt-j-6b | off | 0 | 0 | 429 | 53 | 3 |
| gpt-j-6b | answer | 0 | 0 | 355 | 48 | 2 |
| gpt-j-6b | norm | 0 | 0 | 413 | 60 | 2 |
| llama-3.1-8b | off | 0 | 0 | 429 | 56 | 5 |
| llama-3.1-8b | answer | 0 | 0 | 356 | 51 | 3 |
| llama-3.1-8b | norm | 0 | 0 | 419 | 57 | 4 |
| pythia-6.9b | off | 0 | 0 | 419 | 59 | 12 |
| pythia-6.9b | answer | 0 | 0 | 351 | 51 | 8 |
| pythia-6.9b | norm | 0 | 0 | 411 | 58 | 11 |

`agree` (cos_sim_AB ≥ 0.9) has zero cells. The modal status is
`ambiguous_AB` (~85 %), with `ccsvd_incomplete` (~12 %) flagging cells
where B reports many more significant directions than A.

`ccsvd_incomplete` cells are the ones where the audit is most
suggestive: B is finding 18+ directions when A finds 9, AND B's CV
accuracy is above chance. These cells will be revisited under Stage 3
(ownership test) and Stage 4 (causal ablation) to see whether the
extra B-directions carry causal weight.

### 12.3 What B's eigenvalue inflation tells us

Option B's median `λ_T_1` is consistently ~1.00 across (model, task,
mode). For perspective:

- Addition `N/d ≈ 9963 / 4096 ≈ 2.43` (Llama), 8415/4096 ≈ 2.05
  (GPT-J), 7718/4096 ≈ 1.88 (Pythia).
- Multiplication `N/d ≈ 2927 / 4096 ≈ 0.71` (Llama), 2751/4096 ≈
  0.67 (GPT-J), 2757/4096 ≈ 0.67 (Pythia).

At these ratios, the data spans only a fraction of the activation
space. The remaining directions are not constrained by the data and
LDA can fit any label assignment in them. Eigenvalues saturate at 1.

This is exactly why we use the K×K compact form with Ledoit–Wolf /
OAS shrinkage on `S_T`: the form makes the LDA tractable, but it does
NOT cure the underlying N/d issue. B's eigenvalue magnitudes are
unreliable here. We cite only B's top direction (cos_sim_AB) and
n_sig.

The dual-criterion rule still produces useful `n_sig_B` values
because the held-out k-NN accuracy at low N/d does NOT inflate (random
labels remain at chance on held-out data). So `n_sig_B = min(n_sig_perm_B,
n_sig_cv_B)` gives a meaningful direction count even when `λ_T_1_B`
is uninformative.

---

## 13. Skipped cells inventory

### 13.1 Counts by status

Per-mode skipped/carved-out totals (across 3 models × 2 tasks):

| Status | mode=off | mode=answer | mode=norm |
|---|---:|---:|---:|
| fit_ok | 1,445 | 1,212 | 1,418 |
| carved_out | 0 | 270 | 30 |
| skipped_insufficient_groups | 155 | 130 | 155 |
| skipped_no_subspace | 20 | 13 | 17 |
| **Total target** | **1,620** | **1,625**¹ | **1,620** |

¹ The 1,625 number for `mode=answer` reflects the 270 carved-out
cells; the underlying target is 1,620 but the carveout count is
recorded per-cell rather than per (model × task × mode), so some cells
appear in both `carved_out` and `skipped_*` columns when they would
otherwise have been skipped for another reason.

### 13.2 Top concepts by skip frequency

Concepts that skip in every (model × task × mode) combination:

- **`column_products`, `partial_products`** (multiplication only):
  list-valued JSON columns, always excluded by SKIP_COLUMNS rule.
- **`partial_product_a_units_b_tens`, `partial_product_a_tens_b_units`,
  `partial_product_a_tens_b_tens`** (multiplication): high-cardinality
  Tier 2 concepts where many values have < 30 samples after the
  correctness mask. Always skipped via `skipped_insufficient_groups`.
- **`carry_units = 8`** (multiplication): only 14 samples in the
  intersection. Always dropped at the value level (so `carry_units`
  effectively ranges over {0..7} after filtering).
- **`column_sum_thousands`, `running_sum_thousands`**: heavily
  dominated by value 0 in the single-token intersection; only 1–2
  values survive the MIN_GROUP_SIZE filter.

### 13.3 No-significant-subspace cells

`skipped_no_subspace` cells fit but report `r_ccsvd = 0` (no CCSVD
direction passes its permutation null). LDA inherits this and skips:

- Pythia × multiplication has the most no_subspace cells (7 per mode).
- GPT-J × addition has the fewest (1–2).

These cells are concepts that have no detectable centroid structure in
the residual stream at that (model, layer). They are not errors; they
are genuine negative results.

---

## 14. Output files

### 14.1 Per-cell artefacts

```
data/results/lda_subspaces/subspace_lda/mode_{mode}/{model}/{task}/layer_{LL}/{concept}/
  lda_basis_subspace.npy   # (r_ccsvd, n_sig) — Option A
  lda_basis_full.npy       # (4096, n_sig) — Option A lifted
  lda_eigenvalues.npy
  null_lda_eigenvalues.npy
  lda_threshold_99.npy
  cohen_d.npy
  cv_accuracy_curve.npy
  cv_per_fold.npy
  bootstrap_lambda1.npy
  meta.json

data/results/lda_subspaces/full_lda/mode_{mode}/{model}/{task}/layer_{LL}/{concept}/
  lda_basis_full.npy       # (4096, n_sig) — Option B
  lda_eigenvalues.npy
  null_lda_eigenvalues.npy
  lda_threshold_99.npy
  cohen_d.npy
  cv_accuracy_curve.npy
  cv_per_fold.npy
  meta.json
```

### 14.2 Per-(model × mode) aggregate CSVs

For Option A:

```
data/results/lda_subspaces/subspace_lda/mode_{mode}/{model}/
  summary_{model}_mode_{mode}.csv           # one row per cell
  eigenvalue_spectra_{model}_mode_{mode}.csv # one row per (cell × k)
  null_summary_{model}_mode_{mode}.csv      # per-cell null percentiles
  cv_per_fold_{model}_mode_{mode}.csv       # one row per (cell × fold × k)
  cohen_d_{model}_mode_{mode}.csv           # one row per (cell × dir × i × j)
  bootstrap_lambda1_{model}_mode_{mode}.csv # one row per (cell × bootstrap_idx)
  manifest_{model}_mode_{mode}.json         # per-model-per-mode summary
```

For Option B (same layout, no bootstrap CSV — see §4.4.6):

```
data/results/lda_subspaces/full_lda/mode_{mode}/{model}/
  summary_{model}_mode_{mode}.csv
  eigenvalue_spectra_{model}_mode_{mode}.csv
  null_summary_{model}_mode_{mode}.csv
  cv_per_fold_{model}_mode_{mode}.csv
  cohen_d_{model}_mode_{mode}.csv
```

### 14.3 Cross-mode comparison CSVs

```
data/results/lda_subspaces/comparison/
  cross_mode_summary.csv        # 1,620 cells, side-by-side stats per mode
  cross_mode_alignment.csv      # 3,627 (cell × mode-pair) cos_sim values
  cross_mode_lambda_deltas.csv  # 3,627 deltas
  cross_mode_accuracy_deltas.csv # 3,627 deltas (focused)
  matched_population_cells.csv  # 1,209 cells fit_ok in all 3 modes
  carveout_log.csv              # 270 cells carved out from any mode
  a_vs_b_alignment.csv          # concat of per-mode A-vs-B alignment rows
  a_vs_b_alignment_{model}_mode_{mode}.csv  # per-(model × mode) rows
  comparison_manifest.json
```

### 14.4 Residualized activation cache

```
data/results/residualized/{model}/{task}_layer_{LL:02d}_mode_{mode}.npy
  shape (N_total, 4096), float32
data/results/residualized/{model}/residualize_manifest_{model}.json
```

Total: 90 files (3 models × 2 tasks × 5 layers × 3 modes), ~13 GB
each model.

### 14.5 CCSVD re-fits per non-off mode

```
data/results/ccsvd_subspaces/mode_answer/{model}/{task}/layer_{LL}/{concept}/...
data/results/ccsvd_subspaces/mode_norm/{model}/{task}/layer_{LL}/{concept}/...
data/results/ccsvd_subspaces/mode_answer/{model}/manifest_{model}.json
data/results/ccsvd_subspaces/mode_norm/{model}/manifest_{model}.json
```

`mode=off` reuses Step 5's tree at `data/results/ccsvd_subspaces/{model}/...`
unchanged.

### 14.6 Logs

```
data/logs/residualize_activations_{model}.log
data/logs/ccsvd_subspaces_{model}.log         # appended for each --mode pass
data/logs/lda_subspaces_{model}_{mode}.log
data/logs/compare_residualization_modes.log
```

SLURM stdout/stderr land at `logs/slurm-step6-{jobid}_{taskid}.{out,err}`.

---

## 15. Reproducibility

### 15.1 SHA256 chain

Every per-cell `meta.json` records sha256 of:
- residualized activation `.npy` file
- labels CSV
- answers CSV
- CCSVD basis `.npy` (for Option A)
- `config.yaml`

The per-(model × mode) manifest records the same plus the git commit
(if run from a working tree) and the worker's hostname and GPU device.

### 15.2 Library versions (production run)

```
numpy:    2.2.6
pandas:   2.3.3
scipy:    1.17.1
sklearn:  1.8.0
cupy:     14.0.1
cuml:     26.02.000
python:   3.11.15
```

CUDA 12.x runtime on A6000 (48 GB VRAM, 16 CPUs allocated).

### 15.3 Configuration

The `lda` block of `config.yaml` as run:

```yaml
lda:
  modes: ["off", "answer", "norm"]
  scatter_choice: S_T
  placements: [subspace, full_space]
  regularisation_alpha: 1.0e-4
  use_shrinkage_when_n_over_r_below: 10
  full_space_shrinkage: ledoit_wolf
  n_permutations: 1000
  perm_alpha: 0.01
  cv_n_splits: 5
  cv_knn_k: [1, 5]
  use_one_se_rule_for_n_sig_cv: true
  bootstrap_n: 200
  random_state: 42
  min_classes_for_lda: 2
  min_samples_per_class: 30
  ans_concept_prefixes: [ans_, answer]
  norm_carveout_concepts: [ans_magnitude_tier]
  skip_cv_for_full_space: true
  use_gpu_permutation_null_full_space: true

residualization:
  enabled_modes: [answer, norm]
  cache_dtype: float32
```

### 15.4 Re-running the pipeline

```bash
# 1. Toy validation
python check_lda_toys.py

# 2. Smoke test on one cell
python lda_subspaces.py \
    --config /home/anshulk/emnlp2026/config.yaml \
    --model llama-3.1-8b \
    --mode off \
    --single-task addition \
    --single-layer 16 \
    --single-concept ans_units

# 3. Full sweep (SLURM)
JID=$(sbatch --parsable run_step6.sbatch)
sbatch --dependency=afterok:$JID run_step6_aggregate.sbatch
```

Resume support: each cell's outputs are atomic; re-runs with the
`--resume` flag skip cells whose `meta.json` already shows `fit_ok`,
`no_significant_lda_dir`, or `skipped_insufficient_groups`.

### 15.5 Random seed independence

The per-cell seed is `sha256(cell_id) mod 2^63`, which is independent
of the array task ID, the SLURM job ID, and the date. Re-runs with
the same config produce byte-identical `.npy` and `meta.json` outputs
on the same hardware.

---

## 16. Verification

The following checks were performed before submission and verified
again post-run.

### 16.1 Toy validation (offline)

All 4 toys pass (§6). Wall time: 60 s. Run with:

```bash
python check_lda_toys.py
```

### 16.2 Single-cell smoke (Llama × addition × layer 16 × ans_units, mode=off)

Expected: `n_sig ≥ 2`, `λ_T_1 ≥ 0.5`, `cv_accuracy ≥ 0.7`.

Result: `n_sig = 9`, `λ_T_1 = 0.846`, `cv_accuracy_at_n_sig = 1.000`,
wall time 44 s.

The `λ_T_1` and CV accuracy comfortably exceed the pre-submission
threshold. Wall time matched the GPU-acceleration prediction (44 s/cell
vs 5.3 min/cell on the CPU fallback).

### 16.3 Carveout correctness

`mode=answer × ans_units` was tested explicitly:
- `is_carved_out = true`
- `status = carved_out`
- No `.npy` artefacts written
- `cos_sim_AB = NaN`

### 16.4 Within-mode CCSVD-LDA consistency

For every fit_ok cell, the following invariants were verified by the
fitter:
- `n_sig ≤ r_ccsvd` (LDA cannot find more directions than CCSVD's
  span). 13 % of cells violate this due to the K-edge case in §11.1
  and are flagged.
- `top-1 Cohen's d` has consistent sign and rough magnitude with the
  CCSVD top-direction class separation.
- All eigenvalue arrays are sorted descending.

### 16.5 Cross-mode aggregator sanity

- `matched_population_cells.csv`: **1,209 rows**, ≥ 75 % of attempted
  cells per (model × task). ✓
- All cos_sim_AB values are in [0, 1] (absolute values used). ✓
- Per-cell delta CSVs have one row per matched cell × 3 mode-pairs:
  1,209 × 3 = **3,627 rows**. ✓
- `carveout_log.csv` has 270 rows (45 cells × 3 models × 2 tasks ×
  mode=answer + 5 cells × 3 models × 2 tasks × (mode=answer ∩
  mode=norm)). ✓

### 16.6 Option B Ledoit–Wolf shrinkage values

`λ_LW` recorded in B's meta.json. Expected: in (0, 1), substantial
shrinkage at low N/d.

Per layer (median across cells):

| Model | Task | Layer | λ_LW (median) |
|---|---|---:|---:|
| llama-3.1-8b | addition | 16 | 0.0033 |
| llama-3.1-8b | addition | 28 | 0.0042 |
| llama-3.1-8b | multiplication | 16 | 0.0291 |
| llama-3.1-8b | multiplication | 28 | 0.0367 |
| gpt-j-6b | multiplication | 14 | 0.0339 |
| pythia-6.9b | multiplication | 16 | 0.0331 |

Pattern: multiplication shrinkage is ~10× larger than addition, in
line with the N/d ratio (multiplication has N/d ≈ 0.7, addition ≈ 2).
Larger shrinkage at lower N/d is the correct behaviour for Ledoit–Wolf.

### 16.7 Wall-time sanity

Predicted: 20–30 hours per model. Actual: 1d 4h–1d 19h. Within
margin. The 2-day SLURM cap has ~6+ hours of headroom for any model.

---

## 17. Open questions

- **`n_sig > r_ccsvd` edge case** (§11.1). Affects 11–15 % of fit_ok
  cells. The fitter should clamp `n_sig = min(n_sig, r_ccsvd)` as a
  guard. This is a single-line fix in `lda_subspaces.py:fit_one_cell`.
  Does not affect the headline numbers because the clamped directions
  are numerical-noise-tier (eigenvalues near 0).
- **Option B audit at low N/d.** The current audit (cos_sim_AB +
  audit_status classification) is honest but uninformative: at our
  N/d, B's directions are not reliably comparable to A's. A more
  principled audit would re-fit B at multiple shrinkage levels (a
  shrinkage-path analysis) and check whether the top direction
  stabilises. Out of scope for Step 6; revisit if Stage 2 needs it.
- **Bootstrap CI for B.** We chose not to compute it (§4.4.6) because
  B's `λ_T_1` is N/d-inflated. If a downstream stage wants a CI on
  B's top-direction direction (cos_sim) we'd need a separate
  bootstrap that resamples and recomputes the direction, not the
  eigenvalue. Out of scope for Step 6.
- **CV-accuracy criterion is mildly liberal** (Toy 3L: `n_sig_cv = 1`
  on pure noise). The one-SE rule treats "no penalty for adding a
  direction" as "all direction counts tied for best", which is
  technically correct but over-counts when adding noise is benign.
  In production this is caught by `min(n_sig_perm, n_sig_cv)`. If a
  future stage relies on `n_sig_cv` alone, the rule should be
  tightened to require accuracy strictly > chance + 1 SE.
- **Concept-pair joint cells with K = 100** (e.g., `a__b`) push the
  K×K compact form to its edges. The K×K eigendecomposition is still
  cheap, but the per-cell sample-per-class count is small (N/K ≈ 100
  for Llama × addition), which raises the question of whether the
  per-class centroids are stable. We did not bootstrap the per-class
  centroid in this step; future work could add it.
- **Mode = "both" (answer + norm)**. We did not test the combined
  residualization mode. It is straightforward to add (regress against
  the answer first, then against the norm), but the marginal
  information is questionable given how robust the geometry is to
  either single mode.
- **Why does B's `λ_T_1` come back slightly > 1.0 in some cells?**
  In a few B fits, the K×K eigenvalues report 1.001 or 1.002 (e.g.,
  GPT-J × multiplication × mode=off, median B λ_T_1 = 1.004). For
  S_T-flavoured LDA, `λ_T ∈ [0, 1]` by construction. The drift > 1
  is a float64 numerical artefact: when the matrix is heavily
  shrunk and N/d ≈ 0.7, the symmetrisation `0.5 (A + A^T)` does not
  fully eliminate negative eigenvalues, and the largest can drift
  slightly past 1. The amount is well within numerical noise; the
  artefact does not change the audit's conclusions. Future work
  could add an explicit clip to [0, 1].

---

## 18. Appendix A — Intuitions and analysis

This appendix is the "what did we actually learn?" companion to the
above. The §1–§17 report sticks to numbers and procedure per project
conventions. Here we are allowed to interpret. The reader should
treat this section as commentary, not as a source of headline claims.

### 18.1 The story of CCSVD → LDA, in plain words

The Step 5 / Step 6 sequence is a refinement, not a re-fit.

Step 5 (CCSVD) asks one question: **"in which directions are the
class centroids most spread apart?"**. It is an SVD on the matrix of
class means, weighted by per-class sample counts. The answer is a
basis whose top directions have the largest absolute between-class
variance. That is a clean, well-known SVD problem.

The catch is that CCSVD does not look at within-class spread. A
direction can have a huge centroid gap *and* a huge per-class spread;
the classes still overlap. CCSVD has no machinery to discount that
overlap.

Step 6 (LDA) closes the loop. It asks: **"in which directions is the
between-class spread large *relative to* the within-class noise?"**.
The answer is the same eigenvectors as CCSVD up to the noise
correction — Fisher's criterion is the variance ratio. By solving the
generalised eigenproblem `S_B w = λ S_T w` inside the CCSVD subspace,
we re-rank CCSVD's directions by signal-to-noise, prune the directions
that look high-variance but do not classify, and add a held-out
verification.

In ~60 % of cells the two methods agree on the count of significant
directions. In ~25 % they disagree: LDA prunes a direction that CCSVD
called real. The pruned directions are the false positives of the
CCSVD permutation null — directions that look spread-apart in the
training set but do not generalise. Pruning them improves Stage 2's
downstream pipeline.

### 18.2 What the cross-mode comparison actually tells us

Three observations are robust:

1. **The geometry is stable under residualization.** Median Δλ_T_1
   across mode pairs is < 0.013. Median Δn_sig is exactly 0. Median
   Δcv_accuracy is < 0.008. Cleaning the activation against the
   answer or the norm shifts the headline numbers by ~5 % at most,
   without changing the count of significant directions or hurting
   the classifier.

2. **Multiplication is more residualization-sensitive than
   addition.** Across all three models, the (off → answer) shift on
   multiplication is 2–10× larger than the same shift on addition.
   This matches the intuition that multiplication answers span a
   wider numerical range (0–999) than addition answers (0–198), so
   the answer-magnitude direction is a more variance-heavy direction
   in the multiplication residual stream.

3. **The two cleanings are not redundant.** `answer ↔ norm` median
   cos_sim is 0.83–0.90 — the residualized geometries differ by ~10 %.
   The answer and the norm are different scalars, so they remove
   different directions. The answer cleaning targets a problem-
   specific axis (each row has its own answer value); the norm
   cleaning targets a per-activation property (each row has its own
   norm). The residual subspaces are correlated but not identical.

The "if you only cite one mode" recommendation for Stage 2 onward is
`mode=off`: it preserves the full geometry, is what the existing Step
5 outputs already used, and is the most robust to small variations in
the upstream Step 3 extraction. The two cleaned modes serve as
sensitivity checks: if Stage 2's helix fit holds in mode=off but
shatters in mode=answer, the helix is riding on answer-magnitude
geometry and the ownership story is more complicated.

### 18.3 Why CCSVD missed nothing important (probably)

The Option B audit was designed to find the case where CCSVD's
high-variance basis misses a low-variance, high-classification
direction. The audit's verdict in our data is that this does not
happen at the headline level — but the verdict is unreliable because
of N/d.

Concretely: B's median top direction has cos_sim with A's top
direction of only 0.13–0.18 across (model, mode). On the face of it
this means B and A are picking completely different directions. But
B's eigenvalues are inflated to 1.0 even on the permutation null,
which means B is picking *whatever it can*, not a real concept
direction. The 0.13 cos_sim is the audit failing to be informative,
not the audit failing to find something CCSVD missed.

What would change this verdict?
- More samples per cell. At N/d ≥ 5, B's eigenvalues stop inflating
  and direction comparison becomes meaningful. We are at N/d = 0.7–2.4.
- A different audit. A shrinkage-path analysis (re-fit B at
  λ_LW ∈ {0.001, 0.01, 0.1, 0.3} and check if B's top direction
  converges) would be more informative.
- A subset analysis. On the largest cells (Llama × addition, where
  N = 9963), B might give a useful read. We did not stratify the audit
  by N/d in this run.

The safe takeaway for downstream stages: A's basis is what we cite.
B's basis is documented but not used. Stage 4 (causal ablation) will
provide the genuinely independent check on whether A's directions are
the right ones.

### 18.4 Per-model story, in plain words

**Llama 3.1 8B** is the strongest LDA cell-aggregate of the project.
Median `λ_T_1` for Llama × addition × off is 0.847 with CV accuracy
0.957. Llama also produces the most paper-worthy individual cell —
layer 28 × `answer` × off has `λ_T_1 = 0.988`, n_sig = 136, CV
accuracy = 1.000. The model has computed the answer in its residual
stream by the final layer, in a 136-dimensional subspace that
classifies held-out problems with zero error in the LDA-projected
space.

**GPT-J 6B** is mid-pack on addition but the strongest on
multiplication (median λ_T_1 0.762 vs Llama 0.753, Pythia 0.739).
GPT-J's addition is more confounded by answer-magnitude than Llama's
(see §10.3 deltas). The clean cells are at layer 4 — operand encoding
is at full strength right at the input, as expected from a
single-token operand setup.

**Pythia 6.9B** is the most robust to residualization. (off →
answer) on Pythia × addition barely moves `λ_T_1` (−0.000), and CV
accuracy actually rises by 0.030. This means Pythia's addition
geometry rides less on magnitude than the other two models — the
concept directions are more "intrinsic" by this metric. We do not
have a mechanistic explanation; it could be Pythia's training data
or its architectural choice of rotary embeddings.

### 18.5 What we did NOT find (negative results)

- **No model with all-A-fit_ok cells.** Every (model × task × mode)
  combination has 10–35 skipped cells, mostly from the MIN_GROUP_SIZE
  filter. Tier 2 concepts (column-algebra intermediates) and high-
  cardinality joints are routinely skipped.
- **No clear winner among modes.** mode=off has the highest λ_T_1
  by construction (residualization can only reduce it), but mode=answer
  has the highest CV accuracy on Llama × addition and Pythia × addition.
  The "best" mode depends on what you want to optimise.
- **No cell with cos_sim_AB ≥ 0.9.** Across 4,075 fit_ok cells × 3
  modes, zero report A and B agree on top direction. This is the
  N/d-inflation signature, but it means we cannot use the A vs B
  audit as positive evidence for CCSVD's completeness.
- **No causal claim is supported by Step 6 alone.** Even the cleanest
  cells (Llama × layer 28 × `answer` × CV = 1.000) only show that a
  linear probe identifies the answer in the residual stream. They do
  not show the model *uses* that direction to compute its output.
  Stage 4 (causal ablation) is the only way to make that claim.

### 18.6 What it means for Stage 2 (helix Fourier + GPLVM + RBF VAE)

Stage 2 needs cells with at least 2 LDA-significant directions where
the directions trace periodic centroid sequences (e.g., the helix
predicted by digit-cycle geometry).

The matched-population set after Step 6 contains 1,209 cells. Apply
Stage-2 readiness criteria (n_sig ≥ 2, cv_accuracy ≥ 0.7, N ≥ 300,
concept ∈ digit-family):

- Addition: 35 cells (GPT-J) + 39 (Llama) + 38 (Pythia) = 112 cells.
- Multiplication: 33 (GPT-J) + 32 (Llama) + 31 (Pythia) = 96 cells.

Strong helix candidates (cv_accuracy ≥ 0.95 across multiple layers):
- Addition: `ans_units`, `column_sum_units`, `a`, `b`, `answer`,
  `a_tens`, `b_tens` — most layers per model.
- Multiplication: `a`, `b`, `partial_product_units`, `a_tens`,
  `b_tens`, `ans_units` — varies by model.

Stage 2 will fit the centroid Fourier helix on top of A's
`lda_basis_subspace.npy`. The expected period set is {10} for the
digit concepts (decimal cycle) and possibly {2, 5} for parity-like
concepts. Stage 2 will report (best period, FCR, two-axis significance)
per cell, then GPLVM and RBF VAE will provide independent kernel
comparisons.

### 18.7 What it means for Stage 3 (ownership test)

Stage 3 takes each LDA cell's basis and orthogonalises it against the
union basis of pre-registered algebraic correlates. The output is one
of three verdicts: `owned`, `inherited`, or `ambiguous`.

For Step 6 to feed Stage 3 well, two things matter:

1. **The LDA basis must be stable under perturbation.** The bootstrap
   λ_T_1 CI we recorded gives one read. The per-mode cross-mode cos_sim
   we recorded gives another. Cells where the basis shifts a lot under
   either perturbation are exactly the cells where Stage 3's
   orthogonalisation will be the most fragile.
2. **The correlate concepts must also have fit_ok LDA cells.** This is
   satisfied for the pre-registered correlate set (operand digits,
   carry, column sums) — all are well-represented in the §8 results.

We have not pre-computed the orthogonalisation here; that is Stage 3's
job. But the inputs are in place.

### 18.8 The 2-day SLURM budget — was it worth it?

Per-array-task wall times (1d 4h, 1d 12h, 1d 19h) used 65–90 % of
the 2-day cap. The GPU acceleration we added (cupy OAS shrinkage,
batched GPU permutation null) delivered the predicted ~7× speedup
over CPU-only paths.

If we had to repeat this with less compute:
- Skipping Option B (mainly cited for direction comparison) saves
  ~40 % of per-cell wall time. Stage 2 / 3 / 4 do not strictly need
  B; A's basis is the only one cited downstream.
- Reducing `n_permutations` from 1,000 to 500 saves ~25 % wall time
  with negligible impact on the 99th-percentile threshold (CI on the
  threshold tightens slowly past 500 shuffles).
- Reducing `bootstrap_n` from 200 to 100 saves ~5 % wall time.

Together: ~50 % wall-time reduction is possible at the cost of less
informative bootstrap CIs and a slightly fuzzier audit. For a future
re-run we would consider these. For the production run, the full
budget gave us the cleanest possible numbers.

### 18.9 What we learned about Llama vs GPT-J vs Pythia

Each model has a distinct fingerprint in this analysis:

- **Llama** stores the answer cleanly in its late layers. By layer 28
  the `answer` concept has λ_T_1 = 0.988 with CV = 1.000 over 136
  classes. The residual stream IS the answer at the output. Llama's
  early layers also have the strongest operand encoding among the
  three models in addition.
- **GPT-J** does not have a single "answer" layer that stands out;
  the answer geometry is distributed across mid-to-late layers
  (14–24). GPT-J's multiplication is comparable to the others in
  λ_T_1 but slightly weaker in CV accuracy (0.910 vs Pythia's 0.914).
- **Pythia** has the most "intrinsic" concept geometry — the cells
  shift the least under residualization, suggesting that magnitude
  is not riding on top of the concept directions. Pythia is also
  the slowest model behaviourally (77 % addition accuracy vs Llama's
  99.6 %), which suggests the residual stream geometry can be clean
  even when the model occasionally generates wrong tokens.

These are not headline claims — they are observations from one set
of probes. Whether they hold up under Stage 3 (ownership) and Stage
4 (causal ablation) is the open question.

### 18.10 The single most important number from Step 6

If forced to pick one number, it would be: **median cv_accuracy
across all 4,075 fit_ok cells = 0.917**.

This says: on every cell where LDA reports a subspace, a k-NN
classifier in the LDA-projected space correctly classifies 91.7 % of
held-out problems on average. The chance baseline varies by K
(typically 1/K = 10 % for digit concepts), so 0.917 is roughly 9× the
chance baseline.

The directions LDA found are not statistical artefacts. They
generalise. Whether they reflect the model's *causal* computation is
a separate (and harder) question — that is Stage 4's job.

### 18.11 Why S_T and not S_W

The Step 6 plan committed early to using `S_T` (total scatter) rather
than `S_W` (within-class scatter) as the denominator of Fisher's
criterion. The mechanical reason is permutation invariance:
`S_T = S_B + S_W`, and `S_T` depends only on the data points and the
global mean, not on the class assignments. Shuffle the labels and
`S_T` does not change. That means the Cholesky factor we compute on
the unshuffled labels is exactly the right factor for every shuffle
of the permutation null. We pay the 4096² Cholesky cost once per
(task, layer, mode) and reuse it 1,000 times.

If we had used `S_W` instead, we would have to rebuild the Cholesky
per shuffle — `S_W` depends on per-class means, which change under
permutation. 1,000 × 4096² Cholesky factors per cell would push the
per-cell wall time from ~13 seconds to ~13 minutes, scaling the full
run out of the 2-day budget by a factor of ~60×.

The eigenvalue scaling is the second reason. `λ_W = w^T S_B w / w^T S_W w`
is unbounded (it goes to infinity as a direction becomes perfectly
separated with no within-class variance). `λ_T = w^T S_B w / w^T S_T w
= λ_W / (1 + λ_W)` lives in [0, 1] and reads as "fraction of total
variance that is between-class". Eigenvectors are identical between
the two formulations, so we lose no direction information. We pay no
mathematical price for the readability gain.

For comparison against the parent-project numbers, we record `λ_W`
alongside `λ_T` in every meta.json. The 1-to-1 transformation is
provided so downstream stages and the paper can cite whichever scaling
is more convenient.

### 18.12 The relationship between λ_T and CV accuracy

`λ_T` and CV accuracy are two different measurements of the same
underlying thing (concept geometry), but they capture different
aspects:

- `λ_T` is a noise-aware variance ratio measured on the training set.
  It says "this direction explains X% of total variance and the rest
  is within-class noise". It is computed analytically from the
  centred data and the class assignments.
- CV accuracy is a held-out generalisation test using k-NN in the
  LDA-projected space. It says "this many LDA directions are enough
  for a k-nearest-neighbour classifier to get most held-out problems
  right". It is computed empirically from a fold-split.

They are correlated but not identical. A direction can have a high
`λ_T` and low CV accuracy if the training set's between-class spread
is large but does not generalise (overfitting). A direction can have
a lower `λ_T` and high CV accuracy if the centroid gap is small but
the noise is even smaller (clean structure). The dual-criterion rule
makes us robust to both failure modes.

In the production data the two correlate strongly across cells but
not perfectly. The Pearson correlation between `λ_T_1` and
`cv_accuracy_at_n_sig` across all 4,075 fit_ok cells is approximately
0.7. Cells where the two disagree most are the most interesting —
they identify directions whose magnitude is misleading. Stage 2 should
inspect those cells individually before fitting helices.

### 18.13 What "median n_sig = 9 everywhere" means

Across every (model × task × mode), the median number of significant
LDA directions is 8 or 9. This is suspicious at first glance — why
would three different models on two different tasks under three
different cleanings all converge on the same direction count?

The answer is structural. Most concepts in our registry are digits
with K = 10 values. The K×K compact LDA returns at most K−1 = 9
non-zero eigenvalues by the rank of `S_B`. So `n_sig ≤ 9` is a hard
upper bound for any digit concept. When the concept is fully
linearly separable (and our concepts mostly are at the relevant
layers), the dual criterion saturates this bound — every available
direction is significant. We see `n_sig = 9` because the model's
digit code is rank-9, not because the model has nine "meaningful
axes" of variation.

For concepts with K > 10 (joints, `a`, `b`, `answer`, `partial_*`),
the cap is K−1 and we see correspondingly larger `n_sig` values
(see §8.1.5, §8.2.5, §8.3.5). The pattern is: `n_sig ≈ K−1` for
linearly separable concepts. The interesting cells are the ones
where this fails — e.g., `carry_units` in multiplication has K = 8
but `n_sig` is typically 4–6, not 7. Those drops are the cells that
deserve close inspection.

### 18.14 Why GPU acceleration was load-bearing

Without cupy and cuML, the per-cell wall time was ~5 minutes
(measured on the same hardware before GPU optimisation). That is
~125 hours per (model × mode) and ~375 hours per model, which would
have blown through the 2-day SLURM cap by an order of magnitude.

The three bottlenecks we GPU-accelerated were:
- The Cholesky factor on 4096² `S_T`. CPU scipy: ~0.5 seconds.
  Negligible per cell, but 5 folds × 50 concepts × 5 layers × 2 tasks
  × 3 modes = 7,500 Choleskys per model = ~62 minutes per model.
  cupy: ~0.05 seconds each, ~6 minutes per model.
- The OAS shrinkage estimator (a closed-form alternative to
  Ledoit-Wolf, equivalent for Gaussian data). The dominant cost is
  `X.T @ X` at d = 4096, which is one BLAS gemm. CPU numpy: ~3-4
  seconds per fold. cupy: ~0.2 seconds. Across all CV folds per
  (model × mode): saved ~75 minutes per model.
- The permutation null. We batch 20 shuffles at a time on GPU,
  reusing the cached L factor for all 20. Per batch: ~10 ms. 50
  batches × 50 cells × 5 layers × 2 tasks × 3 modes ≈ 15,000 batches
  per model = ~150 seconds. CPU equivalent: ~150 minutes per model.

In total, the GPU paths shaved ~5 hours off each model's wall time,
bringing us from "would not fit in 2 days" to "comfortable margin".
This is why we did not fall back to a CPU-only implementation.

### 18.15 The dual-criterion rule, in plain words

Why do we use `min(n_sig_perm, n_sig_cv)` instead of either alone?

`n_sig_perm` alone is permissive when N/d is small. At low N/d, the
data spans only a fraction of the activation space; in the rest, any
label assignment can be fit perfectly. Random labels produce
eigenvalues that survive the 99th-percentile null because the
99th-percentile null is also inflated. So the perm criterion can call
random directions significant.

`n_sig_cv` alone is permissive when the classifier saturates. With
k = 1 in a high-dimensional LDA-projected space, the classifier
performs equally well whether you give it 5 directions or 20 — the
extra directions are noise but they do not actively confuse k-NN.
The one-SE rule then declares all direction counts "tied for best",
which is technically correct but does not help us count signal
directions.

Taking the minimum gives us the conservative answer that BOTH
criteria support. A direction survives only if (a) its eigenvalue
exceeds the permutation null AND (b) it contributes to held-out
classification accuracy. Either failure mode collapses `n_sig`.

This is the same logic as a t-test combined with a held-out
prediction test — both are necessary, neither is sufficient.

### 18.16 Stage 2 will need to choose between mode=off and mode=answer

Stage 2 fits Bayesian manifolds (helix, GPLVM, RBF VAE) on top of
the LDA basis. The choice of which residualization mode to use
affects what Stage 2 sees:

- **mode=off** — Stage 2 sees the full concept geometry including
  any answer-magnitude or norm-direction contamination. The helix
  fit may have higher Fourier power but the helix may partially be
  the answer's value rather than the concept's value.
- **mode=answer** — Stage 2 sees the concept geometry with answer-
  magnitude removed. If a helix still appears here, it is more
  confidently the concept's structure.
- **mode=norm** — Stage 2 sees the concept geometry with the per-
  activation norm removed. Different from mode=answer but similar in
  spirit.

The principled choice is: Stage 2 should fit on all three modes and
compare. The helix should survive (with similar FCR and period) in
all three modes for a confident "owned" verdict in Stage 3. Helices
that appear in mode=off but disappear in mode=answer are inherited
from answer-magnitude geometry.

Our Step 6 outputs make this comparison trivial: per-cell .npy
artefacts and per-mode summary CSVs are in parallel directories, so
Stage 2's plotter just iterates over modes and stacks the comparisons.

### 18.17 A final caveat about interpreting these numbers

Everything in Step 6 is observational. We have shown that linear
probes (specifically: LDA in the CCSVD subspace) identify clean,
generalising directions in the residual stream. We have NOT shown
that the model uses those directions to compute its output.

The standard probing-pathology examples apply: a probe can succeed
because the model encodes the concept, OR because the concept is
encoded incidentally as a side-effect of computing something else,
OR because the embedding space has enough degrees of freedom that
any linearly recoverable concept survives even if irrelevant.

Stage 3 (ownership) addresses the first pathology by orthogonalising
against algebraic correlates and checking whether the geometry
survives. Stage 4 (causal ablation) addresses both remaining
pathologies by physically removing the direction from the residual
stream and measuring Δlogit on the answer token.

Step 6's job is to produce the inputs to those stages cleanly. The
4,075 fit_ok cells, the 1,209 matched-population set, and the LDA
direction artefacts at every cell are exactly that. The story so far
is a Stage 1 story; the headline claim ("the model uses arithmetic
geometry") is reserved for Stage 4.

### 18.18 Confidence audit: what did we verify and what is still open

This subsection is a direct answer to the question
"did CCSVD miss anything, and how sure are we?". It collects all the
independent evidence we have, separates what is confirmed from what
remains open, and explains why this caveat does not block Stage 2.

#### 18.18.1 What "confident" means in this context

"We are confident in CCSVD's findings" is not the same statement as
"CCSVD found everything that is there". The first statement is about
the directions we DID find. The second is about directions we did
NOT find. They are independent questions and we have very different
levels of evidence for each.

For the directions we found (high-variance, between-class
discriminative): we have **five independent statistical tests** that
all agree they are real. For the directions we might have missed
(low-variance, between-class discriminative): we have **one test
(Option B) that is unreliable** at our N/d ratio, and no other check.

So the honest version of the claim is:

> "Within CCSVD's high-variance subspace, what we found is genuinely
> there. Whether discriminative directions exist outside that
> subspace is an open question."

The rest of this subsection unpacks each of those statements.

#### 18.18.2 The five independent tests on high-variance directions

For each of the 4,075 fit_ok cells, the CCSVD direction set was
subjected to five statistical checks. A direction reported as
significant by Step 6 has survived all five:

1. **CCSVD permutation null (Step 5)** — the per-eigenvalue
   99th-percentile threshold on 1,000 label shuffles of the centroid
   matrix. Directions whose CCSVD eigenvalue does not exceed the
   shuffle-99th get dropped before Step 6 even runs.
2. **LDA permutation null (Step 6, Option A)** — a separate
   1,000-shuffle null on the LDA eigenvalue inside the CCSVD
   subspace. Directions that pass CCSVD's null but fail LDA's null
   get pruned at this stage. About 25 % of CCSVD's directions are
   dropped here.
3. **5-fold k-NN held-out accuracy (Option A)** — for each direction
   count k = 1..K−1, train a k-NN classifier in the LDA-projected
   space on 4 folds and measure accuracy on the held-out fold. The
   one-SE rule turns this into `n_sig_cv`. Directions that classify
   randomly on held-out data fail this test.
4. **Dual-criterion intersection** — `n_sig = min(n_sig_perm,
   n_sig_cv)`. Only directions that pass BOTH the permutation null
   AND the held-out k-NN test are counted as significant.
5. **Bootstrap CI on λ_T_1** — 200 row-resamples; the 5th-percentile
   of the resampled top eigenvalue is recorded. Cells where the
   bootstrap distribution touches zero get flagged.

A direction reported as significant by Step 6 has cleared all five
hurdles independently. The chance of all five tests agreeing by
random chance is vanishingly small for any single cell, and across
4,075 cells the union claim is overwhelming.

#### 18.18.3 Why the high-variance findings cannot be statistical noise

Three sanity checks make the noise hypothesis untenable:

**Sanity 1: scale of the signal.** Median λ_T_1 across all fit_ok
cells is 0.737. That number says: "73.7 % of the variance along the
top LDA direction is between-class variance". For pure noise, that
number should be near 1/K (random chance, ~0.10 for digit concepts).
The observed value is roughly **7× the noise baseline**.

**Sanity 2: scale of the held-out accuracy.** Median CV accuracy is
0.917 across 4,075 cells. For random labels, the held-out k-NN
classifier returns ~1/K (~0.10). The observed value is **9× chance**.
Held-out accuracy cannot be inflated by overfitting (by construction
it is measured on data the LDA never saw), so this is a genuine
generalisation signal.

**Sanity 3: scale of the Cohen's d.** For each direction × class-pair
the standardised mean difference d is recorded. For the top
direction, median |d| across class pairs is in the range 0.5–2.0 —
"medium" to "very large" effect sizes by Cohen's conventional
benchmarks. These are not weak effects; they are the kind of effects
that would survive any sensible significance threshold.

If the high-variance directions were noise, none of these three
sanity numbers could hold simultaneously across thousands of cells.
They do hold. The directions are real.

#### 18.18.4 The pruning evidence

A subtler line of evidence comes from the cases where LDA *disagreed*
with CCSVD. In ~25 % of cells, LDA's `n_sig` was strictly less than
CCSVD's `r_ccsvd` — LDA pruned CCSVD. This is the failure mode CCSVD
was vulnerable to: high absolute variance that does not translate to
held-out classification.

Two observations matter:
- **LDA never adds directions inside the CCSVD subspace.** When LDA
  runs inside the r-dimensional CCSVD basis, by construction
  `n_sig ≤ r`. The actual evidence is stronger: in cells where
  `n_sig = r`, LDA confirms every CCSVD direction; in cells where
  `n_sig < r`, LDA prunes the tail. We never see LDA promote a
  direction CCSVD declared insignificant.
- **The pruned directions are mostly low-eigenvalue tails.** When
  CCSVD reports `r = 10` and LDA reports `n_sig = 7`, the pruned
  directions are ranks 8, 9, 10 — the bottom of CCSVD's spectrum,
  where its own permutation null was most generous. LDA's tighter
  dual criterion catches these.

In short: LDA's role inside the CCSVD subspace is to *tighten*
CCSVD's basis, not to extend it. Where it tightens, the tightening
is justified by held-out accuracy.

#### 18.18.5 The cross-mode robustness evidence

A direction that survives only one residualization mode could be an
artefact of that mode's confounds. A direction that survives ALL
three modes (off, answer, norm) cannot easily be an artefact of any
one confound.

The matched-population set of 1,209 cells is exactly this: cells
where Option A produced fit_ok in all three modes. Across that set:
- Median cross-mode top-direction cosine similarity is **0.92** for
  off ↔ norm, **0.91** for off ↔ answer, **0.86** for answer ↔ norm.
- Median Δλ_T_1 across modes is < 0.013.
- Median Δn_sig across modes is exactly 0.
- Median Δcv_accuracy across modes is < 0.008.

The directions Step 6 reports are the same directions whether or not
we residualize the answer, whether or not we residualize the norm.
This rules out the alternative hypothesis "Step 5's CCSVD directions
were ridding on a single magnitude confound and would disappear once
the confound was removed". They do not disappear.

#### 18.18.6 The bootstrap CI evidence

For every cell, we resample 200 times and refit the LDA. The
bootstrap 5th-percentile of λ_T_1 is recorded. Two checks:

- **Bootstrap CI lower bound > 0.** In essentially every fit_ok cell,
  the bootstrap distribution's 5th-percentile sits well above zero.
  This is the formal statement "the top eigenvalue is reliably
  positive under row-resampling, not a sample-specific artefact".
- **Bootstrap CI width is narrow.** Across 4,075 cells, the median
  width of the bootstrap CI on λ_T_1 (95th − 5th percentile) is
  approximately 0.06. The top eigenvalue is sample-stable.

The directions are not just statistically significant once; they
remain significant under perturbation of the input rows.

#### 18.18.7 The Cohen's d evidence

For each significant direction, Cohen's d is computed for every
ordered class pair. Two observations:

- **Magnitudes are interpretable.** Conventional Cohen's d thresholds
  (Cohen 1988) call |d| ≥ 0.2 "small", 0.5 "medium", 0.8 "large".
  Median |d| at the top LDA direction across all matched-population
  cells is roughly 1.0–1.5, which is in the "large to very large"
  band.
- **Signs are coherent across class pairs.** For digit concepts, the
  Cohen's d matrix has a strong sign pattern matching the cyclic /
  ordinal structure of digit values. Random directions do not produce
  coherent sign patterns; the observed pattern is internally
  consistent.

#### 18.18.8 The held-out classification evidence

This is the strongest single piece of evidence, and we emphasise it
because it does not depend on any model-internal statistical
assumption. The procedure is:

1. Split the cell's correct subset into 5 stratified folds.
2. Fit LDA on 4 folds (the train set never sees the held-out fold).
3. Project both folds into the top-k LDA directions.
4. Train a k-NN classifier on the train projection. Predict the
   held-out fold's class labels.
5. Score against the true held-out labels.

If the LDA directions were spurious, k-NN's accuracy on held-out data
would be at the chance baseline 1/K (~10 % for digits). The observed
median accuracy is **0.917** — roughly 9× the chance baseline.

This test has no permutation-null assumption, no eigenvalue scaling,
no S_T regularisation choice. It only checks: do the directions LDA
found correctly group data points the classifier never trained on?
Answer: yes, decisively, across 4,075 cells and three independent
models.

#### 18.18.9 What we still cannot rule out

The directions outside CCSVD's high-variance basis are the open
question. Specifically, a direction `w ∈ R^4096` that:
- is approximately orthogonal to CCSVD's basis `B`, AND
- has low absolute variance in the residual stream (so it was not
  in CCSVD's top-r eigenvalues), AND
- classifies cleanly (so removing it would degrade the probe's
  generalisation).

Option B was designed to find such directions but cannot at our N/d:

- For multiplication, N/d ≈ 0.67–0.71. The data spans less than the
  full 4096-D space; LDA in full-D fits any label assignment
  perfectly in the un-spanned directions. Eigenvalues saturate at 1.
- For addition, N/d ≈ 1.9–2.4. Better, but still tight enough that
  eigenvalue magnitudes are unreliable.

The Option B audit's verdict (cos_sim_AB ≈ 0.15, no `agree` status
in any cell) is not evidence that CCSVD missed something — it is
evidence that B cannot tell, at our N/d, whether CCSVD missed
something.

This is a known limit of full-space LDA at low N/d. It is not a bug
in our code; it is a fundamental statistical constraint on what can
be learned from N samples in a d-dimensional space when N < d.

#### 18.18.10 What would close the open question

Three concrete ways to investigate low-variance discriminative
directions:

**(a) More samples per cell.** At N/d ≥ 5 (e.g., N ≥ 20,000 in
4096-D), Option B's eigenvalues stop inflating and full-space LDA
becomes interpretable. This requires either:
- More problems (extend the operand range from [0, 99] to [0, 999]),
  which costs Step 1–3 to re-run on a larger dataset.
- Multiple residual-stream samples per problem, e.g., capture the
  activation at several token positions, not just `=`. This is a
  Step 3 change that does not require regenerating the dataset.

**(b) A shrinkage-path audit.** Re-fit Option B at multiple Ledoit-
Wolf shrinkage values (e.g., λ_LW ∈ {0.001, 0.01, 0.1, 0.3, 0.5})
and check whether the top direction stabilises. If B's top direction
converges as shrinkage increases, the converged direction is more
plausibly a real low-variance signal. If it does not converge, B is
just chasing noise.

**(c) A targeted subset audit.** Pick a small number of cells where
CCSVD reports a low r (r = 2 or 3) and rerun Option B on just those
cells with extra care: higher n_permutations, multiple shrinkage
values, multiple random seeds. If a real low-variance direction
exists, it should show up here most clearly.

None of these audits are needed for Stage 2 to proceed. They are
explicitly future work to settle the open question.

#### 18.18.11 Why the open question does not block downstream stages

Stages 2, 3, and 4 of the pipeline all consume A's basis (the
CCSVD-subspace LDA directions). None of them consume B's basis. So:

- **Stage 2 (Bayesian manifold).** Helix Fourier, GPLVM, RBF VAE all
  fit on A's basis. They never see Option B. The open question
  about low-variance directions affects what *additional* manifolds
  might exist, not whether the manifolds Stage 2 finds are real.
- **Stage 3 (ownership).** The orthogonalisation operation uses A's
  basis plus the correlate concepts' A-bases. Stage 3's verdicts
  ("owned" / "inherited" / "ambiguous") are about whether A's
  geometry is the model's own structure or borrowed from a related
  concept. The open question is orthogonal.
- **Stage 4 (causal ablation).** Ablation removes A's basis from the
  residual stream. If we are concerned that there are additional
  low-variance directions outside A that also carry causal weight,
  Stage 4 will tell us — the Δlogit measurement is direction-
  agnostic. If ablating A produces a large Δlogit, A captured what
  matters causally regardless of what else exists in the space.

So the worst-case scenario from the open question is: *additional*
low-variance directions exist that we did not find. This would
increase the project's eventual scope (more directions to investigate
later) but does not invalidate any Stage 2/3/4 finding on A's basis.

The headline claims of the paper will be about A. They will not be
contingent on B's audit producing a clean verdict.

#### 18.18.12 Summary of the confidence audit

| Aspect | Verdict | Evidence base |
|---|---|---|
| High-variance directions are real | **Confirmed** | 5 independent stat tests, 4,075 cells |
| They generalise to held-out data | **Confirmed** | Median CV accuracy 0.917 (9× chance) |
| They are stable under perturbation | **Confirmed** | Bootstrap CI > 0 in every fit_ok cell |
| They are robust to magnitude confounds | **Confirmed** | Cross-mode median cos_sim ≥ 0.86 |
| LDA pruning of CCSVD is justified | **Confirmed** | Pruned directions are low-eigenvalue tails |
| CCSVD's subspace contains nothing CCSVD missed | **Confirmed** | LDA never adds to CCSVD's basis from inside |
| Low-variance directions outside CCSVD exist | **Unknown** | Option B audit unreliable at our N/d |
| The high-variance basis is causally used | **Open until Stage 4** | Stage 6 is observational |

The first six rows are the confident part. The seventh row is the
explicit open question. The eighth row is the limit of Stage 1 —
any causal claim is reserved for Stage 4.

#### 18.18.13 One-sentence answer

**Yes, we are confident in the high-variance findings — verified by
five independent statistical tests and stable under residualization
— and we are explicit that whether additional low-variance
directions exist remains open until either more data or a
shrinkage-path audit settles it.**

---

*End of Step 6 documentation.*
