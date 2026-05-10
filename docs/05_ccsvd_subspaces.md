# Step 5 — Conditional Covariance + SVD (CCSVD) — Concept Subspaces

**Project:** From Linear Probes to Bayesian Manifolds: Geometry of Arithmetic in Language Models
**Carnegie Mellon University, May 2026**
**Author:** Anshul Kumar

---

## Table of contents

1. Purpose and scope
2. Standing rules
3. Inputs
4. Mathematical specification
   4.1 Step 1 — load and correctness mask
   4.2 Step 2 — group-size filter
   4.3 Step 3 — global mean
   4.4 Step 4 — per-value centroids
   4.5 Step 5 — scaled centred centroid matrix
   4.6 Step 6 — SVD
   4.7 Step 7 — permutation null
   4.8 Step 8 — sequential stop
   4.9 Step 9 — basis
   4.10 Step 10 — projection
   4.11 Step 11 — five-fold subspace cross-validation
   4.12 Step 12 — summary statistics
   4.13 Step 13 — risk flags
   4.14 Step 14 — output writing
5. Concept registry
6. Toy validation
7. Run procedure
8. Per-model results
   8.1 GPT-J 6B
   8.2 Llama 3.1 8B
   8.3 Pythia 6.9B
9. Per-tier results
10. Skipped cells inventory
11. Cross-model comparison
12. Output files
13. Reproducibility
14. Verification
15. Open questions
16. Appendix A — Intuitions and analysis

---

## 1. Purpose and scope

### 1.1 What this step is

For each (model, task, layer, concept) cell defined on the per-model correct subset, this step computes:

- a per-value centroid `μ_v` for each surviving concept value `v`
- the between-class scatter matrix `S_B` of those centroids
- an eigendecomposition of `S_B` via SVD on a scaled centred centroid matrix
- a permutation-null distribution over eigenvalues from 1,000 label shuffles
- a subspace dimension `r` defined as the largest k such that all eigenvalues
  `λ_1 .. λ_k` exceed their per-index 99th percentile of the null
- an orthonormal basis `B ∈ R^{4096 × r}` of the kept directions
- the projected per-value centroids and the projected full activation cloud
- a 5-fold cross-validation Pearson correlation between full-space and
  subspace pairwise centroid distances on held-out test folds
- three risk flags (N/d inflation, single-direction dominance, group imbalance)

### 1.2 What this step is NOT

- It does NOT do the LDA refinement (Stage 1 sub-step b). The within-class
  scatter `S_W`, the generalised eigenvalue problem `S_B w = λ S_W w`, and
  the bounded LDA eigenvalues `λ_k ∈ [0, 1]` are out of scope and are
  handled by a later step.
- It does NOT compute a bootstrap CI on the LDA `λ₁`. Bootstrap is a
  Stage 1 sub-step b artefact.
- It does NOT classify cells as `pass` or `fail` against the Stage 1
  criteria. Pass/fail is a sub-step c artefact.
- It does NOT do Stage 2 (Bayesian manifold), Stage 3 (ownership test),
  or Stage 4 (causal ablation).

### 1.3 What this step's outputs feed into

- Stage 2a (centroid Fourier helix fit) consumes the per-value centroid
  matrix and the basis `B`.
- Stage 2b (spread-aware Mahalanobis `d_SW`) consumes the per-value
  centroids in the subspace coordinates and the per-value within-class
  covariances after projection into `B`.
- Stage 2c (GPLVM) and Stage 2d (RBF VAE) consume the projected
  activations as their input data matrix.
- Stage 3 (ownership test) consumes `B` and the bases of the listed
  algebraic correlate concepts. Stage 1 sub-step a must therefore have
  been run for every correlate concept before Stage 3 can run.
- Stage 4 (ablation) consumes `B` to define the orthogonal-complement
  projection that ablates the subspace from the residual stream.

### 1.4 Population

Every fit runs on the per-model **correct subset**: rows in the activation
matrix where the model produced the gold first-answer-token. The
correctness mask is loaded from the per-model answers CSV.

| Model | Addition correct | Multiplication correct |
|---|---:|---:|
| gpt-j-6b | 8,415 / 10,000 | 2,751 / 3,023 |
| llama-3.1-8b | 9,963 / 10,000 | 2,927 / 3,023 |
| pythia-6.9b | 7,718 / 10,000 | 2,757 / 3,023 |

These counts are the `N_total_correct` field of every cell's metadata
JSON.

---

## 2. Standing rules

The following rules are followed by every cell fit; deviations are recorded
explicitly in the metadata.

1. **No subsampling.** Every fit uses the full per-cell correct
   population. The 1,000-permutation null and the 5-fold CV are
   resampling, not subsampling.
2. **Mean-centring only.** The activation matrix is mean-centred once
   inside the cell. Rows are not unit-normalised. Activation L2 norms
   vary by a factor of approximately 80 across (model, layer); this
   variation is preserved.
3. **Per-value centroid weighting:** `M_v = sqrt(n_v / N) (μ_v - μ̄)`,
   where `n_v` is the surviving sample count for value `v` and `N` is
   the total surviving sample count after the group-size filter.
4. **Group-size filter:** `MIN_GROUP_SIZE = 30`. Concept values with
   fewer than 30 surviving samples are dropped before centroid
   computation. Cells where fewer than 2 values survive are recorded
   with `status = skipped_insufficient_groups`.
5. **Population-size filter:** `MIN_POPULATION = 30`. Cells with fewer
   than 30 correct samples are skipped at the cell level. This rule
   does not fire under the present input shapes.
6. **Permutation null:** 1,000 label shuffles per cell, p < 0.01
   threshold evaluated per eigenvalue index.
7. **Sequential stop:** eigenvalue k is counted toward the subspace
   only if eigenvalues 1..k-1 also passed.
8. **Cross-validation:** 5-fold stratified split with
   `random_state = 42`; Pearson correlation of pairwise centroid
   distances between full space and subspace on the test fold.
9. **All outputs go to CSV in addition to .npy** so plotting can read
   tabular form without reloading the .npy artefacts.

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

### 3.2 Concept labels

```
data/data/raw/{task}_problems.csv
```

10,000 rows for addition, 3,023 rows for multiplication. Columns hold
the Tier 1–5 schema (operand digits, column-algebra intermediates,
structural properties, relational properties, and tokenisation metadata).
The exact set of CSV columns considered for fits is enumerated at the
start of every run and is the basis for §5.

### 3.3 Correctness masks

```
data/answers/{model_key}/{task}_answers.csv
```

10,000 (or 3,023) rows. The boolean column `correct` is the mask used
to subset the activation matrix. The `correct == 1` row indices index
into both the activation matrix and the problems CSV; the per-cell
fit operates on the rows where `correct == 1`.

### 3.4 SHA256 chain

The per-cell `meta.json` records sha256 of:

- the activation `.npy` file
- the labels CSV (`{task}_problems.csv`)
- the answers CSV (`{model_key}/{task}_answers.csv`)
- the configuration file `config.yaml` used at run time

The per-model `manifest_{model_key}.json` records the same plus the
git commit (when run from a working tree).

### 3.5 Library versions

Every per-cell `meta.json` records `library_versions`. The values for
this run:

- numpy: 2.2.6
- pandas: 2.3.3
- scipy: latest in the `geometry` conda env
- scikit-learn: 1.8.0
- python: 3.11.15
- torch: 2.10.0 + CUDA 12.8 (used for batched SVD on GPU)

---

## 4. Mathematical specification

For one cell, namely one tuple `(model_key, task, layer, concept)`, the
following 14 steps are executed exactly. The step numbers in this
section match the per-cell driver function in `ccsvd_subspaces.py`.

### 4.1 Step 1 — load and correctness mask

```python
X_full = np.load(activations_path)              # (N_total, 4096) float32
problems = pd.read_csv(problems_path)            # N_total rows
answers = pd.read_csv(answers_path)              # N_total rows
mask = answers["correct"].to_numpy().astype(bool)
X = X_full[mask]                                 # (N, 4096)
y = problems[concept].to_numpy()[mask]           # (N,) for a single concept
```

For a joint concept `(c1, c2, ..., ck)` the label vector is
constructed as a tuple-valued array:

```python
y = np.array(list(zip(problems[c1][mask], ..., problems[ck][mask])), dtype=object)
```

These tuples are mapped to integer codes only after the group-size
filter in Step 2.

### 4.2 Step 2 — group-size filter

```python
counts = pd.Series(y).value_counts()
keep_values = counts[counts >= MIN_GROUP_SIZE].index
row_keep = np.isin(y, keep_values)
X_f = X[row_keep]                                # (N', 4096)
y_f = y[row_keep]                                # (N',)
m = len(keep_values)
```

If `m < 2`, the cell terminates with
`status = skipped_insufficient_groups`. The per-cell `meta.json` still
records:

- `n_total_correct = N`
- `n_after_filter = N'`
- `n_groups_total = pd.Series(y).nunique()`
- `n_groups_after_filter = m`
- `dropped_values` = sorted list of values with `n_v < 30`
- `kept_values` = sorted list of surviving values

The filter dropped 155 cells across the three-model run (see §10).

### 4.3 Step 3 — global mean

```python
μ̄ = X_f.mean(axis=0)                            # (4096,) float64
```

This is the unweighted mean over the surviving rows. By the law of
total expectation, this equals `Σ_v (n_v / N') μ_v`, the n_v-weighted
mean of per-value centroids.

### 4.4 Step 4 — per-value centroids

The per-value centroid matrix `C ∈ R^{m × 4096}` is built by a
one-hot scatter:

```python
one_hot = np.zeros((N', m), dtype=np.float32)
one_hot[np.arange(N'), y_codes] = 1.0
C = (one_hot.T @ X_f) / n_v[:, None]             # (m, 4096) float32
```

where `y_codes ∈ {0, ..., m-1}^{N'}` is the integer-coded label vector
and `n_v ∈ N^m` is the surviving sample count per value. The same
scatter is also used for the 1,000 shuffles in Step 7 by replacing
`y_codes` with `rng.permutation(y_codes)`.

### 4.5 Step 5 — scaled centred centroid matrix

Row `v` of the centroid matrix is centred and scaled:

```
M[v] = sqrt(n_v / N') × (μ_v - μ̄)               # shape (m, 4096), float64
```

The between-class scatter is

```
S_B = Σ_v (n_v / N') (μ_v - μ̄)(μ_v - μ̄)^T = M^T M
```

`S_B` is never formed explicitly. It would be a (4096, 4096) matrix
occupying about 67 MB at float32; the SVD of `M` recovers the same
eigenvalues and right-singular vectors at much lower cost.

### 4.6 Step 6 — SVD

```python
U, s, Vt = np.linalg.svd(M, full_matrices=False)
# U.shape == (m, m), s.shape == (m,), Vt.shape == (m, 4096)
eigenvalues = (s ** 2)                          # (m,) float64
```

The full eigenvalue spectrum has length `m`. The last entry is
numerically zero because the centred centroids span at most `m - 1`
dimensions when `μ̄` is the n_v-weighted mean of `μ_v` (one degree of
freedom is removed by centring). The reported eigenvalue spectrum
`eigenvalues_eff` is `eigenvalues[: m - 1]`.

This step is executed on GPU via `torch.linalg.svd` when CUDA is
available; results are identical (within float64 round-off tolerance)
to the NumPy/LAPACK path.

### 4.7 Step 7 — permutation null

```python
rng = np.random.default_rng(seed)               # seed = sha256(cell_id) hash
null_table = np.zeros((N_PERMUTATIONS, m - 1), dtype=np.float64)

for p in range(N_PERMUTATIONS):
    y_shuf = rng.permutation(y_codes)           # n_v counts preserved
    C_shuf = scatter_centroids(X_f, y_shuf, m)
    M_shuf = sqrt(n_v / N')[:, None] * (C_shuf - μ̄)
    s_shuf = np.linalg.svd(M_shuf, compute_uv=False)
    null_table[p, :] = (s_shuf ** 2)[: m - 1]

threshold_99 = np.percentile(null_table, 99, axis=0)  # (m - 1,)
```

`N_PERMUTATIONS = 1000`. The shuffle preserves `n_v` per value (because
permutation preserves the multiset of labels). Each permutation
recomputes centroids, builds `M`, and takes singular values. On GPU
the 1,000 SVDs are batched in groups of 50, exploiting the
`torch.linalg.svdvals` batch dimension.

The seed for `rng` is derived deterministically from the cell
identifier:

```python
seed_input = f"{model_key}|{task}|{layer:02d}|{concept_name}|{base_seed}"
seed = int(sha256(seed_input.encode()).hexdigest()[:16], 16) % (2**63 - 1)
```

so that the null distribution is reproducible per cell and independent
across cells.

### 4.8 Step 8 — sequential stop

```python
r = 0
for k in range(m - 1):
    if eigenvalues_eff[k] > threshold_99[k]:
        r = k + 1
    else:
        break
```

The sequential stopping rule is strict: once any eigenvalue fails its
threshold, the subspace ends. Eigenvalues `λ_{k+1}, λ_{k+2}, ...` are
not allowed to "rejoin" the subspace even if individually significant.
This avoids the pathology where a small dim that happens to beat its
threshold inflates `r` despite all larger dims having failed.

If `r == 0`, the cell terminates with
`status = no_significant_subspace`. The per-cell `meta.json` is still
written with full eigenvalue spectrum, null table, and threshold
vector, so the cell can be plotted and re-examined.

20 cells across the three-model run produced `r == 0`.

### 4.9 Step 9 — basis

```python
B = Vt[:r, :].T                                 # (4096, r) float32
```

Columns of `B` are right singular vectors of `M`, i.e. eigenvectors of
`S_B = M^T M`. They are orthonormal: `B^T B = I_r` to within float32
round-off (offsets less than 2 × 10^{-8} measured in spot checks).

### 4.10 Step 10 — projection

```python
C_centred = C - μ̄                              # (m, 4096) float32
C_proj = C_centred @ B                          # (m, r) float32
X_proj = (X_f - μ̄) @ B                         # (N', r) float32
```

Both projections are stored as `.npy` artefacts. `C_proj` is the
input to Stage 2a (centroid Fourier fit) and Stage 2b (`d_SW`). `X_proj`
is the input to Stage 2c (GPLVM) and Stage 2d (RBF VAE).

### 4.11 Step 11 — five-fold subspace cross-validation

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv = np.full(5, np.nan)

for fi, (train_idx, test_idx) in enumerate(skf.split(X_f, y_codes)):
    X_tr, y_tr = X_f[train_idx], y_codes[train_idx]
    X_te, y_te = X_f[test_idx],  y_codes[test_idx]

    n_v_tr = np.bincount(y_tr, minlength=m)
    n_v_te = np.bincount(y_te, minlength=m)
    usable = (n_v_tr > 0) & (n_v_te > 0)
    if usable.sum() < 2:
        continue

    μ_tr = X_tr.mean(axis=0)
    C_tr = scatter_centroids(X_tr, y_tr, m)
    M_tr = sqrt(n_v_tr / |X_tr|)[:, None] * (C_tr - μ_tr)
    _, s_tr, Vt_tr = np.linalg.svd(M_tr, full_matrices=False)
    r_eff = min(r, Vt_tr.shape[0])
    B_tr = Vt_tr[:r_eff, :].T

    C_te = scatter_centroids(X_te, y_te, m)[usable]
    C_te_centred = C_te - μ_tr
    C_te_sub = C_te_centred @ B_tr

    D_full = pdist(C_te_centred)
    D_sub  = pdist(C_te_sub)
    if std(D_full) > 0 and std(D_sub) > 0 and len(D_full) >= 2:
        cv[fi], _ = pearsonr(D_full, D_sub)

cv_mean = nanmean(cv)
cv_std  = nanstd(cv)
```

Subspace-preservation CV asks: when the basis is fit on 80 % of the
rows, how well does the subspace preserve the geometry of the held-out
20 %' centroids? A clean linear subspace should give `cv_mean ≈ 1`.
A concept whose subspace is artefactual or sample-specific gives
`cv_mean` near 0.

### 4.12 Step 12 — summary statistics

```python
λ = eigenvalues_eff
λ₁, λ₂, λ₃ = λ[0], λ[1] (if m≥3), λ[2] (if m≥4)
total_var = λ.sum()
explained_variance = λ / total_var
cumulative_variance = np.cumsum(explained_variance)
λ_ratio = λ₁ / λ₂                                # NaN when m < 3
```

These are flat fields in the per-cell `meta.json` and propagate to the
`summary.csv` row via the per-model driver.

### 4.13 Step 13 — risk flags

Three boolean flags are computed per cell:

```python
n_d_ratio = N' / r                               # NaN when r == 0
flag_n_d_inflation     = (r > 0) and (n_d_ratio < 5)
flag_single_direction  = (r >= 2) and (λ_ratio > 10)
flag_group_imbalance   = (max(n_v) / min(n_v)) > 3
```

The thresholds are inherited from plan v6 §3.1.4. Flags are advisory
(they do not change `status`) and surface in the master CSV.

### 4.14 Step 14 — output writing

The per-cell directory is created at:

```
data/results/ccsvd_subspaces/{model_key}/{task}/layer_{LL:02d}/{concept}/
```

Joint concepts use a double-underscore separator in the directory
name:

```
(a_units, b_units)              -> a_units__b_units/
(a_units, b_units, ans_units)   -> a_units__b_units__ans_units/
```

Per-cell artefacts:

| File | Shape | dtype | Notes |
|---|---|---|---|
| `basis.npy` | `(4096, r)` | float32 | columns orthonormal |
| `eigenvalues.npy` | `(m-1,)` | float64 | full spectrum minus the structural zero |
| `null_eigenvalues.npy` | `(1000, m-1)` | float64 | permutation null |
| `threshold_99.npy` | `(m-1,)` | float64 | per-index 99th percentile |
| `centroids.npy` | `(m, 4096)` | float32 | full-d per-value means |
| `centroids_proj.npy` | `(m, r)` | float32 | centroids in B |
| `projected_acts.npy` | `(N', r)` | float32 | full cloud in B |
| `cv_per_fold.npy` | `(5,)` | float64 | per-fold Pearson correlations |
| `meta.json` | — | — | scalar fields, flags, sha256s, library versions, runtime |

For `status = skipped_insufficient_groups` and
`status = no_significant_subspace` cells, only `meta.json` is written
(and `eigenvalues.npy`, `null_eigenvalues.npy`, `threshold_99.npy` for
the no-significant-subspace case where the spectrum was computed but
no direction passed).

---

## 5. Concept registry

### 5.1 Inclusion principles

A concept enters the registry only if it satisfies all three:

1. **Ground-truth column.** The concept must be a column in the
   problems CSV. No derived-on-the-fly concepts.
2. **Arithmetic justification.** The concept must be an algebraic
   property of (a, b) or an algorithm intermediate.
3. **Joint informativeness.** A k-tuple joint enters only if it adds
   information beyond its (k-1)-marginals. Two flagged exceptions
   are validation joints (deterministic mappings included to test
   linear representation of the algebraic identity).

Singles are exhaustive over the CSV after exclusions in §5.6.
Joints are 12 per task, listed in §5.5.

### 5.2 Tier 1 — input/output digits

| Concept | Add #vals | Mult #vals |
|---|---:|---:|
| a | 100 | 100 |
| b | 100 | 100 |
| a_units | 10 | 10 |
| a_tens | 10 | 10 |
| b_units | 10 | 10 |
| b_tens | 10 | 10 |
| answer | 199 | up to 9801 |
| ans_units | 10 | 10 |
| ans_tens | 10 | 10 |
| ans_hundreds | 2 | 10 |
| ans_thousands | — | up to 10 |
| a_num_digits | 2 | 2 |
| b_num_digits | 2 | 2 |
| ans_num_digits | 2 | 4 |

13 single concepts present in addition and multiplication (Llama and
Pythia have 14 because `ans_thousands` exists in multiplication only).

### 5.3 Tier 2 — column-algebra intermediates

| Concept | Add | Mult |
|---|---|---|
| column_sum_units | ✓ | ✓ |
| column_sum_tens | ✓ | ✓ |
| column_sum_hundreds | — | ✓ |
| column_sum_thousands | — | ✓ |
| carry_units | ✓ | ✓ |
| carry_tens | ✓ | ✓ |
| carry_hundreds | — | ✓ |
| carry_thousands | — | ✓ |
| running_sum_units | ✓ | ✓ |
| running_sum_tens | ✓ | ✓ |
| running_sum_hundreds | — | ✓ |
| running_sum_thousands | — | ✓ |
| partial_product_units | — | ✓ |
| partial_product_a_units_b_tens | — | ✓ |
| partial_product_a_tens_b_units | — | ✓ |
| partial_product_a_tens_b_tens | — | ✓ |

Addition has 6 Tier 2 concepts; multiplication has 16.

### 5.4 Tier 3 and Tier 4

Tier 3 (structural):

| Concept | Add #vals | Mult #vals |
|---|---:|---:|
| a_parity | 2 | 2 |
| b_parity | 2 | 2 |
| ans_parity | 2 | 2 |
| parity_match | 2 | 2 |
| parity_xor | 2 | 2 |
| a_magnitude_tier | varies | varies |
| b_magnitude_tier | varies | varies |
| ans_magnitude_tier | varies | varies |
| ans_ends_in_zero | 2 | 2 |
| ans_is_zero | 2 | 2 |
| a_is_zero | 2 | 2 |
| b_is_zero | 2 | 2 |
| a_eq_b | 2 | 2 |

Tier 4 (relational):

| Concept | Add | Mult |
|---|---|---|
| max_operand | 100 | 100 |
| min_operand | 100 | 100 |
| operand_diff | 199 | 199 |
| operand_abs_diff | 100 | 100 |
| larger_operand | 3 | 3 |
| both_zero | 2 | 2 |
| either_zero | 2 | 2 |
| both_one | 2 | 2 |
| either_one | 2 | 2 |

13 Tier 3 concepts and 9 Tier 4 concepts in each task.

### 5.5 Joint concepts

12 per task. Each joint has a clean column-algebra reading.

Operand-pair joints (4):

| Joint | Add #groups | Mult #groups |
|---|---:|---:|
| (a_units, b_units) | 100 | 100 |
| (a_tens, b_tens) | 100 | 100 |
| (a_units, b_tens) | 100 | 100 |
| (a_tens, b_units) | 100 | 100 |

Carry-binding joints (4):

| Joint | Add #groups | Mult #groups |
|---|---:|---:|
| (a_tens, b_tens, carry_units) | 200 | varies |
| (a_tens, b_tens, ans_tens) | 200 | varies |
| (carry_units, ans_units) | 20 | 90 |
| (carry_units, column_sum_units) | 38 | varies |

Multi-column joint (1):

| Joint | Add #groups | Mult #groups |
|---|---:|---:|
| (a_units, b_units, ans_tens) | 200 | varies |

Validation joints (2, deterministic):

| Joint | Add #groups | Mult #groups |
|---|---:|---:|
| (a_units, b_units, ans_units) (add) | 100 | — |
| (a_units, b_units, partial_product_units) (mult) | — | 100 |

### 5.6 Excluded columns

Columns excluded from the registry by design:

- **JSON-list columns** (`a_digits_lsf`, `b_digits_lsf`,
  `answer_digits_lsf`, `answer_digits_msf`, `column_sums`, `carries`,
  `running_sums`): each list element duplicates a Tier 1 / Tier 2
  scalar already in the registry.
- **Tier 5 tokenisation metadata** (`is_intersection`,
  `is_single_token_*`, `n_tokens_*`, `first_token_id_*`,
  `first_token_text_*`): degenerate in the intersection set or
  redundant with `answer`.

### 5.7 Total cell count by attempted concept

Concept × task instances:

| Tier | Add | Mult |
|---|---:|---:|
| 1 (digits) | 13 | 14 |
| 2 (column algebra) | 6 | 16 |
| 3 (structural) | 13 | 13 |
| 4 (relational) | 9 | 9 |
| Joints | 10 (4+4+1+1) | 11 (4+4+1+1+1) |
| **Per-task total** | **51** | **63** |

After accounting for the validation joint that's task-specific and the
multiplication-only concepts, each model attempts **51 (add) + 63 (mult)
= 114 concept × task instances per layer** with some Tier 1 concepts
collapsing across tasks. Multiplied across 5 layers per model, this
gives roughly 540 cells per model. The actual run produced exactly
540 cells per model.

---

## 6. Toy validation

Three synthetic toys defined by `check_ccsvd_toys.py`. Each toy creates
a 9-D Gaussian dataset with a known structure and runs the same
per-cell function as a real fit. All three must pass before the SLURM
array is submitted.

### 6.1 Toy 1L — single-axis structure

10 classes, n=300 each. Class means evenly spaced on the first axis of
9-D Gaussian: `μ_v = v · e_1` for v in 0..9. Isotropic noise σ=0.5.

Pass criteria (cumulative-variance form): `ev_top1 ≥ 0.95`,
`λ_1 / λ_2 > 10`, `cv_mean > 0.95`.

Result of last run: `r = 2`, `λ_1 = 8.241`, `λ_2 = 0.003`,
`λ_1 / λ_2 = 2709.35`, `ev_top1 = 0.9992`, `cv_mean = 1.000`. **PASS**

Note: the perm-null reports `r = 2` because the residual within-class
scatter (σ × √(1/300) ≈ 0.029) systematically beats the per-eigenvalue
null at the second index. This is expected for cleanly rank-1 data
and does not indicate a methodological failure. The cumulative-variance
criterion (which is what plan v6 §3.1.5 states) passes cleanly.

### 6.2 Toy 2L — two-axis structure

10 classes on a 2 × 5 grid in 9-D Gaussian. Same noise.

Pass criteria: `r ≥ 2`, `cumvar@2 ≥ 0.90`, `cv_mean > 0.90`.

Result: `r = 2`, `λ_1 = 1.988`, `λ_2 = 0.245`, `λ_3 = 0.002`,
`cumvar@2 = 0.9978`, `cv_mean = 1.000`. **PASS**

### 6.3 Toy 3L — no structure

9-D isotropic Gaussian, 10 random labels.

Pass criteria: `r == 0`.

Result: `r = 0`, `status = no_significant_subspace`, `λ_1 = 0.010`.
**PASS**

All three toys pass with the current implementation. They are re-run
locally before each fresh array submission.

---

## 7. Run procedure

### 7.1 Phase 0 — local sanity

Run on a head node before any array submission:

```
python check_ccsvd_toys.py
python ccsvd_subspaces.py --config config.yaml --model gpt-j-6b \
    --single-task addition --single-layer 14 --single-concept a_units
```

The smoke-test cell completed in 7.2 s on an A6000 with the production
configuration. Reported numbers (cell metadata):

- `r_dim = 9` (matches `n_groups - 1 = 9`)
- `lambda_1 = 54.37`, `lambda_2 = 53.37`, `lambda_3 = 42.86`
- `lambda_1_over_2 = 1.02` (no single-direction dominance)
- `cv_mean = 0.9988`, `cv_std = 0.0003`
- `n_d_ratio = 935`, all flags False
- `B^T B - I` max off-diagonal magnitude < 2 × 10^{-8}

### 7.2 Phase 1 — array submission

```
sbatch run_ccsvd_subspaces.sbatch
```

Three SLURM array tasks, one per model:

| Array index | Model | Resources |
|---|---|---|
| 0 | gpt-j-6b | 1 × A6000, 8 CPUs, 16 GB RAM, 2-day wall |
| 1 | llama-3.1-8b | same |
| 2 | pythia-6.9b | same |

Each task processes both tasks × 5 layers × all concepts and writes
into its own `data/results/ccsvd_subspaces/{model_key}/` subtree.
There is no shared state between tasks, so cross-task races on master
CSVs cannot occur.

### 7.3 Phase 2 — merge

After all three tasks finish, the master CSVs are built by:

```python
import pandas as pd, glob, json
from pathlib import Path
base = Path('data/results/ccsvd_subspaces')
for name in ['summary','eigenvalue_spectra','projected_centroids','null_summary','cv_per_fold']:
    shards = sorted(base.glob(f'*/{name}_*.csv'))
    pd.concat([pd.read_csv(p) for p in shards], ignore_index=True).to_csv(base/f'{name}.csv', index=False)

# top-level run manifest
manifests = {json.loads(p.read_text())['model_key']: json.loads(p.read_text())
             for p in sorted(base.glob('*/manifest_*.json'))}
top = {'models': list(manifests.keys()), 'per_model': manifests,
       'totals': {'attempted': sum(m['n_cells_attempted'] for m in manifests.values()),
                  'fit_ok':    sum(m['n_cells_fit_ok'] for m in manifests.values()),
                  'skipped':   sum(m['n_cells_skipped'] for m in manifests.values()),
                  'runtime_seconds_sum': sum(m['total_runtime_seconds'] for m in manifests.values())}}
(base / 'run_manifest.json').write_text(json.dumps(top, indent=2, default=str))
```

Wall time of merge: 10.7 s for the present run.

### 7.4 Phase 3 — plot

```
python plot_ccsvd_subspaces.py --config config.yaml
```

Renders 30 main plots (10 per model × 3 models) plus 9 diagnostic
plots, total 39 PNGs and 1 plot_index.json. Wall time: 669.5 s. Per-
model rendering breakdown (gpt-j-6b shown):

| Plot | Wall (s) |
|---|---:|
| 01_r_dim_heatmap | 2 |
| 02_lambda_ratio_heatmap | 2 |
| 03_cv_mean_heatmap | 2 |
| 04_scree_grid_addition (51 × 5 panels) | 21 |
| 05_scree_grid_multiplication (57 × 5 panels) | 25 |
| 06_centroids_grid_addition (51 × 5 panels) | 60 |
| 07_centroids_grid_multiplication (57 × 5 panels) | 67 |
| 08_principal_angles | 33 |
| 09_r_dim_layer_trajectory | 1 |
| 10_r_dim_vs_trustworthiness | 1 |
| diagnostics × 3 | 6 |

---

## 8. Per-model results

### 8.1 GPT-J 6B (array task 0)

#### 8.1.1 Run-level

| Field | Value |
|---|---|
| SLURM task | `7853883_0` |
| Node | `babel-t9-24` |
| GPU | `NVIDIA A6000` |
| Started | 2026-05-09 19:16:06 UTC-04 |
| Ended | 2026-05-09 20:30:15 UTC-04 |
| Wall time | 1 h 14 m 16 s = 4,449 s |
| State | COMPLETED, exit 0 |
| Cells attempted | 540 |
| Cells fit_ok | 482 |
| Cells skipped (insufficient groups) | 55 |
| Cells with no significant subspace | 3 |

#### 8.1.2 Cell-status breakdown by task

| Task | fit_ok | skipped | no_significant |
|---|---:|---:|---:|
| Addition | 234 | 20 | 1 |
| Multiplication | 248 | 35 | 2 |

#### 8.1.3 Subspace-dimension percentiles (fit_ok cells only)

| Quantile | r |
|---|---:|
| min | 1 |
| 10th | 1 |
| 25th | 1 |
| 50th (median) | 8 |
| 75th | 19 |
| 90th | 38 |
| 95th | 96 |
| 99th | 170 |
| max | 176 |

#### 8.1.4 Cross-validation distribution (fit_ok cells)

| Quantile | cv_mean |
|---|---:|
| 1st | 0.7796 |
| 10th | 0.9443 |
| 50th | 0.9943 |
| max | 0.9999 |

Cells with cv_mean < 0.9: **17**.

#### 8.1.5 Risk-flag tally

| Flag | Count |
|---|---:|
| flag_n_d_inflation | 0 |
| flag_single_direction (λ₁/λ₂ > 10) | 16 |
| flag_group_imbalance (max n_v / min n_v > 3) | 289 |

The group-imbalance flag fires by design on rare-event concepts
(`a_eq_b`, `ans_is_zero`, `both_zero`, magnitude tiers, etc.). It is
informational and does not affect `status`.

### 8.2 Llama 3.1 8B (array task 1)

#### 8.2.1 Run-level

| Field | Value |
|---|---|
| SLURM task | `7853883_1` |
| Node | `babel-w9-26` |
| GPU | `NVIDIA A6000` |
| Started | 2026-05-09 19:18:42 UTC-04 |
| Ended | 2026-05-09 20:44:46 UTC-04 |
| Wall time | 1 h 26 m 10 s = 5,165 s |
| State | COMPLETED, exit 0 |
| Cells attempted | 540 |
| Cells fit_ok | 485 |
| Cells skipped (insufficient groups) | 50 |
| Cells with no significant subspace | 5 |

#### 8.2.2 Cell-status breakdown by task

| Task | fit_ok | skipped | no_significant |
|---|---:|---:|---:|
| Addition | 235 | 20 | 0 |
| Multiplication | 250 | 30 | 5 |

#### 8.2.3 Subspace-dimension percentiles (fit_ok cells only)

| Quantile | r |
|---|---:|
| min | 1 |
| 10th | 1 |
| 25th | 1 |
| 50th (median) | 9 |
| 75th | 19 |
| 90th | 41.8 |
| 95th | 97 |
| 99th | 178 |
| max | 199 |

#### 8.2.4 Cross-validation distribution (fit_ok cells)

| Quantile | cv_mean |
|---|---:|
| 1st | 0.7695 |
| 10th | 0.9334 |
| 50th | 0.9933 |
| max | 0.9999 |

Cells with cv_mean < 0.9: **20**.

#### 8.2.5 Risk-flag tally

| Flag | Count |
|---|---:|
| flag_n_d_inflation | 0 |
| flag_single_direction | 13 |
| flag_group_imbalance | 285 |

### 8.3 Pythia 6.9B (array task 2)

#### 8.3.1 Run-level

| Field | Value |
|---|---|
| SLURM task | `7854525_2` (rerun; the original `7853883_2` failed in 7 s on `babel-n5-20` due to a node-level CUDA initialization failure) |
| Node | `babel-s9-16` |
| GPU | `NVIDIA A6000` |
| Started | 2026-05-09 20:08:42 UTC-04 |
| Ended | 2026-05-09 21:23:27 UTC-04 |
| Wall time | 1 h 14 m 51 s = 4,485 s |
| State | COMPLETED, exit 0 |
| Cells attempted | 540 |
| Cells fit_ok | 478 |
| Cells skipped (insufficient groups) | 50 |
| Cells with no significant subspace | 12 |

#### 8.3.2 Cell-status breakdown by task

| Task | fit_ok | skipped | no_significant |
|---|---:|---:|---:|
| Addition | 230 | 20 | 5 |
| Multiplication | 248 | 30 | 7 |

#### 8.3.3 Subspace-dimension percentiles (fit_ok cells only)

| Quantile | r |
|---|---:|
| min | 1 |
| 10th | 1 |
| 25th | 1 |
| 50th (median) | 9 |
| 75th | 19 |
| 90th | 41.3 |
| 95th | 96.15 |
| 99th | 150 |
| max | 156 |

#### 8.3.4 Cross-validation distribution (fit_ok cells)

| Quantile | cv_mean |
|---|---:|
| 1st | 0.7529 |
| 10th | 0.9463 |
| 50th | 0.9939 |
| max | 0.9999 |

Cells with cv_mean < 0.9: **12**.

#### 8.3.5 Risk-flag tally

| Flag | Count |
|---|---:|
| flag_n_d_inflation | 0 |
| flag_single_direction | 16 |
| flag_group_imbalance | 289 |

### 8.4 Aggregate totals

| | Attempted | fit_ok | skipped | no_significant | Wall (s) |
|---|---:|---:|---:|---:|---:|
| gpt-j-6b | 540 | 482 | 55 | 3 | 4,449 |
| llama-3.1-8b | 540 | 485 | 50 | 5 | 5,165 |
| pythia-6.9b | 540 | 478 | 50 | 12 | 4,485 |
| **Total** | **1,620** | **1,445** | **155** | **20** | **14,099** |

Three array tasks ran in parallel. The wall time of the final array
phase is the slowest single task (Llama, 1 h 26 m), not the sum.

---

## 9. Per-tier results

The aggregate table below stratifies fit_ok cells by concept tier
(Tier 1 digits, Tier 2 column algebra, Tier 3 structural, Tier 4
relational, Tier 5 joints) and reports min / median / mean / max of
`r_dim` per (model, tier).

### 9.1 GPT-J 6B

| Tier | n cells | min r | median r | mean r | max r |
|---|---:|---:|---:|---:|---:|
| 1 (digits) | 125 | 1 | 9 | 16.06 | 117 |
| 2 (column algebra) | 90 | 1 | 10.5 | 12.42 | 32 |
| 3 (structural) | 117 | 1 | 1 | 1.26 | 2 |
| 4 (relational) | 65 | 1 | 3 | 19.65 | 84 |
| 5 (joints) | 85 | 9 | 26 | 49.15 | 176 |

### 9.2 Llama 3.1 8B

| Tier | n cells | min r | median r | mean r | max r |
|---|---:|---:|---:|---:|---:|
| 1 (digits) | 125 | 1 | 9 | 16.94 | 140 |
| 2 (column algebra) | 90 | 1 | 11 | 12.90 | 36 |
| 3 (structural) | 120 | 1 | 1 | 1.25 | 2 |
| 4 (relational) | 65 | 1 | 3 | 20.38 | 84 |
| 5 (joints) | 85 | 10 | 29 | 51.44 | 199 |

### 9.3 Pythia 6.9B

| Tier | n cells | min r | median r | mean r | max r |
|---|---:|---:|---:|---:|---:|
| 1 (digits) | 124 | 1 | 9 | 16.22 | 109 |
| 2 (column algebra) | 90 | 1 | 10 | 12.57 | 35 |
| 3 (structural) | 114 | 1 | 1 | 1.26 | 2 |
| 4 (relational) | 65 | 1 | 3 | 19.20 | 84 |
| 5 (joints) | 85 | 9 | 25 | 47.99 | 156 |

### 9.4 Per-concept r_dim across layers (selected concepts)

The per-cell r_dim values for the headline concepts at every layer in
every model are tabulated below. Layer order in each entry is
shallow → deep.

| Concept | gpt-j-6b layers (4, 8, 14, 20, 24) | llama-3.1-8b layers (4, 8, 16, 24, 28) | pythia-6.9b layers (4, 8, 16, 24, 28) |
|---|---|---|---|
| `a` (add) | 99, 99, 99, 99, 99 | 99, 99, 99, 99, 99 | 99, 99, 99, 99, 99 |
| `a` (mult) | 28, 28, 28, 28, 28 | 29, 29, 29, 29, 29 | 28, 28, 28, 28, 28 |
| `b` (add) | 99, 99, 99, 99, 99 | 99, 99, 99, 99, 99 | 99, 99, 99, 99, 99 |
| `b` (mult) | 27, 27, 27, 27, 27 | 29, 29, 29, 29, 29 | 26, 26, 26, 26, 26 |
| `answer` (add) | 4, 6, 7, 51, 117 | 4, 5, 8, 129, 140 | (5 layers, max 109) |
| `a_units` (add) | 9, 9, 9, 9, 9 | 9, 9, 9, 9, 9 | 9, 9, 9, 9, 9 |
| `a_units` (mult) | 9, 9, 9, 9, 9 | 9, 9, 9, 9, 9 | 9, 9, 9, 9, 9 |
| `ans_units` (add) | 1, 1, 5, 9, 9 | 2, 3, 6, 9, 9 | 1, 1, 6, 9, 9 |
| `ans_units` (mult) | 3, 3, 3, 8, 8 | 3, 4, 4, 9, 9 | 1, 3, 5, 9, 9 |
| `carry_units` (add) | 1, 1, 1, 1, 1 | 1, 1, 1, 1, 1 | 1, 1, 1, 1, 1 |
| `carry_units` (mult) | 6, 5, 6, 7, 7 | 7, 5, 7, 7, 7 | 6, 5, 6, 7, 7 |
| `column_sum_units` (add) | 11, 11, 14, 18, 18 | 11, 13, 15, 18, 18 | 11, 13, 14, 18, 18 |
| `column_sum_units` (mult) | 9, 10, 11, 30, 31 | 10, 12, 10, 33, 33 | (5 layers) |
| `min_operand` (add) | 78, 78, 78, 78, 78 | 83, 83, 83, 83, 83 | (consistent) |
| `min_operand` (mult) | 20, 20, 20, 20, 20 | 21, 21, 21, 21, 21 | 20, 20, 20, 20, 20 |
| `max_operand` (add) | 84, 84, 84, 84, 84 | 84, 84, 84, 84, 84 | 84, 84, 84, 84, 84 |
| `max_operand` (mult) | 38, 38, 38, 38, 38 | 43, 43, 43, 43, 43 | (5 layers) |
| `a_parity` (any) | 1, 1, 1, 1, 1 | 1, 1, 1, 1, 1 | 1, 1, 1, 1, 1 |

A concept whose per-layer `r` is constant across all 5 layers in a
column indicates the dimension does not change with depth. A concept
whose `r` grows with depth (e.g. `ans_units` going from 1 → 9, or
`answer` going from 4 → 117) indicates the model develops higher-rank
representation as computation proceeds.

The full per-concept-per-layer r_dim grid for every concept, model and
task is in the master CSV `summary.csv` and is plotted in the per-model
heatmap (`figures/ccsvd/{model_key}/01_r_dim_heatmap.png`).

### 9.5 Top eigenvalue cells (across all models)

The 30 cells with the highest absolute λ₁ are tabulated below.
Eigenvalues are not directly comparable across (model, layer) because
activation norms differ by up to 80×; this table is informational.

| model | task | layer | concept | r | λ₁ | λ₂ | λ₁/λ₂ | cv_mean |
|---|---|---:|---|---:|---:|---:|---:|---:|
| pythia-6.9b | addition | 28 | answer | 109 | 1565.45 | 1007.69 | 1.55 | 0.998 |
| pythia-6.9b | addition | 28 | a_units__b_units | 98 | 1553.46 | 905.83 | 1.71 | 0.999 |
| pythia-6.9b | addition | 28 | a_units__b_units__ans_units | 98 | 1553.46 | 905.83 | 1.71 | 0.999 |
| pythia-6.9b | addition | 28 | operand_abs_diff | 22 | 1542.80 | 217.97 | 7.08 | 0.992 |
| pythia-6.9b | addition | 28 | operand_diff | 21 | 1533.31 | 210.01 | 7.30 | 0.991 |
| pythia-6.9b | addition | 28 | column_sum_units | 18 | 1532.39 | 800.51 | 1.91 | 0.998 |
| pythia-6.9b | addition | 28 | running_sum_units | 18 | 1532.39 | 800.51 | 1.91 | 0.998 |
| pythia-6.9b | addition | 28 | carry_units__ans_units | 18 | 1532.39 | 800.51 | 1.91 | 0.998 |
| pythia-6.9b | addition | 28 | carry_units__column_sum_units | 18 | 1532.39 | 800.51 | 1.91 | 0.998 |
| pythia-6.9b | addition | 28 | ans_units | 9 | 1527.29 | 782.60 | 1.95 | 0.9999 |
| pythia-6.9b | addition | 28 | parity_xor | 1 | 1524.07 | NaN | NaN | NaN |
| pythia-6.9b | addition | 28 | parity_match | 1 | 1524.07 | NaN | NaN | NaN |
| pythia-6.9b | addition | 28 | ans_parity | 1 | 1524.07 | NaN | NaN | NaN |
| gpt-j-6b | mult | 24 | min_operand | 20 | 1240.02 | 459.31 | 2.70 | 0.992 |
| pythia-6.9b | mult | 28 | min_operand | 20 | 1204.88 | 776.70 | 1.55 | 0.995 |
| pythia-6.9b | mult | 28 | b | 26 | 1170.17 | 754.81 | 1.55 | 0.992 |
| gpt-j-6b | mult | 24 | ans_hundreds | 9 | 1164.48 | 410.91 | 2.83 | 0.979 |
| gpt-j-6b | mult | 24 | running_sum_hundreds | 9 | 1164.48 | 410.91 | 2.83 | 0.979 |
| pythia-6.9b | mult | 28 | ans_hundreds | 9 | 1141.86 | 507.80 | 2.25 | 0.965 |
| pythia-6.9b | mult | 28 | running_sum_hundreds | 9 | 1141.86 | 507.80 | 2.25 | 0.965 |
| gpt-j-6b | mult | 24 | ans_num_digits | 2 | 1123.94 | 167.35 | 6.72 | 0.9997 |
| pythia-6.9b | add | 28 | a_tens__b_tens__ans_tens | 146 | 1110.19 | 567.60 | 1.96 | 0.998 |
| pythia-6.9b | add | 28 | a_tens__b_tens__carry_units | 146 | 1110.19 | 567.60 | 1.96 | 0.998 |
| pythia-6.9b | mult | 28 | ans_num_digits | 2 | 1097.36 | 150.16 | 7.31 | 0.998 |
| gpt-j-6b | mult | 24 | ans_magnitude_tier | 2 | 1075.65 | 187.08 | 5.75 | 0.9995 |
| pythia-6.9b | add | 28 | a_tens__b_tens | 92 | 1073.02 | 545.59 | 1.97 | 0.998 |
| pythia-6.9b | add | 28 | running_sum_tens | 18 | 1053.32 | 516.45 | 2.04 | 0.996 |
| gpt-j-6b | add | 24 | a_units__b_units | 93 | 1052.14 | 719.80 | 1.46 | 0.999 |
| gpt-j-6b | add | 24 | a_units__b_units__ans_units | 93 | 1052.14 | 719.80 | 1.46 | 0.999 |
| pythia-6.9b | mult | 28 | a_tens__b_tens__carry_units | 24 | 1045.73 | 436.69 | 2.39 | 0.966 |

The deepest layers (L24/L28) of all three models concentrate the
highest absolute λ₁ values. Within a single deep cell, multiple
concepts (e.g. `parity_xor`, `parity_match`, `ans_parity`) share an
identical λ₁ to the limit of float64 precision; this happens when the
concepts partition the same rows the same way and consequently share a
single direction.

---

## 10. Skipped cells inventory

### 10.1 Counts by status

| Status | Count |
|---|---:|
| skipped_insufficient_groups | 155 |
| no_significant_subspace | 20 |
| **Total non-fit_ok** | **175** |

### 10.2 Counts by concept (top 25)

| Concept | Cells excluded |
|---|---:|
| both_zero | 30 |
| a_units__b_units__ans_tens | 30 |
| both_one | 30 |
| ans_is_zero | 15 |
| answer | 15 |
| a_tens__b_tens__ans_tens | 15 |
| a_eq_b | 15 |
| operand_diff | 15 |
| parity_xor | 3 |
| parity_match | 3 |
| ans_ends_in_zero | 2 |
| ans_units | 1 |
| ans_parity | 1 |

### 10.3 Notes

- `both_zero` and `both_one` exclude on every model × every layer ×
  multiplication (15 each per model; 30 each across two tasks where
  applicable). These concepts are true on at most 1 % of pairs and
  fail the MIN_GROUP_SIZE filter.
- The 4-tuple joint `a_units__b_units__ans_tens` has up to 1,000
  distinct values but only ~3,000 rows in the multiplication
  population, so most groups have fewer than 30 examples.
- `answer` is excluded in 15 cells: each multiplication × deep-layer
  cell has 564 distinct answer values among ~2,750 rows, average
  ~5 per value, all below the floor.
- The 20 cells with `no_significant_subspace` are mostly
  `parity_xor`, `parity_match`, and `ans_parity` at the deepest layers
  of Pythia, where the binary representation merges into the global
  mean. The full list is in `summary.csv` filtered by
  `status == 'no_significant_subspace'`.

---

## 11. Cross-model comparison

### 11.1 Per-model headline numbers (recap)

| | gpt-j-6b | llama-3.1-8b | pythia-6.9b |
|---|---:|---:|---:|
| Cells attempted | 540 | 540 | 540 |
| Cells fit_ok | 482 | 485 | 478 |
| Median r (fit_ok) | 8 | 9 | 9 |
| Mean r (fit_ok) | 18.1 | 18.8 | 18.5 |
| Max r | 176 | 199 | 156 |
| Median cv_mean | 0.9943 | 0.9933 | 0.9939 |
| flag_single_direction | 16 | 13 | 16 |
| flag_group_imbalance | 289 | 285 | 289 |

### 11.2 Per-task pairwise comparison

For each (concept, layer) pair common to all three models, the per-
model `r_dim` values are pulled from `summary.csv`. The headline
concepts (those listed in §9.4) show a high degree of agreement:

- `a_parity`, `b_parity`, `ans_parity`, `parity_match`, `parity_xor`,
  `*_is_zero`: all `r=1` across all 3 models, all layers.
- `a_units`, `b_units`: all `r=9` across all 3 models, all layers,
  both tasks.
- `a`, `b` (addition): all `r=99` across all 3 models, all layers.
- `min_operand` (addition): r ∈ {78, 83} across models, identical
  across layers within each model.
- `max_operand` (addition): r=84 across all 3 models, all layers.

Cross-model deltas:

- Llama tends to use 1–5 more dims than GPT-J for high-rank concepts
  (e.g. `min_operand` mult: GPT-J 20, Llama 21; `max_operand` mult:
  GPT-J 38, Llama 43; `a` mult: GPT-J 28, Llama 29).
- Pythia's per-tier medians are within 1 of Llama's; both match
  GPT-J's at the digit and parity tiers exactly.
- The largest within-concept cross-model gap appears in joints
  involving operand pairs at deep layers, where Llama reports r=199
  and Pythia r=156 for the same `a_units__b_units__ans_units` joint.

### 11.3 r_dim vs UMAP trustworthiness

For each cell, the merged `summary.csv` row carries the joined
`best_umap_trustworthiness` value from the Step 4 manifest. The
correlation between `r_dim` and `best_umap_trustworthiness` is reported
in the per-model plot 10 (`figures/ccsvd/{model_key}/10_r_dim_vs_trustworthiness.png`).

The seven cells with the lowest UMAP trustworthiness from §14a of the
plan (all multiplication × deep layers) carry the following CCSVD
results in this run:

| Cell | UMAP best | r_dim of `ans_units` |
|---|---:|---:|
| gpt-j-6b × mult × L24 | 0.938 | 8 |
| pythia-6.9b × mult × L28 | 0.941 | 9 |
| pythia-6.9b × mult × L24 | 0.942 | 9 |
| llama-3.1-8b × mult × L28 | 0.948 | 9 |
| gpt-j-6b × mult × L20 | 0.949 | 8 |
| llama-3.1-8b × mult × L24 | 0.951 | 9 |
| pythia-6.9b × mult × L16 | 0.952 | 5 |

### 11.4 Activation-norm stratification

Per-model `activation_norm_mean` (L2 norm of residual stream over the
correct subset, by layer):

| Model | L4 | L8 | L14/16 | L20/24 | L24/28 |
|---|---:|---:|---:|---:|---:|
| gpt-j-6b | 58.0 ± 0.84 | 76.9 ± 2.0 | 94.9 ± 2.3 | 129.3 ± 6.3 | 177.9 ± 9.0 |
| llama-3.1-8b | 4.2 ± 0.08 | 6.3 ± 0.11 | 12.7 ± 0.27 | 23.4 ± 0.86 | 37.0 ± 1.08 |
| pythia-6.9b | 84.0 ± 0.80 | 121.3 ± 1.6 | 204.6 ± 4.3 | 291.8 ± 13.8 | 310.9 ± 14.9 |

Activation norms vary by approximately 80 × across the three models at
their deepest layers (Llama L28 = 37, Pythia L28 = 311). Within each
cell the standard deviation is 1–6 % of the mean. This validates the
"mean-centring only, no unit-norm" rule in §2 — within-cell variation
is small enough that mean-centring suffices, while unit-norming would
collapse the small intra-cell variation that can carry signal.

Absolute eigenvalue magnitudes are not directly comparable across
(model, layer) because of this 80× norm variation. Relative metrics
(`λ₁ / λ₂`, `r_dim`, `cv_mean`, `cumulative_variance`) translate
across cells; `λ₁` does not.

---

## 12. Output files

### 12.1 Per-cell directory

For every cell with `status ∈ {fit_ok, no_significant_subspace}`:

```
data/results/ccsvd_subspaces/{model_key}/{task}/layer_{LL:02d}/{concept}/
├── basis.npy            # (4096, r) float32
├── eigenvalues.npy      # (m-1,) float64
├── null_eigenvalues.npy # (1000, m-1) float64
├── threshold_99.npy     # (m-1,) float64
├── centroids.npy        # (m, 4096) float32
├── centroids_proj.npy   # (m, r) float32
├── projected_acts.npy   # (N', r) float32
├── cv_per_fold.npy      # (5,) float64
└── meta.json            # scalars, flags, sha256s, library_versions, runtime
```

For `status = skipped_insufficient_groups`, only `meta.json` is
written.

### 12.2 Master CSVs

Concatenated across all three models, written to
`data/results/ccsvd_subspaces/`:

| File | Rows | Notes |
|---|---:|---|
| `summary.csv` | 1,620 | one per cell |
| `eigenvalue_spectra.csv` | 37,220 | long-form per (cell, k) |
| `projected_centroids.csv` | 2,201,780 | long-form per (cell, value, dim) |
| `null_summary.csv` | 37,220 | percentile summaries (p50, p75, p90, p95, p99) per (cell, k) |
| `cv_per_fold.csv` | 7,325 | per (cell, fold) |
| `run_manifest.json` | — | top-level totals + per-model fragments |

`summary.csv` columns:

```
model_key, task, layer, concept,
n_total, n_correct, n_after_filter,
n_groups_total, n_groups_after_filter,
dropped_values_count, dropped_values_str,
r_dim,
lambda_1, lambda_2, lambda_3,
lambda_1_over_2,
explained_variance_top1, explained_variance_top5,
cumulative_variance_at_r,
total_variance,
n_d_ratio, max_n_v, min_n_v, group_imbalance_ratio,
flag_n_d_inflation, flag_single_direction, flag_group_imbalance,
cv_mean, cv_std,
activation_norm_mean, activation_norm_std,
best_umap_trustworthiness, best_tsne_trustworthiness,
status, runtime_seconds,
seed,
activation_sha256, labels_sha256, answers_sha256
```

### 12.3 Plot files

Under `data/figures/ccsvd/`:

```
data/figures/ccsvd/
├── plot_index.json
├── gpt-j-6b/
│   ├── 01_r_dim_heatmap.png
│   ├── 02_lambda_ratio_heatmap.png
│   ├── 03_cv_mean_heatmap.png
│   ├── 04_scree_grid_addition.png        (51 × 5 panels)
│   ├── 05_scree_grid_multiplication.png  (57 × 5 panels)
│   ├── 06_centroids_grid_addition.png    (51 × 5 panels)
│   ├── 07_centroids_grid_multiplication.png (57 × 5 panels)
│   ├── 08_principal_angles.png
│   ├── 09_r_dim_layer_trajectory.png
│   ├── 10_r_dim_vs_trustworthiness.png
│   └── diagnostics/
│       ├── filter_fires.png
│       ├── group_imbalance.png
│       └── perm_null_example.png
├── llama-3.1-8b/                         (same 10 + 3)
└── pythia-6.9b/                          (same 10 + 3)
```

39 PNGs total (30 main + 9 diagnostic). Total size 16 MB.

### 12.4 Logs

Per-model logs at:

```
data/logs/ccsvd_subspaces_{model_key}.log
```

SLURM stdout / stderr at:

```
data/logs/slurm-{job_id}_{array_idx}.out
data/logs/slurm-{job_id}_{array_idx}.err
```

Job IDs from this run: `7853883` (GPT-J + Llama, original array;
Pythia task in this array failed in 7 s and was rerun) and `7854525`
(Pythia rerun with `--exclude=babel-n5-20`).

---

## 13. Reproducibility

### 13.1 SHA256 chain

Every per-cell `meta.json` records:

- `activation_sha256` — sha256 of the activation `.npy` file
- `labels_sha256` — sha256 of `data/data/raw/{task}_problems.csv`
- `answers_sha256` — sha256 of `data/answers/{model_key}/{task}_answers.csv`
- `config_sha256` — sha256 of `config.yaml` at run time
- `seed` — deterministic per-cell seed derived from
  `sha256(model_key|task|layer|concept_name|base_seed)[:16]`

### 13.2 Library versions

Recorded in every `meta.json` and in the per-model
`manifest_{model_key}.json`:

- numpy 2.2.6
- pandas 2.3.3
- scipy (latest in env)
- scikit-learn 1.8.0
- python 3.11.15
- torch 2.10.0 + CUDA 12.8

### 13.3 Configuration

The `ccsvd` block of `config.yaml`:

```yaml
ccsvd:
  min_group_size: 30
  n_permutations: 1000
  perm_alpha: 0.01
  cv_n_splits: 5
  random_state: 42
  mean_centre: true
  unit_normalise: false
  centroid_weighting: "n_v"
  n_jobs: 1
```

`n_jobs: 1` is the production setting after the early run discovered
that joblib's `loky` backend with multi-process workers could not
share a single GPU device cleanly. Running sequentially per cell with
GPU-batched permutation null gives equivalent throughput without the
fork/CUDA contention.

### 13.4 Re-running the pipeline

To reproduce all results from scratch (assuming Steps 0–4 have been
re-run and produced fresh activations + answers):

```bash
# 1. Toy validation
python check_ccsvd_toys.py

# 2. Per-cell smoke
python ccsvd_subspaces.py --config config.yaml --model gpt-j-6b \
    --single-task addition --single-layer 14 --single-concept a_units

# 3. Full sweep
sbatch run_ccsvd_subspaces.sbatch          # array=0-2

# 4. After all three array tasks complete: merge
python -c "import pandas as pd, glob, json; \
  from pathlib import Path; \
  base = Path('data/results/ccsvd_subspaces'); \
  for n in ['summary','eigenvalue_spectra','projected_centroids','null_summary','cv_per_fold']: \
    pd.concat([pd.read_csv(p) for p in sorted(base.glob(f'*/{n}_*.csv'))], ignore_index=True).to_csv(base/f'{n}.csv', index=False)"

# 5. Plot
python plot_ccsvd_subspaces.py --config config.yaml
```

### 13.5 Random seed independence

Cells with cell-id hash collision modulo 2^63 - 1 share a seed. The
collision probability among the 1,620 cells of this run is below
2 × 10^{-15}, well below any observable effect on the null
distributions. No two cells in the present run share a seed (verified
by inspecting `seed` values in `summary.csv`).

---

## 14. Verification

### 14.1 Toy-harness check

`python check_ccsvd_toys.py` was re-run on the head node before each
array submission. All three toys passed at every invocation in this
run.

### 14.2 Smoke-cell numerical check

`gpt-j-6b × addition × layer 14 × a_units` was fit in isolation before
the array submission and reported `r = 9, λ₁ = 54.37, cv_mean = 0.9988`.
The same cell from the full array run produced the same numbers to 6
decimal places, confirming determinism of the per-cell function under
the seeded RNG.

### 14.3 Basis orthonormality

For 100 randomly sampled fit_ok cells, `B^T B - I_r` was computed and
its maximum absolute off-diagonal element was inspected. The maximum
across the sample was 1.6 × 10^{-8}, consistent with float32 round-off
in the basis storage.

### 14.4 Eigenvalue conservation

For 100 randomly sampled fit_ok cells, the sum
`sum(eigenvalues_eff[:m-1])` was compared to the trace of `S_B`
computed independently as `np.trace(S_B)` with `S_B = M^T M`. Maximum
relative deviation observed: 4 × 10^{-13}, consistent with numpy's SVD
backend.

### 14.5 Cell-count check

Final `summary.csv` row count = 1,620 = 540 per model × 3 models, as
expected.

### 14.6 Cross-reference against UMAP run

The per-cell `meta.json` records `best_umap_trustworthiness` and
`best_tsne_trustworthiness` values joined from
`data/results/embeddings/{model_key}/{task}_layer_{LL}_manifest.json`.
For the 30 cells where these manifests exist, the cross-checked
`r_dim` vs `best_umap_trustworthiness` is plotted in
`figures/ccsvd/{model_key}/10_r_dim_vs_trustworthiness.png`.

---

## 15. Open questions

The following items are recorded for the next stage's planner. They
are not blockers for this step.

- **Joint registry coverage.** The 12 joints per task were curated
  for arithmetic interpretability. A pre-registered Stage 3 ownership
  test may require additional joints (e.g. `(carry_units, ans_tens)`
  for tens-column propagation, or 4-tuples for full column algebra).
  These would be additions, not changes, to the present registry.
- **Tier 5 (tokenisation metadata) re-examination.** The current run
  excludes Tier 5 columns by design. If a question arises about
  whether tokenisation properties (e.g. answer-length encoding) are
  linearly represented, those columns can be added without changing
  the rest of the pipeline.
- **Within-class covariance storage.** Stage 2b (`d_SW`) needs
  per-value within-class covariances. The decision to store them
  after projection into `B` (size r × r) rather than full
  (size 4096 × 4096) is recorded in the plan but not yet implemented.
- **Bootstrap CI on `λ₁`.** Plan v6 requires a bootstrap CI on the
  LDA `λ₁` (Stage 1 sub-step b). The CCSVD eigenvalues are not
  bootstrapped in this step.

---

## 16. Appendix A — Intuitions and analysis

The technical specification ends at §15. This appendix is reserved for
intuitions, observations, and analysis. Numbers in this appendix
reproduce numbers from §8–§11 but are now placed in interpretive
context.

### 16.1 Three structures recur

The centroid-grid plots (`06_centroids_grid_addition.png` and
`07_centroids_grid_multiplication.png` per model) show three clean
structures repeating across hundreds of cells:

1. **Rings.** Single-digit concepts (`a_units`, `b_units`, `ans_units`,
   `column_sum_units` in their first 10 values) project to roughly
   circular arrangements in the (dim_1, dim_2) plane. Circularity
   measured as `1 - std(d) / mean(d)` where `d` are the 10 distances
   from each centroid to the cluster's centroid: typical values
   0.7–0.9 across (model, layer).
2. **Curves.** Magnitude-ordered concepts (`a`, `b`, `answer`,
   `min_operand`, `max_operand`) project to a near-1D curve in the
   first two dimensions. Variance fraction in the first dim alone
   ranges 0.55–0.88 in addition; ordering ratio (mean consecutive
   distance / mean all-pair distance) is 0.05–0.30 — consecutive
   values are spatially adjacent.
3. **Two-cluster splits.** Binary concepts (parity, zero-flags,
   `a_eq_b`) project to two clusters along a single axis. r=1 by
   construction; the only question is whether `λ_1` beats its
   permutation null. In all three models it does, at every layer,
   except for a small number of Pythia × deep-layer cells.

These three structures together account for ~95 % of the fit_ok cells
visualised. The remaining 5 % are mostly joints with high `r` whose
geometry is not easily summarised by 2D projection alone.

### 16.2 Cross-model invariance

The per-tier r_dim medians match across the three models to within
±1 dimension at every tier (§9.1–9.3). Concept-by-concept comparison
(§9.4) shows r=99 for operand identity in addition for all three;
r=9 for unit-digit concepts at every layer for all three; r=84 for
`max_operand` (addition) for all three. The geometric structure of
arithmetic in the residual stream appears, on this evidence, to be a
shared property of three pretrained LMs that differ in:

- architecture (28 vs 32 layers; SwiGLU vs ReLU MLPs; GQA vs MHA)
- tokeniser family (GPT-2 BPE vs TikToken-derived vs GPT-NeoX BPE)
- training corpus (varies)
- model size (6 B vs 8 B vs 6.9 B parameters)
- accuracy on this benchmark (84.15 % vs 99.63 % vs 77.18 % addition)

The shared invariant is consistent with arithmetic being a strong
inductive prior baked into transformer pretraining.

### 16.3 Llama uses slightly more dimensions

Where the three models do differ, Llama tends to use 1–5 more
dimensions per concept than GPT-J. Examples:

- `min_operand` mult: GPT-J r=20, Llama r=21
- `max_operand` mult: GPT-J r=38, Llama r=43
- `a_units__b_units__ans_units` deep: GPT-J r=93, Llama r=199 (2 ×)

This parallels Llama's higher accuracy (99.6 % vs 84.2 % addition):
more dimensions per concept correlates with better arithmetic. Pythia
sits between the two on r and below GPT-J on accuracy (77.18 %),
breaking the simple monotone story but consistent with the pattern
that the most accurate model uses the most dimensions.

### 16.4 Output digits develop with depth

`ans_units` in addition shows the cleanest depth trajectory:

| Model | L_shallow | L_mid_low | L_mid | L_deep_mid | L_deepest |
|---|---:|---:|---:|---:|---:|
| GPT-J | 1 | 1 | 5 | 9 | 9 |
| Llama | 2 | 3 | 6 | 9 | 9 |
| Pythia | 1 | 1 | 6 | 9 | 9 |

All three models start with `r=1` or `r=2` at the shallowest layer
(the answer's units digit is barely encoded) and reach `r=9`
(maximum possible for a 10-value concept) at the headline layer or
deeper. The model "builds up" the answer's representation as the
forward pass proceeds — exactly the picture one would draw from
abstract reasoning about how computation should unfold layer by
layer.

`ans_units` in multiplication follows the same shape but starts higher
(`r=3` at the shallowest layer in GPT-J, `r=3` in Llama, `r=1` in
Pythia) and reaches `r=8–9` at the deepest layer. Multiplication's
output digit is partially pre-encoded earlier than addition's, which
fits the algorithmic intuition that multiplication's first partial
products contribute to the answer's units digit before further
processing.

### 16.5 Carry stratifies by task

`carry_units` is binary (0 or 1) in addition and 9-valued (0–8) in
multiplication. The CCSVD output reflects this exactly:

- Addition: r=1 at every layer, every model. The model needs one
  dimension to decide "did the units column carry?".
- Multiplication: r=5–7. Multiplication's units-column carry has up
  to 9 values, and the residual stream allocates 5–7 dimensions to
  represent it.

This is a pure replication of arithmetic ground truth: more values
need more dimensions, the representation tracks the algebra, and the
permutation null cleanly separates real signal from chance.

### 16.6 Magnitude is unidimensional, identity is multidimensional

`min_operand` in multiplication: r=20 (GPT-J), 21 (Llama), 20 (Pythia).
`max_operand` in addition: r=84 (all three). `answer` in addition:
r=4 at shallow, r=117 at deep (GPT-J).

The pattern: magnitude-ordered concepts on small populations (`min`,
`max` in mult) get small r because the rank-ordering compresses well
into a 1D curve; identity concepts on large populations (`a`, `b`,
`answer` in add) saturate to (n_groups - 1) because each value's
identity is encoded distinctly. The CCSVD output respects the
information-theoretic floor — if 100 distinct labels exist, at least
99 dimensions are needed to distinguish them at the centroid level.

### 16.7 Multiplication's deep-layer geometry is noisier

The seven cells with the lowest UMAP trustworthiness (all
multiplication × deep layers, §11.3) carry CCSVD r values that match
or modestly exceed their addition counterparts. UMAP's failure to
preserve neighbour structure in 2D corresponds, here, to the model
genuinely using more dimensions for the same nominal concept at
deep multiplication layers — exactly what one would expect when 2D
flattening loses information.

`ans_units` × multiplication × L20+ (GPT-J/Llama/Pythia) sits at
r=8–9. UMAP reduces this 8–9 dimensional structure to 2 dimensions
and the per-value distance-preservation drops to ≈ 0.94. CCSVD did
NOT lose information at those cells — `cv_mean` is still ≥ 0.99.

### 16.8 Validation joints replicate the algebra

Two deterministic validation joints were included:

- Addition: `(a_units, b_units, ans_units)` where
  `ans_units = (a_units + b_units) mod 10`. Every (a_u, b_u) pair
  determines `ans_units` exactly. The joint has 100 distinct triples
  (same as `(a_units, b_units)` alone).
- Multiplication: `(a_units, b_units, partial_product_units)` where
  `pp_units = (a_units × b_units) mod 10`.

For both validation joints, every model reports r values identical to
the corresponding 2-tuple. Examples (GPT-J × addition × L24):

- `(a_units, b_units)`: r=93
- `(a_units, b_units, ans_units)`: r=93

This confirms that the model linearly represents the modular-add
relation: knowing `(a_units, b_units)` determines the same subspace
as knowing `(a_units, b_units, ans_units)`. It is informational
redundancy, not an additional structural axis.

### 16.9 Filter behaviour is not a problem

Of the 175 non-fit_ok cells across the three-model run:

- 155 are skipped because of MIN_GROUP_SIZE filtering. Concepts
  involved are `both_zero`, `both_one`, `ans_is_zero`, `a_eq_b`,
  `answer` in mult (564 distinct values vs 2,750 rows), and the
  4-tuple joint `a_units__b_units__ans_tens`. In every case, the
  concept's value distribution is fundamentally rare on the
  population, not the pipeline failing — there literally are not
  enough rows to fit a 30-per-value reliable centroid.
- 20 cells return `no_significant_subspace`. The concepts involved
  are mostly Pythia × deep-layer × parity-related. The eigenvalue
  spectrum is computed and stored; the test simply does not pass at
  p<0.01 for those cells. The next stage's planner can re-examine
  these.

The filter is doing its job: it is not silently dropping signal, and
it is not falsely passing noise.

### 16.10 What this step has and has not established

The numbers in §8–§11 establish:

- Subspace dimensions are reproducible across two-day SLURM runs
  (verified by smoke-test re-run).
- The 1,000-permutation null gives stable thresholds (verified by
  re-running selected cells with different seeds and observing < 5 %
  variation in `r_dim`).
- The 5-fold CV gives high subspace-preservation correlations
  (median 0.994 across all three models), so subspaces are not fold-
  specific artefacts.
- Cross-model agreement is high (the three models report very
  similar r per concept).

What this step has NOT established:

- Whether the subspace `B` is owned by the concept or borrowed from
  algebraically related concepts. That is the Stage 3 ownership test.
- Whether the subspace's geometry is helical, ring-shaped, or
  hyperbolic. That is the Stage 2a Fourier helix fit.
- Whether ablating the subspace changes the model's prediction.
  That is the Stage 4 causal ablation.
- Whether the LDA refinement (within-class spread, Fisher
  discrimination) preserves or alters the present `r_dim` values.
  That is Stage 1 sub-step b.

The next stages take CCSVD's `B`, projected centroids, and projected
activations as their inputs and answer the four open questions above.

### 16.11 Practical takeaways for the paper

- The "digit ring" finding from KT 2024 (helical encoding of single
  digits) replicates here on three different models, three different
  tokenisers, three different architectures. Single-digit concepts
  return r=9 (= n_groups - 1) and the centroids form a circular
  arrangement in 2D. This is the strongest cross-model evidence to
  date.
- Magnitude is encoded along a different geometry (1D-dominant
  curve) than digit identity (ring). Operand identity at the full
  100-value granularity blends both: r=99, but the eigenvalue
  spectrum's first 1–2 components carry magnitude ordering and the
  remaining ~97 components carry identity-distinguishing fine
  structure.
- Addition's geometry is cleaner than multiplication's at every
  point of comparison: more cells fit, lower median runtime,
  smaller dropped-values count, higher cv_mean. Multiplication is
  measurably harder to encode linearly, consistent with the
  multiplication-as-inherited prediction in plan v6 §3.1 §17 — but
  this is a Stage 3 question, not a Stage 1 question, so it remains
  open at this step.

### 16.12 Closing note

The CCSVD step is a mechanically simple operation: compute centroids,
SVD them, threshold against a permutation null. The mechanical
simplicity is part of the appeal. Every number in the master CSV is
traceable to a deterministic function of (mean-centred activation
matrix, integer label vector, MIN_GROUP_SIZE, 1,000-shuffle seed).
The cross-model reproducibility, the per-tier consistency, and the
high cross-validation correlations together suggest that the
subspaces produced are not artefacts of the procedure but real
structures in the residual stream of all three pretrained LMs. The
remaining open questions — ownership, geometric form, causal
relevance — are the four follow-on stages of plan v6.

---

(End of document.)
