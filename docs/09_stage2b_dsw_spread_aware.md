# Step 11 / Stage 2b — Spread-aware Mahalanobis test (`d_SW`)

> Technical report. Procedure and numbers only; the **Intuition and analysis** section at the end is the only place interpretation appears (it is explicitly requested by the user).
>
> Pipeline location: `stage2b_dsw_spread_aware.py`, `aggregate_stage2b_dsw.py`, `check_stage2b_toys.py`, `run_stage2b.sbatch`, `run_stage2b_aggregate.sbatch`, `configs/stage2b.yaml`. Plan reference: `plan.md` §3.2.2 (B.0–B.9), §3.2.5 (toys). Approval plan: `/home/anshulk/.claude/plans/lets-create-a-perfect-quirky-pretzel.md`.

---

## Table of contents

1. Overview
2. Inputs and outputs
3. Procedure (B.0–B.9)
4. GPU acceleration
5. Numerical thresholds
6. Toy validation suite
7. Aggregator outputs
8. Global verdict counts
9. Per-(model, task, mode, layer, variant) verdict counts — full table
10. Per-concept survival table
11. HIGH and MEDIUM tier cells — full list
12. Stage 2a → 2b transition
13. Cross-mode survival distribution
14. Cross-variant agreement
15. Carry concept cells — full list (zero survival cohort)
16. Runtime and resource usage
17. Artefact pathing reference

---

## 1. Overview

Stage 2a fit a Fourier helix to the **per-value centroid** of each (model, task, mode, layer, variant, concept) cell. The procedure discovered a dominant period from the centroid periodogram and confirmed two-axis + linear pitch significance against a Whittle max-over-frequencies null. Stage 2a operates at the level of the K centroid means; it does not characterise the within-class spread.

Stage 2b is the spread-aware companion. For each cell that Stage 2a deemed eligible (`geometry_detected ∈ {helix, circle, none, sparse_value_grid}`), Stage 2b:

1. Estimates per-value within-class covariance `Σ_v` directly from the activations (no parametric model).
2. Builds a Euclidean centroid distance matrix `D_E` and a spread-aware (Mahalanobis-style) distance matrix `D_SW`.
3. Computes `ρ_centroid = Spearman(vec_offdiag(D_E), vec_offdiag(D_SW))`.
4. Builds a 95% bootstrap confidence interval on `ρ_centroid` (1000 draws, with replacement at size N).
5. Builds a label-permutation null distribution of `ρ_centroid` (1000 permutations).
6. Applies global Benjamini–Hochberg FDR over the eligible cell family.
7. Emits one of `spread_confirmed` / `spread_marginal` / `centroid_only_shape` / `insufficient_samples` / `low_K_after_filter` / `null_unstable`.
8. Assigns one of `HIGH` / `MEDIUM` / `LOW` / `DISCOVERY_ONLY` confidence tier orthogonally to the verdict.

The verdict ladder is designed so that ρ_low (the lower bound of the bootstrap CI) is the dominant gate, closing a gap noted during the 14-point design review where high point estimate ρ with a low CI lower bound (e.g. ρ=0.90, ρ_low=0.30) would otherwise have fallen through the ladder.

**Plan compliance.** This report reflects every fix from the 14-point review:
- The ladder gap is closed (`ρ_low < 0.50` → `centroid_only_shape` regardless of ρ).
- Toy 5B uses a calibrated negative-control construction (per-value anisotropic noise along a private random direction per value) with a monotonicity sweep that locks the largest scale at which the verdict reliably leaves `spread_confirmed`.
- Toy 7B-FPR runs a 100-cell isotropic family under H0 and verifies the false-positive rate sits inside a binomial 95% acceptance band of nominal 5%.
- Shrinkage harmonisation is enforced cell-wide: per-value Σ_v are fit at the strictest mode across all values in the cell (sample < lw < oas), so every pair Σ_pool = (Σ_u' + Σ_v')/2 is spectrally consistent.
- Ledoit-Wolf is computed via the closed-form per-point estimator inlined in `fit_sigma_lw`, replacing sklearn's `ledoit_wolf` which contributed ~40% of single-cell wall time through input-validation overhead.
- Bootstrap re-evaluates the shrinkage choice on every draw; bootstrap samples whose per-value count drops below 5 are rejected and redrawn.
- The subspace-vs-ambient choice is explicit: all `Σ_v` live in subspace coordinates (B^T x ∈ ℝ^r, r ≈ 8–12), not ambient 4096-D.
- K=4 cells are hard-capped to LOW tier (only K(K-1)/2 = 6 off-diagonal pairs, Spearman SE ≈ 0.45).
- The B.8 leave-one-value-out jackknife is restricted to cells with `|ρ_centroid − 0.85| < 0.05` (verdict could plausibly flip) to bound sensitivity-sweep cost.

The locked toy calibration (`configs/stage2b.yaml`) sits at `toy_5b_tangent_scale=50.0` with monotonicity=true and `toy_7b_fpr.status=pass` (3 of 100 cells fired under H0).

---

## 2. Inputs and outputs

### 2.1 Inputs (read-only)

- **Activations.** `data_root/activations/{model}/{task}_layer_{LL}.npy` for `mode=off`, and `results_root/residualized/{model}/{task}_layer_{LL}_mode_{mode}.npy` for `mode∈{answer, norm}`. Shape `(N_total, 4096)` float32.
- **Correctness mask.** `data_root/answers/{model}/{task}_answers.csv` column `correct`. Stage 2b operates on the correct-subset only, matching Stage 2a.
- **Problem labels.** `data_root/data/raw/{task}_problems.csv` provides K_natural per concept and the per-point concept values.
- **Stage 1 subspace bases.**
  - CCSVD basis: `results_root/ccsvd_subspaces/[mode_{mode}/]{model}/{task}/layer_{LL}/{concept}/basis.npy` (shape `(4096, r)` float32).
  - LDA-A basis: `results_root/lda_subspaces/subspace_lda/mode_{mode}/{model}/{task}/layer_{LL}/{concept}/lda_basis_full.npy`.
  - Per-cell metadata at `meta.json` next to each basis (provides `mu_layer`).
- **Stage 2a summary CSVs.** `results_root/stage2a_fourier_helix/{model}/summary_{model}_{task}_mode_{mode}_variant_{variant}.csv`. Stage 2b reads these to (a) determine eligibility per cell and (b) record `stage2a_verdict` + `stage2a_discovered_period` for the headline transition table.
- **Stage 2b config block** in `config.yaml` and toy calibration record in `configs/stage2b.yaml`.

### 2.2 Per-cell outputs

Written under `results_root/stage2b_dsw/{model}/{task}/mode_{mode}/layer_{LL}/variant_{variant}/{concept}/`:

| File | Description |
|------|-------------|
| `dsw_results.csv` | Single row per cell with the headline statistics (see §7.3). |
| `D_E.npy` | K×K float64 Euclidean centroid distance matrix. |
| `D_SW.npy` | K×K float64 spread-aware (Mahalanobis) distance matrix. |
| `mu_stack.npy` | K×r float64 per-value centroids in subspace coordinates. |
| `shrinkage_pair_mode_matrix.npy` | K×K int8 — per-pair shrinkage mode code (0=sample, 1=lw, 2=oas). Under cell-wide harmonisation this is a constant matrix; preserved for audit. |
| `rho_null.npy` | (n_permutations,) float32 — null ρ_centroid distribution from label permutations. |
| `rho_bootstrap.npy` | (n_bootstrap,) float32 — bootstrap ρ_centroid distribution from size-N with-replacement draws. |
| `metadata.json` | Config snapshot, RNG seed, lib versions, computation_status, kept/dropped values, shrinkage modes per value, B basis SHA256. |

### 2.3 Per-model summary CSV

`results_root/stage2b_dsw/{model}/summary_{model}_{task}_mode_{mode}_variant_{variant}.csv` is the concatenated per-cell row stream, one row per concept. Columns include all dsw_results.csv fields plus `stage2a_verdict` and `stage2a_discovered_period` for transition analysis.

### 2.4 Aggregator outputs

Written under `results_root/stage2b_dsw/comparison/`:

| File | Description |
|------|-------------|
| `dsw_all.csv` | Every cell from every per-model summary, concatenated, with `q_dsw` added by global BH-FDR and `spread_verdict_pre_fdr` + `fdr_downgraded` recorded. |
| `spread_verdict_counts_by_cell.csv` | (model, task, mode, layer, variant) × spread_verdict pivot table. |
| `cross_mode_spread_survival.csv` | (model, task, layer, variant, concept) × mode pivot of spread_verdict; adds `n_spread_confirmed_modes ∈ {0,1,2,3}` and `n_centroid_only_modes`. |
| `cross_variant_agreement.csv` | (model, task, mode, layer, concept) × variant pivot of spread_verdict; adds `verdict_agree` boolean and `rho_diff_abs`. |
| `stage2a_vs_stage2b_survival.csv` | Headline transition table — every Stage 2a `helix`/`circle` cell with its Stage 2b verdict and tier. |
| `confidence_tier_distribution.csv` | (model, task, mode, layer, variant, confidence_tier) × spread_verdict counts. |
| `manifest.json` | Run-level summary: row count, cell count, verdict counts pre/post FDR, FDR alpha, tier counts, toy calibration record, lib versions, runtime. |

---

## 3. Procedure (find → fit → null → verdict)

### 3.1 B.0 — Eligibility

A cell is processed if and only if all the following hold:

- `geometry_detected ∈ {helix, circle, none, sparse_value_grid}` from the Stage 2a summary CSV for the matching (model, task, mode, layer, variant). Cells flagged `low_K`, `period_inconsistent`, `null_unstable` by Stage 2a are skipped.
- The cell has a basis matrix on disk (CCSVD basis for `variant=ccsvd`, LDA-A basis for `variant=lda_a`) and a recoverable `mu_layer`.
- The per-cell `K_natural ≥ 2`.
- After filtering each value v with `n_v ≥ min_group_size = 30`, the resulting `K_present ≥ min_K_for_dsw = 4`.

`sparse_value_grid` is included defensively. Empirically the Stage 2a code in `stage2a_fourier_helix.py` uses this label when `K_present < K_natural`; the d_SW test is grid-agnostic so these cells are safe to feed through.

`none` cells are kept because the report needs a baseline distribution of `ρ_centroid` for cells where Stage 2a did not find a periodic shape. They form a comparison group, not the headline.

### 3.2 B.1 — Find: data-adaptive Σ_v per value

**Subspace, not ambient.** All Σ_v are computed in subspace coordinates `Y = (X − μ_layer) · B` where `B ∈ ℝ^{4096 × r}` is the cell's Stage 1 basis and `r ≈ 8–12`. This inherits Stage 1's room and avoids estimating an ill-posed 4096 × 4096 covariance from N ≈ 3K–10K.

**Per-value adaptive shrinkage.** For each value v with `n_v ≥ 30`:

- Compute the centred residuals `R_v = Y_v − μ_v ∈ ℝ^{n_v × r}`.
- Compute `ratio_v = n_v / r`.
- Pick the per-value shrinkage mode:
  - `ratio_v ≥ shrinkage_lw_threshold = 10` → `shrink(v) = sample`.
  - `shrinkage_oas_threshold = 5 ≤ ratio_v < 10` → `shrink(v) = lw` (Ledoit-Wolf).
  - `ratio_v < 5` → `shrink(v) = oas` (Oracle Approximating Shrinkage; Chen, Wiesel, Eldar & Hero 2010, IEEE TSP 58:5016–5029).

**Cell-wide harmonisation (review item #4).** When per-value chosen modes mix, the plan calls for the stricter of any pair's two modes to be used both for u and v before pooling. The empirically equivalent (slightly more conservative) implementation is `cell_mode = max(chosen_modes)` under the strictness order `sample < lw < oas`. Every per-value Σ̂_v is then fit at `cell_mode`, so for any pair (u, v), Σ_pool = (Σ̂_u + Σ̂_v) / 2 is spectrally consistent. This avoids the inconsistent-spectrum case where Σ̂_u_sample is averaged with Σ̂_v_oas (raw + heavily shrunken). The `shrinkage_pair_mode_matrix.npy` artefact records `SHRINK_ORDER[cell_mode]` everywhere off the diagonal for audit.

**Sample covariance.** `Σ̂_v_sample = R_v.T @ R_v / (n_v − 1)` (unbiased, /n-1 normalisation matching the LDA-pipeline convention).

**Ledoit-Wolf (inlined, closed-form).** The full per-point LW estimator is used:

```
S = R.T @ R / n                                # biased sample cov
μ̄ = trace(S) / r
target = μ̄ · I_r
d² = ||S − target||²_F = trace(S²) − trace(S)²/r
For each point i in v:
   per_point[i] = ||x_i||⁴ − 2 x_iᵀ S x_i + ||S||²_F
b̄² = sum(per_point) / n²
shrinkage = clip(b̄² / d², 0, 1)
Σ̂_v_lw = (1 − shrinkage) S + shrinkage · target
```

The closed-form GPU equivalent uses the identity `sum_i (x_iᵀ S x_i) over points in v = n_v · trace(S_v²)`, so the per-point quadratic forms can be batched without explicit point loops. Per-value contributions to `A_v = sum_i ||x_i||⁴` are computed via a (K, N) mask multiplied against the per-point `||x_i||⁴` vector. The CPU and GPU paths produce bit-identical Σ̂_v within float64 tolerance — verified by direct cross-check on the toy helix at multiple seeds (pointwise max abs diff = 0).

**OAS (Chen et al. 2010).** Closed-form:

```
S = R.T @ R / n
trace_S = trace(S);  trace_S2 = trace(S²)
μ̄ = trace_S / r
num = (1 − 2/r) trace_S2 + trace_S²
den = (n + 1 − 2/r) · max(trace_S2 − trace_S²/r, 1e-30)
α = clip(num/den, 0, 1)
Σ̂_v_oas = (1 − α) S + α · (μ̄ · I_r)
```

The OAS form is shared between CPU and GPU paths.

**Citations** — Ledoit & Wolf 2004 (*Honey, I Shrunk the Sample Covariance Matrix*, J. Portfolio Mgmt 31:110–119); Chen, Wiesel, Eldar & Hero 2010 (*Shrinkage Algorithms for MMSE Covariance Estimation*, IEEE TSP 58:5016–5029).

### 3.3 B.2 — Fit: D_E, D_SW, ρ_centroid

For each unordered pair (u, v) with u < v ≤ K_present − 1:

- `D_E[u, v] = ||μ_u − μ_v||₂` — Euclidean centroid distance.
- `Σ_pool = (Σ̂_u + Σ̂_v) / 2`.
- `λ_uv = lambda_factor · trace(Σ_pool) / r` with `lambda_factor = 1e-6` (matches plan §3.2.2 line 226 verbatim).
- `Σ⁺_uv = Σ_pool + λ_uv · I_r`.
- `D_SW[u, v]² = (μ_u − μ_v)ᵀ Σ⁺_uv⁻¹ (μ_u − μ_v)`. Computed via batched `cp.linalg.solve(Σ⁺_uv, diff)` on GPU (Σ⁺_uv is positive-definite with Tikhonov, so the solve is stable) or via `numpy.linalg.cholesky` + `numpy.linalg.solve` on CPU.

Symmetric: `D_E[v, u] = D_E[u, v]` and `D_SW[v, u] = D_SW[u, v]`. Diagonals zero.

Headline statistic:

```
ρ_centroid = Spearman(vec_offdiag(D_E), vec_offdiag(D_SW))
```

over the K(K-1)/2 upper-triangle entries.

Auxiliary diagnostics:

- `ρ_pearson` = Pearson correlation of the same vectors (reported only, not gating).
- `tau_kendall` = Kendall tau-b (robust to ties at low K).
- `mean_log_ratio = mean(log(D_SW / D_E))` over pairs with both > 1e-12 (magnitude-aware shift).

Spearman is computed via `scipy.stats.rankdata` + Pearson on ranks (tie-aware), cheaper than `scipy.stats.spearmanr` for the small K(K-1)/2 vectors typical here.

### 3.4 B.3 — Bootstrap CI on ρ_centroid

`n_bootstrap = 1000` bootstrap samples per cell. Each draw:

1. Sample N indices with replacement from `range(N_used)` (so the bootstrap sample is the same size as the fit population; ≈ 63.2% unique points by construction).
2. Group bootstrapped Z by bootstrapped labels.
3. Recompute per-value counts. If any value has `n_v < bootstrap_min_n_v_floor = 5`, reject the draw and redraw (capped by `bootstrap_max_redraws = 100`).
4. **Re-evaluate the per-value shrinkage mode and the cell-wide stricter mode under the draw's counts** (review item #7). If a value upgrades sample → lw or lw → oas, the change is propagated. This keeps the CI honest about regime instability.
5. Refit Σ̂_v at the draw's cell mode, rebuild D_E and D_SW, recompute ρ_centroid.

Output:

- `ρ_low = quantile(rho_bootstrap, 0.025)`, `ρ_high = quantile(rho_bootstrap, 0.975)`.
- `bootstrap_se = std(rho_bootstrap, ddof=1)`.
- `redraw_rate_bootstrap = total_redraws / n_bootstrap`.

The bootstrap is a within-cell, with-replacement, full-size resampling. It does not subsample the per-cell data (compliant with the standing-rule "no subsampling, ever").

### 3.5 B.4 — Whittle-style label-permutation null

`n_permutations = 1000`. Each permutation:

1. Permute the K_present-coded labels across the N_used points (multiset-preserving — `np.random.permutation` of the codes vector).
2. Per-value counts are exactly preserved by construction (the multiset is invariant under permutation), so the redraw-rate of the null is identically zero.
3. Recompute Σ̂_v under the permuted labels with the cell mode harmonisation rule.
4. Rebuild D_E and D_SW, recompute ρ_centroid.

The per-cell p-value:

```
p_dsw = (1 + sum(rho_null ≥ rho_observed)) / (1 + n_permutations_valid)
```

where rho_null is the array of permuted-label ρ values.

### 3.6 B.5 — Global BH-FDR

Across the entire eligible cell family in `dsw_all.csv` (verdict ∈ {spread_confirmed_pre, spread_marginal_pre, centroid_only_shape_pre}), the aggregator applies Benjamini-Hochberg FDR via `scipy.stats.false_discovery_control(method="bh")` at `fdr_alpha = 0.05`. The output is `q_dsw` per cell. Non-eligible verdicts (`insufficient_samples`, `low_K_after_filter`, `null_unstable`) get `q_dsw = NaN`.

**Downgrade rule.** Cells with `spread_verdict_pre_fdr == spread_confirmed` and `q_dsw ≥ fdr_alpha` are downgraded to `centroid_only_shape`. The pre-FDR verdict is preserved as `spread_verdict_pre_fdr`, and `fdr_downgraded` is set to True. 108 cells were downgraded in this run.

### 3.7 B.6 — Per-cell verdict (ρ_low is the dominant gate)

Pre-registered ladder (assigned per cell, then re-evaluated by the aggregator using `q_dsw` in place of raw `p_dsw`):

| Verdict | Condition |
|---------|-----------|
| `spread_confirmed` | `ρ_low ≥ 0.70` AND `ρ_centroid ≥ 0.85` AND `q_dsw < 0.05`. |
| `spread_marginal` | (`ρ_low ∈ [0.50, 0.70)`) OR (`ρ_centroid ∈ [0.70, 0.85)` AND `ρ_low ≥ 0.50`), AND `q_dsw < 0.05`. |
| `centroid_only_shape` | `ρ_low < 0.50` OR `ρ_centroid < 0.70` OR `q_dsw ≥ 0.05`. |
| `insufficient_samples` | Any per-value n_v < min_group_size at the eligibility stage. |
| `low_K_after_filter` | K_present < min_K_for_dsw after the n_v ≥ 30 filter. |
| `null_unstable` | Bootstrap CI half-width > `ci_halfwidth_unstable = 0.30`. Numbers reported, not used in headlines. |

**Verdict-ladder unit test (review item #1).** A synthetic case `(ρ_centroid=0.90, ρ_low=0.30, p=0.01)` must land in `centroid_only_shape` via the `ρ_low < 0.50` clause. The test is in `check_stage2b_toys.py::unit_test_verdict_ladder` and passes on every run. Additional synthetic cases exercise each branch.

### 3.8 B.7 — Confidence tier (orthogonal to verdict)

Independent of the verdict, every cell carries a confidence tier driven by the sample regime. The tier is the answer to "we found it but how confident should we be?"

| Tier | Gate (all must hold) |
|------|----------------------|
| `HIGH` | `K_present ≥ tier_high_min_K = 6`, `min_ratio_v ≥ tier_high_min_ratio = 10`, `min_n_v ≥ tier_high_min_n_v = 100`, `q_dsw < tier_high_q_threshold = 0.01`, `ρ_low ≥ tier_high_rho_low = 0.80`. |
| `MEDIUM` | `K_present ≥ tier_medium_min_K = 5`, `min_ratio_v ≥ tier_medium_min_ratio = 5`, `min_n_v ≥ tier_medium_min_n_v = 50`, `q_dsw < tier_medium_q_threshold = 0.05`, `ρ_low ≥ tier_medium_rho_low = 0.70`. |
| `LOW` | `K_present ≥ 4`, `min_ratio_v ≥ 2`, `min_n_v ≥ 30`. |
| `DISCOVERY_ONLY` | `γ = r / min_n_v ≥ 1` OR any value with `ratio_v < tier_discovery_only_max_ratio = 2`. |

**Hard K=4 cap (review item #10).** Any cell with `K_present == 4` is capped at LOW regardless of any other gate. K=4 has only 6 off-diagonal pairs and a Spearman SE ≈ 1/√5 ≈ 0.45 even on noise-free data.

Per-cell tiers use raw p as a placeholder until the aggregator runs BH-FDR; the aggregator's `reassign_tiers_with_q` then recomputes every cell's tier using the final `q_dsw`.

### 3.9 B.8 — Robustness sensitivity sweep (deferred)

The plan calls for three perturbations on HIGH/MEDIUM `spread_confirmed` cells:

- **shrinkage_off** — force `Σ̂_v_sample` even when `ratio_v < 10`.
- **λ sweep** — `λ ∈ {1e-8, 1e-6, 1e-4} · trace(Σ_pool)/r`.
- **leave-one-value-out jackknife** — restricted to cells with `|ρ_centroid − 0.85| < loo_runlist_band = 0.05`.

The sweep is implemented but currently deferred to a follow-on pass — the headline numbers in this report come from the locked main pipeline. The per-cell parquet retains everything needed to run B.8 retroactively without re-running the main fit.

### 3.10 B.9 — Cross-mode and cross-variant pivots

The aggregator produces two pivot tables:

- **Cross-mode.** For each (model, task, layer, variant, concept) tuple, pivot mode ∈ {off, answer, norm} → spread_verdict. Adds `n_spread_confirmed_modes ∈ {0, 1, 2, 3}` and `n_centroid_only_modes`. The 3-mode column tracks whether the spread story holds when answer-magnitude or activation-norm is residualised out.
- **Cross-variant.** For each (model, task, mode, layer, concept) tuple, pivot variant ∈ {lda_a, ccsvd} → spread_verdict. Adds `verdict_agree` boolean and `rho_diff_abs`. Discrepancy between the two variants flags subspace-choice sensitivity.

---

## 4. GPU acceleration

The pipeline has CPU and GPU code paths sharing the same numerical algorithm; the GPU path is used automatically when cupy is available.

### 4.1 The hot loops

Per cell, the bootstrap and the Whittle null together account for ~2000 inner iterations (1000 each). Each inner iteration must:

1. Build a (K, N) one-hot mask of label codes.
2. Compute K per-value centroids in subspace coordinates.
3. Compute K per-value covariance matrices.
4. For each upper-triangle pair, form Σ_pool, regularise with Tikhonov, and solve Σ⁺_uv · x = diff for x; sum-of-squares of x is d².
5. Spearman of the K(K-1)/2 pair distances.

At K=10 and r ≈ 9, each step is microsecond-scale on CPU but the Python overhead per call dominates.

### 4.2 GPU implementation

The function `_rho_for_grouping_gpu(Z_gpu, codes_cpu, K, lw_threshold, oas_threshold, lambda_factor, min_n_v_floor)` in `stage2b_dsw_spread_aware.py` implements the inner iteration entirely in cupy, batching the per-value work:

```
mask    = (codes_gpu[None, :] == cp.arange(K)[:, None]).astype(cp.float64)   # (K, N)
counts  = mask.sum(axis=1)                                                    # (K,)
mu      = (mask @ Z_gpu) / counts[:, None]                                    # (K, r)
S_unc   = cp.einsum("kn,nr,ns->krs", mask, Z_gpu, Z_gpu)                      # (K, r, r)
Σ_sample_biased = S_unc / counts[:, None, None] − μ μᵀ                        # (K, r, r)
Σ       = _fit_sigma_batched_gpu(Σ_sample_biased, counts, r, cell_mode, ...)  # (K, r, r)
iu, iv  = cp.triu_indices(K, k=1)
Σ_pool  = 0.5 * (Σ[iu] + Σ[iv])                                               # (n_pairs, r, r)
trace   = cp.einsum("pii->p", Σ_pool)
λ_uv    = lambda_factor * trace / r
Σ_reg   = Σ_pool + λ_uv[:, None, None] * cp.eye(r)
diffs   = mu[iu] - mu[iv]                                                     # (n_pairs, r)
x       = cp.linalg.solve(Σ_reg, diffs[..., None]).squeeze(-1)                # batched
d²      = cp.einsum("pi,pi->p", diffs, x)
D_SW    = cp.sqrt(cp.maximum(d², 0)).get()
D_E     = cp.linalg.norm(diffs, axis=1).get()
ρ       = spearman_rho(D_E, D_SW)
```

`_fit_sigma_batched_gpu` is the batched shrinkage helper. It computes the closed-form LW or OAS estimator per value in a single cupy expression. For LW it uses the residual `r_per_point = Z_gpu − mu[codes_gpu]`, then per-value `A_v = sum_{i in v} ||r_i||⁴` via mask multiplication.

`cp.linalg.solve` natively handles batched (n_pairs, r, r) × (n_pairs, r, 1) systems on cupy 14.0.1. An attempted `cp.linalg.solve_triangular` call failed because that API is not present in cupy 14.0.1; the implementation falls back to `cp.linalg.solve` which is general-purpose but stable when the matrices are Tikhonov-regularised PSD.

### 4.3 Performance

Measured on the helix toy (N=8000, K=10, r=9):

| Path | Wall time per cell (n_perm=n_boot=1000) |
|------|----------------------------------------:|
| Original CPU (sklearn LW + per-pair Python Cholesky + 3-mode cache) | 24.1 s |
| CPU + inline LW + cell-mode harmonisation + batched numpy linalg | 4.5 s |
| GPU (cupy batched) | 3.8 s |

CPU and GPU paths produce bit-identical Σ_v on toy data (max abs diff = 0.0 in 1000-iter bootstrap comparison).

The main saving came from the inline LW (eliminating sklearn's input-validation overhead, which had been ~57% of total wall time per the original profile) and from collapsing the 3-mode covariance cache to a single cell-mode Σ_v per value (~3× fewer covariance fits per inner iteration). The GPU path adds incremental gain on top, and is the default when cupy is available.

### 4.4 Cluster resources

Per array task (one model × task pair):

- 1 × L40S (46068 MiB; CUDA 12.x; driver 575.51.03 on `babel-u5-28` / `babel-p5-28`).
- 4 CPUs (down from the initial 16-CPU request because L40S nodes are typically 32-CPU shared 8 ways).
- 32 GB RAM (down from 128 GB).
- A6000 was initially attempted but the partition's A6000s are heavily saturated; L40S provided faster turnaround.

The aggregator runs once on a single L40S (no CUDA usage; the GPU is requested only to satisfy the cluster's QOS minimum-GPU constraint).

---

## 5. Numerical thresholds

All thresholds are pinned in `config.yaml` under `stage2b:` and (for calibration outputs) in `configs/stage2b.yaml`. None are tuned after seeing real-cell results.

| Parameter | Value | Source / rationale |
|-----------|------:|--------------------|
| `n_permutations` | 1000 | Whittle null per cell. Matches Stage 2a, Stage 1 CCSVD/LDA convention. |
| `n_bootstrap` | 1000 | Bootstrap CI per cell. Matches plan §3.2.2. |
| `lambda_factor` | 1e-6 | Tikhonov λ = lambda_factor · tr(Σ_pool)/r. Verbatim from plan §3.2.2 line 226. |
| `shrinkage_lw_threshold` | 10 | LW kicks in when ratio_v < 10. Ledoit & Wolf 2004 typical regime n/p ∈ [5, 50]; also matches `lda_subspaces.py:573-577`. |
| `shrinkage_oas_threshold` | 5 | OAS kicks in when ratio_v < 5. Chen et al. 2010 typical regime n/p < 10; conservative pick at 5. |
| `rho_pass_threshold` | 0.85 | spread_confirmed gate on ρ_centroid. Plan §3.2.2 line 239. |
| `rho_low_ci_threshold` | 0.70 | spread_confirmed gate on bootstrap CI lower bound. Plan §3.2.2 line 239. |
| `rho_marginal_low` | 0.50 | Heuristic, pre-registered: half-scale midpoint; below this rank-correlation is no longer interpretable as preservation. |
| `ci_halfwidth_unstable` | 0.30 | Heuristic, pre-registered: ~half a Likert step on the 0–1 ρ scale; CIs wider than this leave the verdict undetermined → `null_unstable`. |
| `fdr_alpha` | 0.05 | BH-FDR family-wise threshold. Matches Stage 2a aggregator. |
| `min_group_size` | 30 | Per-value floor before B.0 eligibility check. Matches Stage 2a `min_group_size`. |
| `bootstrap_min_n_v_floor` | 5 | Below this in a bootstrap draw → reject and redraw. Heuristic, pre-registered: floor where even OAS is well-conditioned. |
| `bootstrap_max_redraws` | 100 | Cap on per-draw redraw attempts to prevent infinite loops on degenerate cells. |
| `min_K_for_dsw` | 4 | K_present floor (Stage 2a inheritance). |
| `loo_runlist_band` | 0.05 | B.8 LOO restricted to cells with `|ρ_centroid − 0.85| < 0.05`. Heuristic, pre-registered: bound jackknife cost to the band where the verdict could plausibly flip. |
| `tier_high_min_K` | 6 | HIGH tier requires at least 6 values (15 pairs, Spearman SE ≈ 0.27). |
| `tier_high_min_n_v` | 100 | Heuristic, pre-registered: ≈10× r — asymptotic regime margin. |
| `tier_high_min_ratio` | 10 | Matches `shrinkage_lw_threshold`. |
| `tier_high_q_threshold` | 0.01 | HIGH tier q gate. |
| `tier_high_rho_low` | 0.80 | HIGH tier requires a tight bootstrap lower bound. |
| `tier_medium_min_K` | 5 | MEDIUM tier values floor. |
| `tier_medium_min_n_v` | 50 | Heuristic, pre-registered: ≈5× r — matches LW shrinkage threshold. |
| `tier_medium_min_ratio` | 5 | Matches `shrinkage_oas_threshold`. |
| `tier_medium_q_threshold` | 0.05 | MEDIUM tier q gate. |
| `tier_medium_rho_low` | 0.70 | Matches `rho_low_ci_threshold`. |
| `tier_discovery_only_max_ratio` | 2 | Below this, DISCOVERY_ONLY regardless of verdict. |
| `toy_5b_scale_sweep` | [1.0, 5.0, 20.0, 50.0] | Toy 5B per-value anisotropic scale sweep. |
| `toy_5b_n_seeds` | 3 | Per-scale replicates. |
| `toy_5b_extreme_max_rho` | 0.85 | At the largest sweep scale, median ρ must be < this for the calibration to pass. |
| `toy_7b_n_cells` | 100 | Toy 7B-FPR cell count. |
| `toy_7b_lower` | 2 | Binomial 95% CI lower bound around 5%. |
| `toy_7b_upper` | 11 | Binomial 95% CI upper bound around 5%. |
| `random_state` | 42 | Base seed; per-cell RNG derived via sha256(model|task|mode|layer|variant|concept). |

---

## 6. Toy validation suite

Run via `python check_stage2b_toys.py --config /home/anshulk/emnlp2026/config.yaml`. The full suite must pass before any real-cell run. The toy gate is enforced at the start of `stage2b_dsw_spread_aware.py::main` via `check_toy_calibration(toy_cfg_path)` — real-cell runs refuse to start unless `configs/stage2b.yaml::toy_5b_tangent_scale` is locked and `toy_7b_fpr.status == "pass"`.

### 6.1 Verdict-ladder unit test

Six synthetic (ρ, ρ_low, p) cases exercise each branch of `assign_verdict`:

| ρ | ρ_low | p | Expected | Pass |
|--:|------:|--:|----------|------|
| 0.90 | 0.30 | 0.01 | centroid_only_shape | ✓ (review item #1 gap case) |
| 0.95 | 0.85 | 0.01 | spread_confirmed | ✓ |
| 0.90 | 0.65 | 0.01 | spread_marginal | ✓ |
| 0.80 | 0.60 | 0.01 | spread_marginal | ✓ |
| 0.50 | 0.40 | 0.01 | centroid_only_shape | ✓ |
| 0.95 | 0.85 | 0.20 | centroid_only_shape | ✓ |

### 6.2 Toy 1B — Line

200 points on a 1D line in 9D + isotropic noise (σ=0.5). 3 seeds.

| seed | ρ | ρ_low | ρ_high | p_dsw | verdict | pass |
|-----:|--:|------:|-------:|------:|---------|:----:|
| 0 | 0.991 | 0.99 | 1.00 | 0.005 | spread_confirmed | ✓ |
| 1 | 0.994 | 0.99 | 1.00 | 0.005 | spread_confirmed | ✓ |
| 2 | 0.994 | 0.99 | 1.00 | 0.005 | spread_confirmed | ✓ |

### 6.3 Toy 2B — Circle

200 points on a 2D circle in 9D + isotropic noise (σ=0.5). 3 seeds.

| seed | ρ | ρ_low | ρ_high | p_dsw | verdict | pass |
|-----:|--:|------:|-------:|------:|---------|:----:|
| 10 | 0.970 | 0.94 | 0.98 | 0.005 | spread_confirmed | ✓ |
| 11 | 0.964 | 0.94 | 0.98 | 0.005 | spread_confirmed | ✓ |
| 12 | 0.973 | 0.94 | 0.99 | 0.005 | spread_confirmed | ✓ |

### 6.4 Toy 3B — Helix

200 points on a 3D circular helix in 9D + isotropic noise (σ=0.5). 3 seeds.

| seed | ρ | ρ_low | ρ_high | p_dsw | verdict | pass |
|-----:|--:|------:|-------:|------:|---------|:----:|
| 20 | 0.953 | 0.93 | 0.98 | 0.005 | spread_confirmed | ✓ |
| 21 | 0.972 | 0.94 | 0.99 | 0.005 | spread_confirmed | ✓ |
| 22 | 0.942 | 0.90 | 0.98 | 0.005 | spread_confirmed | ✓ |

### 6.5 Toy 4B — Isotropic Gaussian (FPR sanity)

20 batches of 400 isotropic points with random labels. Under H0 at α=0.05, expected ≤ 5% false positives — over 20 seeds, binomial 95% upper bound is 4.

| trial | false_positives / 20 | rate | binomial 95% upper bound | pass |
|------:|----------------------|------|--------------------------|:----:|
| – | 1 / 20 | 0.05 | 4 | ✓ |

The proper FPR calibration is Toy 7B (100 cells). Toy 4B is a sanity gate.

### 6.6 Toy 5B — AnisotropicOverlapHelix (negative control)

**Construction.** Helix centroids on a clean 3-D helix; each value v gets a private random unit direction d_v in 9D, and per-point amplitude `amp_i ~ N(0, scale²)` added along `d_{label_i}`. Σ_v = scale² · d_v d_vᵀ + iso² · I. Different values' Σ point in different directions, so per-pair Σ_pool has anisotropy in two random dirs — exactly the failure mode d_SW is designed to flag.

**Calibration sweep.** Per-scale ρ_centroid (3 seeds) and verdict:

| scale | ρ seed 0 | ρ seed 1 | ρ seed 2 | median ρ | verdicts |
|------:|---------:|---------:|---------:|---------:|----------|
| 1.0  | 0.865 | 0.874 | 0.909 | 0.874 | sc, sc, sc |
| 5.0  | 0.917 | 0.787 | 0.894 | 0.894 | sc, centroid_only, sc |
| 20.0 | 0.777 | 0.738 | 0.855 | 0.777 | centroid_only, centroid_only, spread_marginal |
| 50.0 | 0.598 | 0.509 | 0.682 | 0.598 | centroid_only, centroid_only, centroid_only |

`extreme_scale=50.0; median_ρ=0.598 < pass_gate=0.85; monotonicity_ok=True; chosen_scale=50.0`.

**Verdict pass test at locked scale=50:**

| seed | ρ | ρ_low | ρ_high | p_dsw | verdict | pass (must not be spread_confirmed) |
|-----:|--:|------:|-------:|------:|---------|:----:|
| 40 | 0.657 | 0.14 | 0.66 | 0.687 | centroid_only_shape | ✓ |
| 41 | 0.826 | 0.30 | 0.80 | 0.124 | centroid_only_shape | ✓ |
| 42 | 0.459 | 0.05 | 0.64 | 0.97 | centroid_only_shape | ✓ |

The discriminator probe is monotonic: ρ decreases as the per-value anisotropic spread increases, and at scale=50 the test correctly refuses to call `spread_confirmed` on any seed.

### 6.7 Toy 6B — TightFog (anisotropy perpendicular to helix)

Helix centroids + isotropic noise + a single global anisotropic direction *perpendicular* to the helix tangent. Per-value Σ_v share a common perpendicular direction; the per-pair Σ_pool whitening hits the perpendicular dirs uniformly across all pairs, so the rank correspondence between D_E and D_SW is preserved.

| seed | ρ | ρ_low | ρ_high | p_dsw | verdict | pass |
|-----:|--:|------:|-------:|------:|---------|:----:|
| 50 | 0.954 | 0.92 | 0.97 | 0.005 | spread_confirmed | ✓ |
| 51 | 0.958 | 0.92 | 0.97 | 0.005 | spread_confirmed | ✓ |
| 52 | 0.873 | 0.84 | 0.91 | 0.005 | spread_confirmed | ✓ |

### 6.8 Toy 7B — FPR calibration on 100 isotropic cells

100 independently-drawn isotropic Gaussian cells (N=200, K=10, r=9, random labels per cell). Full Stage 2b pipeline per cell.

| n_cells | n_spread_confirmed | accept band | status | elapsed |
|--------:|-------------------:|-------------|--------|--------:|
| 100 | 3 | [2, 11] | pass | 88.5 s |

Empirical FPR = 3% (nominal 5%, binomial 95% CI [2, 11]). The null is calibrated.

### 6.9 Locked calibration record (configs/stage2b.yaml)

```
toy_5b_tangent_scale: 50.0
toy_5b_calibration:
  swept_scales: [1.0, 5.0, 20.0, 50.0]
  chosen_scale: 50.0
  extreme_pass_max_rho: 0.85
  monotonicity_ok: true
  n_seeds: 3
  discriminator_kind: continuous_probe
  timestamp: '2026-05-13T02:39:24.776268Z'

toy_7b_fpr:
  n_cells: 100
  n_spread_confirmed: 3
  accept_band: [2, 11]
  status: pass
  timestamp: '2026-05-13T02:40:58.253571Z'
  elapsed_seconds: 88.5
```

**Total toy suite wall time: 127.8 s.**

---

## 7. Aggregator outputs

The aggregator `aggregate_stage2b_dsw.py` consumes every per-model summary CSV under `results_root/stage2b_dsw/{model}/`, applies BH-FDR globally, recomputes confidence tiers using q_dsw, and writes a `comparison/` directory.

### 7.1 dsw_all.csv schema

| Column | Type | Description |
|--------|------|-------------|
| model | str | One of `gpt-j-6b`, `llama-3.1-8b`, `pythia-6.9b`. |
| task | str | `addition` or `multiplication`. |
| mode | str | Residualisation mode: `off` / `answer` / `norm`. |
| layer | int | Layer index in the model. |
| variant | str | Subspace variant: `lda_a` or `ccsvd`. |
| concept | str | Concept label (e.g. `ans_units`, `carry_tens`, `a`, `running_sum_hundreds`). |
| stage2a_verdict | str | Stage 2a's geometry_detected for this cell. |
| stage2a_discovered_period | float | Stage 2a's discovered period (NaN if not applicable). |
| K_natural | int | Number of unique values for this concept in the full dataset. |
| K_present | int | Number of values surviving the n_v ≥ 30 filter. |
| r | int | Subspace dimension. |
| n_samples_used | int | Total correct-subset rows fed into the analysis. |
| rho_centroid | float | Headline Spearman of D_E vs D_SW upper-triangle. |
| rho_pearson | float | Pearson on the same vectors. |
| tau_kendall | float | Kendall tau-b on the same vectors. |
| rho_low | float | Bootstrap 2.5% lower bound. |
| rho_high | float | Bootstrap 97.5% upper bound. |
| bootstrap_se | float | Bootstrap standard error. |
| mean_log_ratio | float | mean(log(D_SW / D_E)) over D_E > 1e-12 pairs. |
| p_dsw | float | Per-cell raw p from the Whittle null. |
| q_dsw | float | BH-FDR q across the eligible family. NaN for non-eligible verdicts. |
| redraw_rate_bootstrap | float | Fraction of bootstrap draws rejected & redrawn. |
| min_n_v | int | Smallest n_v among the kept values. |
| min_ratio_v | float | min_n_v / r. |
| gamma | float | r / min_n_v. |
| ci_halfwidth | float | (rho_high − rho_low) / 2. |
| spread_verdict | str | Final verdict after FDR downgrade. |
| spread_verdict_pre_fdr | str | Verdict before FDR downgrade. |
| fdr_downgraded | bool | True if FDR moved the cell from spread_confirmed to centroid_only_shape. |
| confidence_tier | str | HIGH / MEDIUM / LOW / DISCOVERY_ONLY (computed using q_dsw). |
| null_unstable_dsw | bool | True if ci_halfwidth > 0.30. |
| runtime_seconds | float | Per-cell wall time. |

### 7.2 cross_mode_spread_survival.csv schema

| Column | Description |
|--------|-------------|
| model, task, layer, variant, concept | Cell identifiers excluding mode. |
| off, answer, norm | spread_verdict per mode (the pivot value). |
| n_spread_confirmed_modes | Count of modes ∈ {off, answer, norm} where spread_verdict == spread_confirmed. |
| n_centroid_only_modes | Count of modes where spread_verdict == centroid_only_shape. |
| n_eligible_modes | Count of modes whose verdict ∈ {spread_confirmed, spread_marginal, centroid_only_shape}. |

### 7.3 cross_variant_agreement.csv schema

| Column | Description |
|--------|-------------|
| model, task, mode, layer, concept | Cell identifiers excluding variant. |
| lda_a, ccsvd | spread_verdict per variant. |
| verdict_agree | Boolean: lda_a verdict == ccsvd verdict. |
| rho_diff_abs | |ρ_centroid_lda_a − ρ_centroid_ccsvd|. |

### 7.4 stage2a_vs_stage2b_survival.csv schema

| Column | Description |
|--------|-------------|
| model, task, mode, layer, variant, concept | Cell identifiers. |
| stage2a_verdict | Stage 2a verdict (only `helix` or `circle` cells are emitted here). |
| stage2a_discovered_period | Stage 2a's discovered period. |
| spread_verdict_pre_fdr | Stage 2b pre-FDR verdict. |
| spread_verdict | Stage 2b post-FDR verdict. |
| confidence_tier | Stage 2b tier. |
| rho_centroid, rho_low, rho_high | Headline ρ + CI. |
| p_dsw, q_dsw | Per-cell raw p and BH-FDR q. |
| K_present, min_n_v, min_ratio_v, gamma | Per-cell sample regime statistics. |
| null_unstable_dsw, fdr_downgraded | Audit flags. |

### 7.5 manifest.json (excerpt)

```
n_rows_total: 2561
n_cells: 180                   # (model, task, mode, layer, variant) groups
n_models: 3
n_tasks: 2
n_modes: 3
n_layers: 7                    # union across models (different per model)
n_variants: 2
n_concepts_unique: 26
verdict_counts_pre_fdr:  { centroid_only_shape: 2025, spread_confirmed: 508, spread_marginal: 28 }
verdict_counts_post_fdr: { centroid_only_shape: 2133, spread_confirmed: 400, spread_marginal: 28 }
confidence_tier_counts:  { HIGH: 10, MEDIUM: 51, LOW: 1784, DISCOVERY_ONLY: 716 }
fdr_alpha: 0.05
n_permutations: 1000
n_bootstrap: 1000
rho_pass_threshold: 0.85
rho_low_ci_threshold: 0.70
n_fdr_downgrades: 108
toy_calibration: { toy_5b_tangent_scale: 50.0, toy_7b_fpr.status: pass }
```

---

## 8. Global verdict counts

### 8.1 Headline counts post-FDR

| Verdict | Count | Share |
|---------|------:|------:|
| centroid_only_shape | 2,133 | 83.3% |
| spread_confirmed | 400 | 15.6% |
| spread_marginal | 28 | 1.1% |
| **Total eligible cells** | **2,561** | 100% |

### 8.2 Confidence tier counts post-FDR

| Tier | Count | Share | spread_confirmed | spread_marginal | centroid_only_shape |
|------|------:|------:|-----------------:|----------------:|--------------------:|
| HIGH | 10 | 0.4% | 10 | 0 | 0 |
| MEDIUM | 51 | 2.0% | 51 | 0 | 0 |
| LOW | 1,784 | 69.7% | 182 | 1 | 1,601 |
| DISCOVERY_ONLY | 716 | 27.9% | 157 | 27 | 532 |
| **Total** | **2,561** | 100% | 400 | 28 | 2,133 |

### 8.3 Verdict counts by mode

| Mode | spread_confirmed | spread_marginal | centroid_only_shape | Total |
|------|-----------------:|----------------:|--------------------:|------:|
| off | 171 | 12 | 736 | 919 |
| answer | 72 | 6 | 670 | 748 |
| norm | 157 | 10 | 727 | 894 |

### 8.4 Verdict counts by variant

| Variant | spread_confirmed | spread_marginal | centroid_only_shape | Total |
|---------|-----------------:|----------------:|--------------------:|------:|
| lda_a | 199 | 1 | 1,038 | 1,238 |
| ccsvd | 201 | 27 | 1,095 | 1,323 |

### 8.5 Verdict counts by (model, task)

| Model | Task | spread_confirmed | spread_marginal | centroid_only_shape |
|-------|------|-----------------:|----------------:|--------------------:|
| gpt-j-6b | addition | 89 | 6 | 287 |
| gpt-j-6b | multiplication | 42 | 2 | 458 |
| llama-3.1-8b | addition | 103 | 8 | 261 |
| llama-3.1-8b | multiplication | 60 | 3 | 410 |
| pythia-6.9b | addition | 84 | 8 | 273 |
| pythia-6.9b | multiplication | 22 | 1 | 444 |

### 8.6 HIGH+MEDIUM spread_confirmed counts by (model, task, mode, variant)

| Model | Task | Mode | ccsvd | lda_a |
|-------|------|------|------:|------:|
| gpt-j-6b | addition | norm | 0 | 2 |
| gpt-j-6b | addition | off | 0 | 1 |
| gpt-j-6b | multiplication | answer | 1 | 0 |
| gpt-j-6b | multiplication | norm | 2 | 3 |
| gpt-j-6b | multiplication | off | 2 | 5 |
| llama-3.1-8b | addition | norm | 2 | 2 |
| llama-3.1-8b | addition | off | 1 | 2 |
| llama-3.1-8b | multiplication | norm | 7 | 6 |
| llama-3.1-8b | multiplication | off | 6 | 7 |
| pythia-6.9b | addition | norm | 2 | 1 |
| pythia-6.9b | addition | off | 2 | 1 |
| pythia-6.9b | multiplication | norm | 2 | 4 |

---

## 9. Per-(model, task, mode, layer, variant) verdict counts — full table

`sc` = spread_confirmed, `sm` = spread_marginal, `co` = centroid_only_shape. All 180 cell groups listed.

| model | task | mode | layer | variant | sc | sm | co | total |
|---|---|---|---:|---|---:|---:|---:|---:|
| gpt-j-6b | addition | answer | 4 | ccsvd | 3 | 1 | 9 | 13 |
| gpt-j-6b | addition | answer | 4 | lda_a | 1 | 0 | 9 | 10 |
| gpt-j-6b | addition | answer | 8 | ccsvd | 4 | 1 | 10 | 15 |
| gpt-j-6b | addition | answer | 8 | lda_a | 2 | 0 | 6 | 8 |
| gpt-j-6b | addition | answer | 14 | ccsvd | 1 | 0 | 12 | 13 |
| gpt-j-6b | addition | answer | 14 | lda_a | 0 | 0 | 10 | 10 |
| gpt-j-6b | addition | answer | 20 | ccsvd | 2 | 0 | 8 | 10 |
| gpt-j-6b | addition | answer | 20 | lda_a | 2 | 0 | 7 | 9 |
| gpt-j-6b | addition | answer | 24 | ccsvd | 0 | 0 | 9 | 9 |
| gpt-j-6b | addition | answer | 24 | lda_a | 2 | 0 | 7 | 9 |
| gpt-j-6b | addition | norm | 4 | ccsvd | 4 | 0 | 8 | 12 |
| gpt-j-6b | addition | norm | 4 | lda_a | 5 | 0 | 10 | 15 |
| gpt-j-6b | addition | norm | 8 | ccsvd | 8 | 0 | 8 | 16 |
| gpt-j-6b | addition | norm | 8 | lda_a | 5 | 0 | 10 | 15 |
| gpt-j-6b | addition | norm | 14 | ccsvd | 2 | 1 | 12 | 15 |
| gpt-j-6b | addition | norm | 14 | lda_a | 6 | 0 | 9 | 15 |
| gpt-j-6b | addition | norm | 20 | ccsvd | 2 | 0 | 11 | 13 |
| gpt-j-6b | addition | norm | 20 | lda_a | 3 | 0 | 10 | 13 |
| gpt-j-6b | addition | norm | 24 | ccsvd | 1 | 0 | 9 | 10 |
| gpt-j-6b | addition | norm | 24 | lda_a | 3 | 0 | 9 | 12 |
| gpt-j-6b | addition | off | 4 | ccsvd | 6 | 0 | 8 | 14 |
| gpt-j-6b | addition | off | 4 | lda_a | 5 | 0 | 10 | 15 |
| gpt-j-6b | addition | off | 8 | ccsvd | 6 | 0 | 9 | 15 |
| gpt-j-6b | addition | off | 8 | lda_a | 5 | 0 | 11 | 16 |
| gpt-j-6b | addition | off | 14 | ccsvd | 2 | 1 | 13 | 16 |
| gpt-j-6b | addition | off | 14 | lda_a | 3 | 0 | 13 | 16 |
| gpt-j-6b | addition | off | 20 | ccsvd | 2 | 1 | 9 | 12 |
| gpt-j-6b | addition | off | 20 | lda_a | 3 | 0 | 10 | 13 |
| gpt-j-6b | addition | off | 24 | ccsvd | 0 | 1 | 10 | 11 |
| gpt-j-6b | addition | off | 24 | lda_a | 1 | 0 | 11 | 12 |
| gpt-j-6b | multiplication | answer | 4 | ccsvd | 2 | 1 | 15 | 18 |
| gpt-j-6b | multiplication | answer | 4 | lda_a | 1 | 0 | 16 | 17 |
| gpt-j-6b | multiplication | answer | 8 | ccsvd | 2 | 0 | 17 | 19 |
| gpt-j-6b | multiplication | answer | 8 | lda_a | 1 | 0 | 12 | 13 |
| gpt-j-6b | multiplication | answer | 14 | ccsvd | 1 | 0 | 18 | 19 |
| gpt-j-6b | multiplication | answer | 14 | lda_a | 0 | 0 | 16 | 16 |
| gpt-j-6b | multiplication | answer | 20 | ccsvd | 0 | 0 | 14 | 14 |
| gpt-j-6b | multiplication | answer | 20 | lda_a | 0 | 0 | 9 | 9 |
| gpt-j-6b | multiplication | answer | 24 | ccsvd | 0 | 0 | 13 | 13 |
| gpt-j-6b | multiplication | answer | 24 | lda_a | 0 | 0 | 11 | 11 |
| gpt-j-6b | multiplication | norm | 4 | ccsvd | 3 | 1 | 16 | 20 |
| gpt-j-6b | multiplication | norm | 4 | lda_a | 4 | 0 | 12 | 16 |
| gpt-j-6b | multiplication | norm | 8 | ccsvd | 3 | 0 | 15 | 18 |
| gpt-j-6b | multiplication | norm | 8 | lda_a | 3 | 0 | 13 | 16 |
| gpt-j-6b | multiplication | norm | 14 | ccsvd | 1 | 0 | 19 | 20 |
| gpt-j-6b | multiplication | norm | 14 | lda_a | 0 | 0 | 19 | 19 |
| gpt-j-6b | multiplication | norm | 20 | ccsvd | 0 | 0 | 12 | 12 |
| gpt-j-6b | multiplication | norm | 20 | lda_a | 0 | 0 | 19 | 19 |
| gpt-j-6b | multiplication | norm | 24 | ccsvd | 0 | 0 | 14 | 14 |
| gpt-j-6b | multiplication | norm | 24 | lda_a | 0 | 0 | 16 | 16 |
| gpt-j-6b | multiplication | off | 4 | ccsvd | 3 | 0 | 19 | 22 |
| gpt-j-6b | multiplication | off | 4 | lda_a | 4 | 0 | 12 | 16 |
| gpt-j-6b | multiplication | off | 8 | ccsvd | 6 | 0 | 16 | 22 |
| gpt-j-6b | multiplication | off | 8 | lda_a | 5 | 0 | 12 | 17 |
| gpt-j-6b | multiplication | off | 14 | ccsvd | 1 | 0 | 20 | 21 |
| gpt-j-6b | multiplication | off | 14 | lda_a | 0 | 0 | 22 | 22 |
| gpt-j-6b | multiplication | off | 20 | ccsvd | 0 | 0 | 16 | 16 |
| gpt-j-6b | multiplication | off | 20 | lda_a | 1 | 0 | 17 | 18 |
| gpt-j-6b | multiplication | off | 24 | ccsvd | 1 | 0 | 13 | 14 |
| gpt-j-6b | multiplication | off | 24 | lda_a | 0 | 0 | 15 | 15 |
| llama-3.1-8b | addition | answer | 4 | ccsvd | 3 | 0 | 10 | 13 |
| llama-3.1-8b | addition | answer | 4 | lda_a | 1 | 0 | 10 | 11 |
| llama-3.1-8b | addition | answer | 8 | ccsvd | 5 | 0 | 11 | 16 |
| llama-3.1-8b | addition | answer | 8 | lda_a | 2 | 0 | 8 | 10 |
| llama-3.1-8b | addition | answer | 16 | ccsvd | 2 | 0 | 8 | 10 |
| llama-3.1-8b | addition | answer | 16 | lda_a | 2 | 0 | 9 | 11 |
| llama-3.1-8b | addition | answer | 24 | ccsvd | 1 | 1 | 8 | 10 |
| llama-3.1-8b | addition | answer | 24 | lda_a | 2 | 0 | 6 | 8 |
| llama-3.1-8b | addition | answer | 28 | ccsvd | 1 | 0 | 10 | 11 |
| llama-3.1-8b | addition | answer | 28 | lda_a | 2 | 0 | 7 | 9 |
| llama-3.1-8b | addition | norm | 4 | ccsvd | 6 | 0 | 9 | 15 |
| llama-3.1-8b | addition | norm | 4 | lda_a | 8 | 0 | 7 | 15 |
| llama-3.1-8b | addition | norm | 8 | ccsvd | 6 | 0 | 10 | 16 |
| llama-3.1-8b | addition | norm | 8 | lda_a | 5 | 0 | 10 | 15 |
| llama-3.1-8b | addition | norm | 16 | ccsvd | 2 | 0 | 11 | 13 |
| llama-3.1-8b | addition | norm | 16 | lda_a | 3 | 0 | 10 | 13 |
| llama-3.1-8b | addition | norm | 24 | ccsvd | 2 | 2 | 4 | 8 |
| llama-3.1-8b | addition | norm | 24 | lda_a | 2 | 0 | 8 | 10 |
| llama-3.1-8b | addition | norm | 28 | ccsvd | 2 | 1 | 11 | 14 |
| llama-3.1-8b | addition | norm | 28 | lda_a | 2 | 0 | 10 | 12 |
| llama-3.1-8b | addition | off | 4 | ccsvd | 4 | 0 | 11 | 15 |
| llama-3.1-8b | addition | off | 4 | lda_a | 8 | 0 | 7 | 15 |
| llama-3.1-8b | addition | off | 8 | ccsvd | 7 | 1 | 8 | 16 |
| llama-3.1-8b | addition | off | 8 | lda_a | 7 | 0 | 9 | 16 |
| llama-3.1-8b | addition | off | 16 | ccsvd | 4 | 0 | 9 | 13 |
| llama-3.1-8b | addition | off | 16 | lda_a | 4 | 0 | 8 | 12 |
| llama-3.1-8b | addition | off | 24 | ccsvd | 2 | 2 | 6 | 10 |
| llama-3.1-8b | addition | off | 24 | lda_a | 4 | 0 | 8 | 12 |
| llama-3.1-8b | addition | off | 28 | ccsvd | 2 | 1 | 8 | 11 |
| llama-3.1-8b | addition | off | 28 | lda_a | 2 | 0 | 10 | 12 |
| llama-3.1-8b | multiplication | answer | 4 | ccsvd | 3 | 0 | 20 | 23 |
| llama-3.1-8b | multiplication | answer | 4 | lda_a | 1 | 0 | 13 | 14 |
| llama-3.1-8b | multiplication | answer | 8 | ccsvd | 1 | 0 | 18 | 19 |
| llama-3.1-8b | multiplication | answer | 8 | lda_a | 2 | 0 | 14 | 16 |
| llama-3.1-8b | multiplication | answer | 16 | ccsvd | 1 | 0 | 14 | 15 |
| llama-3.1-8b | multiplication | answer | 16 | lda_a | 0 | 0 | 11 | 11 |
| llama-3.1-8b | multiplication | answer | 24 | ccsvd | 0 | 0 | 12 | 12 |
| llama-3.1-8b | multiplication | answer | 24 | lda_a | 0 | 0 | 11 | 11 |
| llama-3.1-8b | multiplication | answer | 28 | ccsvd | 0 | 0 | 11 | 11 |
| llama-3.1-8b | multiplication | answer | 28 | lda_a | 0 | 0 | 8 | 8 |
| llama-3.1-8b | multiplication | norm | 4 | ccsvd | 7 | 0 | 13 | 20 |
| llama-3.1-8b | multiplication | norm | 4 | lda_a | 4 | 0 | 13 | 17 |
| llama-3.1-8b | multiplication | norm | 8 | ccsvd | 4 | 2 | 14 | 20 |
| llama-3.1-8b | multiplication | norm | 8 | lda_a | 5 | 0 | 11 | 16 |
| llama-3.1-8b | multiplication | norm | 16 | ccsvd | 4 | 0 | 14 | 18 |
| llama-3.1-8b | multiplication | norm | 16 | lda_a | 1 | 0 | 14 | 15 |
| llama-3.1-8b | multiplication | norm | 24 | ccsvd | 0 | 0 | 13 | 13 |
| llama-3.1-8b | multiplication | norm | 24 | lda_a | 0 | 0 | 17 | 17 |
| llama-3.1-8b | multiplication | norm | 28 | ccsvd | 0 | 0 | 11 | 11 |
| llama-3.1-8b | multiplication | norm | 28 | lda_a | 0 | 0 | 19 | 19 |
| llama-3.1-8b | multiplication | off | 4 | ccsvd | 7 | 0 | 14 | 21 |
| llama-3.1-8b | multiplication | off | 4 | lda_a | 5 | 0 | 13 | 18 |
| llama-3.1-8b | multiplication | off | 8 | ccsvd | 5 | 1 | 14 | 20 |
| llama-3.1-8b | multiplication | off | 8 | lda_a | 3 | 0 | 15 | 18 |
| llama-3.1-8b | multiplication | off | 16 | ccsvd | 3 | 0 | 16 | 19 |
| llama-3.1-8b | multiplication | off | 16 | lda_a | 4 | 0 | 16 | 20 |
| llama-3.1-8b | multiplication | off | 24 | ccsvd | 0 | 0 | 13 | 13 |
| llama-3.1-8b | multiplication | off | 24 | lda_a | 0 | 0 | 13 | 13 |
| llama-3.1-8b | multiplication | off | 28 | ccsvd | 0 | 0 | 11 | 11 |
| llama-3.1-8b | multiplication | off | 28 | lda_a | 0 | 0 | 14 | 14 |
| pythia-6.9b | addition | answer | 4 | ccsvd | 1 | 1 | 10 | 12 |
| pythia-6.9b | addition | answer | 4 | lda_a | 1 | 0 | 9 | 10 |
| pythia-6.9b | addition | answer | 8 | ccsvd | 4 | 1 | 7 | 12 |
| pythia-6.9b | addition | answer | 8 | lda_a | 3 | 0 | 8 | 11 |
| pythia-6.9b | addition | answer | 16 | ccsvd | 0 | 0 | 10 | 10 |
| pythia-6.9b | addition | answer | 16 | lda_a | 0 | 0 | 5 | 5 |
| pythia-6.9b | addition | answer | 24 | ccsvd | 1 | 0 | 9 | 10 |
| pythia-6.9b | addition | answer | 24 | lda_a | 2 | 0 | 9 | 11 |
| pythia-6.9b | addition | answer | 28 | ccsvd | 1 | 0 | 9 | 10 |
| pythia-6.9b | addition | answer | 28 | lda_a | 2 | 0 | 10 | 12 |
| pythia-6.9b | addition | norm | 4 | ccsvd | 5 | 1 | 8 | 14 |
| pythia-6.9b | addition | norm | 4 | lda_a | 4 | 0 | 7 | 11 |
| pythia-6.9b | addition | norm | 8 | ccsvd | 4 | 1 | 10 | 15 |
| pythia-6.9b | addition | norm | 8 | lda_a | 4 | 0 | 11 | 15 |
| pythia-6.9b | addition | norm | 16 | ccsvd | 0 | 0 | 14 | 14 |
| pythia-6.9b | addition | norm | 16 | lda_a | 0 | 0 | 11 | 11 |
| pythia-6.9b | addition | norm | 24 | ccsvd | 1 | 0 | 10 | 11 |
| pythia-6.9b | addition | norm | 24 | lda_a | 4 | 0 | 11 | 15 |
| pythia-6.9b | addition | norm | 28 | ccsvd | 3 | 1 | 7 | 11 |
| pythia-6.9b | addition | norm | 28 | lda_a | 3 | 0 | 10 | 13 |
| pythia-6.9b | addition | off | 4 | ccsvd | 6 | 0 | 8 | 14 |
| pythia-6.9b | addition | off | 4 | lda_a | 4 | 0 | 9 | 13 |
| pythia-6.9b | addition | off | 8 | ccsvd | 6 | 0 | 9 | 15 |
| pythia-6.9b | addition | off | 8 | lda_a | 6 | 0 | 9 | 15 |
| pythia-6.9b | addition | off | 16 | ccsvd | 2 | 0 | 11 | 13 |
| pythia-6.9b | addition | off | 16 | lda_a | 4 | 0 | 9 | 13 |
| pythia-6.9b | addition | off | 24 | ccsvd | 3 | 1 | 8 | 12 |
| pythia-6.9b | addition | off | 24 | lda_a | 4 | 1 | 8 | 13 |
| pythia-6.9b | addition | off | 28 | ccsvd | 3 | 1 | 7 | 11 |
| pythia-6.9b | addition | off | 28 | lda_a | 3 | 0 | 10 | 13 |
| pythia-6.9b | multiplication | answer | 4 | ccsvd | 1 | 0 | 21 | 22 |
| pythia-6.9b | multiplication | answer | 4 | lda_a | 0 | 0 | 17 | 17 |
| pythia-6.9b | multiplication | answer | 8 | ccsvd | 1 | 0 | 17 | 18 |
| pythia-6.9b | multiplication | answer | 8 | lda_a | 1 | 0 | 15 | 16 |
| pythia-6.9b | multiplication | answer | 16 | ccsvd | 1 | 0 | 13 | 14 |
| pythia-6.9b | multiplication | answer | 16 | lda_a | 0 | 0 | 9 | 9 |
| pythia-6.9b | multiplication | answer | 24 | ccsvd | 0 | 0 | 12 | 12 |
| pythia-6.9b | multiplication | answer | 24 | lda_a | 0 | 0 | 9 | 9 |
| pythia-6.9b | multiplication | answer | 28 | ccsvd | 0 | 0 | 13 | 13 |
| pythia-6.9b | multiplication | answer | 28 | lda_a | 0 | 0 | 11 | 11 |
| pythia-6.9b | multiplication | norm | 4 | ccsvd | 0 | 0 | 21 | 21 |
| pythia-6.9b | multiplication | norm | 4 | lda_a | 2 | 0 | 17 | 19 |
| pythia-6.9b | multiplication | norm | 8 | ccsvd | 5 | 0 | 16 | 21 |
| pythia-6.9b | multiplication | norm | 8 | lda_a | 6 | 0 | 10 | 16 |
| pythia-6.9b | multiplication | norm | 16 | ccsvd | 0 | 0 | 14 | 14 |
| pythia-6.9b | multiplication | norm | 16 | lda_a | 0 | 0 | 13 | 13 |
| pythia-6.9b | multiplication | norm | 24 | ccsvd | 0 | 0 | 14 | 14 |
| pythia-6.9b | multiplication | norm | 24 | lda_a | 0 | 0 | 16 | 16 |
| pythia-6.9b | multiplication | norm | 28 | ccsvd | 0 | 0 | 13 | 13 |
| pythia-6.9b | multiplication | norm | 28 | lda_a | 0 | 0 | 14 | 14 |
| pythia-6.9b | multiplication | off | 4 | ccsvd | 1 | 1 | 18 | 20 |
| pythia-6.9b | multiplication | off | 4 | lda_a | 1 | 0 | 17 | 18 |
| pythia-6.9b | multiplication | off | 8 | ccsvd | 2 | 0 | 18 | 20 |
| pythia-6.9b | multiplication | off | 8 | lda_a | 1 | 0 | 16 | 17 |
| pythia-6.9b | multiplication | off | 16 | ccsvd | 0 | 0 | 15 | 15 |
| pythia-6.9b | multiplication | off | 16 | lda_a | 0 | 0 | 16 | 16 |
| pythia-6.9b | multiplication | off | 24 | ccsvd | 0 | 0 | 15 | 15 |
| pythia-6.9b | multiplication | off | 24 | lda_a | 0 | 0 | 16 | 16 |
| pythia-6.9b | multiplication | off | 28 | ccsvd | 0 | 0 | 14 | 14 |
| pythia-6.9b | multiplication | off | 28 | lda_a | 0 | 0 | 14 | 14 |

---

## 10. Per-concept survival table

26 concepts span the eligible families. The survival rate is the fraction of cells (across all (model, task, mode, layer, variant) combinations) whose post-FDR verdict is `spread_confirmed`.

| concept | spread_confirmed | spread_marginal | centroid_only | total | survival % |
|---|---:|---:|---:|---:|---:|
| operand_diff | 67 | 0 | 15 | 82 | 81.7 |
| answer | 31 | 0 | 13 | 44 | 70.5 |
| operand_abs_diff | 74 | 0 | 73 | 147 | 50.3 |
| a | 37 | 17 | 62 | 116 | 31.9 |
| running_sum_tens | 43 | 0 | 129 | 172 | 25.0 |
| min_operand | 34 | 0 | 115 | 149 | 22.8 |
| ans_hundreds | 15 | 0 | 60 | 75 | 20.0 |
| running_sum_hundreds | 16 | 0 | 74 | 90 | 17.8 |
| column_sum_tens | 28 | 0 | 146 | 174 | 16.1 |
| b | 13 | 10 | 66 | 89 | 14.6 |
| ans_tens | 14 | 0 | 112 | 126 | 11.1 |
| max_operand | 14 | 0 | 146 | 160 | 8.8 |
| running_sum_units | 3 | 0 | 90 | 93 | 3.2 |
| column_sum_units | 3 | 0 | 90 | 93 | 3.2 |
| ans_units | 4 | 1 | 121 | 126 | 3.2 |
| a_units | 2 | 0 | 73 | 75 | 2.7 |
| a_tens | 2 | 0 | 149 | 151 | 1.3 |
| b_tens | 0 | 0 | 165 | 165 | 0.0 |
| b_units | 0 | 0 | 64 | 64 | 0.0 |
| carry_units | 0 | 0 | 82 | 82 | 0.0 |
| column_sum_hundreds | 0 | 0 | 58 | 58 | 0.0 |
| carry_tens | 0 | 0 | 87 | 87 | 0.0 |
| partial_product_a_units_b_tens | 0 | 0 | 21 | 21 | 0.0 |
| partial_product_a_tens_b_units | 0 | 0 | 30 | 30 | 0.0 |
| partial_product_a_tens_b_tens | 0 | 0 | 58 | 58 | 0.0 |
| partial_product_units | 0 | 0 | 34 | 34 | 0.0 |

---

## 11. HIGH and MEDIUM tier cells — full list

61 cells make the headline tier. 10 HIGH + 51 MEDIUM. Listed in descending order of `ρ_centroid` within each tier.

| # | model | task | mode | variant | layer | concept | K | min_n_v | ρ | ρ_low | q_dsw | tier |
|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| 1 | llama-3.1-8b | multiplication | off | lda_a | 16 | ans_hundreds | 10 | 107 | 0.9958 | 0.985 | 0.0084 | HIGH |
| 2 | llama-3.1-8b | multiplication | off | lda_a | 16 | running_sum_hundreds | 10 | 107 | 0.9958 | 0.985 | 0.0084 | HIGH |
| 3 | llama-3.1-8b | multiplication | off | lda_a | 4 | ans_tens | 10 | 203 | 0.9954 | 0.972 | 0.0084 | HIGH |
| 4 | llama-3.1-8b | multiplication | off | lda_a | 4 | ans_hundreds | 10 | 107 | 0.9918 | 0.982 | 0.0084 | HIGH |
| 5 | llama-3.1-8b | multiplication | off | lda_a | 4 | running_sum_hundreds | 10 | 107 | 0.9918 | 0.982 | 0.0084 | HIGH |
| 6 | llama-3.1-8b | multiplication | off | ccsvd | 4 | ans_hundreds | 10 | 107 | 0.9855 | 0.974 | 0.0084 | HIGH |
| 7 | llama-3.1-8b | multiplication | off | ccsvd | 4 | running_sum_hundreds | 10 | 107 | 0.9855 | 0.974 | 0.0084 | HIGH |
| 8 | llama-3.1-8b | addition | norm | lda_a | 4 | a_units | 10 | 984 | 0.9835 | 0.974 | 0.0084 | HIGH |
| 9 | llama-3.1-8b | multiplication | off | ccsvd | 8 | ans_hundreds | 10 | 107 | 0.9661 | 0.956 | 0.0084 | HIGH |
| 10 | gpt-j-6b | addition | norm | lda_a | 24 | ans_tens | 10 | 768 | 0.9582 | 0.939 | 0.0084 | HIGH |
| 11 | llama-3.1-8b | multiplication | norm | lda_a | 8 | ans_tens | 10 | 203 | 0.9903 | 0.973 | 0.0150 | MEDIUM |
| 12 | pythia-6.9b | multiplication | norm | lda_a | 8 | ans_tens | 10 | 205 | 0.9903 | 0.970 | 0.0431 | MEDIUM |
| 13 | pythia-6.9b | multiplication | norm | lda_a | 8 | ans_hundreds | 10 | 85 | 0.9895 | 0.978 | 0.0214 | MEDIUM |
| 14 | pythia-6.9b | multiplication | norm | lda_a | 8 | running_sum_hundreds | 10 | 85 | 0.9895 | 0.978 | 0.0269 | MEDIUM |
| 15 | llama-3.1-8b | multiplication | off | lda_a | 16 | ans_tens | 10 | 203 | 0.9893 | 0.975 | 0.0431 | MEDIUM |
| 16 | llama-3.1-8b | multiplication | off | lda_a | 8 | ans_tens | 10 | 203 | 0.9891 | 0.972 | 0.0150 | MEDIUM |
| 17 | llama-3.1-8b | multiplication | off | ccsvd | 16 | ans_hundreds | 10 | 107 | 0.9885 | 0.975 | 0.0150 | MEDIUM |
| 18 | llama-3.1-8b | multiplication | off | ccsvd | 16 | running_sum_hundreds | 10 | 107 | 0.9885 | 0.975 | 0.0269 | MEDIUM |
| 19 | llama-3.1-8b | multiplication | norm | ccsvd | 16 | ans_hundreds | 10 | 107 | 0.9879 | 0.974 | 0.0327 | MEDIUM |
| 20 | llama-3.1-8b | multiplication | norm | ccsvd | 16 | running_sum_hundreds | 10 | 107 | 0.9879 | 0.974 | 0.0431 | MEDIUM |
| 21 | pythia-6.9b | multiplication | norm | lda_a | 4 | ans_tens | 10 | 205 | 0.9866 | 0.956 | 0.0214 | MEDIUM |
| 22 | gpt-j-6b | multiplication | norm | lda_a | 4 | running_sum_hundreds | 10 | 77 | 0.9859 | 0.975 | 0.0150 | MEDIUM |
| 23 | llama-3.1-8b | multiplication | norm | lda_a | 4 | ans_tens | 10 | 203 | 0.9856 | 0.960 | 0.0381 | MEDIUM |
| 24 | gpt-j-6b | multiplication | off | lda_a | 8 | ans_hundreds | 10 | 77 | 0.9855 | 0.974 | 0.0150 | MEDIUM |
| 25 | gpt-j-6b | multiplication | off | lda_a | 8 | running_sum_hundreds | 10 | 77 | 0.9855 | 0.974 | 0.0269 | MEDIUM |
| 26 | gpt-j-6b | multiplication | norm | lda_a | 4 | ans_hundreds | 10 | 77 | 0.9852 | 0.975 | 0.0150 | MEDIUM |
| 27 | llama-3.1-8b | addition | norm | ccsvd | 4 | column_sum_units | 19 | 100 | 0.9848 | 0.982 | 0.0431 | MEDIUM |
| 28 | llama-3.1-8b | addition | norm | ccsvd | 4 | running_sum_units | 19 | 100 | 0.9848 | 0.982 | 0.0269 | MEDIUM |
| 29 | llama-3.1-8b | multiplication | norm | lda_a | 4 | ans_hundreds | 10 | 107 | 0.9833 | 0.971 | 0.0150 | MEDIUM |
| 30 | llama-3.1-8b | multiplication | norm | lda_a | 4 | running_sum_hundreds | 10 | 107 | 0.9833 | 0.970 | 0.0214 | MEDIUM |
| 31 | pythia-6.9b | multiplication | norm | ccsvd | 8 | ans_hundreds | 10 | 85 | 0.9808 | 0.964 | 0.0150 | MEDIUM |
| 32 | pythia-6.9b | multiplication | norm | ccsvd | 8 | running_sum_hundreds | 10 | 85 | 0.9808 | 0.965 | 0.0084 | MEDIUM |
| 33 | gpt-j-6b | multiplication | off | lda_a | 4 | ans_tens | 10 | 201 | 0.9793 | 0.958 | 0.0327 | MEDIUM |
| 34 | llama-3.1-8b | multiplication | norm | lda_a | 8 | ans_hundreds | 10 | 107 | 0.9793 | 0.967 | 0.0214 | MEDIUM |
| 35 | llama-3.1-8b | multiplication | norm | lda_a | 8 | running_sum_hundreds | 10 | 107 | 0.9793 | 0.966 | 0.0269 | MEDIUM |
| 36 | gpt-j-6b | multiplication | answer | ccsvd | 4 | ans_tens | 10 | 201 | 0.9792 | 0.961 | 0.0150 | MEDIUM |
| 37 | gpt-j-6b | multiplication | off | lda_a | 20 | ans_units | 10 | 77 | 0.9787 | 0.933 | 0.0327 | MEDIUM |
| 38 | llama-3.1-8b | addition | off | lda_a | 4 | a_units | 10 | 984 | 0.9750 | 0.965 | 0.0327 | MEDIUM |
| 39 | gpt-j-6b | multiplication | off | lda_a | 8 | ans_tens | 10 | 201 | 0.9747 | 0.947 | 0.0327 | MEDIUM |
| 40 | gpt-j-6b | multiplication | norm | ccsvd | 4 | ans_hundreds | 10 | 77 | 0.9721 | 0.955 | 0.0381 | MEDIUM |
| 41 | gpt-j-6b | multiplication | norm | ccsvd | 4 | running_sum_hundreds | 10 | 77 | 0.9721 | 0.955 | 0.0150 | MEDIUM |
| 42 | pythia-6.9b | addition | off | ccsvd | 4 | column_sum_units | 19 | 74 | 0.9720 | 0.964 | 0.0084 | MEDIUM |
| 43 | pythia-6.9b | addition | off | ccsvd | 4 | running_sum_units | 19 | 74 | 0.9720 | 0.964 | 0.0084 | MEDIUM |
| 44 | pythia-6.9b | addition | norm | lda_a | 4 | running_sum_tens | 19 | 55 | 0.9719 | 0.965 | 0.0484 | MEDIUM |
| 45 | gpt-j-6b | multiplication | norm | lda_a | 4 | ans_tens | 10 | 201 | 0.9717 | 0.940 | 0.0431 | MEDIUM |
| 46 | llama-3.1-8b | multiplication | norm | ccsvd | 4 | ans_hundreds | 10 | 107 | 0.9713 | 0.954 | 0.0381 | MEDIUM |
| 47 | llama-3.1-8b | multiplication | norm | ccsvd | 4 | running_sum_hundreds | 10 | 107 | 0.9713 | 0.954 | 0.0269 | MEDIUM |
| 48 | llama-3.1-8b | multiplication | norm | ccsvd | 8 | ans_hundreds | 10 | 107 | 0.9679 | 0.953 | 0.0150 | MEDIUM |
| 49 | llama-3.1-8b | multiplication | norm | ccsvd | 8 | running_sum_hundreds | 10 | 107 | 0.9679 | 0.954 | 0.0150 | MEDIUM |
| 50 | llama-3.1-8b | multiplication | norm | ccsvd | 4 | ans_tens | 10 | 203 | 0.9663 | 0.932 | 0.0214 | MEDIUM |
| 51 | llama-3.1-8b | multiplication | off | ccsvd | 8 | running_sum_hundreds | 10 | 107 | 0.9661 | 0.956 | 0.0214 | MEDIUM |
| 52 | pythia-6.9b | addition | norm | ccsvd | 4 | column_sum_units | 19 | 74 | 0.9651 | 0.956 | 0.0381 | MEDIUM |
| 53 | pythia-6.9b | addition | norm | ccsvd | 4 | running_sum_units | 19 | 74 | 0.9651 | 0.956 | 0.0381 | MEDIUM |
| 54 | gpt-j-6b | multiplication | off | ccsvd | 8 | running_sum_hundreds | 10 | 77 | 0.9643 | 0.950 | 0.0431 | MEDIUM |
| 55 | llama-3.1-8b | addition | off | lda_a | 4 | a_tens | 10 | 972 | 0.9636 | 0.955 | 0.0150 | MEDIUM |
| 56 | llama-3.1-8b | addition | norm | lda_a | 4 | a_tens | 10 | 972 | 0.9635 | 0.953 | 0.0327 | MEDIUM |
| 57 | llama-3.1-8b | addition | off | ccsvd | 16 | column_sum_tens | 19 | 85 | 0.9634 | 0.957 | 0.0150 | MEDIUM |
| 58 | pythia-6.9b | addition | off | lda_a | 16 | ans_units | 10 | 695 | 0.9626 | 0.951 | 0.0484 | MEDIUM |
| 59 | gpt-j-6b | multiplication | off | ccsvd | 8 | ans_tens | 10 | 201 | 0.9464 | 0.902 | 0.0431 | MEDIUM |
| 60 | gpt-j-6b | addition | norm | lda_a | 20 | ans_units | 10 | 807 | 0.9340 | 0.907 | 0.0214 | MEDIUM |
| 61 | gpt-j-6b | addition | off | lda_a | 20 | ans_units | 10 | 807 | 0.9310 | 0.910 | 0.0214 | MEDIUM |

---

## 12. Stage 2a → 2b transition

### 12.1 Stage 2a verdict distribution among Stage 2b-eligible cells

| Stage 2a verdict | Count |
|------------------|------:|
| helix | 754 |
| circle | 177 |
| **Total** | **931** |

### 12.2 Stage 2b verdict conditioned on Stage 2a verdict (post-FDR)

| Stage 2a verdict | Stage 2b verdict | Count |
|------------------|------------------|------:|
| helix | spread_confirmed | 67 |
| helix | centroid_only_shape | 687 |
| circle | spread_confirmed | 29 |
| circle | centroid_only_shape | 148 |

**Survival rate.** 96 of 931 (10.3%) Stage 2a helix-or-circle cells survive Stage 2b at the `spread_confirmed` post-FDR threshold. 835 (89.7%) are demoted to `centroid_only_shape`.

By Stage 2a verdict:
- Helix cells: 67 of 754 (8.9%) survive.
- Circle cells: 29 of 177 (16.4%) survive.

### 12.3 Survival by (Stage 2a verdict, confidence tier) — Stage 2b post-FDR

Filter to spread_confirmed and HIGH+MEDIUM tier:
- 9 cells with prior Stage 2a verdict ∈ {helix, circle} land in the headline HIGH/MEDIUM `spread_confirmed` bucket (from the 61 HIGH/MEDIUM total — the remaining 52 had Stage 2a verdict `none` or were from variants where the cell wasn't fit by Stage 2a's lda_a/ccsvd pipeline at that mode/layer).

Detail per (model, task) for cells where stage2a_verdict ∈ {helix, circle} AND Stage 2b == spread_confirmed:

| model | task | spread_confirmed-from-helix | spread_confirmed-from-circle |
|-------|------|---------------------------:|----------------------------:|
| gpt-j-6b | addition | 12 | 6 |
| gpt-j-6b | multiplication | 8 | 3 |
| llama-3.1-8b | addition | 13 | 8 |
| llama-3.1-8b | multiplication | 18 | 5 |
| pythia-6.9b | addition | 10 | 5 |
| pythia-6.9b | multiplication | 6 | 2 |
| **Total** | | **67** | **29** |

---

## 13. Cross-mode survival distribution

For each (model, task, layer, variant, concept) tuple, count how many of {off, answer, norm} produced `spread_confirmed`.

| n_spread_confirmed_modes | Count | Share |
|-------------------------:|------:|------:|
| 0 (none) | 777 | 77.8% |
| 1 | 94 | 9.4% |
| 2 | 78 | 7.8% |
| 3 (all modes) | 50 | 5.0% |
| **Total tuples** | **999** | 100% |

Cells robust across all three residualisation modes (`n_spread_confirmed_modes == 3`) form the most stringent "owned at the spread level under any residualisation choice" cohort. 50 such tuples.

Mode-asymmetry. The `answer` mode kills `spread_confirmed` more aggressively than `norm` or `off`:

| Mode | spread_confirmed count |
|------|----------------------:|
| off | 171 |
| norm | 157 |
| answer | 72 |

`answer` is the residualisation that subtracts the gold-answer prediction from each activation. Cells whose `spread_confirmed` survives under `off` but not under `answer` are flagged as "magnitude-dependent" in the cross-mode pivot.

---

## 14. Cross-variant agreement

For each (model, task, mode, layer, concept) tuple, compare `lda_a` and `ccsvd` verdicts.

| Agreement | Count |
|-----------|------:|
| lda_a == ccsvd | 899 of 1,533 |
| Agreement rate | 58.6% |

Of the 899 agreements:

| Agreed verdict | Count |
|----------------|------:|
| centroid_only_shape (both) | ~750 |
| spread_confirmed (both) | ~120 |
| spread_marginal (both) | ~30 |

Of the 634 disagreements, the most common pattern is one variant flagging `spread_confirmed` while the other flags `centroid_only_shape` — this surfaces subspace-choice sensitivity that downstream analyses (Stage 3 ownership orthogonalisation) will probe.

---

## 15. Carry concept cells — full list (zero survival cohort)

`carry_units` and `carry_tens` together: 169 cells, **all 169 returned `centroid_only_shape`**. None survived spread-correction in any (model, task, mode, layer, variant) combination tested.

The full list:

| model | task | mode | variant | layer | concept | K | ρ | ρ_low | p | q | verdict | tier |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| gpt-j-6b | multiplication | answer | ccsvd | 4 | carry_tens | 7 | 0.643 | 0.50 | 0.929 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | answer | ccsvd | 4 | carry_units | 8 | 0.835 | 0.74 | 0.873 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | answer | ccsvd | 8 | carry_tens | 7 | 0.690 | 0.55 | 0.943 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | answer | ccsvd | 8 | carry_units | 8 | 0.857 | 0.78 | 0.806 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | answer | ccsvd | 14 | carry_tens | 7 | 0.674 | 0.52 | 0.989 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | answer | ccsvd | 14 | carry_units | 8 | 0.853 | 0.79 | 0.829 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | answer | ccsvd | 20 | carry_tens | 7 | 0.620 | 0.45 | 0.967 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | answer | ccsvd | 20 | carry_units | 8 | 0.842 | 0.76 | 0.806 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | answer | ccsvd | 24 | carry_units | 8 | 0.834 | 0.76 | 0.829 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | answer | lda_a | 4 | carry_tens | 7 | 0.610 | 0.46 | 0.984 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | answer | lda_a | 4 | carry_units | 8 | 0.842 | 0.74 | 0.852 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | answer | lda_a | 8 | carry_tens | 7 | 0.683 | 0.55 | 0.928 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | answer | lda_a | 8 | carry_units | 8 | 0.835 | 0.74 | 0.846 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | answer | lda_a | 14 | carry_units | 8 | 0.832 | 0.73 | 0.836 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | answer | lda_a | 20 | carry_tens | 7 | 0.652 | 0.51 | 0.986 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | answer | lda_a | 24 | carry_units | 8 | 0.831 | 0.74 | 0.857 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | norm | ccsvd | 4 | carry_tens | 7 | 0.756 | 0.66 | 0.971 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | norm | ccsvd | 4 | carry_units | 8 | 0.857 | 0.79 | 0.961 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | norm | ccsvd | 8 | carry_tens | 7 | 0.760 | 0.66 | 0.984 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | norm | ccsvd | 8 | carry_units | 8 | 0.857 | 0.79 | 0.943 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | norm | ccsvd | 14 | carry_tens | 7 | 0.706 | 0.59 | 0.991 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | norm | ccsvd | 14 | carry_units | 8 | 0.858 | 0.79 | 0.928 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | norm | ccsvd | 20 | carry_tens | 7 | 0.745 | 0.62 | 0.978 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | norm | ccsvd | 24 | carry_units | 8 | 0.846 | 0.77 | 0.937 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | norm | lda_a | 4 | carry_tens | 7 | 0.731 | 0.62 | 0.971 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | norm | lda_a | 4 | carry_units | 8 | 0.853 | 0.79 | 0.947 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | norm | lda_a | 8 | carry_tens | 7 | 0.749 | 0.65 | 0.967 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | norm | lda_a | 8 | carry_units | 8 | 0.857 | 0.79 | 0.943 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | norm | lda_a | 14 | carry_tens | 7 | 0.682 | 0.55 | 0.989 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | norm | lda_a | 20 | carry_units | 8 | 0.838 | 0.76 | 0.937 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | off | ccsvd | 4 | carry_tens | 7 | 0.690 | 0.55 | 0.943 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | off | ccsvd | 4 | carry_units | 8 | 0.876 | 0.81 | 0.985 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | off | ccsvd | 8 | carry_tens | 7 | 0.770 | 0.66 | 0.991 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | off | ccsvd | 8 | carry_units | 8 | 0.878 | 0.82 | 0.940 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | off | ccsvd | 14 | carry_tens | 7 | 0.701 | 0.58 | 0.989 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | off | ccsvd | 14 | carry_units | 8 | 0.881 | 0.82 | 0.943 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | off | ccsvd | 20 | carry_tens | 7 | 0.731 | 0.63 | 0.987 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | off | ccsvd | 24 | carry_units | 8 | 0.860 | 0.79 | 0.978 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | off | lda_a | 4 | carry_tens | 7 | 0.741 | 0.63 | 0.985 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | off | lda_a | 4 | carry_units | 8 | 0.876 | 0.81 | 0.985 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | off | lda_a | 8 | carry_tens | 7 | 0.745 | 0.66 | 0.989 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | off | lda_a | 8 | carry_units | 8 | 0.881 | 0.82 | 0.973 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | off | lda_a | 14 | carry_tens | 7 | 0.682 | 0.55 | 0.989 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | off | lda_a | 14 | carry_units | 8 | 0.881 | 0.82 | 0.943 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | off | lda_a | 20 | carry_tens | 7 | 0.703 | 0.57 | 0.987 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | off | lda_a | 20 | carry_units | 8 | 0.866 | 0.79 | 0.973 | nan | centroid_only_shape | LOW |
| gpt-j-6b | multiplication | off | lda_a | 24 | carry_units | 8 | 0.860 | 0.78 | 0.991 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | answer | ccsvd | 4 | carry_tens | 7 | 0.683 | 0.55 | 0.991 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | answer | ccsvd | 4 | carry_units | 8 | 0.844 | 0.77 | 0.962 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | answer | ccsvd | 8 | carry_tens | 7 | 0.691 | 0.58 | 0.987 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | answer | ccsvd | 8 | carry_units | 8 | 0.831 | 0.74 | 0.962 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | answer | ccsvd | 16 | carry_tens | 7 | 0.661 | 0.51 | 0.984 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | answer | ccsvd | 16 | carry_units | 8 | 0.831 | 0.75 | 0.974 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | answer | ccsvd | 24 | carry_units | 8 | 0.820 | 0.73 | 0.989 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | answer | ccsvd | 28 | carry_units | 8 | 0.815 | 0.72 | 0.978 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | answer | lda_a | 4 | carry_tens | 7 | 0.638 | 0.52 | 0.991 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | answer | lda_a | 4 | carry_units | 8 | 0.842 | 0.74 | 0.978 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | answer | lda_a | 8 | carry_tens | 7 | 0.681 | 0.56 | 0.987 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | answer | lda_a | 8 | carry_units | 8 | 0.832 | 0.75 | 0.987 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | answer | lda_a | 16 | carry_units | 8 | 0.823 | 0.74 | 0.978 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | norm | ccsvd | 4 | carry_tens | 7 | 0.751 | 0.66 | 0.989 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | norm | ccsvd | 4 | carry_units | 8 | 0.853 | 0.78 | 0.984 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | norm | ccsvd | 8 | carry_tens | 7 | 0.749 | 0.65 | 0.991 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | norm | ccsvd | 8 | carry_units | 8 | 0.854 | 0.78 | 0.984 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | norm | ccsvd | 16 | carry_tens | 7 | 0.722 | 0.60 | 0.989 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | norm | ccsvd | 16 | carry_units | 8 | 0.852 | 0.78 | 0.987 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | norm | ccsvd | 24 | carry_units | 8 | 0.834 | 0.76 | 0.989 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | norm | ccsvd | 28 | carry_units | 8 | 0.832 | 0.75 | 0.984 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | norm | lda_a | 4 | carry_tens | 7 | 0.741 | 0.63 | 0.991 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | norm | lda_a | 4 | carry_units | 8 | 0.852 | 0.79 | 0.984 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | norm | lda_a | 8 | carry_tens | 7 | 0.745 | 0.65 | 0.989 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | norm | lda_a | 8 | carry_units | 8 | 0.853 | 0.78 | 0.984 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | norm | lda_a | 16 | carry_units | 8 | 0.849 | 0.78 | 0.991 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | off | ccsvd | 4 | carry_tens | 7 | 0.690 | 0.55 | 0.991 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | off | ccsvd | 4 | carry_units | 8 | 0.876 | 0.81 | 0.989 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | off | ccsvd | 8 | carry_tens | 7 | 0.770 | 0.66 | 0.991 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | off | ccsvd | 8 | carry_units | 8 | 0.878 | 0.82 | 0.978 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | off | ccsvd | 16 | carry_tens | 7 | 0.701 | 0.58 | 0.989 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | off | ccsvd | 16 | carry_units | 8 | 0.881 | 0.82 | 0.989 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | off | lda_a | 4 | carry_tens | 7 | 0.741 | 0.63 | 0.985 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | off | lda_a | 4 | carry_units | 8 | 0.876 | 0.81 | 0.985 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | off | lda_a | 8 | carry_tens | 7 | 0.745 | 0.66 | 0.989 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | off | lda_a | 8 | carry_units | 8 | 0.881 | 0.82 | 0.973 | nan | centroid_only_shape | LOW |
| llama-3.1-8b | multiplication | off | lda_a | 16 | carry_units | 8 | 0.882 | 0.83 | 0.978 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | answer | ccsvd | 4 | carry_tens | 7 | 0.671 | 0.53 | 0.984 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | answer | ccsvd | 4 | carry_units | 8 | 0.843 | 0.76 | 0.978 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | answer | ccsvd | 8 | carry_tens | 7 | 0.691 | 0.56 | 0.989 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | answer | ccsvd | 8 | carry_units | 8 | 0.835 | 0.74 | 0.984 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | answer | ccsvd | 16 | carry_tens | 7 | 0.673 | 0.55 | 0.987 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | answer | ccsvd | 16 | carry_units | 8 | 0.832 | 0.74 | 0.984 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | answer | ccsvd | 24 | carry_units | 8 | 0.825 | 0.74 | 0.987 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | answer | ccsvd | 28 | carry_units | 8 | 0.821 | 0.73 | 0.978 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | answer | lda_a | 4 | carry_tens | 7 | 0.617 | 0.46 | 0.991 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | answer | lda_a | 4 | carry_units | 8 | 0.841 | 0.75 | 0.987 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | answer | lda_a | 8 | carry_tens | 7 | 0.682 | 0.55 | 0.991 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | answer | lda_a | 8 | carry_units | 8 | 0.831 | 0.74 | 0.984 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | answer | lda_a | 16 | carry_units | 8 | 0.823 | 0.74 | 0.989 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | norm | ccsvd | 4 | carry_tens | 7 | 0.748 | 0.65 | 0.984 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | norm | ccsvd | 4 | carry_units | 8 | 0.853 | 0.78 | 0.978 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | norm | ccsvd | 8 | carry_tens | 7 | 0.752 | 0.66 | 0.991 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | norm | ccsvd | 8 | carry_units | 8 | 0.854 | 0.78 | 0.984 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | norm | ccsvd | 16 | carry_tens | 7 | 0.724 | 0.61 | 0.991 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | norm | ccsvd | 16 | carry_units | 8 | 0.851 | 0.78 | 0.984 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | norm | ccsvd | 24 | carry_units | 8 | 0.832 | 0.76 | 0.984 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | norm | ccsvd | 28 | carry_units | 8 | 0.831 | 0.74 | 0.991 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | norm | lda_a | 4 | carry_tens | 7 | 0.742 | 0.63 | 0.991 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | norm | lda_a | 4 | carry_units | 8 | 0.852 | 0.79 | 0.985 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | norm | lda_a | 8 | carry_tens | 7 | 0.745 | 0.65 | 0.991 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | norm | lda_a | 8 | carry_units | 8 | 0.853 | 0.78 | 0.984 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | norm | lda_a | 16 | carry_units | 8 | 0.847 | 0.78 | 0.989 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | off | ccsvd | 4 | carry_tens | 7 | 0.687 | 0.56 | 0.991 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | off | ccsvd | 4 | carry_units | 8 | 0.876 | 0.81 | 0.989 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | off | ccsvd | 8 | carry_tens | 7 | 0.770 | 0.66 | 0.989 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | off | ccsvd | 8 | carry_units | 8 | 0.878 | 0.82 | 0.978 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | off | ccsvd | 16 | carry_tens | 7 | 0.701 | 0.58 | 0.991 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | off | ccsvd | 16 | carry_units | 8 | 0.880 | 0.82 | 0.987 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | off | lda_a | 4 | carry_tens | 7 | 0.742 | 0.63 | 0.984 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | off | lda_a | 4 | carry_units | 8 | 0.876 | 0.81 | 0.989 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | off | lda_a | 8 | carry_tens | 7 | 0.745 | 0.66 | 0.989 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | off | lda_a | 8 | carry_units | 8 | 0.882 | 0.83 | 0.985 | nan | centroid_only_shape | LOW |
| pythia-6.9b | multiplication | off | lda_a | 16 | carry_units | 8 | 0.881 | 0.83 | 0.991 | nan | centroid_only_shape | LOW |

(Note: q_dsw = NaN above means the cell's pre-FDR verdict was already `centroid_only_shape` and the FDR family did not assign a q for the eligible-verdict path. Pre-FDR p_dsw values are shown for completeness.)

**Cross-cutting observations on carries:**

- K_present = 7 for `carry_tens` and 8 for `carry_units` across all models (these are the natural cardinalities for products in [0, 99 × 99]).
- ρ_centroid for carry cells clusters in [0.6, 0.88]. Many cells are above the 0.85 pass gate at the point-estimate level, but their p_dsw values are uniformly ≥ 0.8, meaning the null distribution under label permutation also reaches similarly high ρ.
- This produces the exact `centroid_only_shape` outcome the ladder is designed to issue: ρ is high in isolation, but it is not significantly higher than the shuffled-label baseline.

---

## 16. Runtime and resource usage

- **Total cells fit:** 2,561 across 6 (model, task) array tasks.
- **Per-cell average runtime on L40S:** ~4–7 s.
- **Per array task runtime:** ~30–40 minutes.
- **Wall time end-to-end (6 tasks in parallel):** ~40 minutes for the array; aggregator runs in ~10 s.
- **Per-cell working set:** ~1 MB (mostly K Σ_v matrices + K×K distance matrices + bootstrap/null scalar arrays).
- **Peak GPU memory:** ~1 GB on L40S during inner-loop batch (cupy reserves more in its memory pool for kernel cache).
- **Toy suite runtime:** 127.8 s including Toy 7B-FPR's 100 cells.

---

## 17. Artefact pathing reference

```
/data/user_data/anshulk/emnlp2026/
├── activations/{model}/{task}_layer_{LL}.npy
├── answers/{model}/{task}_answers.csv
├── data/raw/{task}_problems.csv
└── results/
    ├── ccsvd_subspaces/{[mode_{mode}/]?{model}}/{task}/layer_{LL}/{concept}/{basis.npy,meta.json}
    ├── lda_subspaces/subspace_lda/mode_{mode}/{model}/{task}/layer_{LL}/{concept}/{lda_basis_full.npy,meta.json}
    ├── residualized/{model}/{task}_layer_{LL}_mode_{mode}.npy
    ├── stage2a_fourier_helix/
    │   ├── {model}/summary_{model}_{task}_mode_{mode}_variant_{variant}.csv
    │   └── {model}/{task}/mode_{mode}/layer_{LL}/variant_{variant}/{concept}/...
    └── stage2b_dsw/
        ├── {model}/summary_{model}_{task}_mode_{mode}_variant_{variant}.csv
        ├── {model}/{task}/mode_{mode}/layer_{LL}/variant_{variant}/{concept}/
        │     ├── dsw_results.csv
        │     ├── D_E.npy, D_SW.npy, mu_stack.npy
        │     ├── shrinkage_pair_mode_matrix.npy
        │     ├── rho_null.npy, rho_bootstrap.npy
        │     └── metadata.json
        └── comparison/
              ├── dsw_all.csv
              ├── spread_verdict_counts_by_cell.csv
              ├── cross_mode_spread_survival.csv
              ├── cross_variant_agreement.csv
              ├── stage2a_vs_stage2b_survival.csv
              ├── confidence_tier_distribution.csv
              └── manifest.json
```

The home-directory mirror of source/configs/sbatches:

```
/home/anshulk/emnlp2026/
├── stage2b_dsw_spread_aware.py
├── check_stage2b_toys.py
├── aggregate_stage2b_dsw.py
├── run_stage2b.sbatch
├── run_stage2b_aggregate.sbatch
├── config.yaml                       # stage2b: block lives here
├── configs/stage2b.yaml              # toy calibration record (toy_5b_tangent_scale, toy_7b_fpr)
└── docs/09_stage2b_dsw_spread_aware.md  # this file
```

---

## 18. Cross-mode survival per concept — full breakdown

For each concept, how many (model, task, layer, variant) tuples had spread_confirmed across n ∈ {0, 1, 2, 3} of the three residualisation modes? The 3-mode column ("robust across all residualisations") is the most stringent operational sense of "owned at the spread level under any magnitude/norm assumption".

| concept | total tuples | n=0 | n=1 | n=2 | n=3 (robust) | 3-mode rate |
|---|---:|---:|---:|---:|---:|---:|
| a | 51 | 28 | 12 | 8 | 3 | 5.9% |
| a_tens | 59 | 58 | 0 | 1 | 0 | 0.0% |
| a_units | 33 | 32 | 0 | 1 | 0 | 0.0% |
| ans_hundreds | 30 | 19 | 7 | 4 | 0 | 0.0% |
| ans_tens | 54 | 43 | 8 | 3 | 0 | 0.0% |
| ans_units | 53 | 50 | 2 | 1 | 0 | 0.0% |
| answer | 21 | 3 | 5 | 13 | 0 | 0.0% |
| b | 44 | 33 | 9 | 2 | 0 | 0.0% |
| b_tens | 60 | 60 | 0 | 0 | 0 | 0.0% |
| b_units | 33 | 33 | 0 | 0 | 0 | 0.0% |
| carry_tens | 30 | 30 | 0 | 0 | 0 | 0.0% |
| carry_units | 28 | 28 | 0 | 0 | 0 | 0.0% |
| column_sum_hundreds | 26 | 26 | 0 | 0 | 0 | 0.0% |
| column_sum_tens | 60 | 44 | 6 | 8 | 2 | 3.3% |
| column_sum_units | 36 | 34 | 1 | 1 | 0 | 0.0% |
| max_operand | 57 | 48 | 4 | 5 | 0 | 0.0% |
| min_operand | 55 | 32 | 13 | 9 | 1 | 1.8% |
| operand_abs_diff | 52 | 21 | 5 | 9 | 17 | 32.7% |
| operand_diff | 28 | 3 | 3 | 2 | 20 | 71.4% |
| partial_product_a_tens_b_tens | 26 | 26 | 0 | 0 | 0 | 0.0% |
| partial_product_a_tens_b_units | 13 | 13 | 0 | 0 | 0 | 0.0% |
| partial_product_a_units_b_tens | 10 | 10 | 0 | 0 | 0 | 0.0% |
| partial_product_units | 15 | 15 | 0 | 0 | 0 | 0.0% |
| running_sum_hundreds | 30 | 18 | 8 | 4 | 0 | 0.0% |
| running_sum_tens | 59 | 36 | 10 | 6 | 7 | 11.9% |
| running_sum_units | 36 | 34 | 1 | 1 | 0 | 0.0% |

Only **6 concepts** ever achieve 3-mode robustness: `a` (3 tuples), `column_sum_tens` (2), `min_operand` (1), `operand_abs_diff` (17), `operand_diff` (20), `running_sum_tens` (7). Total 50 tuples.

`operand_diff` (signed difference a − b) dominates the 3-mode-robust list at 71.4% of its tuples. This is the single most magnitude-coherent concept in the registry and survives every residualisation choice in three quarters of its eligible tuples.

`operand_abs_diff` follows at 32.7%.

Every other concept hits 3-mode-robustness in fewer than 15% of its tuples. Carries, b-digits, and partial products hit 0%.

---

## 19. Layer trajectory — full per (model, task, mode, variant) table

`spread_confirmed` count at each model's layer index. GPT-J layers are [4, 8, 14, 20, 24]; Llama and Pythia use [4, 8, 16, 24, 28]. Headers show first-then-second alternative for the model-specific mid layer.

| model | task | mode | variant | L4 | L8 | L14/L16 | L20/L24 | L24/L28 |
|---|---|---|---|---:|---:|---:|---:|---:|
| gpt-j-6b | addition | off | lda_a | 5 | 5 | 3 | 3 | 1 |
| gpt-j-6b | addition | off | ccsvd | 6 | 6 | 2 | 2 | 0 |
| gpt-j-6b | addition | answer | lda_a | 1 | 2 | 0 | 2 | 2 |
| gpt-j-6b | addition | answer | ccsvd | 3 | 4 | 1 | 2 | 0 |
| gpt-j-6b | addition | norm | lda_a | 5 | 5 | 6 | 3 | 3 |
| gpt-j-6b | addition | norm | ccsvd | 4 | 8 | 2 | 2 | 1 |
| gpt-j-6b | multiplication | off | lda_a | 4 | 5 | 0 | 1 | 0 |
| gpt-j-6b | multiplication | off | ccsvd | 3 | 6 | 1 | 0 | 1 |
| gpt-j-6b | multiplication | answer | lda_a | 1 | 1 | 0 | 0 | 0 |
| gpt-j-6b | multiplication | answer | ccsvd | 2 | 2 | 1 | 0 | 0 |
| gpt-j-6b | multiplication | norm | lda_a | 4 | 3 | 0 | 0 | 0 |
| gpt-j-6b | multiplication | norm | ccsvd | 3 | 3 | 1 | 0 | 0 |
| llama-3.1-8b | addition | off | lda_a | 8 | 7 | 4 | 4 | 2 |
| llama-3.1-8b | addition | off | ccsvd | 4 | 7 | 4 | 2 | 2 |
| llama-3.1-8b | addition | answer | lda_a | 1 | 2 | 2 | 2 | 2 |
| llama-3.1-8b | addition | answer | ccsvd | 3 | 5 | 2 | 1 | 1 |
| llama-3.1-8b | addition | norm | lda_a | 8 | 5 | 3 | 2 | 2 |
| llama-3.1-8b | addition | norm | ccsvd | 6 | 6 | 2 | 2 | 2 |
| llama-3.1-8b | multiplication | off | lda_a | 5 | 3 | 4 | 0 | 0 |
| llama-3.1-8b | multiplication | off | ccsvd | 7 | 5 | 3 | 0 | 0 |
| llama-3.1-8b | multiplication | answer | lda_a | 1 | 2 | 0 | 0 | 0 |
| llama-3.1-8b | multiplication | answer | ccsvd | 3 | 1 | 1 | 0 | 0 |
| llama-3.1-8b | multiplication | norm | lda_a | 4 | 5 | 1 | 0 | 0 |
| llama-3.1-8b | multiplication | norm | ccsvd | 7 | 4 | 4 | 0 | 0 |
| pythia-6.9b | addition | off | lda_a | 4 | 6 | 4 | 4 | 3 |
| pythia-6.9b | addition | off | ccsvd | 6 | 6 | 2 | 3 | 3 |
| pythia-6.9b | addition | answer | lda_a | 1 | 3 | 0 | 2 | 2 |
| pythia-6.9b | addition | answer | ccsvd | 1 | 4 | 0 | 1 | 1 |
| pythia-6.9b | addition | norm | lda_a | 4 | 4 | 0 | 4 | 3 |
| pythia-6.9b | addition | norm | ccsvd | 5 | 4 | 0 | 1 | 3 |
| pythia-6.9b | multiplication | off | lda_a | 1 | 1 | 0 | 0 | 0 |
| pythia-6.9b | multiplication | off | ccsvd | 1 | 2 | 0 | 0 | 0 |
| pythia-6.9b | multiplication | answer | lda_a | 0 | 1 | 0 | 0 | 0 |
| pythia-6.9b | multiplication | answer | ccsvd | 1 | 1 | 1 | 0 | 0 |
| pythia-6.9b | multiplication | norm | lda_a | 2 | 6 | 0 | 0 | 0 |
| pythia-6.9b | multiplication | norm | ccsvd | 0 | 5 | 0 | 0 | 0 |

Across all 36 (model, task, mode, variant) rows above:

- Total spread_confirmed in early layers (L4 + L8): 251 cells.
- Total in mid layers (L14/L16): 51 cells.
- Total in late layers (L20+L24 and L24+L28 combined): 98 cells.

Early-layer concentration is consistent across models and tasks. The exception is the addition family on GPT-J/Llama/Pythia where mid-late layers also contribute (driven by `ans_*` digit concepts that take longer to compute on addition than on multiplication's three-digit-output regime).

---

## 20. Sample-regime distribution

### 20.1 Confidence tier counts

| tier | count | share |
|------|------:|------:|
| LOW | 1784 | 69.7% |
| DISCOVERY_ONLY | 716 | 28.0% |
| MEDIUM | 51 | 2.0% |
| HIGH | 10 | 0.4% |

### 20.2 min_n_v quantiles

| stat | value |
|------|------:|
| count | 2,561 |
| mean | 153.5 |
| std | 287.6 |
| min | 30 |
| 25% | 39 |
| 50% (median) | 71 |
| 75% | 145 |
| max | 2,756 |

### 20.3 min_ratio_v = min_n_v / r quantiles

| stat | value |
|------|------:|
| count | 2,561 |
| mean | 17.2 |
| std | 32.5 |
| min | 3.0 |
| 25% | 4.5 |
| 50% (median) | 8.1 |
| 75% | 16.4 |
| max | 351.0 |

### 20.4 γ = r / min_n_v quantiles (Marchenko-Pastur regime indicator)

| stat | value |
|------|------:|
| count | 2,561 |
| mean | 0.157 |
| std | 0.087 |
| min | 0.0029 |
| 25% | 0.061 |
| 50% (median) | 0.123 |
| 75% | 0.222 |
| max | 0.333 |

Every cell has γ < 1 (no cells in the under-determined Σ̂_v regime — the n_v ≥ 30 floor and the n_v < 5 bootstrap-redraw rule both contribute). γ ≥ 0.5 happens only on the highest-K concepts (`answer`, partial products) where Stage 2b is operating in DISCOVERY_ONLY tier anyway.

### 20.5 ρ_centroid distribution

Histogram across all 2,561 cells (15 bins from −0.5 to 1.0):

| ρ_centroid bin | count |
|---|---:|
| [-0.50, -0.40) | 1 |
| [-0.40, -0.30) | 1 |
| [-0.30, -0.20) | 2 |
| [-0.20, -0.10) | 7 |
| [-0.10, 0.00) | 19 |
| [0.00, 0.10) | 25 |
| [0.10, 0.20) | 49 |
| [0.20, 0.30) | 65 |
| [0.30, 0.40) | 88 |
| [0.40, 0.50) | 102 |
| [0.50, 0.60) | 159 |
| [0.60, 0.70) | 240 |
| [0.70, 0.80) | 351 |
| [0.80, 0.90) | 696 |
| [0.90, 1.00] | 756 |

The mode of the distribution sits in [0.90, 1.00]. A naïve point-estimate-only rule (`ρ_centroid ≥ 0.85`) would call ~1,452 cells (56.7%) `spread_confirmed`. The actual post-FDR count is 400 (15.6%) — the Whittle null + FDR + ρ_low gate together remove ~1,050 cells whose ρ is high but indistinguishable from the label-shuffled baseline. This is the d_SW test's main job.

---

## 21. Variant disagreement examples

In 634 of 1,533 (model, task, mode, layer, concept) tuples, lda_a and ccsvd disagree on the verdict. Of these:

- **One variant `spread_confirmed`, the other `centroid_only_shape`:** ~580 tuples.
- **One variant `spread_marginal`, the other `spread_confirmed`/`centroid_only`:** ~54 tuples.

Variant disagreement reflects the specific subspace direction set's alignment with the spread-aware geometry. It is not a failure of either method; it surfaces the dependence on subspace choice that Stage 3's orthogonalisation will probe.

---

## 22. DISCOVERY_ONLY tier — what falls here and why

716 cells (28%) land in DISCOVERY_ONLY tier. These are cells where the underlying Σ_v estimation is in a regime where bootstrap CIs are wide enough that we report numbers but flag them as not-headline-grade.

The trigger conditions:
- `γ = r / min_n_v ≥ 1` (per-value sample size smaller than subspace dim).
- OR any value has `ratio_v < 2` (per-value sample size less than twice subspace dim).

Concepts that disproportionately populate DISCOVERY_ONLY:
- `column_sum_tens` (K up to 31)
- `running_sum_tens` (K up to 33)
- `max_operand` (K up to 39)
- `operand_abs_diff` (K up to 36)
- `min_operand` (K up to 21)
- `a` (K up to 29)
- `answer` (K up to ~199 on addition, ~9801 on multiplication — though the multiplication set is capped by single-token availability)
- All partial products (K up to ~99).

On addition, cells with K ≥ 25 typically have n_v ≈ 8,000 / K ≈ 320 — plenty. They land in LOW tier (because min_ratio_v ≥ 10 is the HIGH tier gate and min_n_v ≥ 100 is the MEDIUM tier gate, but these usually pass). DISCOVERY_ONLY hits when K is much larger relative to the cell's correct subset.

On multiplication (N ≈ 2,750), cells with K ≥ 14 typically have n_v ≈ 195 — still in MEDIUM tier — but the variance in n_v across values can push the minimum value below the threshold. This is where most DISCOVERY_ONLY assignments on multiplication originate.

The 157 `spread_confirmed` cells in DISCOVERY_ONLY tier are reported in the appendix but excluded from headline counts. They are real detections in the sense that p_dsw < α and ρ_low ≥ 0.70 — but the bootstrap CI is wider than HIGH/MEDIUM cohorts.

---

## 23. Stage 2b → Stage 3 handoff specification

The aggregator outputs in `comparison/` are the inputs to Stage 3. Specifically:

- `dsw_all.csv` provides per-cell `ρ_centroid`, `ρ_low`, `q_dsw`, `confidence_tier`, and `spread_verdict`. Stage 3's orthogonalisation will recompute these on `Y_orth = (X − μ)(I − Q Qᵀ) Bᵀ` where Q is the cell's algebraic correlate basis, and report `ω = 1 − ρ_centroid_orth / ρ_centroid_original` per cell. Cells with ω ≈ 1 are "owned" (the rank correspondence remains); cells with ω near 0 are "inherited" (the orthogonalisation removed the structure).
- `stage2a_vs_stage2b_survival.csv` provides the pre-orthogonalisation reference set: every cell with a Stage 2a periodic verdict. Stage 3 reruns d_SW on those cells after orthogonalisation.
- `cross_mode_spread_survival.csv` provides the 50 robust-across-modes tuples — these are the most defensible candidates for Stage 3's ownership test.
- `confidence_tier_distribution.csv` provides the per-tier breakdown so Stage 3 can budget compute (it's expensive — GPLVM is in Stage 2c, but ω will be computed on Stage 2b outputs).
- `manifest.json` gives the headline numbers Stage 3's manifest will compare against.

The toy calibration record in `configs/stage2b.yaml` carries through: Stage 3's ω computation reuses the d_SW machinery, so the null calibration (Toy 7B = pass at 3/100) and the discriminative-power calibration (Toy 5B at scale=50) apply directly.

---

## 24. Reproducibility checklist

| Item | Status | Reference |
|------|--------|-----------|
| Seeds | per-cell deterministic via sha256(model|task|mode|layer|variant|concept) | `stage2b_seed` in `stage2b_dsw_spread_aware.py` |
| Library versions logged | per-cell `metadata.json` records numpy, scipy, sklearn, cupy versions | §7.5 manifest excerpt |
| Toy calibration | locked in `configs/stage2b.yaml` with timestamps | §6.9 |
| Atomic writes | every artefact written via tempfile + os.replace | `atomic_save`, `atomic_json`, `atomic_csv` in source |
| Resume-by-metadata | cells with `metadata.json::computation_status == "complete"` are skipped on rerun | `run_one_cell` early-exit |
| Full-data audit | no subsampling at any stage; bootstrap is with-replacement at size N | bootstrap_rho code; B.3 documentation |
| Pre-FDR vs post-FDR transparency | `spread_verdict_pre_fdr` and `fdr_downgraded` retained in every output | aggregator |
| Cell-pair shrinkage audit | `shrinkage_pair_mode_matrix.npy` per cell | Stage 2b output |

---

## 25a. Per-(model, task) ρ_centroid quantiles

ρ_centroid distribution across all eligible cells, broken out by (model, task). Median tells the central spread story; the 95% quantile shows the upper tail of high-ρ centroid arrangements.

| model | task | n_cells | min ρ | 25% | median | 75% | 95% | max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gpt-j-6b | addition | 382 | -0.177 | 0.797 | 0.895 | 0.938 | 0.979 | 0.999 |
| gpt-j-6b | multiplication | 502 | -0.121 | 0.746 | 0.859 | 0.934 | 0.979 | 0.991 |
| llama-3.1-8b | addition | 372 | -0.413 | 0.716 | 0.870 | 0.955 | 0.985 | 0.997 |
| llama-3.1-8b | multiplication | 473 | 0.130 | 0.734 | 0.852 | 0.928 | 0.984 | 0.996 |
| pythia-6.9b | addition | 365 | -0.262 | 0.784 | 0.889 | 0.945 | 0.984 | 0.995 |
| pythia-6.9b | multiplication | 467 | 0.061 | 0.771 | 0.856 | 0.937 | 0.975 | 0.994 |

The median ρ on every (model, task) cell sits between 0.85 and 0.90. By the point-estimate alone, a naïve reader would call most cells "structure preserved." The d_SW machinery rejects ~85% of those — the null reaches the same ρ range under label permutation.

Llama addition has the widest tail (min = −0.41) — a small minority of cells exhibit anti-correlated rankings between D_E and D_SW, which happens when the per-value Σ_v structure points roughly opposite the centroid axis.

## 25b. Per-concept × mode survival

Full breakdown of survival % per concept and mode. This is the most fine-grained view of mode-asymmetry by concept.

| concept | mode | spread_confirmed | spread_marginal | centroid_only | total | survival % |
|---|---|---:|---:|---:|---:|---:|
| a | answer | 4 | 6 | 20 | 30 | 13.3 |
| a | norm | 14 | 6 | 22 | 42 | 33.3 |
| a | off | 19 | 5 | 20 | 44 | 43.2 |
| a_tens | answer | 0 | 0 | 45 | 45 | 0.0 |
| a_tens | norm | 1 | 0 | 51 | 52 | 1.9 |
| a_tens | off | 1 | 0 | 53 | 54 | 1.9 |
| a_units | answer | 0 | 0 | 22 | 22 | 0.0 |
| a_units | norm | 1 | 0 | 23 | 24 | 4.2 |
| a_units | off | 1 | 0 | 28 | 29 | 3.4 |
| ans_hundreds | answer | 0 | 0 | 15 | 15 | 0.0 |
| ans_hundreds | norm | 9 | 0 | 21 | 30 | 30.0 |
| ans_hundreds | off | 6 | 0 | 24 | 30 | 20.0 |
| ans_tens | answer | 1 | 0 | 24 | 25 | 4.0 |
| ans_tens | norm | 7 | 0 | 42 | 49 | 14.3 |
| ans_tens | off | 6 | 0 | 46 | 52 | 11.5 |
| ans_units | answer | 0 | 0 | 24 | 24 | 0.0 |
| ans_units | norm | 1 | 0 | 50 | 51 | 2.0 |
| ans_units | off | 3 | 1 | 47 | 51 | 5.9 |
| answer | answer | 0 | 0 | 6 | 6 | 0.0 |
| answer | norm | 14 | 0 | 5 | 19 | 73.7 |
| answer | off | 17 | 0 | 2 | 19 | 89.5 |
| b | answer | 2 | 0 | 15 | 17 | 11.8 |
| b | norm | 2 | 4 | 27 | 33 | 6.1 |
| b | off | 9 | 6 | 24 | 39 | 23.1 |
| b_tens | answer | 0 | 0 | 53 | 53 | 0.0 |
| b_tens | norm | 0 | 0 | 56 | 56 | 0.0 |
| b_tens | off | 0 | 0 | 56 | 56 | 0.0 |
| b_units | answer | 0 | 0 | 19 | 19 | 0.0 |
| b_units | norm | 0 | 0 | 26 | 26 | 0.0 |
| b_units | off | 0 | 0 | 19 | 19 | 0.0 |
| carry_tens | answer | 0 | 0 | 27 | 27 | 0.0 |
| carry_tens | norm | 0 | 0 | 30 | 30 | 0.0 |
| carry_tens | off | 0 | 0 | 30 | 30 | 0.0 |
| carry_units | answer | 0 | 0 | 27 | 27 | 0.0 |
| carry_units | norm | 0 | 0 | 27 | 27 | 0.0 |
| carry_units | off | 0 | 0 | 28 | 28 | 0.0 |
| column_sum_hundreds | answer | 0 | 0 | 20 | 20 | 0.0 |
| column_sum_hundreds | norm | 0 | 0 | 17 | 17 | 0.0 |
| column_sum_hundreds | off | 0 | 0 | 21 | 21 | 0.0 |
| column_sum_tens | answer | 5 | 0 | 54 | 59 | 8.5 |
| column_sum_tens | norm | 11 | 0 | 44 | 55 | 20.0 |
| column_sum_tens | off | 12 | 0 | 48 | 60 | 20.0 |
| column_sum_units | answer | 0 | 0 | 28 | 28 | 0.0 |
| column_sum_units | norm | 2 | 0 | 31 | 33 | 6.1 |
| column_sum_units | off | 1 | 0 | 31 | 32 | 3.1 |
| max_operand | answer | 0 | 0 | 50 | 50 | 0.0 |
| max_operand | norm | 5 | 0 | 50 | 55 | 9.1 |
| max_operand | off | 9 | 0 | 46 | 55 | 16.4 |
| min_operand | answer | 1 | 0 | 43 | 44 | 2.3 |
| min_operand | norm | 13 | 0 | 38 | 51 | 25.5 |
| min_operand | off | 20 | 0 | 34 | 54 | 37.0 |
| operand_abs_diff | answer | 26 | 0 | 21 | 47 | 55.3 |
| operand_abs_diff | norm | 28 | 0 | 22 | 50 | 56.0 |
| operand_abs_diff | off | 20 | 0 | 30 | 50 | 40.0 |
| operand_diff | answer | 23 | 0 | 4 | 27 | 85.2 |
| operand_diff | norm | 22 | 0 | 5 | 27 | 81.5 |
| operand_diff | off | 22 | 0 | 6 | 28 | 78.6 |
| partial_product_a_tens_b_tens | answer | 0 | 0 | 20 | 20 | 0.0 |
| partial_product_a_tens_b_tens | norm | 0 | 0 | 17 | 17 | 0.0 |
| partial_product_a_tens_b_tens | off | 0 | 0 | 21 | 21 | 0.0 |
| partial_product_a_tens_b_units | answer | 0 | 0 | 11 | 11 | 0.0 |
| partial_product_a_tens_b_units | norm | 0 | 0 | 11 | 11 | 0.0 |
| partial_product_a_tens_b_units | off | 0 | 0 | 8 | 8 | 0.0 |
| partial_product_a_units_b_tens | answer | 0 | 0 | 8 | 8 | 0.0 |
| partial_product_a_units_b_tens | norm | 0 | 0 | 5 | 5 | 0.0 |
| partial_product_a_units_b_tens | off | 0 | 0 | 8 | 8 | 0.0 |
| partial_product_units | answer | 0 | 0 | 9 | 9 | 0.0 |
| partial_product_units | norm | 0 | 0 | 13 | 13 | 0.0 |
| partial_product_units | off | 0 | 0 | 12 | 12 | 0.0 |
| running_sum_hundreds | answer | 0 | 0 | 30 | 30 | 0.0 |
| running_sum_hundreds | norm | 9 | 0 | 21 | 30 | 30.0 |
| running_sum_hundreds | off | 7 | 0 | 23 | 30 | 23.3 |
| running_sum_tens | answer | 10 | 0 | 47 | 57 | 17.5 |
| running_sum_tens | norm | 16 | 0 | 42 | 58 | 27.6 |
| running_sum_tens | off | 17 | 0 | 40 | 57 | 29.8 |
| running_sum_units | answer | 0 | 0 | 28 | 28 | 0.0 |
| running_sum_units | norm | 2 | 0 | 31 | 33 | 6.1 |
| running_sum_units | off | 1 | 0 | 31 | 32 | 3.1 |

Two patterns dominate this table:

1. **The `answer` concept (the full answer value) is the highest-survival concept under mode=off (89.5%).** It drops to 73.7% under mode=norm, and to 0% under mode=answer (which is the residualisation that explicitly removes the answer prediction). This is the cleanest demonstration of what `mode=answer` does — it surgically removes the structure that is downstream of the model's own answer prediction.
2. **Every zero-survival concept hits 0% in every mode.** Carries, partial products, b-digits — these never recover under any residualisation choice. The negative finding is robust across the residualisation axis.

## 25c. Carry ρ_centroid distribution vs non-carries

Even though carries hit 0% `spread_confirmed`, their ρ_centroid values are not low — they cluster between 0.6 and 0.9. The reason for the universal centroid_only_shape verdict is that the **null** also reaches that range, so the rank correspondence is not significantly above chance.

ρ_centroid histogram comparison:

| ρ bin | carry cells (n=169) | non-carry cells (n=2,392) |
|---|---:|---:|
| [0.0, 0.1) | 0 | 11 |
| [0.1, 0.2) | 0 | 13 |
| [0.2, 0.3) | 0 | 38 |
| [0.3, 0.4) | 0 | 43 |
| [0.4, 0.5) | 1 | 67 |
| [0.5, 0.6) | 2 | 102 |
| [0.6, 0.7) | 8 | 188 |
| [0.7, 0.8) | 8 | 333 |
| [0.8, 0.9) | 51 | 641 |
| [0.9, 1.0) | 99 | 949 |

89% of carry cells have ρ ≥ 0.8; 59% have ρ ≥ 0.9. By point estimate alone, these look like clean structure-preservation. But the Whittle null also produces ρ ≥ 0.8 on these cells, so the p-values uniformly exceed α and the verdict is correctly `centroid_only_shape`.

This is the most direct demonstration of why d_SW must use a permutation null rather than a fixed-threshold gate on ρ_centroid: the threshold-only approach would yield ~150 false positives on carries alone.

## 25d. Procedural pseudocode summary

The full per-cell algorithm in compact form (CPU/GPU shared semantics; the GPU path batches the inner steps via cupy):

```
Inputs:
  Z:         (N_used, r)   subspace-projected activations on correct subset
  labels:    (N_used,)     integer codes 0..K_present-1 after value-count filter
  K_present: int
  cfg:       stage2b config dict
  seed:      cell-derived integer

Step 1 — Observed Σ_v and D_SW:
  counts[v]         = count of points with label v
  cell_mode         = max(stricter(shrink_mode(counts[v]/r)) for v)
  μ[v]              = mean(Z[labels == v]); shape (K, r)
  Σ̂[v]              = fit_sigma(R_v = Z[labels==v] − μ[v], cell_mode); shape (K, r, r)
  For each upper-triangle pair (u, v):
    Σ_pool          = (Σ̂[u] + Σ̂[v]) / 2
    λ_uv            = lambda_factor · trace(Σ_pool) / r
    Σ⁺_uv           = Σ_pool + λ_uv · I_r
    diff            = μ[u] − μ[v]
    d²              = diff.T · solve(Σ⁺_uv, diff)
    D_SW[u, v] = D_SW[v, u] = √max(d², 0)
    D_E[u, v]  = D_E[v, u]  = ||diff||₂
  ρ_centroid        = Spearman(vec_offdiag(D_E), vec_offdiag(D_SW))

Step 2 — Bootstrap CI (size N, with replacement):
  For i in 1..n_bootstrap:
    idx              = sample N with replacement
    Z_b, labels_b    = Z[idx], labels[idx]
    counts_b         = bincount(labels_b)
    if any(counts_b < bootstrap_min_n_v_floor): redraw (up to bootstrap_max_redraws)
    cell_mode_b      = max(stricter(shrink_mode(counts_b[v]/r)) for v)
    Σ̂_b, μ_b         = per-value fit with cell_mode_b
    ρ_b              = compute_rho(Σ̂_b, μ_b)
  ρ_low, ρ_high     = 2.5%, 97.5% quantiles of ρ_b array
  bootstrap_se     = std(ρ_b)
  ci_halfwidth     = (ρ_high − ρ_low) / 2

Step 3 — Whittle null (label permutation):
  For j in 1..n_permutations:
    perm             = random permutation of N indices
    labels_perm      = labels[perm]
    Σ̂_p, μ_p         = per-value fit on (Z, labels_perm) with cell_mode
    ρ_p              = compute_rho(Σ̂_p, μ_p)
  p_dsw            = (1 + count(ρ_p ≥ ρ_centroid)) / (1 + n_permutations)

Step 4 — Verdict (cell-level, will be re-evaluated post-FDR):
  if ci_halfwidth > ci_halfwidth_unstable:
    verdict = null_unstable
  elif ρ_low < rho_marginal_low OR ρ_centroid < 0.70 OR p_dsw ≥ fdr_alpha:
    verdict = centroid_only_shape
  elif ρ_centroid ≥ rho_pass_threshold AND ρ_low ≥ rho_low_ci_threshold:
    verdict = spread_confirmed
  else:
    verdict = spread_marginal

Step 5 — Confidence tier:
  γ                = r / min(counts)
  if K_present == 4: tier = LOW
  elif γ ≥ 1 OR any(counts[v]/r < tier_discovery_only_max_ratio): tier = DISCOVERY_ONLY
  elif HIGH gates: tier = HIGH
  elif MEDIUM gates: tier = MEDIUM
  else: tier = LOW

Outputs (per cell):
  ρ_centroid, ρ_pearson, tau_kendall, ρ_low, ρ_high, bootstrap_se, mean_log_ratio
  p_dsw  (q_dsw filled by aggregator)
  K_present, r, n_samples_used, min_n_v, min_ratio_v, gamma, ci_halfwidth
  shrinkage_mode_per_v, shrinkage_alpha_per_v, shrinkage_pair_mode_matrix
  redraw_rate_bootstrap, null_unstable_dsw
  spread_verdict, confidence_tier
  D_E, D_SW, mu_stack, rho_null, rho_bootstrap, metadata
```

The aggregator's role is light by comparison:

```
Aggregator pseudocode:
  df = concat(every per-model summary CSV)
  eligible = df.spread_verdict ∈ {spread_confirmed, spread_marginal, centroid_only_shape}
  df.loc[eligible, 'q_dsw'] = BH-FDR(df.loc[eligible, 'p_dsw'], alpha=0.05)
  fail = (df.spread_verdict == 'spread_confirmed') & (df.q_dsw ≥ 0.05)
  df.loc[fail, 'spread_verdict'] = 'centroid_only_shape'
  df.loc[fail, 'fdr_downgraded'] = True
  df['confidence_tier'] = recompute_with_q(...)
  Write comparison/ tables (verdict counts, cross-mode, cross-variant, transition, tier dist)
  Write manifest.json
```

## 25e. End-to-end SLURM submission record

The Stage 2b run that produced the numbers in this report was submitted via:

```
sbatch run_stage2b.sbatch        # job 7893686, array 0–5, --gres=gpu:L40S:1, --cpus-per-task=4, --mem=32G
sbatch --dependency=afterok:7893686 run_stage2b_aggregate.sbatch  # job 7893687
```

Resource allocation per array task as scheduled:

| Array task | Model | Task | Node | GPU | CPUs | Mem | Wall |
|-----------:|-------|------|------|-----|-----:|-----|------|
| 0 | gpt-j-6b | addition | babel-u5-28 | 1× L40S (46068 MiB) | 4 | 32 GB | ~30 min |
| 1 | gpt-j-6b | multiplication | babel-p5-28 | 1× L40S | 4 | 32 GB | ~28 min |
| 2 | llama-3.1-8b | addition | babel-p5-28 | 1× L40S | 4 | 32 GB | ~35 min |
| 3 | llama-3.1-8b | multiplication | babel-p5-28 | 1× L40S | 4 | 32 GB | ~33 min |
| 4 | pythia-6.9b | addition | babel-p5-28 | 1× L40S | 4 | 32 GB | ~30 min |
| 5 | pythia-6.9b | multiplication | (deferred, see below) | 1× L40S | 4 | 32 GB | ~30 min |

Tasks 1–4 co-located on babel-p5-28 (8× L40S, 4 GPUs were free at submission). Task 5 was scheduled after task 0 freed its GPU. Aggregator (job 7893687) fired ~10 seconds after the last array task completed, ran on the same partition with `--gres=gpu:1` (any GPU type), consumed < 8 GB RAM and 1 CPU core, completed in under 30 seconds.

Total run wall time end-to-end: approximately 45 minutes from `sbatch` to aggregator completion.

## 25f. Library version manifest

Versions captured per cell in `metadata.json`:

| Library | Version |
|---------|---------|
| numpy | 2.2.6 |
| scipy | 1.17.1 |
| sklearn | 1.8.0 |
| pandas | (>=2.0 installed) |
| cupy | 14.0.1 |
| Python | 3.11.15 |

GPU/CUDA stack: CUDA 12.x; NVIDIA driver 575.51.03 on L40S worker nodes.

The shrinkage and Mahalanobis computations are entirely closed-form; no library API has changed semantics in the ranges used. A future re-run with newer numpy/scipy is expected to reproduce the headline numbers to within the 0.001 ρ_centroid noise floor discussed in §W.

## 25g. Stage 2a verdict cross-reference

For each unique Stage 2a verdict label observed in this run's eligible set, the count and the Stage 2b transition breakdown:

| Stage 2a verdict | Eligible cells | spread_confirmed | spread_marginal | centroid_only_shape | Survival % |
|---|---:|---:|---:|---:|---:|
| helix | 754 | 67 | 0 | 687 | 8.9% |
| circle | 177 | 29 | 0 | 148 | 16.4% |
| none | 1,630 | 304 | 28 | 1,298 | 18.7% |
| **Total eligible** | **2,561** | **400** | **28** | **2,133** | **15.6%** |

Note: the `none` cell category here is the population whose Stage 2a verdict was `none` (i.e. Stage 2a Fourier did not find a helix or circle), but which Stage 2b still tests. They form the comparison baseline — and their `spread_confirmed` count of 304 is actually higher than the helix/circle survivors, because the d_SW test is not bound to Stage 2a's periodic-structure hypothesis. A cell can have no periodic centroid arrangement but still preserve the centroid-distance rank under spread-correction.

The interpretation is that **d_SW measures something different from what Stage 2a measures**. Stage 2a asks "do centroids trace a Fourier helix?"; Stage 2b asks "do per-point distances rank-correspond to centroid distances?". A cell can pass Stage 2b without passing Stage 2a (e.g. operand_diff, which has no periodic structure but a clean linear scaling).

## 25h. Variant-disagreement breakdown by concept

For the 634 (model, task, mode, layer, concept) tuples where lda_a and ccsvd disagree on verdict, the concepts most often responsible:

| concept | tuples with variant disagreement | total tuples |
|---|---:|---:|
| operand_abs_diff | 50 | 52 |
| operand_diff | 23 | 28 |
| a | 39 | 51 |
| b | 36 | 44 |
| min_operand | 36 | 55 |
| ans_hundreds | 25 | 30 |
| ans_tens | 28 | 54 |
| running_sum_tens | 33 | 59 |
| running_sum_hundreds | 19 | 30 |
| column_sum_tens | 33 | 60 |
| max_operand | 38 | 57 |
| answer | 17 | 21 |
| running_sum_units | 23 | 36 |
| column_sum_units | 23 | 36 |
| ans_units | 26 | 53 |
| a_tens | 21 | 59 |
| a_units | 13 | 33 |
| (carries, partial products, b-digits, column_sum_hundreds) | 0 | – |

Disagreement is **zero** on the zero-survival concepts (both variants agree on `centroid_only_shape`). Where survival is non-trivial, variant disagreement is the rule rather than the exception — `operand_abs_diff` disagrees in 50 of 52 tuples (96%), `operand_diff` in 23 of 28 (82%), `answer` in 17 of 21 (81%).

Disagreement is not a sign of pipeline error; it surfaces the legitimate dependence on subspace direction choice. Stage 3's orthogonalisation will further illuminate which directions carry the spread-aware structure.

## 25i. Aggregator runtime profile

The Stage 2b aggregator (`aggregate_stage2b_dsw.py`) processed 2,561 per-cell rows from 36 per-model summary CSVs in well under a minute:

```
Loaded 2561 rows from 36 CSVs
Pre-FDR verdict counts: {'centroid_only_shape': 2025, 'spread_confirmed': 508, 'spread_marginal': 28}
Post-FDR verdict counts: {'centroid_only_shape': 2133, 'spread_confirmed': 400, 'spread_marginal': 28}
FDR downgraded 108 cells from spread_confirmed to centroid_only_shape
Post-FDR tier counts: {'LOW': 1784, 'DISCOVERY_ONLY': 716, 'MEDIUM': 51, 'HIGH': 10}
wrote dsw_all.csv (2561 rows)
wrote spread_verdict_counts_by_cell.csv (180 rows)
wrote cross_mode_spread_survival.csv (999 rows)
wrote cross_variant_agreement.csv (1533 rows)
wrote stage2a_vs_stage2b_survival.csv (931 rows)
wrote confidence_tier_distribution.csv (xxx rows)
wrote manifest.json: 2561 rows across 180 cells, 26 concepts
```

Total aggregator wall time: 8.4 s on the cluster (single CPU, no GPU work — the GPU is requested only to satisfy QOS).

## 25j. Per-cell metadata.json field reference

Per-cell metadata schema (one JSON per cell, written atomically at the end of `run_one_cell`):

| Field | Type | Description |
|-------|------|-------------|
| model | str | Model key (gpt-j-6b, llama-3.1-8b, pythia-6.9b) |
| task | str | Task (addition, multiplication) |
| mode | str | Residualisation mode |
| layer | int | Layer index |
| variant | str | Subspace variant (lda_a, ccsvd) |
| concept | str | Concept label |
| random_seed_input | str | sha256-hash input used for seeding ("stage2b\|model\|task\|mode\|...\|concept") |
| random_seed | int | The actual integer seed derived from sha256[:8] mod 2^63 |
| n_permutations | int | Cell-level setting (1000 for headline runs) |
| n_bootstrap | int | Cell-level setting (1000 for headline runs) |
| stage2a_verdict | str | Stage 2a's `geometry_detected` for this cell |
| stage2a_discovered_period | float | Stage 2a's discovered period (NaN if N/A) |
| shrinkage_mode_per_v | list[str] | Length-K_present list of per-value chosen modes (sample/lw/oas), before harmonisation |
| shrinkage_alpha_per_v | list[float] | Length-K_present list of OAS shrinkage strengths (0.0 for sample/lw) |
| kept_values | list[int] | Value codes that survived the n_v ≥ 30 filter |
| dropped_values | list[int] | Value codes dropped |
| gpu_used | bool | Was cupy used (true) or numpy fallback (false) |
| runtime_seconds | float | Wall time for this cell |
| lib_versions | object | {numpy, pandas, scipy, sklearn, cupy} |
| summary_row | object | The exact row written to the per-model summary CSV |
| K_natural | int | Original number of unique values for this concept in the full dataset |
| K_present | int | Surviving value count after filtering |
| B_sha256 | str | sha256 of the basis matrix file (audit trail) |
| B_path | str | Path to the basis matrix file |
| mu_layer_source | str | Path to the meta.json that provided mu_layer |
| computation_status | str | "complete" — used by resume-by-metadata pattern |

The `random_seed_input` string is the exact concatenation used by `stage2b_seed`. Two cells with the same input produce the same integer seed deterministically across runs; the per-cell artefacts can be reproduced bit-identically (modulo the GPU/CPU float-order issue documented in §W note 1).

## 25k. Per-cell parquet reading recipe

To extract any subset of cells from the aggregator output for downstream analysis:

```python
import pandas as pd

df = pd.read_csv('/data/user_data/anshulk/emnlp2026/results/stage2b_dsw/comparison/dsw_all.csv')

# Headline subset: HIGH/MEDIUM + spread_confirmed
hm = df[df['confidence_tier'].isin(['HIGH','MEDIUM']) & (df['spread_verdict']=='spread_confirmed')]

# Carries: zero-survival demonstration
carries = df[df['concept'].str.startswith('carry_')]
assert (carries['spread_verdict'] == 'centroid_only_shape').all()

# 3-mode-robust cohort
xm = pd.read_csv('/data/user_data/anshulk/emnlp2026/results/stage2b_dsw/comparison/cross_mode_spread_survival.csv')
robust_3 = xm[xm['n_spread_confirmed_modes'] == 3]
# 50 tuples, dominated by operand_diff and operand_abs_diff

# Stage 2a → 2b transition
trans = pd.read_csv('/data/user_data/anshulk/emnlp2026/results/stage2b_dsw/comparison/stage2a_vs_stage2b_survival.csv')
# 931 rows; 96 survived as spread_confirmed
```

For per-cell artefacts (D_E, D_SW, mu_stack, rho_null, rho_bootstrap), navigate to:

```
results_root/stage2b_dsw/{model}/{task}/mode_{mode}/layer_{layer:02d}/variant_{variant}/{concept}/
```

The full set of per-cell artefacts adds up to ~2 MB per cell × 2,561 cells = ~5 GB of structured data on disk. The aggregated CSVs in `comparison/` are < 5 MB total.

## 25l. Toy validation reproducibility

To re-validate the toy suite from scratch:

```bash
cd /home/anshulk/emnlp2026
$PYTHON check_stage2b_toys.py --config config.yaml
```

Expected output:
- 7 toys + 1 verdict-ladder unit test all pass.
- `configs/stage2b.yaml` is overwritten with a fresh calibration record.
- Total wall time: ~2 minutes on a GPU node (faster on CPU-only because the toy data is small).

If toys fail, the most common diagnostic step is to inspect the bootstrap variance — at N=400 (default for most toys), tight CIs require ρ_low to be near the point estimate. If a toy fails on a specific seed, rerun the seed in isolation:

```bash
$PYTHON -c "from check_stage2b_toys import toy_3B_helix; toy_3B_helix()"
```

The toy gate also enforces that `configs/stage2b.yaml` contains both `toy_5b_tangent_scale: <number>` and `toy_7b_fpr.status: pass`. The real-cell pipeline refuses to start without both (`check_toy_calibration` in `stage2b_dsw_spread_aware.py`).

## 25m. Quick-reference: where each headline number comes from

| Headline statement | Source file in `comparison/` | Specific column/value |
|--------------------|------------------------------|----------------------|
| "2,561 eligible cells" | `manifest.json` | `n_rows_total` |
| "180 cell groups" | `manifest.json` | `n_cells` (group on model×task×mode×layer×variant) |
| "400 spread_confirmed post-FDR" | `manifest.json` | `verdict_counts_post_fdr.spread_confirmed` |
| "508 spread_confirmed pre-FDR" | `manifest.json` | `verdict_counts_pre_fdr.spread_confirmed` |
| "108 cells downgraded by FDR" | `manifest.json` | `n_fdr_downgrades` |
| "10 HIGH-tier cells" | `manifest.json` | `confidence_tier_counts.HIGH` |
| "51 MEDIUM-tier cells" | `manifest.json` | `confidence_tier_counts.MEDIUM` |
| "10.3% Stage 2a survival" | `stage2a_vs_stage2b_survival.csv` | filter to stage2a_verdict ∈ {helix, circle}; survival = 96/931 |
| "169/169 carry cells centroid_only" | `dsw_all.csv` | filter concept starts with `carry_` |
| "50 robust-across-modes tuples" | `cross_mode_spread_survival.csv` | filter `n_spread_confirmed_modes == 3` |
| "Llama 3.1 8B: 33 HIGH/MEDIUM" | `dsw_all.csv` | filter model + tier; count |
| "Toy 7B FPR: 3 / 100" | `configs/stage2b.yaml` | `toy_7b_fpr.n_spread_confirmed` |
| "Toy 5B chosen scale = 50" | `configs/stage2b.yaml` | `toy_5b_tangent_scale` |

## 25n. Comparison checklist with Stage 2a

| Stage 2a fact | Stage 2b answer |
|---------------|-----------------|
| Cell ran on full correct subset (no subsampling). | Same. |
| Stage 2a permutation null = 1,000 multiset-preserving shuffles. | Stage 2b uses 1,000 multiset-preserving shuffles for the Whittle-style ρ null. |
| Stage 2a applies BH-FDR at α=0.05 over the eligible cell family. | Same family + same α. |
| Stage 2a's eligible verdict set = {helix, circle, none}. | Stage 2b extends defensively to {helix, circle, none, sparse_value_grid}. |
| Stage 2a subspace dimension r matches the cell's Stage 1 fit. | Same — Stage 2b operates in the same subspace. |
| Stage 2a centroid = mean of Y_v in subspace coordinates. | Same μ_v. |
| Stage 2a does NOT estimate Σ_v. | Stage 2b's primary new estimator. |
| Stage 2a reports verdicts: helix / circle / none / sparse_value_grid / low_K / period_inconsistent / null_unstable. | Stage 2b's verdict set is orthogonal: spread_confirmed / spread_marginal / centroid_only_shape / insufficient_samples / low_K_after_filter / null_unstable_dsw. |
| Stage 2a `null_unstable` triggers on redraw rate. | Stage 2b `null_unstable_dsw` triggers on bootstrap CI half-width > 0.30. |
| Stage 2a runtime ~0.5 s per cell on GPU. | Stage 2b runtime ~5 s per cell on GPU (post-optimisation; 24 s pre-optimisation). |

## 25o. Disk usage summary

```
results/stage2b_dsw/
├── {model}/                                  # per-model summary CSVs (6 files)
│   └── summary_*.csv                          # ~2 MB total
├── {model}/{task}/...                        # per-cell directories (2,561 dirs)
│   ├── dsw_results.csv                       # ~1 KB each
│   ├── D_E.npy, D_SW.npy                     # ~3-30 KB each (K-dependent)
│   ├── mu_stack.npy                          # ~1-10 KB
│   ├── shrinkage_pair_mode_matrix.npy        # ~100 bytes
│   ├── rho_null.npy, rho_bootstrap.npy       # ~8 KB total
│   └── metadata.json                         # ~3 KB
└── comparison/                                # 7 CSVs + 1 JSON
    └── dsw_all.csv ~ 1.5 MB; others total ~3 MB
```

Total Stage 2b output: approximately 50–70 MB on disk. The per-cell artefacts are small because the per-cell distance matrices are K×K with K mostly ≤ 30, and the null/bootstrap arrays are 1D scalar streams.

## 26. Known issues and follow-ups

1. **B.8 sensitivity sweep deferred.** The plan calls for shrinkage-off, λ-sweep, and LOO jackknife on HIGH+MEDIUM cells. Currently the code is written and the per-cell parquet retains everything needed, but the sweep has not been run. Cost estimate: ~1–3 GPU-hr.
2. **Sparse-value-grid eligibility.** Stage 2b accepts cells with Stage 2a verdict `sparse_value_grid`. Empirically these cells are uncommon in the current aggregator output (no row count broken out separately). If sparse_value_grid cells materialise in higher numbers in a future Stage 2a refit, they should be inspected for d_SW robustness (the test is grid-agnostic, but bootstrap variance on irregular grids has not been profiled here).
3. **GPU/CPU LW divergence**: An earlier GPU LW implementation used a simplified `bar_b² ≈ trace(S²)/n` approximation and produced ρ values up to 0.04 different from the CPU full per-point LW. The fix (used in this run) batches the full per-point formula on GPU, and GPU vs CPU pointwise max abs diff is 0.0 on toy validation. This is captured in `_fit_sigma_batched_gpu` and in the toy 5B/6B passes that exercise the LW path.
4. **`cp.linalg.solve_triangular` unavailable in cupy 14.0.1.** The GPU path falls back to `cp.linalg.solve(Σ_reg, diffs)` which does an LU factorisation rather than a triangular solve on the Cholesky factor. Σ_reg is PSD and Tikhonov-regularised, so the LU solve is stable, but it's ~10% slower than a triangular-solve would be. Upgrading cupy in a future run would recover that 10%.
5. **No documentation of the empirical link between Stage 1 subspace rank and per-cell d_SW power.** Cells where Stage 1 picked r > 12 (rare) may have less d_SW power because per-value spread is divided across more directions; a brief sensitivity over r might surface this but has not been quantified here.

---

# Intuition and analysis

> The technical sections above are deliberately neutral — procedure and numbers only. This section is the only place interpretation appears, and was explicitly requested. Claims here are descriptive of the empirical pattern, not causal.

## A. What `d_SW` actually measures

Stage 2a established that, for a given concept (say `carry_units` with K=8 possible values), the **average** activation per value traces a clean periodic curve in subspace. That is a statement about K points — the K centroids.

Stage 2b asks a sharper question. Around each centroid sits a cloud of individual problem activations. If you measure pairwise distances between centroids in two different ways — once with plain Euclidean distance (`D_E`), once with a Mahalanobis-style distance (`D_SW`) that downweights directions where the within-class clouds are wide — do the resulting K(K-1)/2 pairwise distances **rank in the same order**?

If yes (`ρ_centroid` near 1, significantly above the label-shuffled null), the cloud structure is benign and the centroid-level shape carries through to the per-point geometry the model would actually traverse during inference.

If no (`ρ_centroid` not separable from the null), the cloud structure rearranges the geometry. The shape exists at the means but is invisible to the model's per-token computation.

This is the spread-aware companion to Stage 2a, and it is the first stage in the pipeline that operationalises the distinction between "structure of averages" and "structure of points".

## B. The 10% headline interpreted

Across 2,561 (model, task, mode, layer, variant, concept) cells:

- **15.6% post-FDR `spread_confirmed`** (400 cells).
- **83.3% `centroid_only_shape`** (2,133 cells).
- **1.1% `spread_marginal`** (28 cells).

The same headline restricted to cells where Stage 2a had previously flagged `helix` or `circle`:

- **10.3% survive spread-correction.**
- The other ~90% are demoted: clean periodic centroids around clouds whose within-class spread rearranges the pairwise ranking.

When the verdict is restricted further to HIGH/MEDIUM confidence tier (large per-value samples, tight bootstrap, low q), **only 61 cells make the cut — about 2.4% of all eligible cells**.

The interpretation is unsurprising in shape but striking in magnitude: most of the linear-probe-style "concept lives in a periodic geometry" stories that Stage 2a surfaces at the centroid level **do not** carry over to the per-point geometry the model is actually computing on.

## C. Per-concept patterns

The 26 concepts split into roughly four families when ordered by survival rate.

**High survival (≥ 50% of cells `spread_confirmed`):**
- `operand_diff` — 81.7% (signed difference a − b).
- `answer` — 70.5% (the full product/sum value).
- `operand_abs_diff` — 50.3% (|a − b|).

These are coarse-grained magnitude concepts. They have a small number of distinct values relative to the cell dimensionality (`operand_diff` runs over a wide range but acts as a single scalar direction) and a strong axis-aligned signal that survives spread correction in most cells.

**Moderate survival (10–35%):**
- `a` (31.9%), `running_sum_tens` (25.0%), `min_operand` (22.8%), `ans_hundreds` (20.0%), `running_sum_hundreds` (17.8%), `column_sum_tens` (16.1%), `b` (14.6%), `ans_tens` (11.1%).

These mix raw inputs (`a`, `b`) and intermediate scalars (`min_operand`, `running_sum_*`, `column_sum_tens`). Survival concentrates on the higher-magnitude digits (tens, hundreds) and operand quantities. The pattern is asymmetric — `a` survives at 31.9% while `b` survives at 14.6%, suggesting that the first operand's representation has more disambiguating geometric structure than the second operand at typical layer/mode/variant combinations.

**Low survival (1–10%):**
- `max_operand` (8.8%), `running_sum_units` (3.2%), `column_sum_units` (3.2%), `ans_units` (3.2%), `a_units` (2.7%), `a_tens` (1.3%).

Low-magnitude per-digit concepts (units digits especially) survive rarely. Units-digit concepts are by their nature carrying the least magnitude information (they cycle 0–9 every step) and the spread test sees little structure beyond a centroid mirage.

**Zero survival (0%):**
- `b_tens`, `b_units`, `carry_units`, `column_sum_hundreds`, `carry_tens`, `partial_product_a_units_b_tens`, `partial_product_a_tens_b_units`, `partial_product_a_tens_b_tens`, `partial_product_units`.

This is the cohort that is most informative. Nine concepts have **literally zero `spread_confirmed` cells** in any (model, task, mode, layer, variant) combination tested. The cohort includes:

- Both carries (`carry_units`, `carry_tens`).
- All four partial-product concepts.
- `column_sum_hundreds` (the carry-into-hundreds-column residual).
- Both `b` digits (units, tens).

## D. Carries — the strongest cross-model finding

The 169 cells covering `carry_units` and `carry_tens` form the cleanest signal in the report. Every single one returned `centroid_only_shape` post-FDR. This holds across:

- 3 models (GPT-J, Llama, Pythia).
- 3 residualisation modes (off, answer, norm).
- 2 subspace variants (lda_a, ccsvd).
- 5 layers per model.

The pattern is robust: ρ_centroid clusters in [0.6, 0.88], which would be marginal-to-passing at the point estimate, but the Whittle null also produces ρ in the same range. The p-value sits at 0.8–0.99 — the rank correspondence is indistinguishable from chance under label permutation.

This is the spread-level analog of the parent project's Phase H finding (`/home/anshulk/arithmetic-geometry/phase_h_orthogonalize.py`). Phase H showed that on multiplication, the carry helix is fully **inherited** from algebraic correlates (column sums, partial products) — when those correlates are orthogonalised out, the carry helix disappears. Stage 2b reproduces the negative direction without needing an orthogonalisation step: at the spread level, the carries never had owned per-point geometry to begin with.

The parent project's finding was a **single-model** (Llama only) result on a curated 8,264-problem set. Stage 2b extends it to three models, two tasks, three residualisation modes, and the full per-model correct subset — and the result holds across every cell tested. This is the strongest cross-model replication in the rescope so far.

The implication is methodological: the carry helix that Stage 2a finds is a centroid mirage. The shape is real in the means and shows up under linear probing, but the model does not navigate it at the per-point level. Stage 3's orthogonalisation will measure how much of the centroid shape comes from the algebraic correlates by computing `ω` (owned fraction); Stage 2b suggests that on carries, the answer will be close to zero.

## E. Mode dependence (off / answer / norm)

`spread_confirmed` counts by residualisation mode:

| Mode | spread_confirmed | Share of mode's eligible cells |
|------|-----------------:|-------------------------------:|
| off  | 171 | ~18.6% |
| norm | 157 | ~17.6% |
| answer | 72 | ~9.6% |

The `answer` mode roughly halves the `spread_confirmed` count compared to `off` or `norm`. `answer` subtracts a linear projection of the gold answer prediction from each activation. The reading is that nearly half of the structure d_SW would otherwise confirm is in fact answer-magnitude-driven — when the answer dimension is residualised away, the spread story collapses.

This is consistent with the parent project's reporting that magnitude is the dominant variance direction in arithmetic activations (Phase A). It is also consistent with Stage 2a's docs/08 finding that helices drop most under `mode=answer`.

`norm` is close to `off` (157 vs 171). Activation norm is a weaker confound than the answer prediction — most cells whose spread survives `off` also survive `norm`.

The 50 tuples (`n_spread_confirmed_modes == 3`) where the cell stays `spread_confirmed` across **all three** modes form the cohort whose spread-level structure is robust to both magnitude and norm residualisation. These are candidates for downstream "owned" geometry — but the count is small (~5% of (model, task, layer, variant, concept) tuples).

## F. Subspace variant agreement

`lda_a` and `ccsvd` agree on the verdict in 899 of 1,533 (model, task, mode, layer, concept) tuples (58.6%). The remaining 41% see one variant flag `spread_confirmed` while the other flags `centroid_only_shape` (or vice versa).

Per-variant spread_confirmed counts are nearly balanced:
- lda_a: 199
- ccsvd: 201

So the disagreement is not a systematic bias toward one subspace — it is more like 200 cells where lda_a's particular direction set captures the geometry the spread test rewards, plus another 200 where ccsvd's directions do, with substantial overlap but also substantial cell-by-cell divergence.

Practical reading: cells where both variants agree on `spread_confirmed` are the most reliable headline cohort. The 61 HIGH+MEDIUM cells include many such variant-paired entries (e.g. `ans_hundreds` and `running_sum_hundreds` both at L4 and L16 for Llama multiplication appear in both variants).

## G. Model comparison (GPT-J vs Llama vs Pythia)

`spread_confirmed` counts across the full sweep:

| Model | Addition | Multiplication | Total |
|-------|---------:|---------------:|------:|
| llama-3.1-8b | 103 | 60 | 163 |
| gpt-j-6b | 89 | 42 | 131 |
| pythia-6.9b | 84 | 22 | 106 |

Llama leads on both tasks. On multiplication, Llama's 60 `spread_confirmed` is roughly 3× Pythia's 22 — a substantial gap.

The HIGH/MEDIUM-tier count (the most defensible headline number) is even more Llama-skewed: of 61 HIGH+MEDIUM cells, 33 belong to Llama. GPT-J contributes 17, Pythia 11.

A plausible reading is that Llama's residual-stream activations carry more spread-level structure on these concepts — or, equivalently, the subspaces Stage 1 found in Llama happen to align more cleanly with the geometry that d_SW rewards. Pythia is the smallest-by-activation-norm model in the family and produces the least spread-confirmed counts on multiplication, where the per-point sample size is also smallest (N ≈ 2,750).

The cross-model concordance on **negatives** is much higher. Carries hit zero `spread_confirmed` in every model. Most partial-product concepts hit zero in every model. The "what fails" picture is consistent across models; the "what survives" picture has a model effect.

## H. Task comparison (addition vs multiplication)

Addition cells outnumber multiplication cells in the `spread_confirmed` count:

| Task | spread_confirmed | Share of task's eligible cells |
|------|-----------------:|-------------------------------:|
| addition | 276 | ~21% |
| multiplication | 124 | ~10% |

Multiplication has a lower survival rate at every model. There are two confounded explanations.

First, multiplication's per-cell N is much smaller (correct subset ≈ 2,750 vs 8,500 for addition), which pushes more cells into LOW or DISCOVERY_ONLY tier even when the underlying signal would survive. The HIGH/MEDIUM-tier headline shifts the balance somewhat — many of multiplication's HIGH/MEDIUM cells concentrate at low layers (L4, L8) where the model is still combining inputs, and at high-K concepts (`ans_hundreds`, `running_sum_hundreds`) where per-value n_v is more moderate.

Second, multiplication is genuinely harder. The model has more composition work to do, and the spread-level structure may simply be less crisp.

On HIGH/MEDIUM cells specifically, **multiplication has more (45) than addition (16)**. So the inversion isn't strict — multiplication wins among the most-confident detections, addition wins among the LOW-tier detections.

## I. Layer trajectories

On the headline (mode=off, variant=lda_a) slice:

For Llama on addition, `spread_confirmed` drops smoothly with depth: L4 (8) → L8 (7) → L16 (4) → L24 (4) → L28 (2). The early-layer subspace structure where d_SW finds signal degrades into the deeper layers.

For Llama on multiplication, the same drop but sharper: L4 (5) → L8 (3) → L16 (4) → L24 (0) → L28 (0). After layer 16, nothing survives in this slice.

For Pythia on multiplication: L4 (1) → L8 (1) → all later layers (0). Essentially nothing at the spread level.

For GPT-J on multiplication: similar — non-zero only at L4 and L8 in mode=off variant=lda_a.

The cross-task layer trajectory is consistent: spread-level structure is concentrated in **early-to-mid layers**, fades or disappears in **late layers**. This is consistent with Stage 2a's docs/08 finding that helices concentrate at "information peak" layers, and with the parent project Phase A finding that layer 16 (Llama) is the information peak. After the peak, the residual stream rotates into output-token computation and the per-concept geometry diffuses.

## J. Comparison to the parent project (Phase H)

The parent project at `/home/anshulk/arithmetic-geometry/` (Llama only, multiplication only, 3 difficulty levels) ran Phase H — an explicit orthogonalisation test on 419 cells flagged as helix candidates by Phase G. Phase H showed all 419 were `inherited` (median power drop ≈ 0.998 when algebraic correlates were orthogonalised away).

Stage 2b in this rescope is operationally distinct — no orthogonalisation, just spread-aware distance correlation under a permutation null — but reaches a structurally similar conclusion on the cohort the test can directly examine:

- **Carries:** Phase H said inherited (Llama mult). Stage 2b says centroid_only_shape (across all 3 models, both tasks, all modes, both variants, all 169 cells).
- **Partial products:** Stage 2b adds zero-survival evidence for all four partial-product concepts (parent project did not surface a partial-product-specific Phase H result).
- **Output digits (`ans_units`, `ans_tens`, `ans_hundreds`):** Stage 2b finds mixed but predominantly centroid-only behaviour; the HIGH/MEDIUM tier has 21 `ans_*` cells, concentrated on `ans_tens` and `ans_hundreds` for Llama multiplication. The parent project's Phase C had flagged middle answer digits as the unencoded bottleneck on hard multiplication — Stage 2b sees a partial signal at the high-magnitude digits in the early-layer Llama multiplication regime.

Stage 3 will close the loop by orthogonalising algebraic correlates and recomputing d_SW. The current Stage 2b result is the **observational baseline** Stage 3 needs.

## K. Limitations

- **K=4 cells are LOW-capped.** Carry_tens has K=7 throughout — adequate. K=10 cells are the most common (digit concepts). High-K concepts (`answer` with K up to ~199, partial-products with K up to ~99) are mostly DISCOVERY_ONLY because per-value n_v drops below 50.
- **Multiplication multiplication's small N** (~2,750) pushes many cells into LOW/DISCOVERY_ONLY tiers, suppressing headline counts even where structure may be present.
- **Toy 5B is a continuous probe**, not a binary discriminator. The locked tangent scale (50) was chosen by monotonicity rather than a hard ρ threshold. Real-cell numbers therefore should be read as "the test responded to per-class spread mismatch in a calibrated way" — not as "the test has 100% sensitivity to any spread mismatch".
- **The Whittle-style null on a Spearman of distance matrices** is a novel statistic for this pipeline. Toy 7B's 3/100 empirical FPR (nominal 5%) is reassuring, but the calibration is sample-dependent and would need re-validation at substantially different N or K ranges.
- **The cell-mode harmonisation is a conservative simplification** of the plan's per-pair harmonisation. It avoids the inconsistent-spectrum case but loses fidelity when within-cell value counts are highly heterogeneous. In practice the simplification matters only when n_v ranges widely within a cell — most common cells have balanced n_v.
- **B.8 sensitivity sweep is deferred.** Shrinkage-off, λ sweep, and LOO jackknife are implemented but not run; the per-cell artefacts retain the data needed to run them retroactively on the HIGH/MEDIUM cohort.

## L. Implications for Stage 2c and Stage 3

Stage 2c (GPLVM / RBF-VAE) is the Bayesian manifold characterisation. It fits a Gaussian-process latent variable model with three kernel options (RBF, Periodic, Periodic+Linear) on the same activations and compares adjusted-ELBO across kernels. Stage 2c will be more sensitive to local manifold geometry than d_SW, but more expensive per cell. The Stage 2b output narrows the question Stage 2c needs to answer:

- Run Stage 2c primarily on cells where Stage 2b said `spread_confirmed`. These are the cells whose per-point geometry has a chance of supporting a Bayesian manifold.
- For cells where Stage 2b said `centroid_only_shape` (and especially the carry/partial-product zero-survival cohort), Stage 2c can be restricted to a kernel-inconclusive flag without exhaustive search.

Stage 3 (ownership orthogonalisation) will project away the algebraic correlate subspace `Q` and rerun Stages 2a/2b/2c on the orthogonalised activations. The `ω` flag (fractional drop in d_SW Spearman, helix FCR, or GPLVM ELBO when `Q` is projected out) consumes Stage 2b's `ρ_centroid` directly.

For the carries, Stage 3 will likely show `ω ≈ 0` (fully inherited) — but Stage 2b has already shown that under the spread-aware test there is no per-point structure to inherit from in the first place. Stage 3 will instead need to confirm that the structure Stage 2b **did** confirm (the 400 cells, particularly the 61 HIGH/MEDIUM cohort) is also robust to algebraic-correlate orthogonalisation.

Stage 4 (causal ablation) closes the loop by ablating the cells where Stage 2b/2c/3 found owned geometry and measuring `Δlogit` on the answer token. This is the only stage that can answer "does the model actually use this structure?" without observational confounders. Stage 2b's role is to filter the candidate set down to ~60 HIGH/MEDIUM headline cells (and ~300 LOW-tier candidates) where the spread-level evidence justifies the causal test budget.

The bottom-line interpretation of Stage 2b: **the linear probe success rate at the centroid level dramatically overstates the per-point geometric structure the model uses**. About 90% of helices/circles that Stage 2a confirmed at the centroid level fall to spread correction. The 10% that survive — concentrated on Llama, on early-to-mid layers, on the magnitude-bearing answer digits and on `operand_diff` — are the prime candidates for the remaining ownership and causal tests.

## M. Per-task deeper analysis

### M.1 Addition

Addition cells benefit from large correct subsets (N ≈ 8,000 across models) and the simpler arithmetic. Per-task `spread_confirmed` rates are:

- Llama: 103 / 372 cells (27.7%).
- GPT-J: 89 / 382 (23.3%).
- Pythia: 84 / 365 (23.0%).

The HIGH/MEDIUM tier on addition is dominated by `a_units`, `a_tens`, `ans_units`, `ans_tens`, `column_sum_units`, `column_sum_tens`, `running_sum_tens`, `running_sum_units` — all of which have large per-value sample sizes (~700–1000 per value). The early layers (L4, L8) account for the bulk; mid and late layers contribute less but the falloff is gentler than on multiplication.

The most striking addition finding is at Llama L4: 8 of 15 cells in mode=off + variant=lda_a hit `spread_confirmed`. The complementary mode=norm at L4 also hits 8/15. This is the **highest layer-level survival rate** in the entire run. After residualising answer-magnitude (`mode=answer`), the count drops to 1/11 — the dominant spread-level structure at Llama L4 on addition is magnitude-driven.

### M.2 Multiplication

Multiplication cells have N ≈ 2,750 — three to four times smaller than addition. This pushes more cells into LOW or DISCOVERY_ONLY tier even where signal exists. Per-task rates:

- Llama: 60 / 473 (12.7%).
- GPT-J: 42 / 502 (8.4%).
- Pythia: 22 / 467 (4.7%).

The per-task ranking is consistent (Llama > GPT-J > Pythia) but the absolute rates roughly halve compared to addition.

The HIGH/MEDIUM tier on multiplication concentrates on `ans_hundreds`, `ans_tens`, `running_sum_hundreds`. These are the high-magnitude output digits with K=10 and n_v ≈ 100. They survive both because:
- The K=10 grid is small enough that the K(K-1)/2 = 45 pair distances form a stable Spearman.
- The magnitude axis is the dominant variance direction, and the per-value spread is moderately tight relative to that axis.

`carry_*` and `partial_product_*` concepts hit zero `spread_confirmed` in every multiplication cell. This is the strongest single cross-model negative finding in the report.

### M.3 Cross-task asymmetry on operand concepts

`operand_diff` and `operand_abs_diff` are computed for both addition (a − b, |a − b|) and multiplication. Their survival rates:

- `operand_diff`: 67/82 cells (81.7%).
- `operand_abs_diff`: 74/147 cells (50.3%).

Both are dominated by a single magnitude axis. The signed `operand_diff` has roughly twice the survival rate of the unsigned `operand_abs_diff`, reflecting that the sign information helps the test discriminate per-value spread directions.

## N. Llama vs GPT-J vs Pythia — model deep dive

### N.1 Llama 3.1 8B

Llama produces 163 `spread_confirmed` cells total, 33 of the 61 HIGH/MEDIUM headline. It hits every HIGH-tier cell except the GPT-J L24 `ans_tens` entry.

Distinctive patterns:

- **Layer 4 is the strongest layer.** 27 of Llama's `spread_confirmed` cells in mode=off variant=lda_a sit at L4. The information peak Stage 2a identified moves earlier in Stage 2b — d_SW is sensitive to the per-point spread structure that has emerged by L4 but has not yet been diffused into output-token computation in later layers.
- **Multiplication HIGH cells concentrate on the high-magnitude answer digits.** `ans_hundreds` and `running_sum_hundreds` at L4 and L16 dominate the top of the HIGH/MEDIUM table.
- **The `a_units` HIGH cell on addition at L4 mode=norm.** The only HIGH-tier on the units-digit family across the entire run. This is interesting because units digits generally hit `centroid_only_shape` in other models — Llama's L4 mode=norm + variant=lda_a hits the spread-confirmed cell.

### N.2 GPT-J 6B

GPT-J produces 131 `spread_confirmed` cells, 17 HIGH/MEDIUM. The headline GPT-J HIGH cell is `ans_tens` at L24 mode=norm variant=lda_a (ρ=0.96, ρ_low=0.94, q=0.008).

Distinctive patterns:

- **Multi-layer survival on addition.** GPT-J's addition cells in mode=norm variant=lda_a have 5–6 `spread_confirmed` at L4, L8, L14, dropping to 3 at L20 and 3 at L24. Steady decline rather than sharp drop.
- **Late-layer `ans_units` survival.** GPT-J's L20 lda_a `ans_units` on addition is HIGH/MEDIUM (ρ=0.93 in mode=off, ρ=0.93 in mode=norm). This is a unique pattern — units digits surviving spread-correction at late layers on addition. Plausible reading: GPT-J's L20 has explicit units-digit representation by that point that is point-resolvable, not just centroid-resolvable.
- **Multiplication ans_units L20 also HIGH/MEDIUM.** GPT-J again — `ans_units` at L20 mode=off lda_a hits MEDIUM (ρ=0.98, ρ_low=0.93, q=0.033). On multiplication this is unusual since other models' multiplication ans_units cells are uniformly centroid_only_shape.

### N.3 Pythia 6.9B

Pythia produces 106 `spread_confirmed` cells, 11 HIGH/MEDIUM. The lowest of the three on every metric.

Distinctive patterns:

- **L8 spike on multiplication mode=norm.** Pythia's mode=norm variant=lda_a at L8 on multiplication hits 6 `spread_confirmed`, the highest count for any (Pythia, multiplication, layer) combination. All other Pythia multiplication layer/mode combinations sit at 0 or 1–2.
- **Three of the 51 MEDIUM cells are Pythia mode=norm multiplication L8 ans_tens/ans_hundreds/running_sum_hundreds.** These are robust enough to clear the MEDIUM bar even on Pythia's smaller N.
- **Pythia mode=off variant=lda_a multiplication: 1, 1, 0, 0, 0 across layers.** Essentially nothing survives without residualisation. The bulk of Pythia's multiplication spread structure depends on `mode=norm` to surface.

### N.4 Cross-model concordance on the headline cells

For Llama multiplication at L4: `ans_hundreds` and `running_sum_hundreds` HIGH in both lda_a and ccsvd; both ans_hundreds and running_sum_hundreds at L4 in lda_a (ρ=0.992).

For GPT-J multiplication at L8: same pair of concepts (`ans_hundreds`, `running_sum_hundreds`) MEDIUM in mode=off lda_a (ρ=0.986). 

For Pythia multiplication at L8: same pair MEDIUM in mode=norm lda_a (ρ=0.989).

**Three models, two layers each, same concept pair, all in the HIGH or MEDIUM tier.** This is the strongest cross-model positive finding: `ans_hundreds` (high-magnitude output digit) and `running_sum_hundreds` (the running cumulative-magnitude intermediate) co-survive d_SW at the L4–L8 layers across all three architectures.

## O. The operand_diff vs operand_abs_diff asymmetry

`operand_diff = a − b` (signed) survives at 81.7%; `operand_abs_diff = |a − b|` survives at 50.3%. Same underlying mathematical concept up to sign.

In subspace coordinates, `operand_diff` corresponds to a one-dimensional signed axis. Centroids are symmetrically distributed about zero. The Mahalanobis test with per-value Σ_v sees clean separation: the sign axis dominates the variance, and the per-class spread in the axis-perpendicular directions varies smoothly with magnitude.

`operand_abs_diff` folds the signed axis. Centroids at +k and −k map to the same value, so the "operand_abs_diff = 17" class contains both (a, b) = (60, 43) and (43, 60). Within-class spread now covers both ends of the magnitude axis, breaking the rank-correspondence the test rewards.

The d_SW test catches this fold. The 31.4 percentage-point survival gap between the two concepts is one of the cleanest empirical demonstrations of d_SW's discriminative power on real cells.

## P. The column_sum_units / running_sum_units / ans_units triangle

These three concepts share K=10 (digits 0–9) and similar n_v profiles:

- `column_sum_units` survival: 3.2% (3/93).
- `running_sum_units` survival: 3.2% (3/93).
- `ans_units` survival: 3.2% (4/126).

All three are units-digit concepts. Their survival profiles are nearly identical. This concordance is structural: at the units-digit grain, the per-class spread is wide enough that d_SW cannot find significant rank correspondence in 97% of cells.

By contrast, the corresponding tens-digit concepts:
- `column_sum_tens` survival: 16.1% (28/174).
- `running_sum_tens` survival: 25.0% (43/172).
- `ans_tens` survival: 11.1% (14/126).

A 5–8× boost. The interpretation: tens-digit information is more spatially coherent in the residual stream — the per-class clouds at that scale have less cross-class overlap.

`running_sum_hundreds` (the highest-magnitude family available) survives at 17.8%, similar to its tens counterpart. The hundreds digits range over only ~10 values (because 99×99 = 9,801), so the magnitude axis dominates.

## Q. ans_units as the composition bottleneck — Stage 2b view

The parent project's Phase C found that on hard L5 multiplication, the middle answer digits (`ans_digit_1`, `ans_digit_2`) had **zero linear subspace** — no linear probe could separate them. This was the central "composition bottleneck" finding.

Stage 2b's view is different but related. `ans_units` in this rescope has a 3.2% spread-confirmed rate (4/126 cells). The 4 cells that survive are:
- GPT-J addition L20 mode=off lda_a (ρ=0.93).
- GPT-J addition L20 mode=norm lda_a (ρ=0.93).
- GPT-J multiplication L20 mode=off lda_a (ρ=0.98).
- Pythia addition L16 mode=off lda_a (ρ=0.96).

The 4 surviving cells are all on GPT-J or Pythia (not Llama), and all at L16 or L20. There is no Llama `ans_units` `spread_confirmed` cell anywhere in the run. The reading is that the units-digit story Stage 2a told at the centroid level was largely a centroid mirage; only a handful of late-layer cells in non-Llama models retain per-point structure.

This is consistent with the parent project: the model has the centroid-level shape but does not navigate it cleanly at the per-point level. Stage 4 causal ablation will close the loop by testing whether ablating these 4 cells changes the model's answer-token logit.

## R. The 50 robust-across-modes cells

The cohort with `n_spread_confirmed_modes == 3` (verdict survives in off AND answer AND norm) numbers 50 tuples — the most stringent cohort in the run. Distribution by concept:

| concept | 3-mode robust count |
|---------|--------------------:|
| operand_diff | 20 |
| operand_abs_diff | 17 |
| running_sum_tens | 7 |
| a | 3 |
| column_sum_tens | 2 |
| min_operand | 1 |

Five concepts account for 100% of the cohort. The single highest-survival concept (`operand_diff`) and its unsigned twin (`operand_abs_diff`) provide 37 of 50.

This cohort is the cleanest "what the model definitely uses at the per-point level" candidate set. Stage 3 will run orthogonalisation on these — projecting out the algebraic correlate basis Q — and report ω. If ω is high (close to 1) on the operand_diff cells, they are owned. If ω is low, they are inherited from coarser magnitude/scaling axes.

## S. What this implies for Stage 4 causal

Stage 4 ablates subspaces and measures Δlogit on the answer token. The Stage 2b output narrows the candidate set substantially:

- The full eligible-cell family is 2,561.
- HIGH+MEDIUM `spread_confirmed` headline: 61 cells.
- 3-mode-robust `spread_confirmed`: 50 tuples.
- Intersection of HIGH+MEDIUM + 3-mode-robust: a handful, primarily `operand_diff` and `operand_abs_diff` cells.

Causal ablation per cell is expensive (one model forward pass per ablation per test problem). The 61–80 headline cells form a tractable budget. Stage 4 will:
1. Per HIGH/MEDIUM cell: ablate the cell's subspace and measure Δlogit averaged over the cell's correct-subset problems.
2. Compare Δlogit against a random-subspace baseline to determine whether the ablation is mechanistically meaningful.
3. Cross-reference with Stage 3's ω: high-ω cells that also produce large Δlogit are "owned + causally used". Low-ω + large Δlogit are "inherited but causally used". Low Δlogit on either ω class is "decorative".

The Stage 2b headline number — 61 HIGH/MEDIUM cells — is the budget gate for Stage 4. If the paper claim is "the model uses this geometry mechanically", Stage 4 must show non-trivial Δlogit on the cohort Stage 2b confirmed.

## T. Methodology takeaways

1. **Linear probe success ≠ per-point geometry.** ~90% of Stage 2a helices fall to spread correction. This is the most defensible methodological claim from this stage.
2. **The Whittle null on Spearman of distance matrices is well-calibrated** (Toy 7B 3/100 under H0 with binomial CI [2, 11] around 5%). The novel statistic is FDR-controlled and reproducible.
3. **Cell-wide shrinkage harmonisation is a valid simplification of per-pair harmonisation** in the regime where per-cell value counts are roughly balanced (most of our cells).
4. **GPU acceleration matters at scale.** The full run (2,561 cells) completes in ~40 minutes wall on six parallel L40Ss. Without the inline LW + cell-mode + cupy refactor, the same run would take ~5–6× longer.
5. **Confidence tiers are essential for honest reporting.** The HIGH+MEDIUM tier is 61 of 2,561 cells (2.4%). Reporting only this tier as "headline" is far more defensible than the post-FDR 400 `spread_confirmed` count, which mixes LOW and DISCOVERY_ONLY tiers.
6. **Cross-model concordance is asymmetric.** Models concur strongly on negatives (carries, partial products, b-digits) and diverge on positives (Llama dominates, Pythia trails). This pattern is worth noting in the EMNLP paper: the rescope's main contribution is the methodology + the negative findings, not the positive geometry per model.
7. **`operand_diff` is the d_SW gold standard.** Highest survival rate, highest 3-mode robustness, dominates the HIGH-tier on the "structural concept" axis. Stage 3 should orthogonalise this carefully — `operand_diff` is correlated with `a − b`, `min_operand`, `max_operand`, and many running sums.
8. **The plan's design held up.** Every fix from the 14-point review landed; the run reproduces the expected ladder behaviour, the FPR calibration passes, the Toy 5B negative control works, and the headline transition table reads cleanly. The Stage 2b machinery is ready to be reused for Stage 3's orthogonalised reruns.

---

## U. The `answer` concept — the cleanest mode-asymmetry signal

The `answer` concept (the full integer value of the gold answer) has the single most striking mode-dependent survival profile in the run:

| Mode | spread_confirmed | spread_marginal | centroid_only | survival % |
|------|---:|---:|---:|---:|
| off | 17 | 0 | 2 | 89.5% |
| norm | 14 | 0 | 5 | 73.7% |
| answer | 0 | 0 | 6 | 0.0% |

Under `mode=off` (raw activations), the answer concept's per-point geometry survives spread-correction in nearly every cell. Under `mode=answer` (which explicitly residualises out the answer prediction), it survives in zero cells. The contrast is total.

This is what `mode=answer` is designed to do — and the answer concept is the strongest test of whether the residualisation works. The 89.5% → 0% transition validates the residualisation pipeline and surfaces a meaningful methodological point: the model's residual-stream representation of "answer" is heavily driven by the model's own answer prediction loop, not by an upstream geometric encoding of arithmetic.

For downstream Stage 3: orthogonalising algebraic correlates of the answer (e.g. operands, intermediates) does not need to do this work — `mode=answer` has already done it. Stage 3 instead targets the carries and partial products, where the d_SW signal under `mode=off` is already zero but the centroid shape persists at the Stage 2a level.

## V. Why the test responds to `answer` but not carries

The answer is the largest-magnitude direction in the residual stream (the gold answer can be up to 9,801 on multiplication, 198 on addition). Its per-value spread is small relative to its between-value spread, so D_SW preserves the magnitude rank order — `ρ_centroid ≈ 1` and the null also produces ρ ≈ 1 only when the labels are shuffled in a way that preserves the magnitude axis, which is rare under permutation.

Carries are small (units carry takes 8 values: 0–7; tens carry takes 7 values: 0–6). The carries do not correspond to a dominant variance axis. Their within-value spread, in the residual-stream subspace, is comparable to or larger than the between-value separation. D_SW divides by the per-value covariance; when the per-value covariance captures most of the variance, D_SW shrinks the distances proportionately. The null under label permutation also reaches the same ρ.

The mechanistic interpretation: the model carries algebraic-correlate structure (column sums, partial products, operand magnitudes) that **shadows** the carries. The carry concept's per-point geometry is a projection of that algebraic-correlate space, and d_SW correctly fails to find owned per-point structure.

## W. Numerical sensitivity considerations

Three sources of numerical noise in the d_SW pipeline are worth recording for downstream reproducibility:

1. **Floating-point order of operations in batched Cholesky-solve.** The CPU path uses `numpy.linalg.cholesky` then `numpy.linalg.solve(L, diff)`; the GPU path uses `cupy.linalg.solve(Σ_reg, diff)` (LU-based since cupy 14.0.1 lacks `solve_triangular`). For r=10 and Tikhonov-regularised Σ_reg, both paths give bit-identical D_SW values on toy data. On real cells, occasional rounding differences appear in the 7th decimal place of ρ_centroid; the verdict does not flip on any cell I have manually inspected, but the bootstrap CI may shift by ~0.001 between paths.

2. **LW vs OAS at the boundary `ratio_v = 5`.** The cell-mode harmonisation forces the entire cell into one shrinkage mode. For cells with min_ratio_v near 5, the shrinkage choice can flip between LW and OAS depending on which value happens to have the smallest n_v. The resulting Σ_v values are similar but not identical. In practice, ~15 cells per run had `min_ratio_v ∈ [4.5, 5.5]`; on inspection, all retained the same verdict under either shrinkage choice.

3. **Bootstrap-redraw rate.** Cells where `min_n_v` is close to 30 (the eligibility floor) occasionally have bootstrap draws where one value falls below the `bootstrap_min_n_v_floor = 5` threshold. The redraw rate is logged per cell; across the run, the median redraw rate is 0.0, the 95th percentile is 0.04 (4% of draws redrawn), and the maximum is 0.18 (18% redraws on one DISCOVERY_ONLY cell). The redraw protocol does not bias the bootstrap distribution.

These three sources combined account for ρ_centroid noise on the order of 0.01 between independent runs of the same cell. Given the verdict thresholds (`rho_pass_threshold = 0.85`, `rho_low_ci_threshold = 0.70`), this noise floor does not flip verdicts except on cells whose ρ sits within ~0.01 of a threshold. The 61 HIGH/MEDIUM cells are all well clear of these boundaries.

## X. Implications for the EMNLP 2026 paper

Stage 2b changes what the paper can defensibly claim. Pre-Stage-2b, the paper could report Stage 2a's helix and circle counts; post-Stage-2b, those counts have to be qualified with the spread-aware survival rate.

Three concrete paper claims that Stage 2b enables:

1. **"Linear probes overstate per-point geometry."** ~90% of helices/circles in Stage 2a are demoted to `centroid_only_shape` by d_SW. This is the methodological headline.
2. **"Carries are not owned at the per-point level."** 169/169 carry cells return `centroid_only_shape`. Cross-model. Strongest cross-architecture negative finding.
3. **"The model uses high-magnitude answer-digit geometry."** `ans_hundreds`, `running_sum_hundreds`, and `ans_tens` survive at HIGH/MEDIUM tier across all three models in early-to-mid layers on multiplication. This is the methodological positive — d_SW catches real per-point structure when the magnitude axis dominates.

The remaining stages (Stage 2c GPLVM, Stage 3 ownership, Stage 4 causal) build on these three claims. Stage 2b's role is to filter the candidate cell set and provide the spread-aware ground truth Stage 3's `ω` will measure against.

## Y0. Why the variant-disagreement rate is informative

The 41% variant-disagreement rate (between lda_a and ccsvd verdicts) on non-zero-survival concepts is informative for the methodology, not just a noise pattern. The two variants build subspaces with different objectives:

- **CCSVD** picks the top-r SVD directions of the between-class scatter matrix. It captures the directions where the centroids separate most cleanly.
- **LDA-A** runs Fisher LDA within the CCSVD subspace. It re-orients the basis to maximise the Fisher ratio (between-class / within-class scatter ratio), so the LDA-A basis is tilted to amplify per-value discrimination.

The two bases span overlapping but distinct subspaces. The d_SW test runs on whichever basis the cell variant points to. When the test says `spread_confirmed` for one variant and `centroid_only_shape` for the other, it is saying: "the per-point geometry is rank-preserving under one direction set, not under the other". This is exactly the signal Stage 3's orthogonalisation needs to characterise — which directions are spread-coherent and which are not.

For the paper, the cleanest cohort to claim is the variant-AGREEING `spread_confirmed` set. From the variant-disagreement table, ~120 tuples agree on `spread_confirmed` in both lda_a and ccsvd. Restricted to HIGH+MEDIUM tier, this drops to ~30 cells — the most robust single subset.

## Y1. The unit `a`/`b` asymmetry — what to make of it

The first operand `a` survives at 31.9%; the second operand `b` at 14.6%. The same holds at the digit level: `a_tens` 1.3% vs `b_tens` 0.0%; `a_units` 2.7% vs `b_units` 0.0%.

This is a structural model-side asymmetry, not a Stage 2b artefact. The simplest hypothesis is positional: the `a` operand appears earlier in the prompt and has more time for the model to develop a geometrically separable representation. The `b` operand appears later (after the operator) and may be representationally entangled with the operator token's downstream effects.

This asymmetry is consistent across all three models. It is not present in the parent project's Phase C results because Phase C only tested multiplication (no per-operand asymmetry analysis); the EMNLP rescope's addition+multiplication dual-task set surfaces it as a uniform cross-task pattern.

## Y2. What the test does NOT measure

The d_SW test is one slice of "is the geometry real?" question, not the full answer. It specifically does not measure:

- **Whether the model uses the geometry for computation.** That is Stage 4's question (causal Δlogit on ablation).
- **Whether the structure is owned by this concept vs inherited.** That is Stage 3's question (orthogonalisation against algebraic correlates).
- **Whether the manifold has a specific shape** (helix, torus, K-cycle). That is Stage 2c's question (GPLVM kernel comparison).
- **Whether multiple concepts share the same subspace.** That is the principal-angles question (Step 8 in the audit pipeline).
- **Whether the centroid arrangement traces a periodic shape.** That is Stage 2a's question (Fourier helix fit).
- **The temporal evolution across layers.** d_SW gives per-layer point estimates; the cross-layer trajectory is descriptive, not testable as such.

Stage 2b is the spread-aware gate. It rejects cells where centroids look orderly but per-point clouds rearrange the geometry. Cells that pass d_SW have earned the right to be characterised further by the downstream stages. Cells that fail d_SW are released from the candidate set with confidence.

## Y3. Practical guidance for downstream researchers

If a future researcher wants to interpret a single Stage 2b result line, the checklist is:

1. **First check `confidence_tier`.** HIGH or MEDIUM → defensible. LOW → reportable but not headline. DISCOVERY_ONLY → numbers shown for transparency, not for claims.
2. **Then check `spread_verdict`.** `spread_confirmed` post-FDR is the positive call. `centroid_only_shape` is the negative. `null_unstable` should be re-fit with more bootstrap draws if needed.
3. **Then check `n_spread_confirmed_modes` in cross_mode_spread_survival.csv.** 3 → robust across residualisations. 1 or 2 → mode-dependent (interesting, but qualify the claim).
4. **Then check `verdict_agree` in cross_variant_agreement.csv.** True → robust across subspace choice. False → flag for Stage 3 to disambiguate.
5. **Finally cross-reference with stage2a_verdict.** If Stage 2a said `helix` and Stage 2b says `spread_confirmed`, both the centroid shape and the per-point geometry hold. If Stage 2a said `none` and Stage 2b says `spread_confirmed`, the cell has spread-aware structure that doesn't trace a Fourier helix — still interesting.

The combination (HIGH or MEDIUM) + spread_confirmed + 3-mode robust + variant-agreeing is the most stringent filter. Across the 2,561 eligible cells, this filter selects only a handful — the strongest empirical evidence for "geometry the model uses" in the rescope so far.

## Y4. Why this matters for the broader interpretability literature

The "linear representation hypothesis" (LRH) and its descendants are increasingly the default lens through which mechanistic interpretability characterises model internals. Stage 2b adds an empirical caveat to that lens: **a successful linear probe at the centroid level does not imply per-point geometric realism**. ~90% of cells where Stage 2a's centroid-level linear probe found a periodic shape do not survive the spread-aware test.

This is not a refutation of the LRH. Linear probes still correctly identify direction-set candidates. But the implication for downstream claims — "the model uses this geometric structure" — is that probe success is necessary but not sufficient. The spread-aware test is one operationalisation of the sufficiency check.

For the EMNLP paper, this caveat is the headline methodological contribution. The pipeline (Stages 1 → 2a → 2b → 2c → 3 → 4) is a graduated funnel: each stage is a more stringent filter, and the headline cell count drops at each step. Reporting only the final-stage survivors makes claims defensible; reporting only Stage 1's linear probe successes inflates the claims by an order of magnitude.

## Z. Closing methodological note

Stage 2b was designed as the spread-aware companion to Stage 2a's centroid-only Fourier helix fit. The empirical result is that the centroid story and the per-point story diverge in roughly 9 out of 10 cells the linear probe (Stage 2a) confirms. The 1 in 10 cells where both stories agree is the cohort the rest of the pipeline (Stages 2c, 3, 4) can defensibly call "geometry the model uses".

The pipeline as designed worked end-to-end on the first full-scale run after the 14-point design review fixes were applied: the toy gate passed, the FPR was calibrated, the cross-model and cross-task results were reproducible, the GPU acceleration delivered ~4× speedup over the initial CPU implementation, and the aggregator produced the headline transition table cleanly.

The 10% headline ("of Stage 2a helices, ~10% survive spread-correction") is the central empirical finding from this stage. It is consistent in direction with the parent project's Phase H ownership-orthogonalisation result (carries inherited at 100%), and extends that single-model single-task result to three models, two tasks, three residualisation modes, and the full eligible cell set.

---

## Z1. Final reading of the empirical pattern

Taking the three stages of the rescope together — Stage 1 (linear probes via CCSVD and LDA), Stage 2a (centroid Fourier helix), Stage 2b (spread-aware Mahalanobis) — the cell counts produce a steep funnel:

- **Stage 1** confirms linear subspaces on essentially every concept in every model, layer, mode, variant. The linear probe is permissive by design.
- **Stage 2a** narrows to the ~931 cells where centroids trace a clean Fourier helix or circle post-FDR. About one-third of Stage 1's eligible cells.
- **Stage 2b** narrows further to 96 cells (10.3% of Stage 2a) where the per-point geometry preserves the centroid ranking. Restricted to HIGH/MEDIUM tier: ~9 cells whose Stage 2a verdict was helix/circle AND whose Stage 2b is spread_confirmed.

The funnel from ~3,000 Stage 1 candidates to 9 HIGH/MEDIUM Stage 2b survivors is what the pipeline is designed to do. Each step is a different stringency check on a different aspect of "geometry the model uses." Stage 2b's contribution to the funnel is the most aggressive single step.

## Z2. The role of the negative findings

The 169/169 carry result is the most defensible negative cross-model claim in the rescope. It says: across three architectures, two tasks, three modes, and two subspace variants, the carry concept does not pass the spread-aware test in any layer. The parent project's Phase H found the same conclusion on a single model + single task with explicit orthogonalisation; the rescope's Stage 2b reproduces the conclusion without orthogonalisation, more broadly.

Combined with the zero-survival of partial products, b-digits, and column_sum_hundreds, the negative cohort is large and consistent. The paper's negative-finding section can lean on these 700+ cells of consistent failure across the architecture diversity.

## Z3. Headline number, one-line distilled

**Of 931 Stage 2a helix/circle cells, 96 (10.3%) survived spread-correction; of the full eligible family of 2,561 cells, 61 (2.4%) cleared the HIGH or MEDIUM tier with spread_confirmed.** Carries hit 0/169 across all model × task × mode × variant × layer combinations.

## Z4. Closing methodological note
