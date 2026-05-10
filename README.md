# EMNLP 2026 — Geometry of Arithmetic in Language Models

**Paper (working title):** *From Linear Probes to Bayesian Manifolds: Geometry of Arithmetic in Language Models*
**Authors:** Anshul Kumar (first), Barnabás Póczos (senior)
**Target venue:** ACL Rolling Review → EMNLP 2026 Main (long paper). Workshop fallback: BlackBoxNLP.

A four-stage Bayesian pipeline that tests whether the geometric structure a linear probe finds for a concept actually *belongs to* that concept. Applied to addition and multiplication (`a, b ∈ [0, 99]`) in three pre-trained LMs (GPT-J 6B, Llama 3.1 8B, Pythia 6.9B) on the per-model correct subset.

---

## 1. Status

| # | Step | Script | Status | Doc | Headline output |
|---|---|---|---|---|---|
| 0 | Model downloads | (manual) | ✓ done | this README | 51 GB of weights in `data/models/` |
| 1 | Tokenization preflight | `check_tokenization_limits.py` | ✓ done | [docs/01](docs/01_tokenization_limits.md) | Single-token integer caps: GPT-J 520, Llama 999, Pythia 530 |
| 2 | Dataset generation | `generate_datasets.py` | ✓ done | [docs/02](docs/02_dataset_generation.md) | Addition 10,000 / Multiplication 3,023 (cross-model intersection); Tier 1–5 concept schema per problem |
| 3 | Activation extraction | `eval_and_extract.py` | ✓ done | [docs/03](docs/03_eval_and_extract.md) | 30 `.npy` files (5 layers × 2 tasks × 3 models); SLURM array wall 79 s |
| 4 | UMAP + t-SNE embeddings | `build_embeddings.py` | ✓ done | [docs/04](docs/04_umap_tsne_embeddings.md) | 30 per-cell CSVs + manifests under `data/results/embeddings/`. CPU-only, total wall 77.9 min. Trustworthiness ≥ 0.94 across all 30 cells |
| 5 | CCSVD subspaces *(Stage 1 sub-step a)* | `ccsvd_subspaces.py` | ✓ done | [docs/05](docs/05_ccsvd_subspaces.md) | Per-cell basis, eigenvalue spectrum, 1,000-permutation null, 5-fold CV. ≈1,620 cells fit (~480 per model). |
| 6 | Residualization + LDA refinement *(Stage 1 sub-steps b + c)* | `residualize_activations.py` + `ccsvd_subspaces.py --mode` + `lda_subspaces.py` | ⏳ submitted | docs/06_lda_subspaces.md (after run) | Three residualization modes (off / answer / norm) × two LDA placements (Option A in CCSVD subspace + Option B in full 4096-D with Ledoit–Wolf shrinkage). Dual significance: permutation null ∩ k-NN CV-accuracy. Cohen's d, bootstrap CI on λ_T_1, A↔B alignment. SLURM array, 1 GPU per model. |
| 7 | Stage 2 — Bayesian manifold | (next) | pending | — | Centroid Fourier helix, d_SW, GPLVM, RBF-VAE |
| 8 | Stage 3 — Ownership test | (next) | pending | — | Orthogonalisation against algebraic correlates; verdict ∈ {owned, inherited, ambiguous} |
| 9 | Stage 4 — Causal ablation | (next) | pending | — | Δlogit on first answer token |

---

## 2. Models, tasks, and accuracies

### Models

| | GPT-J 6B (primary) | Llama 3.1 8B | Pythia 6.9B |
|---|---|---|---|
| HF repo | `EleutherAI/gpt-j-6B` | `meta-llama/Llama-3.1-8B` | `EleutherAI/pythia-6.9b` |
| Architecture | 28 layers, 4096 hidden | 32 layers, SwiGLU, GQA | 32 layers, parallel attn+MLP |
| Tokenizer | GPT-2 BPE | TikToken | GPT-NeoX BPE |
| Vocab | 50,257 | 128,000 | 50,254 |
| Single-token integer cap | **520** | **999** | **530** |
| Layers extracted | 4, 8, 14, 20, 24 | 4, 8, 16, 24, 28 | 4, 8, 16, 24, 28 |
| Headline layer | 14 | 16 | 16 |
| Weights size | ~23 GB | ~15 GB | ~13 GB |

### Tasks

- **Addition** `a + b`, `a, b ∈ [0, 99]`. Full Cartesian product = **10,000 problems**. Single-token answer in all three tokenizers.
- **Multiplication** `a × b`, `a, b ∈ [0, 99]`. **Cross-model single-token intersection = 3,023 problems**. Answer range `[0, 999]`.

Correctness: first-answer-token match against the gold first-token id (precomputed in Step 1). Equivalent to exact-match.

### Accuracy (per-model correct subsets — the population every later step runs on)

| Model | Addition | Multiplication |
|---|---:|---:|
| GPT-J 6B | **84.15 %** (8,415 / 10,000) | **91.00 %** (2,751 / 3,023) |
| Llama 3.1 8B | **99.63 %** (9,963 / 10,000) | **96.82 %** (2,927 / 3,023) |
| Pythia 6.9B | **77.18 %** (7,718 / 10,000) | **91.20 %** (2,757 / 3,023) |

---

## 3. Pipeline

Stages and the steps that implement each:

1. **Stage 1 — Linear probe** (per-cell, per-concept)
   - **(a) Conditional Covariance + SVD (CCSVD).** Centroids → between-class scatter → SVD → permutation-null filter → orthonormal basis `B ∈ ℝ^{4096 × r}`. **Step 5**.
   - **(b) LDA refinement.** Generalised eigenvalue `S_B w = λ S_W w`, λ_k ∈ [0,1], Cohen's d. **Step 6**.
   - **(c) 5-fold CV.** Spearman correlation of predicted vs true class label. **Step 6**.
   - Pass criteria: `λ₁ ≥ 0.5`, bootstrap CI lower-bound > 0, `cv_correlation ≥ 0.7`.

2. **Stage 2 — Bayesian manifold characterisation.** (a) centroid Fourier helix fit, (b) spread-aware Mahalanobis `d_SW`, (c) GPLVM, (d) RBF-precision VAE.

3. **Stage 3 — Ownership test.** Orthogonalise against pre-registered algebraic correlates; re-run Stage 2; verdict ∈ {owned, inherited, ambiguous}.

4. **Stage 4 — Causal ablation.** Raw vs orthogonalised subspace ablation against random-subspace null; measure Δlogit on first answer token.

### Headline matrix to populate

|  | Addition | Multiplication |
|---|---|---|
| Operand | predicted owned (trivial) | predicted owned (trivial) |
| Intermediate | **TEST** | predicted inherited (Phase H from arithmetic-geometry replicated 419 / 419) |
| Output | **TEST** | **TEST** |

Either Finding A (asymmetric) or Finding B (uniform inheritance) is publishable — the pipeline is the constant.

---

## 4. Step 5 method — Conditional Covariance + SVD

Per (model, task, layer, concept) cell on the correct subset:

1. **Filter** values with `n_v < 30` (`MIN_GROUP_SIZE`); skip cell if fewer than 2 values survive.
2. **Centroids** `μ_v = mean(X | concept = v)`; global mean `μ̄`.
3. **Scaled centred matrix** `M_v = √(n_v / N) (μ_v − μ̄)`. Then `S_B = M^T M` is between-class scatter.
4. **SVD** `M = U Σ V^T`; eigenvalues of `S_B` are `s_k²`.
5. **Permutation null** — 1,000 label shuffles, recompute eigenvalues; per-eigenvalue 99th percentile is the p<0.01 threshold.
6. **Sequential stop** — `r` = largest k such that all `λ_1 .. λ_k` beat their thresholds.
7. **Basis** `B = V[:r, :]^T` ∈ ℝ^{4096 × r}; project centroids and full cloud into B.
8. **5-fold subspace-preservation CV** — Pearson on pairwise centroid distances (full-space vs subspace).
9. **Risk flags** — N/d inflation (N/r < 5), single-direction dominance (λ₁/λ₂ > 10), group imbalance (max n_v / min n_v > 3).

---

## 4b. Step 6 method — Residualization + LDA (Stage 1 sub-steps b + c)

Step 6 layers two upgrades on Step 5:

**(i) Residualization** — remove one global confounding direction from activations *before* CCSVD or LDA see them. Three modes run in parallel:
- `off` — passthrough (no residualization). Reuses Step 5's CCSVD output verbatim.
- `answer` — OLS-regress activations on the gold answer (`a+b` for addition, `a·b` for multiplication); keep residual. Mirrors the parent project's *product residualization*.
- `norm` — OLS-regress activations on `||x||` (per-row L2 norm); keep residual. Removes the magnitude-of-activation confound.

For non-`off` modes, **CCSVD is re-fit on the residualized activations** so the subspace basis is not contaminated by the very direction we removed. `mode=off` reuses the existing Step 5 output.

**(ii) LDA refinement** — two placements per cell:
- **Option A (headline)** — LDA in the CCSVD subspace (r ≤ ~18 directions). N/r ≈ 100+ → eigenvalues trustworthy. K×K compact form of the generalised eigenproblem `S_B w = λ_T S_T w`. λ_T ∈ [0, 1] reads as "fraction of variance that is between-class."
- **Option B (audit)** — LDA in the full 4096-D residualized space, with **Ledoit–Wolf-style (OAS) shrinkage** on `S_T` (mandatory at d=4096, where N/d may be < 1 on multiplication × GPT-J). B's eigenvalue *magnitudes* are not cited — only directions and `n_sig` are. **Cosine similarity between A's and B's top direction is reported as a structural audit per cell.**

**Significance — dual criterion** (n_sig = min(n_sig_perm, n_sig_cv)):
- `n_sig_perm` — sequential 99th-percentile permutation null over 1,000 label shuffles (S_T's Cholesky is invariant under permutation, so we cache it once and reuse on GPU; per-shuffle cost is one cupy `solve_triangular` + a K×K eigh).
- `n_sig_cv` — 5-fold stratified k-NN classification accuracy in the LDA-projected space, with the *one-SE rule* picking the largest k whose accuracy is within 1 SE of the maximum.
- A direction is "real" only if both criteria agree.

**Carve-outs.** `mode=answer` skips concepts named `ans_*` or `answer` (residualizing the answer onto an answer-derived label is circular). `mode=norm` skips `ans_magnitude_tier`. Carved cells are recorded explicitly in the per-cell `meta.json` and `comparison/carveout_log.csv`.

**Cross-mode comparison.** After all three modes finish, `compare_residualization_modes.py` produces:
- `cross_mode_summary.csv` — every cell with all three modes' n_sig / λ_T_1 / cv_accuracy side by side.
- `cross_mode_alignment.csv` — pairwise top-1 cosine similarity (off↔answer, off↔norm, answer↔norm).
- `cross_mode_lambda_deltas.csv` and `cross_mode_accuracy_deltas.csv` — pairwise deltas.
- `matched_population_cells.csv` — only cells where Option A succeeded in **all three** modes (the headline comparison set).
- `a_vs_b_alignment.csv` — concatenated A↔B alignment per cell × mode.

**GPU acceleration.** Bottlenecks pushed to GPU via cupy + cuML:
- OLS residualization (cupy)
- CCSVD permutation null (existing Step 5 path, torch GPU SVD)
- Full-space LDA's S_T builder (cupy OAS shrinkage)
- Full-space LDA's permutation null (cupy `solve_triangular` reusing the cached lower-Cholesky factor)
- 5-fold k-NN CV-accuracy (cuML `KNeighborsClassifier`)
Sklearn / numpy fallback if cuML or cupy is unavailable on a node, with explicit log line.

### Concept registry (no concept subsampling — every CSV-grounded concept is fit)

- **Tier 1** (digits): `a, b, answer, *_units, *_tens, ans_hundreds, ans_thousands, *_num_digits` (14 concepts)
- **Tier 2** (column algebra): `column_sum_*, carry_*, running_sum_*, partial_product_*` (6 add / 16 mult)
- **Tier 3** (structural): `*_parity, parity_match, parity_xor, *_magnitude_tier, *_is_zero, ans_ends_in_zero, a_eq_b` (13)
- **Tier 4** (relational): `max_operand, min_operand, operand_diff, operand_abs_diff, larger_operand, both_zero, either_zero, both_one, either_one` (9)
- **Joints** (12 per task): operand-pair joints + carry-binding joints + multi-column joints + 2 validation joints (`(a_units, b_units, ans_units)` for addition; `(a_units, b_units, partial_product_units)` for multiplication).

Excluded by design: JSON-list columns (redundant with their per-position scalars), Tier 5 tokenization metadata (degenerate or redundant with `answer`).

### Standing rule

Every fit uses the full per-cell correct population. 5-fold CV and 1,000-permutation null are *resampling*, not subsampling. No silent truncation.

---

## 5. Repository layout

```
emnlp2026/
├── plan.md                          # plan v6 — source of truth for stage definitions and thresholds
├── README.md                        # this file
├── config.yaml                      # paths, models, dataset, tokenization, eval, ccsvd settings
│
├── check_tokenization_limits.py     # Step 1
├── generate_datasets.py             # Step 2
├── eval_and_extract.py              # Step 3
├── run_eval_and_extract.sbatch      # Step 3 SLURM array
│
├── build_embeddings.py              # Step 4 (UMAP + t-SNE per cell)
├── select_and_plot_embeddings.py    # Step 4 plotter (45 selected PNGs)
│
├── ccsvd_subspaces.py               # Step 5 + Step 6 (--mode flag for residualized re-fits)
├── check_ccsvd_toys.py              # Step 5 toy validation (1L / 2L / 3L)
├── run_ccsvd_subspaces.sbatch       # Step 5 SLURM array (1 A6000 per model)
├── plot_ccsvd_subspaces.py          # Step 5 plotter (10 plots × 3 models + 9 diagnostics)
│
├── residualize_activations.py       # Step 6 preprocessor (modes: off / answer / norm)
├── lda_subspaces.py                 # Step 6 fitter — Option A (subspace) + Option B (full 4096-D) LDA
├── check_lda_toys.py                # Step 6 toy validation (1L / 2L / 3L / 4L sample-starved)
├── compare_residualization_modes.py # Step 6 cross-mode + A-vs-B aggregator
├── run_step6.sbatch                 # Step 6 SLURM array — residualize → CCSVD re-fit → LDA, all 3 modes
├── run_step6_aggregate.sbatch       # Step 6 dependent CPU job — cross-mode comparison
│
├── docs/
│   ├── 01_tokenization_limits.md
│   ├── 02_dataset_generation.md
│   ├── 03_eval_and_extract.md
│   ├── 04_umap_tsne_embeddings.md
│   ├── 05_ccsvd_subspaces.md
│   └── 06_lda_subspaces.md          # written after Step 6 completes
│
└── data/                            # symlink → /data/user_data/anshulk/emnlp2026
    ├── models/                      # 51 GB
    ├── data/raw/                    # Step 2 outputs
    ├── activations/                 # Step 3 outputs (30 .npy files, ~3 GB)
    ├── answers/                     # Step 3 outputs (per-problem predictions)
    ├── results/
    │   ├── tokenization_limits/
    │   ├── embeddings/              # Step 4
    │   ├── ccsvd_subspaces/         # Step 5; Step 6 adds mode_answer/ and mode_norm/ subtrees
    │   ├── residualized/            # Step 6 cache: {model}/{task}_layer_{LL}_mode_{mode}.npy
    │   └── lda_subspaces/           # Step 6
    │       ├── subspace_lda/mode_{off,answer,norm}/{model}/...   # Option A (headline)
    │       ├── full_lda/mode_{off,answer,norm}/{model}/...       # Option B (audit)
    │       └── comparison/                                        # cross-mode + A↔B CSVs
    ├── figures/
    │   ├── embeddings/              # Step 4 plots
    │   └── ccsvd/                   # Step 5 plots
    └── logs/                        # Run logs + SLURM stdout/stderr
```

The `data/` symlink points at cluster scratch (`/data/user_data/anshulk/emnlp2026`); the home directory holds only code and docs.

---

## 6. How to run

All scripts read `config.yaml` for paths, model lists, prompts, and settings.

### Step 3 — extract activations (one model at a time, GPU)

```bash
sbatch run_eval_and_extract.sbatch     # array=0-2, one task per model, A6000
```

### Step 4 — UMAP + t-SNE (CPU)

```bash
python build_embeddings.py --config config.yaml
python select_and_plot_embeddings.py --config config.yaml
```

### Step 5 — CCSVD (1 GPU per model)

```bash
# Toys first (synthetic 1L / 2L / 3L sanity check)
python check_ccsvd_toys.py

# Smoke-test one cell
python ccsvd_subspaces.py --config config.yaml --model gpt-j-6b \
    --single-task addition --single-layer 14 --single-concept a_units

# Full sweep
sbatch run_ccsvd_subspaces.sbatch     # array=0-2, one task per model

# After all three tasks finish, merge per-model CSVs
python -c "import pandas as pd, glob; \
  for n in ['summary','eigenvalue_spectra','projected_centroids','null_summary','cv_per_fold']: \
    pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f'data/results/ccsvd_subspaces/*/{n}_*.csv'))]).to_csv(f'data/results/ccsvd_subspaces/{n}.csv', index=False)"

# Plot (39 PNGs total)
python plot_ccsvd_subspaces.py --config config.yaml
```

### Step 6 — Residualization + LDA (1 GPU per model, 3 modes per task)

```bash
# Toys first (synthetic 1L / 2L / 3L / 4L sanity check)
python check_lda_toys.py

# Smoke-test one cell (mode=off uses existing Step 5 CCSVD; ~44s/cell with GPU)
python lda_subspaces.py --config config.yaml --model llama-3.1-8b \
    --mode off --single-task addition --single-layer 16 --single-concept ans_units

# Full sweep — main job + dependent aggregator (chained):
JID=$(sbatch --parsable run_step6.sbatch)
sbatch --dependency=afterok:$JID run_step6_aggregate.sbatch

# Each main array task per model runs Phase 1 (residualize 90 files for that model),
# Phase 2 (CCSVD re-fits for mode=answer and mode=norm), Phase 3 (LDA in both placements
# A and B for all 3 modes). All on a single A6000.

# Manual aggregation (if you want it before the dependent job runs):
python compare_residualization_modes.py --config config.yaml
```

The pipeline writes plot-ready long-form CSVs (`eigenvalue_spectra_*`, `null_summary_*`, `cv_per_fold_*`, `cohen_d_*`, `bootstrap_lambda1_*` per mode and placement; `cross_mode_*`, `a_vs_b_alignment.csv`, `matched_population_cells.csv`, `carveout_log.csv` after aggregation).

---

## 7. Outputs and reproducibility

### Per-cell artifacts (Step 5)
Under `data/results/ccsvd_subspaces/{model_key}/{task}/layer_{LL:02d}/{concept}/`:

| File | Shape | Notes |
|---|---|---|
| `basis.npy` | `(4096, r)` float32 | orthonormal subspace basis |
| `eigenvalues.npy` | `(m−1,)` float64 | full S_B spectrum |
| `null_eigenvalues.npy` | `(1000, m−1)` float64 | permutation null |
| `threshold_99.npy` | `(m−1,)` float64 | per-index 99th percentile |
| `centroids.npy` | `(m, 4096)` float32 | per-value full-d centroids |
| `centroids_proj.npy` | `(m, r)` float32 | centroids in B |
| `projected_acts.npy` | `(N′, r)` float32 | full cloud in B |
| `cv_per_fold.npy` | `(5,)` float64 | per-fold Pearson correlations |
| `meta.json` | scalar fields, flags, sha256s | reproducibility metadata |

### Master CSVs (after merge step)
Under `data/results/ccsvd_subspaces/`:
- `summary.csv` — one row per cell with `r_dim`, top eigenvalues, flags, cv_mean, joined trustworthiness from Step 4 manifest
- `eigenvalue_spectra.csv` — long-form per-eigenvalue rows
- `projected_centroids.csv` — long-form `(cell, value, dim_idx, dim_value)`
- `null_summary.csv` — long-form percentiles of null distribution
- `cv_per_fold.csv` — long-form per-fold CV
- `run_manifest.json` — config sha, library versions, total runtime, cell counts by status

### Per-cell artifacts (Step 6)

Under `data/results/lda_subspaces/subspace_lda/mode_{mode}/{model_key}/{task}/layer_{LL}/{concept}/` (Option A; headline):

| File | Shape | Notes |
|---|---|---|
| `lda_basis_subspace.npy` | `(n_sig, r_ccsvd)` float32 | LDA directions in CCSVD subspace |
| `lda_basis_full.npy` | `(n_sig, 4096)` float32 | A's directions lifted back to 4096-D |
| `lda_eigenvalues.npy` | `(K-1,)` float64 | full LDA spectrum (λ_T) |
| `null_lda_eigenvalues.npy` | `(1000, K-1)` float64 | permutation null |
| `lda_threshold_99.npy` | `(K-1,)` float64 | per-index 99th percentile |
| `cohen_d.npy` | `(n_sig, K, K)` float64 | per-direction × class-pair Cohen's d |
| `cv_accuracy_curve.npy` | `(K-1,)` float64 | k-NN held-out accuracy per direction count |
| `cv_per_fold.npy` | `(5, K-1)` float64 | per-fold accuracy |
| `bootstrap_lambda1.npy` | `(200,)` float64 | bootstrap CI on λ_T_1 |
| `meta.json` | scalar fields, flags | mode, placement=A, n_sig_perm, n_sig_cv, n_sig, cos_sim_AB, audit_status |

Under `data/results/lda_subspaces/full_lda/mode_{mode}/{model_key}/{task}/layer_{LL}/{concept}/` (Option B; audit) — same schema, minus `bootstrap_lambda1` and `lda_basis_subspace`. B's eigenvalue magnitudes are **not** cited; only directions and `n_sig`.

### Master CSVs (Step 6, plot-ready long-form)
Per (model × mode), under `data/results/lda_subspaces/{subspace_lda,full_lda}/mode_{mode}/{model}/`:
- `summary_{model}_mode_{mode}.csv` — one row per cell.
- `eigenvalue_spectra_{model}_mode_{mode}.csv` — long-form per-eigenvalue rows.
- `null_summary_{model}_mode_{mode}.csv` — long-form per-eigenvalue null percentiles.
- `cv_per_fold_{model}_mode_{mode}.csv` — long-form per-fold per-direction accuracies.
- `cohen_d_{model}_mode_{mode}.csv` — long-form per-direction per-class-pair d.
- `bootstrap_lambda1_{model}_mode_{mode}.csv` (subspace_lda only).

Cross-mode + A↔B aggregates, under `data/results/lda_subspaces/comparison/`:
- `cross_mode_summary.csv` — one row per cell with all 3 modes' summaries side by side.
- `cross_mode_alignment.csv` — pairwise top-1 cosine similarity (off↔answer, off↔norm, answer↔norm).
- `cross_mode_lambda_deltas.csv` and `cross_mode_accuracy_deltas.csv`.
- `matched_population_cells.csv` — cells where Option A succeeded in all 3 modes.
- `a_vs_b_alignment.csv` — concatenated per-cell A↔B cosine similarity.
- `carveout_log.csv` — cells carved out from any mode.

### Reproducibility manifests
Each step writes a manifest with sha256s of its inputs and outputs:

| Step | Manifest path |
|---|---|
| 2 | `data/data/raw/build_manifest.json` |
| 3 | `data/activations/{model_key}/extraction_manifest.json` |
| 4 | `data/results/embeddings/{model_key}/{task}_layer_{LL}_manifest.json` |
| 5 | per-cell `meta.json` + per-model `manifest_{model_key}.json` |
| 6 | per-cell `meta.json` (in `subspace_lda/...` and `full_lda/...`) + per-(model, mode) `manifest_{model}_mode_{mode}.json` + `comparison/comparison_manifest.json` + `residualized/{model}/residualize_manifest_{model}.json` |

---

## 8. Environment

- **Conda env:** `geometry` at `/data/user_data/anshulk/miniconda3/envs/geometry`
- **Python** 3.11.15 · **PyTorch** 2.10.0 + CUDA 12.8 · **Transformers** 5.3.0 · **NumPy** 2.2.6 · **scikit-learn** 1.8.0 · **scipy** (latest in env) · **cupy** 14.0.1 · **cuML** 26.02 (Step 6 GPU paths: cupy OAS shrinkage, cupy `solve_triangular`, cuML `KNeighborsClassifier`)
- **GPU:** A6000 (48 GB VRAM, NVLink). Step 3 ran at batch=512 with 96–100 % GPU utilization. Step 5 runs SVD on GPU via `torch.linalg.svd`. Step 6 keeps a 4096² lower-Cholesky factor on GPU per (task, layer, mode) and reuses it across the 1,000-shuffle permutation null.
- **SLURM partition:** `general`. Step 5 sbatch requests 1 A6000 + 8 CPUs + 16 GB + 2-day wall. Step 6 sbatch requests 1 A6000 + 16 CPUs + 128 GB + 2-day wall per model. The aggregator job is gated on `afterok` of the main array.
- **Important:** SLURM scripts use the **absolute conda env Python** (`/data/user_data/anshulk/miniconda3/envs/geometry/bin/python`) to avoid system-Python (3.9) shadowing on compute nodes. `conda activate` alone is not always sufficient in batch contexts.

---

## 9. Compute and storage budget

| Step | Compute | Wall | On-disk size |
|---|---|---|---|
| 1 (tokenization) | CPU | minutes | ~2 GB CSVs |
| 2 (dataset) | CPU | seconds | ~30 MB |
| 3 (activations) | GPU (A6000 × 3 in array) | 79 s | ~3.2 GB |
| 4 (UMAP + t-SNE) | CPU | 77.9 min | ~25 MB |
| 5 (CCSVD) | GPU (A6000 × 3 in array) | ~70–90 min per task | ~3 GB |
| 6 (Residualize + LDA) | GPU (A6000 × 3 in array) | ~20–25 h per task with GPU paths | residualized cache ~14 GB; per-cell LDA artifacts + cross-mode CSVs ~5–10 GB |
| 7–9 (Stages 2–4) | GPU | plan v6 budget: ~250 GPU-h total | TBD |

**Models on disk:** ~51 GB (23 GB GPT-J + 15 GB Llama + 13 GB Pythia).

**Plan v6 timeline:** 16 weeks targeting EMNLP 2026.

---

## 10. Plan v7 corrigenda (so far)

Steps 1–3 surfaced four numerical errors in plan v6 that plan v7 should incorporate:

1. **§4.2 GPT-J cap:** 361 → **520** (plan inherited the 1.3B-variant number; the 6B cap is 520).
2. **§4.3 Llama × multiplication count:** ~6,700 → **3,390** (plan number was for a different operand range).
3. **§4.3 (new):** Pythia 6.9B cap **530**, multiplication count **3,488**.
4. **§21.5 risk:** **resolved**. The cross-model single-token intersection (3,023 problems) sidesteps the multi-token-answer-rate decision rule entirely.

Step 3 also gives plan v7 actual numbers for the §3.1 expected-correct-rate predictions (see [docs/03 §26](docs/03_eval_and_extract.md)).

---

## 11. References

- **Plan source of truth:** [plan.md](plan.md) (v6, ~84 KB). Pre-registration (Part 12), per-stage thresholds (Part 14), week-by-week timeline (Part 11), risks/fallbacks (Part 21), reviewer-attack rebuttals (Part 23), figure/table list (Part 17).
- **Parent project:** [/home/anshulk/arithmetic-geometry/](/home/anshulk/arithmetic-geometry) — emnlp2026 is a deliberate rescope. Mirror its idioms (logger, config, doc skeleton, manifest schema, validation block). Step 5 visual conventions also follow `arithmetic-geometry/plots/phase_c/` (heatmaps, scree, principal-angle trajectories).
- **External:** Kantamneni & Tegmark (2024) *Language Models Use Trigonometry to Do Addition* (KT 2024) — primary baseline for GPT-J × addition (reports 80.5 % accuracy; this work measures 84.15 % on `[0, 99]²`).

---

> **Note on Pythia.** Plan v6 is written against two models. Pythia 6.9B is added here as a third replication target — KT 2024 used it as their appendix replication, so the same six headline cells transfer cleanly. Treat plan.md as the source of truth for stage definitions; treat this README's three-model framing as the working extension.
