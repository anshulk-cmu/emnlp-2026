# EMNLP 2026 — Geometry of Arithmetic in Language Models

**Paper (working title):** *From Linear Probes to Bayesian Manifolds: Geometry of Arithmetic in Language Models*
**Authors:** Anshul Kumar (first), Barnabás Póczos (senior)
**Target venue:** ACL Rolling Review → EMNLP 2026 Main (long paper). Workshop fallback: BlackBoxNLP.

We test whether the geometric structure a linear probe finds for an arithmetic concept actually belongs to that concept, or is inherited from algebraically related concepts that share residual-stream dimensions. Pipeline: linear probe → audit (variance budget, between-concept overlap, distance preservation) → Bayesian manifold characterisation → ownership orthogonalisation → causal ablation. Three pre-trained LMs (GPT-J 6B, Llama 3.1 8B, Pythia 6.9B); two tasks (addition, multiplication, both `a, b ∈ [0, 99]`); per-model correct subset.

---

## 1. Status

| # | Step | Script | Status | Doc | Headline output |
|---|---|---|---|---|---|
| 0 | Model downloads | (manual) | done | this README §3 | 51 GB of weights in `data/models/` |
| 1 | Tokenization preflight | `check_tokenization_limits.py` | done | [docs/01](docs/01_tokenization_limits.md) | Single-token integer caps: GPT-J 520, Llama 999, Pythia 530 |
| 2 | Dataset generation | `generate_datasets.py` | done | [docs/02](docs/02_dataset_generation.md) | Addition 10,000; multiplication 3,023 (cross-model single-token intersection) |
| 3 | Activation extraction | `eval_and_extract.py` | done | [docs/03](docs/03_eval_and_extract.md) | 30 `.npy` files (5 layers × 2 tasks × 3 models), 79 s SLURM wall |
| 4 | UMAP + t-SNE embeddings | `build_embeddings.py` | done | [docs/04](docs/04_umap_tsne_embeddings.md) | 30 per-cell CSVs; trustworthiness ≥ 0.94 on every cell |
| 5 | CCSVD subspaces *(Stage 1 sub-step a)* | `ccsvd_subspaces.py` | done | [docs/05](docs/05_ccsvd_subspaces.md) | Per-cell orthonormal basis + 1000-perm null + 5-fold CV; ~480 fit-ok cells per model |
| 6 | Residualization + LDA refinement *(Stage 1 sub-steps b + c)* | `residualize_activations.py` + `ccsvd_subspaces.py --mode` + `lda_subspaces.py` | done | docs/06_lda_subspaces.md *(in flight)* | 3 modes × 2 placements (Option A in CCSVD subspace, Option B in full 4096-D with shrinkage); 1209 cells matched-population across modes |
| 7 | Residual hunting *(audit)* | `residual_hunting.py` | done | [docs/07](docs/07_audit_pipeline.md) | Per-cell variance budget 70-95% across 90 cells; 1000-perm BH-FDR returns 0/90 significant residual correlates; 2 union variants (`merged`, `generous`); Stage 3 correlate-set unions pre-computed |
| 8 | Principal angles *(audit)* | `principal_angles.py` | done | [docs/07](docs/07_audit_pipeline.md) | All-pair angles between LDA-A concept subspaces (orthonormalised on load); 1000-trial empirical baseline cached on disk; superposition rate 76-92% across cells; multiplication consistently more entangled than addition |
| 9 | JL distance preservation *(audit)* | `jl_distance.py` | done | [docs/07](docs/07_audit_pipeline.md) | All N(N−1)/2 pairs (no subsampling); Spearman ρ ≥ 0.9994 across every cell × variant; distance-variance-explained ≥ 0.999 on addition, ≥ 0.992 on multiplication; full-pair float64 Pythagorean check passes |
| 10 | Stage 2 — Bayesian manifold | (next) | pending | — | Centroid Fourier helix → spread-aware `d_SW` → GPLVM → RBF-precision VAE |
| 11 | Stage 3 — Ownership test | (next) | pending | — | Orthogonalise against algebraic correlates; verdict ∈ {owned, inherited, ambiguous} |
| 12 | Stage 4 — Causal ablation | (next) | pending | — | Δlogit on first answer token |

---

## 2. Pipeline

### Stage 1 — Linear probe (Steps 5–6, done)

For each (model, task, layer, concept, mode) cell on the per-model correct subset:

**(a) CCSVD** — Step 5. Per-value centroids → between-class scatter → SVD → 1000-permutation null filter → orthonormal basis `B ∈ ℝ^{4096 × r}`.

**(b) LDA refinement** — Step 6. Generalised eigenproblem `S_B w = λ S_T w`, λ ∈ [0,1]. Two placements per cell:
- *Option A (headline)* — LDA inside the CCSVD subspace. N/r ≈ 100+ → eigenvalues trustworthy.
- *Option B (audit)* — LDA in the full 4096-D residualised space with Ledoit-Wolf / OAS shrinkage. Eigenvalue magnitudes not cited; only directions and `n_sig`. Cosine similarity vs A reported per cell.

**(c) Significance** — dual criterion `n_sig = min(n_sig_perm, n_sig_cv)`:
- `n_sig_perm` — sequential 99th-percentile permutation null over 1000 label shuffles.
- `n_sig_cv` — 5-fold k-NN classification accuracy with the one-SE rule.

**Residualization modes** — Step 6 runs three modes in parallel and produces a matched-population comparison table:
- `off` — raw activations.
- `answer` — OLS-regress activations on the gold answer, keep residual. Carves out `ans_*` and `answer` concepts (circular).
- `norm` — OLS-regress on `||x||₂`, keep residual. Carves out `ans_magnitude_tier`.

### Audit (Steps 7–9, done)

Three audit phases between Stage 1 and Stage 2. Their job is to honestly answer:
- "Have we captured every linearly organised concept?" — *Step 7 (residual hunting).*
- "Do concept subspaces overlap, and which Stage-3 tests will be load-bearing?" — *Step 8 (principal angles).*
- "Does our union-of-concepts subspace preserve the pairwise geometry?" — *Step 9 (JL distance preservation).*

**Step 7 — Residual hunting** (per (model, task, mode, layer) cell, 2 union variants each):
- Union variants: `merged` = SVD-orthonormalisation of (CCSVD ∪ LDA-A) bases + mode-specific β scalar direction (β_answer for `off`; β_norm + β_answer for `norm`); `generous` = `merged` ∪ LDA-B (audit-only).
- Project X onto V_all; randomised SVD on the residual → top 500 eigenvalues.
- Marchenko-Pastur cliff: trace-based σ²; `n_above_mp`; `mp_reliable_flag = (γ < 0.7)`.
- Correlation sweep (merged only — LDA-B dominates `generous` with N/d noise, so its residual is not swept): top n_above_mp directions × every metadata column + derived columns (carry interactions, mod-10 sums, partial-product cross terms, predicted-digit features); observed Spearman + Pearson, 1000-permutation null, Benjamini-Hochberg FDR across the grid.
- Stage 3 correlate-set unions pre-computed per task target (`ans_units`, `ans_tens`, `answer`, `carry_units`).

**Step 8 — Principal angles** (per cell):
- All `C(K, 2)` pairs of LDA-A bases (orthonormalised on load).
- Principal angles via SVD of B_a @ B_b.T.
- 1000-trial empirical random baseline per (min(dim_a, dim_b), max(dim_a, dim_b)), cached on disk across runs.
- Superposition flag: `angle_1 < baseline_p5 − 10°`. Per-pair empirical p-value and BH-FDR.

**Step 9 — JL distance preservation** (per cell, both variants):
- All `N(N−1)/2` pairs, no subsampling.
- Pairwise distances in full 4096-D and projected k-D, batched on GPU.
- Spearman ρ + Pearson r + mean/max relative error + distance-variance-explained.
- Full-pair Pythagorean check in float64 on GPU.

### Audit headlines (production run, 2026-05-12)

90 cells (3 models × 2 tasks × 3 modes × 5 layers); 0 cells failed; 270 cell evaluations across the three steps.

| Headline | Number |
|---|---|
| Variance captured by named-concept union (mode=off, merged, median per model) | GPT-J 0.866 / Llama 0.858 / Pythia 0.949 on addition; 0.739 / 0.755 / 0.874 on multiplication |
| FDR-significant residual correlates after 1000-perm BH (|ρ_s| > 0.15 AND q < 0.05) | **0 / 90 cells** |
| Median superposition rate across concept pairs | 80% on addition; 90% on multiplication |
| Pairwise-distance preservation Spearman ρ (mode=off, merged) | ≥ 0.9994 every cell; addition median 0.99996; multiplication median 0.99963 |
| MP cliff regime | Reliable (γ < 0.7) on every addition cell; unreliable (γ > 1) on every multiplication cell |
| Total chained wall time | ~13 hours on 6 concurrent A6000s |

The full per-cell numbers, the cross-mode breakdown, the mathematical framework, and the analysis appendix live in [docs/07_audit_pipeline.md](docs/07_audit_pipeline.md) (3,060 lines).

### Stages 2–4 (Steps 10–12, planned)

**Stage 2** — Bayesian manifold characterisation: (a) centroid Fourier helix, (b) spread-aware Mahalanobis `d_SW`, (c) Bayesian GPLVM, (d) RBF-precision VAE.
**Stage 3** — Ownership test via orthogonalisation against pre-registered algebraic correlates; re-run Stage 2.
**Stage 4** — Causal ablation: raw vs orthogonalised subspace ablation; measure Δlogit on first answer token.

### Headline matrix to populate

|  | Addition | Multiplication |
|---|---|---|
| Operand | predicted owned (trivial) | predicted owned (trivial) |
| Intermediate | TEST | predicted inherited (Phase H from arithmetic-geometry replicated 419 / 419) |
| Output | TEST | TEST |

Either Finding A (asymmetric ownership across tasks) or Finding B (uniform inheritance) is publishable — the pipeline is the constant.

---

## 3. Models, tasks, accuracies

| | GPT-J 6B (primary) | Llama 3.1 8B | Pythia 6.9B |
|---|---|---|---|
| HF repo | `EleutherAI/gpt-j-6B` | `meta-llama/Llama-3.1-8B` | `EleutherAI/pythia-6.9b` |
| Architecture | 28 layers, 4096 hidden | 32 layers, SwiGLU, GQA | 32 layers, parallel attn+MLP |
| Single-token integer cap | 520 | 999 | 530 |
| Layers extracted | 4, 8, 14, 20, 24 | 4, 8, 16, 24, 28 | 4, 8, 16, 24, 28 |
| Headline layer | 14 | 16 | 16 |
| Weights on disk | ~23 GB | ~15 GB | ~13 GB |

**Tasks.** Addition `a + b`, `a, b ∈ [0, 99]`, 10,000 problems. Multiplication `a × b`, same range, 3,023 cross-model single-token intersection.

**Correctness.** First-answer-token match against the gold first-token id (precomputed in Step 1).

**Per-model correct subsets (the population every later step runs on):**

| Model | Addition | Multiplication |
|---|---:|---:|
| GPT-J 6B | 8,415 / 10,000 (84.15 %) | 2,751 / 3,023 (91.00 %) |
| Llama 3.1 8B | 9,963 / 10,000 (99.63 %) | 2,927 / 3,023 (96.82 %) |
| Pythia 6.9B | 7,718 / 10,000 (77.18 %) | 2,757 / 3,023 (91.20 %) |

---

## 4. Repository layout

```
emnlp2026/
├── plan.md                              # plan v6 — source of truth for stage definitions, thresholds, pre-registration
├── README.md                            # this file
├── config.yaml                          # paths, models, dataset, tokenization, eval, ccsvd, lda, residualization
│
├── check_tokenization_limits.py         # Step 1
├── generate_datasets.py                 # Step 2
├── eval_and_extract.py                  # Step 3 (activation extraction)
├── run_eval_and_extract.sbatch          # Step 3 SLURM array
├── build_embeddings.py                  # Step 4 (UMAP + t-SNE)
├── select_and_plot_embeddings.py        # Step 4 plotter
│
├── ccsvd_subspaces.py                   # Step 5 (CCSVD); Step 6 re-fits with --mode flag
├── check_ccsvd_toys.py                  # Step 5 toys
├── run_ccsvd_subspaces.sbatch           # Step 5 SLURM array
├── plot_ccsvd_subspaces.py              # Step 5 plotter
│
├── residualize_activations.py           # Step 6 phase 1 — OLS residualisation cache
├── lda_subspaces.py                     # Step 6 fitter — Option A + Option B
├── compare_residualization_modes.py     # Step 6 cross-mode + A↔B aggregator
├── check_lda_toys.py                    # Step 6 toys
├── run_step6.sbatch                     # Step 6 SLURM array (residualise → CCSVD re-fit → LDA, all 3 modes)
├── run_step6_aggregate.sbatch           # Step 6 dependent CPU aggregator
│
├── check_step6_complete.py              # Step 7/8/9 pre-flight: verifies Step 6 outputs are complete
├── residual_hunting.py                  # Step 7 worker
├── principal_angles.py                  # Step 8 worker
├── jl_distance.py                       # Step 9 worker
├── aggregate_residual_hunting.py        # Step 7 aggregator
├── aggregate_principal_angles.py        # Step 8 aggregator
├── aggregate_jl_distance.py             # Step 9 aggregator
├── check_audit_pipeline_toys.py         # Combined toys for Steps 7+8+9
├── run_step7.sbatch                     # Step 7 SLURM array (6 array tasks, max 3 concurrent A6000s)
├── run_step8.sbatch                     # Step 8 SLURM array (depends on Step 7 outputs)
├── run_step9.sbatch                     # Step 9 SLURM array (depends on Step 7 outputs)
├── run_step7_aggregate.sbatch           # Step 7 dependent CPU aggregator
├── run_step8_aggregate.sbatch           # Step 8 dependent CPU aggregator
├── run_step9_aggregate.sbatch           # Step 9 dependent CPU aggregator
│
├── docs/                                # one Markdown file per finished step
│   ├── 01_tokenization_limits.md
│   ├── 02_dataset_generation.md
│   ├── 03_eval_and_extract.md
│   ├── 04_umap_tsne_embeddings.md
│   ├── 05_ccsvd_subspaces.md
│   ├── 06_lda_subspaces.md
│   └── 07_audit_pipeline.md             # combined Steps 7/8/9 report (3,060 lines)
│
└── data/                                # symlink → /data/user_data/anshulk/emnlp2026
    ├── models/                          # 51 GB weights
    ├── data/raw/                        # Step 2 outputs (problems CSVs)
    ├── activations/                     # Step 3 (.npy per (model, task, layer))
    ├── answers/                         # Step 3 (per-problem predictions + correctness)
    └── results/
        ├── tokenization_limits/         # Step 1
        ├── embeddings/                  # Step 4
        ├── ccsvd_subspaces/             # Step 5; Step 6 adds mode_answer/, mode_norm/ subtrees
        ├── residualized/                # Step 6 OLS-residualised activations cache
        ├── lda_subspaces/               # Step 6 (subspace_lda/ + full_lda/ + comparison/)
        ├── residual_hunting/            # Step 7
        ├── principal_angles/            # Step 8
        ├── jl_distance/                 # Step 9
        └── figures/                     # plots (off-path; can regenerate from CSVs)
```

The `data/` symlink points at cluster scratch (`/data/user_data/anshulk/emnlp2026`); the home directory holds only code and docs.

---

## 5. How to run

All scripts read `config.yaml` for paths, model lists, prompts, and settings. SLURM scripts use the absolute conda env Python (`/data/user_data/anshulk/miniconda3/envs/geometry/bin/python`) to avoid system-Python (3.9) shadowing on compute nodes.

### Steps 1–6 (already done; commands here for reference)

```bash
# Step 1 — tokenization preflight (CPU, minutes)
python check_tokenization_limits.py --config config.yaml

# Step 2 — dataset generation (CPU, seconds)
python generate_datasets.py --config config.yaml

# Step 3 — activation extraction (1 GPU per model)
sbatch run_eval_and_extract.sbatch        # array=0-2

# Step 4 — UMAP + t-SNE (CPU, ~78 min)
python build_embeddings.py --config config.yaml
python select_and_plot_embeddings.py --config config.yaml

# Step 5 — CCSVD (1 GPU per model, ~70-90 min/task)
python check_ccsvd_toys.py
sbatch run_ccsvd_subspaces.sbatch

# Step 6 — Residualisation + LDA (1 GPU per model, ~3-6 h/task)
python check_lda_toys.py
JID=$(sbatch --parsable run_step6.sbatch)
sbatch --dependency=afterok:$JID run_step6_aggregate.sbatch
```

### Steps 7–9 (audit phases, ready to launch)

```bash
# Pre-flight: confirm Step 6 outputs are complete (read-only)
python check_step6_complete.py --config config.yaml

# Toys: validate Step 7, 8, 9 algorithms on synthetic data (CPU+GPU, ~3 min)
python check_audit_pipeline_toys.py

# Step 7 — Residual hunting (must run first; Steps 8 and 9 consume its union bases)
S7=$(sbatch --parsable run_step7.sbatch)
sbatch --parsable --dependency=afterok:$S7 run_step7_aggregate.sbatch

# Step 8 — Principal angles (parallel with Step 9)
S8=$(sbatch --parsable --dependency=afterok:$S7 run_step8.sbatch)
sbatch --parsable --dependency=afterok:$S8 run_step8_aggregate.sbatch

# Step 9 — JL distance preservation
S9=$(sbatch --parsable --dependency=afterok:$S7 run_step9.sbatch)
sbatch --parsable --dependency=afterok:$S9 run_step9_aggregate.sbatch
```

**Job-array geometry.** Each worker sbatch uses `--array=0-5%3`: 6 tasks (3 models × 2 tasks), max 3 concurrent (one A6000 each). Each array task processes all 3 modes × 5 layers = 15 cells sequentially. All sbatch files specify `--time=2-00:00:00` (48 h cluster max) as a safety margin; expected wall is ~3 h for Step 7, ~30 min for Step 8, ~1.5 h for Step 9.

**Smoke test a single cell** (recommended before the full sweep):

```bash
python residual_hunting.py --config config.yaml --model gpt-j-6b --task multiplication --mode off --layer 14
python principal_angles.py --config config.yaml --model gpt-j-6b --task multiplication --mode off --layer 14
python jl_distance.py --config config.yaml --model gpt-j-6b --task multiplication --mode off --layer 14
```

### Standing rules

- Every fit uses the full per-cell correct population. 5-fold CV and 1000-permutation nulls are *resampling*, not subsampling.
- No silent truncation, no random row sampling, no subsetting of N for any metric.
- Atomic writes (tempfile + `os.replace`) and resume-by-metadata on every per-cell job.
- Spearman ρ AND Pearson r reported side-by-side for every correlation measurement.
- Permutation / random-baseline trials: 1000 everywhere.

---

## 6. Outputs and reproducibility

Every step writes a manifest (sha256 of inputs and outputs, library versions, config sha, total runtime). Every per-cell job writes a `metadata.json` with `computation_status: "complete"` so reruns are idempotent.

### Step 5 per-cell artifacts
Under `data/results/ccsvd_subspaces/{model}/{task}/layer_{LL}/{concept}/`:
`basis.npy (4096, r)`, `eigenvalues.npy`, `null_eigenvalues.npy (1000, m-1)`, `threshold_99.npy`, `centroids.npy`, `centroids_proj.npy`, `projected_acts.npy`, `cv_per_fold.npy`, `meta.json`.

### Step 6 per-cell artifacts
Under `data/results/lda_subspaces/{subspace_lda,full_lda}/mode_{mode}/{model}/{task}/layer_{LL}/{concept}/`:
`lda_basis_subspace.npy (n_sig, r)` *(Option A only)*, `lda_basis_full.npy (4096, n_sig)`, `lda_eigenvalues.npy`, `null_lda_eigenvalues.npy (1000, K-1)`, `lda_threshold_99.npy`, `cohen_d.npy`, `cv_accuracy_curve.npy`, `cv_per_fold.npy`, `bootstrap_lambda1.npy` *(A only)*, `meta.json`.

Cross-mode aggregates under `data/results/lda_subspaces/comparison/`: `cross_mode_summary.csv`, `cross_mode_alignment.csv`, `cross_mode_lambda_deltas.csv`, `cross_mode_accuracy_deltas.csv`, `matched_population_cells.csv` (1209 rows where all 3 modes succeeded), `a_vs_b_alignment.csv`, `carveout_log.csv` (270 rows).

### Step 7 per-cell artifacts
Under `data/results/residual_hunting/{model}/{task}/mode_{mode}/layer_{LL}/`:
`union_basis_merged.npy (k_merged, 4096)`, `union_basis_generous.npy (k_generous, 4096)`, `eigenvalues_{variant}.npy`, `eigenvectors_{variant}.npy (500, 4096)`, `mp_info_{variant}.json`, `correlation_sweep_merged.csv`, `union_meta.json`, `stage3_unions/union_correlates_<target>.npy + .json`, `metadata.json`.

Per-model summaries: `data/results/residual_hunting/{model}/summary_{model}_{task}_mode_{mode}.csv` (2 rows per cell — one per variant).

Aggregator outputs under `data/results/residual_hunting/comparison/`: `summary_all.csv`, `var_explained_cross_mode.csv`, `n_above_mp_cross_mode.csv`, `k_union_cross_mode.csv`, `gamma_cross_mode.csv`, `residual_top_correlate_cross_mode.csv`, `variant_delta.csv`, `summary_with_matched_count.csv`.

### Step 8 per-cell artifacts
Under `data/results/principal_angles/{model}/{task}/mode_{mode}/layer_{LL}/`:
`angles_pairwise.csv` (one row per pair: concept_a, concept_b, tier_a, tier_b, dim_a, dim_b, angle_1..angle_5, angle_median, angle_max, baseline_theta1_{mean,std,p1,p5}, perm_p, superposition_flag, fdr_q), `self_angles.csv` (sanity check: every concept basis vs itself, max angle should be < 1°), `metadata.json`.

Per-model summary: `data/results/principal_angles/{model}/summary_{model}_{task}_mode_{mode}.csv`. Shared baseline cache at `data/results/principal_angles/random_baseline_cache.npy`.

Aggregator outputs under `data/results/principal_angles/comparison/`: `pairwise_all.csv`, `summary_all.csv`, `superposition_rate_by_cell.csv`, `superposition_by_tier_pair.csv`, `cross_mode_superposition.csv`.

### Step 9 per-cell artifacts
Under `data/results/jl_distance/{model}/{task}/mode_{mode}/layer_{LL}/`:
`jl_metrics_{merged,generous}.json` (Spearman ρ, Pearson r, mean/max relative error, distance variance explained, Pythagorean max/mean relative error + violation count), `d_full_{variant}.npy` + `d_proj_{variant}.npy` *(when N ≤ 5000)* or `d_full_sample_{variant}.npy` + `d_proj_sample_{variant}.npy` *(10k subsample for plotting; metrics computed on all pairs regardless)*, `d_hist_{variant}.npz` (2-D histogram of d_full vs d_proj), `metadata.json`.

Per-model summary: `data/results/jl_distance/{model}/summary_{model}_{task}_mode_{mode}.csv`.

Aggregator outputs under `data/results/jl_distance/comparison/`: `summary_all.csv`, `spearman_cross_mode.csv`, `pearson_cross_mode.csv`, `distance_var_explained_cross_mode.csv`, `mean_rel_error_cross_mode.csv`, `max_rel_error_cross_mode.csv`, `pyth_max_rel_error_cross_cell.csv`, `variant_delta_jl.csv`.

---

## 7. Environment

- **Conda env:** `geometry` at `/data/user_data/anshulk/miniconda3/envs/geometry`
- **Python** 3.11.15 · **PyTorch** 2.10.0 + CUDA 12.8 · **Transformers** 5.3.0 · **NumPy** 2.2.6 · **scikit-learn** 1.8.0 · **scipy** (latest in env) · **cupy** 14.0.1 · **cuML** 26.02
- **GPU:** A6000 (48 GB VRAM, NVLink). Step 3 ran at batch=512 with 96–100 % util on 6–8B models in bf16. Step 5 runs SVD on GPU via `torch.linalg.svd`. Step 6 keeps a 4096² lower-Cholesky factor on GPU per (task, layer, mode). Steps 7/9 use CuPy projection and float64 Pythagorean check.
- **SLURM partition:** `general`. Worker sbatch requests 1 A6000 + 16 CPUs + 128 GB + 2-day wall. Aggregator sbatch (CPU-only) requests 8 CPUs + 64 GB + 2-day wall.
- **Important:** SLURM scripts use the absolute env Python path. `conda activate` alone is unreliable in batch contexts on babel compute nodes.

---

## 8. Compute and storage budget

| Step | Compute | Wall | On-disk |
|---|---|---|---|
| 1 — tokenization | CPU | minutes | ~2 GB CSVs |
| 2 — dataset | CPU | seconds | ~30 MB |
| 3 — activations | GPU (A6000 × 3) | 79 s | ~3.2 GB |
| 4 — UMAP + t-SNE | CPU | 77.9 min | ~25 MB |
| 5 — CCSVD | GPU (A6000 × 3) | ~70–90 min/task | ~3 GB |
| 6 — Residualise + LDA | GPU (A6000 × 3) | ~3–6 h/task | residualised cache ~14 GB + LDA artifacts ~5–10 GB |
| 7 — Residual hunting | GPU (A6000 × 6) | 17 min wall, 1.1 GPU-h total | 7.0 GB |
| 8 — Principal angles | GPU (A6000 × 6) | 12.5 h wall, ~44 GPU-h total | 39 MB |
| 9 — JL distance | GPU (A6000 × 6) | 57 min wall, 2.5 GPU-h total | 2.7 GB |
| 10–12 — Stages 2–4 | GPU | plan v6 budget ~250 GPU-h | TBD |

Models on disk: ~51 GB (23 GB GPT-J + 15 GB Llama + 13 GB Pythia).

---

## 9. References

- **Plan source of truth:** [plan.md](plan.md) (v6). Pre-registration (Part 12), per-stage thresholds (Part 14), week-by-week timeline (Part 11), risks/fallbacks (Part 21), reviewer-attack rebuttals (Part 23), figure/table list (Part 17).
- **Steps 7–9 plan:** `~/.claude/plans/lets-write-a-perfect-wiggly-feigenbaum.md` — the audit-phases spec the current Step 7/8/9 implementations follow.
- **Parent project:** [/home/anshulk/arithmetic-geometry/](/home/anshulk/arithmetic-geometry) — emnlp2026 is a rescope. Mirror its idioms (logger, config, doc skeleton, manifest schema, validation block). Steps 7–9 port Phase E, Phase F, and Phase JL respectively.
- **External:** Kantamneni & Tegmark (2024) *Language Models Use Trigonometry to Do Addition* (KT 2024) — primary baseline for GPT-J × addition (reports 80.5 % accuracy; this work measures 84.15 % on `[0, 99]²`).
