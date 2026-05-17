# EMNLP 2026 — Geometry of Arithmetic in Language Models

**Paper (final title):** *From Linear Probes to Bayesian Manifolds: Geometry of Arithmetic in Language Models*
**Authors:** Anshul Kumar (primary, CMU), Deeksha Varshney (advisor, IIT Jodhpur), Manoj Kumar (advisor, IIT Roorkee), Barnabás Póczos (main advisor, CMU)
**Target venue:** ACL Rolling Review → EMNLP 2026 Main (long paper). Workshop fallback: BlackBoxNLP.

Tests whether the geometric structure a linear probe finds for an arithmetic concept actually belongs to that concept, or is inherited from algebraically related concepts that share residual-stream dimensions. Pipeline: linear probe → audit (variance budget, between-concept overlap, distance preservation) → Bayesian manifold characterisation → ownership orthogonalisation → causal ablation. Three pre-trained LMs (GPT-J 6B, Llama 3.1 8B, Pythia 6.9B); two tasks (addition, multiplication, both `a, b ∈ [0, 99]`); per-model correct subset.

---

## 1. Status as of 2026-05-17

| # | Step | Script | Status | Doc | Headline output |
|---|---|---|---|---|---|
| 0 | Model downloads | (manual) | done | this README §3 | 51 GB of weights in `data/models/` |
| 1 | Tokenization preflight | `check_tokenization_limits.py` | done | [docs/01](docs/01_tokenization_limits.md) | Single-token integer caps: GPT-J 520, Llama 999, Pythia 530 |
| 2 | Dataset generation | `generate_datasets.py` | done | [docs/02](docs/02_dataset_generation.md) | Addition 10,000; multiplication 3,023 |
| 3 | Activation extraction | `eval_and_extract.py` | done | [docs/03](docs/03_eval_and_extract.md) | 30 `.npy` files (5 layers × 2 tasks × 3 models) |
| 4 | UMAP + t-SNE embeddings | `build_embeddings.py` | done | [docs/04](docs/04_umap_tsne_embeddings.md) | 30 per-cell CSVs; trustworthiness ≥ 0.94 |
| 5 | CCSVD subspaces | `ccsvd_subspaces.py` | done | [docs/05](docs/05_ccsvd_subspaces.md) | ~480 fit-ok cells per model |
| 6 | Residualization + LDA | `lda_subspaces.py` | done | [docs/06](docs/06_lda_subspaces.md) | 1209 matched-population cells across 3 modes |
| 7 | Residual hunting (audit) | `residual_hunting.py` | done | [docs/07](docs/07_audit_pipeline.md) | 0/90 cells with FDR-significant residual correlates |
| 8 | Principal angles (audit) | `principal_angles.py` | done | [docs/07](docs/07_audit_pipeline.md) | Superposition rate 76–92% across cells |
| 9 | JL distance preservation | `jl_distance.py` | done | [docs/07](docs/07_audit_pipeline.md) | Spearman ρ ≥ 0.9994 every cell |
| 10 | Stage 2a — Fourier helix | `stage2a_fourier_helix.py` | done | [docs/08](docs/08_stage2a_fourier_helix.md) | 273 helix verdicts across models |
| 11 | Stage 2b — Spread-aware d_SW | `stage2b_dsw_spread_aware.py` | done | [docs/09](docs/09_stage2b_dsw_spread_aware.md) | 2561 cells with d_SW Spearman ρ |
| 12 | Stage 2c — Bayesian GPLVM | `stage2c_gplvm.py` | **fix-verified, ready to relaunch** | — | 1463 eligible cells across all 3 models |
| 13 | Stage 3 — Ownership test | (next) | pending | — | Orthogonalise against algebraic correlates |
| 14 | Stage 4 — Causal ablation | (smoke-validated) | pending | — | Δlogit on first answer token |

**Today's smoke validation (2026-05-17):** on `gpt-j-6b/multiplication/off/L14/ans_units` we confirmed Stage 2c picks K4_Torus by ~73,000 nats over runner-up, and a manual Stage-4-style ablation showed the torus is causally used — zeroing the 2D torus subspace dropped accuracy 9% while zeroing a random 2D subspace dropped 0%. The geometry is real **and** load-bearing.

---

## 2. Pipeline

### Stage 1 — Linear probe (Steps 5–6, done)

For each (model, task, layer, concept, mode) cell on the per-model correct subset:

**(a) CCSVD** — Step 5. Per-value centroids → between-class scatter → SVD → 1000-perm null → orthonormal basis `B ∈ ℝ^{4096 × r}`.

**(b) LDA refinement** — Step 6. Generalised eigenproblem `S_B w = λ S_T w`, two placements per cell:
- *Option A (headline)* — LDA inside the CCSVD subspace.
- *Option B (audit)* — LDA in the full 4096-D residualised space with Ledoit-Wolf / OAS shrinkage.

**(c) Significance** — `n_sig = min(n_sig_perm, n_sig_cv)` (1000-perm null + 5-fold k-NN with one-SE rule).

**Residualization modes**: `off` / `answer` / `norm`. The aggregator produces matched-population comparisons.

### Audit (Steps 7–9, done)

- **Step 7 — Residual hunting**: variance budget per cell; randomised SVD on residual; Marchenko-Pastur cliff; 1000-perm null on top-`n_above_mp` directions × every metadata column; BH-FDR.
- **Step 8 — Principal angles**: all-pair angles between LDA-A concept subspaces; 1000-trial empirical baseline; superposition flag.
- **Step 9 — JL distance preservation**: all `N(N−1)/2` pairs (no subsampling); Spearman + Pearson + Pythagorean check in fp64.

### Stage 2a — Fourier helix (done)

Per-cell discover-then-fit on per-value centroids:
1. **Discover** period via Whittle log-likelihood (improper flat prior on log-P).
2. **Fit** circle/helix Bayesian model; verdict ∈ {helix, circle, none, sparse_value_grid, low_K, period_inconsistent}.
3. Two-axis significance via 1000-perm null; BH-FDR.

### Stage 2b — Spread-aware Mahalanobis `d_SW` (done)

For each cell, compute the spread-aware Mahalanobis distance between every value-pair; Spearman correlation against the cyclic ground-truth distance. Tests whether the "geometry" is just a noise artefact at small per-class spread.

### Stage 2c — Bayesian manifold via point-cloud GPLVM (current)

For each eligible cell on the **Union(LDA-A, CCSVD)** subspace:
1. Project all correct activations onto B_u (typically 4096 → 6–18 dims).
2. Fit an exact GPLVM with strong-Wolfe LBFGS on the full per-cell point cloud.
3. **Six kernels compete**:

| # | Kernel | Hypothesis |
|---|---|---|
| K1 | RBF | smooth 1D curve |
| K2 | Periodic | 1D circle |
| K3 | Periodic + Linear | helix (d=2, linear on orthogonal axis) |
| K4 | Torus | two periods, d=2 |
| K5 | Concentric | two harmonics at same period, d=1 |
| K6 | Periodic + RBF | circle + smooth non-linear axis, d=2 |

4. **Verdict gate** — winner must pass all three:
   - BF gap ≥ 10 nats (K ≤ 10) / 5 nats (K ≥ 11)
   - 3-seed log-likelihood agreement within 1 nat
   - 5-fold hold-out MSE ≤ runner-up's MSE − 1 SE

5. **Significance** — 1000-perm column-shuffle null on the winner only; global BH-FDR across cells.
6. **Dim-only fallback** — for cells failing the gate, report ARD `p(d ≥ k)` and bootstrap-PR `d̂` (1–5). Cells with `d̂ ≥ 1.5` or `P(d≥1) ≥ 0.95` get a `dim_only` verdict.
7. **Confidence tier** — `HIGH` / `MEDIUM` / `LOW` / `DISCOVERY_ONLY` per plan §C.10.

**Configuration (locked 2026-05-17):**
- `SUBSAMPLE_N_MAX = 10000` (full per-cell N)
- `JITTER_INIT = 1e-4`, `JITTER_MAX = 1e-1`
- LBFGS with **`line_search_fn="strong_wolfe"`** (root fix — composite kernels deterministically fail without this)
- `parallel=True` on `fit_kernel_three_seeds` + `holdout_mse` → CUDA-stream parallelism on 3 seeds and 5 folds
- BIC penalty includes latent dimension count
- K3 d=2 with `half_normal(0.3)` outputscale prior on the linear arm

### Stages 3 and 4 (planned)

- **Stage 3 (Ownership):** for each cell with positive Stage 2c verdict, orthogonalise B_u against the pre-registered correlate set (e.g. for `ans_units` multiplication: `a, b, units(a), units(b), partial_product_units, carry_units`). Re-run Stage 2c. Verdict ∈ {owned, inherited, ambiguous}.
- **Stage 4 (Causal):** intervene at the layer where the geometry lives; ablate `B_u` and the GPLVM-identified subspace; measure Δlogit on the first answer token vs random-subspace control of matching rank. Pre-validated today on `ans_units` mult.

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

**Per-model correct subsets:**

| Model | Addition | Multiplication |
|---|---:|---:|
| GPT-J 6B | 8,415 / 10,000 (84.15 %) | 2,751 / 3,023 (91.00 %) |
| Llama 3.1 8B | 9,963 / 10,000 (99.63 %) | 2,927 / 3,023 (96.82 %) |
| Pythia 6.9B | 7,718 / 10,000 (77.18 %) | 2,757 / 3,023 (91.20 %) |

**Stage 2c eligible cells** (Stage 2a verdict ∈ {helix, circle, none, sparse_value_grid} AND both LDA-A and CCSVD bases on disk):

| Model | Cells |
|---|---:|
| GPT-J 6B | 497 |
| Llama 3.1 8B | 491 |
| Pythia 6.9B | 475 |
| **Total** | **1,463** |

---

## 4. Repository layout

```
emnlp2026/
├── plan.md                              # plan v6 — source of truth
├── README.md                            # this file
├── config.yaml                          # paths, models, dataset, eval, ccsvd, lda, residualization
│
├── check_tokenization_limits.py         # Step 1
├── generate_datasets.py                 # Step 2
├── eval_and_extract.py                  # Step 3 (activation extraction)
├── build_embeddings.py                  # Step 4 (UMAP + t-SNE)
├── select_and_plot_embeddings.py        # Step 4 plotter
│
├── ccsvd_subspaces.py                   # Step 5 (CCSVD)
├── check_ccsvd_toys.py                  # Step 5 toys
├── plot_ccsvd_subspaces.py              # Step 5 plotter
│
├── residualize_activations.py           # Step 6 phase 1 — OLS residualisation cache
├── lda_subspaces.py                     # Step 6 fitter — Option A + Option B
├── compare_residualization_modes.py     # Step 6 cross-mode aggregator
├── check_lda_toys.py                    # Step 6 toys
│
├── residual_hunting.py                  # Step 7 worker
├── principal_angles.py                  # Step 8 worker
├── jl_distance.py                       # Step 9 worker
├── aggregate_*.py                       # per-step aggregators
├── check_audit_pipeline_toys.py         # combined toys for Steps 7+8+9
├── check_step6_complete.py              # Step 7/8/9 pre-flight
│
├── stage2a_fourier_helix.py             # Stage 2a worker
├── aggregate_stage2a.py                 # Stage 2a aggregator
├── check_stage2a_toys.py                # Stage 2a toys
│
├── stage2b_dsw_spread_aware.py          # Stage 2b worker
├── aggregate_stage2b_dsw.py             # Stage 2b aggregator
├── check_stage2b_toys.py                # Stage 2b toys
│
├── stage2c_gplvm.py                     # Stage 2c worker
├── stage2c_kernels.py                   # 6-kernel zoo
├── aggregate_stage2c.py                 # Stage 2c aggregator
├── check_stage2c_toys.py                # Stage 2c toys
├── configs/stage2c.yaml                 # toy-calibrated BF thresholds + ARD epsilon
│
├── sbatch/                              # all 18 SLURM scripts (organised 2026-05-17)
│   ├── run_eval_and_extract.sbatch
│   ├── run_ccsvd_subspaces.sbatch
│   ├── run_step{6,7,8,9}.sbatch                 + per-step aggregators
│   ├── run_stage2a.sbatch                       + run_stage2a_aggregate.sbatch
│   ├── run_stage2b.sbatch                       + run_stage2b_aggregate.sbatch
│   ├── run_stage2c_{gptj,llama,pythia}.sbatch   # per-model, partition-aware
│   └── run_stage2c_aggregate.sbatch
│
├── docs/                                # one Markdown file per finished step
│   ├── 01_tokenization_limits.md
│   ├── 02_dataset_generation.md
│   ├── 03_eval_and_extract.md
│   ├── 04_umap_tsne_embeddings.md
│   ├── 05_ccsvd_subspaces.md
│   ├── 06_lda_subspaces.md
│   ├── 07_audit_pipeline.md             # Steps 7/8/9 combined
│   ├── 08_stage2a_fourier_helix.md
│   └── 09_stage2b_dsw_spread_aware.md
│
└── data/                                # symlink → /data/user_data/anshulk/emnlp2026
    ├── models/                          # 51 GB weights
    ├── data/raw/                        # Step 2 outputs (problems CSVs)
    ├── activations/                     # Step 3 .npy per (model, task, layer)
    ├── answers/                         # Step 3 per-problem predictions + correctness
    └── results/
        ├── tokenization_limits/         # Step 1
        ├── embeddings/                  # Step 4
        ├── ccsvd_subspaces/             # Step 5; Step 6 adds mode_answer/, mode_norm/ subtrees
        ├── residualized/                # Step 6 OLS-residualised activations cache
        ├── lda_subspaces/               # Step 6 (subspace_lda/ + full_lda/ + comparison/)
        ├── residual_hunting/            # Step 7
        ├── principal_angles/            # Step 8
        ├── jl_distance/                 # Step 9
        ├── stage2a_fourier_helix/       # Stage 2a
        ├── stage2b_dsw/                 # Stage 2b
        ├── stage2c_gplvm/               # Stage 2c (currently empty — relaunching post-fix)
        └── figures/
```

---

## 5. How to run

All scripts read `config.yaml`. SLURM scripts use the absolute env Python (`/data/user_data/anshulk/miniconda3/envs/geometry/bin/python`) to avoid system-Python (3.9) shadowing on babel compute nodes.

### Steps 1–9 + Stage 2a + Stage 2b (already done; commands for reference)

```bash
# Step 1 — tokenization preflight (CPU, minutes)
python check_tokenization_limits.py --config config.yaml

# Step 2 — dataset generation (CPU, seconds)
python generate_datasets.py --config config.yaml

# Step 3 — activation extraction (1 GPU per model)
sbatch sbatch/run_eval_and_extract.sbatch       # array=0-2

# Step 4 — UMAP + t-SNE (CPU, ~78 min)
python build_embeddings.py --config config.yaml

# Step 5 — CCSVD
sbatch sbatch/run_ccsvd_subspaces.sbatch

# Step 6 — Residualisation + LDA
JID=$(sbatch --parsable sbatch/run_step6.sbatch)
sbatch --dependency=afterok:$JID sbatch/run_step6_aggregate.sbatch

# Steps 7-9 — audit
S7=$(sbatch --parsable sbatch/run_step7.sbatch)
sbatch --dependency=afterok:$S7 sbatch/run_step7_aggregate.sbatch
S8=$(sbatch --parsable --dependency=afterok:$S7 sbatch/run_step8.sbatch)
sbatch --dependency=afterok:$S8 sbatch/run_step8_aggregate.sbatch
S9=$(sbatch --parsable --dependency=afterok:$S7 sbatch/run_step9.sbatch)
sbatch --dependency=afterok:$S9 sbatch/run_step9_aggregate.sbatch

# Stage 2a — Fourier helix
J2A=$(sbatch --parsable sbatch/run_stage2a.sbatch)
sbatch --dependency=afterok:$J2A sbatch/run_stage2a_aggregate.sbatch

# Stage 2b — Spread-aware d_SW
J2B=$(sbatch --parsable sbatch/run_stage2b.sbatch)
sbatch --dependency=afterok:$J2B sbatch/run_stage2b_aggregate.sbatch
```

### Stage 2c (current — three per-model jobs + aggregator)

```bash
# Toy validation (CPU+GPU, ~3 min)
python check_stage2c_toys.py --quick

# Single-cell smoke (optional)
python stage2c_gplvm.py --config config.yaml \
    --model gpt-j-6b --task multiplication --mode off --layer 14 --concept ans_units

# Production sweep — 3 per-model jobs + aggregator
JID_GPT=$(sbatch --parsable sbatch/run_stage2c_gptj.sbatch)     # 8 tasks general/normal
JID_LLA=$(sbatch --parsable sbatch/run_stage2c_llama.sbatch)    # 8 tasks preempt
JID_PYT=$(sbatch --parsable sbatch/run_stage2c_pythia.sbatch)   # 8 tasks preempt
sbatch --dependency=afterany:$JID_GPT:$JID_LLA:$JID_PYT \
       sbatch/run_stage2c_aggregate.sbatch
```

**Per-cell time at production N (with strong_wolfe, parallel streams, 2-worker GPU packing):**
- Addition cells (N≈8k): ~30–60 min/cell
- Multiplication cells (N≈2.7k): ~5–10 min/cell

**Expected wall time:** GPT-J ~24–48 h on 8 general GPUs; Llama/Pythia in parallel on preempt. Aggregator ~5 min once all three finish.

### Standing rules

- Every fit uses the full per-cell correct population. 5-fold CV and 1000-permutation nulls are *resampling*, not subsampling.
- No silent truncation, no random row sampling, no subsetting of N for any metric.
- Atomic writes (tempfile + `os.replace`) and resume-by-metadata on every per-cell job.
- Spearman ρ AND Pearson r reported side-by-side for every correlation measurement.
- Permutation / random-baseline trials: 1000 everywhere.
- BIC parsimony penalty in Stage 2c counts hyperparameters **plus** latent dimensions, so kernels with d=2 are penalised vs d=1.

---

## 6. Outputs and reproducibility

Every step writes a manifest (sha256 of inputs and outputs, library versions, config sha, total runtime). Every per-cell job writes a `metadata.json` with `computation_status: "complete"` so reruns are idempotent.

### Stage 2c per-cell artifacts
Under `data/results/stage2c_gplvm/{model}/{task}/mode_{mode}/layer_{LL}/{concept}/`:
- `gplvm_results.csv` (single summary row: winner_kernel, BF gap, p_2c, q_2c, verdict, tier, per-kernel adj_ml)
- `elbo_per_kernel_seed.npy` (6 × 3 matrix)
- `mu_stack.npy`, `noise_scalar.npy`, `Lambda_full.npy` (centroid summaries)
- `perm_null.npy` (1000-perm BIC-adjusted log-lik distribution)
- `bootstrap_d_hat.npy` (200-draw participation ratio)
- `kernel_hyperparams.json` (final periods, lengthscales, noise per winning seed)
- `ard_posterior.json` (per-axis active probabilities + bootstrap d̂)
- `union_basis_meta.json` (LDA + CCSVD contributions + SVD info)
- `metadata.json`

Aggregator outputs under `data/results/stage2c_gplvm/comparison/`:
- `gplvm_all.csv`, `verdict_counts_by_cell.csv`, `cross_mode_kernel_survival.csv`
- `kernel_concept_matrix.csv`, `dim_only_table.csv`
- `stage2a_2b_2c_survival.csv` (joins all three stages), `headline_tier_cells.csv`
- `manifest.json`

---

## 7. Environment

- **Conda env:** `geometry` at `/data/user_data/anshulk/miniconda3/envs/geometry`
- **Python** 3.11.15 · **PyTorch** 2.10.0 + CUDA 12.8 · **GPyTorch** 1.15.2 · **Transformers** 5.3.0 · **NumPy** 2.2.6 · **scikit-learn** 1.8.0 · **CuPy** 14.0.1
- **GPU:** A6000 (48 GB VRAM). Stage 2c uses TF32 matmul + FP32 inner Cholesky for ~30× speedup over FP64.
- **SLURM partitions:** `general` (normal QoS, 2-day wall, 8 GPU/user cap) and `preempt` (preempt_qos, 31-day wall, 24 GPU/user cap).
- **Important:** SLURM scripts use the absolute env Python path. `conda activate` alone is unreliable in batch contexts on babel compute nodes.

---

## 8. Compute and storage budget

| Step | Compute | Wall | On-disk |
|---|---|---|---|
| 1 — tokenization | CPU | minutes | ~2 GB |
| 2 — dataset | CPU | seconds | ~30 MB |
| 3 — activations | GPU × 3 | 79 s | ~3.2 GB |
| 4 — UMAP + t-SNE | CPU | 77.9 min | ~25 MB |
| 5 — CCSVD | GPU × 3 | ~70–90 min/task | ~3 GB |
| 6 — Residualise + LDA | GPU × 3 | ~3–6 h/task | ~14 GB residualised + ~10 GB LDA |
| 7 — Residual hunting | GPU × 6 | 17 min, 1.1 GPU-h | 7.0 GB |
| 8 — Principal angles | GPU × 6 | 12.5 h, ~44 GPU-h | 39 MB |
| 9 — JL distance | GPU × 6 | 57 min, 2.5 GPU-h | 2.7 GB |
| 2a — Fourier helix | GPU × 6 | ~3 h, 17 GPU-h | ~1 GB |
| 2b — d_SW | GPU × 6 | ~1.5 h, 9 GPU-h | ~500 MB |
| 2c — GPLVM | GPU × 24 | ~24–48 h (post-fix estimate) | est. ~10 GB |
| 3–4 — orthogonality + causal | GPU | est. 100–200 GPU-h | TBD |

Models on disk: ~51 GB (23 GB GPT-J + 15 GB Llama + 13 GB Pythia).

---

## 9. References

- **Plan source of truth:** [plan.md](plan.md) (v6). Pre-registration (Part 12), per-stage thresholds (Part 14), week-by-week timeline (Part 11), risks/fallbacks (Part 21), reviewer-attack rebuttals (Part 23), figure/table list (Part 17).
- **Parent project:** [/home/anshulk/arithmetic-geometry/](/home/anshulk/arithmetic-geometry) — emnlp2026 is a rescope. Mirror its idioms (logger, config, doc skeleton, manifest schema, validation block).
- **External:** Kantamneni & Tegmark (2024) *Language Models Use Trigonometry to Do Addition* (KT 2024) — primary baseline for GPT-J × addition.

---

## 10. Recent decisions / changelog

**2026-05-17 — Stage 2c production-readiness fixes**
- Fixed `noise_scalar` NameError in `analyze_cell` result-dict assembly (root cause of the 14-h zero-output sweep)
- Reorganised all 18 sbatch files into `sbatch/` folder; deleted the dead `run_stage2c.sbatch` (24-task preempt-only generic, superseded by per-model files)
- Enabled `line_search_fn="strong_wolfe"` in LBFGS (root fix for composite-kernel Cholesky failures at production N)
- Bumped `JITTER_MAX` to 1e-1 (defensive)
- Reverted K3 to d=2 with tighter `half_normal(0.3)` outputscale prior on linear arm
- Added `dim_only` fallback for inconclusive cells (reports ARD `P(d≥k)` and bootstrap d̂ up to 5)
- BIC adjustment now includes `d_latent` as effective parameter count
- Smoke-validated K4_Torus causal: 2D ablation drops accuracy 9% on `ans_units` mult vs 0% for random 2D control
- Split production into 3 per-model jobs: `gptj` on general/normal, `llama`+`pythia` on preempt

**Earlier**
- 2026-05-12 — Stage 2b spread-aware d_SW done across 2,561 cells
- 2026-05-10 — Stage 2a Fourier helix done with 1000-perm Whittle null
- 2026-05-08 — Audit Steps 7/8/9 complete: 0/90 cells have FDR-significant residual correlates
- 2026-05-05 — Step 6 residualisation + LDA done across 3 modes
