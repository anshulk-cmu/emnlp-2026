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
| 5 | CCSVD subspaces *(Stage 1 sub-step a)* | `ccsvd_subspaces.py` | ⏳ in progress | docs/05_ccsvd_subspaces.md (after run) | Per-cell basis, eigenvalue spectrum, 1,000-permutation null, 5-fold CV. ≈1,700 cells. SLURM array, 1 GPU per model |
| 6 | LDA refinement *(Stage 1 sub-steps b + c)* | (next) | pending | — | `S_W` + generalised eigenvalue, λ_k ∈ [0,1], bootstrap CI on λ₁, Spearman cv_correlation |
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
├── ccsvd_subspaces.py               # Step 5 (CCSVD)
├── check_ccsvd_toys.py              # Step 5 toy validation (1L / 2L / 3L)
├── run_ccsvd_subspaces.sbatch       # Step 5 SLURM array (1 A6000 per model)
├── plot_ccsvd_subspaces.py          # Step 5 plotter (10 plots × 3 models + 9 diagnostics)
│
├── docs/
│   ├── 01_tokenization_limits.md
│   ├── 02_dataset_generation.md
│   ├── 03_eval_and_extract.md
│   ├── 04_umap_tsne_embeddings.md
│   └── 05_ccsvd_subspaces.md        # written after Step 5 completes
│
└── data/                            # symlink → /data/user_data/anshulk/emnlp2026
    ├── models/                      # 51 GB
    ├── data/raw/                    # Step 2 outputs
    ├── activations/                 # Step 3 outputs (30 .npy files, ~3 GB)
    ├── answers/                     # Step 3 outputs (per-problem predictions)
    ├── results/
    │   ├── tokenization_limits/
    │   ├── embeddings/              # Step 4
    │   └── ccsvd_subspaces/         # Step 5
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

### Reproducibility manifests
Each step writes a manifest with sha256s of its inputs and outputs:

| Step | Manifest path |
|---|---|
| 2 | `data/data/raw/build_manifest.json` |
| 3 | `data/activations/{model_key}/extraction_manifest.json` |
| 4 | `data/results/embeddings/{model_key}/{task}_layer_{LL}_manifest.json` |
| 5 | per-cell `meta.json` + per-model `manifest_{model_key}.json` |

---

## 8. Environment

- **Conda env:** `geometry` at `/data/user_data/anshulk/miniconda3/envs/geometry`
- **Python** 3.11.15 · **PyTorch** 2.10.0 + CUDA 12.8 · **Transformers** 5.3.0 · **NumPy** 2.2.6 · **scikit-learn** 1.8.0 · **scipy** (latest in env)
- **GPU:** A6000 (48 GB VRAM, NVLink). Step 3 ran at batch=512 with 96–100 % GPU utilization. Step 5 runs SVD on GPU via `torch.linalg.svd`.
- **SLURM partition:** `general`. Step 5 sbatch requests 1 A6000 + 8 CPUs + 16 GB + 2-day wall.

---

## 9. Compute and storage budget

| Step | Compute | Wall | On-disk size |
|---|---|---|---|
| 1 (tokenization) | CPU | minutes | ~2 GB CSVs |
| 2 (dataset) | CPU | seconds | ~30 MB |
| 3 (activations) | GPU (A6000 × 3 in array) | 79 s | ~3.2 GB |
| 4 (UMAP + t-SNE) | CPU | 77.9 min | ~25 MB |
| 5 (CCSVD) | GPU (A6000 × 3 in array) | ~70–90 min per task | ~3 GB (estimate) |
| 6–9 (Stages 1b–4) | GPU | plan v6 budget: ~270 GPU-h total | TBD |

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
