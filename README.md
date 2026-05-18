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
| 12 | Stage 2c — BSMI-R (Bayesian Shape Manifold Inference with Refusal) | `stage2c_gplvm.py`, `stage2c_shapes.py`, `stage2c_modules.py` | **running 2026-05-18** | [docs/gplvm.md](docs/gplvm.md) | 497 eligible GPT-J cells, similar for Pythia / Llama |
| 13 | Stage 3 — Ownership test | (next) | pending | — | Orthogonalise against algebraic correlates |
| 14 | Stage 4 — Causal ablation + patching | `causal_validation.py`, `aggregate_stage4_causal.py` | **ready 2026-05-18** | — | M1 ablation + M2 patching at both subspace + geometry granularity |

**Stage 2c BSMI-R (2026-05-18 launch).** Three SLURM array jobs (4 stripes each on A6000) running on all three models. Pipeline reference: [docs/gplvm.md](docs/gplvm.md). Core principle: every module returns evidence — no early gates. Empirical-Bayes α̂ per cell, 10,000-permutation null, family-level Bayes factor, ripser-backed persistent homology, 5-fold + LOO holdout CV. Output tiers: `tier_A_named_shape` / `tier_A_named_family` / `tier_B_family` / `tier_C_dim_only` / `tier_D_refuse`.

**Stage 4 causal (2026-05-18, sbatch ready).** Tests two questions on every BSMI-R cell with a declared shape: **(M1)** does ablating the subspace at the cell's layer hurt the gold-token logit more than ablating a random subspace of matching rank? **(M2)** does patching a donor's projection onto the subspace into a recipient pull the recipient toward the donor's gold token? Both methods run at **both** the union basis `B_u` granularity AND the BSMI-R recovered geometry `Q_geom` granularity, with 5 random-subspace controls each.

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

### Stage 2c — BSMI-R: Bayesian Shape Manifold Inference with Refusal

Reference: `docs/gplvm.md`. **Core principle — no module gates early; every module returns evidence; the final Tier A/B/C/D decision is made once after all evidence is collected.**

For each eligible cell on the **Union(LDA-A, CCSVD)** subspace:

1. **Stage 0 — per-label noise estimates.** Run on the **full point cloud** (all N correct activations). Estimate per-label noise σ²_v from within-cluster scatter. The N×n_basis design matrix Phi is built by indexing per-value basis rows.

2. **Stages 1–2 — shape priors K_0..K_6:**

| # | Shape | Basis Phi(v; theta) | Latent dim | Expected Betti |
|---|---|---|---|---|
| K_0 | Generic (smooth, "none of the above") | polynomial in t = v/(K-1) | 1 | (None, None, None) |
| K_1 | Line | [1, t] | 1 | (1, 0, 0) |
| K_2 | Circle | [1, cos(2πv/P), sin(2πv/P)] | 2 | (1, 1, 0) |
| K_3 | Open helix | [1, cos, sin, t] | 1 | (1, 0, 0) — neutral |
| K_4 | Torus | [1, cos φ, sin φ, cos ψ, sin ψ] | 2 | (1, 2, 1) |
| K_5 | Concentric | [1, r(v)cos, r(v)sin, r(v)] | 2 | neutral |
| K_6 | Ribbon | [1, cos, sin, t, t·cos, t·sin] | 2 | neutral |

3. **Stages 3–5 — Bayesian shape evidence on the full point cloud.** For each shape `K_k`, compute the closed-form Gaussian marginal likelihood log p(Y | theta, K_k) on all N points (Phi rows indexed by each point's label) under the conjugate prior `W ~ N(0, alpha I)`, then **multimodal Laplace integration over period(s)**:
   log Z_k ≈ logsumexp_m [ log p(Y | theta_m, K_k) + curvature correction ] − log M.
   Stage 2a's discovered period seeds the proposal modes {P, 2P, 3P, P/2}.

4. **Stage 6 — refined evidence audit.** For the top-3 candidates plus K_0, importance-weighted refinement around the best theta produces log Z and SE.

5. **Independent evidence modules (Stages 7–14, none gate early):**
   - **Stage 7 — intrinsic dimension:** TwoNN + Levina-Bickel + PCA participation ratio, with bootstrap CIs.
   - **Stage 8 — persistent homology (DEMOTED):** Betti numbers via ripser/gudhi; `β₁ = 0` is **neutral** for line/helix/ribbon, never a rejection.
   - **Stage 9 — Fourier diagnostics:** proposes periods and two-axis flag; never decides the shape.
   - **Stage 10 — posterior differential geometry:** curvature `κ`, torsion `τ`, and the K_3 helix drift test ‖d_⊥span(cos,sin)‖ > 0. **This is what distinguishes shapes that share topology.**
   - **Stage 11 — label alignment:** Spearman or circular correlation between recovered latent and true label codes.
   - **Stage 12 — holdout adequacy:** within-label + leave-value-out. Relaxed rule `mse_winner ≤ 1.10 × min(mse_others)`, not the prior over-strict "beat runner-up by 1 SE".
   - **Stage 13 — seed and prior stability:** vary (seed × alpha-prior); reject if log-evidence std > 2 nats.
   - **Stage 14 — 1000-permutation null** with right-tailed empirical p-value `p = (1 + #ge) / (B + 1)`.

6. **Stage 15 — global BH-FDR (aggregator)** with Benjamini–Yekutieli sensitivity check.

7. **Stage 17 — Tier decision** (made once, after all evidence collected):
   - **Tier A — named shape:** evidence gap ≥ 5 nats (or 10 for small K), alignment ≥ 0.5, geom signature supports, holdout adequate, seeds stable, perm survives FDR, PH not contradictory.
   - **Tier B — geometric family:** family-vs-K_0 gap ≥ 2 nats with alignment ≥ 0.3, but exact shape ambiguous.
   - **Tier C — dimension only:** dim estimators agree, no named shape supported.
   - **Tier D — refuse:** signals disagree.

**Configuration:** `configs/stage2c.yaml`. All thresholds, FDR alpha, seeds, and prior ranges live there.

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
├── stage2c_gplvm.py                     # Stage 2c BSMI-R worker (Stages 0-17 orchestrator)
├── stage2c_shapes.py                    # K_0..K_6 shape priors + family map
├── stage2c_modules.py                   # independent evidence modules (dim, PH, Fourier, geom, align, holdout, perm)
├── aggregate_stage2c.py                 # BH-FDR + BY-FDR aggregator
├── configs/stage2c.yaml                 # BSMI-R thresholds, FDR alpha, period prior
│
├── causal_validation.py                 # Stage 4 — M1 ablation + M2 patching (subspace + geometry)
├── aggregate_stage4_causal.py           # Stage 4 aggregator
│
├── sbatch/                              # all SLURM scripts
│   ├── run_eval_and_extract.sbatch
│   ├── run_ccsvd_subspaces.sbatch
│   ├── run_step{6,7,8,9}.sbatch                 + per-step aggregators
│   ├── run_stage2a.sbatch                       + run_stage2a_aggregate.sbatch
│   ├── run_stage2b.sbatch                       + run_stage2b_aggregate.sbatch
│   ├── run_stage2c_{gptj,llama,pythia}.sbatch   # per-model, partition-aware (4 stripes each)
│   ├── run_stage2c_aggregate.sbatch
│   ├── run_stage4_{gptj,llama,pythia}.sbatch    # 4 stripes each, A6000
│   └── run_stage4_aggregate.sbatch
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
│   ├── 09_stage2b_dsw_spread_aware.md
│   └── gplvm.md                          # Stage 2c BSMI-R spec
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
        ├── stage2c_gplvm/               # Stage 2c BSMI-R per-cell artifacts
        ├── stage4_causal/                # Stage 4 per-cell ablation + patching results
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

### Stage 2c BSMI-R (three per-model jobs + aggregator)

```bash
# Single-cell smoke
python stage2c_gplvm.py --config config.yaml \
    --model gpt-j-6b --task addition --mode off --layer 14 --concept ans_tens

# Production sweep — 4 array tasks per model, A6000 each, 2-day wall
JID_GPT=$(sbatch --parsable sbatch/run_stage2c_gptj.sbatch)    # general
JID_PYT=$(sbatch --parsable sbatch/run_stage2c_pythia.sbatch)  # general
JID_LLA=$(sbatch --parsable sbatch/run_stage2c_llama.sbatch)   # preempt
sbatch --dependency=afterany:$JID_GPT:$JID_PYT:$JID_LLA \
       sbatch/run_stage2c_aggregate.sbatch
```

**Per-cell time (point cloud + closed-form Bayesian linear evidence + 10,000-perm null):**
- Addition cells (N≈8k, K≈10): ~2-3 min. Heavyweights (K≈100-200): up to ~10 min.
- Multiplication cells (N≈2.7k): ~1 min on average.

The N × n_basis Phi matrix scales linearly in N with n_basis ≤ 6, so single evidence calls stay in the millisecond range; the perm-test loop dominates. Expected total wall on 4 GPUs per model: ~6 h.

### Stage 4 causal (three per-model jobs + aggregator)

```bash
# Single-cell, all layers (model loads once)
python causal_validation.py --config config.yaml \
    --model gpt-j-6b --task addition --mode off --layer all --concept b_tens

# Sweep — depends on BSMI-R artifacts already on disk; resume-friendly
JID_G=$(sbatch --parsable sbatch/run_stage4_gptj.sbatch)
JID_P=$(sbatch --parsable sbatch/run_stage4_pythia.sbatch)
JID_L=$(sbatch --parsable sbatch/run_stage4_llama.sbatch)
sbatch --dependency=afterany:$JID_G:$JID_P:$JID_L \
       sbatch/run_stage4_aggregate.sbatch
```

Stage 4 auto-skips cells without a BSMI-R-declared shape (Tier C dim_only, Tier D refuse, low_K). Per-cell cost is ~7-15 s on a warm-loaded model.

### Standing rules

- Every fit uses the full per-cell correct population. K-fold CV, LOO CV, and 10,000-permutation nulls are *resampling*, not subsampling.
- No silent truncation, no random row sampling, no subsetting of N for any metric.
- Atomic writes (tempfile + `os.replace`) and resume-by-metadata on every per-cell job.
- Spearman ρ AND Pearson r reported side-by-side for every correlation measurement.
- Permutation / random-baseline trials: 10,000 in BSMI-R, 5 random-subspace controls in Stage 4.

---

## 6. Outputs and reproducibility

Every step writes a manifest (sha256 of inputs and outputs, library versions, config sha, total runtime). Every per-cell job writes a `metadata.json` with `computation_status: "complete"` so reruns are idempotent.

### Stage 2c BSMI-R per-cell artifacts
Under `data/results/stage2c_gplvm/{model}/{task}/mode_{mode}/layer_{LL}/{concept}/`:
- `gplvm_results.csv` — single summary row: `best_shape`, `tier`, `evidence_gap`, `evidence_gap_se`, `logZ_best`, `logZ_runnerup`, `logZ_K0`, `dim_hat` + CI, `PH_status`, `Betti`, `Fourier_period`, `geom_status`, `alignment_score`, `holdout_mse`, `holdout_mse_lvo`, `seed_stable`, `perm_p`, `verdict_pre_fdr`, `verdict_post_fdr`. Backwards-compat columns `winner_kernel`, `P_top1`, `P_top2` are kept for `causal_validation.py`.
- `evidence_per_shape.csv` — one row per K_k with `log_E`, `log_E_refined`, `log_E_se`, `alignment_score`, `geom_status`, `mse_holdout`, `mse_lvo`, `n_basis`, `best_theta`.
- `perm_null.npy` — 1000 permutation null statistics.
- `latent_winner.npy`, `W_winner.npy` — the winner shape's recovered latent (K_present × d_embed) and basis weights, ready for `causal_validation.py` to read directly without refitting.
- `metadata.json` — full evidence vector (Stage 16): dim module, PH module, Fourier module, per-shape geometry / alignment / holdout / seed_stability, permutation summary, union basis meta.

Aggregator outputs under `data/results/stage2c_gplvm/comparison/`:
- `bsmir_all.csv` — every cell, every column, with `q_BH` (BH-FDR) and `q_BY` (Benjamini-Yekutieli sensitivity) and post-FDR verdict.
- `verdict_counts_by_tier.csv` — per (model, task, mode, layer) tier histogram.
- `shape_winner_matrix.csv` — wide-form named-shape winners across modes.
- `dim_only_cells.csv` — Tier-C cells with dim + CI + Betti.
- `refusals.csv` — Tier-D cells with refusal reason.
- `cross_mode_shape_survival.csv` — per-cell shape consistency across modes.
- `aggregator_meta.json` — manifest (counts, alpha, generation time).

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
