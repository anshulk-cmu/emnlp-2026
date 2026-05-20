# EMNLP 2026 — Geometry of Arithmetic in Language Models

**Paper:** *From Linear Probes to Bayesian Manifolds: Geometry of Arithmetic in Language Models*
**Authors:** Anshul Kumar (CMU, primary) · Deeksha Varshney (IIT Jodhpur) · Manoj Kumar (IIT Roorkee) · Barnabás Póczos (CMU, advisor)
**Venue:** ACL Rolling Review → EMNLP 2026 Main. Fallback: BlackBoxNLP.

## 1. Headline finding

When a linear probe locates a clean geometric shape (helix, torus, ribbon, …) for an arithmetic concept in a transformer's residual stream, **does the shape belong to that concept, or is it inherited from algebraically related concepts that share residual-stream dimensions, and is the shape actually causally used by the model?**

Across **1 463 (model × task × layer × mode × concept) cells** on **GPT-J 6B, Llama 3.1 8B, Pythia 6.9B** doing **addition** and **multiplication** with `a, b ∈ [0, 99]`, we find:

| | GPT-J | Llama | Pythia | All 3 |
|---|---|---|---|---|
| Owned shapes (concept-specific after orthogonalisation) | 74 / 420 (18 %) | 88 / 465 (19 %) | 91 / 445 (20 %) | **253 / 1 330 (19 %)** |
| Inherited shapes | 346 (82 %) | 377 (81 %) | 354 (80 %) | **1 077 (81 %)** |
| Mean B_u causal Δacc | −15.4 % | −5.0 % | −18.1 % | — |
| Mean Q_geom causal Δacc | −10.6 % | −2.1 % | −10.8 % | — |
| geom / B_u causal ratio | 64 % | 42 % | 66 % | — |

**Four-in-five clean LRH-style shapes are inherited.** The 19 % that *are* owned carry roughly two-thirds of the full probe-subspace's causal weight — the geometry **is** the operational handle, but only where ownership holds.

## 2. Pipeline (15 steps, all complete)

| Step | Script | Output | Result |
|---|---|---|---|
| 1. Tokenization preflight | `check_tokenization_limits.py` | per-model single-token integer caps | GPT-J 520, Llama 999, Pythia 530 |
| 2. Dataset generation | `generate_datasets.py` | `data/raw/{addition,multiplication}_problems.csv` | 10 000 add, 3 023 mult |
| 3. Activation extraction | `eval_and_extract.py` | 30 `.npy` files (5 L × 2 T × 3 M) | per-model correct: 84–99 % add, 91–97 % mult |
| 4. UMAP + t-SNE | `build_embeddings.py` | 30 per-cell CSVs | trustworthiness ≥ 0.94 |
| 5. CCSVD subspaces | `ccsvd_subspaces.py` | per-cell orthonormal basis B ∈ ℝ^{4096×r} | ~480 fit-ok cells / model |
| 6. Residualisation + LDA | `lda_subspaces.py` | LDA-A subspace per cell, 3 modes (off/answer/norm) | 1 209 matched cells |
| 7. Residual hunting | `residual_hunting.py` | per-cell BH-FDR scan | **0 / 90 cells with FDR-significant residual correlates** |
| 8. Principal angles | `principal_angles.py` | concept-pair angles + empirical baseline | superposition rate **76–92 %** across cells |
| 9. JL distance preservation | `jl_distance.py` | Spearman ρ on all N(N-1)/2 pairs | ρ ≥ **0.9994** every cell |
| 10. Stage 2a — Fourier helix | `stage2a_fourier_helix.py` | per-cell verdict ∈ {helix, circle, none, …} | 667 helices, 141 circles / 6 886 rows |
| 11. Stage 2b — spread-aware d_SW | `stage2b_dsw_spread_aware.py` | Spearman vs cyclic ground-truth | 400 / 2 561 spread-confirmed |
| 12. Stage 2c — BSMI-R | `stage2c_gplvm.py`, `stage2c_shapes.py`, `stage2c_modules.py` | 7-shape Bayesian evidence, 10 k perm null, Tier A/B/C/D | **1 463 / 1 463 cells**, 415 Tier-A named-shape (28 %), 922 Tier-B family (63 %), 54 Tier-C dim-only, 72 Tier-D refuse |
| 13. Stage 3 — Ownership (binary) | `stage3_ownership.py`, `configs/stage3.yaml` | per-cell `owned` vs `inherited` from sign of Δgap_vs_K0 | **253 owned (19 %) / 1 077 inherited (81 %) / 133 skipped** |
| 14. Stage 4 — Causal (M1 + M2) | `causal_validation.py` | M1 ablation + M2 patching at B_u and Q_geom granularity, 5 random controls each | **1 330 / 1 330 cells**; B_u mean Δacc −12.5 %, Q_geom mean Δacc −7.8 % |
| 15. Joint Stage 3 × Stage 4 | `analyze_joint.py` | descriptive merge on (model, task, mode, layer, concept) | 5 sliced tables under `data/results/joint_analysis/` |

## 3. Method

**Stage 1 (linear probe).** Per (concept, layer, model, task, mode) cell on the correct-only subset: CCSVD between-class scatter SVD with permutation null + LDA refinement (`S_B w = λ S_T w`). Three residualisation modes (off / answer / norm) give matched-population sweeps.

**Stage 2 (Bayesian geometric inference).**
- Stage 2a: discover-then-fit Fourier helix per cell with Whittle null.
- Stage 2b: spread-aware Mahalanobis distance `d_SW(u,v)² = (μ_u−μ_v)ᵀ[(Σ_u+Σ_v)/2 + λI]⁻¹(μ_u−μ_v)` vs the centroid Euclidean baseline.
- Stage 2c (BSMI-R): full point-cloud closed-form Bayesian evidence on 7 shape priors {Generic, Line, Circle, Open helix, Torus, Concentric, Ribbon} with multimodal Laplace integration over period, 10 000-permutation null with alignment-augmented statistic, intrinsic dimension (TwoNN + Levina-Bickel + PCA-PR), persistent homology (ripser β₀,β₁,β₂), differential geometry (κ, τ), label alignment (circular / Spearman), within-label + leave-value-out holdout, seed/prior stability. **No module hard-gates** — all evidence flows into the final Tier A/B/C/D decision.

**Stage 3 (ownership, binary).** Build `Q = orthonormal span([B_u(c') for c' in C(c)])` where `C(c)` is the pre-registered correlate set (e.g. `ans_units → {a, b, a_units, b_units, column_sum_units, carry_units}`). Form `Y_orth = (X − μ)(I − QQᵀ) B_u`. Re-run **full** BSMI-R on Y_orth with **α and P̂ locked** to the raw cell's values. Verdict = sign(`Δgap_vs_K0 = (raw_winner − raw_K0) − (orth_winner − orth_K0)`): negative → owned, positive → inherited. Empirically bimodal; no cell has |Δgap| ∈ (0, 5).

**Stage 4 (causal).** At the cell's layer L: project the last-token residual onto the orthogonal complement of either **(a)** the full union basis `B_u` (subspace granularity) or **(b)** `Q_geom = QR(B_u @ W_winnerᵀ)` — the n_basis-dim row-space of BSMI-R's shape regression (geometry granularity). Compare to 5 random-rank-matched controls. M2 patches the donor's projection onto the recipient stream and measures the donor-gold-token logit shift.

## 4. Layer-resolved trajectory

Averaged across all 3 models × 2 tasks × 3 modes (B_u Δacc; corrected Q_geom track is the joint table):

| Layer position | Inputs (a, b) | Intermediates (column_sum, carry, partial_product) | Answer pieces (ans_units, ans_tens, …) |
|---|---|---|---|
| 4 (early) | −0.9 % | −0.3 % | −0.1 % |
| 8 (early-mid) | −9.7 % | −8.8 % | −4.5 % |
| 14 (mid, GPT-J headline) | **−75.6 %** | **−49.0 %** | −22.2 % |
| 16 (mid, Llama / Pythia headline) | −54.2 % | −38.1 % | **−30.2 %** |
| 20 (late-mid) | −15.7 % | −34.6 % | −22.5 % |
| 24 (late) | −12.9 % | −31.1 % | −17.2 % |
| 28 (late) | −15.5 % | −27.1 % | −20.2 % |

This is the canonical **input → compute → answer** information-flow pattern: input identity matters at L4–14, intermediate computation matters mid-late, answer-piece directions matter most at L16 (just before the unembed).

## 5. Models & accuracies

| | GPT-J 6B | Llama 3.1 8B | Pythia 6.9B |
|---|---|---|---|
| HF repo | `EleutherAI/gpt-j-6B` | `meta-llama/Llama-3.1-8B` | `EleutherAI/pythia-6.9b` |
| Layers | 28, hidden 4096 | 32, SwiGLU + GQA | 32, parallel attn+MLP |
| Layers probed | 4, 8, 14, 20, 24 | 4, 8, 16, 24, 28 | 4, 8, 16, 24, 28 |
| Headline layer | 14 | 16 | 16 |
| Add acc | 84.2 % | 99.6 % | 77.2 % |
| Mult acc | 91.0 % | 96.8 % | 91.2 % |

## 6. Repo layout

```
emnlp2026/
├── README.md (this file) · plan.md (v6 source of truth) · config.yaml · configs/{stage2c,stage2b,stage3}.yaml
├── check_tokenization_limits.py · generate_datasets.py · eval_and_extract.py · build_embeddings.py
├── ccsvd_subspaces.py · residualize_activations.py · lda_subspaces.py
├── residual_hunting.py · principal_angles.py · jl_distance.py
├── stage2a_fourier_helix.py · stage2b_dsw_spread_aware.py
├── stage2c_gplvm.py · stage2c_shapes.py · stage2c_modules.py
├── stage3_ownership.py · causal_validation.py · analyze_joint.py
├── aggregate_*.py · check_*_toys.py
├── sbatch/ run_{step,stage}*.sbatch (one per step, per model)
├── docs/ 01_..10_ per-step doc (one Markdown per finished stage)
└── data/ → /data/user_data/anshulk/emnlp2026
    ├── models/ activations/ answers/ data/raw/
    └── results/{ccsvd_subspaces,lda_subspaces,residual_hunting,principal_angles,
                    jl_distance,stage2a_fourier_helix,stage2b_dsw,stage2c_gplvm,
                    stage3_ownership,stage4_causal,joint_analysis}
```

## 7. How to run

All scripts read `config.yaml`. SLURM scripts use absolute env Python (`/data/user_data/anshulk/miniconda3/envs/geometry/bin/python`).

```bash
# Full sweep — bottom-to-top dependency chain
sbatch sbatch/run_eval_and_extract.sbatch                       # Step 3
sbatch sbatch/run_ccsvd_subspaces.sbatch                        # Step 5
JID=$(sbatch --parsable sbatch/run_step6.sbatch); \
  sbatch --dependency=afterok:$JID sbatch/run_step6_aggregate.sbatch
# … similarly for steps 7-9, stages 2a/2b/2c (per-model arrays)
# Stage 3 (4 stripes per model, GPU-batched perm path; ~25 min/model on 4 A6000s)
sbatch sbatch/run_stage3_{gptj,pythia}.sbatch                   # general
sbatch sbatch/run_stage3_llama.sbatch                           # preempt, any GPU
sbatch sbatch/run_stage3_aggregate.sbatch                       # afterany on 3 above
# Stage 4 (full BSMI-R + ablation + patching at B_u and Q_geom; ~30 min on 12 GPUs)
sbatch sbatch/run_stage4_{gptj,pythia,llama}.sbatch
sbatch sbatch/run_stage4_aggregate.sbatch
# Joint analysis (CPU, < 1 s)
python analyze_joint.py --config config.yaml
```

## 8. Reproducibility

Every per-cell job writes `metadata.json` with `computation_status: complete`, full input SHA-256, library versions, config SHA, and total wall time. Atomic writes (`tempfile` + `os.replace`). Resume-by-metadata on all per-cell jobs. **No subsampling.** Every fit uses the full per-cell correct population; cross-validation, leave-value-out, and 10 000-permutation nulls are *resampling*, never *subsampling*. Spearman ρ and Pearson r reported side-by-side for every correlation. Conda env `geometry` at `/data/user_data/anshulk/miniconda3/envs/geometry` (Python 3.11.15, PyTorch 2.10.0 + CUDA 12.8, GPyTorch 1.15.2, Transformers 5.3.0, NumPy 2.2.6, scikit-learn 1.8.0).
