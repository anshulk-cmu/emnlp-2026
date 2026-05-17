# Complete End-to-End EMNLP Plan
## From Linear Probes to Bayesian Manifolds: Geometry of Arithmetic in Language Models

**Version:** v6 (data scope locked).
**Status:** Living document. Update as decisions lock and experiments land.
**Primary author:** Anshul Kumar (CMU).
**Advisor:** Deeksha Varshney (IIT Jodhpur).
**Advisor:** Manoj Kumar (IIT Roorkee).
**Main advisor:** Barnabás Póczos (CMU).
**Target venue:** ACL Rolling Review → EMNLP 2026 Main, long paper. Workshop fallback: BlackBoxNLP.
**Length budget:** 8 pages content + unlimited references + appendices + mandatory limitations.
**Primary model:** GPT-J 6B (following Kantamneni & Tegmark 2024).
**Secondary model:** Llama 3.1 8B base (replication, runs after GPT-J results land).
**Tasks:** Addition (a + b, a, b ∈ [0, 99]) and multiplication (a × b, a, b ∈ [0, 99]).
**Population:** Correct only. Wrong population is out of scope for this paper.

---

## What changed in v6 (delta from v5)

1. **Data scope locked to a, b ∈ [0, 99] for both tasks.** Addition gives 10,000 single-token-answer problems in both models. Multiplication gives 10,000 problems whose answers range up to 9801 — multi-token for 50–66% of problems depending on model.
2. **Multi-token answer handling defined.** For multiplication, "correct" is operationalised as **first-answer-token correctness** (matches KT 2024's effective protocol). Two alternatives — single-token-restricted or full-sequence-correctness — are flagged but not chosen.
3. **Wrong population dropped.** All experiments run on the **correct** subset only. Cells reduce from 36 to 18 per model. Compute approximately halves.
4. **Phase E's correct/wrong findings deprioritised.** Spearman ≫ Pearson at L5/wrong was a Phase E result; it does not appear in this paper. Phase H's inherited verdict on `correct` is unchanged (it held across all three populations, 419/419 each).
5. **Two new limitations added** (§18.9, §18.10): correct-only restriction and first-token-correctness restriction.
6. **One new risk added** (§21.5): multi-token answer rate in GPT-J multiplication may reduce sample size below comfortable thresholds.

---

## PART 0 — TITLE AND POSITIONING

### 0.1 The final title

> **From Linear Probes to Bayesian Manifolds: Geometry of Arithmetic in Language Models**

The title does four things in one line:
1. Names the methodological arc (linear probes → Bayesian manifolds).
2. Names the methodological contribution (Bayesian manifold characterisation).
3. Names the subject (geometry of arithmetic).
4. Names the model class (language models; pre-trained LMs in general, not a single model).

### 0.2 Alternative titles (rank-ordered)

If Póczos prefers a different register, here are alternatives in decreasing professionalism:

1. *"Owned or Inherited? A Bayesian Pipeline for Testing Probe Geometry in GPT-J Arithmetic"*.
2. *"Testing Representational Ownership: A Bayesian Geometric Pipeline for Arithmetic Probes in GPT-J"*.
3. *"From Linear Probes to Bayesian Manifolds: Testing Geometric Ownership in GPT-J Arithmetic"*.
4. *"Inherited Ingredients: Bayesian Tests of Linear Probe Geometry in GPT-J Arithmetic"* (plays on the project's "ingredients without a recipe" metaphor).
5. *"When Helices Don't Belong: A Bayesian Test of Linear Probes in GPT-J Arithmetic"* (references KT 2024 directly).
6. *"Whose Helix Is It? Testing Probe Ownership in Arithmetic"* (memorable but informal).

### 0.3 The one-sentence positioning

> "We propose a four-stage Bayesian pipeline that tests whether the geometric structure a linear probe finds for a concept actually belongs to that concept, and apply it to addition and multiplication (a, b ∈ [0, 99]) in GPT-J 6B on correct answers; for multiplication, the apparent helical geometry of carries vanishes once we orthogonalise against algebraically related intermediates and barely affects model behaviour under ablation."

### 0.4 The two-sentence abstract opening (for early drafts)

> "Recent work shows that pre-trained language models encode arithmetic numbers as helices in residual stream subspaces and use these helices to compute. We ask a different question: when a linear probe identifies a clean geometric structure for a concept, does that structure belong to the concept itself, or is it inherited from algebraically related concepts that share residual-stream dimensions?"

---

## PART 1 — THE CONTRIBUTION AND WHERE THE PROJECT STANDS

### 1.1 What is already in hand (the experimental capital we have)

Five solid foundations, all on Llama 3.1 8B and all verified against logs:

1. **Phase A through F/JL (linear pipeline):** 2,844 concept subspaces, 42,049 principal-angle measurements, 43.9 billion pairwise distances, union-subspace preservation Spearman ≥ 0.99 across 99 slices.
2. **Phase G (Fourier screening):** 500 helix detections in 3,480 cells, almost all carries, periods 9, 18, 27, 19, 10. Operand-digit positions return zero hits.
3. **Phase H / B.2 (orthogonalisation control):** 419 carry helix targets across 1,676 rows. **Every single one classified `inherited`** after orthogonalising against algebraic correlates. The verdict held across `all`, `correct`, and `wrong` populations — 419 inherited each — so restricting this paper to `correct` only does not weaken the headline.
4. **Curated 8,264-problem set:** difficulty-balanced across L3/L4/L5, ready for downstream Bayesian fits and causal experiments.
5. **A working SLURM-based pipeline with reproducibility gates:** byte-identity checks against Phase G centroids, registered hyperparameters, FDR-corrected p-values.

### 1.2 What this paper adds to that capital

Six new contributions, each named so it can be referred to in the paper's introduction:

**C1 — The four-stage pipeline.** A concrete sequence of tests: linear probe (Stage 1) → Bayesian manifold characterisation with uncertainty (Stage 2) → ownership test via orthogonalisation against algebraic correlates (Stage 3) → causal ablation comparing raw vs orthogonalised subspaces (Stage 4). Each stage has a binary pass criterion. The pipeline is task-agnostic and reusable.

**C2 — Bayesian manifold characterisation with centroid AND spread analysis.** Phase G's centroid-only Fourier fit is supplemented by a spread-aware Mahalanobis distance test (`d_SW`) and a Bayesian GPLVM fit with held-out reconstruction validation. Together these gate the manifold claim against the "the average traces a curve but the data don't sit on it" failure mode.

**C3 — Cross-task application on a uniform data scope.** The same pipeline runs on addition and multiplication in GPT-J 6B with `a, b ∈ [0, 99]` for both tasks. Two outcomes are pre-registered as valid headlines:
- **Finding A — addition owned, multiplication inherited.** Asymmetric verdicts, with a mechanistic interpretation about compositional structure.
- **Finding B — both inherited.** Uniform verdicts, with a stronger interpretation about LRH-style methodology overstating ownership.

Either outcome is publishable. The pipeline is the constant.

**C4 — A concrete cross-model replication on Llama 3.1 8B.** After GPT-J results land, the pipeline runs on Llama for a subset of cells. Cross-model agreement strengthens whichever finding holds; cross-model disagreement is itself a contribution.

**C5 — A reusable diagnostic.** The "ingredient signal minus owned signal" gap (Stage 1 strength minus Stage 3 ownership strength) is a single number that flags inherited geometry in any concept-probe study. We propose this as a future tool.

**C6 — Code, curated dataset, and reproducibility artefacts.** Full release of pipeline scripts, the 8,264-problem curated multiplication set, the addition data generation script, and per-cell metadata.

### 1.3 The pre-registered central claim

The paper's central claim is methodological, not empirical. It is:

> "Linear probe success at finding a clean geometric structure for a concept does not imply that the structure belongs to the concept. We propose a four-stage Bayesian pipeline that tests ownership directly, and demonstrate the pipeline on addition and multiplication in GPT-J 6B on the model's correct answers."

Whether the empirical evidence shows asymmetric verdicts (Finding A) or uniform inheritance (Finding B) is *evidence about LRH-style methodology*, not about the pipeline's correctness. The pipeline is correct if it gives consistent verdicts under perturbation, agrees with toy data, and the verdicts predict causal-ablation outcomes.

### 1.4 Why this is novel

Three reviewer-attack-resistant differentiators:

**Differentiator 1 — Against KT 2024 (the addition + helix paper).** They propose a structure (the helix) and verify it patches behaviour on correct addition prompts. They do not test whether the helix belongs to the number or to algebraically related quantities. Our pipeline tests this directly. Our addition section *replicates* KT (same prompt format, same a, b ∈ [0, 99] range, same correct-only filtering) and *extends* it with the ownership test.

**Differentiator 2 — Against Bai et al. 2025 (multiplication in toy models).** They study small transformers explicitly trained for multiplication. We study a production-scale pretrained LLM. Their Fourier features may or may not transfer; ours is the first systematic test in a pretrained LLM for multiplication.

**Differentiator 3 — Against Phase H's existing inherited verdict on Llama.** That result is currently a single-model, single-task observation. The cross-task and cross-model extension turns it into a publishable methodological story.

---

## PART 2 — WHY GPT-J PRIMARY, LLAMA SECONDARY

### 2.1 KT 2024 used GPT-J as their primary model

Following KT 2024 directly:
- **GPT-J 6B** was their main analysis model because its MLPs are simple (no gating) and individual neurons are easier to interpret.
- **Pythia-6.9B** and **Llama 3.1 8B** were appendix replications.
- KT explicitly state Llama 3.1 8B's helix is much weaker than GPT-J's (their Figure 23).

We follow the same choice. GPT-J is the cleanest setting for the addition baseline. If Stage 1 succeeds in GPT-J × addition, it should succeed; if it succeeds in Llama × addition too, that's a bonus.

### 2.2 GPT-J specifics

| Property | Value |
|---|---|
| Parameters | 6 billion |
| Layers | 28 |
| Hidden dim | 4096 |
| MLP type | Simple (no gating) |
| Attention heads | 16 per layer |
| Single-token integer cap | 361 |
| Tokeniser | GPT-2 BPE |
| License | Apache-2.0 |
| Weight host | EleutherAI on Hugging Face |

### 2.3 Llama 3.1 8B specifics (for the secondary phase)

| Property | Value |
|---|---|
| Parameters | 8 billion |
| Layers | 32 |
| Hidden dim | 4096 |
| MLP type | Gated (SwiGLU) |
| Attention heads | 32 per layer (GQA) |
| Single-token integer cap | 999 |
| Tokeniser | TikToken |
| License | Llama Community License |
| Weight host | Meta on Hugging Face |

### 2.4 Why two models even though GPT-J is primary

A single-model paper invites the reviewer comment "does this happen in any other model?". Adding Llama as a replication step blocks that comment. The cross-model section is short (~0.5 pages) and runs after GPT-J results lock. If Llama agrees, we have cross-architecture robustness. If Llama disagrees, we have a cross-model finding (also publishable).

### 2.5 The pre-registered Llama gate

We commit to running Llama only on the cells where GPT-J gives a clean Stage 1 + Stage 3 + Stage 4 verdict. Llama is replication, not exploration. If GPT-J × addition is `owned`, we test Llama × addition for `owned-confirmed` or `owned-but-weaker`. If GPT-J × multiplication is `inherited` (essentially predicted by Phase H), we test Llama × multiplication for cross-model agreement.

---

## PART 3 — THE FOUR-STAGE PIPELINE (DETAILED)

This section has full technical detail. Each stage has: what we run, the math, the pass criterion, the failure modes, the toy validation it must pass.

### 3.1 Stage 1 — Linear probe

#### 3.1.1 Purpose

Find candidate cells where a linear probe identifies a clean concept subspace. This is the LRH baseline. Stage 1 is necessary but not sufficient for any ownership claim.

#### 3.1.2 Method (Phase C + Phase D combined)

For each cell `(concept, layer, model, task)` (note: no longer `× population`; correct-only):

**Sub-step a — Conditional covariance + SVD.** Compute the per-value mean activations `μ_v` for each concept value `v ∈ V`. Form the between-class covariance `S_B = Σ_v n_v (μ_v − μ̄)(μ_v − μ̄)ᵀ / N`. Take the top eigenvectors via SVD; report the dimension where eigenvalues exceed the permutation null at p < 0.01. This gives the Phase C subspace.

**Sub-step b — LDA refinement.** Compute the within-class covariance `S_W = Σ_v Σ_{i ∈ v} (x_i − μ_v)(x_i − μ_v)ᵀ / N`. Solve the generalised eigenvalue problem `S_B w = λ S_W w` by computing `S_W^{-1/2} S_B S_W^{-1/2}` and taking its eigendecomposition. Report the eigenvalues `λ_k` (these are the LDA eigenvalues, bounded between 0 and 1) and per-direction Cohen's `d`.

**Sub-step c — Cross-validation.** Five-fold CV: hold out one fold, fit the LDA on the other four, project the held-out fold, measure the Spearman correlation between predicted and true class labels. Report `cv_correlation`.

#### 3.1.3 Pass criterion

A cell passes Stage 1 if all three are met:
- λ₁ ≥ 0.5 (substantial Fisher discrimination).
- Bootstrap CI lower bound on λ₁ > 0 (signal not from noise).
- `cv_correlation` ≥ 0.7 (the directions generalise).

Bootstrap: 200 resamples on the input cell, recompute λ₁, take the 5th percentile.

#### 3.1.4 Failure modes

- **N/d inflation.** When N/d is small, all eigenvalues inflate toward 1 by sample chance. Cells with N/d < 5 are flagged and excluded from headline claims. **Risk amplified by correct-only restriction:** for multiplication in GPT-J, correct rate may be low (~25%), giving N ≈ 2,500 of 10,000 problems. In a 4096-dim residual stream, this is N/d < 1 if we treat the full residual stream as the embedding space, but we work in the much smaller Phase C subspace (d ≈ 9–18), giving N/d ≈ 150 — comfortably above floor.
- **Single-direction dominance.** If only λ₁ is large and λ₂ collapses, the "subspace" is really a single direction and may behave anisotropically. Cells with `λ₁ / λ₂ > 10` are flagged for inspection.
- **Group-imbalance.** Per-value sample counts `n_v` should be within a factor of 3 of each other; otherwise the LDA solution is biased. Cells with `max(n_v) / min(n_v) > 3` get a `group_imbalance` flag. **Risk amplified by correct-only restriction:** for multiplication in GPT-J, easy problems (a × b small) are overrepresented in the correct set; small product values get more samples than large ones. Stratification to balance per-value counts is in §7.4.

#### 3.1.5 Toy validation

Stage 1 must pass on:
- **Toy 1L — Linear separation.** 9D Gaussian data with 10 classes whose means lie on a 1D line. Expected: λ₁ ≥ 0.95, λ₂ ≈ 0.
- **Toy 2L — Two-axis structure.** 10 classes on a 2D subspace. Expected: λ₁ and λ₂ both ≥ 0.5, λ₃ ≈ 0.
- **Toy 3L — No structure.** 9D isotropic Gaussian with random class assignment. Expected: all λ_k below the permutation null at p < 0.01.

If Stage 1 fails on any toy, debug before any real-data fit.

### 3.2 Stage 2 — Bayesian manifold characterisation

This is the new big stage. Four sub-components.

#### 3.2.1 Stage 2a — Centroid Fourier helix fit

For each cell, compute per-value centroids `μ_v` and project them onto the Stage 1 subspace. Run the Fourier-power test for periodic structure across the centroid sequence. Report:
- Best period `P` (from the Phase G prior set: 2, 3, 5, 7, 10, 18, 27, 19; for addition the relevant periods are likely 10 and 100).
- Helix FCR (the fraction of total power explained by the (cos(2π v/P), sin(2π v/P)) pair on the top two coordinates).
- Two-axis significance via FDR.

Pass criterion: helix FCR ≥ 0.30 with FDR-corrected q < 0.05.

#### 3.2.2 Stage 2b — Spread-aware test (`d_SW`)

For each cell, compute the spread-aware Mahalanobis distance between centroids:

```
d_SW(u, v)² = (μ_u − μ_v)ᵀ [(Σ_u + Σ_v)/2 + λ I]⁻¹ (μ_u − μ_v)
```

with `λ = 10⁻⁶ tr((Σ_u + Σ_v)/2) / d` for numerical stability. Each `Σ_v` is the within-class covariance for value `v` *on the correct subset*.

Form two distance matrices:
- `D_Euclidean`: the Euclidean centroid distance matrix.
- `D_SW`: the spread-aware distance matrix.

Compute the Spearman correlation between the two:

```
ρ_centroid = Spearman(vec(D_Euclidean), vec(D_SW))
```

Pass criterion: `ρ_centroid ≥ 0.85` with bootstrap 95 % CI lower bound ≥ 0.70.

If Stage 2b fails, the cell's verdict is downgraded to `centroid_only_shape`. The interpretation: the centroids trace a curve, but the per-value spread is so large that the data themselves are not on the curve.

#### 3.2.3 Stage 2c — GPLVM (primary Bayesian method)

Fit Bayesian GPLVM (Titsias-Lawrence 2010, with stochastic variational inference) on the full point cloud of each cell *on correct examples only*. Inference details:
- Latent dimension cap: d̂ = 5, with ARD pruning.
- Inducing points: M = 200, placed by k-means on the projected data.
- Optimiser: Adam at 1e-3 for 1000 ELBO steps.
- Three random seeds.

Compare three kernels:
- **K1 — RBF.** No periodicity assumption.
- **K2 — Periodic.** Period prior fixed at the value reported by Stage 2a.
- **K3 — Periodic + Linear.** Encodes the helix kernel.

For each kernel, compute the optimised ELBO. Report:
- Best kernel by ELBO with BIC-style penalty: `adjusted_ELBO = ELBO − ½ |θ| log N`.
- Held-out reconstruction MSE on a 20% holdout.
- ARD-pruned latent dimension `d̂_ARD`.

Pass criterion for "Bayesian manifold confirmed at this cell":
- Best kernel adjusted-ELBO gap to runner-up ≥ 5 nats (Kass-Raftery decisive).
- Held-out MSE for best kernel ≤ runner-up's MSE − 1 bootstrap SE.
- Three-seed agreement on best kernel within 1 nat.

If decisive AND held-out MSE check passes, label the cell `strong_score_evidence`. If 2.3–5 nats AND held-out MSE passes, label `moderate_score_evidence`. Otherwise `kernel_inconclusive`.

#### 3.2.4 Stage 2d — RBF-precision VAE (consistency check)

Architecture per Hauberg 2018 / Arvanitidis et al. 2018:
- **Encoder**: 2-layer MLP, 64 hidden units, latent dim 5 (matching GPLVM).
- **Decoder mean**: 2-layer MLP, 64 hidden units.
- **Decoder precision (the RBF part)**:
  ```
  σ(z)⁻¹ = β₀ + Σᵢ wᵢ exp(−|z − cᵢ|² / 2λᵢ²)
  ```
  with `wᵢ > 0` (softplus), `λᵢ` learnable, `cᵢ` initialised at the encoder mean for training example `i`.

Training: ELBO with KL warmup over the first 50 epochs. 500 epochs total. Adam at 1e-3.

What we extract: same as GPLVM (latent coordinates, pullback metric `J(z)ᵀ J(z)`, geodesic distances, ARD-pruned `d̂`).

**Critical framing.** GPLVM and RBF VAE share the growing-uncertainty mechanism. Their agreement is *consistency*, not *independent corroboration*. We report agreement rate across cells (kernel choice agreement, `d̂` agreement, posterior-mean L² distance between manifolds) but never claim "two independent methods agree." Genuine independent evidence is Stage 4 causal.

Pass criterion: RBF VAE picks the same kernel-equivalent and the same `d̂_ARD` as GPLVM in ≥ 80% of cells, AND the per-cell posterior-mean manifolds differ by Hausdorff distance ≤ 0.5 (in normalised units).

#### 3.2.5 Toy validation for Stage 2

Four toys (the project's pre-registered set):
- **Toy 1B — Line.** 200 points on a 1D line in 9D + Gaussian noise (σ = 0.05).
- **Toy 2B — Circle.** 200 points uniformly on a 2D circle.
- **Toy 3B — Helix.** 200 points on a circular helix in 9D.
- **Toy 4B — Isotropic Gaussian.** 200 points from N(0, I_9).

Both methods must pass all four toys before any real-data fit.

### 3.3 Stage 3 — Ownership test (Phase H, extended)

#### 3.3.1 Purpose

Test whether the Stage 2 manifold is owned by the probed concept or inherited from algebraically related concepts that share residual-stream geometry.

#### 3.3.2 The orthogonalisation operation

Identify the algebraic correlate set for the concept. For multiplication carries in our 0–99 range:
- `carry_units → {column_sum_units, partial_product_a_units_b_units}` (the carry from the units column).

For addition in our 0–99 range:
- `units(a+b) → {a, b, units(a), units(b)}`.
- `tens(a+b) → {a, b, tens(a), tens(b), carry_units}`.
- `a + b → {a, b}` (almost trivial — only the inputs themselves).

The correlate set is registered in advance per concept. We do not change it after seeing results.

For each cell, build the orthonormal basis `Q` for the union of the correlate concepts' Phase C subspaces (computed via QR). Then form the orthogonalised activations:

```
Y_orth = (X − μ)(I − Q Qᵀ) Bᵀ
```

where `B` is the cell's Phase C basis and `μ` is the layer's training-mean activation (computed on correct examples).

#### 3.3.3 Re-run Stages 2a–d on the orthogonalised activations

For each cell, compute on `Y_orth`:
- Helix FCR, period, two-axis significance.
- d_SW Spearman.
- GPLVM kernel comparison and ARD.
- RBF VAE kernel-equivalent comparison and ARD.

#### 3.3.4 Verdict rule (pre-registered)

For each cell, the Stage 3 verdict is one of three labels:

**Owned.** All four sub-stages survive orthogonalisation:
- Helix FCR drop < 0.30 AND orthogonalised q < 0.05.
- d_SW Spearman remains ≥ 0.85.
- GPLVM still picks the same periodic kernel with adjusted-ELBO ≥ 5 nats over runner-up.
- RBF VAE agrees with GPLVM (consistency).

**Inherited.** Two or more sub-stages collapse:
- Helix FCR drop ≥ 0.50 OR orthogonalised q ≥ 0.10.
- GPLVM no longer picks a periodic kernel decisively.
- The structure that linear probe and Stage 2a found is gone after orthogonalisation.

**Ambiguous.** Mixed pattern. Reported in the appendix; not used for headline claims.

#### 3.3.5 The crucial limitation

Orthogonalisation removes linear nuisance only. If the target concept and a correlate share neurons via nonlinear entanglement, removing the correlate's directions removes some target signal too. We acknowledge this in limitations.

#### 3.3.6 Toy validation for Stage 3

Two toys, hand-constructed:
- **Toy 1O — Owned helix.** A helix with a separate concept whose values are independent of the helix structure. Expected verdict: `owned`.
- **Toy 2O — Inherited helix.** A helix with a separate concept whose values are a deterministic function of the helix angle. Expected verdict: `inherited`.

Both toys must pass before any real-data Stage 3 run.

### 3.4 Stage 4 — Causal ablation

#### 3.4.1 Purpose

The genuinely independent validation. Stages 2 and 3 are observational; Stage 4 asks whether the model's behaviour responds to ablating the manifold structure.

#### 3.4.2 The two ablation conditions

For each cell that passed Stage 1 and has a Stage 3 verdict, run two conditions:

**Condition A — Raw probe ablation.** Project the residual stream at layer L onto the orthogonal complement of the Stage 1 subspace `B`:
```
h_ablated = h − (h · Bᵀ) B
```
Continue the forward pass from layer L+1. Measure the change in **first-answer-token logit**:
```
Δlogit = logit(correct_first_token | original) − logit(correct_first_token | ablated)
```

**Note on multi-token answers.** For addition (max answer 198, single-token in both models), the first answer token is the entire answer. For multiplication (max answer 9801, multi-token in both models for some problems), we measure logit difference at the first answer token only. This matches KT 2024's effective protocol and is consistent with our "first-token correctness" definition for multiplication (§4.3).

**Condition B — Orthogonalised-subspace ablation.** Project onto the orthogonal complement of the *owned component* (the part of `B` not explained by `Q`). For inherited cells, this should be a much smaller subspace. Measure the same `Δlogit`.

**Control C — Random-subspace ablation.** For each ablation condition, run 100 trials with a random subspace of the same rank. Compute the null distribution of `Δlogit`.

#### 3.4.3 Pass criteria

For "owned and causally used":
- Raw ablation `Δlogit` > 95th percentile of random-control distribution.
- Orthogonalised ablation `Δlogit` > 95th percentile of random-control distribution.
- Effect size `Δlogit_orth / Δlogit_raw` > 0.5 (the owned component carries most of the causal effect).

For "inherited and not causally used":
- Raw ablation `Δlogit` > 95th percentile (the structure DOES affect output — but recall the structure is mostly the correlates).
- Orthogonalised ablation `Δlogit` ≤ 95th percentile of random control (the *owned-only* part doesn't matter).

For "owned in geometry but not in causation":
- Raw and orthogonalised ablations both show small effects below random control. The structure is owned geometrically but the model's downstream computation doesn't actually use it. We report this honestly if it occurs.

#### 3.4.4 Compute budget for Stage 4

Per cell:
- Number of test problems: up to 1,000 correct examples from the held-out set.
- Number of layers ablated: 1 (the cell's layer).
- Number of conditions: 2 (raw, orthogonalised) + 100 random controls.
- Total forward passes: 1,000 × 102 ≈ 102,000.
- Time per forward pass: ~50 ms on A100 with KV-cache.
- Total per cell: ~85 minutes.

For 6 headline cells (3 concepts × 2 tasks, no population dimension) × 1 layer × 1 model: ~9 GPU-hours.
For both models: ~18 GPU-hours.

#### 3.4.5 Toy validation for Stage 4

Sanity-check protocol on the GPT-J × addition cell first: ablate a known-irrelevant direction (a random position-encoding direction at an early layer) and verify `Δlogit` is below random control.

---

## PART 4 — TWO TASKS: ADDITION AND MULTIPLICATION

### 4.1 Why these two tasks

**Addition.** KT 2024's primary task. Allows direct comparison and methodology replication. Single-step operation: `a + b → a+b`. Few intermediate variables. If our pipeline's prediction holds (compositional structure → inheritance), addition is the lower-inheritance side of the contrast.

**Multiplication.** The project's existing focus. Multi-step operation: `a × b → partial_products → column_sums → carries → ans_digits`. Many intermediate variables. Phase H already established multiplication is inherited in Llama; we extend to GPT-J and add Stages 2 and 4.

### 4.2 Addition setup (locked)

**Prompt format (following KT 2024 exactly):**
```
"Output ONLY a number. {a}+{b}="
```

**Range:** `a, b ∈ [0, 99]`. **10,000 problems total.** Stratified by carry status during data generation; carry-balanced before any analysis.

**Single-token verification:** Answer range is [0, 198]. Both GPT-J (cap 361) and Llama 3.1 8B (cap 999) tokenise this as a single token. **No multi-token confound for addition.**

**Correctness definition:** Standard exact-match. The model's first generated token must equal the gold answer's single token. Since the answer is single-token, this is full-answer correctness.

**Expected correct rate:** GPT-J ≈ 80–85% (matching KT's 80.5%); Llama ≈ 95%+. So:
- GPT-J × addition correct subset ≈ 8,000–8,500 problems.
- Llama × addition correct subset ≈ 9,500+ problems.

**Concept vocabulary (six concepts; on the correct subset):**
1. `a` — first operand value (100 unique values: 0–99).
2. `b` — second operand value (100 unique values: 0–99).
3. `a + b` — the answer (199 unique values: 0–198).
4. `units(a + b)` — units digit of the answer (10 unique values).
5. `tens(a + b)` — tens digit of the answer (20 unique values, since answer can be 100–198).
6. `carry_units` — boolean: is `units(a) + units(b) ≥ 10`?

**Algebraic correlate sets for Stage 3:**
- `units(a+b) → {a, b, units(a), units(b)}`
- `tens(a+b) → {a, b, tens(a), tens(b), carry_units}`
- `a + b → {a, b}` (almost trivial)

Note the asymmetry: addition concepts have *very few* algebraic correlates. This is the structural reason we *might* see ownership for addition.

### 4.3 Multiplication setup (locked)

**Prompt format:**
```
"Output ONLY a number. {a}*{b}="
```

We use KT's prompt template with `*` substituted for `+`, to maximise comparability with the addition replication.

**Range:** `a, b ∈ [0, 99]`. **10,000 problems total.** Same range as addition for clean cross-task comparison.

**Multi-token answer handling.** Answer range is [0, 9801]. Tokenisation:
- **GPT-J (cap 361):** ~3,400 problems give single-token answers (a × b ≤ 361). The remaining ~6,600 give multi-token answers.
- **Llama (cap 999):** ~6,700 problems give single-token answers (a × b ≤ 999). The remaining ~3,300 give multi-token answers.

We do NOT restrict to single-token answers. Instead, we operationalise correctness as:

**First-answer-token correctness.** A problem is `correct` if the model's first generated token (immediately after `=`) matches the first token of the gold answer's BPE tokenisation. This is a strictly weaker criterion than full-answer exact match but it has three advantages:
- Maximises sample size on the correct subset.
- Matches KT 2024's effective protocol (they only used single-token answers, which is equivalent to first-token-correctness when the answer happens to be one token).
- Aligns with Stage 4's first-answer-token Δlogit measurement.

**Two alternatives flagged but not chosen:**
- **(a) Single-token-restricted.** Restrict to a × b ≤ 361 (GPT-J) or a × b ≤ 999 (Llama). Loses the "0–99" framing but preserves full-answer correctness.
- **(c) Full-sequence-correctness.** Define `correct` as exact match across all answer tokens. Most rigorous but yields the smallest correct set.

Both alternatives are described in §4.5. They remain available if (b) — first-token-correctness — has an unforeseen issue that surfaces during the smoke test.

**Expected correct rate (under first-token correctness):**
- GPT-J × multiplication: probably 25–40%, so ~2,500–4,000 correct examples.
- Llama × multiplication: probably 50–70%, so ~5,000–7,000 correct examples.

These are large enough for Stage 1 LDA in a Phase C subspace (d ≈ 9–18, giving N/d well above 100). They may be tight for GPLVM if a concept-value bin has fewer than 30 examples; per-value stratification (§7.4) is essential.

**Concept vocabulary (three headline concepts on the correct subset):**
1. `a_units` — units digit of `a` (10 unique values).
2. `carry_units` — the units-column carry: `floor((units(a) × units(b)) / 10)` (10 unique values: 0–8 in practice).
3. `ans_units` — units digit of the (full) answer.

Supplementary concepts in appendix: `a_tens`, `b_units`, `partial_product_units`.

**Algebraic correlate sets for Stage 3:**
- `a_units → {}` (no algebraic correlates).
- `carry_units → {column_sum_units, partial_product_units}` (where `column_sum_units = units(a) + units(b)` and `partial_product_units = units(a) × units(b)`).
- `ans_units → {column_sum_units, carry_units, partial_product_units}`.

### 4.4 The structural prediction visualised

The mechanistic prediction is about the *strength* of correlate sharing, not just the count. In addition, `units(a+b) = (units(a) + units(b)) mod 10` is a deterministic function of two variables; the shared variance with operands is bounded. In multiplication, `carry_units` is determined by `units(a) × units(b)`, which is also determined by `partial_product_units` and contributes to `column_sum_units`. Multiple intermediates encode the same numerical quantity through different algebraic functions, and they share residual-stream dimensions. The paper has to make this asymmetry precise, probably as a small schematic figure.

### 4.5 Multi-token correctness alternatives (for reference)

If first-token correctness has an issue (e.g., GPT-J's tokeniser splits multi-token answers in unexpected ways), we have two alternatives pre-registered:

**Alternative (a) — Single-token restriction.**
- Multiplication restricted to a × b ≤ 361 in GPT-J (~3,400 problems).
- Multiplication restricted to a × b ≤ 999 in Llama (~6,700 problems).
- Cross-model comparability requires using the GPT-J subset for both, giving ~3,400 problems each.
- Loses the "0–99 in both tasks" symmetry but gives clean exact-match correctness.

**Alternative (c) — Full-sequence correctness.**
- A problem is correct iff the model emits the exact answer token sequence.
- Smallest correct subset; probably 15–25% in GPT-J for multiplication.
- Most rigorous but may push N/d below the comfortable threshold for some concepts.

The decision between (b) primary, (a) backup, (c) backup is locked. If smoke testing on 100 multiplication problems reveals (b) gives unexpectedly low correct rates (< 15% in GPT-J), we fall back to (a).

---

## PART 5 — CONCEPT VOCABULARY (THE CONTROLLED TESTBED)

### 5.1 Why a small concept set

Three concepts per task is the right number for an EMNLP paper:
- One operand (clean baseline; no correlates, should be owned).
- One intermediate (the headline test; predicted inherited for multiplication).
- One output (the harder concept; mixed predictions).

Three concepts × two tasks × three layers × **one population (correct only)** × two models = 36 total cells (down from 72 in v5). Each cell can be reported individually.

### 5.2 The full concept selection

| Task | Concept | Type | Algebraic correlate set | Predicted Stage 3 verdict |
|---|---|---|---|---|
| Addition | `a` | Operand | {} | owned |
| Addition | `units(a+b)` | Output | {a, b, units(a), units(b)} | owned (but testable) |
| Addition | `carry_units` | Intermediate | {a, b, units(a), units(b)} | owned (single-step, weak inheritance) |
| Multiplication | `a_units` | Operand | {} | owned |
| Multiplication | `carry_units` | Intermediate | {column_sum_units, partial_product_units} | inherited (Phase H confirmed in Llama) |
| Multiplication | `ans_units` | Output | {column_sum_units, carry_units, partial_product_units} | inherited |

### 5.3 The cells we run Stage 4 on (causal)

To save compute, Stage 4 runs only on the six headline cells (one per task × concept combination), at the middle layer per model:
- Addition × `a` × GPT-J × layer 14.
- Addition × `units(a+b)` × GPT-J × layer 14.
- Addition × `carry_units` × GPT-J × layer 14.
- Multiplication × `a_units` × GPT-J × layer 14.
- Multiplication × `carry_units` × GPT-J × layer 14.
- Multiplication × `ans_units` × GPT-J × layer 14.

For Llama, repeat on the same six concepts at Llama layer 16.

Total Stage 4 cells: 12 (6 GPT-J + 6 Llama).

### 5.4 Concept-vocabulary failure modes

**Failure mode A — concept doesn't exist linearly in GPT-J at any layer.** If Stage 1 fails for a concept across all three tested layers, we drop the concept and pick a substitute. Pre-registered substitutes:
- `a` → `b` (symmetric).
- `units(a+b)` → `tens(a+b)`.
- `carry_units` (addition) → `magnitude(a+b)` (whether the answer is < 10, 10–99, or 100+).
- `a_units` → `b_units`.
- `carry_units` (multiplication) → `column_sum_units`.
- `ans_units` → `partial_product_units`.

**Failure mode B — concept correlates with another at probe level.** If two concepts in our vocabulary have inter-concept correlation > 0.7 (Pearson) on activations, we drop one and report only the cleaner one.

**Failure mode C — per-value sample size below 30 on correct subset.** For multiplication in GPT-J, the correct subset will be small (~2,500–4,000). Some `ans_units` values may have < 30 examples. We pre-register: any value with < 30 examples is dropped from the LDA fit and reported separately.

---

## PART 6 — MODEL SETUP

### 6.1 GPT-J 6B specifics

**Source:** EleutherAI/gpt-j-6B on Hugging Face.
**Precision:** bfloat16 for inference, float32 for activation extraction.
**Layers extracted:** 4, 8, 14, 20, 24 (five layers; can downsample to three for headline).
**Position extracted:** the `=` token (last input-side token, before answer generation begins).
**Activation file format:** float32 numpy arrays, shape (n_correct_problems, 4096), one file per (task, layer) combination.

### 6.2 Llama 3.1 8B specifics

**Source:** meta-llama/Meta-Llama-3.1-8B (base, not instruct) on Hugging Face.
**Precision:** bfloat16 for inference, float32 for activation extraction.
**Layers extracted:** 4, 8, 16, 24, 28.
**Position extracted:** `=` token.
**Activation file format:** same as GPT-J.

### 6.3 Why the chosen layers

For an 8-page paper we report three layers per model:
- **Early (layer 4 in both):** the residual stream is still close to embeddings.
- **Middle (layer 14 in GPT-J, layer 16 in Llama):** the heaviest computation.
- **Late (layer 24 in GPT-J, layer 28 in Llama):** output preparation.

Five layers in the actual pipeline run; three layers in the headline figures.

### 6.4 Activation extraction details

Code uses Hugging Face hooks at each transformer block's output (post-LayerNorm if present). One forward pass per problem. Activations cached per problem before any analysis. We extract activations on **all 10,000 problems per task per model**, then filter to the correct subset post-hoc using the model's first-token output.

This separation matters: extracting on all 10,000 lets us verify correctness criteria after the fact and rerun under alternative criteria (a) or (c) without re-extraction.

### 6.5 Reproducibility hash

Each activation file ships with model checkpoint hash, tokeniser version, random seed, PyTorch version, CUDA version. Necessary for the EMNLP reproducibility checklist.

---

## PART 7 — DATA GENERATION

### 7.1 Addition data

Script: `generate_addition.py`.
- 10,000 problems, `a, b ∈ [0, 99]`.
- All 10,000 (a, b) pairs (the full Cartesian product).
- For each problem, compute the six addition concept labels.
- Write to `addition_problems.csv` and the per-(layer) activation files.

Validation:
- Every prompt verified to produce a single-token answer in both GPT-J and Llama.
- Every concept value present at least 30 times across the 10,000 problems.
- No NaN/Inf in activation files.

### 7.2 Multiplication data

Script: `generate_multiplication.py` (new; supersedes the existing project's L1–L5 generation).
- 10,000 problems, `a, b ∈ [0, 99]`.
- All 10,000 (a, b) pairs (the full Cartesian product).
- For each problem, compute the three headline concept labels and the algebraic correlate labels.
- Write to `multiplication_problems.csv` and per-(layer) activation files.

Validation:
- Every prompt is well-formed.
- For each model, identify the correct subset post-hoc by running inference and checking first-token output.
- Verify per-value sample counts on the correct subset; flag any concept value with < 30 correct examples.

### 7.3 Forward-pass activation extraction

For both models × both tasks:
- Run the model in inference mode on each of the 10,000 problems.
- Hook at each chosen layer.
- Extract the residual stream at the `=` token position.
- Save as float32 numpy array.
- **Separately**, run inference all the way through to first-token logits and record the model's predicted first answer token.

Total compute: ~10 GPU-hours for both tasks × both models (10,000 forward passes per task per model × 5 layers extracted × ~2 minutes per layer for 10,000 examples on A100).

### 7.4 Per-value stratification on the correct subset

After identifying the correct subset, check per-value counts for each concept. If any value has fewer than 30 examples:
- For Stage 1 (LDA), drop that value from the fit.
- For Stage 2 (Bayesian methods), drop that value from the centroid computation; the GPLVM/RBF VAE fits on individual points include all values that have ≥ 30 examples.
- Report the dropped values explicitly per cell.

**Expected dropping rate:** rare for addition (correct rate is high, distribution is balanced); more common for multiplication (correct rate is lower, easy problems overrepresented). Pre-registered: if more than 30% of concept values are dropped for any cell, the cell is excluded from headline claims.

### 7.5 Curated set (legacy from existing project)

For the existing Llama × multiplication work, the 8,264-problem curated set was difficulty-balanced. For the new 0–99 setup, we do **not** use this curated set — we use the full 10,000 with correct-only filtering. The curated set remains useful for the appendix replication of Phase H's earlier finding but is not the headline data source.

---

## PART 8 — THE 2x3 VERDICT MATRIX

### 8.1 The core matrix

The headline result is a 2×3 matrix indexed by task and concept type:

| Concept type | Addition | Multiplication |
|---|---|---|
| Operand (a / a_units) | predicted owned | predicted owned |
| Intermediate (carry_units) | TEST | predicted inherited (Phase H confirmed in Llama) |
| Output (units(a+b) / ans_units) | TEST | TEST |

The two cells already mostly settled:
- Operand × addition × GPT-J: trivially owned (no algebraic correlates).
- Intermediate × multiplication × Llama: 419/419 inherited (Phase H, on `correct` and other populations).

The cells we test new:
- Intermediate × addition × GPT-J: the key new comparison cell.
- Intermediate × multiplication × GPT-J: cross-model replication of Phase H.
- Output × addition × GPT-J: secondary new comparison.
- Output × multiplication × GPT-J: secondary new comparison.
- Operand × multiplication × GPT-J: trivially predicted owned but worth verifying.

### 8.2 Reading the matrix

The matrix is what the paper's headline figure shows. A reviewer reading just the figure should be able to extract:
- Whether linear probes succeed (Stage 1 column).
- Whether Bayesian manifolds confirm (Stage 2).
- Whether ownership survives orthogonalisation (Stage 3).
- Whether ablation confirms causal use (Stage 4).
- The contrast pattern across task and concept type.

### 8.3 Pre-registered cell verdicts before any addition experiments run

We pre-register expected verdicts for each cell. Pre-registration means we commit to these in writing before seeing results, and we report deviations honestly.

| Cell | Pre-registered prediction |
|---|---|
| `a` × addition × GPT-J × correct | owned (trivial) |
| `units(a+b)` × addition × GPT-J × correct | owned with moderate confidence |
| `carry_units` × addition × GPT-J × correct | owned with low confidence (could go either way) |
| `a_units` × multiplication × GPT-J × correct | owned (trivial) |
| `carry_units` × multiplication × GPT-J × correct | inherited (replicating Phase H) |
| `ans_units` × multiplication × GPT-J × correct | inherited (algebraically similar to carries) |

Two of the six cells are predictions that genuinely could go either way (`carry_units` and `units(a+b)` in addition). These are the experimental headlines.

---

## PART 9 — THREE POSSIBLE RESULT SCENARIOS

The paper's framing is robust to all three.

### 9.1 Scenario A — Asymmetric verdicts (addition owned, multiplication inherited)

**Headline:** "Geometric ownership is task-dependent. Single-step arithmetic (addition) produces owned representations; compositional arithmetic (multiplication) produces inherited geometry."

**Mechanistic interpretation:** Tasks with few intermediate variables that share residual-stream geometry produce ownership-stable concept representations. Tasks with many intermediate variables produce inheritance because the multiple intermediates all encode the same numerical quantity through different algebraic functions, and they share residual-stream dimensions.

**Paper structure if Scenario A obtains:**
- Section 5 (Results) leads with the asymmetry.
- Discussion (§6) develops the mechanistic story.
- Limitations: "we observe this in two tasks; generalisation to more tasks is future work."
- Target venue tier: EMNLP Main.

### 9.2 Scenario B — Uniform inheritance (both tasks inherited)

**Headline:** "Linear probe success does NOT imply representational ownership, even for the simplest arithmetic operations. Apparent geometric structure in pre-trained LLMs may consistently reflect shared encoding of algebraic intermediates rather than concept-specific geometry."

**Mechanistic interpretation:** The representational sharing observed in multiplication carries also exists in addition, because even single-step operations have intermediate variables (the operands themselves, and their digit-level decompositions) that share residual-stream geometry with the output concept. Linear probes systematically attribute geometric structure to whichever concept is being probed.

**Paper structure if Scenario B obtains:**
- Section 5 leads with the uniformity finding.
- Discussion: this is a stronger claim against LRH-style methodology than Scenario A. The prior literature's reading of "linear probe → ownership" is broadly wrong for arithmetic.
- Target venue tier: EMNLP Main.

### 9.3 Scenario C — Reverse asymmetry (addition inherited, multiplication owned)

Unlikely given Phase H's existing Llama result, but possible if GPT-J's multiplication geometry differs from Llama's.

**Headline:** "Geometric ownership is model-dependent rather than task-dependent. Multiplication carries are owned in GPT-J but inherited in Llama 3.1 8B."

**Paper structure if Scenario C obtains:**
- Cross-model contrast becomes the headline rather than cross-task.
- Discussion: implications for architecture-dependent interpretability.
- Target venue tier: EMNLP Main with careful framing.

### 9.4 Scenario D — All cells ambiguous

If pass criteria are missed across many cells, the paper becomes a methodology paper without a clean empirical headline.

**Paper structure if Scenario D obtains:**
- Cut Stage 4 from main text; report as preliminary in appendix.
- Lead with the pipeline contribution; relegate cross-task contrast to appendix.
- Target venue tier: EMNLP Findings or BlackboxNLP workshop.

### 9.5 Pre-registered scenario decision rule

The scenario is determined within 48 hours of the Stage 3 verdicts on all six headline cells (3 concepts × 2 tasks on GPT-J, correct only). Once determined, the paper structure pivots according to §9.1–9.4.

---

## PART 10 — COMPUTE BUDGET

### 10.1 Per-cell breakdown

For one cell (one concept × one layer × one task × one model, correct-only), with N ≈ 2,500–9,500 depending on task and model:

| Stage | Time per cell |
|---|---|
| 0 — projection + orthogonalisation | < 1 min |
| 1 — Phase C/D linear probe + bootstrap | 5 min |
| 2a — Fourier on centroids | 2 min |
| 2b — d_SW + Spearman | 1 min |
| 2c — Bayesian GPLVM (3 kernels × 3 seeds × 1 prior) | 3 hours |
| 2d — RBF VAE (3 kernel-equivalents × 2 seeds) | 1 hour |
| 3 — Phase H orthogonalisation + replay of 2a–d | 3 hours |
| 4 — causal ablation (only headline cells) | 1.5 hours |
| **Per cell, headline (with Stage 4)** | **~9 hours** |
| **Per cell, non-headline (no Stage 4)** | **~7 hours** |

### 10.2 Per-model totals

For GPT-J 6B, we run:
- 6 concepts × 3 layers × **1 population (correct)** = 18 cells.
- 6 cells get Stage 4 (the headline cells, middle layer only).
- 12 cells skip Stage 4.
- Total: 6 × 9 + 12 × 7 = 138 GPU-hours.

For Llama 3.1 8B (replication phase):
- Most Phase A–G done; we re-run Phase H + Stage 2 on 6 headline cells + Stage 4 on 6 cells.
- ~60 GPU-hours.

Plus initial activation extraction: ~10 GPU-hours.

**Total: ~210 GPU-hours.** Down from v5's ~310, mainly because dropping the wrong population halves the cell count.

### 10.3 Storage budget

Per cell, per model:
- Activations: 10,000 × 4096 × 4 = ~160 MB per layer (extracted on all problems before correct-filtering).
- Phase C/D outputs: 1 MB per cell.
- Phase G outputs: 1 MB per cell.
- Phase H outputs: 5 MB per cell.
- GPLVM outputs: 30 MB per cell.
- RBF VAE outputs: 20 MB per cell.

For 18 cells × 5 layers + per-model activation storage: ~5 GB per model. ~10 GB total.

### 10.4 Memory peaks

The largest in-memory operation is the GPLVM kernel matrix at N ≈ 9,500 (addition × Llama × correct subset) in float64. That's ~720 MB per kernel evaluation. With three kernels in parallel and gradient overhead: ~3 GB peak. Fits on 40 GB A100.

For the smallest cell (multiplication × GPT-J × correct subset, N ≈ 2,500), this drops to ~50 MB.

### 10.5 Cluster scheduling

Recommended setup: 4 concurrent A100s on the Babel cluster. With a 3-week dedicated allocation:
- Days 1–5: GPT-J Phase A–G on both tasks.
- Days 6–12: Phase H + Stage 2 on all GPT-J cells.
- Days 13–15: Stage 4 on GPT-J headline cells.
- Days 16–21: Llama replication.

Tighter than v5 because compute is half. Allow 1 week of buffer for re-runs.

---

## PART 11 — WEEK-BY-WEEK TIMELINE (16 WEEKS)

Down from v5's 18 weeks because compute is roughly halved.

### Week 1 — Lock decisions, set up infrastructure

- Day 1–2: Lock the rescope decision with Póczos. Confirm 0–99 scope, correct-only, first-token-correctness for multiplication.
- Day 3–4: Build addition + multiplication data generation scripts.
- Day 5: Smoke test: run GPT-J on 100 addition + 100 multiplication problems end-to-end. Verify first-token tokenisation works as expected.

**Deliverable:** addition + multiplication data, GPT-J infrastructure verified.

### Week 2 — GPT-J infrastructure

- Day 1–3: Port existing Llama Phase A–H scripts to GPT-J for both tasks.
- Day 4: Run inference on all 10,000 problems × both tasks; record correct subset.
- Day 5: Per-value count audit on correct subsets. Verify coverage.

**Deliverable:** GPT-J pipeline running on both tasks.

### Weeks 3–4 — GPT-J Phases C, D, G on both tasks

Run linear pipeline plus Fourier screening on all selected cells in GPT-J for both addition and multiplication. ~30 GPU-hours.

**Deliverable:** Stage 1 + Stage 2a results table for all GPT-J cells. **Pass-gate decision:** if Stage 1 fails for headline concepts, escalate.

### Week 5 — Stage 3 (Phase H) on GPT-J

Run the orthogonalisation control on all GPT-J cells. ~20 GPU-hours.

**Deliverable:** Stage 3 verdicts for all GPT-J cells. **Scenario decision:** within 48 hours, lock the paper's framing.

### Week 6 — Toy validation for Stage 2

Run the four toys for both GPLVM and RBF VAE.

**Deliverable:** toy validation report.

### Weeks 7–8 — GPLVM + RBF VAE on GPT-J cells

The biggest single compute block. ~80 GPU-hours.

**Deliverable:** Stage 2c + 2d results for all GPT-J cells.

### Week 9 — Stage 4 causal ablation on GPT-J headline cells

Run on 6 cells (one per concept × task at middle layer). ~9 GPU-hours.

**Deliverable:** Stage 4 ablation results.

### Weeks 10–11 — Llama replication

Run Phase H + Stage 2 + Stage 4 on the 6 headline cells in Llama. ~60 GPU-hours.

**Deliverable:** cross-model agreement panel.

### Week 12 — Figures lock

Generate all figures from final data.

**Deliverable:** all figures in PDF, regenerable from scripts.

### Weeks 13–14 — Draft

Draft order: Methods → Results → Related Work → Limitations → Discussion → Intro → Abstract → Title.

**Deliverable:** complete first draft.

### Week 15 — Revise and LaTeX port

**Deliverable:** submission-ready manuscript.

### Week 16 — Submission

**Deliverable:** submitted to ACL ARR or EMNLP main.

### Buffer

If schedule slips, drop in this order:
1. RBF VAE (Stage 2d).
2. Llama replication.
3. `ans_units` cells.
4. Two of the three layers.

---

## PART 12 — PRE-REGISTRATION

Pre-registration is the project's strongest defence against post-hoc rationalisation.

### 12.1 Pre-registered hypotheses

**H1 (probe success).** Stage 1 will pass (λ₁ ≥ 0.5 with bootstrap CI > 0) for all six headline concepts at the middle layer of GPT-J on the correct subset.

**H2 (manifold confirmation).** Stage 2 will give `strong_score_evidence` for at least one kernel choice in GPT-J × multiplication × `carry_units` × middle layer × correct.

**H3 (Phase H replication).** Stage 3 will give verdict `inherited` for GPT-J × multiplication × `carry_units` on the correct subset, replicating Phase H's Llama finding (which itself was 419/419 inherited including the correct-only branch).

**H4 (causal effect).** Stage 4 will show `Δlogit_raw > 95th percentile of random control` for at least the operand cells in both tasks.

### 12.2 Pre-registered thresholds

All thresholds are listed in §3 by stage. A summary:

| Threshold | Value | Sensitivity sweep |
|---|---|---|
| Stage 1 λ₁ | ≥ 0.5 | ±0.1 |
| Stage 1 cv_correlation | ≥ 0.7 | ±0.1 |
| Stage 2a helix FCR | ≥ 0.30 | ±0.05 |
| Stage 2b d_SW Spearman | ≥ 0.85 | ±0.10 |
| Stage 2c adjusted ELBO gap | ≥ 5 nats decisive | 2.3 nat moderate |
| Stage 3 owned-FCR drop | < 0.30 | sensitivity at 0.20, 0.50 |
| Stage 4 effect size | > 95th percentile of control | 90, 99 |

### 12.3 Pre-registered correlate sets for Stage 3

Locked per concept; we do not modify after seeing results.

### 12.4 Pre-registered scenario decision rule

§9.5 specifies the 48-hour decision rule from Stage 3 verdicts.

### 12.5 Pre-registered fallback rules

§11 buffer specifies the order in which features get dropped if compute slips.

### 12.6 Pre-registered correctness criterion

Multiplication uses **first-answer-token correctness** as the primary `correct` definition. Addition uses standard exact-match. Alternatives (a) single-token-restricted and (c) full-sequence-correctness are documented as fallbacks (§4.5) but not used unless the smoke test reveals an unexpected failure.

### 12.7 Pre-registered sensitivity analyses

For each headline result, report sensitivity at:
- Three permutation seeds for null distributions.
- Three GPLVM optimization seeds.
- Two layers (the headline layer and one neighbour).
- Two correctness criteria (primary first-token; backup single-token-restricted).

---

## PART 13 — TOY VALIDATION SUITE

### 13.1 Why toys

Every method in the pipeline must pass on synthetic data with known ground truth before being trusted on real activations.

### 13.2 The toys

For each toy: 200 points in 9 dimensions, σ_noise = 0.05.

**Toy L — Line.** 10 classes whose means lie on a 1D line.

**Toy C — Circle.** 200 points uniformly on a 2D circle.

**Toy H — Helix.** Circular helix in 3 of 9 dimensions.

**Toy G — Gaussian.** Isotropic 9D Gaussian.

### 13.3 Stage 3 toys

**Toy O — Owned helix.** Helix with a separate concept independent of helix structure. Verdict: `owned`.

**Toy I — Inherited helix.** Helix with a separate concept that's a deterministic function of helix angle. Verdict: `inherited`.

### 13.4 Stage 4 sanity check

Run on real GPT-J activations: ablate a known-irrelevant direction, verify Δlogit < random control.

### 13.5 Toy results report

The toy validation results go in the appendix (1 page).

---

## PART 14 — STATISTICAL DECISIONS AND THRESHOLDS

### 14.1 Significance level

α = 0.05 for all single-test comparisons. FDR-corrected at q = 0.10 across multiple cells per stage.

### 14.2 Bootstrap iterations

200 resamples for confidence intervals. 1,000 for permutation null distributions.

### 14.3 Multiple-comparison correction

For each stage, FDR-corrected within the cells of that stage. Across stages, no adjustment because the stages are pre-registered as conditional.

### 14.4 Effect sizes vs significance

Report effect sizes (Cohen's d for probe results, FCR drop for ownership, Δlogit ratio for ablation), not just p-values.

### 14.5 Confidence intervals

Bootstrap 95% CIs for every reported number.

### 14.6 The pre-registration document

A short document (~2 pages) gets timestamped and committed to git before any addition experiments run. Cite this in the methods section.

---

## PART 15 — THE 8-PAGE PAPER OUTLINE

| Section | Pages | Word count | What it does |
|---|---|---|---|
| 1. Introduction | 1.00 | ~600 | Hook, thesis, three contributions. |
| 2. Related work | 0.50 | ~300 | KT 2024, Bai 2025, Park 2024, Engels 2025, Hauberg 2018. |
| 3. Setup and tasks | 0.50 | ~300 | GPT-J 6B (primary), Llama (secondary), 0–99 addition + multiplication, correct only, six concepts. |
| 4. Methodology | 1.50 | ~900 | Four stages, applied identically. Pre-registration + correct-only + first-token-correctness paragraphs. |
| 5. Results | 2.50 | ~1500 | Verdict matrix, per-stage breakdowns, cross-task agreement, manifold visualisation. |
| 6. Discussion | 0.50 | ~300 | Mechanistic interpretation. |
| 7. Limitations (mandatory) | 0.50 | ~300 | Ten specific limitations including correct-only and first-token-correctness. |
| 8. Conclusion | 0.10 | ~80 | One paragraph. |
| References | unlimited | — | ~40 references. |
| Appendix | unlimited | — | Per-cell tables, toy validation, hyperparameter sensitivity, correctness-criterion sensitivity. |

Total: 8.00 content pages.

---

## PART 16 — SECTION-BY-SECTION PAPER PLAN

### 16.1 Title

Locked at submission.

### 16.2 Abstract (~180–200 words)

Beats:
1. Recent work shows pre-trained LLMs encode arithmetic numbers as helices and use these helices to compute (KT 2024).
2. We ask whether the helix actually belongs to the number.
3. Four-stage Bayesian pipeline.
4. Applied to addition and multiplication, a, b ∈ [0, 99], in GPT-J 6B on correct answers.
5. **Headline finding (Scenario A or B).**
6. Cross-model replication on Llama 3.1 8B.
7. Causal ablation results.
8. Reusable diagnostic.

### 16.3 Introduction structure (~7 paragraphs, 1 page)

Paragraph 1 — Hook.

Paragraph 2 — LRH background (Park 2024).

Paragraph 3 — The gap.

Paragraph 4 — Our approach.

Paragraph 5 — Contributions (5 bullet points).

Paragraph 6 — Roadmap.

Paragraph 7 — Pre-registration note: "All hypotheses, thresholds, scenario decisions, and the correct-only / first-token-correctness criteria were pre-registered before addition experiments ran."

### 16.4 Related work (~5–6 paragraphs)

Subsections cover LRH, arithmetic interpretability, probabilistic manifolds, causal interpretation, concept-specific subspaces, and explicit positioning against KT.

### 16.5 Setup and tasks (~4 paragraphs)

Subsection 3.1 — Models.

Subsection 3.2 — Tasks (addition and multiplication, both a, b ∈ [0, 99]; 10,000 problems each).

Subsection 3.3 — Correctness criterion: standard exact-match for addition (single-token); first-answer-token correctness for multiplication (with explicit acknowledgement of multi-token answer handling and reference to the alternatives in the appendix).

Subsection 3.4 — Concepts (six total: three per task, evaluated on the correct subset).

### 16.6 Methodology (~12 sub-paragraphs)

Subsection 4.1 — Pipeline overview.

Subsection 4.2 — Stage 1: linear probe.

Subsection 4.3 — Stage 2: Bayesian manifold characterisation.

Subsections 4.4–4.7 — Stages 2a–d.

Subsection 4.8 — Stage 3: orthogonalisation against algebraic correlates.

Subsection 4.9 — Stage 4: causal ablation with first-answer-token Δlogit.

Subsection 4.10 — Pre-registration note.

Subsection 4.11 — Toy validation summary.

Subsection 4.12 — Hauberg framing paragraph.

### 16.7 Results (~5 sub-paragraphs)

Subsection 5.1 — The verdict matrix (one figure).

Subsection 5.2 — Stage-by-stage breakdown for headline cells.

Subsection 5.3 — Cross-task pattern.

Subsection 5.4 — Cross-model agreement (Llama replication).

Subsection 5.5 — Manifold visualisation.

### 16.8 Discussion

Mechanistic interpretation of whichever scenario obtains.

### 16.9 Limitations (mandatory; ten items)

Ten limitations including the new correct-only and first-token-correctness ones; see Part 18.

### 16.10 Conclusion

One paragraph.

---

## PART 17 — FIGURES AND TABLES

### 17.1 Figure 1 — Pipeline overview (full-width)

Schematic of the four stages, colour-coded for owned/inherited/ambiguous.

### 17.2 Figure 2 — The verdict matrix (full-width)

The 2×3 matrix with checkmarks/crosses per stage per cell. Headline figure.

### 17.3 Figure 3 — Manifold visualisation panel (full-width)

Headline cell (multiplication × `carry_units` × GPT-J × layer 14, correct):
- Subplot A: Stage 2a centroid Fourier plot.
- Subplot B: Stage 2c GPLVM latent space with uncertainty bars.
- Subplot C: Stage 2d RBF VAE latent space (consistency check).
- Subplot D: Stage 3 orthogonalised version (helix collapsed).

### 17.4 Figure 4 — Cross-model agreement (single-column)

Bar chart for the six headline cells, GPT-J vs Llama on Stage 1 λ₁, Stage 3 FCR drop, Stage 4 Δlogit ratio.

### 17.5 Figure 5 — Sensitivity analysis (single-column)

For headline cells: stability across seeds, thresholds, layers, correctness criteria.

### 17.6 Tables

**Table 1 — Concept vocabulary.** Six concepts with algebraic correlate sets.

**Table 2 — Stage 1 results.** All cells, Stage 1 metrics (correct only).

**Table 3 — Stage 3 verdicts.** All cells, owned/inherited/ambiguous verdicts.

**Table 4 — Stage 4 results.** Headline cells, Δlogit raw vs orthogonalised vs random control.

**Table 5 — Correct-rate breakdown (new for v6).** Per task, per model, total correct count, per-value distribution after stratification.

### 17.7 Appendix figures

Per-cell GPLVM kernel comparisons, RBF VAE vs GPLVM agreement scatter, toy validation results, Phase H FCR/power scatter, **and a new correctness-criterion sensitivity panel** showing how verdicts change under (b) first-token-correct vs (a) single-token-restricted vs (c) full-sequence-correct.

---

## PART 18 — LIMITATIONS DRAFTED IN ADVANCE

Ten items now (up from eight in v5). The two new ones (§18.9 and §18.10) capture the v6 changes.

### 18.1 Limitation 1 — Two tasks, two models

We test addition and multiplication in two pre-trained LLMs. Generalisation to other arithmetic operations and to other models is future work.

### 18.2 Limitation 2 — 0–99 range only

We restrict to `a, b ∈ [0, 99]`. Larger operand ranges (3-digit, 4-digit) may show different patterns.

### 18.3 Limitation 3 — Finite correlate registry

Stage 3 orthogonalises against a hand-specified set of algebraic correlates. Concepts not in the registry remain invisible.

### 18.4 Limitation 4 — Linear orthogonalisation only

Stage 3 removes linear nuisance only. Nonlinear entanglement is not addressed.

### 18.5 Limitation 5 — GPLVM and RBF VAE share a mechanism

Both Bayesian methods rely on growing predictive uncertainty. Their agreement is consistency, not independent corroboration.

### 18.6 Limitation 6 — Causation is ablation-based

Stage 4 confirms ablation matters; it does not specify *how* the model uses the manifold.

### 18.7 Limitation 7 — Single position (the `=` token)

We extract activations at the `=` token only.

### 18.8 Limitation 8 — Pre-trained models, not training dynamics

We characterise representations in trained models.

### 18.9 Limitation 9 — Correct answers only (NEW)

> "All experiments are conducted on the model's correct answers only. We do not analyse representations from incorrect predictions. Our findings characterise the geometry of successful arithmetic computation; whether the same inheritance pattern holds for incorrect predictions is an open question. Earlier work in this project (Phase H, on Llama 3.1 8B multiplication) found that the inherited verdict held across `correct`, `wrong`, and combined populations (419 inherited each), suggesting our restriction does not bias the headline verdict, but we do not test this for the addition task or for GPT-J."

### 18.10 Limitation 10 — Multi-token answer handling for multiplication (NEW)

> "Multiplication answers in our 0–99 setup range up to 9801. For approximately 50–66% of multiplication problems (depending on model), this exceeds the single-token integer cap of the tokeniser. We define `correct` for multiplication as first-answer-token correctness — the model's first generated token after `=` matches the first BPE token of the gold answer. This matches Kantamneni and Tegmark 2024's effective protocol but is strictly weaker than full-sequence correctness. The model could generate the correct first token and then deviate; such cases are counted as `correct` in our protocol. Two alternative criteria — single-token-restricted and full-sequence-correct — are documented in the appendix; sensitivity analyses comparing them are included as Appendix figures."

---

## PART 19 — KT 2024 DIFFERENTIATION

### 19.1 Where we cite KT in the paper

Five citations, same as v5.

### 19.2 Three explicit differentiations

**Differentiator 1 — Bayesian uncertainty.** KT's helix fit is deterministic; ours is Bayesian with credible intervals.

**Differentiator 2 — Ownership test.** KT have no equivalent.

**Differentiator 3 — Multiplication application.** KT studied only addition.

**(One additional point we now share with KT v6 onwards.)** KT also restricted to correct answers only (their patching protocol filters to "correct prompts"). Our correct-only filtering is therefore *consistent with* KT's, not a deviation. This makes our addition replication tighter.

### 19.3 Rebuttal letter template

Same as v5; the addition of correct-only matches KT's protocol, so this strengthens rather than weakens the differentiation.

---

## PART 20 — HAUBERG FRAMING AND CITATION PLAN

Same as v5. No changes from data scope decisions.

---

## PART 21 — RISKS AND FALLBACKS

### 21.1 Stage-specific risks

| Risk | Probability | Fallback |
|---|---|---|
| Stage 1 fails on a headline concept in GPT-J | low | substitute concept per §5.4 |
| Stage 2c GPLVM doesn't converge for a cell | medium | report as `inference_failed` |
| Stage 2d RBF VAE diverges from GPLVM | medium-high | report disagreement rate |
| Stage 3 FCR drop ambiguous | medium | tighten threshold or drop cell |
| Stage 4 effect size below random control | low-medium | downgrade causal claim |

### 21.2 Scenario-specific paper structure pivots

Same as §9.

### 21.3 Compute-slip cascade

If schedule slips:
1. Drop RBF VAE.
2. Drop Llama replication.
3. Drop `ans_units` cells.
4. Drop two of three layers.

### 21.4 The "publishable backstop"

Single-model, single-task, Stages 1–3 only paper for BlackboxNLP workshop.

### 21.5 Multi-token answer rate risk (NEW for v6)

**Risk:** GPT-J multiplication on 0–99 with first-token correctness may give a correct rate below 20%, shrinking the correct subset to ≤ 2,000 problems. With three concepts each potentially having 8–10 distinct values, per-value sample counts could fall below the 30-floor for several values.

**Mitigation A:** Stratify the correct subset to maximise per-value coverage. Drop overrepresented values down rather than upsample underrepresented ones.

**Mitigation B:** If correct rate is below 15%, fall back to alternative (a) single-token-restricted (a × b ≤ 361 for GPT-J). Loses the 0–99 framing but recovers correctness rates above 50%.

**Mitigation C:** Report the multiplication × GPT-J cells as "exploratory" with explicit sample-size caveat in the paper. The headline carries on the addition × GPT-J cells and the multiplication × Llama cells (which have larger N).

**Decision rule:** Lock decision after week 2 smoke test. If GPT-J × multiplication first-token correct rate is:
- ≥ 30%: proceed with primary criterion (b) on full 0–99.
- 15–30%: proceed with (b) but mark cells exploratory.
- < 15%: fall back to (a) single-token-restricted.

### 21.6 Correct-only restriction risk (NEW for v6)

**Risk:** A reviewer asks "why correct only — doesn't this bias the geometry analysis toward problems the model already solved?"

**Response:** Phase H's existing inherited verdict on Llama held across correct, wrong, and combined populations (419 inherited each). This evidence already addresses the concern for at least one cell. We acknowledge this in Limitation 9 and report it as a sensitivity check in the appendix where compute permits.

**Pre-emptive defence:** Include a one-paragraph appendix section titled "Sensitivity to correct-vs-wrong restriction" that re-runs Phase H on Llama × multiplication × `carry_units` for the wrong subset (data already exists from the legacy project) and confirms the verdict holds.

---

## PART 22 — WHAT THIS PAPER DOES NOT DO

Updated for v6:

- **Causal mechanism beyond ablation.** Stage 4 ablation confirms directions matter; it does not specify how.
- **Steering or recovery.** Paper 2 territory.
- **More than two tasks.** Subtraction, division, modular arithmetic are next-paper material.
- **More than two models.** Pythia, OPT, Mistral are out of scope.
- **Operands beyond 0–99.** 3-digit and 4-digit operations are out of scope.
- **Wrong population (NEW for v6).** All experiments on correct only. Earlier project data on the wrong population is referenced as supporting evidence only.
- **Number-token positions (KT's primary position).** We use the `=` position; KT use number tokens.
- **The 9-stage geometric pipeline.** Persistent homology, multi-chart flows, SPD are all Paper-2 / Paper-3.
- **Sparse-autoencoder decomposition.** Acknowledged as the proper fix for unregistered superposition.
- **Training-dynamics analysis.** Trained models only.
- **Multi-token-answer geometry analysis (NEW for v6).** We use first-token correctness as the gating criterion but do not analyse the geometry of multi-token generation directly.

---

## PART 23 — REVIEWER-ATTACK PREP AND REBUTTAL TEMPLATES

### 23.1 The six most likely attacks (was five in v5; +1 for v6)

**Attack 1.** "This is just KT 2024 with extra steps." → Response in §19.3.

**Attack 2.** "Your GPLVM and RBF VAE aren't independent." → Acknowledged in §3.2.4; Stage 4 is the genuinely independent validation.

**Attack 3.** "Why GPT-J? Llama is more relevant for current research." → KT's primary model; Llama replication included; cross-model agreement is part of the contribution.

**Attack 4.** "Your prediction is too loose; either outcome is publishable." → Pre-registered both outcomes; this is more honest than rationalising one.

**Attack 5.** "Two tasks isn't enough for a general claim about LRH." → Agreed; framed as "consistent with"; subtraction/division are next steps.

**Attack 6 (NEW for v6).** "Correct-only filtering biases the analysis." → Phase H showed inherited verdict holds across correct, wrong, all populations; cite Limitation 9 and appendix sensitivity check; the bias concern is addressed.

**Attack 7 (NEW for v6).** "First-token correctness for multiplication is a weak criterion." → Matches KT's effective protocol; alternatives (a) and (c) are sensitivity-analysed in appendix; verdicts are stable across criteria.

### 23.2 Rebuttal letter template

Standard EMNLP rebuttal format. Pre-write responses to the seven attacks now.

---

## PART 24 — LLAMA 3.1 8B AS SECOND-STEP REPLICATION

### 24.1 When Llama starts

After GPT-J Stages 1–4 complete (week 9). Llama replication runs in weeks 10–11.

### 24.2 What we leverage from existing Llama work

- Phase A/B/C/D/E/F/JL: complete on Llama. We re-extract activations on the new 0–99 corpus rather than using the legacy 146,287-problem set.
- Phase G: re-run on the new corpus, focused on the six headline concepts.
- Phase H: re-run on the correct subset of the new corpus.
- Bayesian Stage 2 + Stage 4: new work.

**Why re-extract.** The legacy Llama work used L1–L5 with broader operand ranges (1–999 in some cells). To make cross-model comparison clean, we re-extract on the same 0–99 setup as GPT-J. ~5 GPU-hours for activation extraction.

### 24.3 Llama setup

Same as GPT-J but with Llama-specific layer indices (4, 16, 28). Same six concepts; same correlate sets; same correct-only filtering with first-token correctness.

### 24.4 The cross-model panel

For each of the six headline cells, report side-by-side GPT-J vs Llama Stage 1, 2, 3, 4 results.

### 24.5 What if Llama disagrees with GPT-J

Pivot framing toward "geometric ownership is partly model-dependent."

---

## PART 25 — APPENDIX MATERIAL PLAN

### 25.1 What goes in the appendix

EMNLP allows unlimited appendix.

**Appendix A — Toy validation.** All toys for Stages 1, 2, 3.

**Appendix B — Per-cell tables.** All cells × all stages.

**Appendix C — GPLVM hyperparameter sensitivity.**

**Appendix D — Phase H full results (legacy and new).**

**Appendix E — Causal ablation full protocol.**

**Appendix F — Pre-registration document.**

**Appendix G — Code release manifest.**

**Appendix H — Reproducibility instructions.**

**Appendix I — Correctness-criterion sensitivity (NEW for v6).** Comparison of Stage 3 verdicts under (b) first-token-correct vs (a) single-token-restricted vs (c) full-sequence-correct. Demonstrates verdict robustness.

**Appendix J — Correct-vs-wrong sensitivity check on Llama × multiplication (NEW for v6).** Re-runs Phase H on Llama wrong subset for `carry_units`; confirms inherited verdict.

Total appendix: ~22 pages.

---

## PART 26 — CODE RELEASE PLAN

### 26.1 What we release

- Pipeline scripts (Phase A–H + new Stages 2, 3, 4).
- Data: 10,000-problem addition set + 10,000-problem multiplication set, both with correctness labels per model.
- Activations: per-(model, layer, task) files, ~10 GB total.
- Configs and hyperparameters.
- Pre-registration document.
- Documentation.

### 26.2 Where we host

- Code: GitHub, Apache 2.0.
- Data + activations: Hugging Face dataset card.
- Manuscript: ACL Anthology after acceptance.

### 26.3 Anonymisation for double-blind

Strip author names, use placeholder URLs.

### 26.4 Reproducibility checklist

The EMNLP responsible NLP checklist requires source code, datasets, hyperparameters, infrastructure, runs, significance — all present.

---

## PART 27 — AUTHORSHIP AND ROLES

### 27.1 Author list

1. **Anshul Kumar** (CMU) — primary author.
2. **Deeksha Varshney** (IIT Jodhpur) — advisor.
3. **Manoj Kumar** (IIT Roorkee) — advisor.
4. **Barnabás Póczos** (CMU) — main advisor.

### 27.2 Roles

Anshul: pipeline implementation, GPT-J port, addition + multiplication data generation, all experiments, first-draft writing.

Deeksha Varshney: advisory review, methodological feedback, paper revision.

Manoj Kumar: advisory review, methodological feedback, paper revision.

Póczos: main-advisor methodological soundness review, statistical decisions, paper revision, technical claims auditing.

### 27.3 Acknowledgements

Compute support, useful discussions, anonymous reviewers.

---

## PART 28 — READING LIST

### 28.1 Must-read papers

Same set as v5: Park 2024, Engels 2025, KT 2024, Bai 2025, Nanda 2023, Hauberg 2018, Arvanitidis 2018, Tosi 2014, Vig 2020, Geiger 2021, Gurnee 2025, Yang 2024.

### 28.2 Methodology checks

Verify against the geometric pipeline document, Park 2024 for LRH, Hauberg 2018 for manifold-learning math.

### 28.3 Adversarial reading

Read the strongest critic for each headline claim.

---

## PART 29 — PRE-SUBMISSION CHECKLIST

### 29.1 Format

Standard EMNLP requirements.

### 29.2 Numbers

Every number checked against source CSVs.

### 29.3 Claims

Every abstract claim in Results.

### 29.4 Style

No em dashes; no "Notably"; active voice.

### 29.5 Reproducibility

Code, data, hyperparameters, seeds, compute.

### 29.6 Pre-submission audit

Read aloud, print and proofread, grayscale figures, cross-references.

### 29.7 New for v6

- [ ] Correct-only filtering documented in Methods §3.3.
- [ ] First-token-correctness for multiplication documented in Methods §3.3.
- [ ] Limitation 9 (correct only) included.
- [ ] Limitation 10 (first-token correctness) included.
- [ ] Appendix I (correctness-criterion sensitivity) included.
- [ ] Appendix J (correct-vs-wrong sensitivity) included.

---

## PART 30 — FINAL NOTES AND DECISION LOG

### 30.1 Locked decisions (May 2026, v6)

1. **Venue:** EMNLP 2026 main, long paper. Backup: BlackboxNLP.
2. **Authors:** Anshul Kumar (CMU, primary), Deeksha Varshney (IIT Jodhpur, advisor), Manoj Kumar (IIT Roorkee, advisor), Barnabás Póczos (CMU, main advisor).
3. **Primary model:** GPT-J 6B.
4. **Secondary model:** Llama 3.1 8B (replication phase).
5. **Tasks:** Addition + multiplication.
6. **Data scope (NEW v6):** a, b ∈ [0, 99] for both tasks; 10,000 problems each.
7. **Population (NEW v6):** Correct only; wrong population out of scope.
8. **Correctness criterion (NEW v6):** Standard exact-match for addition (single-token); first-answer-token correctness for multiplication.
9. **Concepts:** Six total (three per task).
10. **Pipeline:** Four stages.
11. **Title:** "From Linear Probes to Bayesian Manifolds: Geometry of Arithmetic in Language Models" (final).
12. **Pre-registration:** Mandatory before any addition experiments.
13. **Code release:** Yes, full.

### 30.2 Open decisions

1. **Specific GPT-J layer choice:** finalise after Phase A on GPT-J runs.
2. **RBF VAE inclusion in main body:** depends on agreement rate with GPLVM.
3. **`ans_units` cells:** depend on whether headline concepts give clean verdicts.
4. **Multi-token correctness fallback:** depends on smoke test outcome (§21.5 decision rule).

### 30.3 Decision log format

Every locked decision goes here with date and one-line justification.

### 30.4 Living-document policy

Updated as experiments land. Track via git. Major revisions get a new version number.

### 30.5 The single sentence that captures the paper

> "We propose a four-stage Bayesian pipeline (probe, manifold characterisation with uncertainty, ownership test via orthogonalisation against algebraic correlates, and causal ablation) for testing whether the geometric structure that linear probes find for a concept actually belongs to that concept, and we apply it to addition and multiplication (a, b ∈ [0, 99]) on the correct subset in GPT-J 6B with cross-model replication on Llama 3.1 8B."

That is the paper. Everything in this document is in service of that sentence.

---

## APPENDIX A — TIMELINE ATTACHMENT (16 WEEKS)

| Week | Activity | Compute | Deliverable | Pass-gate |
|---|---|---|---|---|
| 1 | Lock decisions; data; smoke test | < 1 GPU-hr | data + GPT-J infra | Smoke test passes; multi-token correct rate decision |
| 2 | GPT-J infra completion + correctness audit | 5 GPU-hr | correct subsets identified | Per-value counts ≥ 30 for ≥ 70% of values |
| 3 | GPT-J Phases C/D/G on addition | 15 GPU-hr | Stage 1 + 2a (addition) | Stage 1 passes |
| 4 | GPT-J Phases C/D/G on multiplication | 15 GPU-hr | Stage 1 + 2a (multiplication) | Stage 1 passes |
| 5 | Phase H on GPT-J both tasks | 20 GPU-hr | Stage 3 verdicts | **Scenario decision** |
| 6 | Toy validation | 5 GPU-hr | Toy report | All toys pass |
| 7 | GPLVM on GPT-J (half cells) | 50 GPU-hr | Stage 2c results (half) | Three-seed agreement |
| 8 | GPLVM on GPT-J (other half) + RBF VAE | 50 GPU-hr | Stage 2c + 2d complete | GPLVM/VAE agreement ≥ 80% |
| 9 | Stage 4 on GPT-J headline cells | 9 GPU-hr | Stage 4 (GPT-J) | Effect sizes within range |
| 10 | Llama re-extract + Stage 2 | 35 GPU-hr | Stage 2 (Llama) | Cross-model agreement |
| 11 | Llama Stage 4 | 6 GPU-hr | Stage 4 (Llama) | |
| 12 | Figures lock | 0 | All figures PDF | Regenerable |
| 13 | Draft Methods + Results | 0 | First half | Co-author review |
| 14 | Draft Related Work + rest | 0 | Full draft | Co-author review |
| 15 | LaTeX + revision | 0 | Submission-ready | Format check |
| 16 | Submission | 0 | Submitted | Confirmation |

---

## APPENDIX B — KEY PHRASINGS WE COMMIT TO

**The thesis (use one of these three; don't mix in a single section):**
- "Linear probe success does not imply representational ownership."
- "The helix that the probe finds may not belong to the concept it labels."
- "Geometric inheritance is detectable with the right pipeline."

**The model:** "GPT-J 6B" first; "GPT-J" thereafter.

**The tasks:** "Addition" and "multiplication."

**The population:** "Correct" — never "successful" or "non-failure."

**The correctness criterion (always specify):** "first-answer-token correctness" for multiplication; "exact-match" for addition.

**The data scope:** "a, b ∈ [0, 99]" — always with the brackets.

**The pipeline stages:** "Stage 1 (linear probe)" etc.

**The verdicts:** "Owned", "inherited", "ambiguous."

**The contribution:** "Four-stage Bayesian pipeline."

**The outcomes:** "Finding A" and "Finding B."

---

## APPENDIX C — SAMPLE ABSTRACT (DRAFT V1, v6)

> Recent work shows that pre-trained large language models encode arithmetic numbers as helices in residual stream subspaces, and use these helices to perform addition (Kantamneni and Tegmark, 2024). The interpretation has been that the helix belongs to the number. We ask whether it does. We propose a four-stage Bayesian pipeline that combines linear probing, Bayesian manifold characterisation with uncertainty (centroid and full-cloud), an ownership test via orthogonalisation against algebraically related correlates, and causal ablation. Applied to addition and multiplication (a, b ∈ [0, 99]) in GPT-J 6B on the model's correct answers, the pipeline reveals that {Finding A: addition has owned helices but multiplication's apparent helix is inherited from algebraic intermediates; Finding B: even addition shows partial inheritance, indicating that linear-probe ownership reading is broadly wrong for arithmetic}. Cross-model replication on Llama 3.1 8B confirms the pattern. Causal ablation shows that ablating the orthogonalised inherited subspace barely affects model behaviour. We propose the "ingredient signal minus owned signal" gap as a reusable diagnostic for inherited geometry in any concept-probe study.

---

## APPENDIX D — SAMPLE INTRODUCTION OPENING (DRAFT V1, v6)

> Linear probes for arithmetic concepts in pre-trained language models find clean geometric structure. Numbers encode as helices (Kantamneni and Tegmark, 2024). Carries encode as low-dimensional subspaces. Operands encode as one-dimensional rays. The interpretation, following the Linear Representation Hypothesis (Park et al., 2024), is that each concept owns the geometry that probes uncover for it.
>
> We ask whether this interpretation is correct. The Linear Representation Hypothesis says that concepts are represented as linear directions; it does not say that linear probes can identify the *concept-specific* directions, free from confounding with structurally related concepts. In arithmetic, many concepts share algebraic structure: a carry value is determined by the partial product, which is determined by the operand digits, which are encoded jointly with their decompositions. If these structurally related concepts share residual-stream geometry, a linear probe for any one of them will find the shared geometry and label it as belonging to the probed concept.
>
> This paper introduces a four-stage Bayesian pipeline that tests directly whether the geometry a probe finds belongs to the probed concept or is inherited from related concepts. We apply the pipeline to addition and multiplication (a, b ∈ [0, 99]) in GPT-J 6B on the model's correct answers, with cross-model replication on Llama 3.1 8B.

---

## APPENDIX E — DECISION CHECKLIST FOR RIGHT NOW

If the rescope is approved, the next ten things to do:

1. Lock the v6 rescope decision with Póczos in writing — specifically confirm 0–99 scope, correct-only restriction, first-token correctness for multiplication.
2. Create a fresh git branch for v6.
3. Write the pre-registration document (~2 pages); commit with timestamp.
4. Build `generate_addition.py` and `generate_multiplication.py`. Verify single-token property for addition; verify first-token tokenisation works for multiplication answers up to 9801.
5. Run GPT-J on 100 addition + 100 multiplication problems as smoke test. Lock the multi-token correctness criterion based on §21.5 decision rule.
6. Port Phase A to GPT-J (one script, one day).
7. Port Phase H to GPT-J (one script, one day).
8. Reserve the GPU pool: 4 A100s for 4 weeks, with priority queue access.
9. Set up the toy validation harness for Stage 2.
10. Schedule the Week 5 scenario-decision meeting with Póczos.

---

*End of complete end-to-end plan, version 6.*
*Last updated: May 2026.*
*Next major revision after Stage 3 verdicts land (week 5).*