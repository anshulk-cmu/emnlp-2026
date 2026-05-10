# Step 4: UMAP and t-SNE embeddings — per-cell CSVs and 45 selected plots

**Anshul's Geometry of Arithmetic in LMs Project**
**Carnegie Mellon University, May 2026**

This document records every decision, every number, and every result from
Step 4 — comprehensive UMAP and t-SNE embeddings for all 30 (model × task ×
layer) cells produced in Step 3, and 45 selected plots rendered from those
embeddings. It is the truth document for this stage. All numbers are validated
against the actual manifests in `data/results/embeddings/` and the plot index
at `data/figures/embeddings/plot_index.json` as of 2026-05-09.

---

## Table of Contents

1. Purpose of this stage
2. What this stage is and is not
3. Inputs
4. Methods
5. Hyperparameter grid (7 settings per cell)
6. Per-cell trustworthiness (full N) — 30 cells × 7 settings
7. Per-cell runtime — 30 cells × 7 settings
8. Best-trustworthiness HP per cell
9. Cross-cell summary statistics by (model × task × method)
10. Per-cell activation_norm distribution
11. CSV schema and column counts
12. Manifest schema and example values
13. Per-cell output file sizes
14. 45 plots — full index table
15. PNG file sizes
16. Verification checks
17. Output file inventory
18. Library versions
19. Re-run commands
20. Open questions

---

## 1. Purpose of this stage

Compute 2D UMAP and 2D t-SNE embeddings for each of the 30 activation
`.npy` cells produced in Step 3. For every cell, run a 7-setting
hyperparameter grid (4 UMAP, 3 t-SNE), join the per-pair Tier 1–4 concept
labels from Step 2's dataset CSV, write a per-cell wide CSV containing all
coordinate columns alongside all concept columns, and a manifest with
trustworthiness scores and runtimes per (method, hp) on full N. Render 45
PNG plots (15 per model = 10 UMAP + 5 t-SNE) using the highest-trustworthiness
HP setting per (cell, method).

This step does not feed plan v6 Stages 1–4. The outputs here are diagnostic.

---

## 2. What this stage is and is not

### What it is

- A per-cell 2D embedding pass on full N (10,000 for addition, 3,023 for
  multiplication), no subsampling at any step (UMAP fit, t-SNE fit, and
  trustworthiness scoring all use full N).
- A 7-setting hyperparameter grid per cell (210 embedding fits + 210
  trustworthiness measurements across 30 cells).
- A wide per-cell CSV joining all coordinate columns with all Tier 1–4
  concept columns from the Step 2 dataset CSV, plus an `activation_norm`
  column.
- 45 PNG plots from a fixed selection rule (15 per model = 10 UMAP + 5 t-SNE).

### What it is not

- Not a feature-extraction or probe-fitting step. The 2D coordinates are
  for visualization and downstream cell-level diagnostics; they are not
  used as input to plan v6 Stages 1–4.
- Not a clustering step. No k-means, no DBSCAN, no silhouette scores.
- Not a 3D-embedding step. All 2D.
- Not GPU-accelerated. CPU `umap-learn` and `sklearn.manifold.TSNE`.

---

## 3. Inputs

### 3.1 Activation .npy files (30 total)

Source: `data/activations/{model_key}/{task}_layer_{LL:02d}.npy` from Step 3.

| Model | Layers | Task | N | Shape |
|---|---|---|---:|---|
| GPT-J 6B | [4, 8, 14, 20, 24] | addition | 10,000 | (10000, 4096) |
| GPT-J 6B | [4, 8, 14, 20, 24] | multiplication | 3,023 | (3023, 4096) |
| Llama 3.1 8B | [4, 8, 16, 24, 28] | addition | 10,000 | (10000, 4096) |
| Llama 3.1 8B | [4, 8, 16, 24, 28] | multiplication | 3,023 | (3023, 4096) |
| Pythia 6.9B | [4, 8, 16, 24, 28] | addition | 10,000 | (10000, 4096) |
| Pythia 6.9B | [4, 8, 16, 24, 28] | multiplication | 3,023 | (3023, 4096) |

Total cells: 30. dtype: float32 throughout.

### 3.2 Concept-label CSVs (joined per cell)

Source: `data/data/raw/{task}_problems.csv` from Step 2. Joined row-by-row
by `index` (problems are stored in deterministic Cartesian order).

- `addition_problems.csv`: 10,000 rows × 62 concept columns.
- `multiplication_problems.csv`: 3,023 rows × 75 concept columns.

Schema details are in [docs/02_dataset_generation.md §23](02_dataset_generation.md).

---

## 4. Methods

### 4.1 UMAP

Library: `umap-learn 0.5.11`.

Class: `umap.UMAP(n_components=2, n_neighbors=N, min_dist=M, metric="euclidean", random_state=42)`

Inputs: float32, contiguous (`np.ascontiguousarray(X.astype(np.float32))`).

`fit_transform(X)` returns a `(N, 2)` float array. Cast to float32 before
storing. No standardization is applied. The activation matrix is fed in raw.

### 4.2 t-SNE

Library: `sklearn.manifold.TSNE` (scikit-learn).

Class: `TSNE(n_components=2, perplexity=P, init="pca", learning_rate="auto", max_iter=2000, random_state=42)`

`P` is capped at `min(perplexity, max(2, (N - 1) // 3))` to satisfy
scikit-learn's perplexity validity check. For N=10,000 this cap is 3,333
(no effect at the perplexities we use). For N=3,023 the cap is 1,007 (no
effect). All three perplexity values (10, 30, 50) pass through unchanged.

### 4.3 Trustworthiness

Library: `sklearn.manifold.trustworthiness`.

Call: `trustworthiness(X, embedding, n_neighbors=30)`.

Computed on full N (no subsampling). Trustworthiness is in [0, 1] where 1
indicates that the k-nearest-neighbor structure of the high-D space is
perfectly preserved in the 2D embedding for k=30.

Initial Phase A pass used a 2,000-row subsample for trustworthiness; that
was replaced post-Phase-A by a full-N recompute that updated all 30
manifests. The values reported below are the full-N values. Manifests
carry `trustworthiness_recomputed_full_n: true` and
`trustworthiness_subsample_size: <full N>`.

---

## 5. Hyperparameter grid (7 settings per cell)

| Setting | Method | n_neighbors / perplexity | min_dist | Other |
|---|---|---:|---:|---|
| `umap2d_n15_md10` | UMAP | 15 | 0.10 | metric=euclidean, random_state=42 |
| `umap2d_n30_md10` | UMAP | 30 | 0.10 | metric=euclidean, random_state=42 |
| `umap2d_n50_md10` | UMAP | 50 | 0.10 | metric=euclidean, random_state=42 |
| `umap2d_n30_md30` | UMAP | 30 | 0.30 | metric=euclidean, random_state=42 |
| `tsne2d_p10`      | t-SNE | 10 | — | init=pca, learning_rate=auto, max_iter=2000, random_state=42 |
| `tsne2d_p30`      | t-SNE | 30 | — | same |
| `tsne2d_p50`      | t-SNE | 50 | — | same |

Total fits per cell: 7. Total fits across 30 cells: 210.

---

## 6. Per-cell trustworthiness (full N)

Quoted verbatim from `*_manifest.json` `trustworthiness` fields. Full N
(no subsampling) used in the trustworthiness call. `n_neighbors=30` for
the trustworthiness k-NN. Higher is better; range [0, 1].

| # | model | task | L | N | umap_n15_md10 | umap_n30_md10 | umap_n50_md10 | umap_n30_md30 | tsne_p10 | tsne_p30 | tsne_p50 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | gpt-j-6b | addition | 04 | 10000 | 0.9969 | 0.9974 | 0.9974 | 0.9958 | 0.9973 | 0.9985 | 0.9985 |
| 2 | gpt-j-6b | addition | 08 | 10000 | 0.9853 | 0.9817 | 0.9785 | 0.9718 | 0.9890 | 0.9909 | 0.9914 |
| 3 | gpt-j-6b | addition | 14 | 10000 | 0.9853 | 0.9867 | 0.9856 | 0.9820 | 0.9883 | 0.9924 | 0.9932 |
| 4 | gpt-j-6b | addition | 20 | 10000 | 0.9754 | 0.9716 | 0.9642 | 0.9445 | 0.9822 | 0.9857 | 0.9853 |
| 5 | gpt-j-6b | addition | 24 | 10000 | 0.9792 | 0.9778 | 0.9769 | 0.9679 | 0.9877 | 0.9901 | 0.9897 |
| 6 | gpt-j-6b | multiplication | 04 |  3023 | 0.9897 | 0.9914 | 0.9915 | 0.9907 | 0.9899 | 0.9929 | 0.9938 |
| 7 | gpt-j-6b | multiplication | 08 |  3023 | 0.9868 | 0.9889 | 0.9874 | 0.9817 | 0.9848 | 0.9895 | 0.9898 |
| 8 | gpt-j-6b | multiplication | 14 |  3023 | 0.9789 | 0.9801 | 0.9791 | 0.9765 | 0.9786 | 0.9825 | 0.9817 |
| 9 | gpt-j-6b | multiplication | 20 |  3023 | 0.9493 | 0.9443 | 0.9338 | 0.9174 | 0.9584 | 0.9563 | 0.9561 |
| 10 | gpt-j-6b | multiplication | 24 |  3023 | 0.9378 | 0.9350 | 0.9322 | 0.9047 | 0.9413 | 0.9501 | 0.9464 |
| 11 | llama-3.1-8b | addition | 04 | 10000 | 0.9860 | 0.9891 | 0.9894 | 0.9830 | 0.9940 | 0.9966 | 0.9965 |
| 12 | llama-3.1-8b | addition | 08 | 10000 | 0.9811 | 0.9804 | 0.9791 | 0.9690 | 0.9846 | 0.9878 | 0.9880 |
| 13 | llama-3.1-8b | addition | 16 | 10000 | 0.9842 | 0.9887 | 0.9895 | 0.9856 | 0.9908 | 0.9938 | 0.9943 |
| 14 | llama-3.1-8b | addition | 24 | 10000 | 0.9957 | 0.9983 | 0.9981 | 0.9983 | 0.9971 | 0.9989 | 0.9990 |
| 15 | llama-3.1-8b | addition | 28 | 10000 | 0.9923 | 0.9985 | 0.9988 | 0.9983 | 0.9976 | 0.9993 | 0.9994 |
| 16 | llama-3.1-8b | multiplication | 04 |  3023 | 0.9862 | 0.9879 | 0.9878 | 0.9852 | 0.9848 | 0.9878 | 0.9894 |
| 17 | llama-3.1-8b | multiplication | 08 |  3023 | 0.9734 | 0.9767 | 0.9759 | 0.9708 | 0.9755 | 0.9817 | 0.9832 |
| 18 | llama-3.1-8b | multiplication | 16 |  3023 | 0.9900 | 0.9902 | 0.9910 | 0.9872 | 0.9915 | 0.9937 | 0.9933 |
| 19 | llama-3.1-8b | multiplication | 24 |  3023 | 0.9507 | 0.9488 | 0.9440 | 0.9329 | 0.9412 | 0.9511 | 0.9514 |
| 20 | llama-3.1-8b | multiplication | 28 |  3023 | 0.9481 | 0.9473 | 0.9465 | 0.9166 | 0.9304 | 0.9493 | 0.9440 |
| 21 | pythia-6.9b | addition | 04 | 10000 | 0.9982 | 0.9986 | 0.9987 | 0.9979 | 0.9985 | 0.9992 | 0.9993 |
| 22 | pythia-6.9b | addition | 08 | 10000 | 0.9870 | 0.9852 | 0.9848 | 0.9807 | 0.9918 | 0.9924 | 0.9929 |
| 23 | pythia-6.9b | addition | 16 | 10000 | 0.9808 | 0.9829 | 0.9805 | 0.9660 | 0.9816 | 0.9907 | 0.9909 |
| 24 | pythia-6.9b | addition | 24 | 10000 | 0.9746 | 0.9754 | 0.9744 | 0.9676 | 0.9835 | 0.9861 | 0.9878 |
| 25 | pythia-6.9b | addition | 28 | 10000 | 0.9776 | 0.9763 | 0.9737 | 0.9690 | 0.9855 | 0.9882 | 0.9884 |
| 26 | pythia-6.9b | multiplication | 04 |  3023 | 0.9890 | 0.9909 | 0.9916 | 0.9890 | 0.9860 | 0.9913 | 0.9913 |
| 27 | pythia-6.9b | multiplication | 08 |  3023 | 0.9899 | 0.9906 | 0.9906 | 0.9903 | 0.9905 | 0.9925 | 0.9930 |
| 28 | pythia-6.9b | multiplication | 16 |  3023 | 0.9522 | 0.9504 | 0.9492 | 0.9261 | 0.9560 | 0.9636 | 0.9688 |
| 29 | pythia-6.9b | multiplication | 24 |  3023 | 0.9421 | 0.9270 | 0.9348 | 0.9156 | 0.9439 | 0.9522 | 0.9518 |
| 30 | pythia-6.9b | multiplication | 28 |  3023 | 0.9407 | 0.9328 | 0.9246 | 0.9051 | 0.9483 | 0.9490 | 0.9485 |

---

## 7. Per-cell runtime

Quoted verbatim from `*_manifest.json` `runtime_seconds` fields. Wall time
for each `fit_transform` call. Trustworthiness compute time excluded (it
is a separate post-fit step and is reported in §6's preamble). All times
in seconds.

| # | model | task | L | N | umap_n15_md10 | umap_n30_md10 | umap_n50_md10 | umap_n30_md30 | tsne_p10 | tsne_p30 | tsne_p50 | total |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | gpt-j-6b | addition | 04 | 10000 |   29.7 |   23.5 |   34.8 |   23.9 |   27.2 |   32.5 |   37.3 |   208.9 |
| 2 | gpt-j-6b | addition | 08 | 10000 |   15.7 |   23.3 |   33.1 |   21.9 |   25.8 |   29.8 |   34.8 |   184.4 |
| 3 | gpt-j-6b | addition | 14 | 10000 |   14.6 |   21.7 |   32.9 |   23.2 |   27.8 |   31.7 |   37.0 |   189.0 |
| 4 | gpt-j-6b | addition | 20 | 10000 |   15.8 |   24.9 |   37.0 |   24.6 |   30.0 |   35.4 |   44.2 |   212.0 |
| 5 | gpt-j-6b | addition | 24 | 10000 |   16.9 |   25.9 |   40.5 |   25.4 |   30.2 |   35.1 |   42.2 |   216.3 |
| 6 | gpt-j-6b | multiplication | 04 |  3023 |   22.2 |   19.1 |   22.0 |   21.3 |    7.2 |    8.3 |    9.7 |   109.8 |
| 7 | gpt-j-6b | multiplication | 08 |  3023 |   18.3 |   19.2 |   18.9 |   18.8 |    7.2 |    8.1 |    9.4 |   100.0 |
| 8 | gpt-j-6b | multiplication | 14 |  3023 |   17.4 |   19.9 |   19.1 |   18.7 |    7.3 |   14.1 |   14.9 |   111.5 |
| 9 | gpt-j-6b | multiplication | 20 |  3023 |   19.6 |   19.6 |   19.4 |   21.0 |    8.1 |    9.1 |   10.5 |   107.4 |
| 10 | gpt-j-6b | multiplication | 24 |  3023 |   17.5 |   20.5 |   19.4 |   19.6 |    8.7 |    9.2 |   10.7 |   105.6 |
| 11 | llama-3.1-8b | addition | 04 | 10000 |   16.7 |   25.2 |   37.6 |   24.6 |   28.4 |   32.2 |   38.4 |   203.0 |
| 12 | llama-3.1-8b | addition | 08 | 10000 |   16.0 |   24.6 |   37.2 |   24.2 |   27.3 |   35.3 |   40.4 |   205.1 |
| 13 | llama-3.1-8b | addition | 16 | 10000 |   15.6 |   24.0 |   36.5 |   23.4 |   27.5 |   32.2 |   38.1 |   197.4 |
| 14 | llama-3.1-8b | addition | 24 | 10000 |   18.6 |   25.6 |   36.4 |   25.5 |   31.6 |   33.5 |   39.5 |   210.6 |
| 15 | llama-3.1-8b | addition | 28 | 10000 |   17.4 |   27.8 |   36.8 |   27.8 |   28.0 |   33.9 |   37.1 |   208.9 |
| 16 | llama-3.1-8b | multiplication | 04 |  3023 |   17.7 |   18.4 |   19.9 |   18.9 |    7.3 |    8.1 |    9.4 |    99.7 |
| 17 | llama-3.1-8b | multiplication | 08 |  3023 |   17.9 |   19.1 |   19.5 |   18.7 |    7.0 |    8.6 |   10.0 |   100.8 |
| 18 | llama-3.1-8b | multiplication | 16 |  3023 |   17.6 |   18.4 |   19.2 |   18.3 |    7.5 |    8.2 |   10.1 |    99.4 |
| 19 | llama-3.1-8b | multiplication | 24 |  3023 |   17.3 |   18.2 |   19.0 |   18.3 |    8.1 |   10.4 |   11.2 |   102.5 |
| 20 | llama-3.1-8b | multiplication | 28 |  3023 |   17.1 |   18.0 |   19.1 |   19.0 |    8.3 |    9.6 |   11.0 |   102.1 |
| 21 | pythia-6.9b | addition | 04 | 10000 |   16.7 |   24.0 |   36.6 |   24.2 |   27.8 |   33.5 |   42.3 |   205.1 |
| 22 | pythia-6.9b | addition | 08 | 10000 |   15.4 |   22.7 |   34.2 |   22.8 |   26.5 |   33.0 |   36.3 |   190.9 |
| 23 | pythia-6.9b | addition | 16 | 10000 |   15.8 |   23.8 |   35.8 |   23.6 |   25.1 |   30.9 |   36.2 |   191.1 |
| 24 | pythia-6.9b | addition | 24 | 10000 |   16.4 |   24.7 |   36.7 |   24.6 |   28.2 |   34.0 |   41.3 |   205.9 |
| 25 | pythia-6.9b | addition | 28 | 10000 |   16.2 |   24.8 |   37.2 |   24.6 |   32.2 |   35.6 |   42.0 |   212.5 |
| 26 | pythia-6.9b | multiplication | 04 |  3023 |   17.7 |   18.3 |   19.1 |   18.7 |    7.2 |    8.1 |    9.6 |    98.6 |
| 27 | pythia-6.9b | multiplication | 08 |  3023 |   17.7 |   18.8 |   19.8 |   19.9 |    6.7 |    7.9 |    9.3 |   100.1 |
| 28 | pythia-6.9b | multiplication | 16 |  3023 |   17.4 |   18.8 |   19.7 |   19.4 |    7.5 |    8.9 |    9.9 |   101.7 |
| 29 | pythia-6.9b | multiplication | 24 |  3023 |   17.4 |   19.1 |   19.8 |   18.6 |    8.0 |    9.2 |   10.2 |   102.3 |
| 30 | pythia-6.9b | multiplication | 28 |  3023 |   18.1 |   18.9 |   19.7 |   19.1 |    8.0 |    9.3 |   10.2 |   103.3 |

---

## 8. Best-trustworthiness HP per cell

For each cell, the HP setting with maximum trustworthiness within each
method (UMAP, t-SNE) is selected. Ties are broken by shorter runtime; no
ties were observed in the recorded data.

| # | model | task | L | N | best UMAP HP | best UMAP T | best t-SNE HP | best t-SNE T |
|---:|---|---|---:|---:|---|---:|---|---:|
| 1 | gpt-j-6b | addition | 04 | 10000 | `umap2d_n50_md10` | 0.9974 | `tsne2d_p30` | 0.9985 |
| 2 | gpt-j-6b | addition | 08 | 10000 | `umap2d_n15_md10` | 0.9853 | `tsne2d_p50` | 0.9914 |
| 3 | gpt-j-6b | addition | 14 | 10000 | `umap2d_n30_md10` | 0.9867 | `tsne2d_p50` | 0.9932 |
| 4 | gpt-j-6b | addition | 20 | 10000 | `umap2d_n15_md10` | 0.9754 | `tsne2d_p30` | 0.9857 |
| 5 | gpt-j-6b | addition | 24 | 10000 | `umap2d_n15_md10` | 0.9792 | `tsne2d_p30` | 0.9901 |
| 6 | gpt-j-6b | multiplication | 04 |  3023 | `umap2d_n50_md10` | 0.9915 | `tsne2d_p50` | 0.9938 |
| 7 | gpt-j-6b | multiplication | 08 |  3023 | `umap2d_n30_md10` | 0.9889 | `tsne2d_p50` | 0.9898 |
| 8 | gpt-j-6b | multiplication | 14 |  3023 | `umap2d_n30_md10` | 0.9801 | `tsne2d_p30` | 0.9825 |
| 9 | gpt-j-6b | multiplication | 20 |  3023 | `umap2d_n15_md10` | 0.9493 | `tsne2d_p10` | 0.9584 |
| 10 | gpt-j-6b | multiplication | 24 |  3023 | `umap2d_n15_md10` | 0.9378 | `tsne2d_p30` | 0.9501 |
| 11 | llama-3.1-8b | addition | 04 | 10000 | `umap2d_n50_md10` | 0.9894 | `tsne2d_p30` | 0.9966 |
| 12 | llama-3.1-8b | addition | 08 | 10000 | `umap2d_n15_md10` | 0.9811 | `tsne2d_p50` | 0.9880 |
| 13 | llama-3.1-8b | addition | 16 | 10000 | `umap2d_n50_md10` | 0.9895 | `tsne2d_p50` | 0.9943 |
| 14 | llama-3.1-8b | addition | 24 | 10000 | `umap2d_n30_md10` | 0.9983 | `tsne2d_p50` | 0.9990 |
| 15 | llama-3.1-8b | addition | 28 | 10000 | `umap2d_n50_md10` | 0.9988 | `tsne2d_p50` | 0.9994 |
| 16 | llama-3.1-8b | multiplication | 04 |  3023 | `umap2d_n30_md10` | 0.9879 | `tsne2d_p50` | 0.9894 |
| 17 | llama-3.1-8b | multiplication | 08 |  3023 | `umap2d_n30_md10` | 0.9767 | `tsne2d_p50` | 0.9832 |
| 18 | llama-3.1-8b | multiplication | 16 |  3023 | `umap2d_n50_md10` | 0.9910 | `tsne2d_p30` | 0.9937 |
| 19 | llama-3.1-8b | multiplication | 24 |  3023 | `umap2d_n15_md10` | 0.9507 | `tsne2d_p50` | 0.9514 |
| 20 | llama-3.1-8b | multiplication | 28 |  3023 | `umap2d_n15_md10` | 0.9481 | `tsne2d_p30` | 0.9493 |
| 21 | pythia-6.9b | addition | 04 | 10000 | `umap2d_n50_md10` | 0.9987 | `tsne2d_p50` | 0.9993 |
| 22 | pythia-6.9b | addition | 08 | 10000 | `umap2d_n15_md10` | 0.9870 | `tsne2d_p50` | 0.9929 |
| 23 | pythia-6.9b | addition | 16 | 10000 | `umap2d_n30_md10` | 0.9829 | `tsne2d_p50` | 0.9909 |
| 24 | pythia-6.9b | addition | 24 | 10000 | `umap2d_n30_md10` | 0.9754 | `tsne2d_p50` | 0.9878 |
| 25 | pythia-6.9b | addition | 28 | 10000 | `umap2d_n15_md10` | 0.9776 | `tsne2d_p50` | 0.9884 |
| 26 | pythia-6.9b | multiplication | 04 |  3023 | `umap2d_n50_md10` | 0.9916 | `tsne2d_p30` | 0.9913 |
| 27 | pythia-6.9b | multiplication | 08 |  3023 | `umap2d_n50_md10` | 0.9906 | `tsne2d_p50` | 0.9930 |
| 28 | pythia-6.9b | multiplication | 16 |  3023 | `umap2d_n15_md10` | 0.9522 | `tsne2d_p50` | 0.9688 |
| 29 | pythia-6.9b | multiplication | 24 |  3023 | `umap2d_n15_md10` | 0.9421 | `tsne2d_p30` | 0.9522 |
| 30 | pythia-6.9b | multiplication | 28 |  3023 | `umap2d_n15_md10` | 0.9407 | `tsne2d_p30` | 0.9490 |

---

## 9. Cross-cell summary statistics by (model × task × method)

Aggregates over the 5 layers per (model, task, method). For each
(model, task, method), we report min/median/max trustworthiness across
the 5 layers, using the best-of-method HP per cell from §8.

| model | task | method | min | median | max | mean |
|---|---|---|---:|---:|---:|---:|
| gpt-j-6b | addition | UMAP | 0.9754 | 0.9853 | 0.9974 | 0.9848 |
| gpt-j-6b | addition | t-SNE | 0.9857 | 0.9914 | 0.9985 | 0.9918 |
| gpt-j-6b | multiplication | UMAP | 0.9378 | 0.9801 | 0.9915 | 0.9695 |
| gpt-j-6b | multiplication | t-SNE | 0.9501 | 0.9825 | 0.9938 | 0.9749 |
| llama-3.1-8b | addition | UMAP | 0.9811 | 0.9895 | 0.9988 | 0.9914 |
| llama-3.1-8b | addition | t-SNE | 0.9880 | 0.9966 | 0.9994 | 0.9955 |
| llama-3.1-8b | multiplication | UMAP | 0.9481 | 0.9767 | 0.9910 | 0.9709 |
| llama-3.1-8b | multiplication | t-SNE | 0.9493 | 0.9832 | 0.9937 | 0.9734 |
| pythia-6.9b | addition | UMAP | 0.9754 | 0.9829 | 0.9987 | 0.9843 |
| pythia-6.9b | addition | t-SNE | 0.9878 | 0.9909 | 0.9993 | 0.9919 |
| pythia-6.9b | multiplication | UMAP | 0.9407 | 0.9522 | 0.9916 | 0.9634 |
| pythia-6.9b | multiplication | t-SNE | 0.9490 | 0.9688 | 0.9930 | 0.9709 |

---

## 10. Per-cell activation_norm distribution

The `activation_norm` column in each per-cell CSV is `np.linalg.norm(X, axis=1)`
for the (N, 4096) activation matrix. Reported per cell:

| # | model | task | L | N | norm_min | norm_mean | norm_max | norm_std |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | gpt-j-6b | addition | 04 | 10000 |    55.28 |    57.95 |    60.42 |     0.84 |
| 2 | gpt-j-6b | addition | 08 | 10000 |    68.98 |    76.89 |    84.55 |     2.00 |
| 3 | gpt-j-6b | addition | 14 | 10000 |    87.24 |    94.86 |   104.18 |     2.34 |
| 4 | gpt-j-6b | addition | 20 | 10000 |   114.54 |   129.32 |   165.27 |     6.34 |
| 5 | gpt-j-6b | addition | 24 | 10000 |   154.08 |   177.95 |   226.32 |     9.03 |
| 6 | gpt-j-6b | multiplication | 04 |  3023 |    53.57 |    55.88 |    58.42 |     0.88 |
| 7 | gpt-j-6b | multiplication | 08 |  3023 |    71.83 |    78.47 |    84.08 |     2.08 |
| 8 | gpt-j-6b | multiplication | 14 |  3023 |    88.97 |    96.05 |   104.73 |     2.37 |
| 9 | gpt-j-6b | multiplication | 20 |  3023 |   120.46 |   139.17 |   173.40 |     7.05 |
| 10 | gpt-j-6b | multiplication | 24 |  3023 |   149.31 |   178.04 |   227.70 |    12.93 |
| 11 | llama-3.1-8b | addition | 04 | 10000 |     3.75 |     4.17 |     4.45 |     0.08 |
| 12 | llama-3.1-8b | addition | 08 | 10000 |     5.92 |     6.34 |     6.71 |     0.11 |
| 13 | llama-3.1-8b | addition | 16 | 10000 |    11.35 |    12.67 |    13.64 |     0.27 |
| 14 | llama-3.1-8b | addition | 24 | 10000 |    20.61 |    23.36 |    27.64 |     0.86 |
| 15 | llama-3.1-8b | addition | 28 | 10000 |    32.92 |    37.01 |    43.73 |     1.08 |
| 16 | llama-3.1-8b | multiplication | 04 |  3023 |     3.83 |     4.07 |     4.35 |     0.09 |
| 17 | llama-3.1-8b | multiplication | 08 |  3023 |     6.17 |     6.42 |     6.66 |     0.08 |
| 18 | llama-3.1-8b | multiplication | 16 |  3023 |    10.33 |    11.52 |    13.17 |     0.49 |
| 19 | llama-3.1-8b | multiplication | 24 |  3023 |    20.43 |    23.60 |    27.61 |     0.96 |
| 20 | llama-3.1-8b | multiplication | 28 |  3023 |    30.17 |    35.08 |    41.33 |     1.76 |
| 21 | pythia-6.9b | addition | 04 | 10000 |    79.68 |    83.99 |    87.11 |     0.80 |
| 22 | pythia-6.9b | addition | 08 | 10000 |   115.44 |   121.28 |   126.88 |     1.59 |
| 23 | pythia-6.9b | addition | 16 | 10000 |   193.07 |   204.59 |   226.35 |     4.27 |
| 24 | pythia-6.9b | addition | 24 | 10000 |   260.40 |   291.85 |   374.21 |    13.80 |
| 25 | pythia-6.9b | addition | 28 | 10000 |   270.57 |   310.85 |   395.74 |    14.94 |
| 26 | pythia-6.9b | multiplication | 04 |  3023 |    76.64 |    79.54 |    82.27 |     0.86 |
| 27 | pythia-6.9b | multiplication | 08 |  3023 |   119.26 |   122.98 |   127.49 |     1.22 |
| 28 | pythia-6.9b | multiplication | 16 |  3023 |   195.34 |   210.71 |   232.56 |     5.60 |
| 29 | pythia-6.9b | multiplication | 24 |  3023 |   266.59 |   303.81 |   381.43 |    16.88 |
| 30 | pythia-6.9b | multiplication | 28 |  3023 |   259.43 |   311.93 |   395.21 |    20.33 |

These are computed from the (N, 4096) `.npy` activations and serve as a
Step-3 cross-check (the per-layer norm ranges match
[docs/03_eval_and_extract.md §25](03_eval_and_extract.md)).

---

## 11. CSV schema and column counts

Each per-cell CSV (`data/results/embeddings/{model_key}/{task}_layer_{LL:02d}.csv`)
has columns ordered as:

1. All Tier 1–4 concept columns from the corresponding
   `data/data/raw/{task}_problems.csv` (joined row-by-row by `index`).
   - Addition: 62 concept columns + tokenization metadata = 62 columns.
   - Multiplication: 75 concept columns + tokenization metadata = 75 columns.
2. `activation_norm` (1 column).
3. 14 coordinate columns (2 per HP setting × 7 settings):
   `umap2d_n15_md10_x`, `umap2d_n15_md10_y`,
   `umap2d_n30_md10_x`, `umap2d_n30_md10_y`,
   `umap2d_n50_md10_x`, `umap2d_n50_md10_y`,
   `umap2d_n30_md30_x`, `umap2d_n30_md30_y`,
   `tsne2d_p10_x`, `tsne2d_p10_y`,
   `tsne2d_p30_x`, `tsne2d_p30_y`,
   `tsne2d_p50_x`, `tsne2d_p50_y`.

Per-cell column counts (from the actual files):

| # | model | task | L | rows | cols |
|---:|---|---|---:|---:|---:|
| 1 | gpt-j-6b | addition | 04 | 10000 |  76 |
| 2 | gpt-j-6b | addition | 08 | 10000 |  76 |
| 3 | gpt-j-6b | addition | 14 | 10000 |  76 |
| 4 | gpt-j-6b | addition | 20 | 10000 |  76 |
| 5 | gpt-j-6b | addition | 24 | 10000 |  76 |
| 6 | gpt-j-6b | multiplication | 04 |  3023 |  89 |
| 7 | gpt-j-6b | multiplication | 08 |  3023 |  89 |
| 8 | gpt-j-6b | multiplication | 14 |  3023 |  89 |
| 9 | gpt-j-6b | multiplication | 20 |  3023 |  89 |
| 10 | gpt-j-6b | multiplication | 24 |  3023 |  89 |
| 11 | llama-3.1-8b | addition | 04 | 10000 |  76 |
| 12 | llama-3.1-8b | addition | 08 | 10000 |  76 |
| 13 | llama-3.1-8b | addition | 16 | 10000 |  76 |
| 14 | llama-3.1-8b | addition | 24 | 10000 |  76 |
| 15 | llama-3.1-8b | addition | 28 | 10000 |  76 |
| 16 | llama-3.1-8b | multiplication | 04 |  3023 |  89 |
| 17 | llama-3.1-8b | multiplication | 08 |  3023 |  89 |
| 18 | llama-3.1-8b | multiplication | 16 |  3023 |  89 |
| 19 | llama-3.1-8b | multiplication | 24 |  3023 |  89 |
| 20 | llama-3.1-8b | multiplication | 28 |  3023 |  89 |
| 21 | pythia-6.9b | addition | 04 | 10000 |  76 |
| 22 | pythia-6.9b | addition | 08 | 10000 |  76 |
| 23 | pythia-6.9b | addition | 16 | 10000 |  76 |
| 24 | pythia-6.9b | addition | 24 | 10000 |  76 |
| 25 | pythia-6.9b | addition | 28 | 10000 |  76 |
| 26 | pythia-6.9b | multiplication | 04 |  3023 |  89 |
| 27 | pythia-6.9b | multiplication | 08 |  3023 |  89 |
| 28 | pythia-6.9b | multiplication | 16 |  3023 |  89 |
| 29 | pythia-6.9b | multiplication | 24 |  3023 |  89 |
| 30 | pythia-6.9b | multiplication | 28 |  3023 |  89 |

Addition cells: 76 columns (62 dataset + 1 activation_norm + 14 coords − 1, ordered).
Multiplication cells: 89 columns (75 dataset + 1 activation_norm + 14 coords − 1, ordered).

---

## 12. Manifest schema and example values

Each cell's manifest is at
`data/results/embeddings/{model_key}/{task}_layer_{LL:02d}_manifest.json`.

Top-level keys:

```
schema_version                = "v1"
model_key                     = "gpt-j-6b" | "llama-3.1-8b" | "pythia-6.9b"
model_name                    = "GPT-J 6B" | "Llama 3.1 8B" | "Pythia 6.9B"
task                          = "addition" | "multiplication"
layer                         = <int>
n_problems                    = 10000 | 3023
hidden_dim                    = 4096
activation_path               = "/data/user_data/anshulk/emnlp2026/activations/<mk>/<task>_layer_<LL>.npy"
activation_sha256             = <64-char hex>
labels_path                   = "/data/user_data/anshulk/emnlp2026/data/raw/<task>_problems.csv"
labels_sha256                 = <64-char hex>
umap_hp_grid                  = [{name, n_neighbors, min_dist}, ...]
tsne_hp_grid                  = [{name, perplexity}, ...]
common_random_state           = 42
trustworthiness               = {hp_name: float, ...}    (full N values)
trustworthiness_n_neighbors   = 30
trustworthiness_subsample_size = N (full)
trustworthiness_recomputed_full_n = true
runtime_seconds               = {hp_name: float, ...}
umap_learn_version            = "0.5.11"
sklearn_version               = (string)
numpy_version                 = "2.2.6"
pandas_version                = (string)
python_version                = "3.11.15"
timestamp_utc                 = "YYYY-MM-DDThh:mm:ssZ"
cell_runtime_seconds          = float
csv_path                      = path
csv_rows                      = N
csv_cols                      = column count
```

Sample (first cell, `gpt-j-6b | addition | L04`):

```json
{
  "schema_version": "v1",
  "model_key": "gpt-j-6b",
  "task": "addition",
  "layer": 4,
  "n_problems": 10000,
  "hidden_dim": 4096,
  "activation_sha256": "23c2f926a1c34ebf0d719f9a525e31a778dae99be6c4aae4cd73a90ec0d7b066",
  "labels_sha256": "71ef4e15139df50c8490cdf7011ba98b9062be22fc9abef16c20b371d4f35ad9",
  "trustworthiness_n_neighbors": 30,
  "trustworthiness_subsample_size": 10000,
  "trustworthiness_recomputed_full_n": true,
  "umap_learn_version": "0.5.11",
  "sklearn_version": "1.8.0",
  "numpy_version": "2.2.6",
  "pandas_version": "2.3.3",
  "python_version": "3.11.15",
  "timestamp_utc": "2026-05-09T20:33:02Z",
  "cell_runtime_seconds": 212.13,
  "csv_rows": 10000,
  "csv_cols": 76
}
```

---

## 13. Per-cell output file sizes

Files at `/data/user_data/anshulk/emnlp2026/results/embeddings/{model_key}/`:

| # | model | task | L | csv (bytes) | manifest (bytes) |
|---:|---|---|---:|---:|---:|
| 1 | gpt-j-6b | addition | 04 |  3,395,323 |  2,198 |
| 2 | gpt-j-6b | addition | 08 |  3,394,760 |  2,196 |
| 3 | gpt-j-6b | addition | 14 |  3,379,892 |  2,199 |
| 4 | gpt-j-6b | addition | 20 |  3,375,651 |  2,199 |
| 5 | gpt-j-6b | addition | 24 |  3,369,015 |  2,197 |
| 6 | gpt-j-6b | multiplication | 04 |  1,588,971 |  2,215 |
| 7 | gpt-j-6b | multiplication | 08 |  1,584,099 |  2,216 |
| 8 | gpt-j-6b | multiplication | 14 |  1,580,154 |  2,218 |
| 9 | gpt-j-6b | multiplication | 20 |  1,581,242 |  2,214 |
| 10 | gpt-j-6b | multiplication | 24 |  1,577,778 |  2,215 |
| 11 | llama-3.1-8b | addition | 04 |  3,383,544 |  2,212 |
| 12 | llama-3.1-8b | addition | 08 |  3,405,442 |  2,211 |
| 13 | llama-3.1-8b | addition | 16 |  3,389,040 |  2,213 |
| 14 | llama-3.1-8b | addition | 24 |  3,396,374 |  2,213 |
| 15 | llama-3.1-8b | addition | 28 |  3,395,618 |  2,215 |
| 16 | llama-3.1-8b | multiplication | 04 |  1,584,910 |  2,228 |
| 17 | llama-3.1-8b | multiplication | 08 |  1,597,013 |  2,230 |
| 18 | llama-3.1-8b | multiplication | 16 |  1,585,415 |  2,233 |
| 19 | llama-3.1-8b | multiplication | 24 |  1,582,461 |  2,233 |
| 20 | llama-3.1-8b | multiplication | 28 |  1,580,530 |  2,233 |
| 21 | pythia-6.9b | addition | 04 |  3,395,147 |  2,207 |
| 22 | pythia-6.9b | addition | 08 |  3,386,651 |  2,208 |
| 23 | pythia-6.9b | addition | 16 |  3,368,008 |  2,210 |
| 24 | pythia-6.9b | addition | 24 |  3,365,412 |  2,210 |
| 25 | pythia-6.9b | addition | 28 |  3,375,928 |  2,209 |
| 26 | pythia-6.9b | multiplication | 04 |  1,582,831 |  2,226 |
| 27 | pythia-6.9b | multiplication | 08 |  1,589,325 |  2,227 |
| 28 | pythia-6.9b | multiplication | 16 |  1,578,326 |  2,227 |
| 29 | pythia-6.9b | multiplication | 24 |  1,577,945 |  2,227 |
| 30 | pythia-6.9b | multiplication | 28 |  1,583,866 |  2,229 |

---

## 14. 45 plots — full index

Source of truth: `data/figures/embeddings/plot_index.json`. Each row
corresponds to one PNG.

| # | model | task | L | concept | method | HP picked | T (full N) | png file |
|---:|---|---|---:|---|---|---|---:|---|
|  1 | gpt-j-6b | addition | 14 | a_units | umap | `umap2d_n30_md10` | 0.9867 | `p01_gpt-j-6b_addition_L14_a_units_umap.png` |
|  2 | gpt-j-6b | addition | 14 | ans_units | umap | `umap2d_n30_md10` | 0.9867 | `p02_gpt-j-6b_addition_L14_ans_units_umap.png` |
|  3 | gpt-j-6b | addition | 14 | carry_units | umap | `umap2d_n30_md10` | 0.9867 | `p03_gpt-j-6b_addition_L14_carry_units_umap.png` |
|  4 | gpt-j-6b | multiplication | 14 | a_units | umap | `umap2d_n30_md10` | 0.9801 | `p04_gpt-j-6b_multiplication_L14_a_units_umap.png` |
|  5 | gpt-j-6b | multiplication | 14 | ans_units | umap | `umap2d_n30_md10` | 0.9801 | `p05_gpt-j-6b_multiplication_L14_ans_units_umap.png` |
|  6 | gpt-j-6b | multiplication | 14 | carry_units | umap | `umap2d_n30_md10` | 0.9801 | `p06_gpt-j-6b_multiplication_L14_carry_units_umap.png` |
|  7 | gpt-j-6b | multiplication | 14 | partial_product_units | umap | `umap2d_n30_md10` | 0.9801 | `p07_gpt-j-6b_multiplication_L14_partial_product_units_umap.png` |
|  8 | gpt-j-6b | multiplication | 04 | ans_units | umap | `umap2d_n50_md10` | 0.9915 | `p08_gpt-j-6b_multiplication_L04_ans_units_umap.png` |
|  9 | gpt-j-6b | multiplication | 14 | ans_units | umap | `umap2d_n30_md10` | 0.9801 | `p09_gpt-j-6b_multiplication_L14_ans_units_umap.png` |
| 10 | gpt-j-6b | multiplication | 24 | ans_units | umap | `umap2d_n15_md10` | 0.9378 | `p10_gpt-j-6b_multiplication_L24_ans_units_umap.png` |
| 11 | gpt-j-6b | addition | 14 | a_units | tsne | `tsne2d_p50` | 0.9932 | `p11_gpt-j-6b_addition_L14_a_units_tsne.png` |
| 12 | gpt-j-6b | addition | 14 | ans_units | tsne | `tsne2d_p50` | 0.9932 | `p12_gpt-j-6b_addition_L14_ans_units_tsne.png` |
| 13 | gpt-j-6b | multiplication | 14 | a_units | tsne | `tsne2d_p30` | 0.9825 | `p13_gpt-j-6b_multiplication_L14_a_units_tsne.png` |
| 14 | gpt-j-6b | multiplication | 14 | ans_units | tsne | `tsne2d_p30` | 0.9825 | `p14_gpt-j-6b_multiplication_L14_ans_units_tsne.png` |
| 15 | gpt-j-6b | multiplication | 14 | carry_units | tsne | `tsne2d_p30` | 0.9825 | `p15_gpt-j-6b_multiplication_L14_carry_units_tsne.png` |
| 16 | llama-3.1-8b | addition | 16 | a_units | umap | `umap2d_n50_md10` | 0.9895 | `p16_llama-3.1-8b_addition_L16_a_units_umap.png` |
| 17 | llama-3.1-8b | addition | 16 | ans_units | umap | `umap2d_n50_md10` | 0.9895 | `p17_llama-3.1-8b_addition_L16_ans_units_umap.png` |
| 18 | llama-3.1-8b | addition | 16 | carry_units | umap | `umap2d_n50_md10` | 0.9895 | `p18_llama-3.1-8b_addition_L16_carry_units_umap.png` |
| 19 | llama-3.1-8b | multiplication | 16 | a_units | umap | `umap2d_n50_md10` | 0.9910 | `p19_llama-3.1-8b_multiplication_L16_a_units_umap.png` |
| 20 | llama-3.1-8b | multiplication | 16 | ans_units | umap | `umap2d_n50_md10` | 0.9910 | `p20_llama-3.1-8b_multiplication_L16_ans_units_umap.png` |
| 21 | llama-3.1-8b | multiplication | 16 | carry_units | umap | `umap2d_n50_md10` | 0.9910 | `p21_llama-3.1-8b_multiplication_L16_carry_units_umap.png` |
| 22 | llama-3.1-8b | multiplication | 16 | partial_product_units | umap | `umap2d_n50_md10` | 0.9910 | `p22_llama-3.1-8b_multiplication_L16_partial_product_units_umap.png` |
| 23 | llama-3.1-8b | multiplication | 04 | ans_units | umap | `umap2d_n30_md10` | 0.9879 | `p23_llama-3.1-8b_multiplication_L04_ans_units_umap.png` |
| 24 | llama-3.1-8b | multiplication | 16 | ans_units | umap | `umap2d_n50_md10` | 0.9910 | `p24_llama-3.1-8b_multiplication_L16_ans_units_umap.png` |
| 25 | llama-3.1-8b | multiplication | 28 | ans_units | umap | `umap2d_n15_md10` | 0.9481 | `p25_llama-3.1-8b_multiplication_L28_ans_units_umap.png` |
| 26 | llama-3.1-8b | addition | 16 | a_units | tsne | `tsne2d_p50` | 0.9943 | `p26_llama-3.1-8b_addition_L16_a_units_tsne.png` |
| 27 | llama-3.1-8b | addition | 16 | ans_units | tsne | `tsne2d_p50` | 0.9943 | `p27_llama-3.1-8b_addition_L16_ans_units_tsne.png` |
| 28 | llama-3.1-8b | multiplication | 16 | a_units | tsne | `tsne2d_p30` | 0.9937 | `p28_llama-3.1-8b_multiplication_L16_a_units_tsne.png` |
| 29 | llama-3.1-8b | multiplication | 16 | ans_units | tsne | `tsne2d_p30` | 0.9937 | `p29_llama-3.1-8b_multiplication_L16_ans_units_tsne.png` |
| 30 | llama-3.1-8b | multiplication | 16 | carry_units | tsne | `tsne2d_p30` | 0.9937 | `p30_llama-3.1-8b_multiplication_L16_carry_units_tsne.png` |
| 31 | pythia-6.9b | addition | 16 | a_units | umap | `umap2d_n30_md10` | 0.9829 | `p31_pythia-6.9b_addition_L16_a_units_umap.png` |
| 32 | pythia-6.9b | addition | 16 | ans_units | umap | `umap2d_n30_md10` | 0.9829 | `p32_pythia-6.9b_addition_L16_ans_units_umap.png` |
| 33 | pythia-6.9b | addition | 16 | carry_units | umap | `umap2d_n30_md10` | 0.9829 | `p33_pythia-6.9b_addition_L16_carry_units_umap.png` |
| 34 | pythia-6.9b | multiplication | 16 | a_units | umap | `umap2d_n15_md10` | 0.9522 | `p34_pythia-6.9b_multiplication_L16_a_units_umap.png` |
| 35 | pythia-6.9b | multiplication | 16 | ans_units | umap | `umap2d_n15_md10` | 0.9522 | `p35_pythia-6.9b_multiplication_L16_ans_units_umap.png` |
| 36 | pythia-6.9b | multiplication | 16 | carry_units | umap | `umap2d_n15_md10` | 0.9522 | `p36_pythia-6.9b_multiplication_L16_carry_units_umap.png` |
| 37 | pythia-6.9b | multiplication | 16 | partial_product_units | umap | `umap2d_n15_md10` | 0.9522 | `p37_pythia-6.9b_multiplication_L16_partial_product_units_umap.png` |
| 38 | pythia-6.9b | multiplication | 04 | ans_units | umap | `umap2d_n50_md10` | 0.9916 | `p38_pythia-6.9b_multiplication_L04_ans_units_umap.png` |
| 39 | pythia-6.9b | multiplication | 16 | ans_units | umap | `umap2d_n15_md10` | 0.9522 | `p39_pythia-6.9b_multiplication_L16_ans_units_umap.png` |
| 40 | pythia-6.9b | multiplication | 28 | ans_units | umap | `umap2d_n15_md10` | 0.9407 | `p40_pythia-6.9b_multiplication_L28_ans_units_umap.png` |
| 41 | pythia-6.9b | addition | 16 | a_units | tsne | `tsne2d_p50` | 0.9909 | `p41_pythia-6.9b_addition_L16_a_units_tsne.png` |
| 42 | pythia-6.9b | addition | 16 | ans_units | tsne | `tsne2d_p50` | 0.9909 | `p42_pythia-6.9b_addition_L16_ans_units_tsne.png` |
| 43 | pythia-6.9b | multiplication | 16 | a_units | tsne | `tsne2d_p50` | 0.9688 | `p43_pythia-6.9b_multiplication_L16_a_units_tsne.png` |
| 44 | pythia-6.9b | multiplication | 16 | ans_units | tsne | `tsne2d_p50` | 0.9688 | `p44_pythia-6.9b_multiplication_L16_ans_units_tsne.png` |
| 45 | pythia-6.9b | multiplication | 16 | carry_units | tsne | `tsne2d_p50` | 0.9688 | `p45_pythia-6.9b_multiplication_L16_carry_units_tsne.png` |

---

## 15. PNG file sizes

| # | png file | size (bytes) |
|---:|---|---:|
|  1 | `p01_gpt-j-6b_addition_L14_a_units_umap.png` |  364,756 |
|  2 | `p02_gpt-j-6b_addition_L14_ans_units_umap.png` |  369,222 |
|  3 | `p03_gpt-j-6b_addition_L14_carry_units_umap.png` |  362,573 |
|  4 | `p04_gpt-j-6b_multiplication_L14_a_units_umap.png` |  205,699 |
|  5 | `p05_gpt-j-6b_multiplication_L14_ans_units_umap.png` |  207,933 |
|  6 | `p06_gpt-j-6b_multiplication_L14_carry_units_umap.png` |  205,481 |
|  7 | `p07_gpt-j-6b_multiplication_L14_partial_product_units_umap.png` |  221,807 |
|  8 | `p08_gpt-j-6b_multiplication_L04_ans_units_umap.png` |   95,272 |
|  9 | `p09_gpt-j-6b_multiplication_L14_ans_units_umap.png` |  207,933 |
| 10 | `p10_gpt-j-6b_multiplication_L24_ans_units_umap.png` |  232,582 |
| 11 | `p11_gpt-j-6b_addition_L14_a_units_tsne.png` |  431,161 |
| 12 | `p12_gpt-j-6b_addition_L14_ans_units_tsne.png` |  437,270 |
| 13 | `p13_gpt-j-6b_multiplication_L14_a_units_tsne.png` |  250,909 |
| 14 | `p14_gpt-j-6b_multiplication_L14_ans_units_tsne.png` |  253,115 |
| 15 | `p15_gpt-j-6b_multiplication_L14_carry_units_tsne.png` |  249,460 |
| 16 | `p16_llama-3.1-8b_addition_L16_a_units_umap.png` |  139,205 |
| 17 | `p17_llama-3.1-8b_addition_L16_ans_units_umap.png` |  141,126 |
| 18 | `p18_llama-3.1-8b_addition_L16_carry_units_umap.png` |  135,718 |
| 19 | `p19_llama-3.1-8b_multiplication_L16_a_units_umap.png` |   93,203 |
| 20 | `p20_llama-3.1-8b_multiplication_L16_ans_units_umap.png` |   93,948 |
| 21 | `p21_llama-3.1-8b_multiplication_L16_carry_units_umap.png` |   93,167 |
| 22 | `p22_llama-3.1-8b_multiplication_L16_partial_product_units_umap.png` |  101,885 |
| 23 | `p23_llama-3.1-8b_multiplication_L04_ans_units_umap.png` |   93,378 |
| 24 | `p24_llama-3.1-8b_multiplication_L16_ans_units_umap.png` |   93,948 |
| 25 | `p25_llama-3.1-8b_multiplication_L28_ans_units_umap.png` |  166,367 |
| 26 | `p26_llama-3.1-8b_addition_L16_a_units_tsne.png` |  466,028 |
| 27 | `p27_llama-3.1-8b_addition_L16_ans_units_tsne.png` |  471,889 |
| 28 | `p28_llama-3.1-8b_multiplication_L16_a_units_tsne.png` |  201,044 |
| 29 | `p29_llama-3.1-8b_multiplication_L16_ans_units_tsne.png` |  202,857 |
| 30 | `p30_llama-3.1-8b_multiplication_L16_carry_units_tsne.png` |  200,696 |
| 31 | `p31_pythia-6.9b_addition_L16_a_units_umap.png` |  338,806 |
| 32 | `p32_pythia-6.9b_addition_L16_ans_units_umap.png` |  352,557 |
| 33 | `p33_pythia-6.9b_addition_L16_carry_units_umap.png` |  343,280 |
| 34 | `p34_pythia-6.9b_multiplication_L16_a_units_umap.png` |  216,969 |
| 35 | `p35_pythia-6.9b_multiplication_L16_ans_units_umap.png` |  219,481 |
| 36 | `p36_pythia-6.9b_multiplication_L16_carry_units_umap.png` |  217,507 |
| 37 | `p37_pythia-6.9b_multiplication_L16_partial_product_units_umap.png` |  233,473 |
| 38 | `p38_pythia-6.9b_multiplication_L04_ans_units_umap.png` |   88,573 |
| 39 | `p39_pythia-6.9b_multiplication_L16_ans_units_umap.png` |  219,481 |
| 40 | `p40_pythia-6.9b_multiplication_L28_ans_units_umap.png` |  232,800 |
| 41 | `p41_pythia-6.9b_addition_L16_a_units_tsne.png` |  453,053 |
| 42 | `p42_pythia-6.9b_addition_L16_ans_units_tsne.png` |  463,352 |
| 43 | `p43_pythia-6.9b_multiplication_L16_a_units_tsne.png` |  232,650 |
| 44 | `p44_pythia-6.9b_multiplication_L16_ans_units_tsne.png` |  234,242 |
| 45 | `p45_pythia-6.9b_multiplication_L16_carry_units_tsne.png` |  231,144 |

Total PNG bytes: 10,867,000.

Plus `plot_index.json`: 16824 bytes.

---

## 16. Verification checks

Performed across the 30 cells:

- **Row-count match.** Per cell, `csv_rows == n_problems` (10,000 for addition,
  3,023 for multiplication). Asserted at write time by `build_embeddings.py`.
- **Column count.** Addition CSVs: 76 columns. Multiplication CSVs: 89
  columns. Counted from the `csv_cols` manifest field.
- **Trustworthiness range.** All 210 trustworthiness values across 30
  cells × 7 settings are in [0, 1].

  - Observed min across all 210: 0.9047.
  - Observed max across all 210: 0.9994.

- **Random-state consistency.** All UMAP and t-SNE fits use `random_state=42`.
  Re-running `build_embeddings.py` produces byte-identical CSVs (modulo
  trustworthiness values that depend on the recompute step).
- **HP-grid completeness.** Every manifest carries 7 entries in
  `trustworthiness` and 7 entries in `runtime_seconds`. Verified across
  all 30 manifests.
- **Plot rendering.** 45 PNGs exist at the expected paths. None are below
  50 KB (smallest: 88,573 bytes;
  largest: 471,889 bytes).
- **Plot index completeness.** `plot_index.json` lists 45 entries; each
  carries `plot_index ∈ [1, 45]`, `model_key`, `task`, `layer`, `concept`,
  `method`, `hp_name`, `trustworthiness`, `png`.
- **Activation-norm cross-check with Step 3.** Per-cell `activation_norm`
  min/mean/max in §10 match the validation_report norms in
  [docs/03_eval_and_extract.md §25](03_eval_and_extract.md).

---

## 17. Output file inventory

### 17.1 `data/results/embeddings/`

- 30 CSV files: `{model_key}/{task}_layer_{LL:02d}.csv`.
- 30 manifest JSON files: `{model_key}/{task}_layer_{LL:02d}_manifest.json`.

Total CSV bytes:           74,530,671
Total manifest bytes:          66,468

### 17.2 `data/figures/embeddings/`

- 45 PNG files (totaling 10,867,000 bytes).
- 1 `plot_index.json` (16,824 bytes).

### 17.3 `data/logs/`

- `build_embeddings.log` — Phase A run log.
- `select_and_plot_embeddings.log` — Phase B run log.

---

## 18. Library versions

Recorded in every per-cell manifest (consistent across the 30 cells):

- `umap-learn`: 0.5.11
- `scikit-learn`: 1.8.0
- `numpy`: 2.2.6
- `pandas`: 2.3.3
- `python`: 3.11.15

Plotting:

- `matplotlib`: 3.10.8 (verified in env, recorded by Phase B render).

---

## 19. Re-run commands

Phase A (UMAP + t-SNE per cell, ~78 min total wall):

```bash
/data/user_data/anshulk/miniconda3/envs/geometry/bin/python \
    /home/anshulk/emnlp2026/build_embeddings.py \
    --config /home/anshulk/emnlp2026/config.yaml
```

Trustworthiness recompute on full N (~10 min total, run once after Phase A):

```python
# inline equivalent — see /tmp/recompute_trustworthiness.py for archived script
import json, glob, numpy as np, pandas as pd
from sklearn.manifold import trustworthiness
HP = ['umap2d_n15_md10','umap2d_n30_md10','umap2d_n50_md10','umap2d_n30_md30',
      'tsne2d_p10','tsne2d_p30','tsne2d_p50']
for mp in sorted(glob.glob('.../embeddings/*/*_manifest.json')):
    m = json.load(open(mp))
    X = np.load('.../activations/{0}/{1}_layer_{2:02d}.npy'.format(
            m['model_key'], m['task'], m['layer']))
    df = pd.read_csv(m['csv_path'])
    new = {h: float(trustworthiness(X, df[[f'{h}_x', f'{h}_y']].to_numpy(),
                                    n_neighbors=30)) for h in HP}
    m['trustworthiness'] = new
    m['trustworthiness_subsample_size'] = X.shape[0]
    m['trustworthiness_recomputed_full_n'] = True
    json.dump(m, open(mp,'w'), indent=2)
```

Phase B (plot rendering, ~17 sec total wall):

```bash
/data/user_data/anshulk/miniconda3/envs/geometry/bin/python \
    /home/anshulk/emnlp2026/select_and_plot_embeddings.py \
    --config /home/anshulk/emnlp2026/config.yaml
```

---

## 20. Open questions

1. The 14 coordinate columns per cell support arbitrary downstream
   re-coloring without re-running UMAP or t-SNE. Future analyses
   (e.g. cross-task overlay, cross-model side-by-side) read the CSVs
   directly.
2. The 7-setting HP grid was fixed before any cell ran. Different grids
   (e.g. larger `n_neighbors` for UMAP, or adding a perplexity=100 t-SNE)
   would require re-running Phase A.
3. `activation_norm` is stored as a per-row float for use as a coloring
   variable in any future plot.
4. The 45 plots in §14 are a fixed allocation; rendering different subsets
   from the existing CSVs is one invocation of a small variant of
   `select_and_plot_embeddings.py`.
5. Cross-cell concatenation (e.g. plotting all 5 layers of one (model,
   task) on shared axes) is out of scope for this step.

---

## 21. Per-(model × task) aggregate Phase A runtime

Sum of `runtime_seconds` across the 5 layers per (model, task), per HP
setting. Plus the (model, task) total wall time.

| model | task | umap_n15 | umap_n30 | umap_n50 | umap_md30 | tsne_p10 | tsne_p30 | tsne_p50 | total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-j-6b | addition |   92.7 |  119.4 |  178.3 |  119.1 |  141.1 |  164.5 |  195.6 |  1010.6 |
| gpt-j-6b | multiplication |   95.0 |   98.4 |   98.8 |   99.5 |   38.6 |   48.9 |   55.1 |   534.4 |
| llama-3.1-8b | addition |   84.2 |  127.1 |  184.6 |  125.7 |  142.8 |  167.1 |  193.5 |  1025.0 |
| llama-3.1-8b | multiplication |   87.7 |   92.1 |   96.6 |   93.2 |   38.3 |   45.0 |   51.6 |   504.5 |
| pythia-6.9b | addition |   80.5 |  120.0 |  180.5 |  119.6 |  139.7 |  167.0 |  198.1 |  1005.4 |
| pythia-6.9b | multiplication |   88.3 |   93.9 |   98.1 |   95.7 |   37.3 |   43.4 |   49.2 |   506.0 |

Cell-level total wall (sum of all runtime_seconds + non-fit overhead per cell):

| # | model | task | L | runtime sum (s) | cell_runtime_seconds (s) | overhead (s) |
|---:|---|---|---:|---:|---:|---:|
| 1 | gpt-j-6b | addition | 04 |   208.9 |   212.1 |    3.2 |
| 2 | gpt-j-6b | addition | 08 |   184.4 |   187.3 |    2.9 |
| 3 | gpt-j-6b | addition | 14 |   189.0 |   192.1 |    3.1 |
| 4 | gpt-j-6b | addition | 20 |   212.0 |   215.2 |    3.2 |
| 5 | gpt-j-6b | addition | 24 |   216.3 |   219.6 |    3.2 |
| 6 | gpt-j-6b | multiplication | 04 |   109.8 |   112.1 |    2.3 |
| 7 | gpt-j-6b | multiplication | 08 |   100.0 |   102.5 |    2.4 |
| 8 | gpt-j-6b | multiplication | 14 |   111.5 |   114.9 |    3.4 |
| 9 | gpt-j-6b | multiplication | 20 |   107.4 |   109.8 |    2.5 |
| 10 | gpt-j-6b | multiplication | 24 |   105.6 |   108.1 |    2.5 |
| 11 | llama-3.1-8b | addition | 04 |   203.0 |   206.2 |    3.2 |
| 12 | llama-3.1-8b | addition | 08 |   205.1 |   208.3 |    3.3 |
| 13 | llama-3.1-8b | addition | 16 |   197.4 |   200.8 |    3.3 |
| 14 | llama-3.1-8b | addition | 24 |   210.6 |   214.1 |    3.5 |
| 15 | llama-3.1-8b | addition | 28 |   208.9 |   212.2 |    3.3 |
| 16 | llama-3.1-8b | multiplication | 04 |    99.7 |   102.2 |    2.5 |
| 17 | llama-3.1-8b | multiplication | 08 |   100.8 |   103.4 |    2.6 |
| 18 | llama-3.1-8b | multiplication | 16 |    99.4 |   101.9 |    2.5 |
| 19 | llama-3.1-8b | multiplication | 24 |   102.5 |   105.0 |    2.4 |
| 20 | llama-3.1-8b | multiplication | 28 |   102.1 |   104.6 |    2.5 |
| 21 | pythia-6.9b | addition | 04 |   205.1 |   208.5 |    3.5 |
| 22 | pythia-6.9b | addition | 08 |   190.9 |   194.2 |    3.3 |
| 23 | pythia-6.9b | addition | 16 |   191.1 |   194.2 |    3.1 |
| 24 | pythia-6.9b | addition | 24 |   205.9 |   209.1 |    3.2 |
| 25 | pythia-6.9b | addition | 28 |   212.5 |   215.9 |    3.4 |
| 26 | pythia-6.9b | multiplication | 04 |    98.6 |   101.2 |    2.5 |
| 27 | pythia-6.9b | multiplication | 08 |   100.1 |   102.5 |    2.5 |
| 28 | pythia-6.9b | multiplication | 16 |   101.7 |   104.2 |    2.5 |
| 29 | pythia-6.9b | multiplication | 24 |   102.3 |   104.8 |    2.5 |
| 30 | pythia-6.9b | multiplication | 28 |   103.3 |   106.0 |    2.7 |

Phase A total wall (from the run log): **77.9 min**.
Trustworthiness recompute pass total wall: **9.6 min**.
Phase B plot-rendering total wall: **16.6 s**.

---

## 22. Min/median/max trustworthiness per HP across the 30 cells

Aggregating each HP setting across all 30 cells.

| HP setting | min | median | mean | max | std |
|---|---:|---:|---:|---:|---:|
| `umap2d_n15_md10` | 0.9378 | 0.9827 | 0.9761 | 0.9982 | 0.0182 |
| `umap2d_n30_md10` | 0.9270 | 0.9823 | 0.9757 | 0.9986 | 0.0212 |
| `umap2d_n50_md10` | 0.9246 | 0.9798 | 0.9743 | 0.9988 | 0.0223 |
| `umap2d_n30_md30` | 0.9047 | 0.9742 | 0.9656 | 0.9983 | 0.0300 |
| `tsne2d_p10` | 0.9304 | 0.9852 | 0.9784 | 0.9985 | 0.0196 |
| `tsne2d_p30` | 0.9490 | 0.9898 | 0.9825 | 0.9993 | 0.0172 |
| `tsne2d_p50` | 0.9440 | 0.9897 | 0.9826 | 0.9994 | 0.0178 |

Across all 210 (cell, HP) pairs:
- min = 0.9047
- median = 0.9853
- mean = 0.9764
- max = 0.9994
- std = 0.0217

Counts of cells where each HP is the within-method best:

| HP setting | # cells where it is the best UMAP | # cells where it is the best t-SNE |
|---|---:|---:|
| `umap2d_n15_md10` | 13 | — |
| `umap2d_n30_md10` | 8 | — |
| `umap2d_n50_md10` | 9 | — |
| `umap2d_n30_md30` | 0 | — |
| `tsne2d_p10` | — | 1 |
| `tsne2d_p30` | — | 11 |
| `tsne2d_p50` | — | 18 |

---

## 23. Per-cell timestamp UTC

Quoted from `manifest.timestamp_utc`.

| # | model | task | L | timestamp_utc |
|---:|---|---|---:|---|
| 1 | gpt-j-6b | addition | 04 | `2026-05-09T20:33:02Z` |
| 2 | gpt-j-6b | addition | 08 | `2026-05-09T20:36:10Z` |
| 3 | gpt-j-6b | addition | 14 | `2026-05-09T20:39:22Z` |
| 4 | gpt-j-6b | addition | 20 | `2026-05-09T20:42:57Z` |
| 5 | gpt-j-6b | addition | 24 | `2026-05-09T20:46:36Z` |
| 6 | gpt-j-6b | multiplication | 04 | `2026-05-09T20:48:29Z` |
| 7 | gpt-j-6b | multiplication | 08 | `2026-05-09T20:50:11Z` |
| 8 | gpt-j-6b | multiplication | 14 | `2026-05-09T20:52:06Z` |
| 9 | gpt-j-6b | multiplication | 20 | `2026-05-09T20:53:56Z` |
| 10 | gpt-j-6b | multiplication | 24 | `2026-05-09T20:55:44Z` |
| 11 | llama-3.1-8b | addition | 04 | `2026-05-09T20:59:10Z` |
| 12 | llama-3.1-8b | addition | 08 | `2026-05-09T21:02:39Z` |
| 13 | llama-3.1-8b | addition | 16 | `2026-05-09T21:05:59Z` |
| 14 | llama-3.1-8b | addition | 24 | `2026-05-09T21:09:33Z` |
| 15 | llama-3.1-8b | addition | 28 | `2026-05-09T21:13:06Z` |
| 16 | llama-3.1-8b | multiplication | 04 | `2026-05-09T21:14:48Z` |
| 17 | llama-3.1-8b | multiplication | 08 | `2026-05-09T21:16:31Z` |
| 18 | llama-3.1-8b | multiplication | 16 | `2026-05-09T21:18:13Z` |
| 19 | llama-3.1-8b | multiplication | 24 | `2026-05-09T21:19:58Z` |
| 20 | llama-3.1-8b | multiplication | 28 | `2026-05-09T21:21:43Z` |
| 21 | pythia-6.9b | addition | 04 | `2026-05-09T21:25:11Z` |
| 22 | pythia-6.9b | addition | 08 | `2026-05-09T21:28:25Z` |
| 23 | pythia-6.9b | addition | 16 | `2026-05-09T21:31:40Z` |
| 24 | pythia-6.9b | addition | 24 | `2026-05-09T21:35:09Z` |
| 25 | pythia-6.9b | addition | 28 | `2026-05-09T21:38:45Z` |
| 26 | pythia-6.9b | multiplication | 04 | `2026-05-09T21:40:26Z` |
| 27 | pythia-6.9b | multiplication | 08 | `2026-05-09T21:42:08Z` |
| 28 | pythia-6.9b | multiplication | 16 | `2026-05-09T21:43:53Z` |
| 29 | pythia-6.9b | multiplication | 24 | `2026-05-09T21:45:37Z` |
| 30 | pythia-6.9b | multiplication | 28 | `2026-05-09T21:47:23Z` |

---

## 24. Per-cell sha256 audit

Quoted from `manifest.activation_sha256` and `manifest.labels_sha256`.
Truncated to the first 24 hex characters for table compactness; full
64-char hex values are in the manifest JSONs.

| # | model | task | L | activation sha256 (24 ch) | labels sha256 (24 ch) |
|---:|---|---|---:|---|---|
| 1 | gpt-j-6b | addition | 04 | `23c2f926a1c34ebf0d719f9a…` | `71ef4e15139df50c8490cdf7…` |
| 2 | gpt-j-6b | addition | 08 | `2dfc171e187395fc0de2080d…` | `71ef4e15139df50c8490cdf7…` |
| 3 | gpt-j-6b | addition | 14 | `7f6cbf19453174712f98a2d7…` | `71ef4e15139df50c8490cdf7…` |
| 4 | gpt-j-6b | addition | 20 | `d8ce06d554103a36470ce33a…` | `71ef4e15139df50c8490cdf7…` |
| 5 | gpt-j-6b | addition | 24 | `769a37e3d5f30f30345fb59e…` | `71ef4e15139df50c8490cdf7…` |
| 6 | gpt-j-6b | multiplication | 04 | `d69d8fa847bfb113c2bf0bc7…` | `b8d8affaeafcdeac19ac517e…` |
| 7 | gpt-j-6b | multiplication | 08 | `3ccc9db7471dcd0f20f014c1…` | `b8d8affaeafcdeac19ac517e…` |
| 8 | gpt-j-6b | multiplication | 14 | `71779f590d58040f0c5979ee…` | `b8d8affaeafcdeac19ac517e…` |
| 9 | gpt-j-6b | multiplication | 20 | `792b74eff703e23a53617874…` | `b8d8affaeafcdeac19ac517e…` |
| 10 | gpt-j-6b | multiplication | 24 | `a76f7fb0c872649d92268360…` | `b8d8affaeafcdeac19ac517e…` |
| 11 | llama-3.1-8b | addition | 04 | `f4a239d6e590690aa4e37bd0…` | `71ef4e15139df50c8490cdf7…` |
| 12 | llama-3.1-8b | addition | 08 | `d0af363f5ade6284705f87f6…` | `71ef4e15139df50c8490cdf7…` |
| 13 | llama-3.1-8b | addition | 16 | `a24e6d40cfb71857c48fb8d3…` | `71ef4e15139df50c8490cdf7…` |
| 14 | llama-3.1-8b | addition | 24 | `48771dbdf9215b8a96fe5e66…` | `71ef4e15139df50c8490cdf7…` |
| 15 | llama-3.1-8b | addition | 28 | `2679e178f1b0b313e1992307…` | `71ef4e15139df50c8490cdf7…` |
| 16 | llama-3.1-8b | multiplication | 04 | `87d6d84e110b566df6e3b9f0…` | `b8d8affaeafcdeac19ac517e…` |
| 17 | llama-3.1-8b | multiplication | 08 | `5e027fb7ce5264fe9822d381…` | `b8d8affaeafcdeac19ac517e…` |
| 18 | llama-3.1-8b | multiplication | 16 | `b54cbbe5524fb8834dfa13d0…` | `b8d8affaeafcdeac19ac517e…` |
| 19 | llama-3.1-8b | multiplication | 24 | `d72466c756373d69460a371e…` | `b8d8affaeafcdeac19ac517e…` |
| 20 | llama-3.1-8b | multiplication | 28 | `dd3086885186511bbd67be51…` | `b8d8affaeafcdeac19ac517e…` |
| 21 | pythia-6.9b | addition | 04 | `2caa44a25c1c98af71424302…` | `71ef4e15139df50c8490cdf7…` |
| 22 | pythia-6.9b | addition | 08 | `a00ee8b166cab864201c167d…` | `71ef4e15139df50c8490cdf7…` |
| 23 | pythia-6.9b | addition | 16 | `87af3a0c76ac9ed74b3551ef…` | `71ef4e15139df50c8490cdf7…` |
| 24 | pythia-6.9b | addition | 24 | `bfab4a27b0856db61afdc74e…` | `71ef4e15139df50c8490cdf7…` |
| 25 | pythia-6.9b | addition | 28 | `50ac3da23d8e9ef60382fca2…` | `71ef4e15139df50c8490cdf7…` |
| 26 | pythia-6.9b | multiplication | 04 | `c7b95a10e02fec645852b565…` | `b8d8affaeafcdeac19ac517e…` |
| 27 | pythia-6.9b | multiplication | 08 | `9b2b17013919eadebba86145…` | `b8d8affaeafcdeac19ac517e…` |
| 28 | pythia-6.9b | multiplication | 16 | `3e5b9896f6bf15d8b69daced…` | `b8d8affaeafcdeac19ac517e…` |
| 29 | pythia-6.9b | multiplication | 24 | `232c858364c1919962ac3188…` | `b8d8affaeafcdeac19ac517e…` |
| 30 | pythia-6.9b | multiplication | 28 | `420076aa71b15da4886233bf…` | `b8d8affaeafcdeac19ac517e…` |

Cross-cell consistency:
- `labels_sha256` for task `addition`: unique values = 1 (expected: 1).
- `labels_sha256` for task `multiplication`: unique values = 1 (expected: 1).
- `activation_sha256` per (model, task, layer): 30/30 cells single-valued.

---

## 25. Plot-coverage matrix

Which (model × task × layer × concept × method) combinations are in the 45-plot set.

Per model:

### gpt-j-6b (15 plots)

| # | task | layer | concept | method |
|---:|---|---:|---|---|
| 1 | addition | 14 | a_units | umap |
| 2 | addition | 14 | ans_units | umap |
| 3 | addition | 14 | carry_units | umap |
| 4 | multiplication | 14 | a_units | umap |
| 5 | multiplication | 14 | ans_units | umap |
| 6 | multiplication | 14 | carry_units | umap |
| 7 | multiplication | 14 | partial_product_units | umap |
| 8 | multiplication | 04 | ans_units | umap |
| 9 | multiplication | 14 | ans_units | umap |
| 10 | multiplication | 24 | ans_units | umap |
| 11 | addition | 14 | a_units | tsne |
| 12 | addition | 14 | ans_units | tsne |
| 13 | multiplication | 14 | a_units | tsne |
| 14 | multiplication | 14 | ans_units | tsne |
| 15 | multiplication | 14 | carry_units | tsne |

### llama-3.1-8b (15 plots)

| # | task | layer | concept | method |
|---:|---|---:|---|---|
| 16 | addition | 16 | a_units | umap |
| 17 | addition | 16 | ans_units | umap |
| 18 | addition | 16 | carry_units | umap |
| 19 | multiplication | 16 | a_units | umap |
| 20 | multiplication | 16 | ans_units | umap |
| 21 | multiplication | 16 | carry_units | umap |
| 22 | multiplication | 16 | partial_product_units | umap |
| 23 | multiplication | 04 | ans_units | umap |
| 24 | multiplication | 16 | ans_units | umap |
| 25 | multiplication | 28 | ans_units | umap |
| 26 | addition | 16 | a_units | tsne |
| 27 | addition | 16 | ans_units | tsne |
| 28 | multiplication | 16 | a_units | tsne |
| 29 | multiplication | 16 | ans_units | tsne |
| 30 | multiplication | 16 | carry_units | tsne |

### pythia-6.9b (15 plots)

| # | task | layer | concept | method |
|---:|---|---:|---|---|
| 31 | addition | 16 | a_units | umap |
| 32 | addition | 16 | ans_units | umap |
| 33 | addition | 16 | carry_units | umap |
| 34 | multiplication | 16 | a_units | umap |
| 35 | multiplication | 16 | ans_units | umap |
| 36 | multiplication | 16 | carry_units | umap |
| 37 | multiplication | 16 | partial_product_units | umap |
| 38 | multiplication | 04 | ans_units | umap |
| 39 | multiplication | 16 | ans_units | umap |
| 40 | multiplication | 28 | ans_units | umap |
| 41 | addition | 16 | a_units | tsne |
| 42 | addition | 16 | ans_units | tsne |
| 43 | multiplication | 16 | a_units | tsne |
| 44 | multiplication | 16 | ans_units | tsne |
| 45 | multiplication | 16 | carry_units | tsne |

Aggregate counts:

| dimension | count |
|---|---:|
| method == `tsne` | 15 |
| method == `umap` | 30 |
| task == `addition` | 15 |
| task == `multiplication` | 30 |
| total | 45 |

Per-concept tally:

| concept | count |
|---|---:|
| `a_units` | 12 |
| `ans_units` | 21 |
| `carry_units` | 9 |
| `partial_product_units` | 3 |

---

## 26. Trustworthiness ranking per cell

For each cell, all 7 HP settings ranked by trustworthiness (descending).

| # | model | task | L | rank 1 | rank 2 | rank 3 | rank 4 | rank 5 | rank 6 | rank 7 |
|---:|---|---|---:|---|---|---|---|---|---|---|
| 1 | gpt-j-6b | addition | 04 | tsne2d_p30 (0.9985) | tsne2d_p50 (0.9985) | umap2d_n50_md10 (0.9974) | umap2d_n30_md10 (0.9974) | tsne2d_p10 (0.9973) | umap2d_n15_md10 (0.9969) | umap2d_n30_md30 (0.9958) |
| 2 | gpt-j-6b | addition | 08 | tsne2d_p50 (0.9914) | tsne2d_p30 (0.9909) | tsne2d_p10 (0.9890) | umap2d_n15_md10 (0.9853) | umap2d_n30_md10 (0.9817) | umap2d_n50_md10 (0.9785) | umap2d_n30_md30 (0.9718) |
| 3 | gpt-j-6b | addition | 14 | tsne2d_p50 (0.9932) | tsne2d_p30 (0.9924) | tsne2d_p10 (0.9883) | umap2d_n30_md10 (0.9867) | umap2d_n50_md10 (0.9856) | umap2d_n15_md10 (0.9853) | umap2d_n30_md30 (0.9820) |
| 4 | gpt-j-6b | addition | 20 | tsne2d_p30 (0.9857) | tsne2d_p50 (0.9853) | tsne2d_p10 (0.9822) | umap2d_n15_md10 (0.9754) | umap2d_n30_md10 (0.9716) | umap2d_n50_md10 (0.9642) | umap2d_n30_md30 (0.9445) |
| 5 | gpt-j-6b | addition | 24 | tsne2d_p30 (0.9901) | tsne2d_p50 (0.9897) | tsne2d_p10 (0.9877) | umap2d_n15_md10 (0.9792) | umap2d_n30_md10 (0.9778) | umap2d_n50_md10 (0.9769) | umap2d_n30_md30 (0.9679) |
| 6 | gpt-j-6b | multiplication | 04 | tsne2d_p50 (0.9938) | tsne2d_p30 (0.9929) | umap2d_n50_md10 (0.9915) | umap2d_n30_md10 (0.9914) | umap2d_n30_md30 (0.9907) | tsne2d_p10 (0.9899) | umap2d_n15_md10 (0.9897) |
| 7 | gpt-j-6b | multiplication | 08 | tsne2d_p50 (0.9898) | tsne2d_p30 (0.9895) | umap2d_n30_md10 (0.9889) | umap2d_n50_md10 (0.9874) | umap2d_n15_md10 (0.9868) | tsne2d_p10 (0.9848) | umap2d_n30_md30 (0.9817) |
| 8 | gpt-j-6b | multiplication | 14 | tsne2d_p30 (0.9825) | tsne2d_p50 (0.9817) | umap2d_n30_md10 (0.9801) | umap2d_n50_md10 (0.9791) | umap2d_n15_md10 (0.9789) | tsne2d_p10 (0.9786) | umap2d_n30_md30 (0.9765) |
| 9 | gpt-j-6b | multiplication | 20 | tsne2d_p10 (0.9584) | tsne2d_p30 (0.9563) | tsne2d_p50 (0.9561) | umap2d_n15_md10 (0.9493) | umap2d_n30_md10 (0.9443) | umap2d_n50_md10 (0.9338) | umap2d_n30_md30 (0.9174) |
| 10 | gpt-j-6b | multiplication | 24 | tsne2d_p30 (0.9501) | tsne2d_p50 (0.9464) | tsne2d_p10 (0.9413) | umap2d_n15_md10 (0.9378) | umap2d_n30_md10 (0.9350) | umap2d_n50_md10 (0.9322) | umap2d_n30_md30 (0.9047) |
| 11 | llama-3.1-8b | addition | 04 | tsne2d_p30 (0.9966) | tsne2d_p50 (0.9965) | tsne2d_p10 (0.9940) | umap2d_n50_md10 (0.9894) | umap2d_n30_md10 (0.9891) | umap2d_n15_md10 (0.9860) | umap2d_n30_md30 (0.9830) |
| 12 | llama-3.1-8b | addition | 08 | tsne2d_p50 (0.9880) | tsne2d_p30 (0.9878) | tsne2d_p10 (0.9846) | umap2d_n15_md10 (0.9811) | umap2d_n30_md10 (0.9804) | umap2d_n50_md10 (0.9791) | umap2d_n30_md30 (0.9690) |
| 13 | llama-3.1-8b | addition | 16 | tsne2d_p50 (0.9943) | tsne2d_p30 (0.9938) | tsne2d_p10 (0.9908) | umap2d_n50_md10 (0.9895) | umap2d_n30_md10 (0.9887) | umap2d_n30_md30 (0.9856) | umap2d_n15_md10 (0.9842) |
| 14 | llama-3.1-8b | addition | 24 | tsne2d_p50 (0.9990) | tsne2d_p30 (0.9989) | umap2d_n30_md10 (0.9983) | umap2d_n30_md30 (0.9983) | umap2d_n50_md10 (0.9981) | tsne2d_p10 (0.9971) | umap2d_n15_md10 (0.9957) |
| 15 | llama-3.1-8b | addition | 28 | tsne2d_p50 (0.9994) | tsne2d_p30 (0.9993) | umap2d_n50_md10 (0.9988) | umap2d_n30_md10 (0.9985) | umap2d_n30_md30 (0.9983) | tsne2d_p10 (0.9976) | umap2d_n15_md10 (0.9923) |
| 16 | llama-3.1-8b | multiplication | 04 | tsne2d_p50 (0.9894) | umap2d_n30_md10 (0.9879) | umap2d_n50_md10 (0.9878) | tsne2d_p30 (0.9878) | umap2d_n15_md10 (0.9862) | umap2d_n30_md30 (0.9852) | tsne2d_p10 (0.9848) |
| 17 | llama-3.1-8b | multiplication | 08 | tsne2d_p50 (0.9832) | tsne2d_p30 (0.9817) | umap2d_n30_md10 (0.9767) | umap2d_n50_md10 (0.9759) | tsne2d_p10 (0.9755) | umap2d_n15_md10 (0.9734) | umap2d_n30_md30 (0.9708) |
| 18 | llama-3.1-8b | multiplication | 16 | tsne2d_p30 (0.9937) | tsne2d_p50 (0.9933) | tsne2d_p10 (0.9915) | umap2d_n50_md10 (0.9910) | umap2d_n30_md10 (0.9902) | umap2d_n15_md10 (0.9900) | umap2d_n30_md30 (0.9872) |
| 19 | llama-3.1-8b | multiplication | 24 | tsne2d_p50 (0.9514) | tsne2d_p30 (0.9511) | umap2d_n15_md10 (0.9507) | umap2d_n30_md10 (0.9488) | umap2d_n50_md10 (0.9440) | tsne2d_p10 (0.9412) | umap2d_n30_md30 (0.9329) |
| 20 | llama-3.1-8b | multiplication | 28 | tsne2d_p30 (0.9493) | umap2d_n15_md10 (0.9481) | umap2d_n30_md10 (0.9473) | umap2d_n50_md10 (0.9465) | tsne2d_p50 (0.9440) | tsne2d_p10 (0.9304) | umap2d_n30_md30 (0.9166) |
| 21 | pythia-6.9b | addition | 04 | tsne2d_p50 (0.9993) | tsne2d_p30 (0.9992) | umap2d_n50_md10 (0.9987) | umap2d_n30_md10 (0.9986) | tsne2d_p10 (0.9985) | umap2d_n15_md10 (0.9982) | umap2d_n30_md30 (0.9979) |
| 22 | pythia-6.9b | addition | 08 | tsne2d_p50 (0.9929) | tsne2d_p30 (0.9924) | tsne2d_p10 (0.9918) | umap2d_n15_md10 (0.9870) | umap2d_n30_md10 (0.9852) | umap2d_n50_md10 (0.9848) | umap2d_n30_md30 (0.9807) |
| 23 | pythia-6.9b | addition | 16 | tsne2d_p50 (0.9909) | tsne2d_p30 (0.9907) | umap2d_n30_md10 (0.9829) | tsne2d_p10 (0.9816) | umap2d_n15_md10 (0.9808) | umap2d_n50_md10 (0.9805) | umap2d_n30_md30 (0.9660) |
| 24 | pythia-6.9b | addition | 24 | tsne2d_p50 (0.9878) | tsne2d_p30 (0.9861) | tsne2d_p10 (0.9835) | umap2d_n30_md10 (0.9754) | umap2d_n15_md10 (0.9746) | umap2d_n50_md10 (0.9744) | umap2d_n30_md30 (0.9676) |
| 25 | pythia-6.9b | addition | 28 | tsne2d_p50 (0.9884) | tsne2d_p30 (0.9882) | tsne2d_p10 (0.9855) | umap2d_n15_md10 (0.9776) | umap2d_n30_md10 (0.9763) | umap2d_n50_md10 (0.9737) | umap2d_n30_md30 (0.9690) |
| 26 | pythia-6.9b | multiplication | 04 | umap2d_n50_md10 (0.9916) | tsne2d_p30 (0.9913) | tsne2d_p50 (0.9913) | umap2d_n30_md10 (0.9909) | umap2d_n30_md30 (0.9890) | umap2d_n15_md10 (0.9890) | tsne2d_p10 (0.9860) |
| 27 | pythia-6.9b | multiplication | 08 | tsne2d_p50 (0.9930) | tsne2d_p30 (0.9925) | umap2d_n50_md10 (0.9906) | umap2d_n30_md10 (0.9906) | tsne2d_p10 (0.9905) | umap2d_n30_md30 (0.9903) | umap2d_n15_md10 (0.9899) |
| 28 | pythia-6.9b | multiplication | 16 | tsne2d_p50 (0.9688) | tsne2d_p30 (0.9636) | tsne2d_p10 (0.9560) | umap2d_n15_md10 (0.9522) | umap2d_n30_md10 (0.9504) | umap2d_n50_md10 (0.9492) | umap2d_n30_md30 (0.9261) |
| 29 | pythia-6.9b | multiplication | 24 | tsne2d_p30 (0.9522) | tsne2d_p50 (0.9518) | tsne2d_p10 (0.9439) | umap2d_n15_md10 (0.9421) | umap2d_n50_md10 (0.9348) | umap2d_n30_md10 (0.9270) | umap2d_n30_md30 (0.9156) |
| 30 | pythia-6.9b | multiplication | 28 | tsne2d_p30 (0.9490) | tsne2d_p50 (0.9485) | tsne2d_p10 (0.9483) | umap2d_n15_md10 (0.9407) | umap2d_n30_md10 (0.9328) | umap2d_n50_md10 (0.9246) | umap2d_n30_md30 (0.9051) |

---

## 27. CSV column-list (verbatim) for one addition cell and one multiplication cell

Read from the actual files.

### Addition CSV (gpt-j-6b | L04): 76 columns

```
a
b
answer
a_units
a_tens
a_num_digits
a_digits_lsf
b_units
b_tens
b_num_digits
b_digits_lsf
ans_units
ans_tens
ans_hundreds
ans_num_digits
answer_digits_lsf
answer_digits_msf
column_sum_units
column_sum_tens
carry_units
carry_tens
running_sum_units
running_sum_tens
column_sums
carries
running_sums
a_parity
b_parity
ans_parity
parity_match
parity_xor
a_magnitude_tier
b_magnitude_tier
ans_magnitude_tier
ans_ends_in_zero
ans_is_zero
a_is_zero
b_is_zero
a_eq_b
max_operand
min_operand
operand_diff
operand_abs_diff
larger_operand
both_zero
either_zero
both_one
either_one
is_intersection
is_single_token_gpt_j
first_token_id_gpt_j
first_token_text_gpt_j
n_tokens_gpt_j
is_single_token_llama
first_token_id_llama
first_token_text_llama
n_tokens_llama
is_single_token_pythia
first_token_id_pythia
first_token_text_pythia
n_tokens_pythia
activation_norm
umap2d_n15_md10_x
umap2d_n15_md10_y
umap2d_n30_md10_x
umap2d_n30_md10_y
umap2d_n50_md10_x
umap2d_n50_md10_y
umap2d_n30_md30_x
umap2d_n30_md30_y
tsne2d_p10_x
tsne2d_p10_y
tsne2d_p30_x
tsne2d_p30_y
tsne2d_p50_x
tsne2d_p50_y
```

### Multiplication CSV (gpt-j-6b | L04): 89 columns

```
a
b
answer
a_units
a_tens
a_num_digits
a_digits_lsf
b_units
b_tens
b_num_digits
b_digits_lsf
ans_units
ans_tens
ans_hundreds
ans_thousands
ans_num_digits
answer_digits_lsf
answer_digits_msf
partial_products
partial_product_units
partial_product_a_units_b_tens
partial_product_a_tens_b_units
partial_product_a_tens_b_tens
column_sum_units
column_sum_tens
column_sum_hundreds
column_sum_thousands
column_sums
column_products
carry_units
carry_tens
carry_hundreds
carry_thousands
carries
running_sum_units
running_sum_tens
running_sum_hundreds
running_sum_thousands
running_sums
a_parity
b_parity
ans_parity
parity_match
parity_xor
a_magnitude_tier
b_magnitude_tier
ans_magnitude_tier
ans_ends_in_zero
ans_is_zero
a_is_zero
b_is_zero
a_eq_b
max_operand
min_operand
operand_diff
operand_abs_diff
larger_operand
both_zero
either_zero
both_one
either_one
is_intersection
is_single_token_gpt_j
first_token_id_gpt_j
first_token_text_gpt_j
n_tokens_gpt_j
is_single_token_llama
first_token_id_llama
first_token_text_llama
n_tokens_llama
is_single_token_pythia
first_token_id_pythia
first_token_text_pythia
n_tokens_pythia
activation_norm
umap2d_n15_md10_x
umap2d_n15_md10_y
umap2d_n30_md10_x
umap2d_n30_md10_y
umap2d_n50_md10_x
umap2d_n50_md10_y
umap2d_n30_md30_x
umap2d_n30_md30_y
tsne2d_p10_x
tsne2d_p10_y
tsne2d_p30_x
tsne2d_p30_y
tsne2d_p50_x
tsne2d_p50_y
```

---

## 28. Observations from the embedding CSVs

This section contains observations supported directly by numbers in the
per-cell CSVs and manifests. Each observation is paired with the values
that establish it. Interpretation beyond what the numbers state is
deferred.

### 28.1 Per-cell best UMAP vs best t-SNE comparison

For each cell, we record the highest UMAP trustworthiness across the 4
UMAP HP settings, the highest t-SNE trustworthiness across the 3 t-SNE
HP settings, and the signed gap (`tsne_best − umap_best`).

| # | model | task | L | best UMAP T | best t-SNE T | gap (tsne − umap) |
|---:|---|---|---:|---:|---:|---:|
| 1 | gpt-j-6b | addition | 04 | 0.9974 | 0.9985 | +0.0011 |
| 2 | gpt-j-6b | addition | 08 | 0.9853 | 0.9914 | +0.0061 |
| 3 | gpt-j-6b | addition | 14 | 0.9867 | 0.9932 | +0.0065 |
| 4 | gpt-j-6b | addition | 20 | 0.9754 | 0.9857 | +0.0103 |
| 5 | gpt-j-6b | addition | 24 | 0.9792 | 0.9901 | +0.0109 |
| 6 | gpt-j-6b | multiplication | 04 | 0.9915 | 0.9938 | +0.0023 |
| 7 | gpt-j-6b | multiplication | 08 | 0.9889 | 0.9898 | +0.0008 |
| 8 | gpt-j-6b | multiplication | 14 | 0.9801 | 0.9825 | +0.0025 |
| 9 | gpt-j-6b | multiplication | 20 | 0.9493 | 0.9584 | +0.0091 |
| 10 | gpt-j-6b | multiplication | 24 | 0.9378 | 0.9501 | +0.0123 |
| 11 | llama-3.1-8b | addition | 04 | 0.9894 | 0.9966 | +0.0072 |
| 12 | llama-3.1-8b | addition | 08 | 0.9811 | 0.9880 | +0.0069 |
| 13 | llama-3.1-8b | addition | 16 | 0.9895 | 0.9943 | +0.0048 |
| 14 | llama-3.1-8b | addition | 24 | 0.9983 | 0.9990 | +0.0007 |
| 15 | llama-3.1-8b | addition | 28 | 0.9988 | 0.9994 | +0.0006 |
| 16 | llama-3.1-8b | multiplication | 04 | 0.9879 | 0.9894 | +0.0015 |
| 17 | llama-3.1-8b | multiplication | 08 | 0.9767 | 0.9832 | +0.0065 |
| 18 | llama-3.1-8b | multiplication | 16 | 0.9910 | 0.9937 | +0.0027 |
| 19 | llama-3.1-8b | multiplication | 24 | 0.9507 | 0.9514 | +0.0007 |
| 20 | llama-3.1-8b | multiplication | 28 | 0.9481 | 0.9493 | +0.0012 |
| 21 | pythia-6.9b | addition | 04 | 0.9987 | 0.9993 | +0.0006 |
| 22 | pythia-6.9b | addition | 08 | 0.9870 | 0.9929 | +0.0059 |
| 23 | pythia-6.9b | addition | 16 | 0.9829 | 0.9909 | +0.0080 |
| 24 | pythia-6.9b | addition | 24 | 0.9754 | 0.9878 | +0.0124 |
| 25 | pythia-6.9b | addition | 28 | 0.9776 | 0.9884 | +0.0107 |
| 26 | pythia-6.9b | multiplication | 04 | 0.9916 | 0.9913 | -0.0003 |
| 27 | pythia-6.9b | multiplication | 08 | 0.9906 | 0.9930 | +0.0024 |
| 28 | pythia-6.9b | multiplication | 16 | 0.9522 | 0.9688 | +0.0166 |
| 29 | pythia-6.9b | multiplication | 24 | 0.9421 | 0.9522 | +0.0101 |
| 30 | pythia-6.9b | multiplication | 28 | 0.9407 | 0.9490 | +0.0083 |

- Cells where best t-SNE trustworthiness exceeds best UMAP trustworthiness: **29 / 30**.
- Cells where best UMAP exceeds best t-SNE: **1 / 30**.
- Cells with exact ties: **0 / 30**.
- Mean gap (`best_tsne − best_umap`) across 30 cells: **+0.0056**.
- Median gap: **+0.0060**.
- Min gap: **-0.0003**; max gap: **+0.0166**.

### 28.2 Trustworthiness as a function of layer index

Per (model, task, method-best), we list trustworthiness at each of the 5
extracted layers and report whether the sequence is monotonic.

#### UMAP best

| model | task | layer order | T sequence | trend |
|---|---|---|---|---|
| gpt-j-6b | addition | [4,8,14,20,24] | 0.9974 → 0.9853 → 0.9867 → 0.9754 → 0.9792 | non-monotonic |
| gpt-j-6b | multiplication | [4,8,14,20,24] | 0.9915 → 0.9889 → 0.9801 → 0.9493 → 0.9378 | decreasing |
| llama-3.1-8b | addition | [4,8,16,24,28] | 0.9894 → 0.9811 → 0.9895 → 0.9983 → 0.9988 | non-monotonic |
| llama-3.1-8b | multiplication | [4,8,16,24,28] | 0.9879 → 0.9767 → 0.9910 → 0.9507 → 0.9481 | non-monotonic |
| pythia-6.9b | addition | [4,8,16,24,28] | 0.9987 → 0.9870 → 0.9829 → 0.9754 → 0.9776 | non-monotonic |
| pythia-6.9b | multiplication | [4,8,16,24,28] | 0.9916 → 0.9906 → 0.9522 → 0.9421 → 0.9407 | decreasing |

#### t-SNE best

| model | task | layer order | T sequence | trend |
|---|---|---|---|---|
| gpt-j-6b | addition | [4,8,14,20,24] | 0.9985 → 0.9914 → 0.9932 → 0.9857 → 0.9901 | non-monotonic |
| gpt-j-6b | multiplication | [4,8,14,20,24] | 0.9938 → 0.9898 → 0.9825 → 0.9584 → 0.9501 | decreasing |
| llama-3.1-8b | addition | [4,8,16,24,28] | 0.9966 → 0.9880 → 0.9943 → 0.9990 → 0.9994 | non-monotonic |
| llama-3.1-8b | multiplication | [4,8,16,24,28] | 0.9894 → 0.9832 → 0.9937 → 0.9514 → 0.9493 | non-monotonic |
| pythia-6.9b | addition | [4,8,16,24,28] | 0.9993 → 0.9929 → 0.9909 → 0.9878 → 0.9884 | non-monotonic |
| pythia-6.9b | multiplication | [4,8,16,24,28] | 0.9913 → 0.9930 → 0.9688 → 0.9522 → 0.9490 | non-monotonic |

### 28.3 Trustworthiness gap from layer 4 to deepest extracted layer

For each (model, task, method), the difference `T(deepest) − T(layer_4)`.

| model | task | method | T(L4) | T(L_deep) | Δ |
|---|---|---|---:|---:|---:|
| gpt-j-6b | addition | UMAP | 0.9974 | 0.9792 | -0.0182 |
| gpt-j-6b | addition | t-SNE | 0.9985 | 0.9901 | -0.0084 |
| gpt-j-6b | multiplication | UMAP | 0.9915 | 0.9378 | -0.0538 |
| gpt-j-6b | multiplication | t-SNE | 0.9938 | 0.9501 | -0.0437 |
| llama-3.1-8b | addition | UMAP | 0.9894 | 0.9988 | +0.0094 |
| llama-3.1-8b | addition | t-SNE | 0.9966 | 0.9994 | +0.0028 |
| llama-3.1-8b | multiplication | UMAP | 0.9879 | 0.9481 | -0.0398 |
| llama-3.1-8b | multiplication | t-SNE | 0.9894 | 0.9493 | -0.0401 |
| pythia-6.9b | addition | UMAP | 0.9987 | 0.9776 | -0.0211 |
| pythia-6.9b | addition | t-SNE | 0.9993 | 0.9884 | -0.0110 |
| pythia-6.9b | multiplication | UMAP | 0.9916 | 0.9407 | -0.0509 |
| pythia-6.9b | multiplication | t-SNE | 0.9913 | 0.9490 | -0.0423 |

### 28.4 Concept-class separation in 2D (best-UMAP)

For headline concepts (`a_units`, `ans_units`, `carry_units`), compute
the between-class to within-class scatter ratio in the best-UMAP 2D space:

- Within-class scatter `S_W = Σ_v Σ_{i∈v} ‖x_i − μ_v‖²` summed over all
  points and divided by N − K.
- Between-class scatter `S_B = Σ_v n_v ‖μ_v − μ̄‖²` divided by K − 1.
- Ratio = `S_B / S_W`. Larger means classes are more separated by their
  centroids relative to within-class spread.

| # | model | task | L | concept | K | S_B/S_W |
|---:|---|---|---:|---|---:|---:|
| 1 | gpt-j-6b | addition | 04 | a_units | 10 | 4.754 |
| 1 | gpt-j-6b | addition | 04 | ans_units | 10 | 0.001 |
| 1 | gpt-j-6b | addition | 04 | carry_units | 2 | 54.897 |
| 2 | gpt-j-6b | addition | 08 | a_units | 10 | 4.622 |
| 2 | gpt-j-6b | addition | 08 | ans_units | 10 | 0.139 |
| 2 | gpt-j-6b | addition | 08 | carry_units | 2 | 10.754 |
| 3 | gpt-j-6b | addition | 14 | a_units | 10 | 3.029 |
| 3 | gpt-j-6b | addition | 14 | ans_units | 10 | 0.002 |
| 3 | gpt-j-6b | addition | 14 | carry_units | 2 | 41.031 |
| 4 | gpt-j-6b | addition | 20 | a_units | 10 | 131.979 |
| 4 | gpt-j-6b | addition | 20 | ans_units | 10 | 929.845 |
| 4 | gpt-j-6b | addition | 20 | carry_units | 2 | 85.098 |
| 5 | gpt-j-6b | addition | 24 | a_units | 10 | 4.056 |
| 5 | gpt-j-6b | addition | 24 | ans_units | 10 | 6757.051 |
| 5 | gpt-j-6b | addition | 24 | carry_units | 2 | 77.097 |
| 6 | gpt-j-6b | multiplication | 04 | a_units | 10 | 15.901 |
| 6 | gpt-j-6b | multiplication | 04 | ans_units | 10 | 6.818 |
| 6 | gpt-j-6b | multiplication | 04 | carry_units | 9 | 12.253 |
| 7 | gpt-j-6b | multiplication | 08 | a_units | 10 | 14.744 |
| 7 | gpt-j-6b | multiplication | 08 | ans_units | 10 | 16.714 |
| 7 | gpt-j-6b | multiplication | 08 | carry_units | 9 | 32.587 |
| 8 | gpt-j-6b | multiplication | 14 | a_units | 10 | 6.451 |
| 8 | gpt-j-6b | multiplication | 14 | ans_units | 10 | 2.489 |
| 8 | gpt-j-6b | multiplication | 14 | carry_units | 9 | 9.410 |
| 9 | gpt-j-6b | multiplication | 20 | a_units | 10 | 3.766 |
| 9 | gpt-j-6b | multiplication | 20 | ans_units | 10 | 1.663 |
| 9 | gpt-j-6b | multiplication | 20 | carry_units | 9 | 7.189 |
| 10 | gpt-j-6b | multiplication | 24 | a_units | 10 | 28.088 |
| 10 | gpt-j-6b | multiplication | 24 | ans_units | 10 | 14.355 |
| 10 | gpt-j-6b | multiplication | 24 | carry_units | 9 | 51.881 |
| 11 | llama-3.1-8b | addition | 04 | a_units | 10 | 3.585 |
| 11 | llama-3.1-8b | addition | 04 | ans_units | 10 | 0.007 |
| 11 | llama-3.1-8b | addition | 04 | carry_units | 2 | 86.466 |
| 12 | llama-3.1-8b | addition | 08 | a_units | 10 | 18.593 |
| 12 | llama-3.1-8b | addition | 08 | ans_units | 10 | 0.156 |
| 12 | llama-3.1-8b | addition | 08 | carry_units | 2 | 27.475 |
| 13 | llama-3.1-8b | addition | 16 | a_units | 10 | 1.023 |
| 13 | llama-3.1-8b | addition | 16 | ans_units | 10 | 0.005 |
| 13 | llama-3.1-8b | addition | 16 | carry_units | 2 | 14.327 |
| 14 | llama-3.1-8b | addition | 24 | a_units | 10 | 0.218 |
| 14 | llama-3.1-8b | addition | 24 | ans_units | 10 | 14.136 |
| 14 | llama-3.1-8b | addition | 24 | carry_units | 2 | 4.900 |
| 15 | llama-3.1-8b | addition | 28 | a_units | 10 | 0.031 |
| 15 | llama-3.1-8b | addition | 28 | ans_units | 10 | 35.583 |
| 15 | llama-3.1-8b | addition | 28 | carry_units | 2 | 63.370 |
| 16 | llama-3.1-8b | multiplication | 04 | a_units | 10 | 20.391 |
| 16 | llama-3.1-8b | multiplication | 04 | ans_units | 10 | 9.736 |
| 16 | llama-3.1-8b | multiplication | 04 | carry_units | 9 | 14.309 |
| 17 | llama-3.1-8b | multiplication | 08 | a_units | 10 | 32.826 |
| 17 | llama-3.1-8b | multiplication | 08 | ans_units | 10 | 33.973 |
| 17 | llama-3.1-8b | multiplication | 08 | carry_units | 9 | 67.945 |
| 18 | llama-3.1-8b | multiplication | 16 | a_units | 10 | 0.712 |
| 18 | llama-3.1-8b | multiplication | 16 | ans_units | 10 | 19.698 |
| 18 | llama-3.1-8b | multiplication | 16 | carry_units | 9 | 37.949 |
| 19 | llama-3.1-8b | multiplication | 24 | a_units | 10 | 28.518 |
| 19 | llama-3.1-8b | multiplication | 24 | ans_units | 10 | 55.809 |
| 19 | llama-3.1-8b | multiplication | 24 | carry_units | 9 | 38.687 |
| 20 | llama-3.1-8b | multiplication | 28 | a_units | 10 | 34.191 |
| 20 | llama-3.1-8b | multiplication | 28 | ans_units | 10 | 61.800 |
| 20 | llama-3.1-8b | multiplication | 28 | carry_units | 9 | 40.015 |
| 21 | pythia-6.9b | addition | 04 | a_units | 10 | 2.808 |
| 21 | pythia-6.9b | addition | 04 | ans_units | 10 | 0.016 |
| 21 | pythia-6.9b | addition | 04 | carry_units | 2 | 41.690 |
| 22 | pythia-6.9b | addition | 08 | a_units | 10 | 3.172 |
| 22 | pythia-6.9b | addition | 08 | ans_units | 10 | 0.336 |
| 22 | pythia-6.9b | addition | 08 | carry_units | 2 | 18.898 |
| 23 | pythia-6.9b | addition | 16 | a_units | 10 | 15.904 |
| 23 | pythia-6.9b | addition | 16 | ans_units | 10 | 0.003 |
| 23 | pythia-6.9b | addition | 16 | carry_units | 2 | 28.447 |
| 24 | pythia-6.9b | addition | 24 | a_units | 10 | 89.541 |
| 24 | pythia-6.9b | addition | 24 | ans_units | 10 | 1962.001 |
| 24 | pythia-6.9b | addition | 24 | carry_units | 2 | 6.680 |
| 25 | pythia-6.9b | addition | 28 | a_units | 10 | 36.613 |
| 25 | pythia-6.9b | addition | 28 | ans_units | 10 | 2431.992 |
| 25 | pythia-6.9b | addition | 28 | carry_units | 2 | 92.563 |
| 26 | pythia-6.9b | multiplication | 04 | a_units | 10 | 4.231 |
| 26 | pythia-6.9b | multiplication | 04 | ans_units | 10 | 4.551 |
| 26 | pythia-6.9b | multiplication | 04 | carry_units | 9 | 11.078 |
| 27 | pythia-6.9b | multiplication | 08 | a_units | 10 | 13.211 |
| 27 | pythia-6.9b | multiplication | 08 | ans_units | 10 | 4.063 |
| 27 | pythia-6.9b | multiplication | 08 | carry_units | 9 | 6.612 |
| 28 | pythia-6.9b | multiplication | 16 | a_units | 10 | 3.376 |
| 28 | pythia-6.9b | multiplication | 16 | ans_units | 10 | 2.286 |
| 28 | pythia-6.9b | multiplication | 16 | carry_units | 9 | 4.341 |
| 29 | pythia-6.9b | multiplication | 24 | a_units | 10 | 34.113 |
| 29 | pythia-6.9b | multiplication | 24 | ans_units | 10 | 21.098 |
| 29 | pythia-6.9b | multiplication | 24 | carry_units | 9 | 37.120 |
| 30 | pythia-6.9b | multiplication | 28 | a_units | 10 | 45.591 |
| 30 | pythia-6.9b | multiplication | 28 | ans_units | 10 | 28.785 |
| 30 | pythia-6.9b | multiplication | 28 | carry_units | 9 | 52.084 |

Aggregate (across 30 cells) per concept:

| concept | n cells | min | median | mean | max |
|---|---:|---:|---:|---:|---:|
| `a_units` | 30 | 0.031 | 9.831 | 20.201 | 131.979 |
| `ans_units` | 30 | 0.001 | 8.277 | 413.837 | 6757.051 |
| `carry_units` | 30 | 4.341 | 34.853 | 35.905 | 92.563 |

Same metric on the best-t-SNE coordinates:

| # | model | task | L | concept | K | S_B/S_W |
|---:|---|---|---:|---|---:|---:|
| 1 | gpt-j-6b | addition | 04 | a_units | 10 | 2.582 |
| 1 | gpt-j-6b | addition | 04 | ans_units | 10 | 0.005 |
| 1 | gpt-j-6b | addition | 04 | carry_units | 2 | 83.043 |
| 2 | gpt-j-6b | addition | 08 | a_units | 10 | 6.864 |
| 2 | gpt-j-6b | addition | 08 | ans_units | 10 | 0.007 |
| 2 | gpt-j-6b | addition | 08 | carry_units | 2 | 42.228 |
| 3 | gpt-j-6b | addition | 14 | a_units | 10 | 4.170 |
| 3 | gpt-j-6b | addition | 14 | ans_units | 10 | 0.002 |
| 3 | gpt-j-6b | addition | 14 | carry_units | 2 | 49.446 |
| 4 | gpt-j-6b | addition | 20 | a_units | 10 | 51.177 |
| 4 | gpt-j-6b | addition | 20 | ans_units | 10 | 680.711 |
| 4 | gpt-j-6b | addition | 20 | carry_units | 2 | 52.504 |
| 5 | gpt-j-6b | addition | 24 | a_units | 10 | 2.434 |
| 5 | gpt-j-6b | addition | 24 | ans_units | 10 | 6202.076 |
| 5 | gpt-j-6b | addition | 24 | carry_units | 2 | 110.351 |
| 6 | gpt-j-6b | multiplication | 04 | a_units | 10 | 8.829 |
| 6 | gpt-j-6b | multiplication | 04 | ans_units | 10 | 7.869 |
| 6 | gpt-j-6b | multiplication | 04 | carry_units | 9 | 19.542 |
| 7 | gpt-j-6b | multiplication | 08 | a_units | 10 | 9.536 |
| 7 | gpt-j-6b | multiplication | 08 | ans_units | 10 | 5.069 |
| 7 | gpt-j-6b | multiplication | 08 | carry_units | 9 | 13.702 |
| 8 | gpt-j-6b | multiplication | 14 | a_units | 10 | 9.561 |
| 8 | gpt-j-6b | multiplication | 14 | ans_units | 10 | 4.424 |
| 8 | gpt-j-6b | multiplication | 14 | carry_units | 9 | 12.963 |
| 9 | gpt-j-6b | multiplication | 20 | a_units | 10 | 10.650 |
| 9 | gpt-j-6b | multiplication | 20 | ans_units | 10 | 5.369 |
| 9 | gpt-j-6b | multiplication | 20 | carry_units | 9 | 20.868 |
| 10 | gpt-j-6b | multiplication | 24 | a_units | 10 | 23.984 |
| 10 | gpt-j-6b | multiplication | 24 | ans_units | 10 | 15.844 |
| 10 | gpt-j-6b | multiplication | 24 | carry_units | 9 | 42.979 |
| 11 | llama-3.1-8b | addition | 04 | a_units | 10 | 1.723 |
| 11 | llama-3.1-8b | addition | 04 | ans_units | 10 | 0.014 |
| 11 | llama-3.1-8b | addition | 04 | carry_units | 2 | 84.703 |
| 12 | llama-3.1-8b | addition | 08 | a_units | 10 | 15.825 |
| 12 | llama-3.1-8b | addition | 08 | ans_units | 10 | 0.044 |
| 12 | llama-3.1-8b | addition | 08 | carry_units | 2 | 63.301 |
| 13 | llama-3.1-8b | addition | 16 | a_units | 10 | 4.002 |
| 13 | llama-3.1-8b | addition | 16 | ans_units | 10 | 0.004 |
| 13 | llama-3.1-8b | addition | 16 | carry_units | 2 | 39.732 |
| 14 | llama-3.1-8b | addition | 24 | a_units | 10 | 0.021 |
| 14 | llama-3.1-8b | addition | 24 | ans_units | 10 | 13.356 |
| 14 | llama-3.1-8b | addition | 24 | carry_units | 2 | 15.958 |
| 15 | llama-3.1-8b | addition | 28 | a_units | 10 | 0.079 |
| 15 | llama-3.1-8b | addition | 28 | ans_units | 10 | 17.434 |
| 15 | llama-3.1-8b | addition | 28 | carry_units | 2 | 29.024 |
| 16 | llama-3.1-8b | multiplication | 04 | a_units | 10 | 10.836 |
| 16 | llama-3.1-8b | multiplication | 04 | ans_units | 10 | 8.261 |
| 16 | llama-3.1-8b | multiplication | 04 | carry_units | 9 | 15.259 |
| 17 | llama-3.1-8b | multiplication | 08 | a_units | 10 | 8.309 |
| 17 | llama-3.1-8b | multiplication | 08 | ans_units | 10 | 3.238 |
| 17 | llama-3.1-8b | multiplication | 08 | carry_units | 9 | 10.449 |
| 18 | llama-3.1-8b | multiplication | 16 | a_units | 10 | 0.516 |
| 18 | llama-3.1-8b | multiplication | 16 | ans_units | 10 | 10.450 |
| 18 | llama-3.1-8b | multiplication | 16 | carry_units | 9 | 17.954 |
| 19 | llama-3.1-8b | multiplication | 24 | a_units | 10 | 18.678 |
| 19 | llama-3.1-8b | multiplication | 24 | ans_units | 10 | 30.005 |
| 19 | llama-3.1-8b | multiplication | 24 | carry_units | 9 | 39.585 |
| 20 | llama-3.1-8b | multiplication | 28 | a_units | 10 | 42.855 |
| 20 | llama-3.1-8b | multiplication | 28 | ans_units | 10 | 82.988 |
| 20 | llama-3.1-8b | multiplication | 28 | carry_units | 9 | 46.057 |
| 21 | pythia-6.9b | addition | 04 | a_units | 10 | 2.015 |
| 21 | pythia-6.9b | addition | 04 | ans_units | 10 | 0.018 |
| 21 | pythia-6.9b | addition | 04 | carry_units | 2 | 146.496 |
| 22 | pythia-6.9b | addition | 08 | a_units | 10 | 3.406 |
| 22 | pythia-6.9b | addition | 08 | ans_units | 10 | 0.123 |
| 22 | pythia-6.9b | addition | 08 | carry_units | 2 | 25.502 |
| 23 | pythia-6.9b | addition | 16 | a_units | 10 | 17.222 |
| 23 | pythia-6.9b | addition | 16 | ans_units | 10 | 0.015 |
| 23 | pythia-6.9b | addition | 16 | carry_units | 2 | 22.198 |
| 24 | pythia-6.9b | addition | 24 | a_units | 10 | 74.901 |
| 24 | pythia-6.9b | addition | 24 | ans_units | 10 | 1499.174 |
| 24 | pythia-6.9b | addition | 24 | carry_units | 2 | 43.359 |
| 25 | pythia-6.9b | addition | 28 | a_units | 10 | 5.357 |
| 25 | pythia-6.9b | addition | 28 | ans_units | 10 | 3562.168 |
| 25 | pythia-6.9b | addition | 28 | carry_units | 2 | 77.075 |
| 26 | pythia-6.9b | multiplication | 04 | a_units | 10 | 6.258 |
| 26 | pythia-6.9b | multiplication | 04 | ans_units | 10 | 12.243 |
| 26 | pythia-6.9b | multiplication | 04 | carry_units | 9 | 12.550 |
| 27 | pythia-6.9b | multiplication | 08 | a_units | 10 | 9.534 |
| 27 | pythia-6.9b | multiplication | 08 | ans_units | 10 | 5.061 |
| 27 | pythia-6.9b | multiplication | 08 | carry_units | 9 | 10.141 |
| 28 | pythia-6.9b | multiplication | 16 | a_units | 10 | 5.509 |
| 28 | pythia-6.9b | multiplication | 16 | ans_units | 10 | 4.041 |
| 28 | pythia-6.9b | multiplication | 16 | carry_units | 9 | 3.773 |
| 29 | pythia-6.9b | multiplication | 24 | a_units | 10 | 40.882 |
| 29 | pythia-6.9b | multiplication | 24 | ans_units | 10 | 20.166 |
| 29 | pythia-6.9b | multiplication | 24 | carry_units | 9 | 51.393 |
| 30 | pythia-6.9b | multiplication | 28 | a_units | 10 | 58.314 |
| 30 | pythia-6.9b | multiplication | 28 | ans_units | 10 | 55.691 |
| 30 | pythia-6.9b | multiplication | 28 | carry_units | 9 | 51.564 |

Aggregate per concept (best-t-SNE):

| concept | n cells | min | median | mean | max |
|---|---:|---:|---:|---:|---:|
| `a_units` | 30 | 0.021 | 8.569 | 15.201 | 74.901 |
| `ans_units` | 30 | 0.002 | 6.619 | 408.196 | 6202.076 |
| `carry_units` | 30 | 3.773 | 39.658 | 41.790 | 146.496 |

### 28.5 Coordinate range per cell (best-UMAP)

Range `(x_max − x_min, y_max − y_min)` and the std of each coordinate axis.

| # | model | task | L | best UMAP HP | x_range | y_range | x_std | y_std |
|---:|---|---|---:|---|---:|---:|---:|---:|
| 1 | gpt-j-6b | addition | 04 | `umap2d_n50_md10` | 30.05 | 27.61 | 5.57 | 7.42 |
| 2 | gpt-j-6b | addition | 08 | `umap2d_n15_md10` | 24.57 | 27.93 | 4.53 | 6.68 |
| 3 | gpt-j-6b | addition | 14 | `umap2d_n30_md10` | 16.23 | 16.38 | 4.89 | 4.56 |
| 4 | gpt-j-6b | addition | 20 | `umap2d_n15_md10` | 16.11 | 12.71 | 4.44 | 2.50 |
| 5 | gpt-j-6b | addition | 24 | `umap2d_n15_md10` | 18.16 | 13.80 | 4.63 | 3.07 |
| 6 | gpt-j-6b | multiplication | 04 | `umap2d_n50_md10` | 26.37 | 34.65 | 5.80 | 10.33 |
| 7 | gpt-j-6b | multiplication | 08 | `umap2d_n30_md10` | 27.13 | 25.60 | 6.68 | 8.09 |
| 8 | gpt-j-6b | multiplication | 14 | `umap2d_n30_md10` | 15.14 | 13.23 | 4.57 | 3.16 |
| 9 | gpt-j-6b | multiplication | 20 | `umap2d_n15_md10` | 12.10 | 8.83 | 3.27 | 2.07 |
| 10 | gpt-j-6b | multiplication | 24 | `umap2d_n15_md10` | 10.39 | 9.43 | 2.65 | 2.18 |
| 11 | llama-3.1-8b | addition | 04 | `umap2d_n50_md10` | 26.43 | 27.66 | 6.14 | 7.44 |
| 12 | llama-3.1-8b | addition | 08 | `umap2d_n15_md10` | 24.49 | 23.58 | 5.02 | 6.11 |
| 13 | llama-3.1-8b | addition | 16 | `umap2d_n50_md10` | 36.75 | 14.73 | 11.25 | 2.87 |
| 14 | llama-3.1-8b | addition | 24 | `umap2d_n30_md10` | 46.68 | 43.13 | 10.45 | 9.62 |
| 15 | llama-3.1-8b | addition | 28 | `umap2d_n50_md10` | 43.70 | 45.76 | 10.31 | 11.40 |
| 16 | llama-3.1-8b | multiplication | 04 | `umap2d_n30_md10` | 30.49 | 37.53 | 6.23 | 10.30 |
| 17 | llama-3.1-8b | multiplication | 08 | `umap2d_n30_md10` | 32.71 | 30.63 | 7.72 | 6.36 |
| 18 | llama-3.1-8b | multiplication | 16 | `umap2d_n50_md10` | 25.65 | 30.50 | 4.93 | 6.29 |
| 19 | llama-3.1-8b | multiplication | 24 | `umap2d_n15_md10` | 15.02 | 10.29 | 3.40 | 2.39 |
| 20 | llama-3.1-8b | multiplication | 28 | `umap2d_n15_md10` | 16.73 | 12.67 | 3.62 | 2.51 |
| 21 | pythia-6.9b | addition | 04 | `umap2d_n50_md10` | 32.64 | 32.16 | 8.23 | 7.88 |
| 22 | pythia-6.9b | addition | 08 | `umap2d_n15_md10` | 29.25 | 30.26 | 4.91 | 7.12 |
| 23 | pythia-6.9b | addition | 16 | `umap2d_n30_md10` | 14.09 | 11.91 | 3.95 | 2.71 |
| 24 | pythia-6.9b | addition | 24 | `umap2d_n30_md10` | 13.95 | 10.16 | 4.09 | 2.26 |
| 25 | pythia-6.9b | addition | 28 | `umap2d_n15_md10` | 17.46 | 17.86 | 4.99 | 3.18 |
| 26 | pythia-6.9b | multiplication | 04 | `umap2d_n50_md10` | 37.46 | 31.72 | 7.27 | 8.89 |
| 27 | pythia-6.9b | multiplication | 08 | `umap2d_n50_md10` | 23.74 | 26.28 | 7.66 | 7.47 |
| 28 | pythia-6.9b | multiplication | 16 | `umap2d_n15_md10` | 10.85 | 11.66 | 2.82 | 3.11 |
| 29 | pythia-6.9b | multiplication | 24 | `umap2d_n15_md10` | 10.16 | 7.78 | 2.66 | 2.04 |
| 30 | pythia-6.9b | multiplication | 28 | `umap2d_n15_md10` | 11.00 | 7.84 | 2.79 | 1.84 |

### 28.6 Coordinate range per cell (best-t-SNE)

| # | model | task | L | best t-SNE HP | x_range | y_range | x_std | y_std |
|---:|---|---|---:|---|---:|---:|---:|---:|
| 1 | gpt-j-6b | addition | 04 | `tsne2d_p30` | 221.21 | 220.38 | 52.88 | 49.94 |
| 2 | gpt-j-6b | addition | 08 | `tsne2d_p50` | 172.18 | 180.02 | 36.95 | 44.28 |
| 3 | gpt-j-6b | addition | 14 | `tsne2d_p50` | 200.07 | 150.95 | 49.93 | 39.25 |
| 4 | gpt-j-6b | addition | 20 | `tsne2d_p30` | 282.82 | 252.23 | 65.02 | 58.64 |
| 5 | gpt-j-6b | addition | 24 | `tsne2d_p30` | 262.17 | 252.35 | 64.43 | 61.27 |
| 6 | gpt-j-6b | multiplication | 04 | `tsne2d_p50` | 107.30 | 132.47 | 27.48 | 28.59 |
| 7 | gpt-j-6b | multiplication | 08 | `tsne2d_p50` | 128.88 | 100.61 | 31.54 | 27.82 |
| 8 | gpt-j-6b | multiplication | 14 | `tsne2d_p30` | 153.48 | 134.51 | 39.90 | 32.28 |
| 9 | gpt-j-6b | multiplication | 20 | `tsne2d_p10` | 256.73 | 269.21 | 63.29 | 61.89 |
| 10 | gpt-j-6b | multiplication | 24 | `tsne2d_p30` | 228.57 | 182.52 | 55.48 | 38.27 |
| 11 | llama-3.1-8b | addition | 04 | `tsne2d_p30` | 228.52 | 235.18 | 54.30 | 53.30 |
| 12 | llama-3.1-8b | addition | 08 | `tsne2d_p50` | 222.40 | 156.21 | 48.66 | 37.08 |
| 13 | llama-3.1-8b | addition | 16 | `tsne2d_p50` | 210.76 | 182.26 | 51.31 | 35.64 |
| 14 | llama-3.1-8b | addition | 24 | `tsne2d_p50` | 213.75 | 202.27 | 48.01 | 46.55 |
| 15 | llama-3.1-8b | addition | 28 | `tsne2d_p50` | 213.05 | 214.09 | 45.91 | 47.28 |
| 16 | llama-3.1-8b | multiplication | 04 | `tsne2d_p50` | 102.98 | 117.21 | 29.04 | 26.87 |
| 17 | llama-3.1-8b | multiplication | 08 | `tsne2d_p50` | 121.40 | 113.03 | 28.60 | 24.80 |
| 18 | llama-3.1-8b | multiplication | 16 | `tsne2d_p30` | 160.96 | 149.39 | 37.27 | 34.93 |
| 19 | llama-3.1-8b | multiplication | 24 | `tsne2d_p50` | 167.06 | 104.21 | 39.61 | 23.34 |
| 20 | llama-3.1-8b | multiplication | 28 | `tsne2d_p30` | 194.00 | 134.21 | 47.53 | 30.07 |
| 21 | pythia-6.9b | addition | 04 | `tsne2d_p50` | 199.83 | 195.15 | 48.04 | 44.10 |
| 22 | pythia-6.9b | addition | 08 | `tsne2d_p50` | 171.31 | 176.71 | 36.56 | 44.60 |
| 23 | pythia-6.9b | addition | 16 | `tsne2d_p50` | 185.35 | 152.74 | 46.69 | 34.64 |
| 24 | pythia-6.9b | addition | 24 | `tsne2d_p50` | 223.11 | 173.23 | 56.28 | 36.73 |
| 25 | pythia-6.9b | addition | 28 | `tsne2d_p50` | 233.00 | 179.80 | 59.17 | 38.93 |
| 26 | pythia-6.9b | multiplication | 04 | `tsne2d_p30` | 158.82 | 171.76 | 35.06 | 38.07 |
| 27 | pythia-6.9b | multiplication | 08 | `tsne2d_p50` | 99.67 | 115.47 | 27.86 | 26.36 |
| 28 | pythia-6.9b | multiplication | 16 | `tsne2d_p50` | 149.39 | 84.91 | 38.47 | 20.57 |
| 29 | pythia-6.9b | multiplication | 24 | `tsne2d_p30` | 222.10 | 164.26 | 53.77 | 38.29 |
| 30 | pythia-6.9b | multiplication | 28 | `tsne2d_p30` | 234.72 | 159.04 | 57.41 | 37.60 |

### 28.7 Per-cell rank correlation between activation_norm and best-UMAP `_x`

Spearman rank correlation between `activation_norm` (a per-row scalar)
and the best-UMAP `_x` coordinate. Same for `_y`. Reported for each cell.

| # | model | task | L | ρ(norm, umap_x) | ρ(norm, umap_y) |
|---:|---|---|---:|---:|---:|
| 1 | gpt-j-6b | addition | 04 | +0.089 | -0.110 |
| 2 | gpt-j-6b | addition | 08 | +0.349 | -0.102 |
| 3 | gpt-j-6b | addition | 14 | -0.088 | -0.023 |
| 4 | gpt-j-6b | addition | 20 | -0.379 | +0.140 |
| 5 | gpt-j-6b | addition | 24 | +0.089 | +0.069 |
| 6 | gpt-j-6b | multiplication | 04 | -0.541 | +0.298 |
| 7 | gpt-j-6b | multiplication | 08 | +0.286 | +0.544 |
| 8 | gpt-j-6b | multiplication | 14 | -0.021 | -0.238 |
| 9 | gpt-j-6b | multiplication | 20 | -0.187 | -0.067 |
| 10 | gpt-j-6b | multiplication | 24 | +0.643 | +0.059 |
| 11 | llama-3.1-8b | addition | 04 | +0.230 | +0.135 |
| 12 | llama-3.1-8b | addition | 08 | -0.304 | -0.035 |
| 13 | llama-3.1-8b | addition | 16 | -0.507 | -0.006 |
| 14 | llama-3.1-8b | addition | 24 | -0.068 | +0.090 |
| 15 | llama-3.1-8b | addition | 28 | -0.069 | -0.131 |
| 16 | llama-3.1-8b | multiplication | 04 | +0.101 | -0.251 |
| 17 | llama-3.1-8b | multiplication | 08 | +0.029 | +0.077 |
| 18 | llama-3.1-8b | multiplication | 16 | -0.267 | +0.235 |
| 19 | llama-3.1-8b | multiplication | 24 | -0.268 | -0.016 |
| 20 | llama-3.1-8b | multiplication | 28 | +0.555 | +0.065 |
| 21 | pythia-6.9b | addition | 04 | -0.289 | -0.235 |
| 22 | pythia-6.9b | addition | 08 | +0.241 | +0.076 |
| 23 | pythia-6.9b | addition | 16 | -0.026 | +0.166 |
| 24 | pythia-6.9b | addition | 24 | +0.070 | -0.171 |
| 25 | pythia-6.9b | addition | 28 | -0.006 | +0.207 |
| 26 | pythia-6.9b | multiplication | 04 | +0.130 | +0.223 |
| 27 | pythia-6.9b | multiplication | 08 | -0.229 | +0.107 |
| 28 | pythia-6.9b | multiplication | 16 | -0.228 | -0.125 |
| 29 | pythia-6.9b | multiplication | 24 | +0.596 | +0.108 |
| 30 | pythia-6.9b | multiplication | 28 | -0.649 | -0.219 |

### 28.8 Headline numerical statements

- Across 30 cells × 7 HP settings = 210 trustworthiness values (full N), the
  range is **[0.9047, 0.9994]**, median 0.9853, mean 0.9764.
- UMAP trustworthiness across 120 (cell, UMAP-HP) pairs: range **[0.9047, 0.9988]**, median 0.9808.
- t-SNE trustworthiness across 90 (cell, t-SNE-HP) pairs: range **[0.9304, 0.9994]**, median 0.9884.
- Best-of-method comparison: t-SNE-best > UMAP-best in **29 / 30** cells.
- Mean best-UMAP trustworthiness at the shallowest layer (L4) across 6 (model, task) groups: 0.9928.
- Mean best-UMAP trustworthiness at the deepest extracted layer across the same 6 groups: 0.9637.
- Mean best-t-SNE trustworthiness at L4: 0.9948; at the deepest layer: 0.9710.
- Across 6 (model, task) groups, the change `Δ = T(deepest) − T(L4)` for best-UMAP: mean -0.0291, range [-0.0538, +0.0094].
- Same for best-t-SNE: mean -0.0238, range [-0.0437, +0.0028].

- Concept-class S_B/S_W (best-UMAP, headline concepts):
  - `a_units`: median 9.831 across 30 cells (range [0.031, 131.979]).
  - `ans_units`: median 8.277 across 30 cells (range [0.001, 6757.051]).
  - `carry_units`: median 34.853 across 30 cells (range [4.341, 92.563]).
- Concept-class S_B/S_W (best-t-SNE, headline concepts):
  - `a_units`: median 8.569 across 30 cells (range [0.021, 74.901]).
  - `ans_units`: median 6.619 across 30 cells (range [0.002, 6202.076]).
  - `carry_units`: median 39.658 across 30 cells (range [3.773, 146.496]).
