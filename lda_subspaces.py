"""Step 6 — LDA refinement of CCSVD subspaces, plus a full-space audit.

Per cell (model, task, layer, concept, residualization-mode):

  Option A  — LDA inside the CCSVD subspace (HEADLINE).
              N/r ≈ 100+ → eigenvalues trustworthy.
              alpha·I regularization on S_T_z, or Ledoit-Wolf when N/r < 10.

  Option B  — LDA in the full 4096-D residualized activation space (AUDIT).
              N/d can be < 1 → eigenvalue MAGNITUDES are not cited.
              Top directions and n_sig are cited; cosine similarity vs A is the
              primary cross-check. Ledoit-Wolf shrinkage on S_T mandatory.

Both placements use the K×K compact form:
    S_T^{-1} S_B  has top K-1 non-zero eigenvalues (S_B has rank ≤ K-1).
    Solve  X = S_T^{-1} M_w^T  via cached Cholesky.
    Form   A_kk = M_w @ X      (K × K).
    Eigendecompose A_kk → λ_T (LDA eigenvalues, ∈ [0, 1] when scaled by S_T).
    Directions in dim-space: W = X @ V_kk.

Significance: dual criterion.
    n_sig_perm — sequential 99th-percentile permutation null (1000 shuffles).
    n_sig_cv   — one-SE rule on 5-fold k-NN classification accuracy.
    n_sig      = min(n_sig_perm, n_sig_cv).

Output trees (per mode):
    data/results/lda_subspaces/subspace_lda/mode_{mode}/{model}/...
    data/results/lda_subspaces/full_lda/mode_{mode}/{model}/...

Run on full data — no subsampling. See plan §2 standing rules.

Usage:
  python lda_subspaces.py --config /home/anshulk/emnlp2026/config.yaml \
      --model gpt-j-6b --mode off
  python lda_subspaces.py --config ... --model gpt-j-6b --mode answer
"""

import argparse
import hashlib
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from logging.handlers import WatchedFileHandler
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.linalg import cho_factor, cho_solve, eigh as eigh_sym
from sklearn.covariance import ledoit_wolf
from sklearn.model_selection import StratifiedKFold

# Optional GPU bits — wrapped so import failures fall back cleanly.
_HAS_CUPY = False
_HAS_CUML_KNN = False
try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    cp = None
try:
    from cuml.neighbors import KNeighborsClassifier as cuKNN
    _HAS_CUML_KNN = True
except Exception:
    cuKNN = None

# Reuse existing CCSVD utilities.
from ccsvd_subspaces import (
    JOINT_REGISTRY,
    SKIP_COLUMNS,
    build_label_array,
    cell_seed,
    centroid_matrix,
    concept_columns,
    enumerate_single_concepts,
    encode_labels,
    safe_concept_name,
    sha256_of,
)


# ───────────────────────────────────────────────────────────────────────────────
# Logging
# ───────────────────────────────────────────────────────────────────────────────

def setup_logging(logs_root: Path, model_key: str, mode: str) -> logging.Logger:
    logs_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"lda_subspaces.{model_key}.{mode}")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = WatchedFileHandler(logs_root / f"lda_subspaces_{model_key}_{mode}.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


# ───────────────────────────────────────────────────────────────────────────────
# LDA core (K×K compact form)
# ───────────────────────────────────────────────────────────────────────────────

def compute_centroids_from_codes(X: np.ndarray, y_codes: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-class centroid mean + count. X (N, dim) float32; y_codes (N,) int."""
    n = X.shape[0]
    counts = np.bincount(y_codes, minlength=K).astype(np.int64)
    one_hot = np.zeros((n, K), dtype=np.float32)
    one_hot[np.arange(n), y_codes] = 1.0
    sums = one_hot.T @ X
    means = sums / counts[:, None].clip(min=1)
    return means.astype(np.float64), counts


def build_M_w(centroids: np.ndarray, mu_bar: np.ndarray, n_v: np.ndarray) -> np.ndarray:
    """M_w[k] = sqrt(n_v[k]) * (centroids[k] - mu_bar). Shape (K, dim) float64."""
    weights = np.sqrt(n_v.astype(np.float64))
    return weights[:, None] * (centroids.astype(np.float64) - mu_bar.astype(np.float64))


def solve_with_chol(c_and_lower, B: np.ndarray) -> np.ndarray:
    """Solve A X = B given A = chol_factor (cho_factor result). Wraps cho_solve."""
    return cho_solve(c_and_lower, B)


def compact_lda_eigen(M_w: np.ndarray, c_and_lower) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K×K compact LDA. Returns (lambdas_T, V_kk, X_solve).

    M_w: (K, dim) float64
    c_and_lower: cached Cholesky factor of S_T (or shrunk version).
    Returns:
        lambdas_T:   (K,) sorted descending
        V_kk:        (K, K) eigenvectors of A_kk, columns aligned to lambdas_T
        X_solve:     (dim, K) = inv(S_T) @ M_w^T
    """
    K, dim = M_w.shape
    X_solve = solve_with_chol(c_and_lower, M_w.T)            # (dim, K)
    A_kk = M_w @ X_solve                                       # (K, K) symmetric (numerically)
    A_kk = 0.5 * (A_kk + A_kk.T)                               # symmetrize against fp drift
    # Use scipy eigh (symmetric eigenproblem) — fast, numerically stable for small K.
    evals, V = eigh_sym(A_kk)
    # Sort descending; eigh returns ascending.
    idx = np.argsort(evals)[::-1]
    return evals[idx], V[:, idx], X_solve


def directions_from_compact(X_solve: np.ndarray, V_kk: np.ndarray) -> np.ndarray:
    """Map K×K eigenvectors to dim-space directions and normalize. Shape (dim, K)."""
    W = X_solve @ V_kk                                         # (dim, K)
    norms = np.linalg.norm(W, axis=0, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return W / norms


# ───────────────────────────────────────────────────────────────────────────────
# S_T builders (with regularization choices)
# ───────────────────────────────────────────────────────────────────────────────

def build_ST_alpha_I(X_centered: np.ndarray, alpha: float) -> tuple[np.ndarray, float, float]:
    """S_T = X_c^T X_c + alpha * trace(S_T)/dim * I. Returns (S_T_reg, used_alpha, lw_shrinkage_or_0)."""
    S_T = X_centered.T.astype(np.float64) @ X_centered.astype(np.float64)
    dim = S_T.shape[0]
    diag_term = alpha * (np.trace(S_T) / max(dim, 1))
    S_T_reg = S_T + diag_term * np.eye(dim, dtype=np.float64)
    return S_T_reg, float(diag_term), 0.0


def build_ST_ledoit_wolf(X_centered: np.ndarray) -> tuple[np.ndarray, float, float]:
    """S_T via Ledoit-Wolf shrinkage (CPU sklearn). Returns (S_T_reg, scaled_diag, lw_shrinkage)."""
    # ledoit_wolf returns sample COVARIANCE = (1/N) (X-mu)^T (X-mu) (with shrinkage).
    # Scatter = N * covariance. We need scatter for the LDA generalized eigenproblem.
    cov_shrunk, lw_alpha = ledoit_wolf(X_centered, assume_centered=True)
    N = X_centered.shape[0]
    S_T_reg = N * cov_shrunk
    return S_T_reg.astype(np.float64), 0.0, float(lw_alpha)


def build_ST_oas_gpu(X_centered: np.ndarray) -> tuple[np.ndarray, float, float]:
    """S_T via Oracle Approximating Shrinkage on GPU.

    OAS (Chen et al. 2010) is the closed-form shrinkage estimator that's
    asymptotically equivalent to Ledoit-Wolf for Gaussian data. Implemented
    purely with cupy gemm + trace ops, so it's ~10x faster than sklearn's
    CPU ledoit_wolf for d=4096.

    Returns (S_T_reg, 0.0, shrinkage). The 0.0 second slot mirrors
    build_ST_ledoit_wolf's return signature.
    """
    if not _HAS_CUPY:
        return build_ST_ledoit_wolf(X_centered)
    X_g = cp.asarray(X_centered, dtype=cp.float64)
    N, d = X_g.shape
    # S_emp = (1/N) X^T X  — sample covariance for centered data.
    S_emp = (X_g.T @ X_g) / N                                   # (d, d)
    trace_S = float(cp.trace(S_emp))
    # trace(S^2) = sum(S * S) — cheaper than another (d, d) matmul.
    trace_S2 = float(cp.sum(S_emp * S_emp))
    mu = trace_S / d
    # Closed-form OAS shrinkage:
    num = (1.0 - 2.0 / d) * trace_S2 + trace_S * trace_S
    den = (N + 1.0 - 2.0 / d) * max(trace_S2 - trace_S * trace_S / d, 1e-30)
    shrinkage = float(min(1.0, max(0.0, num / den)))
    target = mu * cp.eye(d, dtype=cp.float64)
    S_shrunk_cov = (1.0 - shrinkage) * S_emp + shrinkage * target
    # Scatter = N * covariance.
    S_T_reg = N * S_shrunk_cov
    return cp.asnumpy(S_T_reg).astype(np.float64), 0.0, shrinkage


# ───────────────────────────────────────────────────────────────────────────────
# Permutation null with cached S_T Cholesky factor
# ───────────────────────────────────────────────────────────────────────────────

def permutation_null(
    X_centered: np.ndarray,
    y_codes: np.ndarray,
    K: int,
    c_and_lower,
    n_perm: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run n_perm label-shuffles; return null_eigvals (n_perm, K-1).

    For each shuffle:
      shuffled centroids → M_w (K, dim) → solve via cached Cholesky → A_kk →
      eigenvalues. Top K-1 are non-zero (S_B has rank ≤ K-1).
    """
    null = np.zeros((n_perm, K - 1), dtype=np.float64)
    n_v = np.bincount(y_codes, minlength=K).astype(np.int64)
    mu_bar = X_centered.mean(axis=0).astype(np.float64)
    for p in range(n_perm):
        y_shuf = rng.permutation(y_codes)
        centroids, _ = compute_centroids_from_codes(X_centered, y_shuf, K)
        # n_v unchanged under permutation (just relabelling).
        M_w = build_M_w(centroids, mu_bar, n_v)
        evals, _, _ = compact_lda_eigen(M_w, c_and_lower)
        # Top K-1 non-zero (trailing eval = 0 by rank constraint).
        null[p, :] = evals[: K - 1]
    return null


def permutation_null_full_gpu(
    X_centered: np.ndarray,
    y_codes: np.ndarray,
    K: int,
    L_lower: np.ndarray,            # (d, d) lower-triangular Cholesky of S_T
    n_perm: int,
    rng: np.random.Generator,
    batch: int = 20,
) -> np.ndarray:
    """GPU-batched permutation null for full-space LDA.

    The Cholesky factor L is invariant under label permutation (S_T does not
    depend on labels), so we ship it to GPU once and reuse across all 1000
    shuffles. Each shuffle: rebuild K class means on GPU (vectorised across
    a batch of `batch` shuffles), solve via L, form the K×K compact form,
    eigendecompose on CPU (K is small).
    """
    if not _HAS_CUPY:
        # No cupy → fall back to scipy CPU path, which is much slower for d=4096.
        from scipy.linalg import solve_triangular as _st_cpu
        null = np.zeros((n_perm, K - 1), dtype=np.float64)
        n_v = np.bincount(y_codes, minlength=K).astype(np.int64)
        weights = np.sqrt(n_v.astype(np.float64))
        mu_bar = X_centered.mean(axis=0).astype(np.float64)
        for p in range(n_perm):
            y_shuf = rng.permutation(y_codes)
            means = np.zeros((K, X_centered.shape[1]), dtype=np.float64)
            for k in range(K):
                m = y_shuf == k
                if m.sum() > 0:
                    means[k] = X_centered[m].mean(axis=0)
            M_w = weights[:, None] * (means - mu_bar)
            Y = _st_cpu(L_lower, M_w.T, lower=True)
            X_solve = _st_cpu(L_lower.T, Y, lower=False)
            A_kk = M_w @ X_solve
            A_kk = 0.5 * (A_kk + A_kk.T)
            evals = np.sort(eigh_sym(A_kk, eigvals_only=True))[::-1]
            null[p, :] = evals[: K - 1]
        return null

    from cupyx.scipy.linalg import solve_triangular as _st_gpu
    X_g = cp.asarray(X_centered, dtype=cp.float64)              # (N, d)
    L_g = cp.asarray(L_lower, dtype=cp.float64)                 # (d, d)
    N, d = X_g.shape
    n_v_int = np.bincount(y_codes, minlength=K).astype(np.int64)
    n_v_g = cp.asarray(n_v_int.astype(np.float64))
    weights_g = cp.sqrt(n_v_g)
    mu_bar = X_g.mean(axis=0)

    null = np.zeros((n_perm, K - 1), dtype=np.float64)
    p_done = 0
    while p_done < n_perm:
        bs = min(batch, n_perm - p_done)
        shuf = np.stack([rng.permutation(y_codes) for _ in range(bs)]).astype(np.int64)
        shuf_g = cp.asarray(shuf)                               # (bs, N)
        # One-hot: (bs, N, K). Vectorised scatter — flatten then assign.
        oh = cp.zeros((bs, N, K), dtype=cp.float64)
        idx_b = cp.broadcast_to(cp.arange(bs)[:, None], (bs, N))
        idx_n = cp.broadcast_to(cp.arange(N)[None, :], (bs, N))
        oh[idx_b.ravel(), idx_n.ravel(), shuf_g.ravel()] = 1.0
        # Class sums: (bs, K, d).
        sums = cp.einsum("bnk,nd->bkd", oh, X_g)
        means = sums / n_v_g[None, :, None].clip(1.0)
        # M_w = sqrt(n_v) * (means - mu_bar). Broadcast.
        M_w_batch = weights_g[None, :, None] * (means - mu_bar[None, None, :])    # (bs, K, d)
        # Solve & form K×K per batch row. (Inner loop is K×K eig; cheap.)
        for b in range(bs):
            Y = _st_gpu(L_g, M_w_batch[b].T, lower=True)
            X_solve = _st_gpu(L_g.T, Y, lower=False)             # (d, K)
            A_kk = cp.asnumpy(M_w_batch[b] @ X_solve)
            A_kk = 0.5 * (A_kk + A_kk.T)
            evals = np.sort(eigh_sym(A_kk, eigvals_only=True))[::-1]
            null[p_done + b, :] = evals[: K - 1]
        p_done += bs
    return null


# ───────────────────────────────────────────────────────────────────────────────
# Sequential significance + one-SE rule
# ───────────────────────────────────────────────────────────────────────────────

def sequential_n_sig(real_evals: np.ndarray, threshold_99: np.ndarray) -> int:
    """Walk top-down; stop at first eigenvalue not exceeding its index threshold."""
    n = 0
    L = min(len(real_evals), len(threshold_99))
    for k in range(L):
        if real_evals[k] > threshold_99[k]:
            n = k + 1
        else:
            break
    return n


def n_sig_one_se_rule(cv_curve: np.ndarray) -> int:
    """One-SE rule: largest k where cv_curve[k-1] is within 1 SE of max."""
    if cv_curve.ndim != 1 or cv_curve.size == 0:
        return 0
    max_acc = float(np.nanmax(cv_curve))
    se = float(np.nanstd(cv_curve)) / max(np.sqrt(np.sum(~np.isnan(cv_curve))), 1)
    threshold = max_acc - se
    n = 0
    for k, acc in enumerate(cv_curve, start=1):
        if not np.isfinite(acc):
            continue
        if acc >= threshold:
            n = k
    return n


# ───────────────────────────────────────────────────────────────────────────────
# CV-accuracy via k-NN in LDA-projected space
# ───────────────────────────────────────────────────────────────────────────────

def _knn_predict(Z_train: np.ndarray, y_train: np.ndarray, Z_test: np.ndarray, k: int) -> np.ndarray:
    """k-NN classification. Use cuML when available, otherwise sklearn (lazy import)."""
    if _HAS_CUML_KNN:
        Zt = cp.asarray(Z_train.astype(np.float32))
        yt = cp.asarray(y_train.astype(np.int32))
        Ze = cp.asarray(Z_test.astype(np.float32))
        knn = cuKNN(n_neighbors=k)
        knn.fit(Zt, yt)
        preds = knn.predict(Ze)
        return cp.asnumpy(preds).astype(np.int64)
    else:
        from sklearn.neighbors import KNeighborsClassifier as skKNN
        knn = skKNN(n_neighbors=k, n_jobs=1)
        knn.fit(Z_train, y_train)
        return knn.predict(Z_test).astype(np.int64)


def cv_accuracy_curve(
    X_centered: np.ndarray,
    y_codes: np.ndarray,
    K: int,
    n_splits: int,
    knn_k: int,
    random_state: int,
    s_t_builder,                     # callable: (X_train_centered) -> (S_T, _, _)
    n_dirs_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per direction-count k=1..K-1, compute mean k-NN held-out accuracy.

    Returns (curve, per_fold) where:
        curve     = (K-1,) mean accuracy across folds for each k
        per_fold  = (n_splits, K-1) per-fold accuracies
    """
    K_minus_1 = max(K - 1, 1)
    n_dirs_max = min(n_dirs_max, K_minus_1)
    per_fold = np.full((n_splits, K_minus_1), np.nan, dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fi, (train_idx, test_idx) in enumerate(skf.split(X_centered, y_codes)):
        X_tr = X_centered[train_idx]
        y_tr = y_codes[train_idx]
        X_te = X_centered[test_idx]
        y_te = y_codes[test_idx]
        # Each surviving class must appear in both splits.
        n_v_tr = np.bincount(y_tr, minlength=K)
        n_v_te = np.bincount(y_te, minlength=K)
        usable = (n_v_tr > 0) & (n_v_te > 0)
        if usable.sum() < 2:
            continue
        # Train LDA on the training fold.
        mu_tr = X_tr.mean(axis=0).astype(np.float64)
        X_tr_c = X_tr.astype(np.float64) - mu_tr
        # Build S_T_train (with the same regularization choice as the main fit).
        S_T_tr, _, _ = s_t_builder(X_tr_c)
        try:
            cf_tr = cho_factor(S_T_tr, lower=True)
        except np.linalg.LinAlgError:
            continue
        centroids_tr, n_v_tr_int = compute_centroids_from_codes(X_tr.astype(np.float32), y_tr, K)
        M_w_tr = build_M_w(centroids_tr, mu_tr, n_v_tr_int)
        evals, V_kk, X_solve = compact_lda_eigen(M_w_tr, cf_tr)
        W_tr = directions_from_compact(X_solve, V_kk)            # (dim, K)
        # Project both folds.
        Z_tr_full = (X_tr.astype(np.float64) - mu_tr) @ W_tr     # (N_tr, K)
        Z_te_full = (X_te.astype(np.float64) - mu_tr) @ W_tr     # (N_te, K)
        for k in range(1, K_minus_1 + 1):
            Z_tr = Z_tr_full[:, :k]
            Z_te = Z_te_full[:, :k]
            try:
                preds = _knn_predict(Z_tr.astype(np.float32), y_tr, Z_te.astype(np.float32), k=knn_k)
                acc = float((preds == y_te).mean())
            except Exception:
                acc = np.nan
            per_fold[fi, k - 1] = acc
    curve = np.nanmean(per_fold, axis=0)
    return curve, per_fold


# ───────────────────────────────────────────────────────────────────────────────
# Cohen's d per direction, all class pairs
# ───────────────────────────────────────────────────────────────────────────────

def cohens_d_per_direction(
    X_centered: np.ndarray, y_codes: np.ndarray, K: int, W: np.ndarray, n_dirs: int,
) -> np.ndarray:
    """Returns (n_dirs, K, K) Cohen's d per direction × class-pair."""
    out = np.zeros((max(n_dirs, 0), K, K), dtype=np.float64)
    if n_dirs <= 0:
        return out
    Z = X_centered.astype(np.float64) @ W[:, :n_dirs]    # (N, n_dirs)
    for d in range(n_dirs):
        z = Z[:, d]
        means = np.zeros(K, dtype=np.float64)
        stds = np.zeros(K, dtype=np.float64)
        for k in range(K):
            mask = y_codes == k
            if mask.sum() >= 2:
                means[k] = float(z[mask].mean())
                stds[k] = float(z[mask].std(ddof=1))
        for i in range(K):
            for j in range(K):
                if i == j:
                    out[d, i, j] = 0.0
                    continue
                pooled_var = 0.5 * (stds[i] ** 2 + stds[j] ** 2)
                if pooled_var > 1e-12:
                    out[d, i, j] = (means[i] - means[j]) / np.sqrt(pooled_var)
    return out


# ───────────────────────────────────────────────────────────────────────────────
# Bootstrap CI on top eigenvalue
# ───────────────────────────────────────────────────────────────────────────────

def bootstrap_lambda1(
    X_centered: np.ndarray, y_codes: np.ndarray, K: int, s_t_builder, n_boot: int, rng: np.random.Generator,
) -> np.ndarray:
    """Resample rows with replacement, refit LDA, capture λ_T_1. Returns (n_boot,)."""
    out = np.full(n_boot, np.nan, dtype=np.float64)
    N = X_centered.shape[0]
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        X_b = X_centered[idx]
        y_b = y_codes[idx]
        # Need each class still represented.
        n_v_b = np.bincount(y_b, minlength=K)
        if (n_v_b == 0).any():
            continue
        mu_b = X_b.mean(axis=0).astype(np.float64)
        X_b_c = X_b.astype(np.float64) - mu_b
        try:
            S_T_b, _, _ = s_t_builder(X_b_c)
            cf_b = cho_factor(S_T_b, lower=True)
            centroids_b, _ = compute_centroids_from_codes(X_b.astype(np.float32), y_b, K)
            M_w_b = build_M_w(centroids_b, mu_b, n_v_b.astype(np.int64))
            evals, _, _ = compact_lda_eigen(M_w_b, cf_b)
            out[b] = float(evals[0]) if len(evals) else np.nan
        except Exception:
            continue
    return out


# ───────────────────────────────────────────────────────────────────────────────
# Per-cell driver — runs both A and B
# ───────────────────────────────────────────────────────────────────────────────

def fit_one_cell(
    *,
    X_correct_resid: np.ndarray,        # (N_correct, d) residualized activations, correctness-masked
    y: np.ndarray,                      # (N_correct,) raw concept labels
    cell_id: dict,
    cfg_lda: dict,
    B_ccsvd: np.ndarray | None,         # (d, r) CCSVD basis for this cell, or None if missing/r=0
    is_carved_out: bool,
    full_space_chol_cache: dict | None, # {"cf": cho_factor result, "mu": (d,) global mean, "lw_alpha": float}
) -> dict:
    """Fit Option A and Option B for a single cell. Returns nested dict with all artifacts."""

    t0 = time.time()
    seed = cell_seed(cell_id["model_key"], cell_id["task"], cell_id["layer"],
                     cell_id["concept_name"], base_seed=cfg_lda["random_state"])

    # ── Filter by min_group_size ──────────────────────────────────────────────
    y_codes, keep_mask, keep_values, dropped = encode_labels(y, cfg_lda["min_samples_per_class"])
    K = len(keep_values)
    n_groups_total = pd.Series(y).nunique()

    base_meta = {
        "cell_id": cell_id,
        "is_carved_out": bool(is_carved_out),
        "n_total_correct": int(len(y)),
        "n_after_filter": int(len(y_codes)),
        "n_groups_total": int(n_groups_total),
        "n_groups_after_filter": int(K),
        "kept_values": [str(v) for v in keep_values],
        "dropped_values": [str(v) for v in dropped],
        "dropped_values_count": int(len(dropped)),
        "seed": int(seed),
    }

    if K < cfg_lda["min_classes_for_lda"]:
        base_meta["status"] = "skipped_insufficient_groups"
        base_meta["runtime_seconds"] = round(time.time() - t0, 3)
        return {"A": dict(base_meta, placement="A"), "B": dict(base_meta, placement="B"),
                "A_artifacts": None, "B_artifacts": None}

    X_f = X_correct_resid[keep_mask]
    N = X_f.shape[0]
    n_v = np.bincount(y_codes, minlength=K).astype(np.int64)

    # Heads-up flags.
    nv_max, nv_min = int(n_v.max()), int(n_v.min())
    group_imbalance_ratio = nv_max / nv_min if nv_min > 0 else float("inf")
    flag_group_imbalance = bool(group_imbalance_ratio > 3)

    # ══════════════════════════════════════════════════════════════════════════
    # Option A — LDA in the CCSVD subspace
    # ══════════════════════════════════════════════════════════════════════════
    A_meta = dict(base_meta, placement="A")
    A_artifacts = None

    if B_ccsvd is None or B_ccsvd.shape[1] == 0:
        A_meta["status"] = "skipped_no_subspace"
        A_meta["lambda_T_1"] = float("nan")
        A_meta["n_sig_perm"] = 0
        A_meta["n_sig_cv"] = 0
        A_meta["n_sig"] = 0
    else:
        r = int(B_ccsvd.shape[1])
        # Project residualized activations into the CCSVD subspace.
        Z = (X_f.astype(np.float64) @ B_ccsvd.astype(np.float64)).astype(np.float32)
        Z_mu = Z.mean(axis=0).astype(np.float64)
        Z_c = Z.astype(np.float64) - Z_mu

        # Pick S_T builder: alpha·I when N/r ≥ threshold, Ledoit-Wolf otherwise.
        n_over_r = N / max(r, 1)
        use_lw_A = (n_over_r < cfg_lda["use_shrinkage_when_n_over_r_below"])
        if use_lw_A:
            s_t_builder_A = build_ST_ledoit_wolf
        else:
            alpha = cfg_lda["regularisation_alpha"]
            s_t_builder_A = lambda X_c: build_ST_alpha_I(X_c, alpha=alpha)

        try:
            S_T_z, used_diag, lw_a = s_t_builder_A(Z_c)
            cf_z = cho_factor(S_T_z, lower=True)
        except Exception as e:
            A_meta["status"] = f"failed_chol_A:{type(e).__name__}"
            A_meta["lambda_T_1"] = float("nan")
            A_meta["n_sig_perm"] = 0
            A_meta["n_sig_cv"] = 0
            A_meta["n_sig"] = 0
        else:
            centroids_z, _ = compute_centroids_from_codes(Z, y_codes, K)
            M_w_z = build_M_w(centroids_z, Z_mu, n_v)
            evals_A, V_A, X_solve_A = compact_lda_eigen(M_w_z, cf_z)
            W_z = directions_from_compact(X_solve_A, V_A)        # (r, K)
            # Permutation null (A is small-dim → CPU is fine).
            rng_A = np.random.default_rng(seed)
            null_A = permutation_null(Z_c, y_codes, K, cf_z, cfg_lda["n_permutations"], rng_A)
            thr99_A = np.percentile(null_A, 100.0 * (1.0 - cfg_lda["perm_alpha"]), axis=0)
            n_sig_perm_A = sequential_n_sig(evals_A[: K - 1], thr99_A)
            # CV-accuracy curve (use first knn_k).
            knn_k_A = cfg_lda["cv_knn_k"][0]
            curve_A, per_fold_A = cv_accuracy_curve(
                Z_c, y_codes, K,
                n_splits=cfg_lda["cv_n_splits"], knn_k=knn_k_A,
                random_state=cfg_lda["random_state"],
                s_t_builder=s_t_builder_A,
                n_dirs_max=K - 1,
            )
            n_sig_cv_A = n_sig_one_se_rule(curve_A) if cfg_lda["use_one_se_rule_for_n_sig_cv"] else int(np.argmax(curve_A) + 1)
            n_sig_A = min(n_sig_perm_A, n_sig_cv_A)
            cohen_A = cohens_d_per_direction(Z_c, y_codes, K, W_z, n_dirs=max(n_sig_A, 1))
            # Bootstrap λ_T_1.
            rng_B = np.random.default_rng(seed + 7)
            boot_A = bootstrap_lambda1(Z_c, y_codes, K, s_t_builder_A, cfg_lda["bootstrap_n"], rng_B)

            # Lift A's directions to 4096-D via the CCSVD basis.
            W_full_A = (B_ccsvd.astype(np.float64) @ W_z[:, : max(n_sig_A, 1)]).astype(np.float32)
            # Re-normalize after lifting (rounding / float drift).
            norms = np.linalg.norm(W_full_A, axis=0, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            W_full_A = W_full_A / norms

            A_meta["status"] = "fit_ok" if n_sig_A > 0 else "no_significant_lda_dir"
            A_meta["r_ccsvd"] = int(r)
            A_meta["N_over_r"] = float(n_over_r)
            A_meta["used_shrinkage"] = bool(use_lw_A)
            A_meta["used_alpha_diag"] = float(used_diag)
            A_meta["lw_shrinkage"] = float(lw_a)
            A_meta["lambda_T"] = evals_A[: K - 1].tolist()
            A_meta["lambda_W"] = [float(l / (1 - l)) if (1 - l) > 1e-12 else float("inf") for l in evals_A[: K - 1]]
            A_meta["lambda_T_1"] = float(evals_A[0]) if len(evals_A) else float("nan")
            A_meta["lambda_T_2"] = float(evals_A[1]) if len(evals_A) > 1 else float("nan")
            A_meta["lambda_T_1_over_2"] = float(evals_A[0] / evals_A[1]) if len(evals_A) > 1 and evals_A[1] > 1e-12 else float("nan")
            A_meta["n_sig_perm"] = int(n_sig_perm_A)
            A_meta["n_sig_cv"] = int(n_sig_cv_A)
            A_meta["n_sig"] = int(n_sig_A)
            A_meta["cv_accuracy_max"] = float(np.nanmax(curve_A)) if curve_A.size and not np.all(np.isnan(curve_A)) else float("nan")
            A_meta["cv_accuracy_at_n_sig"] = float(curve_A[n_sig_A - 1]) if n_sig_A >= 1 and n_sig_A <= len(curve_A) else float("nan")
            A_meta["cv_accuracy_random_baseline"] = float(1.0 / K)
            A_meta["bootstrap_lambda1_mean"] = float(np.nanmean(boot_A))
            A_meta["bootstrap_lambda1_p5"] = float(np.nanpercentile(boot_A, 5))
            A_meta["null_p99_top1"] = float(thr99_A[0]) if len(thr99_A) else float("nan")
            A_meta["flag_group_imbalance"] = flag_group_imbalance
            A_meta["max_n_v"] = nv_max
            A_meta["min_n_v"] = nv_min
            A_meta["group_imbalance_ratio"] = float(group_imbalance_ratio)

            A_artifacts = dict(
                lda_basis_subspace=W_z[:, : max(n_sig_A, 1)].astype(np.float32),
                lda_basis_full=W_full_A.astype(np.float32),
                lda_eigenvalues=evals_A.astype(np.float64),
                null_lda_eigenvalues=null_A.astype(np.float64),
                lda_threshold_99=thr99_A.astype(np.float64),
                cohen_d=cohen_A.astype(np.float64),
                cv_accuracy_curve=curve_A.astype(np.float64),
                cv_per_fold=per_fold_A.astype(np.float64),
                bootstrap_lambda1=boot_A.astype(np.float64),
            )

    # ══════════════════════════════════════════════════════════════════════════
    # Option B — LDA in the full residualized 4096-D space
    # ══════════════════════════════════════════════════════════════════════════
    B_meta = dict(base_meta, placement="B")
    B_artifacts = None

    if full_space_chol_cache is None:
        # If no cache passed, build it locally (slower; per-cell).
        try:
            mu_full = X_f.mean(axis=0).astype(np.float64)
            X_f_c = X_f.astype(np.float64) - mu_full
            S_T_full, _, lw_full = build_ST_oas_gpu(X_f_c) if _HAS_CUPY else build_ST_ledoit_wolf(X_f_c)
            cf_full = cho_factor(S_T_full, lower=True)
            L_lower_full = np.tril(cf_full[0]).astype(np.float64)
        except Exception as e:
            B_meta["status"] = f"failed_chol_B_local:{type(e).__name__}"
            B_meta["lambda_T_1"] = float("nan")
            B_meta["n_sig_perm"] = 0
            B_meta["n_sig_cv"] = 0
            B_meta["n_sig"] = 0
            return {"A": A_meta, "B": B_meta, "A_artifacts": A_artifacts, "B_artifacts": None}
    else:
        cf_full = full_space_chol_cache["cf"]
        mu_full = full_space_chol_cache["mu"]
        lw_full = float(full_space_chol_cache["lw_alpha"])
        L_lower_full = full_space_chol_cache["L_lower"]

    try:
        X_f_c = X_f.astype(np.float64) - mu_full
        centroids_full, _ = compute_centroids_from_codes(X_f, y_codes, K)
        M_w_full = build_M_w(centroids_full, mu_full, n_v)
        evals_B, V_B, X_solve_B = compact_lda_eigen(M_w_full, cf_full)
        W_full_B = directions_from_compact(X_solve_B, V_B)        # (d, K)
        # Permutation null for B — GPU-batched if cupy available, CPU fallback.
        rng_Bp = np.random.default_rng(seed + 13)
        null_B = permutation_null_full_gpu(
            X_f_c, y_codes, K, L_lower_full, cfg_lda["n_permutations"], rng_Bp,
        )
        thr99_B = np.percentile(null_B, 100.0 * (1.0 - cfg_lda["perm_alpha"]), axis=0)
        n_sig_perm_B = sequential_n_sig(evals_B[: K - 1], thr99_B)
        # CV-accuracy curve (rebuilds S_T per fold inside, using GPU OAS when available).
        s_t_builder_B = build_ST_oas_gpu if _HAS_CUPY else build_ST_ledoit_wolf
        knn_k_B = cfg_lda["cv_knn_k"][0]
        curve_B, per_fold_B = cv_accuracy_curve(
            X_f_c, y_codes, K,
            n_splits=cfg_lda["cv_n_splits"], knn_k=knn_k_B,
            random_state=cfg_lda["random_state"],
            s_t_builder=s_t_builder_B,
            n_dirs_max=K - 1,
        )
        n_sig_cv_B = n_sig_one_se_rule(curve_B) if cfg_lda["use_one_se_rule_for_n_sig_cv"] else int(np.argmax(curve_B) + 1)
        n_sig_B = min(n_sig_perm_B, n_sig_cv_B)
        cohen_B = cohens_d_per_direction(X_f_c, y_codes, K, W_full_B, n_dirs=max(n_sig_B, 1))

        B_meta["status"] = "fit_ok" if n_sig_B > 0 else "no_significant_lda_dir"
        B_meta["N"] = int(N)
        B_meta["d"] = int(X_f.shape[1])
        B_meta["N_over_d"] = float(N / X_f.shape[1])
        B_meta["used_shrinkage"] = True
        B_meta["lw_shrinkage"] = float(lw_full)
        B_meta["lambda_T"] = evals_B[: K - 1].tolist()
        B_meta["lambda_T_1"] = float(evals_B[0]) if len(evals_B) else float("nan")
        B_meta["lambda_T_2"] = float(evals_B[1]) if len(evals_B) > 1 else float("nan")
        B_meta["n_sig_perm"] = int(n_sig_perm_B)
        B_meta["n_sig_cv"] = int(n_sig_cv_B)
        B_meta["n_sig"] = int(n_sig_B)
        B_meta["cv_accuracy_max"] = float(np.nanmax(curve_B)) if curve_B.size and not np.all(np.isnan(curve_B)) else float("nan")
        B_meta["cv_accuracy_at_n_sig"] = float(curve_B[n_sig_B - 1]) if n_sig_B >= 1 and n_sig_B <= len(curve_B) else float("nan")
        B_meta["cv_accuracy_random_baseline"] = float(1.0 / K)
        B_meta["null_p99_top1"] = float(thr99_B[0]) if len(thr99_B) else float("nan")
        B_meta["flag_group_imbalance"] = flag_group_imbalance

        B_artifacts = dict(
            lda_basis_full=W_full_B[:, : max(n_sig_B, 1)].astype(np.float32),
            lda_eigenvalues=evals_B.astype(np.float64),
            null_lda_eigenvalues=null_B.astype(np.float64),
            lda_threshold_99=thr99_B.astype(np.float64),
            cohen_d=cohen_B.astype(np.float64),
            cv_accuracy_curve=curve_B.astype(np.float64),
            cv_per_fold=per_fold_B.astype(np.float64),
        )
    except Exception as e:
        B_meta["status"] = f"failed_B_inner:{type(e).__name__}"
        B_meta["lambda_T_1"] = float("nan")
        B_meta["n_sig_perm"] = 0
        B_meta["n_sig_cv"] = 0
        B_meta["n_sig"] = 0

    # ── A vs B alignment ──────────────────────────────────────────────────────
    cos_sim_AB = float("nan")
    audit_status = "unknown"
    if A_artifacts is not None and B_artifacts is not None:
        try:
            wA = A_artifacts["lda_basis_full"][:, 0].astype(np.float64)
            wB = B_artifacts["lda_basis_full"][:, 0].astype(np.float64)
            cos_sim_AB = float(abs(np.dot(wA, wB)) / (np.linalg.norm(wA) * np.linalg.norm(wB) + 1e-12))
            if cos_sim_AB >= 0.9:
                audit_status = "agree"
            elif cos_sim_AB >= 0.7:
                audit_status = "partial"
            else:
                audit_status = "ambiguous_AB"
            # Promote to ccsvd_incomplete if B finds many more directions than A.
            if A_meta.get("n_sig", 0) > 0 and B_meta.get("n_sig", 0) >= 2 * A_meta["n_sig"]:
                audit_status = "ccsvd_incomplete"
        except Exception:
            pass

    A_meta["cos_sim_AB"] = cos_sim_AB
    A_meta["audit_status"] = audit_status
    B_meta["cos_sim_AB"] = cos_sim_AB
    B_meta["audit_status"] = audit_status

    A_meta["runtime_seconds"] = round(time.time() - t0, 3)
    B_meta["runtime_seconds"] = round(time.time() - t0, 3)
    return {"A": A_meta, "B": B_meta, "A_artifacts": A_artifacts, "B_artifacts": B_artifacts}


# ───────────────────────────────────────────────────────────────────────────────
# I/O helpers — write per-cell artifacts atomically
# ───────────────────────────────────────────────────────────────────────────────

def atomic_save(arr: np.ndarray, path: Path):
    """Atomic write: tempfile in same directory, then os.replace.

    np.save appends '.npy' to its argument if absent, so the tempfile suffix
    is '.npy' to avoid double-extension; the rename target preserves the
    caller's chosen filename.
    """
    import os
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(suffix=".npy", dir=str(path.parent))
    os.close(fd)
    np.save(tmp_str, arr)
    os.replace(tmp_str, path)


def write_cell_outputs(cell_dir: Path, meta: dict, artifacts: dict | None):
    cell_dir.mkdir(parents=True, exist_ok=True)
    if artifacts is not None:
        for k, v in artifacts.items():
            atomic_save(v, cell_dir / f"{k}.npy")
    with open(cell_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


# ───────────────────────────────────────────────────────────────────────────────
# Carve-out logic
# ───────────────────────────────────────────────────────────────────────────────

def is_concept_carved_out(concept_name: str, mode: str, cfg_lda: dict) -> bool:
    if mode == "answer":
        prefixes = cfg_lda.get("ans_concept_prefixes", ["ans_", "answer"])
        for p in prefixes:
            if concept_name == p.rstrip("_") or concept_name.startswith(p):
                return True
    if mode == "norm":
        carveouts = cfg_lda.get("norm_carveout_concepts", ["ans_magnitude_tier"])
        if concept_name in carveouts:
            return True
    return False


# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["off", "answer", "norm"],
        help="Residualization mode. Determines which CCSVD tree and which "
             "residualized activations to read.",
    )
    parser.add_argument("--single-task", default=None)
    parser.add_argument("--single-layer", type=int, default=None)
    parser.add_argument("--single-concept", default=None)
    parser.add_argument("--limit-concepts", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip cells whose meta.json already shows status='fit_ok' or 'no_significant_lda_dir'.")
    parser.add_argument("--skip-full-space", action="store_true",
                        help="DEBUG: only run Option A; skip Option B fitter (saves time during smoke tests).")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    paths = cfg["paths"]
    cfg_lda = cfg["lda"]

    logs_root = Path(paths["logs_root"])
    logger = setup_logging(logs_root, args.model, args.mode)

    model_cfg = next((m for m in cfg["models"] if m["key"] == args.model), None)
    if model_cfg is None:
        raise ValueError(f"Model {args.model} not in config")

    results_root = Path(paths["results_root"])
    data_root = Path(paths["data_root"])

    # CCSVD basis location: mode_off uses original tree, modes answer/norm use mode_X subtree.
    if args.mode == "off":
        ccsvd_root = results_root / "ccsvd_subspaces"
    else:
        ccsvd_root = results_root / "ccsvd_subspaces" / f"mode_{args.mode}"

    # Output trees (always per-mode, always per-placement).
    A_root = results_root / "lda_subspaces" / "subspace_lda" / f"mode_{args.mode}"
    B_root = results_root / "lda_subspaces" / "full_lda" / f"mode_{args.mode}"
    A_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_full_space:
        B_root.mkdir(parents=True, exist_ok=True)

    # Activation source: residualized cache for non-off, original for off.
    if args.mode == "off":
        act_root = data_root / "activations"
    else:
        act_root = results_root / "residualized"

    logger.info("=" * 78)
    logger.info("LDA subspaces — model=%s mode=%s", args.model, args.mode)
    logger.info("ccsvd_root: %s", ccsvd_root)
    logger.info("A_root:     %s", A_root)
    logger.info("B_root:     %s", "DISABLED" if args.skip_full_space else B_root)
    logger.info("act_root:   %s", act_root)
    logger.info("cuML kNN:   %s", "available" if _HAS_CUML_KNN else "NOT available — using sklearn")
    logger.info("cupy:       %s", "available" if _HAS_CUPY else "NOT available — CPU only")

    t_run0 = time.time()
    rows_A = []                      # one row per cell — A summary
    rows_B = []                      # one row per cell — B summary
    rows_audit = []                  # one row per cell — A vs B alignment
    eig_rows_A = []                  # one row per (cell × k)
    eig_rows_B = []                  # one row per (cell × k)
    null_rows_A = []                 # one row per (cell × k) with permutation-null percentiles
    null_rows_B = []                 # one row per (cell × k)
    cv_rows_A = []                   # one row per (cell × fold × k)
    cv_rows_B = []                   # one row per (cell × fold × k)
    cohen_rows_A = []                # one row per (cell × direction × i × j)
    cohen_rows_B = []                # one row per (cell × direction × i × j)
    bootstrap_rows_A = []            # one row per (cell × bootstrap_idx)

    tasks = [args.single_task] if args.single_task else ["addition", "multiplication"]
    layers = [args.single_layer] if args.single_layer else model_cfg["layers"]

    for task in tasks:
        labels_path = data_root / "data" / "raw" / f"{task}_problems.csv"
        problems_df = pd.read_csv(labels_path)
        answers_path = data_root / "answers" / args.model / f"{task}_answers.csv"
        answers_df = pd.read_csv(answers_path)
        if len(answers_df) != len(problems_df):
            raise RuntimeError(f"row count mismatch on {task}")
        correct_mask = answers_df["correct"].to_numpy().astype(bool)

        singles = enumerate_single_concepts(problems_df)
        joints = JOINT_REGISTRY.get(task, [])
        if args.single_concept:
            concepts = [args.single_concept]
        else:
            concepts = singles + joints
            if args.limit_concepts:
                concepts = concepts[: args.limit_concepts]

        logger.info("[%s] N_correct=%d singles=%d joints=%d total=%d",
                    task, int(correct_mask.sum()), len(singles), len(joints), len(concepts))

        for layer in layers:
            # Resolve activation path for this (task, layer, mode).
            if args.mode == "off":
                act_path = act_root / args.model / f"{task}_layer_{layer:02d}.npy"
            else:
                act_path = act_root / args.model / f"{task}_layer_{layer:02d}_mode_{args.mode}.npy"
            if not act_path.exists():
                logger.warning("missing activations for layer %d: %s — skipping", layer, act_path)
                continue
            X_full = np.load(act_path)
            X_correct = np.ascontiguousarray(X_full[correct_mask].astype(np.float32))
            del X_full
            d = X_correct.shape[1]

            # Build full-space S_T Cholesky cache once per (task, layer, mode).
            # GPU path: cupy OAS shrinkage + cupy.linalg.cholesky → keep L on GPU.
            full_cache = None
            t_chol = time.time()
            mu_full = X_correct.mean(axis=0).astype(np.float64)
            X_correct_c = X_correct.astype(np.float64) - mu_full
            S_T_full, _, lw_alpha = build_ST_oas_gpu(X_correct_c) if _HAS_CUPY else build_ST_ledoit_wolf(X_correct_c)
            try:
                cf_full = cho_factor(S_T_full, lower=True)
                # Extract clean lower triangle for GPU reuse.
                L_lower = np.tril(cf_full[0]).astype(np.float64)
                full_cache = {
                    "cf": cf_full, "mu": mu_full, "lw_alpha": lw_alpha, "L_lower": L_lower,
                }
                logger.info("  layer %d: full-space S_T+Cholesky cached  d=%d  shrinkage=%.4f  dt=%.1fs  backend=%s",
                            layer, d, lw_alpha, time.time() - t_chol, "GPU" if _HAS_CUPY else "CPU")
            except Exception as e:
                logger.error("  layer %d: full-space Cholesky failed (%s) — Option B disabled for this layer",
                             layer, type(e).__name__)
                full_cache = None

            t_layer = time.time()
            n_processed = 0
            for concept in concepts:
                concept_name = safe_concept_name(concept)
                cell_id = {
                    "model_key": args.model,
                    "task": task,
                    "layer": layer,
                    "mode": args.mode,
                    "concept_name": concept_name,
                    "concept_columns": concept_columns(concept),
                }
                # Validate columns exist.
                missing = [c for c in concept_columns(concept) if c not in problems_df.columns]
                if missing:
                    logger.debug("  skip %s: missing columns %s", concept_name, missing)
                    continue

                # Carve-out check (mode-specific).
                carved = is_concept_carved_out(concept_name, args.mode, cfg_lda)
                if carved:
                    logger.debug("  carve-out: %s under mode=%s", concept_name, args.mode)

                # Resume check.
                A_dir = A_root / args.model / task / f"layer_{layer:02d}" / concept_name
                B_dir = B_root / args.model / task / f"layer_{layer:02d}" / concept_name
                if args.resume:
                    a_meta = A_dir / "meta.json"
                    if a_meta.exists():
                        try:
                            existing = json.loads(a_meta.read_text())
                            if existing.get("status") in ("fit_ok", "no_significant_lda_dir", "skipped_insufficient_groups"):
                                continue
                        except Exception:
                            pass

                # Load CCSVD basis B.
                B_path = ccsvd_root / args.model / task / f"layer_{layer:02d}" / concept_name / "basis.npy"
                B_basis = None
                if carved:
                    # When carved-out we do NOT run A or B. Record an explicit cell.
                    A_meta = {
                        "cell_id": cell_id, "is_carved_out": True, "status": "carved_out",
                        "n_sig_perm": 0, "n_sig_cv": 0, "n_sig": 0,
                        "lambda_T_1": float("nan"), "cv_accuracy_at_n_sig": float("nan"),
                        "cos_sim_AB": float("nan"), "audit_status": "carved_out",
                    }
                    write_cell_outputs(A_dir, A_meta, None)
                    if not args.skip_full_space:
                        B_meta = dict(A_meta, placement="B")
                        write_cell_outputs(B_dir, B_meta, None)
                    rows_A.append(_summary_row(A_meta, placement="A"))
                    if not args.skip_full_space:
                        rows_B.append(_summary_row(dict(A_meta, placement="B"), placement="B"))
                    n_processed += 1
                    continue

                if B_path.exists():
                    try:
                        B_basis = np.load(B_path)
                    except Exception as e:
                        logger.warning("  could not load CCSVD basis at %s (%s) — A will be skipped",
                                       B_path, type(e).__name__)
                        B_basis = None
                else:
                    logger.debug("  no CCSVD basis at %s — A will be skipped", B_path)

                y = build_label_array(problems_df, concept, correct_mask)

                t_cell = time.time()
                res = fit_one_cell(
                    X_correct_resid=X_correct,
                    y=y,
                    cell_id=cell_id,
                    cfg_lda=cfg_lda,
                    B_ccsvd=B_basis,
                    is_carved_out=False,
                    full_space_chol_cache=(None if args.skip_full_space else full_cache),
                )
                dt = time.time() - t_cell

                # Write outputs.
                write_cell_outputs(A_dir, res["A"], res["A_artifacts"])
                if not args.skip_full_space:
                    write_cell_outputs(B_dir, res["B"], res["B_artifacts"])

                rows_A.append(_summary_row(res["A"], placement="A"))
                if not args.skip_full_space:
                    rows_B.append(_summary_row(res["B"], placement="B"))
                # A vs B alignment row.
                rows_audit.append({
                    "model_key": args.model, "task": task, "layer": layer,
                    "concept": concept_name, "mode": args.mode,
                    "n_sig_A": res["A"].get("n_sig", 0),
                    "n_sig_B": res["B"].get("n_sig", 0),
                    "lambda_T_1_A": res["A"].get("lambda_T_1"),
                    "lambda_T_1_B": res["B"].get("lambda_T_1"),
                    "cv_accuracy_at_n_sig_A": res["A"].get("cv_accuracy_at_n_sig"),
                    "cv_accuracy_at_n_sig_B": res["B"].get("cv_accuracy_at_n_sig"),
                    "cos_sim_AB": res["A"].get("cos_sim_AB"),
                    "audit_status": res["A"].get("audit_status"),
                })

                # ── Long-form CSV rows ───────────────────────────────────────
                base_id = dict(model_key=args.model, task=task, layer=layer,
                               concept=concept_name, mode=args.mode)

                def _push_eigs(rows, meta, artifacts, placement_label):
                    if artifacts is None:
                        return
                    evs = artifacts.get("lda_eigenvalues")
                    thr99 = artifacts.get("lda_threshold_99")
                    if evs is None or thr99 is None:
                        return
                    K_minus_1 = min(len(evs), len(thr99))
                    total = float(np.sum(evs[:K_minus_1])) if K_minus_1 else 0.0
                    cum = 0.0
                    for k in range(K_minus_1):
                        lam = float(evs[k])
                        t99 = float(thr99[k])
                        ex = (lam / total) if total > 0 else float("nan")
                        cum += ex if total > 0 else 0.0
                        rows.append(dict(
                            **base_id, placement=placement_label, k=k + 1,
                            lambda_T_k=lam, threshold_99_k=t99,
                            significant=bool(lam > t99),
                            explained_variance_k=ex, cumulative_variance_k=cum,
                        ))

                def _push_nulls(rows, meta, artifacts, placement_label):
                    if artifacts is None:
                        return
                    nulls = artifacts.get("null_lda_eigenvalues")
                    if nulls is None or nulls.size == 0:
                        return
                    K_minus_1 = nulls.shape[1]
                    p50 = np.percentile(nulls, 50, axis=0)
                    p75 = np.percentile(nulls, 75, axis=0)
                    p90 = np.percentile(nulls, 90, axis=0)
                    p95 = np.percentile(nulls, 95, axis=0)
                    p99 = np.percentile(nulls, 99, axis=0)
                    for k in range(K_minus_1):
                        rows.append(dict(
                            **base_id, placement=placement_label, k=k + 1,
                            null_p50=float(p50[k]), null_p75=float(p75[k]),
                            null_p90=float(p90[k]), null_p95=float(p95[k]),
                            null_p99=float(p99[k]),
                        ))

                def _push_cv(rows, meta, artifacts, placement_label):
                    if artifacts is None:
                        return
                    per_fold = artifacts.get("cv_per_fold")
                    curve = artifacts.get("cv_accuracy_curve")
                    if per_fold is None:
                        return
                    n_folds, K_minus_1 = per_fold.shape
                    for fi in range(n_folds):
                        for k in range(K_minus_1):
                            rows.append(dict(
                                **base_id, placement=placement_label,
                                fold=fi + 1, k=k + 1,
                                accuracy=float(per_fold[fi, k]) if not np.isnan(per_fold[fi, k]) else None,
                                accuracy_mean_at_k=float(curve[k]) if curve is not None and k < len(curve) and not np.isnan(curve[k]) else None,
                            ))

                def _push_cohen(rows, meta, artifacts, placement_label):
                    if artifacts is None:
                        return
                    cohen = artifacts.get("cohen_d")
                    if cohen is None or cohen.size == 0:
                        return
                    kept = meta.get("kept_values", [])
                    n_dirs, K, _ = cohen.shape
                    for d_idx in range(n_dirs):
                        for i in range(K):
                            for j in range(K):
                                if i == j:
                                    continue
                                rows.append(dict(
                                    **base_id, placement=placement_label,
                                    direction=d_idx + 1,
                                    class_i=str(kept[i]) if i < len(kept) else str(i),
                                    class_j=str(kept[j]) if j < len(kept) else str(j),
                                    cohen_d=float(cohen[d_idx, i, j]),
                                ))

                def _push_bootstrap(rows, artifacts, placement_label):
                    if artifacts is None:
                        return
                    boot = artifacts.get("bootstrap_lambda1")
                    if boot is None:
                        return
                    for bi, val in enumerate(boot):
                        rows.append(dict(
                            **base_id, placement=placement_label,
                            bootstrap_idx=bi + 1,
                            lambda_T_1_boot=float(val) if not np.isnan(val) else None,
                        ))

                _push_eigs(eig_rows_A, res["A"], res["A_artifacts"], "A")
                _push_nulls(null_rows_A, res["A"], res["A_artifacts"], "A")
                _push_cv(cv_rows_A, res["A"], res["A_artifacts"], "A")
                _push_cohen(cohen_rows_A, res["A"], res["A_artifacts"], "A")
                _push_bootstrap(bootstrap_rows_A, res["A_artifacts"], "A")
                if not args.skip_full_space:
                    _push_eigs(eig_rows_B, res["B"], res["B_artifacts"], "B")
                    _push_nulls(null_rows_B, res["B"], res["B_artifacts"], "B")
                    _push_cv(cv_rows_B, res["B"], res["B_artifacts"], "B")
                    _push_cohen(cohen_rows_B, res["B"], res["B_artifacts"], "B")

                n_processed += 1
                if n_processed % 25 == 0:
                    logger.info("  layer %d  %d/%d concepts done  last cell=%.1fs",
                                layer, n_processed, len(concepts), dt)

            logger.info("  layer %d done in %.1fs (%d concepts)", layer, time.time() - t_layer, n_processed)
            del X_correct
            full_cache = None

    # ── Per-mode aggregate CSVs (full long-form, plot-ready) ─────────────────
    out_dir_A = A_root / args.model
    out_dir_A.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_A).to_csv(out_dir_A / f"summary_{args.model}_mode_{args.mode}.csv", index=False)
    pd.DataFrame(eig_rows_A).to_csv(out_dir_A / f"eigenvalue_spectra_{args.model}_mode_{args.mode}.csv", index=False)
    pd.DataFrame(null_rows_A).to_csv(out_dir_A / f"null_summary_{args.model}_mode_{args.mode}.csv", index=False)
    pd.DataFrame(cv_rows_A).to_csv(out_dir_A / f"cv_per_fold_{args.model}_mode_{args.mode}.csv", index=False)
    pd.DataFrame(cohen_rows_A).to_csv(out_dir_A / f"cohen_d_{args.model}_mode_{args.mode}.csv", index=False)
    pd.DataFrame(bootstrap_rows_A).to_csv(out_dir_A / f"bootstrap_lambda1_{args.model}_mode_{args.mode}.csv", index=False)
    if not args.skip_full_space:
        out_dir_B = B_root / args.model
        out_dir_B.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_B).to_csv(out_dir_B / f"summary_{args.model}_mode_{args.mode}.csv", index=False)
        pd.DataFrame(eig_rows_B).to_csv(out_dir_B / f"eigenvalue_spectra_{args.model}_mode_{args.mode}.csv", index=False)
        pd.DataFrame(null_rows_B).to_csv(out_dir_B / f"null_summary_{args.model}_mode_{args.mode}.csv", index=False)
        pd.DataFrame(cv_rows_B).to_csv(out_dir_B / f"cv_per_fold_{args.model}_mode_{args.mode}.csv", index=False)
        pd.DataFrame(cohen_rows_B).to_csv(out_dir_B / f"cohen_d_{args.model}_mode_{args.mode}.csv", index=False)
    audit_dir = results_root / "lda_subspaces" / "comparison"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_csv = audit_dir / f"a_vs_b_alignment_{args.model}_mode_{args.mode}.csv"
    pd.DataFrame(rows_audit).to_csv(audit_csv, index=False)
    logger.info("CSV outputs:")
    logger.info("  A summary:    %s", out_dir_A / f"summary_{args.model}_mode_{args.mode}.csv")
    logger.info("  A eig spectra: %s", out_dir_A / f"eigenvalue_spectra_{args.model}_mode_{args.mode}.csv")
    logger.info("  A null summ:   %s", out_dir_A / f"null_summary_{args.model}_mode_{args.mode}.csv")
    logger.info("  A cv per-fold: %s", out_dir_A / f"cv_per_fold_{args.model}_mode_{args.mode}.csv")
    logger.info("  A cohen d:     %s", out_dir_A / f"cohen_d_{args.model}_mode_{args.mode}.csv")
    logger.info("  A bootstrap:   %s", out_dir_A / f"bootstrap_lambda1_{args.model}_mode_{args.mode}.csv")
    logger.info("  A vs B align:  %s", audit_csv)

    # Per-(model, mode) manifest.
    manifest = {
        "model_key": args.model,
        "mode": args.mode,
        "n_cells_A": len(rows_A),
        "n_cells_B": len(rows_B) if not args.skip_full_space else 0,
        "n_cells_carved_out": int(sum(1 for r in rows_A if r.get("status") == "carved_out")),
        "n_cells_fit_ok_A": int(sum(1 for r in rows_A if r.get("status") == "fit_ok")),
        "n_cells_fit_ok_B": int(sum(1 for r in rows_B if r.get("status") == "fit_ok")) if not args.skip_full_space else 0,
        "lda_settings": cfg_lda,
        "library_versions": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "sklearn": __import__("sklearn").__version__,
            "cupy_available": _HAS_CUPY,
            "cuml_knn_available": _HAS_CUML_KNN,
            "python": platform.python_version(),
        },
        "config_sha256": sha256_of(Path(args.config)),
        "total_runtime_seconds": round(time.time() - t_run0, 2),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest_path = A_root / args.model / f"manifest_{args.model}_mode_{args.mode}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info("=" * 78)
    logger.info("DONE  total wall=%.1f min  cells_A=%d (fit_ok=%d)  cells_B=%d (fit_ok=%d)",
                (time.time() - t_run0) / 60.0,
                manifest["n_cells_A"], manifest["n_cells_fit_ok_A"],
                manifest["n_cells_B"], manifest["n_cells_fit_ok_B"])


def _summary_row(meta: dict, placement: str) -> dict:
    cid = meta.get("cell_id", {})
    return dict(
        model_key=cid.get("model_key"),
        task=cid.get("task"),
        layer=cid.get("layer"),
        mode=cid.get("mode"),
        concept=cid.get("concept_name"),
        placement=placement,
        status=meta.get("status"),
        is_carved_out=meta.get("is_carved_out"),
        n_total_correct=meta.get("n_total_correct"),
        n_after_filter=meta.get("n_after_filter"),
        n_groups_after_filter=meta.get("n_groups_after_filter"),
        r_ccsvd=meta.get("r_ccsvd"),
        N_over_r=meta.get("N_over_r"),
        N_over_d=meta.get("N_over_d"),
        used_shrinkage=meta.get("used_shrinkage"),
        lw_shrinkage=meta.get("lw_shrinkage"),
        lambda_T_1=meta.get("lambda_T_1"),
        lambda_T_2=meta.get("lambda_T_2"),
        n_sig_perm=meta.get("n_sig_perm"),
        n_sig_cv=meta.get("n_sig_cv"),
        n_sig=meta.get("n_sig"),
        cv_accuracy_max=meta.get("cv_accuracy_max"),
        cv_accuracy_at_n_sig=meta.get("cv_accuracy_at_n_sig"),
        cv_accuracy_random_baseline=meta.get("cv_accuracy_random_baseline"),
        bootstrap_lambda1_p5=meta.get("bootstrap_lambda1_p5"),
        cos_sim_AB=meta.get("cos_sim_AB"),
        audit_status=meta.get("audit_status"),
        flag_group_imbalance=meta.get("flag_group_imbalance"),
        runtime_seconds=meta.get("runtime_seconds"),
    )


if __name__ == "__main__":
    main()
