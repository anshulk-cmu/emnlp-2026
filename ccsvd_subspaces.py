"""Phase 1 — Conditional Covariance + SVD (CCSVD) for concept subspaces.

For each (model, task, layer, concept) cell on the per-model correct subset:
  1. Per-value centroids µ_v on the cell's correct subset.
  2. Scaled centred centroid matrix M[v] = sqrt(n_v / N) * (µ_v − µ̄).
  3. SVD of M → eigenvalues λ_k = s_k² and right singular vectors V_k.
  4. Permutation null (1000 shuffles, sequential 99th-percentile stop) → r.
  5. Basis B = V[:r].T (4096 × r), projections of centroids and full cloud.
  6. 5-fold subspace-preservation CV (Pearson on pairwise centroid distances).

Concepts are read from problems CSV columns plus a curated set of joint tuples.
Run on full data — no subsampling. See plan §2 standing rules.

Usage:
  python ccsvd_subspaces.py --config /home/anshulk/emnlp2026/config.yaml --model gpt-j-6b
"""

import argparse
import hashlib
import json
import logging
import platform
import re
import sys
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler, WatchedFileHandler
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import torch
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr


# ═══════════════════════════════════════════════════════════════════════════════
# GPU SVD HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" or (prefer == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def batched_centroids_torch(X_t: torch.Tensor, y_codes_batch: torch.Tensor, m: int) -> torch.Tensor:
    """Compute centroids for B label assignments in one shot.

    Args:
        X_t: (N, d) float32 on device.
        y_codes_batch: (B, N) int64 on device.
        m: number of classes.

    Returns:
        centroids: (B, m, d) float32 on device.
    """
    B, N = y_codes_batch.shape
    d = X_t.shape[1]
    out = torch.zeros((B, m, d), dtype=X_t.dtype, device=X_t.device)
    counts = torch.zeros((B, m), dtype=torch.float32, device=X_t.device)
    # index_add per batch row (cheap loop on B, vectorised over N×d).
    for b in range(B):
        out[b].index_add_(0, y_codes_batch[b], X_t)
        counts[b].index_add_(0, y_codes_batch[b], torch.ones(N, dtype=torch.float32, device=X_t.device))
    out = out / counts.unsqueeze(-1).clamp_min(1e-12)
    return out


def build_M_torch(C: torch.Tensor, mu_bar: torch.Tensor, n_v: torch.Tensor, N: int) -> torch.Tensor:
    """M[v] = sqrt(n_v / N) * (C[v] - mu_bar). Supports batched C of shape (B, m, d) or (m, d)."""
    weights = torch.sqrt(n_v.to(torch.float64) / float(N))      # (m,) or (B, m)
    if C.ndim == 3:
        # broadcast n_v across batch — assumes label permutation preserves n_v
        return (weights.view(1, -1, 1) * (C.to(torch.float64) - mu_bar.to(torch.float64))).to(torch.float64)
    return (weights.view(-1, 1) * (C.to(torch.float64) - mu_bar.to(torch.float64))).to(torch.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# CONCEPT REGISTRY  (plan §6)
# ═══════════════════════════════════════════════════════════════════════════════

# Columns we never run as concepts (plan §6 "intentionally excluded").
SKIP_COLUMNS = {
    # JSON-list columns (each list element duplicates Tier 1/2 scalars)
    "a_digits_lsf", "b_digits_lsf", "answer_digits_lsf", "answer_digits_msf",
    "column_sums", "carries", "running_sums",
    "column_products", "partial_products",
    # Tier 5 tokenization metadata (degenerate or redundant with `answer`)
    "is_intersection",
    "is_single_token_gpt_j", "is_single_token_llama", "is_single_token_pythia",
    "first_token_id_gpt_j", "first_token_id_llama", "first_token_id_pythia",
    "first_token_text_gpt_j", "first_token_text_llama", "first_token_text_pythia",
    "n_tokens_gpt_j", "n_tokens_llama", "n_tokens_pythia",
}

# Curated joints (plan §6e).
JOINT_REGISTRY = {
    "addition": [
        # Operand-pair joints (4)
        ("a_units", "b_units"),
        ("a_tens", "b_tens"),
        ("a_units", "b_tens"),
        ("a_tens", "b_units"),
        # Carry-binding joints (4)
        ("a_tens", "b_tens", "carry_units"),
        ("a_tens", "b_tens", "ans_tens"),
        ("carry_units", "ans_units"),
        ("carry_units", "column_sum_units"),
        # Multi-column joint (1)
        ("a_units", "b_units", "ans_tens"),
        # Validation joint (1) — deterministic mapping
        ("a_units", "b_units", "ans_units"),
    ],
    "multiplication": [
        # Operand-pair joints (4)
        ("a_units", "b_units"),
        ("a_tens", "b_tens"),
        ("a_units", "b_tens"),
        ("a_tens", "b_units"),
        # Carry-binding joints (4)
        ("a_tens", "b_tens", "carry_units"),
        ("a_tens", "b_tens", "ans_tens"),
        ("carry_units", "ans_units"),
        ("carry_units", "column_sum_units"),
        # Multi-column joint (1)
        ("a_units", "b_units", "ans_tens"),
        # Validation joint (1) — deterministic mapping
        ("a_units", "b_units", "partial_product_units"),
    ],
}


def enumerate_single_concepts(problems_df: pd.DataFrame) -> list[str]:
    """Return all CSV columns eligible as single concepts (plan §6a-d).

    Filters out: SKIP_COLUMNS, columns that are constant across rows,
    columns that look like list-of-list strings (start with '[').
    """
    out = []
    for c in problems_df.columns:
        if c in SKIP_COLUMNS:
            continue
        col = problems_df[c]
        # Skip columns that store JSON list strings (defensive).
        if col.dtype == object:
            sample = col.dropna().iloc[0] if len(col.dropna()) else None
            if isinstance(sample, str) and sample.startswith("["):
                continue
        # Skip columns that are constant.
        if col.nunique(dropna=True) < 2:
            continue
        out.append(c)
    return out


def safe_concept_name(concept: str | tuple) -> str:
    """File-safe directory name for a concept (single or joint)."""
    if isinstance(concept, tuple):
        return "__".join(concept)
    return str(concept)


def concept_columns(concept: str | tuple) -> list[str]:
    """Return the source CSV columns this concept depends on."""
    return list(concept) if isinstance(concept, tuple) else [concept]


def build_label_array(problems_df: pd.DataFrame, concept: str | tuple, mask: np.ndarray) -> np.ndarray:
    """Return label vector y (object dtype for joints, native dtype for singles), masked.

    For joints, returns an array of tuples. The downstream filter+codec turns these
    into integer codes after MIN_GROUP_SIZE filtering.
    """
    if isinstance(concept, tuple):
        cols = [problems_df[c].to_numpy() for c in concept]
        # Build (N,) array of tuples
        y = np.empty(len(problems_df), dtype=object)
        y[:] = list(zip(*cols))
        return y[mask]
    return problems_df[concept].to_numpy()[mask]


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(logs_root: Path, model_key: str):
    logs_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"ccsvd_subspaces.{model_key}")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    # WatchedFileHandler re-opens the file if it detects inode changed
    # (handles deletion-while-running on NFS, log rotation, etc.).
    fh = WatchedFileHandler(logs_root / f"ccsvd_subspaces_{model_key}.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cell_seed(model_key: str, task: str, layer: int, concept_name: str, base_seed: int = 42) -> int:
    """Deterministic per-cell seed (64-bit hash truncated to int64)."""
    s = f"{model_key}|{task}|{layer:02d}|{concept_name}|{base_seed}"
    h = hashlib.sha256(s.encode()).hexdigest()
    return int(h[:16], 16) % (2**63 - 1)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE: per-cell CCSVD fit
# ═══════════════════════════════════════════════════════════════════════════════

def encode_labels(y: np.ndarray, min_group_size: int) -> tuple[np.ndarray, np.ndarray, list, list]:
    """Filter by MIN_GROUP_SIZE and encode labels to integer codes 0..m-1.

    Returns:
        y_codes:     (N',) int32 array of integer codes
        keep_mask:   (N,) bool array indicating which input rows survive
        keep_values: list of m surviving label values (in canonical order)
        dropped:     list of dropped label values
    """
    s = pd.Series(y)
    counts = s.value_counts()
    keep = counts[counts >= min_group_size]
    drop = counts[counts < min_group_size]
    keep_values = sorted(keep.index.tolist(), key=lambda v: (str(type(v).__name__), str(v)))
    dropped = sorted(drop.index.tolist(), key=lambda v: (str(type(v).__name__), str(v)))
    keep_mask = s.isin(keep_values).to_numpy()
    code_map = {v: i for i, v in enumerate(keep_values)}
    y_codes = np.array([code_map[v] for v in s[keep_mask]], dtype=np.int32)
    return y_codes, keep_mask, keep_values, dropped


def centroid_matrix(X: np.ndarray, y_codes: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Return per-value centroids C (m, d) and counts n_v (m,).

    Uses a one-hot @ X formulation for vectorisation.
    """
    n = X.shape[0]
    counts = np.bincount(y_codes, minlength=m).astype(np.int64)
    one_hot = np.zeros((n, m), dtype=np.float32)
    one_hot[np.arange(n), y_codes] = 1.0
    sums = one_hot.T @ X                          # (m, d)
    C = sums / counts[:, None]                    # (m, d)
    return C, counts


def scaled_centred_M(C: np.ndarray, mu_bar: np.ndarray, n_v: np.ndarray, N: int) -> np.ndarray:
    """Build M[v] = sqrt(n_v / N) * (µ_v − µ̄). Shape (m, d), float64."""
    weights = np.sqrt(n_v.astype(np.float64) / float(N))
    return weights[:, None] * (C.astype(np.float64) - mu_bar.astype(np.float64))


def fit_cell(
    X: np.ndarray,                # (N_correct, d) float32, already correctness-masked
    y: np.ndarray,                # (N_correct,) raw labels (any dtype)
    *,
    cell_id: dict,                # {model_key, task, layer, concept_name}
    min_group_size: int,
    n_permutations: int,
    perm_alpha: float,
    cv_n_splits: int,
    random_state: int,
    mean_centre: bool,
    unit_normalise: bool,
    device: torch.device | None = None,
    perm_batch: int = 50,         # how many shuffles to SVD per GPU batch
) -> dict:
    """One CCSVD fit on one cell. Returns a result dict with all artifacts.

    SVD on GPU when device is cuda. The 1,000 permutation-null SVDs are computed
    in batches of `perm_batch` via torch.linalg.svdvals on a (perm_batch, m, d)
    tensor.
    """

    t0 = time.time()
    seed = cell_seed(cell_id["model_key"], cell_id["task"], cell_id["layer"],
                     cell_id["concept_name"], base_seed=random_state)
    if device is None:
        device = get_device("auto")

    # ── Step 2: filter by group size and encode labels ────────────────────────
    y_codes, keep_mask, keep_values, dropped = encode_labels(y, min_group_size)
    n_groups_total = pd.Series(y).nunique()

    if len(keep_values) < 2:
        return {
            "cell_id": cell_id,
            "status": "skipped_insufficient_groups",
            "n_total_correct": int(len(y)),
            "n_after_filter": int(len(y_codes)),
            "n_groups_total": int(n_groups_total),
            "n_groups_after_filter": int(len(keep_values)),
            "dropped_values": [str(v) for v in dropped],
            "dropped_values_count": int(len(dropped)),
            "kept_values": [str(v) for v in keep_values],
            "seed": int(seed),
            "runtime_seconds": round(time.time() - t0, 3),
        }

    X_f = X[keep_mask]
    N = X_f.shape[0]
    m = len(keep_values)
    n_v = np.bincount(y_codes, minlength=m).astype(np.int64)

    # ── Step 3: global mean ───────────────────────────────────────────────────
    if not unit_normalise and mean_centre:
        mu_bar = X_f.mean(axis=0).astype(np.float64)
    elif mean_centre:
        mu_bar = X_f.mean(axis=0).astype(np.float64)
    else:
        mu_bar = np.zeros(X_f.shape[1], dtype=np.float64)

    # ── Move to device ────────────────────────────────────────────────────────
    X_t = torch.from_numpy(np.ascontiguousarray(X_f.astype(np.float32))).to(device)
    y_codes_t = torch.from_numpy(y_codes.astype(np.int64)).to(device)
    mu_bar_t = torch.from_numpy(mu_bar).to(device)
    n_v_t = torch.from_numpy(n_v).to(device)

    # ── Step 4: per-value centroids ───────────────────────────────────────────
    C_t = batched_centroids_torch(X_t, y_codes_t.unsqueeze(0), m)[0]   # (m, d)
    C = C_t.cpu().numpy().astype(np.float32)

    # ── Step 5: build M ───────────────────────────────────────────────────────
    M_t = build_M_torch(C_t, mu_bar_t, n_v_t, N)                       # (m, d) float64

    # ── Step 6: real SVD ──────────────────────────────────────────────────────
    U_t, s_t, Vt_t = torch.linalg.svd(M_t, full_matrices=False)
    eigenvalues = (s_t ** 2).cpu().numpy()                              # (m,)
    eigenvalues_eff = eigenvalues[: m - 1]
    Vt_real = Vt_t.cpu().numpy()                                        # (m, d) float64

    # ── Step 7: permutation null (BATCHED on device) ──────────────────────────
    rng = np.random.default_rng(seed)
    null_table = np.zeros((n_permutations, m - 1), dtype=np.float64)
    p_done = 0
    while p_done < n_permutations:
        bs = min(perm_batch, n_permutations - p_done)
        # Generate bs shuffled label vectors on CPU, ship to device.
        shuf = np.stack([rng.permutation(y_codes) for _ in range(bs)]).astype(np.int64)
        shuf_t = torch.from_numpy(shuf).to(device)
        C_batch = batched_centroids_torch(X_t, shuf_t, m)               # (bs, m, d)
        M_batch = build_M_torch(C_batch, mu_bar_t, n_v_t, N)            # (bs, m, d)
        s_batch = torch.linalg.svdvals(M_batch)                         # (bs, m)
        null_eigs = (s_batch ** 2).cpu().numpy()
        null_table[p_done : p_done + bs, :] = null_eigs[:, : m - 1]
        p_done += bs

    threshold_99 = np.percentile(null_table, 100.0 * (1.0 - perm_alpha), axis=0)

    # ── Step 8: sequential stopping → r ───────────────────────────────────────
    r = 0
    for k in range(m - 1):
        if eigenvalues_eff[k] > threshold_99[k]:
            r = k + 1
        else:
            break

    # ── Step 9: basis B ───────────────────────────────────────────────────────
    if r > 0:
        B = Vt_real[:r, :].T.astype(np.float32)         # (d, r)
    else:
        B = np.zeros((X_f.shape[1], 0), dtype=np.float32)

    # ── Step 10: project centroids and full cloud ─────────────────────────────
    C_centred = (C.astype(np.float64) - mu_bar).astype(np.float32)
    C_proj = (C_centred @ B).astype(np.float32)
    X_proj = ((X_f.astype(np.float64) - mu_bar).astype(np.float32) @ B).astype(np.float32) if r > 0 \
        else np.zeros((N, 0), dtype=np.float32)

    # ── Step 11: 5-fold subspace-preservation CV ─────────────────────────────
    cv_per_fold = np.full(cv_n_splits, np.nan, dtype=np.float64)
    if r > 0:
        skf = StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=random_state)
        for fi, (train_idx, test_idx) in enumerate(skf.split(X_f, y_codes)):
            X_tr, y_tr = X_f[train_idx], y_codes[train_idx]
            X_te, y_te = X_f[test_idx], y_codes[test_idx]
            # need each surviving value to appear in both splits; if not, NaN this fold
            n_v_tr = np.bincount(y_tr, minlength=m)
            n_v_te = np.bincount(y_te, minlength=m)
            usable = (n_v_tr > 0) & (n_v_te > 0)
            if usable.sum() < 2:
                continue
            mu_tr = X_tr.mean(axis=0).astype(np.float64)
            C_tr, _ = centroid_matrix(X_tr, y_tr, m)
            M_tr = scaled_centred_M(C_tr, mu_tr, n_v_tr.astype(np.int64).clip(min=1), len(X_tr))
            try:
                _, s_tr, Vt_tr = np.linalg.svd(M_tr, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            r_eff = min(r, Vt_tr.shape[0])
            B_tr = Vt_tr[:r_eff, :].T                                    # (d, r_eff)
            # Compute test centroids restricted to usable values.
            C_te, _ = centroid_matrix(X_te, y_te, m)
            uv = np.where(usable)[0]
            C_te_u = C_te[uv]
            C_te_centred = C_te_u.astype(np.float64) - mu_tr             # use train mean for consistency
            C_te_sub = C_te_centred @ B_tr
            # pairwise distances upper-triangular
            from scipy.spatial.distance import pdist
            D_full = pdist(C_te_centred)
            D_sub = pdist(C_te_sub)
            if np.std(D_full) > 0 and np.std(D_sub) > 0 and len(D_full) >= 2:
                corr, _ = pearsonr(D_full, D_sub)
                cv_per_fold[fi] = corr

    valid = ~np.isnan(cv_per_fold)
    cv_mean = float(np.nanmean(cv_per_fold)) if valid.any() else float("nan")
    cv_std = float(np.nanstd(cv_per_fold)) if valid.any() else float("nan")

    # ── Step 12-13: summary stats and flags ───────────────────────────────────
    lam = eigenvalues_eff
    total_var = float(lam.sum())
    explained_variance = (lam / total_var).tolist() if total_var > 0 else [float("nan")] * len(lam)
    cumulative_variance = np.cumsum(explained_variance).tolist() if total_var > 0 else explained_variance

    lam_1 = float(lam[0]) if len(lam) >= 1 else float("nan")
    lam_2 = float(lam[1]) if len(lam) >= 2 else float("nan")
    lam_3 = float(lam[2]) if len(lam) >= 3 else float("nan")
    lam_1_over_2 = (lam_1 / lam_2) if (len(lam) >= 2 and lam_2 > 0) else float("nan")

    n_d_ratio = (N / r) if r > 0 else float("nan")
    flag_n_d_inflation = bool(r > 0 and n_d_ratio < 5)
    flag_single_direction = bool(r >= 2 and lam_1_over_2 > 10)
    nv_max, nv_min = int(n_v.max()), int(n_v.min())
    group_imbalance_ratio = nv_max / nv_min if nv_min > 0 else float("inf")
    flag_group_imbalance = bool(group_imbalance_ratio > 3)

    activation_norm = np.linalg.norm(X_f, axis=1)

    status = "fit_ok" if r > 0 else "no_significant_subspace"

    return {
        "cell_id": cell_id,
        "status": status,
        "n_total_correct": int(len(y)),
        "n_after_filter": int(N),
        "n_groups_total": int(n_groups_total),
        "n_groups_after_filter": int(m),
        "dropped_values": [str(v) for v in dropped],
        "dropped_values_count": int(len(dropped)),
        "kept_values": [str(v) for v in keep_values],
        "n_v": n_v.tolist(),
        "r_dim": int(r),
        "eigenvalues": eigenvalues_eff.tolist(),
        "threshold_99": threshold_99.tolist(),
        "explained_variance": explained_variance,
        "cumulative_variance": cumulative_variance,
        "lambda_1": lam_1,
        "lambda_2": lam_2,
        "lambda_3": lam_3,
        "lambda_1_over_2": lam_1_over_2,
        "total_variance": total_var,
        "n_d_ratio": n_d_ratio,
        "max_n_v": nv_max,
        "min_n_v": nv_min,
        "group_imbalance_ratio": group_imbalance_ratio,
        "flag_n_d_inflation": flag_n_d_inflation,
        "flag_single_direction": flag_single_direction,
        "flag_group_imbalance": flag_group_imbalance,
        "cv_per_fold": cv_per_fold.tolist(),
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "activation_norm_mean": float(activation_norm.mean()),
        "activation_norm_std": float(activation_norm.std()),
        "null_summary": {
            "p50": np.percentile(null_table, 50, axis=0).tolist(),
            "p75": np.percentile(null_table, 75, axis=0).tolist(),
            "p90": np.percentile(null_table, 90, axis=0).tolist(),
            "p95": np.percentile(null_table, 95, axis=0).tolist(),
            "p99": np.percentile(null_table, 99, axis=0).tolist(),
        },
        "seed": int(seed),
        "runtime_seconds": round(time.time() - t0, 3),
        # Heavy artifacts (not stored in JSON; written separately to .npy):
        "_basis": B,
        "_centroids": C.astype(np.float32),
        "_centroids_proj": C_proj,
        "_projected_acts": X_proj,
        "_null_table": null_table,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PER-CELL DRIVER (returns one row's worth of CSV info + writes per-cell artifacts)
# ═══════════════════════════════════════════════════════════════════════════════

def process_cell(
    *,
    model_cfg: dict,
    task: str,
    layer: int,
    concept,
    X_correct: np.ndarray,
    problems_df: pd.DataFrame,
    correct_mask: np.ndarray,
    activations_path: Path,
    labels_path: Path,
    answers_path: Path,
    out_root: Path,
    cfg_ccsvd: dict,
    trustworthiness: dict,
) -> dict:
    """Fit one cell, write its per-cell artifacts, return a flat row dict."""
    concept_name = safe_concept_name(concept)
    cell_id = {
        "model_key": model_cfg["key"],
        "task": task,
        "layer": layer,
        "concept_name": concept_name,
        "concept_columns": concept_columns(concept),
    }

    # Validate that all source columns exist; if not, skip with explicit reason.
    for c in concept_columns(concept):
        if c not in problems_df.columns:
            return {
                "model_key": model_cfg["key"], "task": task, "layer": layer,
                "concept": concept_name, "status": "skipped_missing_column",
                "missing_column": c, "n_total_correct": int(correct_mask.sum()),
                "n_after_filter": 0, "n_groups_total": 0, "n_groups_after_filter": 0,
                "r_dim": 0, "runtime_seconds": 0.0,
            }

    y = build_label_array(problems_df, concept, correct_mask)
    res = fit_cell(
        X_correct, y,
        cell_id=cell_id,
        min_group_size=cfg_ccsvd["min_group_size"],
        n_permutations=cfg_ccsvd["n_permutations"],
        perm_alpha=cfg_ccsvd["perm_alpha"],
        cv_n_splits=cfg_ccsvd["cv_n_splits"],
        random_state=cfg_ccsvd["random_state"],
        mean_centre=cfg_ccsvd["mean_centre"],
        unit_normalise=cfg_ccsvd["unit_normalise"],
    )

    # ── Write per-cell artifacts ──────────────────────────────────────────────
    cell_dir = out_root / model_cfg["key"] / task / f"layer_{layer:02d}" / concept_name
    cell_dir.mkdir(parents=True, exist_ok=True)

    if "_basis" in res:
        np.save(cell_dir / "basis.npy", res["_basis"])
        np.save(cell_dir / "centroids.npy", res["_centroids"])
        np.save(cell_dir / "centroids_proj.npy", res["_centroids_proj"])
        np.save(cell_dir / "projected_acts.npy", res["_projected_acts"])
        np.save(cell_dir / "null_eigenvalues.npy", res["_null_table"])
        np.save(cell_dir / "eigenvalues.npy", np.array(res["eigenvalues"]))
        np.save(cell_dir / "threshold_99.npy", np.array(res["threshold_99"]))
        np.save(cell_dir / "cv_per_fold.npy", np.array(res["cv_per_fold"]))

    meta = {k: v for k, v in res.items() if not k.startswith("_")}
    meta["activations_path"] = str(activations_path)
    meta["labels_path"] = str(labels_path)
    meta["answers_path"] = str(answers_path)
    meta["best_umap_trustworthiness"] = trustworthiness.get("best_umap")
    meta["best_tsne_trustworthiness"] = trustworthiness.get("best_tsne")
    meta["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta["library_versions"] = {
        "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": __import__("scipy").__version__,
        "sklearn": __import__("sklearn").__version__,
        "python": platform.python_version(),
    }
    with open(cell_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # ── Build the flat summary row ────────────────────────────────────────────
    row = dict(
        model_key=model_cfg["key"],
        task=task,
        layer=layer,
        concept=concept_name,
        n_total_correct=res.get("n_total_correct"),
        n_after_filter=res.get("n_after_filter"),
        n_groups_total=res.get("n_groups_total"),
        n_groups_after_filter=res.get("n_groups_after_filter"),
        dropped_values_count=res.get("dropped_values_count"),
        dropped_values_str=";".join(res.get("dropped_values", [])),
        r_dim=res.get("r_dim", 0),
        lambda_1=res.get("lambda_1"),
        lambda_2=res.get("lambda_2"),
        lambda_3=res.get("lambda_3"),
        lambda_1_over_2=res.get("lambda_1_over_2"),
        explained_variance_top1=(res.get("explained_variance") or [None])[0],
        explained_variance_top5=sum((res.get("explained_variance") or [])[:5]) if res.get("explained_variance") else None,
        cumulative_variance_at_r=(res.get("cumulative_variance") or [None])[res.get("r_dim", 0) - 1] if res.get("r_dim", 0) > 0 else None,
        total_variance=res.get("total_variance"),
        n_d_ratio=res.get("n_d_ratio"),
        max_n_v=res.get("max_n_v"),
        min_n_v=res.get("min_n_v"),
        group_imbalance_ratio=res.get("group_imbalance_ratio"),
        flag_n_d_inflation=res.get("flag_n_d_inflation"),
        flag_single_direction=res.get("flag_single_direction"),
        flag_group_imbalance=res.get("flag_group_imbalance"),
        cv_mean=res.get("cv_mean"),
        cv_std=res.get("cv_std"),
        activation_norm_mean=res.get("activation_norm_mean"),
        activation_norm_std=res.get("activation_norm_std"),
        best_umap_trustworthiness=trustworthiness.get("best_umap"),
        best_tsne_trustworthiness=trustworthiness.get("best_tsne"),
        status=res.get("status"),
        runtime_seconds=res.get("runtime_seconds"),
        seed=res.get("seed"),
    )
    return row


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_trustworthiness(embeddings_root: Path, model_key: str, task: str, layer: int) -> dict:
    """Read the prior UMAP/t-SNE manifest if available."""
    manifest_path = embeddings_root / model_key / f"{task}_layer_{layer:02d}_manifest.json"
    out = {"best_umap": None, "best_tsne": None}
    if not manifest_path.exists():
        return out
    try:
        m = json.loads(manifest_path.read_text())
        tw = m.get("trustworthiness", {})
        u = [v for k, v in tw.items() if k.startswith("umap")]
        t = [v for k, v in tw.items() if k.startswith("tsne")]
        out["best_umap"] = max(u) if u else None
        out["best_tsne"] = max(t) if t else None
    except Exception:
        pass
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, help="Model key from config: gpt-j-6b | llama-3.1-8b | pythia-6.9b")
    parser.add_argument("--limit-concepts", type=int, default=None,
                        help="Smoke-test: process at most this many concepts per (task, layer).")
    parser.add_argument("--single-concept", default=None,
                        help="Smoke-test: only fit this one concept (single name only, not joint).")
    parser.add_argument("--single-task", default=None, help="Smoke-test: only this task.")
    parser.add_argument("--single-layer", type=int, default=None, help="Smoke-test: only this layer.")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    paths = cfg["paths"]
    cfg_ccsvd = cfg["ccsvd"]

    logs_root = Path(paths["logs_root"])
    logger = setup_logging(logs_root, args.model)

    model_cfg = next((m for m in cfg["models"] if m["key"] == args.model), None)
    if model_cfg is None:
        raise ValueError(f"Model {args.model} not in config")

    out_root = Path(paths["results_root"]) / "ccsvd_subspaces"
    out_root.mkdir(parents=True, exist_ok=True)
    embeddings_root = Path(paths["results_root"]) / "embeddings"
    data_root = Path(paths["data_root"])

    t_run0 = time.time()
    logger.info("=" * 78)
    logger.info("CCSVD subspaces — model=%s", args.model)
    logger.info("config=%s", args.config)
    logger.info("ccsvd settings: %s", cfg_ccsvd)

    # Build cell list
    rows = []
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
        N_total, N_correct = len(problems_df), int(correct_mask.sum())

        # Concept list for this task
        singles = enumerate_single_concepts(problems_df)
        joints = JOINT_REGISTRY.get(task, [])
        if args.single_concept:
            concepts = [args.single_concept]
        else:
            concepts = singles + joints
            if args.limit_concepts:
                concepts = concepts[: args.limit_concepts]

        logger.info("[%s] N_total=%d N_correct=%d singles=%d joints=%d total=%d",
                    task, N_total, N_correct, len(singles), len(joints), len(concepts))

        for layer in layers:
            activations_path = data_root / "activations" / args.model / f"{task}_layer_{layer:02d}.npy"
            if not activations_path.exists():
                logger.warning("missing activations: %s — skipping layer", activations_path)
                continue
            X_full = np.load(activations_path)
            X_correct = np.ascontiguousarray(X_full[correct_mask].astype(np.float32))
            del X_full
            tw = fetch_trustworthiness(embeddings_root, args.model, task, layer)

            t_layer = time.time()
            logger.info("  layer %d: fitting %d concepts on N=%d (n_jobs=%d)",
                        layer, len(concepts), X_correct.shape[0], cfg_ccsvd["n_jobs"])
            results = Parallel(n_jobs=cfg_ccsvd["n_jobs"], backend="loky", verbose=0)(
                delayed(process_cell)(
                    model_cfg=model_cfg, task=task, layer=layer, concept=c,
                    X_correct=X_correct, problems_df=problems_df, correct_mask=correct_mask,
                    activations_path=activations_path, labels_path=labels_path,
                    answers_path=answers_path, out_root=out_root,
                    cfg_ccsvd=cfg_ccsvd, trustworthiness=tw,
                ) for c in concepts
            )
            rows.extend(results)
            logger.info("  layer %d done in %.1fs", layer, time.time() - t_layer)

    # ── Write per-model master CSVs ───────────────────────────────────────────
    model_out = out_root / args.model
    model_out.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(model_out / f"summary_{args.model}.csv", index=False)
    logger.info("wrote %s (rows=%d)", model_out / f"summary_{args.model}.csv", len(summary_df))

    # eigenvalue_spectra long-form
    eig_rows = []
    centroid_rows = []
    null_rows = []
    cv_rows = []
    for r in rows:
        if r.get("status") not in ("fit_ok", "no_significant_subspace"):
            continue
        cell_dir = out_root / r["model_key"] / r["task"] / f"layer_{r['layer']:02d}" / r["concept"]
        meta_path = cell_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        evs = meta.get("eigenvalues", [])
        thr = meta.get("threshold_99", [])
        ev_total = sum(evs) if evs else 0.0
        cum = 0.0
        for k, (lam, t99) in enumerate(zip(evs, thr), start=1):
            ex = (lam / ev_total) if ev_total > 0 else float("nan")
            cum += ex
            eig_rows.append(dict(
                model_key=r["model_key"], task=r["task"], layer=r["layer"], concept=r["concept"],
                k=k, lambda_k=lam, threshold_99_k=t99,
                significant=bool(lam > t99),
                explained_variance_k=ex, cumulative_variance_k=cum,
            ))
        # null_summary
        ns = meta.get("null_summary", {})
        for k in range(len(evs)):
            null_rows.append(dict(
                model_key=r["model_key"], task=r["task"], layer=r["layer"], concept=r["concept"], k=k + 1,
                null_p50=ns.get("p50", [float("nan")] * (k + 1))[k] if ns.get("p50") else float("nan"),
                null_p75=ns.get("p75", [float("nan")] * (k + 1))[k] if ns.get("p75") else float("nan"),
                null_p90=ns.get("p90", [float("nan")] * (k + 1))[k] if ns.get("p90") else float("nan"),
                null_p95=ns.get("p95", [float("nan")] * (k + 1))[k] if ns.get("p95") else float("nan"),
                null_p99=ns.get("p99", [float("nan")] * (k + 1))[k] if ns.get("p99") else float("nan"),
            ))
        # CV per fold
        for fi, corr in enumerate(meta.get("cv_per_fold", []), start=1):
            cv_rows.append(dict(
                model_key=r["model_key"], task=r["task"], layer=r["layer"], concept=r["concept"],
                fold=fi, pearson_corr=corr,
            ))
        # projected centroids (long-form)
        kept = meta.get("kept_values", [])
        n_v = meta.get("n_v", [])
        cp_path = cell_dir / "centroids_proj.npy"
        if cp_path.exists():
            cp = np.load(cp_path)
            for vi, val in enumerate(kept):
                for di in range(cp.shape[1]):
                    centroid_rows.append(dict(
                        model_key=r["model_key"], task=r["task"], layer=r["layer"], concept=r["concept"],
                        value=str(val), n_v=n_v[vi] if vi < len(n_v) else None,
                        dim_idx=di + 1, dim_value=float(cp[vi, di]),
                    ))

    pd.DataFrame(eig_rows).to_csv(model_out / f"eigenvalue_spectra_{args.model}.csv", index=False)
    pd.DataFrame(null_rows).to_csv(model_out / f"null_summary_{args.model}.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(model_out / f"cv_per_fold_{args.model}.csv", index=False)
    pd.DataFrame(centroid_rows).to_csv(model_out / f"projected_centroids_{args.model}.csv", index=False)

    # ── Per-model manifest fragment ───────────────────────────────────────────
    manifest = {
        "model_key": args.model,
        "n_cells_attempted": len(rows),
        "n_cells_fit_ok": int(sum(1 for r in rows if r.get("status") == "fit_ok")),
        "n_cells_skipped": int(sum(1 for r in rows if str(r.get("status", "")).startswith("skipped"))),
        "n_cells_no_significant_subspace": int(sum(1 for r in rows if r.get("status") == "no_significant_subspace")),
        "ccsvd_settings": cfg_ccsvd,
        "library_versions": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "sklearn": __import__("sklearn").__version__,
            "python": platform.python_version(),
        },
        "config_sha256": sha256_of(Path(args.config)),
        "total_runtime_seconds": round(time.time() - t_run0, 2),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(model_out / f"manifest_{args.model}.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info("wrote per-model manifest")

    logger.info("=" * 78)
    logger.info("DONE  total wall=%.1f min  cells=%d  fit_ok=%d  skipped=%d",
                (time.time() - t_run0) / 60.0, len(rows),
                manifest["n_cells_fit_ok"], manifest["n_cells_skipped"])


if __name__ == "__main__":
    main()
