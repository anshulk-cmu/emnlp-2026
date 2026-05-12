"""Step 7 — Residual hunting.

For each (model, task, mode, layer) cell:

  1. Build two global union variants:
       merged   = SVD-orthonormalised stack of (CCSVD ∪ LDA-Option-A)
                  + mode-specific β scalar direction.
       generous = SVD-orthonormalised stack of (CCSVD ∪ LDA-Option-A ∪ LDA-Option-B)
                  + mode-specific β scalar direction.

  2. For each variant:
       - X_proj = (X @ V.T) @ V  on GPU
       - X_resid = X − X_proj
       - var_explained = 1 − ||X_resid||²/||X||²
       - randomized SVD of X_centered → top N_PCA_COMPONENTS eigenvalues + vectors
       - trace-based σ², MP edge, n_above_mp.

  3. Correlation sweep (merged variant only — generous is audit-only):
       - top n_above_mp directions (or top 50 if 0)
       - Spearman + Pearson against every metadata column + derived columns
       - 1000-permutation FDR null (BH correction across the (direction × column) grid)

  4. Pre-compute Stage 3 correlate-set unions per task target (plan §4.2-4.3).

  5. Resume logic via metadata.json.

Run on full data — no subsampling, no random row sampling. See plan §2 standing rules.

Usage:
  python residual_hunting.py --config /home/anshulk/emnlp2026/config.yaml \
      --model gpt-j-6b --task addition --mode off --all-layers
  python residual_hunting.py --config ... --model gpt-j-6b --task addition \
      --mode off --layer 14
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from logging.handlers import WatchedFileHandler
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import false_discovery_control, rankdata
from sklearn.utils.extmath import randomized_svd

try:
    import cupy as cp
    _HAS_CUPY = cp.cuda.is_available()
except Exception:
    _HAS_CUPY = False
    cp = None

from ccsvd_subspaces import (
    JOINT_REGISTRY,
    SKIP_COLUMNS,
    cell_seed,
    enumerate_single_concepts,
    safe_concept_name,
    sha256_of,
)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

N_PCA_COMPONENTS = 500
SVD_TOLERANCE_FACTOR = 1e-10
N_PERMUTATIONS = 1000
CORR_FLAG_THRESHOLD = 0.15
FDR_ALPHA = 0.05
GAMMA_RELIABLE_MAX = 0.7
N_TOP_DIRECTIONS_IF_NULL_MP = 50

STAGE3_CORRELATE_SETS = {
    "addition": {
        "ans_units": ["a", "b", "a_units", "b_units"],
        "ans_tens":  ["a", "b", "a_tens", "b_tens", "carry_units"],
        "answer":    ["a", "b"],
    },
    "multiplication": {
        "carry_units": ["column_sum_units", "partial_product_units"],
        "ans_units":   ["column_sum_units", "carry_units", "partial_product_units"],
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(logs_root: Path, model_key: str, task: str, mode: str) -> logging.Logger:
    logs_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"residual_hunting.{model_key}.{task}.{mode}")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = WatchedFileHandler(logs_root / f"residual_hunting_{model_key}_{task}_{mode}.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


# ──────────────────────────────────────────────────────────────────────────────
# Atomic I/O
# ──────────────────────────────────────────────────────────────────────────────

def atomic_save(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".npy", dir=str(path.parent))
    os.close(fd)
    np.save(tmp, arr)
    os.replace(tmp, path)


def atomic_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".json", dir=str(path.parent))
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


# ──────────────────────────────────────────────────────────────────────────────
# Basis loaders
# ──────────────────────────────────────────────────────────────────────────────

def ccsvd_basis_path(results_root: Path, model: str, task: str, layer: int, concept: str, mode: str) -> Path:
    if mode == "off":
        return results_root / "ccsvd_subspaces" / model / task / f"layer_{layer:02d}" / concept / "basis.npy"
    return results_root / "ccsvd_subspaces" / f"mode_{mode}" / model / task / f"layer_{layer:02d}" / concept / "basis.npy"


def lda_a_basis_path(results_root: Path, model: str, task: str, layer: int, concept: str, mode: str) -> Path:
    return (results_root / "lda_subspaces" / "subspace_lda" / f"mode_{mode}"
            / model / task / f"layer_{layer:02d}" / concept / "lda_basis_full.npy")


def lda_b_basis_path(results_root: Path, model: str, task: str, layer: int, concept: str, mode: str) -> Path:
    return (results_root / "lda_subspaces" / "full_lda" / f"mode_{mode}"
            / model / task / f"layer_{layer:02d}" / concept / "lda_basis_full.npy")


def load_basis_rows(path: Path) -> np.ndarray:
    """Load a basis stored on disk as (D, r) columns and return (r, D) rows."""
    if not path.exists():
        return np.zeros((0, 4096), dtype=np.float32)
    B = np.load(path)
    if B.ndim != 2:
        return np.zeros((0, 4096), dtype=np.float32)
    if B.shape[0] == 4096:
        return B.T.astype(np.float32, copy=False)
    if B.shape[1] == 4096:
        return B.astype(np.float32, copy=False)
    return np.zeros((0, 4096), dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Concept filter (read from Step 6 LDA-A summary)
# ──────────────────────────────────────────────────────────────────────────────

def load_concept_filter(results_root: Path, model: str, task: str, mode: str, layer: int) -> dict:
    """Return {concept_name: status_dict} for every concept attempted in Step 6.

    status_dict has keys: status, n_sig, is_carved_out, lambda_T_1.
    Only concepts with status='fit_ok' and n_sig>=1 and is_carved_out=False
    are eligible for the union.
    """
    summary_path = (results_root / "lda_subspaces" / "subspace_lda" / f"mode_{mode}"
                    / model / f"summary_{model}_mode_{mode}.csv")
    df = pd.read_csv(summary_path)
    df = df[(df["task"] == task) & (df["layer"] == layer)]
    out = {}
    for _, row in df.iterrows():
        out[str(row["concept"])] = {
            "status": str(row.get("status", "")),
            "n_sig": int(row.get("n_sig", 0)) if not pd.isna(row.get("n_sig")) else 0,
            "is_carved_out": bool(row.get("is_carved_out", False)),
            "lambda_T_1": float(row.get("lambda_T_1", float("nan"))),
        }
    return out


def eligible_concepts(filter_dict: dict) -> list[str]:
    return [c for c, s in filter_dict.items()
            if s["status"] == "fit_ok" and s["n_sig"] >= 1 and not s["is_carved_out"]]


# ──────────────────────────────────────────────────────────────────────────────
# β scalar directions per mode
# ──────────────────────────────────────────────────────────────────────────────

def beta_scalar_direction(X: np.ndarray, scalar: np.ndarray) -> np.ndarray:
    """OLS slope of mean-centred X on mean-centred scalar, normalised to unit length.

    Returns (4096,) unit vector or zeros if scalar is constant.
    """
    s = scalar.astype(np.float64) - scalar.astype(np.float64).mean()
    denom = float(np.dot(s, s))
    if denom < 1e-12:
        return np.zeros(X.shape[1], dtype=np.float32)
    Xc = X.astype(np.float64) - X.astype(np.float64).mean(axis=0, keepdims=True)
    beta = (Xc.T @ s) / denom
    n = float(np.linalg.norm(beta))
    if n < 1e-12:
        return np.zeros(X.shape[1], dtype=np.float32)
    return (beta / n).astype(np.float32)


def append_mode_betas(stacked: np.ndarray, X: np.ndarray, mode: str, answer_scalar: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Append β_answer and/or β_norm rows to a stacked basis matrix."""
    extras = []
    extra_labels = []
    if mode == "off":
        b = beta_scalar_direction(X, answer_scalar)
        if np.linalg.norm(b) > 0:
            extras.append(b); extra_labels.append("beta_answer")
    elif mode == "norm":
        row_norm = np.linalg.norm(X, axis=1)
        b1 = beta_scalar_direction(X, row_norm)
        b2 = beta_scalar_direction(X, answer_scalar)
        if np.linalg.norm(b1) > 0: extras.append(b1); extra_labels.append("beta_norm")
        if np.linalg.norm(b2) > 0: extras.append(b2); extra_labels.append("beta_answer")
    # mode == "answer": cache already nulls the answer scalar; do not re-add.
    if not extras:
        return stacked, extra_labels
    return np.vstack([stacked, np.stack(extras)]), extra_labels


# ──────────────────────────────────────────────────────────────────────────────
# Union builder
# ──────────────────────────────────────────────────────────────────────────────

def build_union(
    results_root: Path,
    model: str, task: str, mode: str, layer: int,
    concepts: list[str],
    variant: str,                            # "merged" or "generous"
    X: np.ndarray,
    answer_scalar: np.ndarray,
    logger: logging.Logger,
) -> tuple[np.ndarray, dict]:
    """Stack per-concept bases per variant, SVD-orthonormalise, append β directions.

    Returns (V_all (k, 4096), meta).
    """
    rows = []
    contributions = []   # (concept, source, n_dims)
    for c in concepts:
        sources = ["ccsvd", "lda_a"] if variant == "merged" else ["ccsvd", "lda_a", "lda_b"]
        for src in sources:
            if src == "ccsvd":
                B = load_basis_rows(ccsvd_basis_path(results_root, model, task, layer, c, mode))
            elif src == "lda_a":
                B = load_basis_rows(lda_a_basis_path(results_root, model, task, layer, c, mode))
            elif src == "lda_b":
                B = load_basis_rows(lda_b_basis_path(results_root, model, task, layer, c, mode))
            else:
                continue
            if B.shape[0] > 0:
                rows.append(B)
                contributions.append({"concept": c, "source": src, "n_dims": int(B.shape[0])})
    if not rows:
        return np.zeros((0, 4096), dtype=np.float32), {
            "variant": variant, "k": 0, "stacked_dim": 0,
            "n_concepts_kept": 0, "contributions": [], "extra_betas": [],
        }
    stacked = np.vstack(rows)
    stacked, beta_labels = append_mode_betas(stacked, X, mode, answer_scalar)
    stacked_dim = stacked.shape[0]
    # SVD orthonormalisation.
    U, S, Vt = np.linalg.svd(stacked, full_matrices=False)
    if S.size == 0:
        return np.zeros((0, 4096), dtype=np.float32), {
            "variant": variant, "k": 0, "stacked_dim": int(stacked_dim),
            "n_concepts_kept": 0, "contributions": contributions, "extra_betas": beta_labels,
        }
    keep = S > SVD_TOLERANCE_FACTOR * S[0]
    V_all = Vt[keep].astype(np.float32)
    return V_all, {
        "variant": variant,
        "k": int(V_all.shape[0]),
        "stacked_dim": int(stacked_dim),
        "n_concepts_kept": int(len(set(d["concept"] for d in contributions))),
        "redundancy_removed": int(stacked_dim - V_all.shape[0]),
        "contributions": contributions,
        "extra_betas": beta_labels,
        "top_singular_value": float(S[0]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Projection + variance
# ──────────────────────────────────────────────────────────────────────────────

def project_and_residual(X: np.ndarray, V_all: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Compute X_residual on GPU (with CPU fallback). Returns (X_residual, var_orig, var_resid, var_explained)."""
    if _HAS_CUPY and V_all.shape[0] > 0:
        X_g = cp.asarray(X)
        V_g = cp.asarray(V_all)
        coords = X_g @ V_g.T
        X_proj = coords @ V_g
        X_resid = X_g - X_proj
        var_orig = float(cp.sum(X_g ** 2))
        var_resid = float(cp.sum(X_resid ** 2))
        out = cp.asnumpy(X_resid)
        del X_g, V_g, coords, X_proj, X_resid
        cp.get_default_memory_pool().free_all_blocks()
    else:
        if V_all.shape[0] == 0:
            X_resid = X.copy()
        else:
            coords = X @ V_all.T
            X_proj = coords @ V_all
            X_resid = X - X_proj
        var_orig = float(np.sum(X.astype(np.float64) ** 2))
        var_resid = float(np.sum(X_resid.astype(np.float64) ** 2))
        out = X_resid
    var_explained = 1.0 - var_resid / max(var_orig, 1e-30)
    return out, var_orig, var_resid, var_explained


# ──────────────────────────────────────────────────────────────────────────────
# PCA + Marchenko-Pastur
# ──────────────────────────────────────────────────────────────────────────────

def pca_with_mp(X_resid: np.ndarray, d_residual: int, seed: int, n_components: int = N_PCA_COMPONENTS) -> dict:
    """Centred randomized SVD on X_resid; trace-based σ²; MP edge.

    Returns dict with eigenvalues, eigenvectors, sigma_sq, gamma, lambda_max_mp,
    lambda_min_mp, n_above_mp, d_residual, n_components, top_eigenvalue.
    """
    N = X_resid.shape[0]
    mu = X_resid.mean(axis=0, keepdims=True).astype(np.float32)
    X_centered = X_resid - mu
    total_var = float(np.sum(X_centered.astype(np.float64) ** 2) / N)
    sigma_sq = total_var / max(d_residual, 1)
    gamma = d_residual / max(N, 1)

    n_components = min(n_components, X_centered.shape[1] - 1, N - 1)
    # randomized_svd is CPU; for our sizes (~10k x 4096) this is the right choice.
    # sklearn requires a uint32 random_state; truncate our 64-bit cell seed.
    U, S, Vt = randomized_svd(X_centered, n_components=n_components, random_state=int(seed) % (2**32 - 1))
    eigenvalues = (S ** 2) / N

    if gamma < 1.0:
        lambda_min_mp = sigma_sq * (1.0 - np.sqrt(gamma)) ** 2
    else:
        lambda_min_mp = 0.0
    lambda_max_mp = sigma_sq * (1.0 + np.sqrt(gamma)) ** 2
    n_above_mp = int(np.sum(eigenvalues > lambda_max_mp))

    return {
        "eigenvalues": eigenvalues.astype(np.float64),
        "eigenvectors": Vt.astype(np.float32),
        "sigma_sq": float(sigma_sq),
        "gamma": float(gamma),
        "lambda_max_mp": float(lambda_max_mp),
        "lambda_min_mp": float(lambda_min_mp),
        "n_above_mp": int(n_above_mp),
        "d_residual": int(d_residual),
        "n_components": int(n_components),
        "top_eigenvalue": float(eigenvalues[0]) if eigenvalues.size > 0 else float("nan"),
        "mp_reliable_flag": bool(gamma < GAMMA_RELIABLE_MAX),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Derived columns for the correlation sweep
# ──────────────────────────────────────────────────────────────────────────────

def _safe_num(s: pd.Series) -> np.ndarray | None:
    """Coerce a Series to numeric numpy (float64), return None if not coerceable to ≥2 unique values."""
    try:
        x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
        if np.isnan(x).all() or len(np.unique(x[~np.isnan(x)])) < 2:
            return None
        return x
    except Exception:
        return None


def _interaction(a: pd.Series, b: pd.Series) -> np.ndarray | None:
    A = _safe_num(a); B = _safe_num(b)
    if A is None or B is None:
        return None
    return A * B


def _mod10(a: pd.Series, b: pd.Series) -> np.ndarray | None:
    A = _safe_num(a); B = _safe_num(b)
    if A is None or B is None:
        return None
    return (A + B) % 10.0


def _consecutive_carry_run(carries_col: pd.Series) -> np.ndarray | None:
    """Longest run of nonzero carries per row (if the column stores list-like data)."""
    out = np.zeros(len(carries_col), dtype=np.float64)
    parsed = 0
    for i, v in enumerate(carries_col.tolist()):
        try:
            if isinstance(v, str):
                v = v.strip("[](){} ").split(",")
                arr = [int(x) for x in v if x.strip() != ""]
            else:
                arr = list(v)
            run = best = 0
            for x in arr:
                if x != 0:
                    run += 1; best = max(best, run)
                else:
                    run = 0
            out[i] = best
            parsed += 1
        except Exception:
            out[i] = np.nan
    if parsed == 0 or np.nanstd(out) == 0:
        return None
    return out


def build_derived_columns(problems_df: pd.DataFrame, answers_df: pd.DataFrame | None, task: str) -> dict:
    """Return {column_name: np.ndarray} of derived columns relevant for the correlation sweep.

    All arrays have len == len(problems_df) (BEFORE correctness masking; caller masks).
    """
    out: dict[str, np.ndarray] = {}
    df = problems_df

    # Interaction pairs
    for (a, b, name) in [
        ("carry_units", "carry_tens",          "carry_units_x_carry_tens"),
        ("a_units",     "b_units",             "a_units_x_b_units"),
        ("a_tens",      "b_tens",              "a_tens_x_b_tens"),
        ("a_units",     "b_tens",              "a_units_x_b_tens"),
        ("a_tens",      "b_units",             "a_tens_x_b_units"),
        ("a_parity",    "b_parity",            "a_parity_x_b_parity"),
        ("a_magnitude_tier", "b_magnitude_tier", "a_magtier_x_b_magtier"),
    ]:
        if a in df.columns and b in df.columns:
            v = _interaction(df[a], df[b])
            if v is not None:
                out[name] = v

    # Mod-10 sums
    for (a, b, name) in [
        ("a_units", "b_units", "a_units_plus_b_units_mod10"),
        ("a_tens",  "b_tens",  "a_tens_plus_b_tens_mod10"),
    ]:
        if a in df.columns and b in df.columns:
            v = _mod10(df[a], df[b])
            if v is not None:
                out[name] = v

    # Consecutive carry run if the carries list-column exists
    if "carries" in df.columns:
        v = _consecutive_carry_run(df["carries"])
        if v is not None:
            out["consecutive_carry_run"] = v

    # Multiplication-only
    if task == "multiplication":
        for (a, b, name) in [
            ("partial_product_units", "partial_product_tens", "pp_units_x_pp_tens"),
            ("column_sum_units",      "column_sum_tens",      "cs_units_x_cs_tens"),
            ("a_units", "b_units", "ab_units_product_mod100"),
        ]:
            if a in df.columns and b in df.columns:
                v = _interaction(df[a], df[b])
                if v is not None:
                    out[name if not name.endswith("_mod100") else name] = v % 100.0 if name.endswith("_mod100") else v

    # Predicted-answer features from answers CSV
    if answers_df is not None:
        for col in ["pred_first_token_text", "predicted", "model_prediction"]:
            if col in answers_df.columns:
                texts = answers_df[col].astype(str)
                # parse leading integer if any
                ints = pd.to_numeric(texts.str.extract(r"^\s*(-?\d+)", expand=False), errors="coerce").to_numpy()
                if not np.all(np.isnan(ints)):
                    out["predicted_value"] = ints
                    out["predicted_units"] = np.where(np.isnan(ints), np.nan, np.abs(ints) % 10)
                    out["predicted_tens"]  = np.where(np.isnan(ints), np.nan, (np.abs(ints) // 10) % 10)
                    out["predicted_n_digits"] = np.where(np.isnan(ints), np.nan, np.floor(np.log10(np.where(np.abs(ints) >= 1, np.abs(ints), 1))).astype(float) + 1)
                break

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Correlation sweep with 1000-permutation FDR null
# ──────────────────────────────────────────────────────────────────────────────

def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (Spearman convention). Returns float64 (N,) on rows without NaN."""
    return rankdata(x, method="average").astype(np.float64)


def _centred(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (centred, std) for Pearson normalisation."""
    xc = x - x.mean()
    s = float(np.sqrt((xc * xc).sum()))
    return xc, s


def _batched_corr(directions: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """For (n_dir, N) directions × (n_col, N) cols: return (n_dir, n_col) Pearson correlation.

    Both arrays must already be centred per-row; we re-center to be safe.
    Returns NaN where a column has zero variance.
    """
    D = directions - directions.mean(axis=1, keepdims=True)
    C = cols - cols.mean(axis=1, keepdims=True)
    d_norm = np.sqrt((D * D).sum(axis=1, keepdims=True))      # (n_dir, 1)
    c_norm = np.sqrt((C * C).sum(axis=1, keepdims=True))      # (n_col, 1)
    num = D @ C.T                                              # (n_dir, n_col)
    denom = d_norm @ c_norm.T                                  # (n_dir, n_col)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom > 0, num / denom, np.nan)
    return out


def correlation_sweep(
    X_resid: np.ndarray,
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    n_top: int,
    metadata_arrays: dict[str, np.ndarray],
    metadata_derived: set[str],
    seed: int,
    n_permutations: int = N_PERMUTATIONS,
) -> pd.DataFrame:
    """Spearman + Pearson sweep with batched 1000-perm FDR.

    Strategy: rank directions and columns once, batch all (dir, col) Pearson
    on ranks (=Spearman) and on raw (=Pearson). Permutation null: permute each
    column once per permutation, recompute batched Pearson(ranks_dir, perm_ranks_col)
    and Pearson(raw_dir, perm_raw_col); collect tail of |corr| ≥ |observed|.

    Returns a long-format DataFrame: direction_idx, eigenvalue, metadata_column,
    is_derived, spearman_rho, spearman_p_perm, spearman_q_fdr,
    pearson_r, pearson_p_perm, pearson_q_fdr, flag.
    """
    if eigenvectors.shape[0] == 0 or X_resid.shape[0] == 0 or not metadata_arrays:
        return pd.DataFrame()

    N = X_resid.shape[0]
    n_top = min(n_top, eigenvectors.shape[0])

    # Project residual onto top directions: (N, n_top)
    Z = X_resid @ eigenvectors[:n_top].T

    # Drop columns that are all-NaN or constant; coerce NaN→column-mean for finite stats
    col_names: list[str] = []
    col_raws: list[np.ndarray] = []
    for name, arr in metadata_arrays.items():
        a = np.asarray(arr, dtype=np.float64)
        if a.shape[0] != N:
            continue
        finite = np.isfinite(a)
        if finite.sum() < 10:
            continue
        if len(np.unique(a[finite])) < 2:
            continue
        # NaN-fill with mean (rare path; most cols clean)
        if not finite.all():
            a = a.copy()
            a[~finite] = a[finite].mean()
        col_names.append(name); col_raws.append(a)

    if not col_raws:
        return pd.DataFrame()

    # Raw matrices
    Z_raw = Z.astype(np.float64).T                            # (n_top, N)
    C_raw = np.stack(col_raws)                                # (n_col, N)

    # Rank matrices (for Spearman)
    Z_rank = np.stack([_ranks(z) for z in Z_raw])
    C_rank = np.stack([_ranks(c) for c in C_raw])

    # Observed correlations
    obs_sp = _batched_corr(Z_rank, C_rank)                    # (n_top, n_col)
    obs_pr = _batched_corr(Z_raw, C_raw)                      # (n_top, n_col)

    # Permutation null — universal across (dir, col), depends only on N for Spearman.
    # We use a tiered approach: one permutation matrix of column INDICES, shape (n_perm, N).
    # For each perm k, recompute batched correlation with permuted cols → (n_top, n_col).
    # Tally how often |perm_corr| ≥ |obs_corr|.
    rng = np.random.default_rng(seed)
    sp_tail = np.zeros_like(obs_sp, dtype=np.int64)
    pr_tail = np.zeros_like(obs_pr, dtype=np.int64)
    perm_chunk = 100
    n_done = 0
    while n_done < n_permutations:
        bs = min(perm_chunk, n_permutations - n_done)
        # generate bs permutations of columns; for each, batch-compute corr
        # We permute columns (cheap), keep Z fixed.
        for _ in range(bs):
            perm = rng.permutation(N)
            sp_pred = _batched_corr(Z_rank, C_rank[:, perm])
            pr_pred = _batched_corr(Z_raw, C_raw[:, perm])
            sp_tail += (np.abs(sp_pred) >= np.abs(obs_sp))
            pr_tail += (np.abs(pr_pred) >= np.abs(obs_pr))
        n_done += bs

    sp_p = (sp_tail + 1) / (n_permutations + 1)
    pr_p = (pr_tail + 1) / (n_permutations + 1)

    # BH FDR across the full (direction × col) grid, per metric.
    sp_q = false_discovery_control(sp_p.ravel(), method="bh").reshape(sp_p.shape)
    pr_q = false_discovery_control(pr_p.ravel(), method="bh").reshape(pr_p.shape)

    # Long-form DataFrame
    rows = []
    for i in range(n_top):
        for j, name in enumerate(col_names):
            rho_s = float(obs_sp[i, j]) if np.isfinite(obs_sp[i, j]) else float("nan")
            rho_p = float(obs_pr[i, j]) if np.isfinite(obs_pr[i, j]) else float("nan")
            flag = (abs(rho_s) > CORR_FLAG_THRESHOLD) and (sp_q[i, j] < FDR_ALPHA)
            rows.append({
                "direction_idx": i,
                "eigenvalue": float(eigenvalues[i]) if i < len(eigenvalues) else float("nan"),
                "metadata_column": name,
                "is_derived": name in metadata_derived,
                "spearman_rho": rho_s,
                "spearman_p_perm": float(sp_p[i, j]),
                "spearman_q_fdr": float(sp_q[i, j]),
                "pearson_r": rho_p,
                "pearson_p_perm": float(pr_p[i, j]),
                "pearson_q_fdr": float(pr_q[i, j]),
                "flag": bool(flag),
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 correlate-set unions
# ──────────────────────────────────────────────────────────────────────────────

def build_stage3_unions(
    results_root: Path, model: str, task: str, mode: str, layer: int,
    eligible: set[str], filter_dict: dict, out_dir: Path,
    logger: logging.Logger,
) -> list[dict]:
    """For each (target → correlate-set) defined in plan §4.2-4.3 for this task,
    build the union of correlate bases (LDA-A) and save under stage3_unions/.
    """
    sets = STAGE3_CORRELATE_SETS.get(task, {})
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for target, correlates in sets.items():
        kept = [c for c in correlates if c in eligible]
        skipped = [c for c in correlates if c not in eligible]
        if not kept:
            rows.append({"target": target, "k": 0, "kept": [], "skipped": skipped})
            continue
        bases = []
        for c in kept:
            B = load_basis_rows(lda_a_basis_path(results_root, model, task, layer, c, mode))
            if B.shape[0] > 0:
                bases.append(B)
        if not bases:
            rows.append({"target": target, "k": 0, "kept": kept, "skipped": skipped})
            continue
        stacked = np.vstack(bases)
        U, S, Vt = np.linalg.svd(stacked, full_matrices=False)
        keep = S > SVD_TOLERANCE_FACTOR * S[0]
        V = Vt[keep].astype(np.float32)
        atomic_save(V, out_dir / f"union_correlates_{target}.npy")
        meta = {
            "target": target, "task": task, "mode": mode, "layer": layer,
            "model": model, "k": int(V.shape[0]), "kept": kept, "skipped": skipped,
            "stacked_dim": int(stacked.shape[0]),
        }
        atomic_json(meta, out_dir / f"union_correlates_{target}_meta.json")
        rows.append({"target": target, "k": int(V.shape[0]), "kept": kept, "skipped": skipped})
        logger.debug(f"stage3 union {target}: k={V.shape[0]}, kept={kept}")
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Per-cell runner
# ──────────────────────────────────────────────────────────────────────────────

def run_cell(
    cfg: dict, model: str, task: str, mode: str, layer: int,
    problems_df: pd.DataFrame, answers_df: pd.DataFrame, correct_mask: np.ndarray,
    X_correct: np.ndarray, answer_scalar_correct: np.ndarray,
    logger: logging.Logger,
) -> dict:
    results_root = Path(cfg["paths"]["results_root"])
    cell_dir = results_root / "residual_hunting" / model / task / f"mode_{mode}" / f"layer_{layer:02d}"
    meta_path = cell_dir / "metadata.json"
    if meta_path.exists():
        try:
            cached = json.loads(meta_path.read_text())
            if cached.get("computation_status") == "complete":
                logger.info(f"[skip] cached {model}/{task}/mode_{mode}/layer_{layer:02d}")
                return cached.get("summary_rows", [])
        except Exception:
            pass

    t0 = time.time()
    cell_dir.mkdir(parents=True, exist_ok=True)
    filter_dict = load_concept_filter(results_root, model, task, mode, layer)
    eligible = eligible_concepts(filter_dict)
    logger.info(f"[run] {model}/{task}/mode_{mode}/layer_{layer:02d}: N={X_correct.shape[0]}, eligible_concepts={len(eligible)}")

    # Derived metadata columns (pre-correctness-mask, then mask)
    derived = build_derived_columns(problems_df, answers_df, task)
    # Combine raw + derived; apply correctness mask
    metadata_arrays: dict[str, np.ndarray] = {}
    raw_skip = set(SKIP_COLUMNS)
    for c in problems_df.columns:
        if c in raw_skip: continue
        col = problems_df[c]
        v = _safe_num(col)
        if v is None: continue
        metadata_arrays[c] = v[correct_mask]
    derived_names: set[str] = set()
    for name, arr in derived.items():
        if arr.shape[0] != len(correct_mask):
            continue
        metadata_arrays[name] = arr[correct_mask]
        derived_names.add(name)
    logger.debug(f"  metadata columns: {len(metadata_arrays)} (derived: {len(derived_names)})")

    seed = cell_seed(model, task, layer, f"residual_hunting_{mode}", base_seed=cfg["lda"]["random_state"])

    # Build both variants
    summary_rows = []
    union_meta_all = {}
    for variant in ("merged", "generous"):
        V_all, umeta = build_union(results_root, model, task, mode, layer, eligible, variant,
                                   X_correct, answer_scalar_correct, logger)
        atomic_save(V_all, cell_dir / f"union_basis_{variant}.npy")
        union_meta_all[variant] = umeta
        logger.info(f"  variant={variant}: k={umeta['k']} (stacked={umeta['stacked_dim']}, "
                    f"redundancy_removed={umeta.get('redundancy_removed', 0)})")

        if V_all.shape[0] == 0:
            summary_rows.append({
                "model": model, "task": task, "mode": mode, "layer": int(layer),
                "variant": variant, "k_union": 0, "status": "empty_union",
                "runtime_seconds": round(time.time() - t0, 2),
            })
            continue

        X_resid, var_orig, var_resid, var_explained = project_and_residual(X_correct, V_all)
        # Sanity check
        assert var_resid <= var_orig * 1.001 + 1e-3, f"var_resid {var_resid} > var_orig {var_orig}"

        # PCA + MP
        d_residual = 4096 - V_all.shape[0]
        pca_info = pca_with_mp(X_resid, d_residual, seed)
        atomic_save(pca_info["eigenvalues"], cell_dir / f"eigenvalues_{variant}.npy")
        atomic_save(pca_info["eigenvectors"], cell_dir / f"eigenvectors_{variant}.npy")
        mp_info_json = {k: v for k, v in pca_info.items() if k not in ("eigenvalues", "eigenvectors")}
        mp_info_json["variant"] = variant
        mp_info_json["var_orig"] = float(var_orig)
        mp_info_json["var_resid"] = float(var_resid)
        mp_info_json["var_explained"] = float(var_explained)
        atomic_json(mp_info_json, cell_dir / f"mp_info_{variant}.json")

        # Correlation sweep ONLY on merged (per plan §7.5; generous is audit-only)
        n_corr_flags = 0
        top_corr_concept = ""
        top_corr_rho = 0.0
        top_corr_q = 1.0
        if variant == "merged":
            n_top = max(pca_info["n_above_mp"], N_TOP_DIRECTIONS_IF_NULL_MP)
            n_top = min(n_top, pca_info["eigenvectors"].shape[0])
            t_sweep = time.time()
            corr_df = correlation_sweep(
                X_resid, pca_info["eigenvectors"], pca_info["eigenvalues"],
                n_top, metadata_arrays, derived_names, seed=seed,
            )
            logger.info(f"  correlation_sweep on {n_top} directions × {len(metadata_arrays)} cols in {time.time()-t_sweep:.1f}s")
            if len(corr_df):
                corr_df.to_csv(cell_dir / f"correlation_sweep_{variant}.csv", index=False)
                flagged = corr_df[corr_df["flag"]]
                n_corr_flags = int(len(flagged))
                if len(flagged):
                    best = flagged.iloc[flagged["spearman_rho"].abs().argmax()]
                    top_corr_concept = str(best["metadata_column"])
                    top_corr_rho = float(best["spearman_rho"])
                    top_corr_q = float(best["spearman_q_fdr"])

        summary_rows.append({
            "model": model, "task": task, "mode": mode, "layer": int(layer),
            "variant": variant,
            "k_union": int(V_all.shape[0]),
            "stacked_dim": int(umeta["stacked_dim"]),
            "n_concepts_kept": int(umeta["n_concepts_kept"]),
            "n_concepts_carved": int(sum(1 for s in filter_dict.values() if s["is_carved_out"])),
            "N": int(X_correct.shape[0]),
            "d_residual": int(d_residual),
            "gamma": float(pca_info["gamma"]),
            "sigma_sq": float(pca_info["sigma_sq"]),
            "lambda_max_mp": float(pca_info["lambda_max_mp"]),
            "top_eigenvalue": float(pca_info["top_eigenvalue"]),
            "n_above_mp": int(pca_info["n_above_mp"]),
            "mp_reliable_flag": bool(pca_info["mp_reliable_flag"]),
            "var_explained": float(var_explained),
            "var_residual": float(var_resid),
            "n_correlation_flags": int(n_corr_flags),
            "top_corr_concept": top_corr_concept,
            "top_corr_rho": float(top_corr_rho),
            "top_corr_q": float(top_corr_q),
            "runtime_seconds": round(time.time() - t0, 2),
            "seed": int(seed),
            "status": "fit_ok",
        })

    # Stage 3 correlate-set unions
    stage3_rows = build_stage3_unions(results_root, model, task, mode, layer,
                                       set(eligible), filter_dict, cell_dir / "stage3_unions", logger)

    # Union metadata + per-cell metadata
    atomic_json({
        "model": model, "task": task, "mode": mode, "layer": int(layer),
        "eligible_concepts": eligible,
        "n_eligible": int(len(eligible)),
        "filter_summary": filter_dict,
        "variants": union_meta_all,
        "stage3_unions": stage3_rows,
        "answer_scalar_sha256": None,
    }, cell_dir / "union_meta.json")

    atomic_json({
        "model": model, "task": task, "mode": mode, "layer": int(layer),
        "computation_status": "complete",
        "summary_rows": summary_rows,
        "runtime_seconds": round(time.time() - t0, 2),
    }, meta_path)

    logger.info(f"[done] {model}/{task}/mode_{mode}/layer_{layer:02d} "
                f"in {time.time()-t0:.1f}s")
    return summary_rows


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=["addition", "multiplication"])
    ap.add_argument("--mode", required=True, choices=["off", "answer", "norm"])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--layer", type=int)
    g.add_argument("--all-layers", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    paths = cfg["paths"]
    logs_root = Path(paths["logs_root"])
    results_root = Path(paths["results_root"])
    data_root = Path(paths["data_root"])
    model_cfg = next(m for m in cfg["models"] if m["key"] == args.model)
    layers = [args.layer] if args.layer is not None else model_cfg["layers"]

    logger = setup_logging(logs_root, args.model, args.task, args.mode)
    logger.info(f"=== Step 7 residual_hunting: model={args.model} task={args.task} mode={args.mode} layers={layers} ===")
    logger.info(f"cupy: {'AVAILABLE' if _HAS_CUPY else 'NOT available — CPU only'}")

    # Load per-task tables (constant across layers)
    problems_df = pd.read_csv(data_root / "data" / "raw" / f"{args.task}_problems.csv")
    answers_df = pd.read_csv(data_root / "answers" / args.model / f"{args.task}_answers.csv")
    if len(answers_df) != len(problems_df):
        raise RuntimeError("answers/problems row count mismatch")
    correct_mask = answers_df["correct"].to_numpy().astype(bool)
    answer_scalar_full = pd.to_numeric(problems_df["answer"], errors="coerce").to_numpy(dtype=np.float64)
    answer_scalar_correct = answer_scalar_full[correct_mask]
    logger.info(f"N_correct={int(correct_mask.sum())} / N_total={len(problems_df)}")

    all_rows = []
    for layer in layers:
        if args.mode == "off":
            act_path = data_root / "activations" / args.model / f"{args.task}_layer_{layer:02d}.npy"
        else:
            act_path = results_root / "residualized" / args.model / f"{args.task}_layer_{layer:02d}_mode_{args.mode}.npy"
        if not act_path.exists():
            logger.warning(f"missing activations: {act_path} — skipping layer {layer}")
            continue
        X_full = np.load(act_path)
        # Both raw and residualized caches store the full task population; mask here.
        X_correct = np.ascontiguousarray(X_full[correct_mask].astype(np.float32))
        del X_full

        rows = run_cell(
            cfg, args.model, args.task, args.mode, layer,
            problems_df, answers_df, correct_mask,
            X_correct, answer_scalar_correct, logger,
        )
        all_rows.extend(rows)

    # Per-(model, task, mode) summary CSV (cumulative across layers in this invocation)
    if all_rows:
        per_model_dir = results_root / "residual_hunting" / args.model
        per_model_dir.mkdir(parents=True, exist_ok=True)
        out_csv = per_model_dir / f"summary_{args.model}_{args.task}_mode_{args.mode}.csv"
        df_new = pd.DataFrame(all_rows)
        if out_csv.exists():
            df_old = pd.read_csv(out_csv)
            key = ["model", "task", "mode", "layer", "variant"]
            df_old = df_old[~df_old.set_index(key).index.isin(df_new.set_index(key).index)]
            df_new = pd.concat([df_old, df_new], ignore_index=True)
        df_new = df_new.sort_values(["task", "mode", "layer", "variant"]).reset_index(drop=True)
        df_new.to_csv(out_csv, index=False)
        logger.info(f"wrote {out_csv} ({len(df_new)} rows)")

    logger.info(f"=== Step 7 residual_hunting DONE: {args.model}/{args.task}/{args.mode} ===")


if __name__ == "__main__":
    main()
