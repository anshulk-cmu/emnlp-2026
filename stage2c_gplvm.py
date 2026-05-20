#!/usr/bin/env python3
"""
Stage 2c — BSMI-R worker: Bayesian Shape Manifold Inference with Refusal.

Reference: docs/gplvm.md.

Per cell c = (model, task, mode, layer, concept):
    Stage 0   load activations on Union(LDA-A, CCSVD); per-label noise sigma_v^2
    Stage 1-2 build candidate shape priors K_0..K_6 with label-aligned latents
    Stage 3-4 mandatory shape-aware Bayesian evidence on the FULL point cloud
    Stage 5   multimodal Laplace integration over period(s)
    Stage 6   refined evidence audit for Tier-A candidates (importance-weighted)
    Stage 7   intrinsic dimension (TwoNN + Levina-Bickel + PCA-PR)
    Stage 8   persistent homology (NEVER hard-rejects)
    Stage 9   Fourier diagnostics (proposes periods only)
    Stage 10  posterior differential geometry (curvature, torsion, helix drift)
    Stage 11  label alignment (circular / Spearman correlation)
    Stage 12  holdout adequacy (within-label + leave-value-out)
    Stage 13  seed and prior stability (winner identity)
    Stage 14  1000-permutation null with alignment-augmented statistic
    Stage 15  (in aggregator) global BH-FDR
    Stage 16  full evidence vector
    Stage 17  Tier A/B/C/D decision

Pipeline mode: **full point cloud**. Every closed-form Bayesian-linear
evidence call indexes Phi at all N points (Phi[i] = Phi_per_value[label_i]).
The label-collapsed centroid path is mathematically equivalent (sufficient
statistics) but the point-cloud path uses all N points exactly.

The core principle: NO module hard-gates. All evidence flows into the Tier
decision in `decide_tier()`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

import stage2c_shapes as shapes_mod
import stage2c_modules as evidence_mod


# ─── Locked configuration ─────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
STAGE2C_CONFIG_DEFAULT = PROJECT_ROOT / "configs" / "stage2c.yaml"

# BSMI-R per-cell limits
MIN_GROUP_SIZE = 30            # n_v floor for a value to enter the cell
MIN_K_FOR_BSMIR = 4            # cells with fewer than this many present values
                               # short-circuit to Tier D / low_K
EVIDENCE_GAP_NATS = 5.0        # Tier-A "decisive" gap (Kass-Raftery very strong)
EVIDENCE_GAP_NATS_SMALL_K = 10.0  # raise the bar when K_present is small
ALIGNMENT_THRESHOLD = 0.5      # |rho| floor for Tier A
HOLDOUT_EPSILON = 1.10         # mse_winner <= 1.10 * min(mse_others)
PH_TOL_Z = 2.0                 # how many sigmas Betti must miss expectation
                               # before PH says "contradictory"
N_PERMUTATIONS = 10000      # raised from 1000 for ~10x tighter p-value resolution
FDR_ALPHA = 0.10
DEFAULT_SEEDS = (42, 43, 44)


# Verdict labels (mirror gplvm.md tier rules)
T_NAMED       = "tier_A_named_shape"           # individual shape decisive
T_NAMED_FAM   = "tier_A_named_family"          # family decisive, within-family ambiguous
T_FAMILY      = "tier_B_family"                # weaker family-level claim
T_DIM_ONLY    = "tier_C_dim_only"
T_REFUSE      = "tier_D_refuse"
T_LOW_K       = "tier_low_K"


# ─── Atomic IO helpers ────────────────────────────────────────────────────────

def atomic_save(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".npy", dir=str(path.parent))
    os.close(fd)
    np.save(tmp, arr)
    os.replace(tmp, path)


def atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".npz", dir=str(path.parent))
    os.close(fd)
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def atomic_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".json", dir=str(path.parent))
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".csv", dir=str(path.parent))
    os.close(fd)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def cell_seed(model: str, task: str, mode: str, layer: int, concept: str,
              shape: str, seed_idx: int) -> int:
    s = f"stage2c_bsmir|{model}|{task}|{mode}|{layer:02d}|{concept}|{shape}|{seed_idx}"
    h = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**63 - 1)


# ─── Config / paths (unchanged from old pipeline — preserves repo layout) ─────

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def derive_paths(cfg: dict) -> dict:
    data_root = Path(cfg["paths"]["data_root"])
    results_root = Path(cfg["paths"]["results_root"])
    activations_root = Path(cfg["paths"].get("activations_root",
                                                data_root / "activations"))
    return {
        "data_root": data_root,
        "results_root": results_root,
        "activations_root": activations_root,
        "logs_root": Path(cfg["paths"].get("logs_root", data_root / "logs")),
    }


def ccsvd_basis_path(results_root: Path, model: str, task: str,
                      layer: int, concept: str, mode: str) -> Path:
    if mode == "off":
        return (results_root / "ccsvd_subspaces" / model / task
                / f"layer_{layer:02d}" / concept / "basis.npy")
    return (results_root / "ccsvd_subspaces" / f"mode_{mode}" / model / task
            / f"layer_{layer:02d}" / concept / "basis.npy")


def lda_basis_path(results_root: Path, model: str, task: str,
                    layer: int, concept: str, mode: str) -> Path:
    return (results_root / "lda_subspaces" / "subspace_lda" / f"mode_{mode}"
            / model / task / f"layer_{layer:02d}" / concept
            / "lda_basis_full.npy")


def activation_path(activations_root: Path, results_root: Path,
                     model: str, task: str, layer: int, mode: str) -> Path:
    if mode == "off":
        return activations_root / model / f"{task}_layer_{layer:02d}.npy"
    return (results_root / "residualized" / model
            / f"{task}_layer_{layer:02d}_mode_{mode}.npy")


def stage2c_cell_dir(results_root: Path, model: str, task: str, mode: str,
                      layer: int, concept: str) -> Path:
    return (results_root / "stage2c_gplvm" / model / task / f"mode_{mode}"
            / f"layer_{layer:02d}" / concept)


def stage2a_summary_path(results_root: Path, model: str, task: str,
                          mode: str, variant: str) -> Path:
    return (results_root / "stage2a_fourier_helix" / model
            / f"summary_{model}_{task}_mode_{mode}_variant_{variant}.csv")


SVD_TOLERANCE_FACTOR = 1e-10


def build_union_basis(results_root: Path, model: str, task: str, mode: str,
                       layer: int, concept: str) -> Tuple[np.ndarray, dict]:
    """SVD-orthonormalised [LDA-A | CCSVD] union basis (unchanged from prior)."""
    paths_tried = []
    rows = []
    contributions = []
    p_lda = lda_basis_path(results_root, model, task, layer, concept, mode)
    if p_lda.exists():
        try:
            B_lda = np.load(p_lda)
            if B_lda.ndim == 2 and B_lda.shape[0] == 4096 and B_lda.shape[1] > 0:
                rows.append(B_lda.T)
                contributions.append({"source": "lda_a", "n_dims": int(B_lda.shape[1]),
                                       "path": str(p_lda),
                                       "sha256": hashlib.sha256(B_lda.tobytes()).hexdigest()[:16]})
        except Exception as e:
            paths_tried.append((str(p_lda), str(e)))
    p_ccsvd = ccsvd_basis_path(results_root, model, task, layer, concept, mode)
    if p_ccsvd.exists():
        try:
            B_ccsvd = np.load(p_ccsvd)
            if B_ccsvd.ndim == 2 and B_ccsvd.shape[0] == 4096 and B_ccsvd.shape[1] > 0:
                rows.append(B_ccsvd.T)
                contributions.append({"source": "ccsvd", "n_dims": int(B_ccsvd.shape[1]),
                                       "path": str(p_ccsvd),
                                       "sha256": hashlib.sha256(B_ccsvd.tobytes()).hexdigest()[:16]})
        except Exception as e:
            paths_tried.append((str(p_ccsvd), str(e)))
    if not rows:
        return (np.zeros((4096, 0), dtype=np.float32),
                {"k_u": 0, "contributions": contributions,
                 "paths_tried": paths_tried, "stacked_dim": 0,
                 "redundancy_removed": 0,
                 "lda_a_present": p_lda.exists(),
                 "ccsvd_present": p_ccsvd.exists()})
    stacked = np.vstack(rows).astype(np.float64)
    stacked_dim = stacked.shape[0]
    U, S, Vt = np.linalg.svd(stacked, full_matrices=False)
    keep = S > SVD_TOLERANCE_FACTOR * S[0]
    Vt_keep = Vt[keep].astype(np.float32)
    B_u = Vt_keep.T.astype(np.float32)
    meta = {
        "k_u": int(B_u.shape[1]),
        "stacked_dim": int(stacked_dim),
        "redundancy_removed": int(stacked_dim - B_u.shape[1]),
        "contributions": contributions,
        "top_singular_value": float(S[0]),
        "smallest_kept_singular_value": float(S[keep].min() if keep.any() else 0.0),
        "svd_tolerance_factor": SVD_TOLERANCE_FACTOR,
    }
    return B_u, meta


def read_stage2a_row(results_root: Path, model: str, task: str,
                      mode: str, variant: str, layer: int, concept: str
                      ) -> Optional[dict]:
    p = stage2a_summary_path(results_root, model, task, mode, variant)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    sub = df[(df["layer"].astype(int) == int(layer)) & (df["concept"] == concept)]
    if sub.empty:
        return None
    return sub.iloc[0].to_dict()


def stage2a_period(s2a_row: Optional[dict]) -> Optional[float]:
    if s2a_row is None:
        return None
    P = s2a_row.get("discovered_period")
    try:
        P = float(P)
    except Exception:
        return None
    if not (np.isfinite(P) and P > 1.5):
        return None
    return P


# ============================================================================
# Stage 0 — Sufficient statistics
# ============================================================================

def compute_sufficient_stats(Z: np.ndarray, label_codes: np.ndarray,
                              K_present: int) -> Dict:
    """Compress N points to per-label means + within-label scatter.

    Returns:
        Z_bar: (K_present, k_u) per-label means
        n_v:   (K_present,) counts per label
        S_v:   (K_present, k_u, k_u) within-label scatter
        sigma2_v: (K_present,) scalar centroid sampling variance (tr Lambda / k_u)
        kept_codes: list of v in 0..K_present-1 that have n_v >= MIN_GROUP_SIZE
    """
    Z = np.asarray(Z, dtype=np.float64)
    N, k_u = Z.shape
    Z_bar = np.zeros((K_present, k_u), dtype=np.float64)
    n_v = np.zeros(K_present, dtype=np.int64)
    S_v = np.zeros((K_present, k_u, k_u), dtype=np.float64)
    for v in range(K_present):
        mask = label_codes == v
        n_v[v] = int(mask.sum())
        if n_v[v] == 0:
            continue
        Yv = Z[mask]
        Z_bar[v] = Yv.mean(axis=0)
        d = Yv - Z_bar[v]
        S_v[v] = d.T @ d
    kept_codes = [int(v) for v in range(K_present) if n_v[v] >= MIN_GROUP_SIZE]
    sigma2_v = np.zeros(K_present, dtype=np.float64)
    for v in range(K_present):
        if n_v[v] > 1:
            sigma2_v[v] = float(np.trace(S_v[v]) / (max(n_v[v] - 1, 1) * k_u))
    return {
        "Z_bar": Z_bar,
        "n_v": n_v,
        "S_v": S_v,
        "sigma2_v": sigma2_v,
        "kept_codes": kept_codes,
    }


# ============================================================================
# Stage 3-4 — Bayesian shape evidence (closed-form, FULL POINT CLOUD)
# ============================================================================
#
# Under a Gaussian linear model on the full N points:
#     y_i = Phi(v_i; theta) @ W + epsilon_i,   epsilon_i ~ N(0, sigma_{v_i}^2 I)
# with conjugate prior W ~ N(0, alpha I_n_basis), the marginal likelihood
# integrating out W is closed-form (Bishop PRML Eq. 3.86).
#
# Compared to the label-collapsed centroid version, this builds an (N, n_basis)
# design matrix Phi by indexing per-label rows of Phi_per_value into N rows by
# `label_codes`. The cost scales linearly in N (N * n_basis^2 flops for PtP)
# rather than the n_basis^2 / k_u flops of the centroid path. For our cell
# sizes (N up to ~10k, n_basis up to 6) the closed-form remains fast (~3 ms
# per shape evaluation) but uses ALL points, not summary statistics.

def _shape_evidence_at_theta(Z_full: np.ndarray, label_codes: np.ndarray,
                              sigma2_v: np.ndarray,
                              shape, theta: Dict[str, float],
                              K_natural: int,
                              alpha_prior: float = 1.0) -> Dict:
    """Full point-cloud closed-form Bayesian evidence at one theta.

    Args:
        Z_full       : (N, r) point cloud in the union subspace.
        label_codes  : (N,) dense codes 0..K_present-1 assigning each point to a value.
        sigma2_v     : (K_present,) per-label noise variance.
        shape        : ShapePrior instance.
        theta        : dict of shape hyperparameters (e.g. {"P": 10.0}).
        K_natural    : total span of natural label codes (e.g. 10 for digits).
        alpha_prior  : variance of the conjugate prior on W.

    Returns dict with log_marg_lik, posterior W mean, latent at present values,
    residuals, and metadata.
    """
    N, r = Z_full.shape
    K_present = int(sigma2_v.shape[0])
    v_unique = np.arange(K_present, dtype=np.int64)
    Phi_per_value = shape.build_basis(theta, v_unique, K_natural)  # (K_present, n_basis)
    n_basis = int(Phi_per_value.shape[1])
    # Fix 4: per-column prior-variance multipliers (default = 1; K6 tightens
    # its cross-term columns to 0.1× so it can't mimic K4_Torus).
    col_factors = shape.column_alpha_factors(theta, K_natural).astype(np.float64)
    prior_var_per_col = np.clip(alpha_prior * col_factors, 1e-12, None)  # (n_basis,)
    Phi = Phi_per_value[label_codes]                                 # (N, n_basis)
    sigma2_i = sigma2_v[label_codes]
    prec_i = 1.0 / np.clip(sigma2_i, 1e-12, None)
    sw = np.sqrt(prec_i)
    Phi_w = Phi * sw[:, None]                                        # (N, n_basis)
    Y_w = Z_full * sw[:, None]                                       # (N, r)
    PtP = Phi_w.T @ Phi_w                                            # (n_basis, n_basis)
    A = np.diag(1.0 / prior_var_per_col) + PtP                       # column-wise prior precision
    try:
        L = np.linalg.cholesky(A + 1e-9 * np.eye(n_basis))
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(A + 1e-3 * np.eye(n_basis))
    PtY = Phi_w.T @ Y_w                                              # (n_basis, r)
    W_mean = np.linalg.solve(L.T, np.linalg.solve(L, PtY))
    Y_pred_w = Phi_w @ W_mean
    resid_w = Y_w - Y_pred_w
    logdet_A = 2.0 * np.log(np.diag(L)).sum()
    # Bishop 3.86 generalised to per-column prior variance:
    #     log p(Y | theta) = -0.5 N log(2π) + 0.5 Σ log(prec_i)
    #                        + 0.5 (Σ log(1/prior_var_per_col) - log|A|)
    #                        - 0.5 ||resid_w||^2
    const = -0.5 * N * math.log(2.0 * math.pi) + 0.5 * np.log(prec_i).sum()
    per_output = (const
                  + 0.5 * (-np.log(prior_var_per_col).sum() - logdet_A)
                  - 0.5 * (resid_w ** 2).sum(axis=0))
    log_marg_lik = float(per_output.sum())
    # Fix 5: BIC sanity check (prior-free, asymptotic Bayes-factor proxy).
    # Use the residual-only Gaussian log-likelihood at the posterior-mean W
    # as a proxy for max log-likelihood (the precision-weighted residuals are
    # standardised, so the per-output log-lik is just -0.5 |resid|^2 plus
    # the same Gaussian constants).
    log_lik_max = float(per_output.sum()
                          + 0.5 * np.log(prior_var_per_col).sum()
                          + 0.5 * logdet_A)
    BIC = -2.0 * log_lik_max + n_basis * math.log(max(N, 2))
    latent = shape.latent_from_basis(theta, v_unique, K_natural)
    return {
        "log_marg_lik": log_marg_lik,
        "log_lik_max": log_lik_max,
        "BIC": float(BIC),
        "W_mean": W_mean,
        "W_chol_factor": L,                       # A = L L^T, used to draw W samples
        "Phi_per_value": Phi_per_value,
        "latent": latent,
        "residuals_weighted": resid_w,
        "theta": dict(theta),
        "n_basis": n_basis,
        "N_used": int(N),
    }


def sample_posterior_W(W_mean: np.ndarray, L: np.ndarray,
                        n_samples: int = 50, rng_seed: int = 0) -> np.ndarray:
    """Draw posterior samples W_s ~ N(W_mean, A^{-1}) using A = L L^T.

    Returns (n_samples, n_basis, r). Each sample is W_mean + L^{-T} Z where
    Z ~ N(0, I_{n_basis, r}). Cost: O(n_samples * n_basis^2 * r).
    """
    rng = np.random.default_rng(rng_seed)
    n_basis, r = W_mean.shape
    Z = rng.standard_normal((n_samples, n_basis, r))
    # A^{-1} = L^{-T} L^{-1}; chol(A^{-1}) = L^{-T}. Solve L^T X = Z for X.
    samples = np.empty_like(Z)
    for s in range(n_samples):
        samples[s] = W_mean + np.linalg.solve(L.T, Z[s])
    return samples


def multimodal_laplace_evidence(Z_full, label_codes, sigma2_v, shape,
                                  K_natural,
                                  P_hat: Optional[float] = None,
                                  P_hat_2: Optional[float] = None,
                                  alpha_prior: float = 1.0,
                                  n_alias: int = 3
                                  ) -> Dict:
    """Stage 5: integrate full point-cloud evidence over period modes via
    multimodal Laplace.

        log Z_k ~= logsumexp_m [ log p(Y | theta_m) + log p(theta_m)
                                  + 0.5 * log(2 pi / |H_m|) ]

    We use a flat prior over period (so log p(theta_m) drops out as a constant),
    a uniform weight log(1 / M), and a smoothness "curvature correction" via
    local-mode discrete second differences of the per-theta evidence.
    """
    K_present = int(sigma2_v.shape[0])
    v_unique = np.arange(K_present, dtype=np.int64)
    proposals = shape.propose_thetas(v_unique, K_natural, P_hat=P_hat,
                                       P_hat_2=P_hat_2, n_alias=n_alias)
    if not proposals:
        proposals = [{}]
    per_theta = []
    for theta in proposals:
        try:
            r = _shape_evidence_at_theta(Z_full, label_codes, sigma2_v, shape,
                                          theta, K_natural, alpha_prior)
        except Exception:
            continue
        per_theta.append(r)
    if not per_theta:
        return {"log_evidence": float("-inf"), "best": None, "per_theta": []}
    lls = np.asarray([r["log_marg_lik"] for r in per_theta])
    # logsumexp over modes — uniform prior over proposed theta_m.
    M = lls.size
    log_evidence = float(_logsumexp(lls) - math.log(M))
    # Curvature correction: approximate H_m via discrete 2nd diff in P-space
    # when shape is periodic and we have >= 3 modes ordered by P.
    if shape.is_periodic and lls.size >= 3 and not shape.is_two_periodic:
        Ps = np.asarray([r["theta"].get("P", float("nan")) for r in per_theta])
        order = np.argsort(Ps)
        Ps_o = Ps[order]
        lls_o = lls[order]
        d2 = np.zeros_like(lls_o)
        for i in range(1, lls_o.size - 1):
            h = 0.5 * (Ps_o[i + 1] - Ps_o[i - 1])
            if h > 1e-6:
                d2[i] = (lls_o[i + 1] - 2 * lls_o[i] + lls_o[i - 1]) / (h ** 2)
        # very rough correction: add +0.5 log(2 pi / max(|d2|, eps)) to each mode
        eps = 1e-6
        corr = 0.5 * np.log(2.0 * math.pi / np.clip(np.abs(d2), eps, None))
        log_evidence = float(_logsumexp(lls_o + corr) - math.log(M))
    best_idx = int(np.argmax(lls))
    return {
        "log_evidence": log_evidence,
        "best": per_theta[best_idx],
        "per_theta": per_theta,
        "log_lik_per_theta": lls.tolist(),
    }


def _logsumexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return float("-inf")
    m = finite.max()
    return float(m + math.log(np.exp(finite - m).sum()))


def empirical_bayes_alpha(Z_full: np.ndarray, label_codes: np.ndarray,
                            sigma2_v: np.ndarray, shapes_list,
                            K_natural: int,
                            P_hat: Optional[float], P_hat_2: Optional[float],
                            alpha_grid: Tuple[float, ...] = (
                              1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1,
                              1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
                            ) -> Dict:
    """Fix 1 — Empirical-Bayes choice of W's prior variance alpha (single value
    per cell, shared across all shapes for a fair model comparison).

    For each candidate alpha, computes the joint marginal log-evidence over
    shapes under a uniform shape prior:

        log p(Y | alpha) = log[(1/K) sum_k p(Y | K_k; alpha)]

    Returns the alpha that maximises this, along with the per-alpha log-marginal
    table for transparency.
    """
    best_alpha = 1.0
    best_log_marg = float("-inf")
    log_marg_per_alpha = {}
    log_E_per_alpha = {}
    for alpha in alpha_grid:
        log_E = []
        for sh in shapes_list:
            try:
                res = multimodal_laplace_evidence(
                    Z_full, label_codes, sigma2_v, sh, K_natural,
                    P_hat=P_hat, P_hat_2=P_hat_2,
                    alpha_prior=alpha, n_alias=3)
                log_E.append(float(res["log_evidence"]))
            except Exception:
                continue
        if not log_E:
            continue
        log_marg = _logsumexp(np.asarray(log_E)) - math.log(len(log_E))
        log_marg_per_alpha[float(alpha)] = float(log_marg)
        log_E_per_alpha[float(alpha)] = log_E
        if log_marg > best_log_marg:
            best_log_marg = log_marg
            best_alpha = float(alpha)
    return {
        "alpha_hat": best_alpha,
        "log_marg_per_alpha": log_marg_per_alpha,
        "log_marg_max": float(best_log_marg),
        "alpha_grid": list(alpha_grid),
    }


# ============================================================================
# Stage 6 — Refined evidence audit (importance-weighted) for Tier-A candidates
# ============================================================================

def refined_evidence_audit(Z_full, label_codes, sigma2_v, shape,
                              K_natural, base_result: Dict,
                              n_samples: int = 64, rng_seed: int = 0,
                              alpha_prior: float = 1.0) -> Dict:
    """Importance-weighted refinement around the best theta found by multimodal
    Laplace, on the full point cloud. Adds local jitter to the period
    parameter and re-evaluates.

    Returns a tighter estimate of log Z_k with a standard error.
    """
    best = base_result.get("best")
    if best is None:
        return {"log_evidence_refined": float("-inf"),
                "se_log_evidence": float("inf")}
    rng = np.random.default_rng(rng_seed)
    base_theta = dict(best.get("theta", {}))
    lls = []
    for _ in range(n_samples):
        theta_s = dict(base_theta)
        if "P" in theta_s:
            theta_s["P"] = float(base_theta["P"]) * (1.0 + 0.05 * rng.normal())
        if "P2" in theta_s:
            theta_s["P2"] = float(base_theta["P2"]) * (1.0 + 0.05 * rng.normal())
        try:
            r = _shape_evidence_at_theta(Z_full, label_codes, sigma2_v, shape,
                                          theta_s, K_natural, alpha_prior)
            lls.append(r["log_marg_lik"])
        except Exception:
            continue
    if not lls:
        return {"log_evidence_refined": base_result["log_evidence"],
                "se_log_evidence": float("inf")}
    lls = np.asarray(lls, dtype=np.float64)
    log_mean = _logsumexp(lls) - math.log(lls.size)
    # bootstrap SE
    rng2 = np.random.default_rng(rng_seed + 1)
    boots = []
    for _ in range(100):
        idx = rng2.integers(0, lls.size, lls.size)
        boots.append(_logsumexp(lls[idx]) - math.log(lls.size))
    se = float(np.std(boots, ddof=1))
    return {
        "log_evidence_refined": float(log_mean),
        "se_log_evidence": se,
        "n_samples": int(lls.size),
    }


# ============================================================================
# Stage 13 — Seed and prior stability
# ============================================================================

def run_seeds_and_priors(Z_full, label_codes, sigma2_v, shape,
                           K_natural,
                           P_hat: Optional[float], P_hat_2: Optional[float],
                           seeds: Tuple[int, ...] = DEFAULT_SEEDS,
                           alpha_priors: Tuple[float, ...] = (0.1, 1.0, 10.0)
                           ) -> Dict:
    """Diagnostic per-shape sweep over (seed, alpha) combinations on the
    full point cloud.

    Kept for the evidence vector but **NOT** used for the Tier-A gate any
    more — Tweak 1 of docs/gplvm.md replaces the absolute log-evidence std
    rule with cross-shape `winner_identity_stable` (see
    `cross_shape_seed_stability` below).
    """
    results = []
    for s in seeds:
        for alpha in alpha_priors:
            rng = np.random.default_rng(s)
            P_h = P_hat * (1.0 + 0.05 * rng.normal()) if P_hat else P_hat
            P_h2 = P_hat_2 * (1.0 + 0.05 * rng.normal()) if P_hat_2 else P_hat_2
            res = multimodal_laplace_evidence(
                Z_full, label_codes, sigma2_v, shape, K_natural,
                P_hat=P_h, P_hat_2=P_h2, alpha_prior=alpha)
            results.append({"seed": int(s), "alpha": float(alpha),
                            "log_evidence": float(res["log_evidence"])})
    lEs = np.asarray([r["log_evidence"] for r in results])
    return {"per_seed_prior": results,
            "log_evidence_mean": float(np.nanmean(lEs)),
            "log_evidence_std": float(np.nanstd(lEs)),
            "stable_absolute_logE": bool(np.isfinite(lEs).all()
                                            and (lEs.std() < 2.0))}


def cross_shape_seed_stability(Z_full, label_codes, sigma2_v, shapes_list,
                                  K_natural,
                                  P_hat: Optional[float], P_hat_2: Optional[float],
                                  seeds: Tuple[int, ...] = DEFAULT_SEEDS,
                                  alpha_priors: Tuple[float, ...] = (0.1, 1.0, 10.0),
                                  majority_threshold: float = 7 / 9,
                                  ) -> Dict:
    """Tweak 1 — robust seed/prior stability via winner-identity.

    For each (seed, alpha) combination, compute the marginal evidence of
    every candidate shape and record which shape wins. The rule:

        winner_identity_stable = (all combinations pick the same argmax shape)

    This is invariant to prior-induced shifts in absolute log Z (the
    Bishop 3.86 `-½ n_basis log α` term moves every shape's log Z, but the
    *ranking* is preserved). It is the correct stability indicator for a
    BSMI-R Tier-A claim.
    """
    per_combo = []
    for s in seeds:
        for alpha in alpha_priors:
            rng = np.random.default_rng(s)
            P_h = P_hat * (1.0 + 0.05 * rng.normal()) if P_hat else P_hat
            P_h2 = P_hat_2 * (1.0 + 0.05 * rng.normal()) if P_hat_2 else P_hat_2
            log_E_per_shape = {}
            for sh in shapes_list:
                try:
                    res = multimodal_laplace_evidence(
                        Z_full, label_codes, sigma2_v, sh, K_natural,
                        P_hat=P_h, P_hat_2=P_h2, alpha_prior=alpha)
                    log_E_per_shape[sh.name] = float(res["log_evidence"])
                except Exception:
                    continue
            if not log_E_per_shape:
                continue
            ranked = sorted(log_E_per_shape.items(), key=lambda kv: kv[1],
                              reverse=True)
            winner = ranked[0][0]
            gap = (ranked[0][1] - ranked[1][1]) if len(ranked) >= 2 else 0.0
            per_combo.append({
                "seed": int(s), "alpha": float(alpha),
                "winner": winner,
                "evidence_gap": float(gap),
                "log_E_per_shape": log_E_per_shape,
            })
    if not per_combo:
        return {"per_combo": [], "winner_identity_stable": False,
                "winner_majority_stable": False,
                "family_identity_stable": False,
                "evidence_gap_stable": False,
                "majority_winner": "",
                "majority_family": "",
                "winners_seen": [], "families_seen": []}
    winners = [r["winner"] for r in per_combo]
    families = [shapes_mod.SHAPE_TO_FAMILY.get(w, "trivial") for w in winners]
    winners_seen = sorted(set(winners))
    families_seen = sorted(set(families))
    majority = max(winners_seen, key=lambda w: winners.count(w))
    majority_family = max(families_seen, key=lambda f: families.count(f))
    winner_identity_stable = (len(winners_seen) == 1)
    # Fix 3: relaxed stability — winner family identical AND winner-shape
    # majority above threshold (default 7/9 = 77.8 %).
    winner_majority_stable = (winners.count(majority) / len(winners)
                                >= majority_threshold)
    family_identity_stable = (len(families_seen) == 1)
    evidence_gap_stable = all(r["evidence_gap"] > 0 for r in per_combo)
    return {
        "per_combo": per_combo,
        "winners_seen": winners_seen,
        "families_seen": families_seen,
        "majority_winner": majority,
        "majority_family": majority_family,
        "winner_identity_stable": bool(winner_identity_stable),
        "winner_majority_stable": bool(winner_majority_stable),
        "family_identity_stable": bool(family_identity_stable),
        "evidence_gap_stable": bool(evidence_gap_stable),
        "majority_threshold": float(majority_threshold),
    }


def compute_family_evidences(log_E_per_shape: Dict[str, float]) -> Dict:
    """Fix 2 — family-level marginal log-evidence under a uniform-within-family
    prior:
        log p(Y | family) = -log |family| + logsumexp_k log p(Y | K_k)
    for k in family.

    Returns {family: log_E_family}, the winning family, and the family
    evidence gap (best − runner-up).
    """
    fam_logE = {}
    for fam, members in shapes_mod.FAMILY_TO_SHAPES.items():
        ms = [log_E_per_shape.get(m, float("-inf")) for m in members
              if m in log_E_per_shape]
        ms = [v for v in ms if math.isfinite(v)]
        if not ms:
            continue
        fam_logE[fam] = _logsumexp(np.asarray(ms)) - math.log(len(ms))
    if not fam_logE:
        return {"family_log_E": {}, "family_winner": "",
                "family_runner_up": "", "family_evidence_gap": 0.0}
    ranked = sorted(fam_logE.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "family_log_E": fam_logE,
        "family_winner": ranked[0][0],
        "family_runner_up": ranked[1][0] if len(ranked) >= 2 else "",
        "family_evidence_gap": float(ranked[0][1] - ranked[1][1])
                                if len(ranked) >= 2 else float("inf"),
        "family_ranking": [r[0] for r in ranked],
    }


# ============================================================================
# Stage 14 — 1000-permutation test (CPU reference + GPU-batched)
# ============================================================================

def _perm_first_theta(shape, v_codes, K_natural,
                       P_hat: Optional[float], P_hat_2: Optional[float]
                       ) -> Optional[Dict]:
    """The single theta proposal used by `multimodal_laplace_evidence(...n_alias=1)`."""
    proposals = shape.propose_thetas(v_codes, K_natural, P_hat=P_hat,
                                       P_hat_2=P_hat_2, n_alias=1)
    if not proposals:
        return None
    return proposals[0]


def permutation_test_for_cell_gpu(Z_full, label_codes, sigma2_v, K_natural,
                                    shapes_to_test, P_hat, P_hat_2,
                                    observed_T: float,
                                    alignment_lambda: float = 1000.0,
                                    B: int = N_PERMUTATIONS,
                                    rng_seed: int = 0,
                                    alpha_prior: float = 1.0,
                                    chunk_size: int = 2000,
                                    device: Optional[str] = None) -> Dict:
    """GPU-batched permutation test, mathematically identical to
    `permutation_test_for_cell` (same RNG → same permutations → same
    null statistic up to float-precision roundoff).

    Strategy: for each shape's single n_alias=1 theta, build Phi_per_value once,
    then evaluate B permutations in chunks on GPU using batched Cholesky.
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    dtype = torch.float64   # use fp64 to match CPU numerics closely

    rng = np.random.default_rng(rng_seed)
    K_present = int(sigma2_v.shape[0])
    N, r = Z_full.shape
    v_codes_true = np.arange(K_present, dtype=np.int64)

    # Generate ALL permutations on CPU (same RNG sequence as CPU path).
    pi_batch = np.empty((B, K_present), dtype=np.int64)
    for b in range(B):
        pi_batch[b] = rng.permutation(K_present)

    # Per-shape: single-theta basis + latent + col-factors.
    K0_shape = shapes_mod.SHAPE_REGISTRY["K0_Generic"]
    K0_theta = _perm_first_theta(K0_shape, v_codes_true, K_present,
                                    P_hat, P_hat_2) or {}
    K0_Phi_pv = K0_shape.build_basis(K0_theta, v_codes_true, K_present)
    K0_col = K0_shape.column_alpha_factors(K0_theta, K_present)

    named_specs = []
    for sh in shapes_to_test:
        if sh.name == "K0_Generic":
            continue
        theta = _perm_first_theta(sh, v_codes_true, K_present, P_hat, P_hat_2)
        if theta is None:
            continue
        Phi_pv  = sh.build_basis(theta, v_codes_true, K_present)
        latent  = sh.latent_from_basis(theta, v_codes_true, K_present)
        col_fac = sh.column_alpha_factors(theta, K_present)
        named_specs.append({"shape": sh, "name": sh.name,
                              "theta": dict(theta), "Phi_pv": Phi_pv,
                              "latent": latent, "col_factors": col_fac,
                              "n_basis": int(Phi_pv.shape[1])})
    if not named_specs:
        return {"p_value": float("nan"),
                "null_samples": [],
                "n_permutations_run": 0,
                "observed_T": float(observed_T),
                "alignment_lambda": float(alignment_lambda)}

    # Push fixed tensors to GPU once.
    Z_t           = torch.tensor(Z_full,      dtype=dtype, device=torch_device)
    sigma2_t      = torch.tensor(sigma2_v,    dtype=dtype, device=torch_device)
    label_codes_t = torch.tensor(label_codes, dtype=torch.long, device=torch_device)
    pi_batch_t    = torch.tensor(pi_batch,    dtype=torch.long, device=torch_device)
    Z_norm2_t     = (Z_t ** 2).sum(dim=-1)                       # (N,)
    twopi_log     = math.log(2.0 * math.pi)

    def _batched_log_marg(shape_name: str, Phi_pv_np: np.ndarray,
                            col_fac_np: np.ndarray, n_basis: int
                            ) -> np.ndarray:
        Phi_pv_t = torch.tensor(Phi_pv_np, dtype=dtype, device=torch_device)
        col_t    = torch.tensor(col_fac_np, dtype=dtype, device=torch_device)
        prior_var_t  = torch.clamp(alpha_prior * col_t, min=1e-12)         # (n_basis,)
        prior_prec_t = 1.0 / prior_var_t
        log_prior_var_sum = torch.log(prior_var_t).sum()
        eye = torch.eye(n_basis, dtype=dtype, device=torch_device)

        out = np.empty(B, dtype=np.float64)
        for b0 in range(0, B, chunk_size):
            b1 = min(b0 + chunk_size, B); cb = b1 - b0
            pi_chunk = pi_batch_t[b0:b1]                          # (cb, K)
            lc_perm  = pi_chunk[:, label_codes_t]                  # (cb, N)
            Phi_perm = Phi_pv_t[lc_perm]                           # (cb, N, n_basis)
            sigma_p  = sigma2_t[lc_perm]
            prec     = 1.0 / torch.clamp(sigma_p, min=1e-12)       # (cb, N)
            M        = Phi_perm * prec.unsqueeze(-1)               # (cb, N, n_basis)
            PtP = torch.einsum('bnk,bnj->bkj', M, Phi_perm)        # (cb, n_basis, n_basis)
            PtY = torch.einsum('bnk,nr->bkr',  M, Z_t)             # (cb, n_basis, r)
            A   = PtP + torch.diag(prior_prec_t)                   # broadcasts to (cb, nb, nb)
            try:
                L = torch.linalg.cholesky(A + 1e-9 * eye)
            except RuntimeError:
                L = torch.linalg.cholesky(A + 1e-3 * eye)
            W = torch.cholesky_solve(PtY, L)                        # (cb, n_basis, r)
            log_det_A = 2.0 * torch.log(
                torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)    # (cb,)
            Y_w_norm2 = torch.einsum('bn,n->b', prec, Z_norm2_t)    # (cb,)
            # ||resid||^2 = ||Y_w||^2 - 2 W^T PtY + W^T PtP W      (Bayesian, NOT OLS)
            WtPtY     = (W * PtY).sum(dim=(-1, -2))                 # (cb,)
            PtP_W     = torch.bmm(PtP, W)                            # (cb, n_basis, r)
            WtPtP_W   = (W * PtP_W).sum(dim=(-1, -2))                # (cb,)
            resid     = Y_w_norm2 - 2.0 * WtPtY + WtPtP_W            # (cb,)
            log_prec_sum = torch.log(prec).sum(dim=-1)              # (cb,)
            # Mirrors CPU log_marg_lik = per_output.sum() with r outputs:
            #   r * const  +  0.5 r (-log_prior_var_sum - log_det_A) - 0.5 * total_resid_sumsq
            const = -0.5 * N * twopi_log + 0.5 * log_prec_sum
            log_marg = (r * const
                         + 0.5 * r * (-log_prior_var_sum - log_det_A)
                         - 0.5 * resid)
            out[b0:b1] = log_marg.detach().cpu().numpy()
        return out

    # Evaluate K0 + all named shapes on GPU.
    log_E_K0_all    = _batched_log_marg("K0_Generic",
                                          K0_Phi_pv, K0_col,
                                          int(K0_Phi_pv.shape[1]))
    log_E_named_all = np.full((B, len(named_specs)), float("-inf"),
                                 dtype=np.float64)
    for s_idx, spec in enumerate(named_specs):
        log_E_named_all[:, s_idx] = _batched_log_marg(
            spec["name"], spec["Phi_pv"], spec["col_factors"], spec["n_basis"])

    # Per-perm: pick winner, compute alignment on CPU (cheap, K-length).
    null = np.empty(B, dtype=np.float64)
    for b in range(B):
        row = log_E_named_all[b]
        if not np.any(np.isfinite(row)):
            null[b] = float("nan"); continue
        best_idx = int(np.argmax(row))
        spec     = named_specs[best_idx]
        latent_remapped = spec["latent"][pi_batch[b]]
        align = evidence_mod.label_alignment(
            spec["shape"], spec["theta"], latent_remapped, v_codes_true,
            K_natural=K_present)
        a = float(abs(align.get("alignment_score", 0.0)))
        null[b] = (row[best_idx] - log_E_K0_all[b]) + alignment_lambda * a

    finite = null[np.isfinite(null)]
    if finite.size == 0:
        p = float("nan")
    else:
        p = (1 + int((finite >= observed_T).sum())) / (finite.size + 1)
    return {"p_value": float(p),
             "null_samples": null.tolist(),
             "n_permutations_run": int(B),
             "observed_T": float(observed_T),
             "alignment_lambda": float(alignment_lambda),
             "device_used": str(torch_device)}


def permutation_test_for_cell(Z_full, label_codes, sigma2_v, K_natural,
                                shapes_to_test, P_hat, P_hat_2,
                                observed_T: float,
                                alignment_lambda: float = 1000.0,
                                B: int = N_PERMUTATIONS,
                                rng_seed: int = 0,
                                alpha_prior: float = 1.0,
                                use_gpu: bool = True) -> Dict:
    """Tweak 2 — permutation null with strengthened statistic, on full points.

    For each permutation b:
        1. draw a permutation pi of the K_present unique label codes
        2. re-map each point's label: label_codes_perm[i] = pi[label_codes[i]]
        3. refit every named shape and K_0 using label_codes_perm
        4. compute the winning named shape's alignment vs the TRUE
           v_codes (un-permuted dense indices arange(K_present))
        5. T_b = (log_E_best_named - log_E_K0) + λ · |alignment_best_vs_TRUE|

    Under H_alt: alignment stays at ~0.99; under H_null: alignment crashes
    to ~0 because labels are scrambled. λ scales alignment up to the same
    magnitude as the evidence gap.

    If `use_gpu=True` and torch+CUDA are available, dispatches to the GPU-
    batched implementation (identical math, ~30-80x faster on large K).
    """
    if use_gpu:
        try:
            import torch  # noqa: F401
            if torch.cuda.is_available():
                return permutation_test_for_cell_gpu(
                    Z_full, label_codes, sigma2_v, K_natural,
                    shapes_to_test, P_hat, P_hat_2, observed_T,
                    alignment_lambda=alignment_lambda, B=B,
                    rng_seed=rng_seed, alpha_prior=alpha_prior)
        except Exception:
            pass  # fall through to CPU
    rng = np.random.default_rng(rng_seed)
    null = []
    K_shape = shapes_mod.SHAPE_REGISTRY["K0_Generic"]
    K_present = int(sigma2_v.shape[0])
    v_codes_true = np.arange(K_present, dtype=np.int64)
    for b in range(B):
        pi = rng.permutation(K_present)
        label_codes_perm = pi[label_codes]
        log_E_K0 = multimodal_laplace_evidence(
            Z_full, label_codes_perm, sigma2_v, K_shape, K_natural,
            P_hat=P_hat, P_hat_2=P_hat_2, n_alias=1,
            alpha_prior=alpha_prior)["log_evidence"]
        best_lE = float("-inf")
        best_shape = None
        best_theta: Dict = {}
        best_latent: Optional[np.ndarray] = None
        for sh in shapes_to_test:
            if sh.name == "K0_Generic":
                continue
            res = multimodal_laplace_evidence(
                Z_full, label_codes_perm, sigma2_v, sh, K_natural,
                P_hat=P_hat, P_hat_2=P_hat_2, n_alias=1,
                alpha_prior=alpha_prior)
            if res["best"] is None:
                continue
            lE = float(res["log_evidence"])
            if lE > best_lE:
                best_lE = lE
                best_shape = sh
                best_theta = dict(res["best"]["theta"])
                best_latent = res["best"]["latent"]
        if best_shape is None or best_latent is None:
            null.append(float("nan"))
            continue
        # latent under permutation is indexed by v_codes_true (0..K-1) because
        # latent_from_basis was called with v_unique = arange(K_present); the
        # PERMUTATION enters through label_codes_perm at fit time, NOT through
        # the latent's row order. To measure alignment "to the true labels",
        # we apply the inverse permutation to recover what each true-value
        # angle would be.
        latent_remapped = best_latent[pi]
        align = evidence_mod.label_alignment(
            best_shape, best_theta, latent_remapped, v_codes_true,
            K_natural=K_natural)
        a = float(abs(align.get("alignment_score", 0.0)))
        T_b = (best_lE - log_E_K0) + alignment_lambda * a
        null.append(float(T_b))
    null = np.asarray(null, dtype=np.float64)
    finite = null[np.isfinite(null)]
    if finite.size == 0:
        p = float("nan")
    else:
        p = (1 + int((finite >= observed_T).sum())) / (finite.size + 1)
    return {"p_value": float(p),
            "null_samples": null.tolist(),
            "n_permutations_run": int(B),
            "observed_T": float(observed_T),
            "alignment_lambda": float(alignment_lambda)}


# ============================================================================
# Stage 16/17 — Evidence vector + Tier decision
# ============================================================================

@dataclass
class CellResult:
    cell_id: str
    K_natural: int
    K_present: int
    N_used: int
    k_u: int

    # Per-shape evidence
    log_E: Dict[str, float] = field(default_factory=dict)
    log_E_refined: Dict[str, float] = field(default_factory=dict)
    log_E_se: Dict[str, float] = field(default_factory=dict)
    best_theta: Dict[str, Dict] = field(default_factory=dict)
    latent: Dict[str, np.ndarray] = field(default_factory=dict)
    W_mean: Dict[str, np.ndarray] = field(default_factory=dict)
    n_basis: Dict[str, int] = field(default_factory=dict)

    # Module outputs
    dim: Dict = field(default_factory=dict)
    ph: Dict = field(default_factory=dict)
    fourier: Dict = field(default_factory=dict)
    geom: Dict[str, Dict] = field(default_factory=dict)
    align: Dict[str, Dict] = field(default_factory=dict)
    holdout: Dict[str, Dict] = field(default_factory=dict)
    seed_stability: Dict[str, Dict] = field(default_factory=dict)
    cross_seed: Dict = field(default_factory=dict)   # Tweak 1: winner-identity sweep
    permutation: Dict = field(default_factory=dict)

    # Decision
    winner: str = ""
    runner_up: str = ""
    evidence_gap: float = 0.0
    tier: str = T_REFUSE
    tier_reason: str = ""
    # Fixes 1, 2, 5
    alpha_hat: float = 1.0                          # empirical-Bayes prior variance
    alpha_eb_details: Dict = field(default_factory=dict)
    family_winner: str = ""
    family_runner_up: str = ""
    family_evidence_gap: float = 0.0
    family_log_E: Dict[str, float] = field(default_factory=dict)
    claim_level: str = ""                            # "shape" | "family" | ""
    BIC: Dict[str, float] = field(default_factory=dict)


def decide_tier(cr: CellResult,
                  evidence_gap_thresh: float = EVIDENCE_GAP_NATS,
                  evidence_gap_thresh_small_K: float = EVIDENCE_GAP_NATS_SMALL_K,
                  alignment_thresh: float = ALIGNMENT_THRESHOLD,
                  holdout_eps: float = HOLDOUT_EPSILON,
                  fdr_alpha: float = FDR_ALPHA,
                  ph_tol_z: float = PH_TOL_Z) -> CellResult:
    """Apply the 8-condition Tier-A rule. Falls back to B/C/D as appropriate.

    The decision uses *only* the evidence already computed; no module gates
    were applied earlier (per docs/gplvm.md core principle).
    """
    if cr.K_present < MIN_K_FOR_BSMIR:
        cr.tier = T_LOW_K
        cr.tier_reason = f"K_present={cr.K_present} < {MIN_K_FOR_BSMIR}"
        return cr

    # Rank shapes by refined evidence (fall back to log_E if no refinement)
    names = list(cr.log_E.keys())
    if not names:
        cr.tier = T_REFUSE
        cr.tier_reason = "no shape evidence computed"
        return cr
    def score(name):
        return cr.log_E_refined.get(name, cr.log_E.get(name, float("-inf")))
    sorted_names = sorted(names, key=score, reverse=True)
    winner = sorted_names[0]
    runner_up = sorted_names[1] if len(sorted_names) > 1 else "K0_Generic"
    cr.winner = winner
    cr.runner_up = runner_up
    cr.evidence_gap = float(score(winner) - score(runner_up))

    # If K0_Generic wins outright (or wins over named shapes), we cannot make
    # a named-shape claim -> Tier B/C.
    if winner == "K0_Generic":
        return _decide_non_named(cr)

    # ─── Common signals used by both shape-level and family-level Tier A ──
    gap_thresh = (evidence_gap_thresh_small_K
                  if cr.K_present <= 6 else evidence_gap_thresh)
    se = cr.log_E_se.get(winner, 0.0)
    decisive_gap_shape = cr.evidence_gap - 1.96 * se >= gap_thresh
    decisive_gap_family = cr.family_evidence_gap >= gap_thresh

    align_pass = evidence_mod.alignment_pass(
        cr.align.get(winner, {}).get("alignment_score", 0.0),
        threshold=alignment_thresh)

    geom_status = cr.geom.get(winner, {}).get("geom_status", "uncertain")
    geom_ok = (geom_status != "contradictory")

    mse_win = cr.holdout.get(winner, {}).get("mse", float("nan"))
    mse_others = [h.get("mse", float("nan")) for n, h in cr.holdout.items()
                  if n != winner]
    holdout_ok = evidence_mod.holdout_pass(mse_win, mse_others,
                                             epsilon_factor=holdout_eps)

    perm_p = cr.permutation.get("p_value", 1.0)
    perm_ok_pre_fdr = perm_p <= max(fdr_alpha, 1e-3)

    ph_status = shapes_mod.topology_consistent(
        shapes_mod.SHAPE_REGISTRY[winner],
        cr.ph.get("betti_obs", (1, 0, 0)),
        cr.ph.get("betti_std", (1.0, 1.0, 1.0)),
        tol_z=ph_tol_z)
    ph_ok_shape = (ph_status != "contradictory")

    # ─── Shape-level Tier A (strictest) ───────────────────────────────────
    seeds_ok_shape = (cr.cross_seed.get("winner_identity_stable", False)
                       and cr.cross_seed.get("majority_winner") == winner)
    shape_reasons: List[str] = []
    if not decisive_gap_shape:
        shape_reasons.append(
            f"gap-1.96se={cr.evidence_gap - 1.96 * se:.2f}<{gap_thresh}")
    if not align_pass:
        shape_reasons.append("weak alignment")
    if not geom_ok:
        shape_reasons.append(f"geom={geom_status}")
    if not holdout_ok:
        shape_reasons.append("holdout inadequate")
    if not seeds_ok_shape:
        shape_reasons.append("shape winner unstable across seeds/priors")
    if not perm_ok_pre_fdr:
        shape_reasons.append(f"perm p={perm_p:.4f}")
    if not ph_ok_shape:
        shape_reasons.append(f"PH={ph_status}")
    if not shape_reasons:
        cr.tier = T_NAMED
        cr.claim_level = "shape"
        cr.tier_reason = "all 7 shape-level conditions met"
        return cr

    # ─── Family-level Tier A (Fix 2) ─────────────────────────────────────
    # PH check at family level: any family member with a non-contradictory
    # signature is enough.
    fam = cr.family_winner
    members = shapes_mod.FAMILY_TO_SHAPES.get(fam, [])
    ph_per_member = [
        shapes_mod.topology_consistent(
            shapes_mod.SHAPE_REGISTRY[m],
            cr.ph.get("betti_obs", (1, 0, 0)),
            cr.ph.get("betti_std", (1.0, 1.0, 1.0)),
            tol_z=ph_tol_z)
        for m in members if m in cr.log_E]
    ph_ok_family = any(s != "contradictory" for s in ph_per_member) if \
        ph_per_member else True
    # Fix 3: family identity stable AND winner shape stable above majority
    seeds_ok_family = (cr.cross_seed.get("family_identity_stable", False)
                       and cr.cross_seed.get("majority_family") == fam
                       and cr.cross_seed.get("winner_majority_stable", False))
    family_reasons: List[str] = []
    if not decisive_gap_family:
        family_reasons.append(
            f"family gap={cr.family_evidence_gap:.2f}<{gap_thresh}")
    if not align_pass:
        family_reasons.append("weak alignment in family's best shape")
    if not holdout_ok:
        family_reasons.append("holdout inadequate for family's best shape")
    if not seeds_ok_family:
        family_reasons.append("family winner unstable across seeds/priors")
    if not perm_ok_pre_fdr:
        family_reasons.append(f"perm p={perm_p:.4f}")
    if not ph_ok_family:
        family_reasons.append("PH contradicts every family member")
    if not family_reasons:
        cr.tier = T_NAMED_FAM
        cr.claim_level = "family"
        cr.tier_reason = (f"family-level Tier A ({fam}); within-family "
                           f"ambiguity (best shape={winner}); "
                           f"shape-level fail = {'; '.join(shape_reasons)}")
        return cr
    # Otherwise fall back
    combined = "shape: " + "; ".join(shape_reasons) + \
                " | family: " + "; ".join(family_reasons)
    return _decide_non_named(cr, fallback_reason=combined)


def _decide_non_named(cr: CellResult, fallback_reason: str = "") -> CellResult:
    """Tier B / C / D decision when Tier A fails."""
    # Tier B: a geometric family is decisive (any periodic / curved shape's
    # evidence dominates K_0 even if exact shape is ambiguous).
    if cr.log_E.get("K0_Generic", float("-inf")) != float("-inf"):
        any_named_beats_K0 = any(
            cr.log_E.get(n, float("-inf")) - cr.log_E.get("K0_Generic", float("-inf"))
                >= 2.0
            for n in cr.log_E if n != "K0_Generic")
    else:
        any_named_beats_K0 = False
    dim_stable = cr.dim.get("estimators_agree", False)

    if any_named_beats_K0 and cr.align.get(cr.winner, {}).get(
            "alignment_score", 0.0) >= 0.3:
        cr.tier = T_FAMILY
        cr.tier_reason = "family decisive vs K0; " + fallback_reason
        return cr
    if dim_stable:
        cr.tier = T_DIM_ONLY
        cr.tier_reason = "dim estimators agree; no named shape; " + fallback_reason
        return cr
    cr.tier = T_REFUSE
    cr.tier_reason = "no module agreement; " + fallback_reason
    return cr


# ============================================================================
# Main pipeline orchestrator (Stages 0-17)
# ============================================================================

def analyze_cell(Z: np.ndarray, label_codes: np.ndarray, K_natural: int,
                  model: str, task: str, mode: str, layer: int, concept: str,
                  union_meta: dict, mu_layer_source: str,
                  stage2a_row_lda: Optional[dict],
                  stage2a_row_ccsvd: Optional[dict],
                  bsmir_cfg: dict, logger: logging.Logger,
                  alpha_override: Optional[float] = None,
                  P_hat_override: Optional[float] = None,
                  P_hat_2_override: Optional[float] = None,
                  ) -> Dict:
    """Run BSMI-R Stages 0-17 on one cell.

    Overrides (used by Stage 3 ownership tests):
        alpha_override   — skip empirical-Bayes; use this alpha for every shape.
        P_hat_override   — skip Stage 9 Fourier; use this as P_hat seed.
        P_hat_2_override — skip Stage 9 second-axis; use this as P_hat_2 seed.
    """
    cell_id = f"{model}|{task}|{mode}|{layer:02d}|{concept}"
    K_present = int(label_codes.max() + 1) if label_codes.size > 0 else 0
    if K_present < MIN_K_FOR_BSMIR:
        return {"status": "low_K", "cell_id": cell_id,
                "K_natural": K_natural, "K_present": K_present,
                "k_u": int(union_meta["k_u"]), "N_used": int(Z.shape[0])}

    # Stage 0 — per-label noise estimates only (we operate on the full point
    # cloud, but still need sigma_v^2 for the heteroscedastic likelihood).
    logger.info("[%s] Stage 0: per-label sigma estimates (point-cloud mode)",
                  cell_id)
    stats = compute_sufficient_stats(Z, label_codes, K_present)
    Z_bar = stats["Z_bar"]
    sigma2_v = stats["sigma2_v"]
    kept_codes = stats["kept_codes"]
    if len(kept_codes) < MIN_K_FOR_BSMIR:
        return {"status": "low_K_after_min_group", "cell_id": cell_id,
                "K_natural": K_natural, "K_present": K_present,
                "k_u": int(union_meta["k_u"]), "N_used": int(Z.shape[0]),
                "kept_codes": kept_codes}

    # Restrict to kept labels in BOTH the point cloud and the noise array.
    keep = np.asarray(kept_codes, dtype=np.int64)
    point_mask = np.isin(label_codes, keep)
    Z_full_k = Z[point_mask]                                   # (N_kept, k_u)
    remap = -np.ones(K_present, dtype=np.int64)
    for i, v in enumerate(keep):
        remap[v] = i
    label_codes_k = remap[label_codes[point_mask]]             # (N_kept,) in 0..K_kept-1
    sigma2_v_k = sigma2_v[keep]                                 # (K_kept,)
    Z_bar_k = Z_bar[keep]                                       # for PH / Fourier diagnostics only
    v_codes_k = np.arange(keep.size, dtype=np.int64)

    cr = CellResult(cell_id=cell_id, K_natural=K_natural,
                     K_present=int(keep.size),
                     N_used=int(Z_full_k.shape[0]),
                     k_u=int(union_meta["k_u"]))

    # Stage 7 — intrinsic dimension (on full data, not centroids)
    logger.info("[%s] Stage 7: intrinsic dimension", cell_id)
    cr.dim = evidence_mod.intrinsic_dimension(Z)

    # Stage 8 — persistent homology (on per-label means + bootstrap)
    logger.info("[%s] Stage 8: persistent homology", cell_id)
    cr.ph = evidence_mod.persistent_homology(Z_bar_k, max_dim=2)

    # Stage 9 — Fourier diagnostics (always run for the evidence vector, even
    # when periods are overridden, so the diagnostic record stays complete).
    logger.info("[%s] Stage 9: fourier diagnostics", cell_id)
    cr.fourier = evidence_mod.fourier_diagnostics(Z_bar_k, v_codes_k, K_natural)
    P_hat = None
    P_hat_2 = None
    if P_hat_override is not None:
        P_hat = float(P_hat_override)
    else:
        # Prefer Stage 2a period; else Stage 9 Fourier estimate.
        s2a_P = stage2a_period(stage2a_row_lda) or stage2a_period(
            stage2a_row_ccsvd)
        if s2a_P is not None:
            P_hat = float(s2a_P)
        else:
            ph_P = cr.fourier.get("P_hat")
            if ph_P and np.isfinite(ph_P):
                P_hat = float(ph_P)
    if P_hat_2_override is not None:
        P_hat_2 = float(P_hat_2_override)
    elif cr.fourier.get("P_hat_2") and np.isfinite(cr.fourier["P_hat_2"]):
        P_hat_2 = float(cr.fourier["P_hat_2"])

    # Fix 1 — Empirical Bayes alpha (single value per cell, shared across
    # all shapes for a fair comparison). Stage 3 ownership locks alpha to the
    # raw cell's value via alpha_override so log Z values are apples-to-apples.
    if alpha_override is not None:
        alpha_hat = float(alpha_override)
        cr.alpha_hat = alpha_hat
        cr.alpha_eb_details = {"alpha_hat": alpha_hat,
                                 "alpha_locked_by_caller": True}
        logger.info("[%s] Stage 3pre: alpha locked to %.3f (caller override)",
                      cell_id, alpha_hat)
    else:
        logger.info("[%s] Stage 3pre: empirical-Bayes alpha", cell_id)
        eb = empirical_bayes_alpha(
            Z_full_k, label_codes_k, sigma2_v_k,
            list(shapes_mod.all_shapes()),
            K_natural=keep.size, P_hat=P_hat, P_hat_2=P_hat_2)
        alpha_hat = float(eb["alpha_hat"])
        cr.alpha_hat = alpha_hat
        cr.alpha_eb_details = eb
        logger.info("[%s]   alpha_hat = %.3f (log_marg over alpha: %s)",
                      cell_id, alpha_hat,
                      {k: round(v, 1) for k, v in
                          eb["log_marg_per_alpha"].items()})

    # Stage 3-5 — Bayesian shape evidence with multimodal Laplace period
    # integration, for every shape K_0..K_6, using empirical-Bayes alpha.
    logger.info("[%s] Stage 3-5: Bayesian shape evidence "
                 "(alpha=%.3f, P_hat=%s, P_hat_2=%s)",
                 cell_id, alpha_hat, P_hat, P_hat_2)
    shape_results: Dict[str, Dict] = {}
    for shape in shapes_mod.all_shapes():
        try:
            res = multimodal_laplace_evidence(
                Z_full_k, label_codes_k, sigma2_v_k, shape,
                K_natural=keep.size, P_hat=P_hat, P_hat_2=P_hat_2,
                alpha_prior=alpha_hat)
        except Exception as e:
            logger.warning("[%s] shape %s failed: %s", cell_id, shape.name, e)
            continue
        if res["best"] is None:
            continue
        shape_results[shape.name] = res
        cr.log_E[shape.name] = float(res["log_evidence"])
        cr.best_theta[shape.name] = dict(res["best"]["theta"])
        cr.latent[shape.name] = res["best"]["latent"]
        cr.W_mean[shape.name] = res["best"]["W_mean"]
        cr.n_basis[shape.name] = int(res["best"]["n_basis"])
        # Fix 5: BIC sanity-check column
        cr.BIC[shape.name] = float(res["best"].get("BIC", float("nan")))

    if not shape_results:
        return {"status": "no_shape_evidence", "cell_id": cell_id,
                "K_present": int(keep.size), "k_u": int(union_meta["k_u"]),
                "N_used": int(Z.shape[0])}

    # Stage 10 — posterior differential geometry per shape
    logger.info("[%s] Stage 10: posterior geometry", cell_id)
    for name, sr in shape_results.items():
        cr.geom[name] = evidence_mod.posterior_geometry(
            name, cr.latent[name], cr.W_mean[name])

    # Stage 11 — label alignment per shape
    logger.info("[%s] Stage 11: label alignment", cell_id)
    for name, sr in shape_results.items():
        shape = shapes_mod.SHAPE_REGISTRY[name]
        cr.align[name] = evidence_mod.label_alignment(
            shape, cr.best_theta[name], cr.latent[name], v_codes_k,
            K_natural=keep.size)

    # Stage 12 — holdout (within-label only; leave-value-out only when K>=8)
    logger.info("[%s] Stage 12: holdout adequacy", cell_id)
    for name, sr in shape_results.items():
        shape = shapes_mod.SHAPE_REGISTRY[name]
        theta_best = cr.best_theta[name]
        # Closure: refit on training point cloud, predict held-out points.
        def make_pred_fn(_shape=shape, _theta=theta_best):
            def fit_fn(Z_train, lbl_train, K_pres):
                # Re-estimate per-label sigma^2 on the training fold and refit.
                stats_tr = compute_sufficient_stats(Z_train, lbl_train, K_pres)
                sig_tr = stats_tr["sigma2_v"]
                fit = _shape_evidence_at_theta(
                    Z_train, lbl_train, sig_tr, _shape, _theta,
                    K_pres, alpha_prior=1.0)
                W = fit["W_mean"]
                def predict(v_query):
                    Phi_q = _shape.build_basis(_theta, np.asarray(v_query),
                                                  K_pres)
                    return Phi_q @ W
                return predict
            return fit_fn
        try:
            ho_within = evidence_mod.within_label_holdout(
                Z, label_codes, K_present, make_pred_fn(), n_folds=5)
            ho_lvo = evidence_mod.leave_value_out_holdout(
                Z, label_codes, K_present, make_pred_fn())
            cr.holdout[name] = {
                "mse": float(ho_within.get("mse", float("nan"))),
                "mse_std": float(ho_within.get("fold_mse_std", float("nan"))),
                "n_folds": int(ho_within.get("n_folds", 0)),
                "mse_lvo": float(ho_lvo.get("mse", float("nan"))),
                "mse_lvo_std": float(ho_lvo.get("fold_mse_std", float("nan"))),
                "lvo_n_folds": int(ho_lvo.get("n_folds", 0)),
                "lvo_status": ho_lvo.get("status", "n/a"),
            }
        except Exception as e:
            logger.warning("[%s] holdout for %s failed: %s", cell_id, name, e)
            cr.holdout[name] = {"mse": float("nan"), "mse_lvo": float("nan")}

    # Stage 13 — seed/prior stability.
    # Per-shape diagnostic sweep (kept for the evidence vector, NOT used
    # for the Tier-A gate). The decisive flag is the cross-shape
    # winner-identity stability below (Tweak 1).
    # Stage 13 stability is now centred at the empirical-Bayes alpha.
    # We perturb alpha by 0.5x and 2x around alpha_hat to verify alpha_hat
    # is a stable optimum, not a knife-edge.
    alpha_sweep = (max(alpha_hat * 0.5, 1e-3),
                    alpha_hat,
                    alpha_hat * 2.0)
    logger.info("[%s] Stage 13a: per-shape seed/prior diagnostic "
                  "(alpha sweep around %.3f)", cell_id, alpha_hat)
    for name, sr in shape_results.items():
        shape = shapes_mod.SHAPE_REGISTRY[name]
        cr.seed_stability[name] = run_seeds_and_priors(
            Z_full_k, label_codes_k, sigma2_v_k, shape, keep.size,
            P_hat=P_hat, P_hat_2=P_hat_2,
            alpha_priors=alpha_sweep)
    logger.info("[%s] Stage 13b: cross-shape winner-identity stability",
                  cell_id)
    cr.cross_seed = cross_shape_seed_stability(
        Z_full_k, label_codes_k, sigma2_v_k,
        [shapes_mod.SHAPE_REGISTRY[n] for n in shape_results.keys()],
        K_natural=keep.size,
        P_hat=P_hat, P_hat_2=P_hat_2,
        alpha_priors=alpha_sweep)

    logger.info("[%s] Stage 6: refined evidence audit (all shapes)", cell_id)
    n_audit = int(bsmir_cfg.get("n_refined_audit_samples", 256))
    for name in shape_results.keys():
        shape = shapes_mod.SHAPE_REGISTRY[name]
        refined = refined_evidence_audit(
            Z_full_k, label_codes_k, sigma2_v_k, shape, keep.size,
            shape_results[name], alpha_prior=alpha_hat, n_samples=n_audit)
        cr.log_E_refined[name] = float(refined.get("log_evidence_refined",
                                                       cr.log_E[name]))
        cr.log_E_se[name] = float(refined.get("se_log_evidence", 0.0))

    # Fix 2 — family-level Bayes factor on the refined evidences.
    refined_or_E = {n: cr.log_E_refined.get(n, cr.log_E.get(n, float("-inf")))
                    for n in cr.log_E}
    fam_info = compute_family_evidences(refined_or_E)
    cr.family_log_E = fam_info.get("family_log_E", {})
    cr.family_winner = fam_info.get("family_winner", "")
    cr.family_runner_up = fam_info.get("family_runner_up", "")
    cr.family_evidence_gap = float(fam_info.get("family_evidence_gap", 0.0))

    # Stage 14 — 1000-permutation test with Tweak 2 strengthened statistic.
    logger.info("[%s] Stage 14: %d permutations (T includes alignment term)",
                  cell_id, N_PERMUTATIONS)
    alignment_lambda = float(bsmir_cfg.get("perm_alignment_lambda", 1000.0))
    log_E_K0 = cr.log_E_refined.get("K0_Generic", cr.log_E.get(
        "K0_Generic", float("-inf")))
    # Winner evidence + alignment for observed_T (mirrors permutation formula)
    named_logEs = {n: cr.log_E_refined.get(n, cr.log_E.get(n, float("-inf")))
                   for n in cr.log_E if n != "K0_Generic"}
    if named_logEs:
        winner_name = max(named_logEs, key=named_logEs.get)
        log_E_best = named_logEs[winner_name]
        winner_align = float(abs(cr.align.get(winner_name, {}).get(
            "alignment_score", 0.0)))
    else:
        winner_name = ""
        log_E_best = float("-inf")
        winner_align = 0.0
    observed_T = (float(log_E_best - log_E_K0) + alignment_lambda * winner_align
                   if math.isfinite(log_E_best) and math.isfinite(log_E_K0)
                   else float("-inf"))
    cr.permutation = permutation_test_for_cell(
        Z_full_k, label_codes_k, sigma2_v_k, K_natural=keep.size,
        shapes_to_test=[shapes_mod.SHAPE_REGISTRY[n] for n in
                          shape_results.keys()],
        P_hat=P_hat, P_hat_2=P_hat_2,
        observed_T=observed_T,
        alignment_lambda=alignment_lambda, B=N_PERMUTATIONS,
        alpha_prior=alpha_hat)
    cr.permutation["winner_alignment_used"] = winner_align
    cr.permutation["winner_shape_at_perm_time"] = winner_name

    # Stage 16/17 — evidence vector + Tier decision
    cr = decide_tier(cr)

    return {
        "status": "ok",
        "cell_result": cr,
        "K_natural": K_natural, "K_present": int(keep.size),
        "k_u": int(union_meta["k_u"]), "N_used": int(Z_full_k.shape[0]),
        "P_hat": P_hat, "P_hat_2": P_hat_2,
        "kept_codes": kept_codes,
        "union_meta": union_meta,
        "mu_layer_source": mu_layer_source,
        "pipeline_mode": "point_cloud_v1",
    }


# ============================================================================
# Cell IO — writes artifacts + summary CSV
# ============================================================================

def write_cell_artifacts(out_dir: Path, result: Dict, logger: logging.Logger
                          ) -> None:
    """Persist the per-cell artifact set (per gplvm.md Stage 16)."""
    if result.get("status") != "ok":
        # write the minimal "this cell could not be analyzed" CSV
        row = _minimal_row(result)
        atomic_csv(pd.DataFrame([row]), out_dir / "gplvm_results.csv")
        atomic_json({"status": result.get("status"),
                      "computation_status": "complete",
                      "union_meta": result.get("union_meta", {})},
                     out_dir / "metadata.json")
        return

    cr: CellResult = result["cell_result"]

    summary_row = _summary_row_from_cr(cr, result)
    atomic_csv(pd.DataFrame([summary_row]), out_dir / "gplvm_results.csv")

    # Per-shape evidence
    evidence_table = []
    for name in cr.log_E:
        evidence_table.append({
            "shape": name,
            "family": shapes_mod.SHAPE_TO_FAMILY.get(name, "trivial"),
            "log_E": cr.log_E[name],
            "log_E_refined": cr.log_E_refined.get(name, float("nan")),
            "log_E_se": cr.log_E_se.get(name, float("nan")),
            "BIC": cr.BIC.get(name, float("nan")),
            "alignment_score": cr.align.get(name, {}).get("alignment_score",
                                                            float("nan")),
            "geom_status": cr.geom.get(name, {}).get("geom_status", "n/a"),
            "mse_holdout": cr.holdout.get(name, {}).get("mse", float("nan")),
            "mse_lvo": cr.holdout.get(name, {}).get("mse_lvo", float("nan")),
            "n_basis": cr.n_basis.get(name, 0),
            "best_theta": cr.best_theta.get(name, {}),
        })
    atomic_csv(pd.DataFrame(evidence_table),
                out_dir / "evidence_per_shape.csv")

    # Permutation null
    perm_arr = np.asarray(cr.permutation.get("null_samples", []),
                            dtype=np.float64)
    atomic_save(perm_arr, out_dir / "perm_null.npy")

    # Full evidence vector + metadata
    atomic_json({
        "cell_id": cr.cell_id,
        "K_natural": cr.K_natural,
        "K_present": cr.K_present,
        "N_used": cr.N_used,
        "k_u": cr.k_u,
        "winner_shape": cr.winner,
        "runner_up_shape": cr.runner_up,
        "evidence_gap": cr.evidence_gap,
        "tier": cr.tier,
        "tier_reason": cr.tier_reason,
        "dim_module": cr.dim,
        "ph_module": {
            "betti_obs": cr.ph.get("betti_obs"),
            "betti_std": cr.ph.get("betti_std"),
            "status": cr.ph.get("status"),
        },
        "fourier_module": cr.fourier,
        "geom_per_shape": cr.geom,
        "align_per_shape": cr.align,
        "holdout_per_shape": cr.holdout,
        "seed_stability_per_shape": cr.seed_stability,
        "cross_shape_seed_stability": cr.cross_seed,
        "alpha_hat": cr.alpha_hat,
        "alpha_eb_details": cr.alpha_eb_details,
        "family_winner": cr.family_winner,
        "family_runner_up": cr.family_runner_up,
        "family_evidence_gap": cr.family_evidence_gap,
        "family_log_E": cr.family_log_E,
        "claim_level": cr.claim_level,
        "BIC_per_shape": cr.BIC,
        "permutation": {
            "p_value": cr.permutation.get("p_value"),
            "observed_T": cr.permutation.get("observed_T"),
            "n_permutations_run": cr.permutation.get("n_permutations_run"),
            "alignment_lambda": cr.permutation.get("alignment_lambda"),
            "winner_alignment_used": cr.permutation.get("winner_alignment_used"),
        },
        "P_hat_stage9_or_stage2a": result.get("P_hat"),
        "P_hat_2": result.get("P_hat_2"),
        "kept_codes": result.get("kept_codes"),
        "union_meta": result.get("union_meta"),
        "mu_layer_source": result.get("mu_layer_source"),
        "pipeline_mode": result.get("pipeline_mode", "point_cloud_v1"),
        "computation_status": "complete",
    }, out_dir / "metadata.json")

    # Save winner-specific artifacts for downstream causal validation
    if cr.winner and cr.winner in cr.W_mean:
        atomic_save(cr.W_mean[cr.winner], out_dir / "W_winner.npy")
        atomic_save(cr.latent[cr.winner], out_dir / "latent_winner.npy")


def _summary_row_from_cr(cr: CellResult, result: Dict) -> Dict:
    """One-row CSV summary, matches docs/gplvm.md final summary table cols."""
    parts = cr.cell_id.split("|")
    model, task, mode, layer, concept = parts[0], parts[1], parts[2], parts[3], parts[4]
    P1 = result.get("P_hat", float("nan"))
    win = cr.winner
    return {
        "model": model,
        "task": task,
        "mode": mode,
        "layer": int(layer),
        "concept": concept,
        "K_present": cr.K_present,
        "N_used": cr.N_used,
        "k_u": cr.k_u,
        "best_shape": win,
        "best_shape_verdict": shapes_mod.SHAPE_TO_VERDICT.get(win, "unknown"),
        "tier": cr.tier,
        "tier_reason": cr.tier_reason,
        "logZ_best": cr.log_E_refined.get(win, cr.log_E.get(win, float("nan"))),
        "logZ_runnerup": cr.log_E_refined.get(cr.runner_up,
                                                  cr.log_E.get(cr.runner_up,
                                                                 float("nan"))),
        "logZ_K0": cr.log_E.get("K0_Generic", float("nan")),
        "evidence_gap": cr.evidence_gap,
        "evidence_gap_se": cr.log_E_se.get(win, 0.0),
        "dim_hat": cr.dim.get("d_hat", float("nan")),
        "dim_ci_low": cr.dim.get("d_ci_low", float("nan")),
        "dim_ci_high": cr.dim.get("d_ci_high", float("nan")),
        "dim_estimators_agree": cr.dim.get("estimators_agree", False),
        "PH_status": cr.ph.get("status", "n/a"),
        "Betti": cr.ph.get("betti_obs", None),
        "Fourier_period": P1,
        "Fourier_period_2": result.get("P_hat_2", float("nan")),
        "Fourier_two_axis": cr.fourier.get("two_axis", False),
        "geom_status": cr.geom.get(win, {}).get("geom_status", "n/a"),
        "alignment_score": cr.align.get(win, {}).get("alignment_score",
                                                       float("nan")),
        "holdout_mse": cr.holdout.get(win, {}).get("mse", float("nan")),
        "holdout_mse_lvo": cr.holdout.get(win, {}).get("mse_lvo", float("nan")),
        "seed_stable_absolute_logE": cr.seed_stability.get(win, {}).get(
            "stable_absolute_logE", False),
        "winner_identity_stable": cr.cross_seed.get(
            "winner_identity_stable", False),
        "winner_majority_stable": cr.cross_seed.get(
            "winner_majority_stable", False),
        "family_identity_stable": cr.cross_seed.get(
            "family_identity_stable", False),
        "majority_winner": cr.cross_seed.get("majority_winner", ""),
        "majority_family": cr.cross_seed.get("majority_family", ""),
        # Fixes 1, 2, 5
        "alpha_hat": cr.alpha_hat,
        "family_winner": cr.family_winner,
        "family_runner_up": cr.family_runner_up,
        "family_evidence_gap": cr.family_evidence_gap,
        "claim_level": cr.claim_level,
        "BIC_winner": cr.BIC.get(win, float("nan")),
        "BIC_K0": cr.BIC.get("K0_Generic", float("nan")),
        "perm_p": cr.permutation.get("p_value", float("nan")),
        "perm_n_run": cr.permutation.get("n_permutations_run", 0),
        # The winner's best theta (period etc.)
        "best_theta_P": cr.best_theta.get(win, {}).get("P", float("nan")),
        "best_theta_P2": cr.best_theta.get(win, {}).get("P2", float("nan")),
        # Causal-validation convenience: same column name as old pipeline
        "winner_kernel": win,
        "P_top1": cr.best_theta.get(win, {}).get("P", float("nan")),
        "P_top2": cr.best_theta.get(win, {}).get("P2", float("nan")),
        # Pre-FDR verdict; aggregator will downgrade after BH-FDR
        "verdict_pre_fdr": cr.tier,
        "verdict_post_fdr": cr.tier,
    }


def _minimal_row(result: Dict) -> Dict:
    parts = result.get("cell_id", "?|?|?|00|?").split("|")
    model, task, mode, layer, concept = (parts + ["?"] * 5)[:5]
    try:
        layer_i = int(layer)
    except Exception:
        layer_i = -1
    return {
        "model": model, "task": task, "mode": mode,
        "layer": layer_i, "concept": concept,
        "K_natural": result.get("K_natural", 0),
        "K_present": result.get("K_present", 0),
        "N_used": result.get("N_used", 0),
        "k_u": result.get("k_u", 0),
        "best_shape": "", "tier": T_LOW_K,
        "tier_reason": result.get("status", "?"),
        "verdict_pre_fdr": T_LOW_K, "verdict_post_fdr": T_LOW_K,
    }


def cell_complete(out_dir: Path) -> bool:
    meta = out_dir / "metadata.json"
    if not meta.exists():
        return False
    try:
        with open(meta) as f:
            j = json.load(f)
        return j.get("computation_status") == "complete"
    except Exception:
        return False


# ============================================================================
# CLI entry points
# ============================================================================

def setup_logging(logs_root: Path, model: str, task: str, mode: str
                    ) -> logging.Logger:
    logs_root.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(logs_root / f"stage2c_{model}_{task}_{mode}.log")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger = logging.getLogger(f"stage2c.{model}.{task}.{mode}")
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def discover_concepts_for_cell(results_root: Path, model: str, task: str,
                                 mode: str, layer: int,
                                 s2a_lda_df: Optional[pd.DataFrame],
                                 s2a_ccsvd_df: Optional[pd.DataFrame]
                                 ) -> List[str]:
    """Same eligibility filter as the old pipeline: union of concepts where any
    Stage 2a variant gave a non-degenerate verdict AND both LDA-A and CCSVD
    bases exist."""
    eligible_verdicts = {"helix", "circle", "none", "sparse_value_grid"}
    concepts: set = set()
    for df in (s2a_lda_df, s2a_ccsvd_df):
        if df is None:
            continue
        sub = df[(df["layer"].astype(int) == int(layer))
                  & (df["geometry_detected"].isin(eligible_verdicts))]
        for c in sub["concept"].unique():
            concepts.add(str(c))
    have = []
    for c in sorted(concepts):
        if (lda_basis_path(results_root, model, task, layer, c, mode).exists()
                and ccsvd_basis_path(results_root, model, task, layer, c,
                                       mode).exists()):
            have.append(c)
    return have


def _load_cell_inputs(paths, model, task, mode, layer, concept, prob_df,
                      correct_mask, X_all):
    """Project activations onto the union basis. Returns Z, label_codes,
    K_natural, union_meta, mu_layer."""
    labels_full = prob_df[concept].to_numpy()
    labels_correct = labels_full[correct_mask]
    unique_full_values = sorted(prob_df[concept].dropna().unique().tolist())
    K_natural = len(unique_full_values)
    value_to_code = {v: i for i, v in enumerate(unique_full_values)}
    label_codes = np.array(
        [value_to_code.get(v, -1) for v in labels_correct], dtype=np.int64)
    keep = label_codes >= 0
    X = X_all[correct_mask][keep].astype(np.float64)
    label_codes = label_codes[keep]
    B_u, union_meta = build_union_basis(
        paths["results_root"], model, task, mode, layer, concept)
    if union_meta["k_u"] < 1:
        return None
    mu_layer = X.mean(axis=0)
    Z = (X - mu_layer).dot(B_u.astype(np.float64))
    return Z, label_codes, K_natural, union_meta, mu_layer


def main_one_cell(args, cfg, paths, bsmir_cfg) -> int:
    model = args.model
    task = args.task
    mode = args.mode
    layer = int(args.layer)
    concept = args.concept
    logger = setup_logging(paths["logs_root"], model, task, mode)
    logger.info("=== BSMI-R single cell: %s/%s/mode_%s/layer_%d/%s ===",
                  model, task, mode, layer, concept)
    out_dir = stage2c_cell_dir(paths["results_root"], model, task, mode,
                                 layer, concept)
    if cell_complete(out_dir) and not args.force:
        logger.info("Cell already complete, skipping (use --force to overwrite).")
        return 0
    X_path = activation_path(paths["activations_root"], paths["results_root"],
                              model, task, layer, mode)
    if not X_path.exists():
        logger.error("Activation file missing: %s", X_path)
        return 1
    X_all = np.load(X_path)
    ans_path = paths["data_root"] / "answers" / model / f"{task}_answers.csv"
    if not ans_path.exists():
        logger.error("Answers file missing: %s", ans_path)
        return 1
    ans_df = pd.read_csv(ans_path)
    correct_mask = ans_df["correct"].astype(bool).to_numpy()
    if X_all.shape[0] != correct_mask.shape[0]:
        logger.error("Activation rows %d != answers rows %d",
                       X_all.shape[0], correct_mask.shape[0])
        return 1
    prob_path = paths["data_root"] / "data" / "raw" / f"{task}_problems.csv"
    if not prob_path.exists():
        logger.error("Problems file missing: %s", prob_path)
        return 1
    prob_df = pd.read_csv(prob_path)
    if concept not in prob_df.columns:
        logger.error("Concept %r not in problems CSV columns", concept)
        return 1
    loaded = _load_cell_inputs(paths, model, task, mode, layer, concept,
                                  prob_df, correct_mask, X_all)
    if loaded is None:
        logger.error("Union basis empty for %s — abort", concept)
        return 2
    Z, label_codes, K_natural, union_meta, mu_layer = loaded
    s2a_lda = read_stage2a_row(paths["results_root"], model, task, mode,
                                 "lda_a", layer, concept)
    s2a_ccsvd = read_stage2a_row(paths["results_root"], model, task, mode,
                                   "ccsvd", layer, concept)
    result = analyze_cell(
        Z, label_codes, K_natural, model, task, mode, layer, concept,
        union_meta, mu_layer_source="X.mean",
        stage2a_row_lda=s2a_lda, stage2a_row_ccsvd=s2a_ccsvd,
        bsmir_cfg=bsmir_cfg, logger=logger,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    result["cell_id"] = f"{model}|{task}|{mode}|{layer:02d}|{concept}"
    write_cell_artifacts(out_dir, result, logger)
    if result.get("status") == "ok":
        cr = result["cell_result"]
        logger.info("[%s] verdict=%s gap=%.2f (%s)",
                      cr.cell_id, cr.tier, cr.evidence_gap, cr.winner)
    return 0


def main_full_sweep(args, cfg, paths, bsmir_cfg) -> int:
    array_idx = int(args.array_task)
    array_size = max(1, int(args.array_size))
    model_tag = args.model or "all"
    task_tag = args.task or "all"
    mode_tag = args.mode if args.mode != "all" else "all"
    logger = setup_logging(paths["logs_root"], model_tag, task_tag, mode_tag)
    logger.info("=== BSMI-R full sweep: model=%s task=%s mode=%s "
                  "array_task=%d / %d  rebalance=%s ===", model_tag, task_tag,
                  mode_tag, array_idx, array_size,
                  bool(args.rebalance_incomplete))
    all_models = [m["key"] for m in cfg["models"]]
    models = [args.model] if args.model else all_models
    all_tasks = ["addition", "multiplication"]
    tasks = [args.task] if args.task else all_tasks
    all_modes = ["off", "answer", "norm"]
    modes = [args.mode] if args.mode and args.mode != "all" else all_modes
    n_done = n_skip = n_err = 0

    # Rebalance: pre-build the set of incomplete cell_ids assigned to this
    # stripe by INDEX in the sorted incomplete-cell list. This redistributes
    # remaining work evenly when resuming a partial sweep on fewer stripes
    # than the original launch.
    my_cells: Optional[set] = None
    if args.rebalance_incomplete:
        all_incomplete: List[str] = []
        for model in models:
            mcfg = next(m for m in cfg["models"] if m["key"] == model)
            for task in tasks:
                if not (paths["data_root"] / "answers" / model
                          / f"{task}_answers.csv").exists():
                    continue
                for mode in modes:
                    sl_p = stage2a_summary_path(paths["results_root"], model,
                                                  task, mode, "lda_a")
                    sc_p = stage2a_summary_path(paths["results_root"], model,
                                                  task, mode, "ccsvd")
                    sl_df = pd.read_csv(sl_p) if sl_p.exists() else None
                    sc_df = pd.read_csv(sc_p) if sc_p.exists() else None
                    for layer in mcfg["layers"]:
                        for concept in discover_concepts_for_cell(
                                paths["results_root"], model, task, mode, layer,
                                sl_df, sc_df):
                            out_dir = stage2c_cell_dir(paths["results_root"],
                                                         model, task, mode,
                                                         layer, concept)
                            if cell_complete(out_dir) and not args.force:
                                continue
                            cid = f"{model}|{task}|{mode}|{layer:02d}|{concept}"
                            all_incomplete.append(cid)
        all_incomplete.sort()
        my_cells = {cid for i, cid in enumerate(all_incomplete)
                     if i % array_size == array_idx}
        logger.info("Rebalance: %d incomplete cells total; this stripe owns %d",
                      len(all_incomplete), len(my_cells))
    for model in models:
        model_cfg = next(m for m in cfg["models"] if m["key"] == model)
        layers_for_model = model_cfg["layers"]
        for task in tasks:
            ans_path = paths["data_root"] / "answers" / model / f"{task}_answers.csv"
            if not ans_path.exists():
                continue
            ans_df = pd.read_csv(ans_path)
            correct_mask = ans_df["correct"].astype(bool).to_numpy()
            prob_path = paths["data_root"] / "data" / "raw" / f"{task}_problems.csv"
            if not prob_path.exists():
                continue
            prob_df = pd.read_csv(prob_path)
            for mode in modes:
                s2a_lda_path = stage2a_summary_path(paths["results_root"],
                                                      model, task, mode, "lda_a")
                s2a_ccsvd_path = stage2a_summary_path(paths["results_root"],
                                                        model, task, mode,
                                                        "ccsvd")
                s2a_lda_df = (pd.read_csv(s2a_lda_path)
                                if s2a_lda_path.exists() else None)
                s2a_ccsvd_df = (pd.read_csv(s2a_ccsvd_path)
                                  if s2a_ccsvd_path.exists() else None)
                for layer in layers_for_model:
                    X_path = activation_path(paths["activations_root"],
                                              paths["results_root"],
                                              model, task, layer, mode)
                    if not X_path.exists():
                        continue
                    X_all = None
                    concepts = discover_concepts_for_cell(
                        paths["results_root"], model, task, mode, layer,
                        s2a_lda_df, s2a_ccsvd_df)
                    for concept in concepts:
                        cell_id = f"{model}|{task}|{mode}|{layer:02d}|{concept}"
                        if my_cells is not None:
                            if cell_id not in my_cells:
                                continue
                        else:
                            stripe = int.from_bytes(
                                hashlib.sha256(cell_id.encode()).digest()[:4],
                                "big") % array_size
                            if stripe != array_idx:
                                continue
                        out_dir = stage2c_cell_dir(paths["results_root"],
                                                     model, task, mode, layer,
                                                     concept)
                        if cell_complete(out_dir) and not args.force:
                            n_skip += 1
                            continue
                        if concept not in prob_df.columns:
                            continue
                        if X_all is None:
                            X_all = np.load(X_path)
                        loaded = _load_cell_inputs(paths, model, task, mode,
                                                       layer, concept, prob_df,
                                                       correct_mask, X_all)
                        if loaded is None:
                            continue
                        Z, label_codes, K_natural, union_meta, mu_layer = loaded
                        s2a_lda = read_stage2a_row(paths["results_root"],
                                                     model, task, mode, "lda_a",
                                                     layer, concept)
                        s2a_ccsvd = read_stage2a_row(paths["results_root"],
                                                       model, task, mode,
                                                       "ccsvd", layer, concept)
                        logger.info(">>> [stripe %d/%d] %s", array_idx,
                                      array_size, cell_id)
                        try:
                            result = analyze_cell(
                                Z, label_codes, K_natural, model, task, mode,
                                layer, concept, union_meta,
                                mu_layer_source="X.mean",
                                stage2a_row_lda=s2a_lda,
                                stage2a_row_ccsvd=s2a_ccsvd,
                                bsmir_cfg=bsmir_cfg, logger=logger)
                        except Exception as e:
                            logger.exception("Cell %s failed: %s", cell_id, e)
                            n_err += 1
                            continue
                        out_dir.mkdir(parents=True, exist_ok=True)
                        result["cell_id"] = cell_id
                        write_cell_artifacts(out_dir, result, logger)
                        if result.get("status") == "ok":
                            cr = result["cell_result"]
                            logger.info(
                                "[%s] verdict=%s gap=%.2f winner=%s",
                                cell_id, cr.tier, cr.evidence_gap, cr.winner)
                        n_done += 1
    logger.info("=== Stripe %d done. processed=%d, skipped=%d, errors=%d ===",
                  array_idx, n_done, n_skip, n_err)
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--stage2c-config", default=str(STAGE2C_CONFIG_DEFAULT))
    p.add_argument("--model", default=None,
                    help="model key (e.g. gpt-j-6b); omit when using --sweep all")
    p.add_argument("--task", default=None,
                    choices=["addition", "multiplication", None],
                    help="task; omit when using --sweep all")
    p.add_argument("--mode", default="off",
                    choices=["off", "answer", "norm", "all"])
    p.add_argument("--layer", type=int, default=-1,
                    help="-1 = all layers in config")
    p.add_argument("--concept", default="",
                    help="single concept; if empty, iterate eligible concepts")
    p.add_argument("--force", action="store_true",
                    help="re-run cells already marked complete")
    p.add_argument("--device", default="auto")
    p.add_argument("--sweep", choices=["", "all"], default="",
                    help="'all' = iterate all eligible cells")
    p.add_argument("--array-task", type=int, default=0)
    p.add_argument("--array-size", type=int, default=1)
    p.add_argument("--rebalance-incomplete", action="store_true",
                    help="Assign INCOMPLETE cells to stripes by sorted-index "
                         "mod array_size instead of hash. Use this when "
                         "resuming a partial sweep on fewer stripes than "
                         "the original so the remaining work distributes "
                         "evenly.")
    return p


def main():
    args = build_argparser().parse_args()
    cfg = load_config(Path(args.config))
    bsmir_cfg = {}
    s2c_path = Path(args.stage2c_config)
    if s2c_path.exists():
        with open(s2c_path) as f:
            bsmir_cfg = yaml.safe_load(f) or {}
    paths = derive_paths(cfg)
    if args.sweep == "all":
        return main_full_sweep(args, cfg, paths, bsmir_cfg)
    if not args.model or not args.task:
        raise SystemExit("Either --sweep all OR (--model X --task Y) is required.")
    if args.concept:
        return main_one_cell(args, cfg, paths, bsmir_cfg)
    # iterate all concepts for the cohort
    return main_full_sweep(args, cfg, paths, bsmir_cfg)


if __name__ == "__main__":
    sys.exit(main() or 0)
