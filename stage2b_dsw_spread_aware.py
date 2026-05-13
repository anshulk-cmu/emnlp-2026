"""Stage 2b — Spread-aware Mahalanobis test (d_SW).

For each (model, task, mode, layer, concept, variant) cell that Stage 2a flagged
as eligible (helix / circle / none / sparse_value_grid):

  1. Project activations onto the cell's Stage 1 subspace (LDA-A or CCSVD).
  2. Group by concept value; keep values with n_v ≥ min_group_size.
  3. Per value v, fit Σ̂_v with adaptive shrinkage (sample / LW / OAS) based on n_v / r.
  4. Build pair-harmonised Σ⁺_uv = (Σ̂_u' + Σ̂_v')/2 + λ I (λ = 1e-6 · tr/r).
  5. Compute D_E (Euclidean centroid) and D_SW (Mahalanobis) K×K matrices.
  6. Headline statistic ρ_centroid = Spearman(vec_offdiag(D_E), vec_offdiag(D_SW)).
  7. Bootstrap CI: 1000 draws WITH replacement at size N; per-draw shrink re-eval.
  8. Whittle null: 1000 label permutations; null distribution of ρ.
  9. Verdict ladder (ρ_low dominant): spread_confirmed / spread_marginal /
     centroid_only_shape / insufficient_samples / low_K_after_filter / null_unstable.
 10. Confidence tier (orthogonal): HIGH / MEDIUM / LOW / DISCOVERY_ONLY.
 11. Write per-cell artefacts (atomic, resume-by-metadata).

Run on full data — no subsampling. Bootstrap is WITH replacement at size N
(≈63.2% unique points per draw, full N as fit population).

CLI:
  python stage2b_dsw_spread_aware.py --config /home/anshulk/emnlp2026/config.yaml \
      --model llama-3.1-8b --task addition --mode off --variant lda_a --layer 16
  python stage2b_dsw_spread_aware.py --config ... --all-layers
"""

import argparse
import hashlib
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
from scipy.stats import rankdata
from sklearn.covariance import ledoit_wolf

try:
    import cupy as cp
    _HAS_CUPY = cp.cuda.is_available()
except Exception:
    _HAS_CUPY = False
    cp = None

# Reuse loaders, paths, and projection from Stage 2a (single source of truth).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2a_fourier_helix import (
    AMBIENT_D,
    ccsvd_basis_path, ccsvd_meta_path,
    lda_a_basis_path, lda_a_meta_path,
    load_basis_matrix, load_mu_layer,
    load_concept_filter, K_natural_for_concept,
    project_to_subspace, sha256_of,
    cell_artifact_dir as stage2a_cell_dir,
)


# ──────────────────────────────────────────────────────────────────────────────
# Verdict labels and shrinkage modes
# ──────────────────────────────────────────────────────────────────────────────

VERDICT_SPREAD_CONFIRMED = "spread_confirmed"
VERDICT_SPREAD_MARGINAL = "spread_marginal"
VERDICT_CENTROID_ONLY = "centroid_only_shape"
VERDICT_INSUFFICIENT = "insufficient_samples"
VERDICT_LOW_K_AFTER_FILTER = "low_K_after_filter"
VERDICT_NULL_UNSTABLE_DSW = "null_unstable"

ALL_VERDICTS = {
    VERDICT_SPREAD_CONFIRMED, VERDICT_SPREAD_MARGINAL, VERDICT_CENTROID_ONLY,
    VERDICT_INSUFFICIENT, VERDICT_LOW_K_AFTER_FILTER, VERDICT_NULL_UNSTABLE_DSW,
}

TIER_HIGH = "HIGH"
TIER_MEDIUM = "MEDIUM"
TIER_LOW = "LOW"
TIER_DISCOVERY_ONLY = "DISCOVERY_ONLY"

SHRINK_SAMPLE = "sample"
SHRINK_LW = "lw"
SHRINK_OAS = "oas"
SHRINK_ORDER = {SHRINK_SAMPLE: 0, SHRINK_LW: 1, SHRINK_OAS: 2}

# Stage 2a verdict set we consume (mirror; sparse_value_grid IS in Stage 2a code).
STAGE2A_ELIGIBLE = {"helix", "circle", "none", "sparse_value_grid"}


# ──────────────────────────────────────────────────────────────────────────────
# Atomic I/O (mirrors stage2a_fourier_helix.py)
# ──────────────────────────────────────────────────────────────────────────────

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


def stage2b_seed(model_key: str, task: str, mode: str, layer: int, variant: str, concept: str) -> tuple[int, str]:
    s = f"stage2b|{model_key}|{task}|{mode}|{layer:02d}|{variant}|{concept}"
    h = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**63 - 1), s


# ══════════════════════════════════════════════════════════════════════════════
# PURE ALGORITHM FUNCTIONS — importable by check_stage2b_toys.py
# ══════════════════════════════════════════════════════════════════════════════

def select_shrink_mode(n_v: int, r: int, lw_threshold: float, oas_threshold: float) -> str:
    """Adaptive shrinkage mode based on ratio n_v/r."""
    if r <= 0:
        return SHRINK_OAS
    ratio = n_v / r
    if ratio >= lw_threshold:
        return SHRINK_SAMPLE
    if ratio >= oas_threshold:
        return SHRINK_LW
    return SHRINK_OAS


def fit_sigma_sample(R_v: np.ndarray) -> np.ndarray:
    """Sample covariance from centred residuals R_v (n_v, r). Returns (r, r) float64."""
    n_v = R_v.shape[0]
    if n_v <= 1:
        r = R_v.shape[1]
        return np.eye(r, dtype=np.float64) * 1e-12
    return (R_v.T.astype(np.float64) @ R_v.astype(np.float64)) / float(n_v - 1)


def fit_sigma_lw(R_v: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf covariance from centred residuals R_v. Returns (r, r) float64.

    Inline closed-form (Ledoit & Wolf 2004) — bypasses sklearn's input validation
    layer which dominates wall-time when called 10K+ times per cell."""
    R = np.asarray(R_v, dtype=np.float64)
    n_v, r = R.shape
    if n_v <= 1:
        return np.eye(r, dtype=np.float64) * 1e-12
    # Sample covariance (assume centered → no further centring).
    S = (R.T @ R) / float(n_v)
    # Shrinkage target: μ I where μ = trace(S) / r.
    mu = float(np.trace(S) / r)
    target = mu * np.eye(r, dtype=np.float64)
    # Estimator of |S − target|_F²
    d2 = float(np.sum((S - target) ** 2))
    # Estimator of E[|S − Σ|_F²] (asymptotic):
    #   bar_b² = (1/n²) Σ_i ||x_i x_i.T − S||_F²
    #   shrinkage = min(bar_b² / d², 1)
    # Vectorised: ||x_i x_iᵀ − S||² = (xᵀx)² − 2 xᵀS x + ||S||²
    XtX = np.einsum("ij,ij->i", R, R)              # (n_v,) = ||x_i||² (per-point)
    xSx = np.einsum("ij,jk,ik->i", R, S, R)        # (n_v,) = x_iᵀ S x_i
    S_norm2 = float(np.sum(S * S))
    per_point = XtX * XtX - 2.0 * xSx + S_norm2
    bar_b2 = float(np.sum(per_point) / (n_v * n_v))
    shrinkage = float(min(1.0, max(0.0, bar_b2 / max(d2, 1e-30))))
    # Scale shrinkage by sample size correction matching sklearn: keep simple here.
    return (1.0 - shrinkage) * S + shrinkage * target


def fit_sigma_oas(R_v: np.ndarray) -> tuple[np.ndarray, float]:
    """OAS shrinkage (Chen et al. 2010) — closed form. Returns (Σ, shrinkage_alpha)."""
    R = R_v.astype(np.float64, copy=False)
    n_v, r = R.shape
    if n_v <= 1:
        return np.eye(r, dtype=np.float64) * 1e-12, 1.0
    S_emp = (R.T @ R) / float(n_v)
    trace_S = float(np.trace(S_emp))
    trace_S2 = float(np.sum(S_emp * S_emp))
    mu = trace_S / max(r, 1)
    denom_inner = trace_S2 - (trace_S * trace_S) / max(r, 1)
    num = (1.0 - 2.0 / max(r, 1)) * trace_S2 + trace_S * trace_S
    den = (n_v + 1.0 - 2.0 / max(r, 1)) * max(denom_inner, 1e-30)
    shrinkage = float(min(1.0, max(0.0, num / den)))
    target = mu * np.eye(r, dtype=np.float64)
    Sigma = (1.0 - shrinkage) * S_emp + shrinkage * target
    return Sigma, shrinkage


def fit_sigma(R_v: np.ndarray, mode: str) -> tuple[np.ndarray, float]:
    """Dispatch to one of the three estimators. Returns (Σ, alpha) where alpha
    is the LW/OAS shrinkage strength (0.0 for sample mode)."""
    if mode == SHRINK_SAMPLE:
        return fit_sigma_sample(R_v), 0.0
    if mode == SHRINK_LW:
        return fit_sigma_lw(R_v), 0.0  # sklearn doesn't return alpha directly here
    return fit_sigma_oas(R_v)


def stricter_mode(m_u: str, m_v: str) -> str:
    return m_u if SHRINK_ORDER[m_u] >= SHRINK_ORDER[m_v] else m_v


def filter_values_by_count(label_codes: np.ndarray, K_natural: int,
                            min_group_size: int) -> tuple[np.ndarray, list[int]]:
    """Return (kept_codes, kept_value_indices) where kept_codes is dense 0..K_present-1
    over the points belonging to surviving values; kept_value_indices lists original
    value codes (0..K_natural-1) that survived."""
    counts = np.bincount(label_codes, minlength=K_natural)
    kept = [v for v in range(K_natural) if counts[v] >= min_group_size]
    kept_set = set(kept)
    mask = np.array([c in kept_set for c in label_codes])
    # Build a dense remap
    remap = {v: i for i, v in enumerate(kept)}
    new_codes = np.array([remap[c] for c in label_codes[mask]], dtype=np.int64)
    return new_codes, kept, mask


def fit_per_value_sigmas(Z: np.ndarray, label_codes: np.ndarray, K_present: int,
                          lw_threshold: float, oas_threshold: float
                          ) -> tuple[np.ndarray, list[np.ndarray], list[str], list[float], np.ndarray]:
    """For each value v in 0..K_present-1, fit centroid μ_v and Σ̂_v under
    adaptive shrinkage with per-pair harmonisation done via cell-wide stricter mode.

    Optimisation: rather than caching all three modes per value (3× cost), we
    compute the cell-wide stricter mode (max across all values' chosen modes) and
    fit each Σ_v in that single mode. This satisfies the plan's pair-harmonisation
    rule conservatively — every pair uses the strictest mode any value needed.

    Returns:
      mu_stack: (K_present, r) centroids
      sigma_stack: list of length K_present; each entry is Σ̂_v (r, r) at cell-mode
      shrink_mode_per_v: list of length K_present (the chosen-mode per value;
                          stored for audit; the ACTUAL Σ uses cell_mode)
      cell_mode: str — the single mode used for all Σ_v in this iter
      counts: (K_present,) int n_v
    """
    r = Z.shape[1]
    counts = np.bincount(label_codes, minlength=K_present)
    chosen_modes = [select_shrink_mode(int(counts[v]), r, lw_threshold, oas_threshold)
                    for v in range(K_present)]
    # Cell-wide stricter mode = max(chosen) — every pair will use this
    cell_mode = chosen_modes[0]
    for m in chosen_modes[1:]:
        cell_mode = stricter_mode(cell_mode, m)

    mu = np.zeros((K_present, r), dtype=np.float64)
    sigma_stack: list[np.ndarray] = []
    shrink_alpha: list[float] = []
    for v in range(K_present):
        idx = np.where(label_codes == v)[0]
        Z_v = Z[idx].astype(np.float64)
        mu[v] = Z_v.mean(axis=0)
        R_v = Z_v - mu[v]
        Sigma, alpha = fit_sigma(R_v, cell_mode)
        sigma_stack.append(Sigma)
        shrink_alpha.append(float(alpha))

    return mu, sigma_stack, chosen_modes, cell_mode, shrink_alpha, counts


def build_dsw_matrix(mu: np.ndarray,
                      sigma_stack: list[np.ndarray] | np.ndarray,
                      cell_mode: str,
                      lambda_factor: float
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Compute K×K Mahalanobis distance matrix in vectorised numpy.

    All Σ_v are at the cell-wide stricter mode, so pair Σ_pool = 0.5*(Σ_u + Σ_v)
    is consistent. We batch all K(K-1)/2 Cholesky-solves via numpy.linalg.cholesky
    on a stacked (n_pairs, r, r) tensor.

    Returns (D_SW, pair_mode_int) where pair_mode_int[u, v] = SHRINK_ORDER[cell_mode]
    everywhere (since all pairs use the same mode under cell-wide harmonisation).
    """
    K = mu.shape[0]
    r = mu.shape[1]
    if isinstance(sigma_stack, list):
        Sigma = np.stack([np.asarray(s, dtype=np.float64) for s in sigma_stack], axis=0)
    else:
        Sigma = np.asarray(sigma_stack, dtype=np.float64)

    # Build pair indices (upper triangle).
    iu, iv = np.triu_indices(K, k=1)
    n_pairs = iu.size
    # Σ_pool = 0.5 (Σ_u + Σ_v) for each pair: (n_pairs, r, r)
    Sigma_pool = 0.5 * (Sigma[iu] + Sigma[iv])
    # Tikhonov: λ_uv I where λ_uv = lambda_factor * tr / r
    traces = np.einsum("pii->p", Sigma_pool)
    lam = lambda_factor * traces / max(r, 1)                  # (n_pairs,)
    Sigma_reg = Sigma_pool + lam[:, None, None] * np.eye(r)[None, :, :]
    # Differences
    diffs = (mu[iu] - mu[iv]).astype(np.float64)              # (n_pairs, r)
    # Batched Cholesky + solve
    try:
        L = np.linalg.cholesky(Sigma_reg)                     # (n_pairs, r, r)
        # Solve L @ y = diffs for y, then ||y||² = d²
        y = np.linalg.solve(L, diffs[..., None]).squeeze(-1)  # (n_pairs, r)
        d2 = np.einsum("pi,pi->p", y, y)
    except np.linalg.LinAlgError:
        # Fallback per-pair pinv (rare)
        d2 = np.zeros(n_pairs, dtype=np.float64)
        for p in range(n_pairs):
            d2[p] = float(diffs[p] @ np.linalg.pinv(Sigma_reg[p]) @ diffs[p])
    d_vals = np.sqrt(np.maximum(d2, 0.0))

    D_SW = np.zeros((K, K), dtype=np.float64)
    D_SW[iu, iv] = d_vals
    D_SW[iv, iu] = d_vals
    pair_mode = np.full((K, K), SHRINK_ORDER[cell_mode], dtype=np.int8)
    np.fill_diagonal(pair_mode, 0)
    return D_SW, pair_mode


def build_euclidean_matrix(mu: np.ndarray) -> np.ndarray:
    K = mu.shape[0]
    D = np.zeros((K, K), dtype=np.float64)
    for u in range(K):
        for v in range(u + 1, K):
            d = float(np.linalg.norm(mu[u] - mu[v]))
            D[u, v] = d
            D[v, u] = d
    return D


def offdiag_vec(D: np.ndarray) -> np.ndarray:
    """Upper-triangle off-diagonal entries (K*(K-1)/2,) — symmetric matrix assumed."""
    iu = np.triu_indices(D.shape[0], k=1)
    return D[iu]


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation; tie-aware via rankdata. Returns 0 for degenerate."""
    if x.size < 2 or y.size < 2 or x.size != y.size:
        return float("nan")
    rx = rankdata(x).astype(np.float64)
    ry = rankdata(y).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall's tau-b via scipy (small K, fine on CPU)."""
    from scipy.stats import kendalltau
    if x.size < 2 or y.size < 2 or x.size != y.size:
        return float("nan")
    t, _ = kendalltau(x, y)
    return float(t) if t is not None else float("nan")


def compute_observed_rho(mu: np.ndarray, sigma_stack: list[np.ndarray] | np.ndarray,
                          cell_mode: str,
                          lambda_factor: float
                          ) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute D_E, D_SW, ρ_centroid (Spearman), ρ_pearson, τ_kendall, mean_log_ratio.
    Returns (rho_spearman, rho_pearson, tau_kendall, D_E, D_SW, pair_mode, mean_log_ratio)."""
    D_E = build_euclidean_matrix(mu)
    D_SW, pair_mode = build_dsw_matrix(mu, sigma_stack, cell_mode, lambda_factor)
    v_E = offdiag_vec(D_E)
    v_SW = offdiag_vec(D_SW)
    rho_s = spearman_rho(v_E, v_SW)
    # Pearson on the raw values (linear sanity)
    if v_E.size > 1 and v_E.std() > 0 and v_SW.std() > 0:
        rho_p = float(np.corrcoef(v_E, v_SW)[0, 1])
    else:
        rho_p = float("nan")
    tau = kendall_tau(v_E, v_SW)
    pos = (v_E > 1e-12) & (v_SW > 1e-12)
    if pos.any():
        mean_log_ratio = float(np.mean(np.log(v_SW[pos] / v_E[pos])))
    else:
        mean_log_ratio = float("nan")
    return rho_s, rho_p, tau, D_E, D_SW, pair_mode, mean_log_ratio


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap (with replacement at size N) and Whittle-style null
# ──────────────────────────────────────────────────────────────────────────────

def _fit_sigma_batched_gpu(Sigma_sample, counts, r, mode: str,
                             Z_gpu=None, mu=None, codes_gpu=None, mask=None):
    """Batched shrinkage on GPU. `Sigma_sample` is biased sample cov (with /n).
    Returns Σ_v (K, r, r) cupy under `mode` ∈ {sample, lw, oas}.

    For LW we need per-point centred residual statistics — pass Z_gpu (N, r),
    mu (K, r), codes_gpu (N,), mask (K, N) so per-value ||r_i||⁴ sums can be
    formed without materialising every R_v.
    """
    if mode == SHRINK_SAMPLE:
        # Unbiased sample cov: multiply by n/(n-1)
        n = counts.astype(cp.float64)
        return Sigma_sample * (n / cp.maximum(n - 1.0, 1.0))[:, None, None]

    n = counts.astype(cp.float64)
    trace_S = cp.einsum("kii->k", Sigma_sample)
    trace_S2 = cp.einsum("kij,kij->k", Sigma_sample, Sigma_sample)
    mu_val = trace_S / max(r, 1)
    d2 = cp.maximum(trace_S2 - (trace_S ** 2) / max(r, 1), 1e-30)

    if mode == SHRINK_OAS:
        num = (1.0 - 2.0 / max(r, 1)) * trace_S2 + trace_S * trace_S
        den = (n + 1.0 - 2.0 / max(r, 1)) * d2
        shrinkage = cp.clip(num / cp.maximum(den, 1e-30), 0.0, 1.0)
    else:  # LW — full per-point formula matching the CPU fit_sigma_lw
        # bar_b² = (1/n²) sum_i ||r_i r_iᵀ − S||²_F
        # which simplifies to: bar_b² = (A_v − n_v * trace_S2[v]) / n_v²
        # where A_v = sum_{i in v} ||r_i||⁴
        r_per_point = Z_gpu - mu[codes_gpu]                          # (N, r)
        xxT = cp.einsum("nr,nr->n", r_per_point, r_per_point)         # (N,) = ||r_i||²
        xx4 = xxT * xxT
        A = mask @ xx4                                                # (K,) per-value sum
        bar_b2 = cp.maximum((A - n * trace_S2) / (n * n), 0.0)
        shrinkage = cp.clip(bar_b2 / d2, 0.0, 1.0)

    target = mu_val[:, None, None] * cp.eye(r, dtype=cp.float64)[None, :, :]
    return (1.0 - shrinkage)[:, None, None] * Sigma_sample + shrinkage[:, None, None] * target


def _rho_for_grouping_gpu(Z_gpu, codes_cpu: np.ndarray, K: int,
                            lw_threshold: float, oas_threshold: float,
                            lambda_factor: float, min_n_v_floor: int) -> float | None:
    """GPU fast path: cell-mode + per-iter Σ_v batched + batched Cholesky on all pairs.
    Returns ρ_centroid (CPU float) or None if any n_v < min_n_v_floor."""
    counts_cpu = np.bincount(codes_cpu, minlength=K)
    if (counts_cpu < min_n_v_floor).any():
        return None
    r = int(Z_gpu.shape[1])
    chosen_modes = [select_shrink_mode(int(counts_cpu[v]), r, lw_threshold, oas_threshold) for v in range(K)]
    cell_mode = chosen_modes[0]
    for m in chosen_modes[1:]:
        cell_mode = stricter_mode(cell_mode, m)

    codes_gpu = cp.asarray(codes_cpu, dtype=cp.int64)
    counts_gpu = cp.asarray(counts_cpu, dtype=cp.float64)
    # mask: (K, N) — one-hot per value
    mask = (codes_gpu[None, :] == cp.arange(K)[:, None]).astype(cp.float64)
    # Per-value mean: (K, r)
    mu = (mask @ Z_gpu) / counts_gpu[:, None]
    # Per-value sample covariance (biased, /n): Σ_v_biased = (1/n_v) Σ z_i z_iᵀ − μ_v μ_vᵀ
    S_unc = cp.einsum("kn,nr,ns->krs", mask, Z_gpu, Z_gpu)
    Sigma_sample_biased = S_unc / counts_gpu[:, None, None] - cp.einsum("kr,ks->krs", mu, mu)
    Sigma = _fit_sigma_batched_gpu(Sigma_sample_biased, counts_gpu, r, cell_mode,
                                     Z_gpu=Z_gpu, mu=mu, codes_gpu=codes_gpu, mask=mask)

    # Build batched Σ_pool over all upper-triangle pairs
    iu_cpu, iv_cpu = np.triu_indices(K, k=1)
    iu = cp.asarray(iu_cpu); iv = cp.asarray(iv_cpu)
    Sigma_pool = 0.5 * (Sigma[iu] + Sigma[iv])                         # (n_pairs, r, r)
    trace = cp.einsum("pii->p", Sigma_pool)
    lam = lambda_factor * trace / max(r, 1)
    eye_r = cp.eye(r, dtype=cp.float64)
    Sigma_reg = Sigma_pool + lam[:, None, None] * eye_r[None, :, :]
    diffs = (mu[iu] - mu[iv])                                          # (n_pairs, r)
    try:
        # cp.linalg.solve natively handles batched (n, r, r) @ (n, r, 1) → (n, r, 1)
        x = cp.linalg.solve(Sigma_reg, diffs[..., None]).squeeze(-1)   # (n_pairs, r)
    except Exception:
        return None
    # d² = diff.T Σ⁻¹ diff = diff · x
    d2 = cp.einsum("pi,pi->p", diffs, x)
    D_SW = cp.sqrt(cp.maximum(d2, 0.0)).get()
    D_E = cp.linalg.norm(diffs, axis=1).get()
    return spearman_rho(D_E, D_SW)


def _rho_for_grouping(Z: np.ndarray, codes: np.ndarray, K: int,
                       lw_threshold: float, oas_threshold: float,
                       lambda_factor: float, min_n_v_floor: int) -> float | None:
    """One ρ_centroid computation under a given (Z, codes) grouping.

    Returns None if any value has n_v < min_n_v_floor (caller decides redraw).
    Uses cell-wide stricter mode (single Σ_v computation per value, not all 3).
    """
    counts = np.bincount(codes, minlength=K)
    if (counts < min_n_v_floor).any():
        return None
    r = Z.shape[1]
    # Determine cell-wide stricter shrinkage mode for THIS grouping.
    chosen_modes = [select_shrink_mode(int(counts[v]), r, lw_threshold, oas_threshold) for v in range(K)]
    cell_mode = chosen_modes[0]
    for m in chosen_modes[1:]:
        cell_mode = stricter_mode(cell_mode, m)

    mu = np.zeros((K, r), dtype=np.float64)
    sigma_stack: list[np.ndarray] = []
    for v in range(K):
        idx = np.where(codes == v)[0]
        Z_v = Z[idx].astype(np.float64)
        mu[v] = Z_v.mean(axis=0)
        R_v = Z_v - mu[v]
        Sigma, _ = fit_sigma(R_v, cell_mode)
        sigma_stack.append(Sigma)

    D_E = build_euclidean_matrix(mu)
    D_SW, _ = build_dsw_matrix(mu, sigma_stack, cell_mode, lambda_factor)
    v_E = offdiag_vec(D_E)
    v_SW = offdiag_vec(D_SW)
    return spearman_rho(v_E, v_SW)


def bootstrap_rho(Z: np.ndarray, codes: np.ndarray, K: int, n_bootstrap: int,
                   lw_threshold: float, oas_threshold: float, lambda_factor: float,
                   min_n_v_floor: int, max_redraws: int, rng: np.random.Generator,
                   use_gpu: bool = True
                   ) -> tuple[np.ndarray, int]:
    """Bootstrap WITH replacement at size N. Per-draw shrinkage re-eval (option b
    in plan §B.3). Uses GPU-batched path when cupy is available."""
    N = Z.shape[0]
    out = np.zeros(n_bootstrap, dtype=np.float64)
    total_redraws = 0
    Z_gpu = cp.asarray(Z.astype(np.float64)) if (use_gpu and _HAS_CUPY) else None
    for i in range(n_bootstrap):
        attempts = 0
        rho = None
        while attempts < max_redraws and rho is None:
            idx = rng.integers(0, N, size=N)
            c_b = codes[idx]
            if Z_gpu is not None:
                Z_b_gpu = Z_gpu[cp.asarray(idx)]
                rho = _rho_for_grouping_gpu(Z_b_gpu, c_b, K, lw_threshold, oas_threshold,
                                              lambda_factor, min_n_v_floor)
            else:
                rho = _rho_for_grouping(Z[idx], c_b, K, lw_threshold, oas_threshold,
                                         lambda_factor, min_n_v_floor)
            attempts += 1
            if rho is None:
                total_redraws += 1
        out[i] = rho if rho is not None else float("nan")
    return out, total_redraws


def whittle_null_rho(Z: np.ndarray, codes: np.ndarray, K: int, n_perms: int,
                      lw_threshold: float, oas_threshold: float, lambda_factor: float,
                      rng: np.random.Generator, use_gpu: bool = True) -> np.ndarray:
    """Label-permutation null preserving per-value counts. GPU-batched when cupy is available."""
    out = np.zeros(n_perms, dtype=np.float64)
    N = codes.shape[0]
    Z_gpu = cp.asarray(Z.astype(np.float64)) if (use_gpu and _HAS_CUPY) else None
    for i in range(n_perms):
        perm = rng.permutation(N)
        c_perm = codes[perm]
        if Z_gpu is not None:
            rho = _rho_for_grouping_gpu(Z_gpu, c_perm, K, lw_threshold, oas_threshold,
                                          lambda_factor, min_n_v_floor=1)
        else:
            rho = _rho_for_grouping(Z, c_perm, K, lw_threshold, oas_threshold,
                                     lambda_factor, min_n_v_floor=1)
        out[i] = rho if rho is not None else float("nan")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Verdict and confidence tier
# ──────────────────────────────────────────────────────────────────────────────

def assign_verdict(rho_centroid: float, rho_low: float, p_value: float, cfg: dict
                   ) -> str:
    """Pre-registered ladder. ρ_low is the dominant gate — the (ρ=0.90, ρ_low=0.30,
    p<0.05) case lands in centroid_only_shape via the ρ_low < 0.50 clause.

    Note: This is per-cell using p_value (raw); the aggregator applies BH-FDR
    and may downgrade spread_confirmed → centroid_only_shape if q ≥ 0.05.
    """
    rho_pass = float(cfg.get("rho_pass_threshold", 0.85))
    rho_low_pass = float(cfg.get("rho_low_ci_threshold", 0.70))
    rho_marg_low = float(cfg.get("rho_marginal_low", 0.50))
    alpha = float(cfg.get("fdr_alpha", 0.05))

    if rho_low < rho_marg_low or rho_centroid < 0.70 or p_value >= alpha:
        return VERDICT_CENTROID_ONLY
    if rho_centroid >= rho_pass and rho_low >= rho_low_pass:
        return VERDICT_SPREAD_CONFIRMED
    # Marginal band
    if (rho_marg_low <= rho_low < rho_low_pass) or (0.70 <= rho_centroid < rho_pass and rho_low >= rho_marg_low):
        return VERDICT_SPREAD_MARGINAL
    return VERDICT_CENTROID_ONLY


def assign_confidence_tier(K_present: int, min_n_v: int, min_ratio: float,
                            q_value: float, rho_low: float, cfg: dict) -> str:
    """K=4 hard-capped to LOW. DISCOVERY_ONLY if any value has ratio < threshold."""
    if K_present <= 0 or min_n_v <= 0:
        return TIER_DISCOVERY_ONLY
    disc_max = float(cfg.get("tier_discovery_only_max_ratio", 2.0))
    if min_ratio < disc_max:
        return TIER_DISCOVERY_ONLY
    if K_present == 4:
        return TIER_LOW  # hard cap
    if (K_present >= int(cfg.get("tier_high_min_K", 6))
            and min_ratio >= float(cfg.get("tier_high_min_ratio", 10))
            and min_n_v >= int(cfg.get("tier_high_min_n_v", 100))
            and q_value < float(cfg.get("tier_high_q_threshold", 0.01))
            and rho_low >= float(cfg.get("tier_high_rho_low", 0.80))):
        return TIER_HIGH
    if (K_present >= int(cfg.get("tier_medium_min_K", 5))
            and min_ratio >= float(cfg.get("tier_medium_min_ratio", 5))
            and min_n_v >= int(cfg.get("tier_medium_min_n_v", 50))
            and q_value < float(cfg.get("tier_medium_q_threshold", 0.05))
            and rho_low >= float(cfg.get("tier_medium_rho_low", 0.70))):
        return TIER_MEDIUM
    return TIER_LOW


# ──────────────────────────────────────────────────────────────────────────────
# Top-level analysis (importable by toys)
# ──────────────────────────────────────────────────────────────────────────────

def analyze_cell_dsw(Z: np.ndarray, label_codes: np.ndarray, K_natural: int,
                     cfg_stage2b: dict, seed: int = 0,
                     logger: logging.Logger | None = None,
                     skip_bootstrap: bool = False, skip_null: bool = False) -> dict:
    """Full Stage 2b analysis on (Z, label_codes). Mirrors stage2a.analyze_cell shape.

    Returns a dict with the row + arrays. p_dsw is per-cell (raw); BH-FDR is applied
    in the aggregator. Verdict is assigned here using raw p as the q-equivalent gate
    so per-cell reads are self-consistent; the aggregator overwrites once q is known.
    """
    log = logger or logging.getLogger("stage2b_dsw")

    n_perms = int(cfg_stage2b.get("n_permutations", 1000))
    n_bootstrap = int(cfg_stage2b.get("n_bootstrap", 1000))
    lam = float(cfg_stage2b.get("lambda_factor", 1e-6))
    lw_thr = float(cfg_stage2b.get("shrinkage_lw_threshold", 10))
    oas_thr = float(cfg_stage2b.get("shrinkage_oas_threshold", 5))
    min_group = int(cfg_stage2b.get("min_group_size", 30))
    min_K = int(cfg_stage2b.get("min_K_for_dsw", 4))
    n_v_floor_boot = int(cfg_stage2b.get("bootstrap_min_n_v_floor", 5))
    max_redraws_boot = int(cfg_stage2b.get("bootstrap_max_redraws", 100))
    ci_unstable = float(cfg_stage2b.get("ci_halfwidth_unstable", 0.30))

    label_codes = np.asarray(label_codes, dtype=np.int64)
    # ────── Step 1: filter by per-value count
    new_codes, kept_values, mask = filter_values_by_count(label_codes, K_natural, min_group)
    Z_keep = np.ascontiguousarray(Z[mask].astype(np.float32))
    K_present = len(kept_values)
    dropped_values = [v for v in range(K_natural) if v not in set(kept_values)]
    N_used = int(Z_keep.shape[0])
    r = int(Z_keep.shape[1])

    if K_present < min_K:
        # Cannot compute D_SW meaningfully.
        return {
            "K_natural": int(K_natural), "K_present": int(K_present),
            "r": int(r), "n_samples_used": int(N_used),
            "kept_values": kept_values, "dropped_values": dropped_values,
            "rho_centroid": float("nan"), "rho_pearson": float("nan"),
            "tau_kendall": float("nan"), "rho_low": float("nan"), "rho_high": float("nan"),
            "bootstrap_se": float("nan"), "mean_log_ratio": float("nan"),
            "p_dsw": float("nan"), "redraw_rate_bootstrap": float("nan"),
            "min_n_v": 0, "min_ratio_v": 0.0, "gamma": float("inf"),
            "shrinkage_mode_per_v": [], "shrinkage_alpha_per_v": [],
            "shrinkage_pair_mode_matrix": None,
            "D_E": None, "D_SW": None, "rho_bootstrap": None, "rho_null": None,
            "spread_verdict": (VERDICT_INSUFFICIENT if K_present > 0 else VERDICT_LOW_K_AFTER_FILTER),
            "confidence_tier": TIER_DISCOVERY_ONLY,
            "null_unstable_dsw": False,
        }

    # ────── Step 2: per-value Σ̂_v at cell-wide stricter shrinkage mode
    mu, sigma_stack, shrink_mode, cell_mode, shrink_alpha, counts = fit_per_value_sigmas(
        Z_keep, new_codes, K_present, lw_thr, oas_thr,
    )
    min_n_v = int(counts.min())
    min_ratio = float(min_n_v / max(r, 1))
    gamma = float(r / max(min_n_v, 1))

    # ────── Step 3: observed ρ_centroid
    rho_s, rho_p, tau, D_E, D_SW, pair_mode, mlr = compute_observed_rho(
        mu, sigma_stack, cell_mode, lam,
    )

    # ────── Step 4: bootstrap CI (with replacement at size N)
    rng = np.random.default_rng(seed)
    if skip_bootstrap:
        rho_boot = np.array([], dtype=np.float64)
        rho_low = rho_high = boot_se = float("nan")
        redraw_total = 0
    else:
        rho_boot, redraw_total = bootstrap_rho(
            Z_keep, new_codes, K_present, n_bootstrap,
            lw_thr, oas_thr, lam, n_v_floor_boot, max_redraws_boot, rng,
        )
        valid = rho_boot[~np.isnan(rho_boot)]
        if valid.size >= 20:
            rho_low = float(np.quantile(valid, 0.025))
            rho_high = float(np.quantile(valid, 0.975))
            boot_se = float(valid.std(ddof=1))
        else:
            rho_low = rho_high = boot_se = float("nan")

    # ────── Step 5: Whittle-style label-permutation null
    if skip_null:
        rho_null = np.array([], dtype=np.float64)
        p_value = float("nan")
    else:
        rng2 = np.random.default_rng(seed + 1)
        rho_null = whittle_null_rho(
            Z_keep, new_codes, K_present, n_perms,
            lw_thr, oas_thr, lam, rng2,
        )
        valid_null = rho_null[~np.isnan(rho_null)]
        if valid_null.size > 0 and not np.isnan(rho_s):
            n_ge = int(np.sum(valid_null >= rho_s))
            p_value = float((1 + n_ge) / (1 + valid_null.size))
        else:
            p_value = float("nan")

    # ────── Step 6: null stability (CI half-width gate)
    ci_halfwidth = 0.5 * (rho_high - rho_low) if (np.isfinite(rho_high) and np.isfinite(rho_low)) else float("nan")
    null_unstable_dsw = bool(np.isfinite(ci_halfwidth) and ci_halfwidth > ci_unstable)

    # ────── Step 7: per-cell verdict (raw p — aggregator may downgrade via q)
    if null_unstable_dsw:
        verdict = VERDICT_NULL_UNSTABLE_DSW
    else:
        verdict = assign_verdict(rho_s, rho_low, p_value, cfg_stage2b)

    # ────── Step 8: confidence tier (uses p as q-equivalent until aggregator updates)
    tier = assign_confidence_tier(K_present, min_n_v, min_ratio, p_value, rho_low, cfg_stage2b)

    return {
        "K_natural": int(K_natural), "K_present": int(K_present),
        "r": int(r), "n_samples_used": int(N_used),
        "kept_values": kept_values, "dropped_values": dropped_values,
        "rho_centroid": float(rho_s),
        "rho_pearson": float(rho_p),
        "tau_kendall": float(tau),
        "rho_low": float(rho_low),
        "rho_high": float(rho_high),
        "bootstrap_se": float(boot_se),
        "mean_log_ratio": float(mlr),
        "p_dsw": float(p_value),
        "redraw_rate_bootstrap": float(redraw_total) / max(n_bootstrap, 1) if not skip_bootstrap else float("nan"),
        "min_n_v": int(min_n_v),
        "min_ratio_v": float(min_ratio),
        "gamma": float(gamma),
        "ci_halfwidth": float(ci_halfwidth),
        "shrinkage_mode_per_v": list(shrink_mode),
        "shrinkage_alpha_per_v": [float(a) for a in shrink_alpha],
        "shrinkage_pair_mode_matrix": pair_mode,
        "D_E": D_E, "D_SW": D_SW,
        "mu_stack": mu,
        "rho_bootstrap": rho_boot, "rho_null": rho_null,
        "spread_verdict": verdict,
        "confidence_tier": tier,
        "null_unstable_dsw": null_unstable_dsw,
    }


# ══════════════════════════════════════════════════════════════════════════════
# I/O — paths, loaders, runners, CLI
# ══════════════════════════════════════════════════════════════════════════════

def stage2b_cell_dir(results_root: Path, model: str, task: str, mode: str,
                     layer: int, variant: str, concept: str) -> Path:
    return (results_root / "stage2b_dsw" / model / task / f"mode_{mode}"
            / f"layer_{layer:02d}" / f"variant_{variant}" / concept)


def setup_logging(logs_root: Path, model_key: str, task: str, mode: str, variant: str) -> logging.Logger:
    logs_root.mkdir(parents=True, exist_ok=True)
    name = f"stage2b.{model_key}.{task}.{mode}.{variant}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = WatchedFileHandler(logs_root / f"stage2b_{model_key}_{task}_{mode}_{variant}.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


def load_stage2a_summary_row(results_root: Path, model: str, task: str, mode: str,
                              variant: str, layer: int, concept: str) -> dict | None:
    """Read one row from Stage 2a's summary CSV for matching (model, task, mode, variant, layer, concept)."""
    csv_path = (results_root / "stage2a_fourier_helix" / model
                / f"summary_{model}_{task}_mode_{mode}_variant_{variant}.csv")
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    sel = df[(df["task"] == task) & (df["mode"] == mode) & (df["variant"] == variant)
              & (df["layer"] == layer) & (df["concept"] == concept)]
    if sel.empty:
        return None
    return sel.iloc[0].to_dict()


def write_cell_artifacts(out_dir: Path, result: dict, meta: dict,
                          store_full_null: bool, logger: logging.Logger) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-cell row CSV (single row; aggregator concats).
    row_keys = [
        "K_natural", "K_present", "r", "n_samples_used",
        "rho_centroid", "rho_pearson", "tau_kendall",
        "rho_low", "rho_high", "bootstrap_se", "mean_log_ratio",
        "p_dsw", "redraw_rate_bootstrap",
        "min_n_v", "min_ratio_v", "gamma", "ci_halfwidth",
        "spread_verdict", "confidence_tier", "null_unstable_dsw",
    ]
    row = {k: result.get(k) for k in row_keys}
    pd.DataFrame([row]).to_csv(out_dir / "dsw_results.csv", index=False)

    if result.get("D_E") is not None:
        atomic_save(np.asarray(result["D_E"], dtype=np.float64), out_dir / "D_E.npy")
    if result.get("D_SW") is not None:
        atomic_save(np.asarray(result["D_SW"], dtype=np.float64), out_dir / "D_SW.npy")
    if result.get("mu_stack") is not None:
        atomic_save(np.asarray(result["mu_stack"], dtype=np.float64), out_dir / "mu_stack.npy")
    if result.get("shrinkage_pair_mode_matrix") is not None:
        atomic_save(np.asarray(result["shrinkage_pair_mode_matrix"], dtype=np.int8),
                    out_dir / "shrinkage_pair_mode_matrix.npy")

    # Big null/bootstrap arrays — keep on detection or always (small, K(K-1)/2 floats each)
    if result.get("rho_null") is not None and result["rho_null"].size > 0:
        atomic_save(np.asarray(result["rho_null"], dtype=np.float32), out_dir / "rho_null.npy")
    if result.get("rho_bootstrap") is not None and result["rho_bootstrap"].size > 0:
        atomic_save(np.asarray(result["rho_bootstrap"], dtype=np.float32), out_dir / "rho_bootstrap.npy")

    atomic_json({**meta, "computation_status": "complete"}, out_dir / "metadata.json")


def run_one_cell(cfg: dict, results_root: Path,
                  model: str, task: str, mode: str, layer: int, variant: str, concept: str,
                  problems_df: pd.DataFrame, correct_mask: np.ndarray,
                  X_correct: np.ndarray, mu_layer_fallback: np.ndarray | None,
                  stage2a_row: dict | None, logger: logging.Logger,
                  ) -> dict | None:
    out_dir = stage2b_cell_dir(results_root, model, task, mode, layer, variant, concept)
    meta_path = out_dir / "metadata.json"
    if meta_path.exists():
        try:
            cached = json.loads(meta_path.read_text())
            if cached.get("computation_status") == "complete":
                logger.info(f"[skip cached] {variant}/{concept}")
                return cached.get("summary_row")
        except Exception:
            pass

    # Load basis matrix per variant — mirror Stage 2a's resolution.
    if variant == "ccsvd":
        B_path = ccsvd_basis_path(results_root, model, task, layer, concept, mode)
        meta_p = ccsvd_meta_path(results_root, model, task, layer, concept, mode)
    elif variant == "lda_a":
        B_path = lda_a_basis_path(results_root, model, task, layer, concept, mode)
        meta_p = lda_a_meta_path(results_root, model, task, layer, concept, mode)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    B = load_basis_matrix(B_path)
    if B is None or B.shape[1] == 0:
        logger.info(f"[skip no-basis] {variant}/{concept}: {B_path}")
        return None
    mu_layer = load_mu_layer(meta_p)
    if mu_layer is None:
        mu_layer = mu_layer_fallback
    if mu_layer is None:
        logger.warning(f"[skip no-mu] {variant}/{concept}: no mu_layer for mode {mode}")
        return None

    Z = project_to_subspace(X_correct, B, mu_layer)
    r = Z.shape[1]
    K_natural = K_natural_for_concept(problems_df, concept)
    if K_natural < 2:
        logger.info(f"[skip constant] {variant}/{concept}: K_natural={K_natural}")
        return None
    labels_full = problems_df[concept].to_numpy()
    labels_correct_raw = labels_full[correct_mask]
    # Map raw values to 0..K_natural-1 dense codes.
    unique_vals, label_codes = np.unique(labels_correct_raw, return_inverse=True)
    K_natural_dense = int(unique_vals.size)

    seed_int, seed_str = stage2b_seed(model, task, mode, layer, variant, concept)
    t0 = time.time()
    result = analyze_cell_dsw(
        Z, label_codes, K_natural_dense,
        cfg_stage2b=cfg.get("stage2b", {}),
        seed=seed_int, logger=logger,
    )
    runtime = time.time() - t0

    stage2a_verdict = (str(stage2a_row.get("geometry_detected")) if stage2a_row else "unknown")
    stage2a_discovered_period = (float(stage2a_row.get("discovered_period", float("nan")))
                                  if stage2a_row else float("nan"))

    summary_row = {
        "model": model, "task": task, "mode": mode, "layer": int(layer),
        "variant": variant, "concept": concept,
        "stage2a_verdict": stage2a_verdict,
        "stage2a_discovered_period": stage2a_discovered_period,
        **{k: result.get(k) for k in [
            "K_natural", "K_present", "r", "n_samples_used",
            "rho_centroid", "rho_pearson", "tau_kendall",
            "rho_low", "rho_high", "bootstrap_se", "mean_log_ratio",
            "p_dsw", "redraw_rate_bootstrap",
            "min_n_v", "min_ratio_v", "gamma", "ci_halfwidth",
            "spread_verdict", "confidence_tier", "null_unstable_dsw",
        ]},
        "runtime_seconds": round(runtime, 2),
    }

    meta = {
        "model": model, "task": task, "mode": mode, "layer": int(layer),
        "variant": variant, "concept": concept,
        "random_seed_input": seed_str, "random_seed": int(seed_int),
        "n_permutations": int(cfg["stage2b"].get("n_permutations", 1000)),
        "n_bootstrap": int(cfg["stage2b"].get("n_bootstrap", 1000)),
        "stage2a_verdict": stage2a_verdict,
        "stage2a_discovered_period": stage2a_discovered_period,
        "shrinkage_mode_per_v": result.get("shrinkage_mode_per_v"),
        "shrinkage_alpha_per_v": result.get("shrinkage_alpha_per_v"),
        "kept_values": result.get("kept_values"),
        "dropped_values": result.get("dropped_values"),
        "gpu_used": _HAS_CUPY,
        "runtime_seconds": round(runtime, 2),
        "lib_versions": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "sklearn": __import__("sklearn").__version__,
            "cupy": cp.__version__ if _HAS_CUPY else None,
        },
        "summary_row": summary_row,
        "K_natural": int(result.get("K_natural", 0)),
        "K_present": int(result.get("K_present", 0)),
        "B_sha256": sha256_of(B_path) if B_path.exists() else None,
        "B_path": str(B_path),
        "mu_layer_source": str(meta_p),
    }
    store_full = bool(cfg["stage2b"].get("store_full_null_only_for_detections", True))
    write_cell_artifacts(out_dir, result, meta, store_full, logger)
    return summary_row


def discover_concepts_for_cell(results_root: Path, model: str, task: str, mode: str,
                                layer: int, variant: str, problems_df: pd.DataFrame) -> list[str]:
    """Eligible concepts: those Stage 2a produced (helix/circle/none/sparse_value_grid)."""
    csv_path = (results_root / "stage2a_fourier_helix" / model
                / f"summary_{model}_{task}_mode_{mode}_variant_{variant}.csv")
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    sel = df[(df["task"] == task) & (df["mode"] == mode) & (df["variant"] == variant)
              & (df["layer"] == layer)]
    if sel.empty:
        return []
    eligible = sel[sel["geometry_detected"].isin(list(STAGE2A_ELIGIBLE))]
    return sorted(eligible["concept"].astype(str).unique().tolist())


def check_toy_calibration(toy_cfg_path: Path) -> None:
    """Abort if Toy 5B scale + Toy 7B FPR haven't been calibrated yet."""
    if not toy_cfg_path.exists():
        raise RuntimeError(
            f"Stage 2b toy calibration file missing: {toy_cfg_path}. "
            "Run check_stage2b_toys.py first.")
    cal = yaml.safe_load(toy_cfg_path.read_text()) or {}
    if cal.get("toy_5b_tangent_scale") is None:
        raise RuntimeError(
            f"Toy 5B scale not yet calibrated in {toy_cfg_path}. "
            "Run `python check_stage2b_toys.py --calibrate-5b` first.")
    fpr = cal.get("toy_7b_fpr") or {}
    if fpr.get("status") != "pass":
        raise RuntimeError(
            f"Toy 7B FPR has not passed in {toy_cfg_path} (status={fpr.get('status')!r}). "
            "Run `python check_stage2b_toys.py --run-7b` first.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=["addition", "multiplication"])
    ap.add_argument("--mode", required=True, choices=["off", "answer", "norm"])
    ap.add_argument("--variant", required=True, choices=["lda_a", "ccsvd"])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--layer", type=int)
    g.add_argument("--all-layers", action="store_true")
    ap.add_argument("--single-concept", type=str, default=None,
                    help="Process only this concept (smoke-test convenience).")
    ap.add_argument("--skip-toy-gate", action="store_true",
                    help="Bypass the toy-calibration check (NOT for headline runs).")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    paths = cfg["paths"]
    logs_root = Path(paths["logs_root"])
    results_root = Path(paths["results_root"])
    data_root = Path(paths["data_root"])
    model_cfg = next(m for m in cfg["models"] if m["key"] == args.model)
    layers = [args.layer] if args.layer is not None else model_cfg["layers"]

    logger = setup_logging(logs_root, args.model, args.task, args.mode, args.variant)
    logger.info(f"=== Stage 2b: model={args.model} task={args.task} mode={args.mode} "
                f"variant={args.variant} layers={layers} ===")
    logger.info(f"cupy: {'AVAILABLE' if _HAS_CUPY else 'NOT available — CPU only'}")

    # Toy gate (must pass before any real-cell run)
    if not args.skip_toy_gate:
        toy_cfg_path = Path(args.config).parent / cfg["stage2b"].get("toy_calibration_path", "configs/stage2b.yaml")
        check_toy_calibration(toy_cfg_path)
        logger.info(f"toy gate: PASS  ({toy_cfg_path})")
    else:
        logger.warning("toy gate: SKIPPED (--skip-toy-gate). Do not use for headline runs.")

    problems_df = pd.read_csv(data_root / "data" / "raw" / f"{args.task}_problems.csv")
    answers_df = pd.read_csv(data_root / "answers" / args.model / f"{args.task}_answers.csv")
    if len(answers_df) != len(problems_df):
        raise RuntimeError("answers/problems row count mismatch")
    correct_mask = answers_df["correct"].to_numpy().astype(bool)
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
        X_correct = np.ascontiguousarray(X_full[correct_mask].astype(np.float32))
        del X_full

        mu_layer_fallback = X_correct.mean(axis=0).astype(np.float32)
        concepts = discover_concepts_for_cell(results_root, args.model, args.task,
                                               args.mode, layer, args.variant, problems_df)
        if args.single_concept:
            concepts = [c for c in concepts if c == args.single_concept]
        logger.info(f"layer={layer:02d} variant={args.variant}: {len(concepts)} concepts to analyze")

        for concept in concepts:
            stage2a_row = load_stage2a_summary_row(results_root, args.model, args.task,
                                                    args.mode, args.variant, layer, concept)
            try:
                row = run_one_cell(
                    cfg, results_root,
                    args.model, args.task, args.mode, layer, args.variant, concept,
                    problems_df, correct_mask, X_correct, mu_layer_fallback,
                    stage2a_row, logger,
                )
                if row is not None:
                    all_rows.append(row)
                    logger.info(f"  {concept:35s}  K={row.get('K_present')}  "
                                f"ρ={row.get('rho_centroid'):.3f}  "
                                f"CI=[{row.get('rho_low'):.2f},{row.get('rho_high'):.2f}]  "
                                f"p={row.get('p_dsw'):.3g}  "
                                f"verdict={row.get('spread_verdict')}  tier={row.get('confidence_tier')}")
            except Exception as exc:
                logger.exception(f"[FAIL] {args.variant}/{concept}: {exc}")

    if all_rows:
        per_model_dir = results_root / "stage2b_dsw" / args.model
        per_model_dir.mkdir(parents=True, exist_ok=True)
        out_csv = per_model_dir / f"summary_{args.model}_{args.task}_mode_{args.mode}_variant_{args.variant}.csv"
        df_new = pd.DataFrame(all_rows)
        if out_csv.exists():
            df_old = pd.read_csv(out_csv)
            key = ["model", "task", "mode", "layer", "variant", "concept"]
            df_old = df_old[~df_old.set_index(key).index.isin(df_new.set_index(key).index)]
            df_new = pd.concat([df_old, df_new], ignore_index=True)
        df_new = df_new.sort_values(["task", "mode", "layer", "variant", "concept"]).reset_index(drop=True)
        df_new.to_csv(out_csv, index=False)
        logger.info(f"wrote {out_csv} ({len(df_new)} rows)")

    logger.info(f"=== Stage 2b DONE: {args.model}/{args.task}/{args.mode}/{args.variant} ===")


if __name__ == "__main__":
    main()
