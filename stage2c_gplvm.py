#!/usr/bin/env python3
"""
Stage 2c — Bayesian manifold characterisation via point-cloud GPLVM.

For each (model, task, mode, layer, concept) cell on the Union(LDA-A, CCSVD)
subspace, fit an exact Bayesian GPLVM (Titsias-Lawrence 2010) on every correct
activation in the cell. Compete six kernel families (RBF, Periodic,
Periodic+Linear, Torus, Concentric, Periodic+RBF) via BIC-adjusted marginal
likelihood with Kass-Raftery 5-nat decisive threshold.

Verdict gate (all three must pass): BF gap ≥ threshold, 5-fold held-out MSE
better than runner-up, 3-seed agreement within 1 nat. Cells passing the gate
get a 1000-permutation column-shuffle null for the statistical significance
test; cells failing the gate are labelled `inconclusive` without consuming
perm-null compute.

FP32 inner Cholesky on TF32-enabled A6000 (~30× faster than FP64 on this GPU)
with FP64 hyperparameters. Mirrors the parent project's `phase_i_gplvm.py`
structure with point-cloud-specific adaptations.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

import gpytorch
from gpytorch.means import ZeroMean

import stage2c_kernels as kerns

# Zero-quality-drop speedups for the A6000:
#   - TF32 for matmul: kernel-matrix builds (matmul-heavy) move from FP32
#     38 TFLOPS to TF32 62 TFLOPS. TF32 has FP32-equivalent dynamic range with
#     10 bits of mantissa — more than enough precision for kernel cross-products.
#   - cudnn.allow_tf32: same for any cudnn-backed ops.
#   - matmul_precision('high'): tells PyTorch we accept TF32 for FP32 matmuls.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


# Filter chatty gpytorch warnings about jitter; we surface jitter in metadata.
warnings.filterwarnings("ignore", category=gpytorch.utils.warnings.NumericalWarning)
warnings.filterwarnings("ignore", category=UserWarning,
                         module="gpytorch")


# ─── Locked configuration ─────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
STAGE2C_CONFIG_DEFAULT = PROJECT_ROOT / "configs" / "stage2c.yaml"

MIN_GROUP_SIZE = 30
MIN_K_FOR_STAGE2C = 4
D_MAX_LATENT = 5
N_LBFGS_ITERS = 100
LBFGS_TOL_GRAD = 1e-5
ARD_PRUNE_MULTIPLIER = 100.0
JITTER_INIT = 1e-4          # FP32 condition-number safety margin
JITTER_MAX = 1e-1           # Bumped from 1e-2 to handle composite-kernel
                            # ill-conditioning at production N≈8K. K3/K4/K5/K6
                            # additive periodic compositions need this headroom
                            # for FP32 Cholesky to factorise.
SEED_AGREEMENT_NATS = 1.0
N_SEEDS = 3
SEEDS = (42, 43, 44)
HOLDOUT_FRACTION = 0.2
N_HOLDOUT_OPT_ITERS = 100
BOOTSTRAP_D_HAT_DRAWS = 200
N_PERMUTATIONS = 1000       # Project standard. Run only on cells passing the
                            # BF + holdout + seed gate (selective significance).
PERM_LBFGS_ITERS = 20
FDR_ALPHA = 0.05
LATENT_VARIANCE_FLOOR = 1e-2

# Period regime constants
REGIME_NARROW = "narrow"
REGIME_WIDE = "wide"
REGIME_DISCOVER = "discover"

# Verdict labels
V_HELIX = "helix"
V_CIRCLE = "circle"
V_TORUS = "torus"
V_CONCENTRIC = "concentric"
V_PERIODIC_SMOOTH = "periodic_smooth"   # K6 PeriodicRBF — periodic+smooth-other-axis
V_SMOOTH_ONLY = "smooth_only"
V_DIM_ONLY = "dim_only"
V_KERNEL_INCONCLUSIVE_SEEDS = "kernel_inconclusive_seeds"
V_INCONCLUSIVE = "inconclusive"
V_LOW_K = "low_K"

KERNEL_TO_VERDICT = {
    "K1_RBF": V_SMOOTH_ONLY,
    "K2_Periodic": V_CIRCLE,
    "K3_PeriodicLinear": V_HELIX,
    "K4_Torus": V_TORUS,
    "K5_Concentric": V_CONCENTRIC,
    "K6_PeriodicRBF": V_PERIODIC_SMOOTH,
}


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ─── Atomic IO helpers (mirror stage2b) ───────────────────────────────────────

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
              kernel: str, seed_idx: int) -> int:
    s = f"stage2c|{model}|{task}|{mode}|{layer:02d}|{concept}|{kernel}|{seed_idx}"
    h = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**63 - 1)


# ─── Config / paths ───────────────────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def derive_paths(cfg: dict) -> dict:
    data_root = Path(cfg["paths"]["data_root"])
    results_root = Path(cfg["paths"]["results_root"])
    activations_root = Path(cfg["paths"].get("activations_root", data_root / "activations"))
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


def ccsvd_meta_path(results_root: Path, model: str, task: str,
                     layer: int, concept: str, mode: str) -> Path:
    if mode == "off":
        return (results_root / "ccsvd_subspaces" / model / task
                / f"layer_{layer:02d}" / concept / "meta.json")
    return (results_root / "ccsvd_subspaces" / f"mode_{mode}" / model / task
            / f"layer_{layer:02d}" / concept / "meta.json")


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


def stage2a_cell_dir(results_root: Path, model: str, task: str, mode: str,
                      layer: int, variant: str, concept: str) -> Path:
    return (results_root / "stage2a_fourier_helix" / model / task
            / f"mode_{mode}" / f"layer_{layer:02d}" / f"variant_{variant}"
            / concept)


# ─── Union basis builder (per-concept, NO β-append) ───────────────────────────

SVD_TOLERANCE_FACTOR = 1e-10


def build_union_basis(results_root: Path, model: str, task: str, mode: str,
                       layer: int, concept: str) -> Tuple[np.ndarray, dict]:
    """Stage 2c per-cell union basis: SVD-orthonormalise [LDA-A | CCSVD] columns.

    Differs from residual_hunting.build_union in two ways:
      - Per-concept, not multi-concept (Stage 2c works on one concept at a time).
      - No β scalar appending (β_answer / β_norm are Step 7 audit artefacts that
        do not belong in a per-concept kernel-comparison subspace).

    Returns:
      B_u: (4096, k_u) float32 union basis (column form)
      meta: dict with contributions, ranks, SHA hashes

    The layer mean `mu_layer` is computed by the caller directly from the
    correct-mask activations (CCSVD meta.json does not store it).
    """
    paths_tried = []
    rows = []
    contributions = []

    # LDA-A
    p_lda = lda_basis_path(results_root, model, task, layer, concept, mode)
    if p_lda.exists():
        try:
            B_lda = np.load(p_lda)
            if B_lda.ndim == 2 and B_lda.shape[0] == 4096 and B_lda.shape[1] > 0:
                rows.append(B_lda.T)                          # (n_sig, 4096)
                contributions.append({"source": "lda_a", "n_dims": int(B_lda.shape[1]),
                                       "path": str(p_lda),
                                       "sha256": hashlib.sha256(B_lda.tobytes()).hexdigest()[:16]})
        except Exception as e:
            paths_tried.append((str(p_lda), str(e)))

    # CCSVD
    p_ccsvd = ccsvd_basis_path(results_root, model, task, layer, concept, mode)
    if p_ccsvd.exists():
        try:
            B_ccsvd = np.load(p_ccsvd)
            if B_ccsvd.ndim == 2 and B_ccsvd.shape[0] == 4096 and B_ccsvd.shape[1] > 0:
                rows.append(B_ccsvd.T)                        # (r_c, 4096)
                contributions.append({"source": "ccsvd", "n_dims": int(B_ccsvd.shape[1]),
                                       "path": str(p_ccsvd),
                                       "sha256": hashlib.sha256(B_ccsvd.tobytes()).hexdigest()[:16]})
        except Exception as e:
            paths_tried.append((str(p_ccsvd), str(e)))

    if not rows:
        return (np.zeros((4096, 0), dtype=np.float32),
                {"k_u": 0, "contributions": contributions, "paths_tried": paths_tried,
                 "stacked_dim": 0, "redundancy_removed": 0,
                 "lda_a_present": p_lda.exists(), "ccsvd_present": p_ccsvd.exists()})

    stacked = np.vstack(rows).astype(np.float64)              # (n_lda + r_c, 4096)
    stacked_dim = stacked.shape[0]
    U, S, Vt = np.linalg.svd(stacked, full_matrices=False)
    keep = S > SVD_TOLERANCE_FACTOR * S[0]
    Vt_keep = Vt[keep].astype(np.float32)                     # (k_u, 4096)
    B_u = Vt_keep.T.astype(np.float32)                        # (4096, k_u)

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


# ─── Centroid + per-centroid noise (reuses stage2b's shrinkage) ───────────────

# Import the shrinkage helpers from stage2b — single source of truth.
sys.path.insert(0, str(PROJECT_ROOT))
from stage2b_dsw_spread_aware import (  # noqa: E402
    fit_per_value_sigmas, filter_values_by_count,
)


def compute_centroids_and_noise(Z: np.ndarray, label_codes: np.ndarray, K_present: int,
                                  lw_threshold: float = 10.0,
                                  oas_threshold: float = 5.0
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """For each value v: μ_v ∈ ℝ^r and centroid-sampling variance scalar σ²_v.

    The plan §C.1 specifies Λ_v = Σ_v_pt / n_v as the centroid sampling covariance.
    The GP likelihood is heteroscedastic Gaussian, so we summarise Λ_v to a scalar
    σ²_v = tr(Λ_v) / r (the average sampling variance per output dim). This makes
    the GP isotropic at each centroid while still respecting per-centroid sample
    sizes. The full Λ_v is kept on disk for audit.

    Returns:
      mu_stack:    (K_present, r) float64 centroids
      noise_scalar:(K_present,)   float64 σ²_v per centroid
      Lambda_full: (K_present, r, r) float64 the full per-centroid sampling cov (audit)
      meta:        dict with chosen_modes, cell_mode, n_v counts
    """
    mu, sigma_stack, chosen_modes, cell_mode, shrink_alpha, counts = fit_per_value_sigmas(
        Z, label_codes, K_present, lw_threshold, oas_threshold)
    r = mu.shape[1]
    Lambda_full = np.zeros((K_present, r, r), dtype=np.float64)
    noise_scalar = np.zeros(K_present, dtype=np.float64)
    for v in range(K_present):
        n_v = int(counts[v])
        Lambda_full[v] = np.asarray(sigma_stack[v], dtype=np.float64) / max(n_v, 1)
        noise_scalar[v] = float(np.trace(Lambda_full[v]) / max(r, 1))
    meta = {
        "chosen_modes": list(chosen_modes),
        "cell_mode": cell_mode,
        "n_v_per_value": [int(c) for c in counts],
        "shrink_alpha_per_value": [float(a) for a in shrink_alpha],
    }
    return mu, noise_scalar, Lambda_full, meta


# ─── Period regime from Stage 2a confidence ───────────────────────────────────

@dataclass
class PeriodSeed:
    """Period seeding for one kernel hyperparameter."""
    P_init: float
    regime: str        # "narrow" | "wide" | "discover"
    source: str        # "stage2a_top1" | "stage2a_top2" | "grid_search"
    grid_candidates: List[float] = None
    grid_elbos: List[float] = None
    grid_winner: float = None


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


def read_stage2a_periodogram(results_root: Path, model: str, task: str,
                              mode: str, variant: str, layer: int, concept: str
                              ) -> Optional[np.ndarray]:
    """Load fourier_spectrum_observed.npy shape (K//2 + 1, r) from Stage 2a."""
    d = stage2a_cell_dir(results_root, model, task, mode, layer, variant, concept)
    p = d / "fourier_spectrum_observed.npy"
    if not p.exists():
        return None
    try:
        return np.load(p)
    except Exception:
        return None


def stage2a_confident(row: dict) -> bool:
    """Compute the cell's Stage 2a confidence flag per plan §C.2.5."""
    if row is None:
        return False
    verdict = str(row.get("geometry_detected", row.get("verdict", ""))).strip()
    if verdict not in {"helix", "circle"}:
        return False
    # Whittle p < 0.01 (Stage 2a column is `p_two_axis`; falls back to legacy names)
    p_two = row.get("p_two_axis", row.get("two_axis_p"))
    if p_two is None or (isinstance(p_two, float) and math.isnan(p_two)):
        p_two = row.get("p_coord_a", 1.0)
    try:
        p_two = float(p_two)
    except Exception:
        p_two = 1.0
    if p_two >= 0.01:
        return False
    period_match = row.get("period_match", False)
    return bool(period_match)


def derive_top2_period(P_spec: np.ndarray, K: int, P_top1: float
                       ) -> Tuple[Optional[float], float, float]:
    """Compute the top-2 periodogram peak averaged across the r output dims.

    The top-1 peak is at k = round(K / P_top1). We exclude k=0 (DC) and the
    top-1 frequency and its 1-bin neighbours, then take the argmax over k of
    the across-coord mean power.

    Returns (P_top2 or None, power_top2, power_top1) where powers are the
    across-coord averaged at the chosen k.
    """
    if P_spec is None or P_spec.ndim != 2:
        return None, 0.0, 0.0
    n_freq = P_spec.shape[0]
    # Power averaged across r coords; index 0 is DC.
    power_mean = P_spec.mean(axis=1)               # (K//2 + 1,)
    if n_freq < 3:
        return None, 0.0, 0.0
    k_top1 = int(round(K / max(P_top1, 1e-9))) if P_top1 > 0 else 0
    excluded = set(range(0, 1)) | {k_top1 - 1, k_top1, k_top1 + 1}
    excluded = {k for k in excluded if 0 <= k < n_freq}
    best_k = -1
    best_pw = -1.0
    for k in range(1, n_freq):
        if k in excluded:
            continue
        if power_mean[k] > best_pw:
            best_pw = float(power_mean[k])
            best_k = k
    pw_top1 = float(power_mean[k_top1]) if 0 <= k_top1 < n_freq else 0.0
    if best_k <= 0:
        return None, 0.0, pw_top1
    P_top2 = float(K / max(best_k, 1))
    return P_top2, float(best_pw), pw_top1


def top2_confident(power_top2: float, power_top1: float, median_floor: float) -> bool:
    if power_top1 <= 0 or median_floor <= 0:
        return False
    return (power_top2 >= 0.5 * power_top1) and (power_top2 >= 3.0 * median_floor)


# ─── Bayesian-manifold GP with observed-input latent positions ───────────────
#
# This is the natural Bayesian model for centroid-only data: the value index v
# IS the manifold parameterisation, so the latent input is observed (no
# learning needed). Each kernel hypothesises a different manifold geometry as
# a function of v. The marginal likelihood is the manifold-hypothesis evidence.
#
# Why not the parent's free-latent BayesianGPLVM: the parent fitted per-point
# data where each point's latent was unknown, so learning latents was
# necessary. For per-value centroids the latent IS the value index — learning
# it freely with K=10 data points and 1-2 latent dims would let composite
# kernels (K3 Periodic+Linear, K6 Periodic+RBF) absorb arbitrary structure via
# their non-periodic axis, which is an over-parameterisation artefact, not a
# Bayesian-manifold property.
#
# Latent layout per kernel: all dims = `v / max(K-1, 1)` ∈ [0, 1]. This puts
# each kernel on equal footing; the kernel's STRUCTURE (Periodic, Linear, RBF,
# Torus, etc.) is what determines its marginal likelihood.


class LatentGPLVM(gpytorch.models.ExactGP):
    """Bayesian GPLVM (Titsias-Lawrence 2010): learnable latent positions.

    Each of the N point-cloud observations gets its own latent position X[n, :]
    ∈ ℝ^{d_latent}. L-BFGS optimises latents jointly with kernel hyperparameters
    against the marginal log-likelihood. Multi-output observations are treated
    as independent given the latent (parent's pattern), so the per-output log-
    likelihoods sum.

    Mirrors phase_i_gplvm.LatentGPLVM from the arithmetic-geometry parent.
    """

    def __init__(self, x_init: torch.Tensor, y: torch.Tensor,
                 kernel_module, likelihood):
        super().__init__(x_init, y[:, 0], likelihood)
        self.X = torch.nn.Parameter(x_init.clone().detach())
        self.mean_module = ZeroMean()
        self.covar_module = kernel_module
        self._train_inputs = (self.X,)
        self.train_targets_full = y                              # (N, D)
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def _marginal_log_likelihood_per_output(model, y_full: torch.Tensor,
                                          likelihood, jitter: float
                                          ) -> torch.Tensor:
    """Sum_d log p(y_d | X, θ) under homoscedastic Gaussian noise — Option C with
    FP32 Cholesky to exploit the A6000's 38 TFLOPS FP32 throughput (~30× faster
    than FP64).

    The kernel hyperparameters and latent positions remain in FP64 for stable
    optimisation; only the inner Cholesky / cholesky_solve / logdet compute in
    FP32. The final log-marginal-likelihood scalar is cast back to FP64 so the
    LBFGS optimiser receives a FP64 loss value (gradients flow back through the
    fp32 → fp64 cast correctly via autograd's mixed-precision handling).

    K_full = K(X, X) + (σ² + jitter)·I, summed over D independent output dims.
    Mirrors phase_i_gplvm._marginal_log_likelihood_per_output (Titsias-Lawrence).
    """
    N, D = y_full.shape
    K = model.covar_module(model.X).evaluate()                        # fp64
    sigma2 = likelihood.noise.detach()
    eye64 = torch.eye(N, dtype=K.dtype, device=K.device)
    K_full = K + (sigma2 + jitter) * eye64                            # fp64
    # ── FP32 inner ops (heavy compute) ──────────────────────────────────────
    K_f32 = K_full.to(torch.float32)
    y_f32 = y_full.to(torch.float32)
    L = torch.linalg.cholesky(K_f32)                                  # fp32
    logdet = 2.0 * torch.sum(torch.log(torch.diagonal(L)))
    alpha = torch.cholesky_solve(y_f32, L)                            # fp32
    quad = torch.sum(y_f32 * alpha, dim=0)
    log_lik_per_d_f32 = -0.5 * quad - 0.5 * logdet - 0.5 * N * math.log(2.0 * math.pi)
    # Cast scalar back to fp64 for the optimiser
    return log_lik_per_d_f32.sum().to(torch.float64)


_PERIODIC_KERNELS = {"K2_Periodic", "K3_PeriodicLinear", "K4_Torus",
                      "K5_Concentric", "K6_PeriodicRBF"}


SUBSAMPLE_N_MAX = 10000    # Full per-cell population (largest cell N≈9,963).
                            # Subsampling was tested at 2k/3k/5k and produced
                            # deterministic Cholesky failure even for K1 — the
                            # LBFGS optimiser drives latents to degenerate
                            # configurations when N is too small to constrain
                            # them. Stay at full N; conditioning for composite
                            # kernels handled via the higher JITTER_MAX below.
# CG settings (kept as a safety net for cells N > SUBSAMPLE_N_MAX — never
# triggered with the current cap):
CG_MAX_CHOLESKY_SIZE = 12000
CG_TOLERANCE = 1e-3
CG_MAX_ITERATIONS = 2000


def stratified_subsample(N: int, labels: np.ndarray, n_max: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Return indices into [0, N) of size ≤ n_max, drawn proportional to
    per-label counts so every value retains coverage."""
    if N <= n_max:
        return np.arange(N)
    unique, counts = np.unique(labels, return_counts=True)
    # Proportional allocation, rounded down; fill rest randomly
    quotas = np.maximum(1, (counts.astype(np.float64) * n_max / N).astype(int))
    while quotas.sum() > n_max:
        # Trim from the largest quota
        i = int(np.argmax(quotas))
        quotas[i] -= 1
    picks = []
    for u, q in zip(unique, quotas):
        pool = np.where(labels == u)[0]
        if q >= pool.size:
            picks.append(pool)
        else:
            picks.append(rng.choice(pool, size=int(q), replace=False))
    out = np.concatenate(picks)
    rng.shuffle(out)
    return out[:n_max]


def _latent_init(y: np.ndarray, kernel_name: str, d_latent: int,
                  period_init: Optional[float] = None,
                  seed: int = 42) -> np.ndarray:
    """Initialise latent coordinates for the GPLVM.

    For periodic kernels with d_latent == 1, recover the angular coordinate
    from the top-2 PCs via atan2 and scale to one period — avoids the cluster-
    collapse minimum L-BFGS would fall into from plain PCA.
    """
    yc = y - y.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(yc, full_matrices=False)
    if kernel_name in _PERIODIC_KERNELS and d_latent == 1 and period_init is not None:
        z2 = u[:, :2] * s[:2]
        z2 = z2 / (z2.std(axis=0, keepdims=True) + 1e-9)
        theta = np.arctan2(z2[:, 1], z2[:, 0])
        return (theta * (period_init / (2.0 * math.pi))).reshape(-1, 1)
    z = u[:, :d_latent] * s[:d_latent]
    return z / (np.std(z, axis=0, keepdims=True) + 1e-9)


def _ard_pruning_and_posterior(kernel, latent_z: Optional[torch.Tensor]
                                 ) -> dict:
    """Inspect ARD lengthscales and the latent-position spread to count active dims.

    Returns:
      d_active_ls:     # axes deemed active by the lengthscale rule (ℓ < 100 × prior median)
      d_active_var:    # axes deemed active by the latent-spread rule (std ≥ 1e-2)
      d_active:        intersection (both rules)
      ls_per_dim:      list of softplus-transformed raw_lengthscale values
      raw_ls_mean:     list of raw_lengthscale parameter values (for credibility)
      raw_ls_var:      list of variational variance approximations (Laplace-style;
                         taken as 1/|second derivative of negative log posterior|;
                         here we use 1/H_diag from the kernel's parameter Fisher info
                         if available, else NaN)
    """
    out = {
        "ls_per_dim": [], "d_active_ls": 0, "d_active_var": 0, "d_active": 0,
        "raw_ls_mean": [], "raw_ls_var": [],
    }
    ls_per_dim = None
    raw_ls_param = None
    for name, p in kernel.named_parameters():
        if "raw_lengthscale" in name:
            ls = torch.nn.functional.softplus(p).detach().flatten().cpu().numpy()
            if ls_per_dim is None or ls.size > ls_per_dim.size:
                ls_per_dim = ls
                raw_ls_param = p.detach().flatten().cpu().numpy()
    if ls_per_dim is None or raw_ls_param is None:
        if latent_z is not None:
            stds = latent_z.detach().cpu().numpy().std(axis=0)
            out["d_active_var"] = int((stds >= LATENT_VARIANCE_FLOOR).sum())
            out["d_active"] = out["d_active_var"]
        return out

    out["ls_per_dim"] = [float(x) for x in ls_per_dim]
    out["raw_ls_mean"] = [float(x) for x in raw_ls_param]

    # Lengthscale-based active dims: ℓ ≤ 100 × prior median (prior median for
    # Gamma(1,1) = ln 2 ≈ 0.693, mirroring the parent's rule).
    prior_median = 0.693
    n_total = ls_per_dim.size
    active_ls = ls_per_dim <= ARD_PRUNE_MULTIPLIER * prior_median
    out["d_active_ls"] = int(active_ls.sum())

    if latent_z is not None:
        stds = latent_z.detach().cpu().numpy().std(axis=0)
        if stds.size < n_total:
            stds = np.pad(stds, (0, n_total - stds.size), constant_values=0.0)
        active_var = stds[:n_total] >= LATENT_VARIANCE_FLOOR
        out["d_active_var"] = int(active_var.sum())
        out["d_active"] = int((active_ls & active_var).sum())
    else:
        out["d_active"] = out["d_active_ls"]

    # Laplace variance approximation: we approximate the posterior variance
    # on each raw_lengthscale by the curvature of the prior at the optimum
    # (a conservative lower bound on the true credibility). Gamma(1,1) prior
    # on softplus(raw_ls) has log-prior ≈ -softplus(x), whose second derivative
    # is σ(x)(1-σ(x)) where σ is sigmoid — small near the optimum. Without a
    # full Hessian solve this is the cheap proxy; the bootstrap d̂ (separate
    # path) is the rigorous companion.
    var_proxy = []
    for x in raw_ls_param:
        sigx = 1.0 / (1.0 + math.exp(-x))
        var_proxy.append(1.0 / max(sigx * (1.0 - sigx), 1e-6))
    out["raw_ls_var"] = var_proxy
    return out


# ─── Single-seed kernel fit ───────────────────────────────────────────────────

def fit_kernel_one_seed(y_centroids: np.ndarray, noise_scalar: np.ndarray,
                         kernel_name: str,
                         period_seed: Optional[PeriodSeed],
                         period_seed_secondary: Optional[PeriodSeed],
                         d_max: int = D_MAX_LATENT,
                         n_iters: int = N_LBFGS_ITERS,
                         seed: int = 42,
                         device: Optional[str] = None
                         ) -> Dict:
    """Fit GPLVM on K centroids for one kernel and one seed.

    Returns a result dict including log marginal likelihood, ARD info, and
    latent coordinates. status='ok' on success; otherwise a status string
    explaining the failure.
    """
    if device is None:
        device = default_device()
    torch.manual_seed(int(seed) % (2**31 - 1))
    np.random.seed(int(seed) % (2**31 - 1))

    # Point-cloud framing: y_centroids is actually the (N, k_u) point cloud.
    # Name preserved for API compatibility; rename locally.
    y_points = y_centroids
    N, D_obs = y_points.shape
    if N < MIN_K_FOR_STAGE2C:
        return {"status": "n_too_small", "N": N}

    spec = kerns.KERNEL_REGISTRY[kernel_name]
    # Each kernel encodes a specific manifold-shape hypothesis. The latent
    # dimensionality is the structural minimum the kernel needs:
    #   K1 RBF      d=1  → "smooth 1D curve"           (vs K2 = circle on same dim)
    #   K2 Periodic d=1  → "1D circle"
    #   K3 P+Linear d=2  → "helix" (circle + linear drift)
    #   K4 Torus    d=2  → "torus" (two periods on orthogonal axes)
    #   K5 Concentr d=1  → "two harmonics at same period on same axis"
    #   K6 P+RBF    d=2  → "periodic + smooth axis"
    # The continuous intrinsic-dimension question (PCA participation ratio) is
    # handled separately by bootstrap_d_hat on the point cloud, NOT by ARD on
    # an over-parameterised K1. This keeps the kernel competition fair —
    # otherwise K1 with d=5 ARD becomes a universal approximator that wins
    # everything by exploiting latent freedom rather than kernel structure.
    d_latent = spec["min_latent_dim"]
    # Identifiability cap (rare): never more latents than points minus one
    d_latent = min(d_latent, max(1, N - 1))

    period_val = period_seed.P_init if period_seed is not None else None
    # PCA-based latent init (parent's _latent_init): angular coord via atan2
    # for periodic kernels with d_latent==1, else plain PCA scaled to unit std.
    z_init_np = _latent_init(y_points, kernel_name, d_latent,
                              period_init=period_val, seed=seed)
    z_init = torch.tensor(z_init_np, dtype=torch.float64, device=device)
    y_t = torch.tensor(y_points, dtype=torch.float64, device=device)

    # Build the kernel with the right regime + periods
    if spec["needs_period"]:
        if kernel_name == "K4_Torus":
            kmod = kerns.build_kernel(
                kernel_name,
                period=period_val,
                regime=period_seed.regime,
                period_secondary=(period_seed_secondary.P_init if period_seed_secondary is not None else None),
                regime_secondary=(period_seed_secondary.regime if period_seed_secondary is not None else period_seed.regime),
            )
        else:
            kmod = kerns.build_kernel(kernel_name, period=period_val, regime=period_seed.regime)
    else:
        kmod = kerns.build_kernel(kernel_name, active_dim_count=d_latent)
    kmod = kmod.to(device).double()

    # Homoscedastic GaussianLikelihood with learnable noise variance
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_prior=kerns.noise_var_prior(),
    ).double().to(device)
    likelihood.noise = torch.tensor(0.1, dtype=torch.float64, device=device)

    model = LatentGPLVM(z_init, y_t, kmod, likelihood).double().to(device)
    model.train()
    likelihood.train()

    # Dedupe parameters to silence torch's duplicate-param warning
    seen_ids = set()
    params = []
    for p in list(model.parameters()) + list(likelihood.parameters()):
        if id(p) in seen_ids:
            continue
        seen_ids.add(id(p))
        params.append(p)
    # Strong-Wolfe line-search LBFGS (matches parent project's working setting).
    # Without a line search, LBFGS takes the full Newton step regardless of
    # whether it destabilises the kernel matrix — this caused deterministic
    # Cholesky failures on composite kernels (K3/K4/K5/K6) at production N.
    optimizer = torch.optim.LBFGS(params, lr=0.1, max_iter=n_iters,
                                   tolerance_grad=LBFGS_TOL_GRAD,
                                   history_size=10,
                                   line_search_fn="strong_wolfe")

    jitter = JITTER_INIT
    last_loss = float("inf")
    n_step = [0]

    def closure():
        optimizer.zero_grad()
        nonlocal jitter, last_loss
        for _ in range(5):
            try:
                ll = _marginal_log_likelihood_per_output(
                    model, y_t, likelihood, jitter)
                loss = -ll
                loss.backward()
                last_loss = float(loss.item())
                n_step[0] += 1
                return loss
            except torch._C._LinAlgError:
                jitter = min(jitter * 10.0, JITTER_MAX)
        # Final attempt
        try:
            ll = _marginal_log_likelihood_per_output(
                model, y_t, likelihood, jitter)
            loss = -ll
            loss.backward()
            return loss
        except Exception as e:
            raise RuntimeError(f"Cholesky failed at jitter={jitter}: {e}")

    t0 = time.time()
    # Option C: GPyTorch CG-based exact inference for N > CG_MAX_CHOLESKY_SIZE.
    # This wraps all kernel-matrix linear solves inside the marginal-likelihood
    # evaluation: direct Cholesky for small N, iterative CG for large N. The
    # returned quantity is the exact Titsias-Lawrence log marginal likelihood
    # regardless of which solver path is taken.
    cg_ctx = (gpytorch.settings.max_cholesky_size(CG_MAX_CHOLESKY_SIZE),
               gpytorch.settings.cg_tolerance(CG_TOLERANCE),
               gpytorch.settings.max_cg_iterations(CG_MAX_ITERATIONS))
    try:
        with cg_ctx[0], cg_ctx[1], cg_ctx[2]:
            optimizer.step(closure)
    except RuntimeError as e:
        return {"status": "numerical_failure", "error": str(e),
                "jitter": jitter, "kernel": kernel_name, "seed": seed}
    elapsed = time.time() - t0

    if jitter > JITTER_MAX:
        return {"status": "numerical_failure_jitter", "jitter": jitter,
                "kernel": kernel_name, "seed": seed}

    # Final clean evaluation under the same CG settings
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        with cg_ctx[0], cg_ctx[1], cg_ctx[2]:
            try:
                log_marg_lik = float(_marginal_log_likelihood_per_output(
                    model, y_t, likelihood, jitter).item())
            except Exception as e:
                return {"status": "numerical_failure_final", "error": str(e),
                        "kernel": kernel_name, "seed": seed}

    ard_info = _ard_pruning_and_posterior(kmod, latent_z=model.X)
    final_periods = kerns.extract_periods(kmod)
    final_ls = kerns.extract_lengthscales(kmod)
    z_final = model.X.detach().cpu().numpy()

    return {
        "status": "ok",
        "kernel": kernel_name,
        "seed": int(seed),
        "N": int(N),
        "D_obs": int(D_obs),
        "d_latent_max": int(d_latent),
        "d_latent_active": int(ard_info["d_active"]),
        "log_marginal_likelihood": float(log_marg_lik),
        "log_marginal_likelihood_per_d": float(log_marg_lik / max(D_obs, 1)),
        "noise_variance": float(likelihood.noise.detach().item()),
        "jitter_used": float(jitter),
        "n_steps": int(n_step[0]),
        "elapsed_seconds": float(elapsed),
        "final_periods": final_periods,
        "final_lengthscales": final_ls,
        "ard_info": ard_info,
        "z_final": z_final.tolist(),
    }


def _run_seed_on_stream(idx: int, results_list: list, stream,
                         args_tuple: tuple) -> None:
    """Thread target: run one seed's fit on a dedicated CUDA stream.

    PyTorch releases the GIL during CUDA-bound operations, so threads on
    different streams run their forward/backward passes concurrently on the
    GPU. The Cholesky/cholesky_solve calls are GPU-side; thread-level Python
    overhead is microseconds.
    """
    try:
        with torch.cuda.stream(stream):
            results_list[idx] = fit_kernel_one_seed(*args_tuple)
    except Exception as e:
        results_list[idx] = {"status": "thread_exception", "error": str(e)}


def fit_kernel_three_seeds(y_centroids: np.ndarray, noise_scalar: np.ndarray,
                            kernel_name: str,
                            period_seed: Optional[PeriodSeed],
                            period_seed_secondary: Optional[PeriodSeed],
                            seeds: Tuple[int, int, int] = SEEDS,
                            d_max: int = D_MAX_LATENT,
                            n_iters: int = N_LBFGS_ITERS,
                            device: Optional[str] = None,
                            logger: Optional[logging.Logger] = None,
                            parallel: bool = False) -> Dict:
    """Run 3 seeds for one kernel; report median LL + seed agreement."""
    results: list = [None] * len(seeds)
    use_parallel = (parallel and device != "cpu"
                    and torch.cuda.is_available()
                    and len(seeds) > 1)
    if use_parallel:
        import threading
        streams = [torch.cuda.Stream() for _ in seeds]
        threads = []
        for i, s in enumerate(seeds):
            args_tuple = (y_centroids, noise_scalar, kernel_name,
                           period_seed, period_seed_secondary,
                           d_max, n_iters, s, device)
            t = threading.Thread(target=_run_seed_on_stream,
                                 args=(i, results, streams[i], args_tuple))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        torch.cuda.synchronize()
    else:
        for i, s in enumerate(seeds):
            results[i] = fit_kernel_one_seed(
                y_centroids, noise_scalar, kernel_name,
                period_seed, period_seed_secondary,
                d_max=d_max, n_iters=n_iters, seed=s, device=device)
    if logger is not None:
        for r, s in zip(results, seeds):
            if r is None:
                logger.info("    %s seed=%d FAILED: no_result", kernel_name, s)
            elif r.get("status") == "ok":
                logger.info("    %s seed=%d log_lik=%.3f d_active=%d in %.1fs",
                            kernel_name, s, r["log_marginal_likelihood"],
                            r["d_latent_active"], r["elapsed_seconds"])
            else:
                logger.info("    %s seed=%d FAILED: %s",
                            kernel_name, s, r.get("status"))

    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        return {"status": "all_seeds_failed", "kernel": kernel_name,
                "per_seed": results}

    lls = [r["log_marginal_likelihood"] for r in ok]
    max_pair_diff = max(lls) - min(lls) if len(lls) >= 2 else 0.0
    if len(ok) == 3 and max_pair_diff <= SEED_AGREEMENT_NATS:
        convergence = "stable"
    elif len(ok) >= 2 and max_pair_diff <= SEED_AGREEMENT_NATS:
        convergence = "stable_with_failure"
    elif len(ok) >= 2:
        pair_diffs = sorted(
            abs(lls[i] - lls[j]) for i in range(len(lls)) for j in range(i + 1, len(lls))
        )
        convergence = ("flagged_optimisation"
                        if pair_diffs and pair_diffs[0] <= SEED_AGREEMENT_NATS
                        else "unstable_optimisation")
    else:
        convergence = "single_seed_only"

    sorted_lls = sorted(lls)
    median_ll = float(np.median(lls))
    median_idx_in_ok = int(np.argsort(lls)[len(lls) // 2])

    return {
        "status": "ok",
        "kernel": kernel_name,
        "n_seeds_attempted": len(seeds),
        "n_seeds_ok": len(ok),
        "convergence": convergence,
        "max_pair_diff_nats": float(max_pair_diff),
        "log_marginal_likelihoods_per_seed": lls,
        "median_log_marginal_likelihood": median_ll,
        "best_seed": int(ok[median_idx_in_ok]["seed"]),
        "median_seed_result": ok[median_idx_in_ok],
        "per_seed": results,
    }


# ─── BIC-style adjustment + K-aware BF gate ──────────────────────────────────

def bic_adjusted_ll(log_lik: float, n_hyp: int, K: int, D_obs: int = 1,
                     d_latent: int = 0) -> float:
    """BIC-adjusted marginal log likelihood.

    Effective sample size = K × D_obs (K observations each contributing D independent
    output-dim observations to the joint likelihood).

    `d_latent` is added to the parameter count to penalise kernels with more
    learnable latent axes. Without this, K3/K4/K6 (d=2) get a structural
    advantage over K1/K2/K5 (d=1) because they can use the extra latent
    freedom to fit anything — including isotropic noise.
    """
    n_params_eff = max(n_hyp + int(d_latent), 1)
    return log_lik - 0.5 * n_params_eff * math.log(max(K * max(D_obs, 1), 2))


def k_aware_bf_threshold(K: int, stage2c_cfg: dict) -> float:
    """Read the K-aware BF threshold from configs/stage2c.yaml (toy-calibrated).

    Falls back to the plan defaults (10 nats for K ≤ 10, 5 nats for K ≥ 11)
    if the config doesn't have an explicit K entry.
    """
    table = stage2c_cfg.get("bf_threshold_per_K", {})
    if str(K) in table:
        return float(table[str(K)])
    if K <= 10:
        return float(stage2c_cfg.get("bf_threshold_default_small_K", 10.0))
    return float(stage2c_cfg.get("bf_threshold_default_large_K", 5.0))


# ─── Held-out reconstruction (5-fold, with frozen kernel hyperparams) ────────

def _holdout_one_fold(fold_idx: int, mses_list: list, stream,
                        args_tuple: tuple) -> None:
    """Thread target: compute one hold-out fold's MSE on its own CUDA stream."""
    try:
        with torch.cuda.stream(stream):
            mses_list[fold_idx] = _compute_one_fold_mse(*args_tuple)
    except Exception as e:
        mses_list[fold_idx] = {"status": "thread_exception", "error": str(e)}


def _compute_one_fold_mse(y_tr, n_tr, y_ho, n_ho, kernel_name,
                            period_seed, period_seed_secondary,
                            d_max, n_iters, seed, device,
                            ho_idx, tr_idx) -> Optional[float]:
    """Single fold's MSE — extracted so it can be threaded.

    Returns the fold's MSE (float), or None on failure.
    """
    fit = fit_kernel_one_seed(y_tr, n_tr, kernel_name,
                                period_seed, period_seed_secondary,
                                d_max=d_max, n_iters=n_iters, seed=seed,
                                device=device)
    if fit.get("status") != "ok":
        return None
    z_tr = np.array(fit["z_final"])
    spec = kerns.KERNEL_REGISTRY[kernel_name]
    d_latent = spec["min_latent_dim"]
    d_latent = min(d_latent, max(1, len(tr_idx) - 1))
    # Linear-interp held-out latents from neighbours by tr_idx value-index
    order = np.argsort(tr_idx)
    tr_idx_sorted = tr_idx[order]
    z_tr_sorted = z_tr[order]
    z_ho_init = np.zeros((ho_idx.size, d_latent), dtype=np.float64)
    for i, v in enumerate(ho_idx):
        for d in range(d_latent):
            z_ho_init[i, d] = float(np.interp(float(v),
                                                tr_idx_sorted.astype(float),
                                                z_tr_sorted[:, d]))
    if spec["needs_period"]:
        if kernel_name == "K4_Torus":
            kmod = kerns.build_kernel(
                kernel_name, period=period_seed.P_init,
                regime=period_seed.regime,
                period_secondary=(period_seed_secondary.P_init if period_seed_secondary is not None else None),
                regime_secondary=(period_seed_secondary.regime if period_seed_secondary is not None else period_seed.regime),
            )
        else:
            kmod = kerns.build_kernel(kernel_name,
                                       period=period_seed.P_init,
                                       regime=period_seed.regime)
    else:
        kmod = kerns.build_kernel(kernel_name, active_dim_count=d_latent)
    kmod = kmod.to(device).double()
    y_tr_t = torch.tensor(y_tr, dtype=torch.float64, device=device)
    y_ho_t = torch.tensor(y_ho, dtype=torch.float64, device=device)
    z_tr_t = torch.tensor(z_tr, dtype=torch.float64, device=device)
    z_ho_t = torch.tensor(z_ho_init, dtype=torch.float64, device=device)
    try:
        with torch.no_grad():
            K_tt = kmod(z_tr_t).evaluate()
            # FP32 inner Cholesky for the held-out predictive (consistent with
            # training fit)
            K_full = (K_tt + (JITTER_INIT) * torch.eye(K_tt.shape[0],
                                                         dtype=K_tt.dtype,
                                                         device=K_tt.device)).to(torch.float32)
            L = torch.linalg.cholesky(K_full)
            alpha = torch.cholesky_solve(y_tr_t.to(torch.float32), L)
            K_ht = kmod(z_ho_t, z_tr_t).evaluate().to(torch.float32)
            mu_pred = K_ht @ alpha
            mse = float(torch.mean((y_ho_t.to(torch.float32) - mu_pred) ** 2).item())
        return mse
    except Exception:
        return None


def holdout_mse(y_centroids: np.ndarray, noise_scalar: np.ndarray,
                  kernel_name: str,
                  period_seed: Optional[PeriodSeed],
                  period_seed_secondary: Optional[PeriodSeed],
                  d_max: int = D_MAX_LATENT,
                  n_folds: int = 5,
                  seed: int = 42,
                  device: Optional[str] = None,
                  parallel: bool = False) -> Dict:
    """5-fold reconstruction MSE for one kernel."""
    if device is None:
        device = default_device()
    K, D = y_centroids.shape
    if K < 4:
        return {"status": "K_too_small", "K": K}

    fold_size = max(1, K // n_folds)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(K)
    n_folds_eff = min(n_folds, K)
    fold_args = []
    for fold in range(n_folds_eff):
        ho_idx = perm[fold * fold_size:(fold + 1) * fold_size]
        if ho_idx.size == 0:
            continue
        tr_idx = np.setdiff1d(np.arange(K), ho_idx)
        if tr_idx.size < 3:
            continue
        y_tr = y_centroids[tr_idx]
        n_tr = noise_scalar[tr_idx] if noise_scalar is not None else None
        y_ho = y_centroids[ho_idx]
        n_ho = noise_scalar[ho_idx] if noise_scalar is not None else None
        fold_args.append((y_tr, n_tr, y_ho, n_ho, kernel_name,
                           period_seed, period_seed_secondary,
                           d_max, N_LBFGS_ITERS, seed, device,
                           ho_idx, tr_idx))

    if not fold_args:
        return {"status": "no_valid_folds"}

    use_parallel = (parallel and device != "cpu"
                    and torch.cuda.is_available()
                    and len(fold_args) > 1)
    fold_results: list = [None] * len(fold_args)
    if use_parallel:
        import threading
        streams = [torch.cuda.Stream() for _ in fold_args]
        threads = []
        for i, args in enumerate(fold_args):
            t = threading.Thread(target=_holdout_one_fold,
                                 args=(i, fold_results, streams[i], args))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        torch.cuda.synchronize()
    else:
        for i, args in enumerate(fold_args):
            fold_results[i] = _compute_one_fold_mse(*args)

    mses = [m for m in fold_results
             if (m is not None
                 and not isinstance(m, dict)
                 and math.isfinite(m))]

    if not mses:
        return {"status": "all_folds_failed"}
    return {
        "status": "ok",
        "n_folds_ok": len(mses),
        "mse_mean": float(np.mean(mses)),
        "mse_se": float(np.std(mses, ddof=1) / math.sqrt(max(len(mses), 1)) if len(mses) >= 2 else 0.0),
        "mse_per_fold": [float(m) for m in mses],
    }


# ─── Discovery grid search for regime=discover periods ────────────────────────

def discover_period_grid(y_centroids: np.ndarray, noise_scalar: np.ndarray,
                          kernel_name: str,
                          K_present: int,
                          regime_secondary: str = "discover",
                          period_secondary: Optional[float] = None,
                          d_max: int = D_MAX_LATENT,
                          n_iters: int = 200,
                          seed: int = 42,
                          device: Optional[str] = None
                          ) -> Tuple[List[float], List[float], float]:
    """For a kernel whose primary period is in regime=discover, perform a
    coarse grid search over candidate integer periods P ∈ {2, ..., ⌊K/2⌋};
    return (candidates, ELBOs, best_candidate).
    """
    # Discovery grid covers plausible periods given the cell's unique label
    # count. Hard-capped at 30 candidates so a misconfigured caller can't
    # explode the discovery loop into thousands of fits.
    max_period = min(max(K_present // 2 + 1, 3), 32)
    candidates = [float(p) for p in range(2, max_period)]
    elbos = []
    for P in candidates:
        ps_primary = PeriodSeed(P_init=P, regime="discover", source="grid_search")
        ps_secondary = None
        if kernel_name == "K4_Torus":
            ps_secondary = PeriodSeed(
                P_init=(period_secondary if period_secondary is not None else P / 2.0),
                regime=regime_secondary, source="grid_search")
        r = fit_kernel_one_seed(
            y_centroids, noise_scalar, kernel_name,
            ps_primary, ps_secondary,
            d_max=d_max, n_iters=n_iters, seed=seed, device=device)
        if r.get("status") == "ok":
            elbos.append(float(r["log_marginal_likelihood"]))
        else:
            elbos.append(float("-inf"))
    if all(e == float("-inf") for e in elbos):
        return candidates, elbos, float(candidates[0])
    best_idx = int(np.argmax(elbos))
    return candidates, elbos, float(candidates[best_idx])


# ─── Bootstrap d̂ distribution on RBF fit ────────────────────────────────────

def bootstrap_d_hat(y_centroids: np.ndarray, noise_scalar: np.ndarray,
                     n_draws: int = BOOTSTRAP_D_HAT_DRAWS,
                     seed: int = 42,
                     device: Optional[str] = None,
                     short_iters: int = 200) -> Dict:
    """Bootstrap-resampled intrinsic-dimension estimate via PCA participation ratio.

    The participation ratio of a singular-value spectrum {s_i} is

        PR = (Σ s_i²)² / Σ s_i⁴ .

    Interpretation: PR equals the rank of M when the singular values are equal,
    drops toward 1 when one direction dominates. It is the standard Bayesian-
    spirit "effective dimensionality" estimator and does NOT require fitting a
    GP — avoiding the K=10 over-parameterisation issue that breaks GPLVM-ARD on
    centroid-only data. 200 bootstrap draws give the 2.5/97.5 percentile CI.

    Inputs `noise_scalar` and `device` are kept in the signature for API
    compatibility with the older GPLVM-based implementation; they are unused
    in the PCA path.
    """
    rng = np.random.default_rng(seed)
    K = y_centroids.shape[0]
    d_hats = []
    for d in range(n_draws):
        idx = rng.integers(0, K, K)
        M_b = np.asarray(y_centroids[idx], dtype=np.float64)
        M_c = M_b - M_b.mean(axis=0, keepdims=True)
        if M_c.shape[0] < 2:
            d_hats.append(0.0)
            continue
        try:
            s = np.linalg.svd(M_c, compute_uv=False)
        except np.linalg.LinAlgError:
            d_hats.append(0.0)
            continue
        s2 = s ** 2
        s4 = s2 ** 2
        denom = float(s4.sum())
        if denom <= 1e-30:
            d_hats.append(0.0)
        else:
            pr = float((s2.sum()) ** 2 / denom)
            d_hats.append(pr)
    arr = np.array(d_hats, dtype=np.float64)
    return {
        "n_draws": int(n_draws),
        "n_ok": int((arr > 0).sum()),
        "n_redraws": 0,
        "d_hat_per_draw": arr.tolist(),
        "d_hat_median": float(np.median(arr)),
        "d_hat_p025": float(np.quantile(arr, 0.025)),
        "d_hat_p975": float(np.quantile(arr, 0.975)),
        "method": "pca_participation_ratio",
    }


# ─── Per-axis credibility on the RBF kernel's ARD lengthscales ───────────────

def ard_credibility(rbf_result: dict, eps: float, d_max: int = D_MAX_LATENT) -> Dict:
    """Compute P(d̂ ≥ k | data) for k = 1..d_max from per-axis credibility.

    Each axis d has a "raw_lengthscale" mean (≈ softplus⁻¹(ℓ_d)). Under the
    Laplace approximation, the posterior is Normal(raw_mean, raw_var). An
    axis is "active" iff its inverse-lengthscale α = 1/ℓ² > ε, i.e.
    ℓ < 1/√ε, i.e. softplus(raw) < 1/√ε. Convert the per-axis credibility
    to P(active) via the Normal CDF.
    """
    out = {
        "epsilon": float(eps),
        "p_axis_active": [],
        "n_axes": 0,
        "d_post": 0,
        "p_d_geq_k": {f"k={k}": 0.0 for k in range(1, d_max + 1)},
    }
    if rbf_result.get("status") != "ok":
        return out
    ard = rbf_result.get("ard_info", {})
    raw_means = ard.get("raw_ls_mean", [])
    raw_vars = ard.get("raw_ls_var", [])
    if not raw_means:
        return out
    threshold_ls = 1.0 / math.sqrt(max(eps, 1e-12))
    # raw_lengthscale_max = softplus⁻¹(threshold_ls). softplus⁻¹(y) = log(exp(y)-1)
    if threshold_ls >= 50:
        raw_threshold = threshold_ls
    else:
        raw_threshold = math.log(math.expm1(threshold_ls))
    p_active = []
    for mu_x, var_x in zip(raw_means, raw_vars):
        sd = math.sqrt(max(var_x, 1e-12))
        # P(raw_lengthscale < raw_threshold) = Φ((raw_threshold − μ) / σ)
        # Use erf-based normal CDF
        z = (raw_threshold - mu_x) / sd
        p = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        p_active.append(float(p))
    out["p_axis_active"] = p_active
    out["n_axes"] = len(p_active)
    # Active count: sum of P(active) gives expected #active dims; integer
    # d̂_post = number of axes with P >= 0.95.
    out["d_post"] = int(sum(1 for p in p_active if p >= 0.95))
    # P(d̂ ≥ k) by treating independence (approximation): sort p_active desc
    sorted_p = sorted(p_active, reverse=True)
    for k in range(1, d_max + 1):
        if k > len(sorted_p):
            out["p_d_geq_k"][f"k={k}"] = 0.0
        else:
            # P(at least k axes active) under independence:
            # take top-k probabilities and compute P(top-k all active)
            prob = 1.0
            for p in sorted_p[:k]:
                prob *= p
            out["p_d_geq_k"][f"k={k}"] = float(prob)
    return out


# ─── Permutation null on best-kernel ──────────────────────────────────────────

def permutation_null(y_points: np.ndarray, noise_scalar_unused: Optional[np.ndarray],
                      best_kernel: str,
                      period_seed: Optional[PeriodSeed],
                      period_seed_secondary: Optional[PeriodSeed],
                      n_perms: int = N_PERMUTATIONS,
                      seed: int = 42,
                      device: Optional[str] = None) -> Dict:
    """Cross-coordinate column-shuffle null on the point cloud.

    Row-permutation of a point cloud is a no-op for GP marginal likelihood
    (it's a set of points, permutation-invariant). The honest null is
    "what would the GP fit look like if the cross-coordinate coupling were
    destroyed?" We shuffle each column of Z independently → marginals per
    coordinate are preserved, but any geometric structure across coordinates
    (which is what a kernel like Periodic, Torus, etc. picks up) is broken.

    For each permutation:
      1. For each column d, draw a random permutation of [0, N) and apply.
      2. Refit the best kernel on the column-shuffled point cloud (1 seed,
         short LBFGS iterations).
      3. Record BIC-adjusted log marginal likelihood.

    The empirical p-value is the fraction of nulls whose adj_ML ≥ observed
    adj_ML. Cells with low p_2c have geometry that cannot be explained by
    independent-per-coordinate marginals.
    """
    if device is None:
        device = default_device()
    rng = np.random.default_rng(seed)
    N, D = y_points.shape
    d_lat_k = int(kerns.KERNEL_REGISTRY[best_kernel]["min_latent_dim"])
    n_hyp = kerns.kernel_n_hyperparams(best_kernel, d_latent=d_lat_k)
    adj_lls = np.zeros(n_perms, dtype=np.float64)
    n_failed = 0
    for i in range(n_perms):
        y_p = np.empty_like(y_points)
        for d in range(D):
            y_p[:, d] = y_points[rng.permutation(N), d]
        r = fit_kernel_one_seed(
            y_p, None, best_kernel,
            period_seed, period_seed_secondary,
            d_max=D_MAX_LATENT, n_iters=PERM_LBFGS_ITERS,
            seed=seed + i, device=device)
        if r.get("status") == "ok":
            adj_lls[i] = bic_adjusted_ll(
                r["log_marginal_likelihood"], n_hyp, N, D_obs=D,
                d_latent=d_lat_k)
        else:
            adj_lls[i] = float("-inf")
            n_failed += 1
    return {
        "n_perms": int(n_perms),
        "n_failed": int(n_failed),
        "adj_ml_null": adj_lls.tolist(),
    }


def _empirical_p_value(observed: float, null_arr: np.ndarray) -> float:
    valid = null_arr[np.isfinite(null_arr)]
    if valid.size == 0:
        return 1.0
    return float((1 + np.sum(valid >= observed)) / (1 + valid.size))


# ─── Confidence tier ──────────────────────────────────────────────────────────

def assign_tier(K_present: int, min_n_v: int, k_u: int,
                 q_dsw: float, bf_gap: float, bf_threshold: float,
                 max_seed_range: float,
                 ratio_v_min: float) -> str:
    """Plan §C.10."""
    if k_u <= 0:
        return "DISCOVERY_ONLY"
    if K_present == 4:
        return "LOW"
    # Discovery: any value with n_v / k_u < 2 (highly underdetermined)
    if ratio_v_min < 2.0 or (k_u / max(min_n_v, 1)) >= 1.0:
        return "DISCOVERY_ONLY"
    if (K_present >= 6 and min_n_v >= 100 and q_dsw < 0.01
            and bf_gap >= 2.0 * bf_threshold and max_seed_range < 0.5):
        return "HIGH"
    if (K_present >= 5 and min_n_v >= 50 and q_dsw < 0.05
            and bf_gap >= bf_threshold):
        return "MEDIUM"
    return "LOW"


# ─── Per-cell analyse ────────────────────────────────────────────────────────

def analyze_cell(
    Z: np.ndarray, label_codes_full: np.ndarray, K_natural: int,
    model: str, task: str, mode: str, layer: int, concept: str,
    union_meta: dict, mu_layer_source: str,
    stage2a_row_lda: Optional[dict], stage2a_row_ccsvd: Optional[dict],
    P_spec_lda: Optional[np.ndarray], P_spec_ccsvd: Optional[np.ndarray],
    stage2c_cfg: dict,
    device: Optional[str],
    logger: logging.Logger,
) -> Dict:
    """Run Stage 2c on one cell, returning the full result dict.

    Z: (N_total, k_u) projected point cloud (post-mode-residualisation,
        post-mu_layer subtraction, post-union-basis projection, correct-mask
        applied).
    label_codes_full: (N_total,) concept value indices for each row, 0..K_natural-1.

    Stage 2c fits the Bayesian GPLVM on the FULL point cloud (one observation
    per correct example), with stratified subsampling to N_MAX = 3000 for
    cells exceeding the cap. The mu_stack / Lambda_full per-value summaries
    are computed on the full data and saved as auxiliary artefacts for
    Stage 3 / plotting — they do NOT enter the GP fit.
    """
    t0 = time.time()
    k_u = Z.shape[1]
    # Filter values with min_group_size
    label_codes, kept_values, mask = filter_values_by_count(
        label_codes_full, K_natural, MIN_GROUP_SIZE)
    if not mask.any():
        return {"status": "no_values_survive_filter", "K_present": 0,
                 "K_natural": int(K_natural), "k_u": int(k_u)}
    Z_full = Z[mask]
    K_present = len(kept_values)
    N_full = Z_full.shape[0]
    if K_present < MIN_K_FOR_STAGE2C:
        return {"status": V_LOW_K, "K_present": K_present,
                 "K_natural": int(K_natural), "k_u": int(k_u),
                 "N_used": N_full}

    # Auxiliary centroid+spread summaries on the FULL point cloud (saved for
    # Stage 3 and plotting; not consumed by the GP fit)
    mu_stack, noise_scalar, Lambda_full, sigma_meta = compute_centroids_and_noise(
        Z_full, label_codes, K_present,
        lw_threshold=10.0, oas_threshold=5.0)
    counts = np.bincount(label_codes, minlength=K_present)
    min_n_v = int(counts.min())
    ratio_v_min = float(min_n_v / max(k_u, 1))

    # Subsample for kernel-matrix tractability (parent's pattern, raised from
    # 1500 to 3000 to use more data when available). Stratified by value so no
    # value loses coverage. Deterministic per cell.
    N_max_cell = int(SUBSAMPLE_N_MAX)
    rng_sub = np.random.default_rng(
        int.from_bytes(hashlib.sha256(
            f"stage2c-sub|{model}|{task}|{mode}|{layer}|{concept}".encode()
        ).digest()[:8], "big"))
    sub_idx = stratified_subsample(N_full, label_codes, N_max_cell, rng_sub)
    Z_used = Z_full[sub_idx]
    label_codes_used = label_codes[sub_idx]
    N_used = Z_used.shape[0]

    # Choose primary stage2a metadata: prefer lda_a if available, else ccsvd
    stage2a_used = "lda_a" if stage2a_row_lda is not None else (
        "ccsvd" if stage2a_row_ccsvd is not None else None)
    stage2a_row = stage2a_row_lda if stage2a_used == "lda_a" else stage2a_row_ccsvd
    P_spec_used = P_spec_lda if stage2a_used == "lda_a" else P_spec_ccsvd
    P_top1 = float(stage2a_row.get("discovered_period", 0.0)) if stage2a_row else 0.0
    confident = stage2a_confident(stage2a_row) if stage2a_row else False
    period_match = bool(stage2a_row.get("period_match", False)) if stage2a_row else False

    # Determine period regime
    if confident:
        regime_primary = REGIME_NARROW
    elif stage2a_row is not None and \
            str(stage2a_row.get("geometry_detected",
                                 stage2a_row.get("verdict", ""))) in {"helix", "circle"}:
        # Stage 2a found something but period_match=False → wide prior
        regime_primary = REGIME_WIDE
    else:
        regime_primary = REGIME_DISCOVER

    # Derive top-2 period for Torus
    P_top2 = None
    pw_top2 = 0.0
    pw_top1 = 0.0
    median_floor = 0.0
    if P_spec_used is not None and P_top1 > 0:
        P_top2, pw_top2, pw_top1 = derive_top2_period(P_spec_used, K_present, P_top1)
        if P_spec_used.size > 0:
            median_floor = float(np.median(P_spec_used.mean(axis=1)))
    top2_conf = top2_confident(pw_top2, pw_top1, median_floor) if P_top2 else False
    regime_secondary = (REGIME_NARROW if (confident and top2_conf) else REGIME_DISCOVER)

    # Build period seeds per kernel (and resolve discovery via grid)
    P_init_periodic = P_top1 if P_top1 > 0 else (K_present / 2.0)
    period_regime_per_kernel = {}
    period_seeds: Dict[str, Tuple[Optional[PeriodSeed], Optional[PeriodSeed]]] = {}

    def maybe_discover(kernel_name, P_primary, regime_primary,
                        P_secondary=None, regime_secondary_in="narrow"):
        ps_primary = None
        if kerns.KERNEL_REGISTRY[kernel_name]["needs_period"]:
            if regime_primary == REGIME_DISCOVER:
                cand, elbos, P_win = discover_period_grid(
                    Z_used, None, kernel_name,
                    N_used, regime_secondary=regime_secondary_in,
                    period_secondary=P_secondary, device=device)
                ps_primary = PeriodSeed(P_init=P_win, regime=REGIME_DISCOVER,
                                          source="grid_search",
                                          grid_candidates=cand,
                                          grid_elbos=elbos,
                                          grid_winner=P_win)
            else:
                ps_primary = PeriodSeed(P_init=P_primary, regime=regime_primary,
                                          source=f"stage2a_top1_{stage2a_used}")
        ps_secondary = None
        if kernel_name == "K4_Torus":
            if P_secondary is None or regime_secondary_in == REGIME_DISCOVER:
                cand2, elbos2, P2_win = discover_period_grid(
                    Z_used, None, kernel_name,
                    N_used, regime_secondary=REGIME_DISCOVER,
                    period_secondary=ps_primary.P_init / 2.0 if ps_primary else None,
                    device=device)
                ps_secondary = PeriodSeed(P_init=P2_win, regime=REGIME_DISCOVER,
                                            source="grid_search",
                                            grid_candidates=cand2,
                                            grid_elbos=elbos2,
                                            grid_winner=P2_win)
            else:
                ps_secondary = PeriodSeed(P_init=P_secondary, regime=regime_secondary_in,
                                            source=f"stage2a_top2_{stage2a_used}")
        return ps_primary, ps_secondary

    for kernel_name in kerns.KERNEL_NAMES:
        spec = kerns.KERNEL_REGISTRY[kernel_name]
        if not spec["needs_period"]:
            period_seeds[kernel_name] = (None, None)
            period_regime_per_kernel[kernel_name] = {}
            continue
        if kernel_name == "K4_Torus":
            ps_p, ps_s = maybe_discover(
                kernel_name, P_init_periodic, regime_primary,
                P_secondary=P_top2, regime_secondary_in=regime_secondary)
            period_seeds[kernel_name] = (ps_p, ps_s)
            period_regime_per_kernel[kernel_name] = {
                "P_1": ps_p.regime, "P_2": ps_s.regime,
            }
        else:
            ps_p, _ = maybe_discover(kernel_name, P_init_periodic, regime_primary)
            period_seeds[kernel_name] = (ps_p, None)
            period_regime_per_kernel[kernel_name] = {"P": ps_p.regime}

    logger.info("  cell %s/%s/mode_%s/L%d/%s N_full=%d N_used=%d K=%d k_u=%d min_n_v=%d",
                model, task, mode, layer, concept, N_full, N_used, K_present, k_u, min_n_v)
    logger.info("    stage2a verdict=%s P_top1=%.2f confident=%s P_top2=%s top2_confident=%s",
                stage2a_row.get("geometry_detected", "?") if stage2a_row else "?",
                P_top1, confident, P_top2, top2_conf)
    logger.info("    period regimes: %s", period_regime_per_kernel)

    # Fit each kernel with 3 seeds on the full point cloud
    kernel_results = {}
    elbo_matrix = np.zeros((len(kerns.KERNEL_NAMES), N_SEEDS), dtype=np.float64)
    for ki, kernel_name in enumerate(kerns.KERNEL_NAMES):
        ps_p, ps_s = period_seeds[kernel_name]
        seeds = (cell_seed(model, task, mode, layer, concept, kernel_name, i) % (2**31 - 1)
                  for i in range(N_SEEDS))
        seeds_tuple = tuple(seeds)
        res = fit_kernel_three_seeds(
            Z_used, None, kernel_name, ps_p, ps_s,
            seeds=seeds_tuple, device=device, logger=logger,
            parallel=True)
        kernel_results[kernel_name] = res
        if res.get("status") == "ok":
            for si, ll in enumerate(res["log_marginal_likelihoods_per_seed"]):
                if si < N_SEEDS:
                    elbo_matrix[ki, si] = ll

    # BIC-adjusted ML per kernel — sample size for BIC is now N_used × k_u
    adj_mls = {}
    median_lls = {}
    seed_ranges = {}
    for kernel_name, res in kernel_results.items():
        if res.get("status") != "ok":
            adj_mls[kernel_name] = float("-inf")
            median_lls[kernel_name] = float("-inf")
            seed_ranges[kernel_name] = float("inf")
            continue
        median_ll = float(res["median_log_marginal_likelihood"])
        d_lat_k = int(kerns.KERNEL_REGISTRY[kernel_name]["min_latent_dim"])
        n_hyp = kerns.kernel_n_hyperparams(kernel_name, d_latent=d_lat_k)
        adj_mls[kernel_name] = bic_adjusted_ll(median_ll, n_hyp, N_used,
                                                  D_obs=k_u,
                                                  d_latent=d_lat_k)
        median_lls[kernel_name] = median_ll
        seed_ranges[kernel_name] = float(res["max_pair_diff_nats"])

    # Determine winner + K-aware BF gap (plan §C.10: 10 nats for K ≤ 10
    # because the kernel competition has too few degrees of freedom to be
    # decisive at 5 nats; 5 nats for K ≥ 11 — standard Kass-Raftery).
    ranked = sorted(adj_mls.items(), key=lambda x: x[1], reverse=True)
    winner_kernel, winner_adj = ranked[0]
    runner_adj = ranked[1][1] if len(ranked) > 1 else float("-inf")
    bf_gap = winner_adj - runner_adj
    bf_threshold = k_aware_bf_threshold(K_present, stage2c_cfg)
    bf_pass = bf_gap >= bf_threshold and math.isfinite(winner_adj)
    seed_pass = seed_ranges.get(winner_kernel, float("inf")) <= SEED_AGREEMENT_NATS

    # Held-out MSE for the winner and runner-up
    holdout = {}
    for kname in (winner_kernel, ranked[1][0]) if len(ranked) > 1 else (winner_kernel,):
        ps_p, ps_s = period_seeds[kname]
        h = holdout_mse(Z_used, None, kname, ps_p, ps_s,
                         device=device, parallel=True)
        holdout[kname] = h
    holdout_pass = False
    if (holdout.get(winner_kernel, {}).get("status") == "ok"
            and holdout.get(ranked[1][0], {}).get("status") == "ok"
            and len(ranked) > 1):
        mse_w = holdout[winner_kernel]["mse_mean"]
        mse_r = holdout[ranked[1][0]]["mse_mean"]
        mse_se_r = holdout[ranked[1][0]].get("mse_se", 0.0)
        holdout_pass = mse_w <= mse_r - mse_se_r

    # Selective permutation null: only run on cells passing the verdict gate
    # (BF + seed + holdout). Cells without a positive verdict make no
    # significance claim, so the 1000-perm null is skipped.
    ps_p_win, ps_s_win = period_seeds[winner_kernel]
    verdict_candidate = bf_pass and seed_pass and holdout_pass
    if verdict_candidate:
        perm = permutation_null(
            Z_used, None, winner_kernel, ps_p_win, ps_s_win,
            n_perms=N_PERMUTATIONS, seed=42, device=device)
        p_2c = _empirical_p_value(
            adj_mls[winner_kernel], np.array(perm["adj_ml_null"]))
    else:
        perm = {"n_perms": 0, "n_failed": 0, "adj_ml_null": []}
        p_2c = float("nan")

    # ARD credibility on the RBF fit + bootstrap d̂ via PCA participation ratio
    rbf_res = kernel_results.get("K1_RBF", {})
    rbf_median = rbf_res.get("median_seed_result") if rbf_res.get("status") == "ok" else None
    epsilon = float(stage2c_cfg.get("ard_epsilon", 0.01))
    ard_post = ard_credibility(rbf_median or {}, eps=epsilon, d_max=D_MAX_LATENT)
    boot_d = bootstrap_d_hat(Z_used, None,
                              n_draws=BOOTSTRAP_D_HAT_DRAWS,
                              seed=42, device=device)

    # Lengthscale-collapse check on composite kernels (K3, K6): if the
    # periodic-arm lengthscale ran away (>50× the non-periodic arm), the
    # winning kernel collapsed to its non-periodic part — downgrade to smooth.
    composite_downgrade = None
    if winner_kernel in {"K3_PeriodicLinear", "K6_PeriodicRBF"}:
        kres = kernel_results.get(winner_kernel, {})
        wm = kres.get("median_seed_result", {}) if kres.get("status") == "ok" else {}
        ls_list = wm.get("final_lengthscales", [])
        if len(ls_list) >= 2 and ls_list[0] > 50.0 * max(ls_list[1], 1e-6):
            composite_downgrade = "K1_RBF"

    # Verdict assignment. The 1000-perm null only ran when verdict_candidate
    # was True; cells without a verdict candidate get p_2c = NaN and verdict
    # = inconclusive regardless.
    pre_fdr_verdict = V_INCONCLUSIVE
    if verdict_candidate and math.isfinite(p_2c) and p_2c < FDR_ALPHA:
        effective = composite_downgrade or winner_kernel
        pre_fdr_verdict = KERNEL_TO_VERDICT.get(effective, V_INCONCLUSIVE)
        if (winner_kernel in _PERIODIC_KERNELS
                and ps_p_win is not None and ps_p_win.regime == REGIME_DISCOVER
                and K_present < 6):
            pre_fdr_verdict = V_DIM_ONLY
    elif bf_pass and not seed_pass:
        pre_fdr_verdict = V_KERNEL_INCONCLUSIVE_SEEDS
    elif (math.isfinite(p_2c) and p_2c < FDR_ALPHA
            and ard_post.get("p_d_geq_k", {}).get("k=1", 0.0) >= 0.95):
        pre_fdr_verdict = V_DIM_ONLY

    # Dimension-only fallback for inconclusive cells: if no kernel won the
    # competition but the point cloud has clear ≥1D structure (bootstrap
    # participation ratio ≥ 1.5 OR ARD posterior says d≥1 with high
    # probability), still report a dim_only verdict so the cell contributes
    # an intrinsic-dimension reading instead of being purely uninformative.
    # The dimension itself (up to D_MAX_LATENT=5) is in the summary row's
    # d_hat_bootstrap_median and ard_d_post fields.
    if pre_fdr_verdict == V_INCONCLUSIVE:
        d_hat = float(boot_d.get("d_hat_median", 0.0))
        p_d_geq_1 = float(ard_post.get("p_d_geq_k", {}).get("k=1", 0.0))
        if d_hat >= 1.5 or p_d_geq_1 >= 0.95:
            pre_fdr_verdict = V_DIM_ONLY

    # Tier (uses p_2c as a placeholder for q in pre-FDR pass; aggregator
    # will recompute with global BH-FDR q)
    tier = assign_tier(
        K_present, min_n_v, k_u, p_2c, bf_gap, bf_threshold,
        max_seed_range=seed_ranges.get(winner_kernel, 0.0),
        ratio_v_min=ratio_v_min)

    summary_row = {
        "model": model, "task": task, "mode": mode, "layer": int(layer),
        "concept": concept,
        "K_natural": int(K_natural), "K_present": int(K_present),
        "N_used": int(N_used), "min_n_v": int(min_n_v),
        "k_u": int(k_u), "ratio_v_min": float(ratio_v_min),
        "stage2a_used": stage2a_used or "",
        "stage2a_confident_period": bool(confident),
        "stage2a_P_top1": float(P_top1),
        "stage2a_P_top2": float(P_top2) if P_top2 is not None else float("nan"),
        "stage2a_period_match": bool(period_match),
        "top2_confidence_flag": bool(top2_conf),
        "winner_kernel": winner_kernel,
        "winner_adj_ml": float(winner_adj),
        "runner_kernel": ranked[1][0] if len(ranked) > 1 else "",
        "runner_adj_ml": float(runner_adj),
        "bf_gap_nats": float(bf_gap),
        "bf_threshold_nats": float(bf_threshold),
        "bf_pass": bool(bf_pass),
        "seed_pass": bool(seed_pass),
        "holdout_pass": bool(holdout_pass),
        "p_2c": float(p_2c),
        "q_2c": float("nan"),                                  # filled by aggregator
        "verdict_pre_fdr": pre_fdr_verdict,
        "verdict_post_fdr": pre_fdr_verdict,                   # placeholder
        "tier": tier,
        "winner_seed_range_nats": float(seed_ranges.get(winner_kernel, 0.0)),
        "ard_d_post": int(ard_post.get("d_post", 0)),
        "ard_P_d_geq_1": float(ard_post.get("p_d_geq_k", {}).get("k=1", 0.0)),
        "ard_P_d_geq_2": float(ard_post.get("p_d_geq_k", {}).get("k=2", 0.0)),
        "ard_P_d_geq_3": float(ard_post.get("p_d_geq_k", {}).get("k=3", 0.0)),
        "d_hat_bootstrap_median": float(boot_d.get("d_hat_median", 0.0)),
        "d_hat_bootstrap_p025": float(boot_d.get("d_hat_p025", 0.0)),
        "d_hat_bootstrap_p975": float(boot_d.get("d_hat_p975", 0.0)),
        "elapsed_seconds": float(time.time() - t0),
    }
    # Add per-kernel adj_ml columns
    for kname in kerns.KERNEL_NAMES:
        summary_row[f"adj_ml_{kname}"] = float(adj_mls.get(kname, float("-inf")))
        summary_row[f"median_ll_{kname}"] = float(median_lls.get(kname, float("-inf")))
        summary_row[f"seed_range_{kname}"] = float(seed_ranges.get(kname, float("inf")))

    return {
        "status": "ok",
        "summary_row": summary_row,
        "kernel_results": kernel_results,
        "period_regime_per_kernel": period_regime_per_kernel,
        "period_seeds": {
            k: {
                "primary": (None if v[0] is None else {
                    "P_init": v[0].P_init, "regime": v[0].regime,
                    "source": v[0].source,
                    "grid_candidates": v[0].grid_candidates,
                    "grid_elbos": v[0].grid_elbos,
                    "grid_winner": v[0].grid_winner,
                }),
                "secondary": (None if v[1] is None else {
                    "P_init": v[1].P_init, "regime": v[1].regime,
                    "source": v[1].source,
                    "grid_candidates": v[1].grid_candidates,
                    "grid_elbos": v[1].grid_elbos,
                    "grid_winner": v[1].grid_winner,
                }),
            } for k, v in period_seeds.items()
        },
        "elbo_per_kernel_seed": elbo_matrix,
        "mu_stack": mu_stack,
        "noise_scalar": noise_scalar,
        "Lambda_full": Lambda_full,
        "holdout": holdout,
        "perm_null_adj_ml": np.array(perm["adj_ml_null"], dtype=np.float64),
        "n_perm_failed": int(perm["n_failed"]),
        "ard_posterior": ard_post,
        "bootstrap_d_hat": boot_d,
        "union_meta": union_meta,
        "sigma_meta": sigma_meta,
        "kept_values": kept_values,
        "mu_layer_source": mu_layer_source,
    }


# ─── Concept discovery for a (model, task, mode, layer) ──────────────────────

def discover_concepts_for_cell(results_root: Path, model: str, task: str,
                                 mode: str, layer: int,
                                 stage2a_summary_lda: Optional[pd.DataFrame],
                                 stage2a_summary_ccsvd: Optional[pd.DataFrame]
                                 ) -> List[str]:
    """Stage 2c eligibility: union of concepts where Stage 2a verdict ∈
    {helix, circle, none, sparse_value_grid} for either lda_a or ccsvd, AND both
    LDA-A and CCSVD bases exist on disk for the cell."""
    eligible_verdicts = {"helix", "circle", "none", "sparse_value_grid"}
    concepts: set = set()
    for df in (stage2a_summary_lda, stage2a_summary_ccsvd):
        if df is None:
            continue
        sub = df[(df["layer"].astype(int) == int(layer))
                  & (df["geometry_detected"].isin(eligible_verdicts))]
        for c in sub["concept"].unique():
            concepts.add(str(c))
    # Filter by basis existence
    have = []
    for c in sorted(concepts):
        if (lda_basis_path(results_root, model, task, layer, c, mode).exists()
                and ccsvd_basis_path(results_root, model, task, layer, c, mode).exists()):
            have.append(c)
    return have


# ─── IO write per cell ───────────────────────────────────────────────────────

def write_cell_artifacts(out_dir: Path, result: dict, logger: logging.Logger) -> None:
    summary = result["summary_row"]
    df = pd.DataFrame([summary])
    atomic_csv(df, out_dir / "gplvm_results.csv")
    atomic_save(np.asarray(result["elbo_per_kernel_seed"], dtype=np.float64),
                 out_dir / "elbo_per_kernel_seed.npy")
    atomic_save(np.asarray(result["mu_stack"], dtype=np.float64),
                 out_dir / "mu_stack.npy")
    atomic_save(np.asarray(result["noise_scalar"], dtype=np.float64),
                 out_dir / "noise_scalar.npy")
    atomic_save(np.asarray(result["Lambda_full"], dtype=np.float64),
                 out_dir / "Lambda_full.npy")
    atomic_save(np.asarray(result["perm_null_adj_ml"], dtype=np.float64),
                 out_dir / "perm_null.npy")
    atomic_save(np.array(result["bootstrap_d_hat"]["d_hat_per_draw"], dtype=np.float64),
                 out_dir / "bootstrap_d_hat.npy")
    # Persist hyperparameters for the median seed of every kernel
    hyp = {}
    for kname, kres in result["kernel_results"].items():
        if kres.get("status") == "ok":
            ms = kres.get("median_seed_result", {})
            hyp[kname] = {
                "final_periods": ms.get("final_periods", []),
                "final_lengthscales": ms.get("final_lengthscales", []),
                "d_latent_active": ms.get("d_latent_active", 0),
                "noise_var_mean": ms.get("noise_var_mean", float("nan")),
                "jitter_used": ms.get("jitter_used", float("nan")),
                "log_marg_lik": ms.get("log_marginal_likelihood", float("nan")),
            }
    atomic_json(hyp, out_dir / "kernel_hyperparams.json")
    atomic_json({"posterior": result["ard_posterior"],
                  "bootstrap": result["bootstrap_d_hat"]},
                 out_dir / "ard_posterior.json")
    atomic_json(result["union_meta"], out_dir / "union_basis_meta.json")
    atomic_json({
        "holdout": result["holdout"],
        "period_regime_per_kernel": result["period_regime_per_kernel"],
        "period_seeds": result["period_seeds"],
        "sigma_meta": result["sigma_meta"],
        "kept_values": result["kept_values"],
        "n_perm_failed": result["n_perm_failed"],
        "summary": summary,
        "mu_layer_source": result["mu_layer_source"],
        "computation_status": "complete",
    }, out_dir / "metadata.json")


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


# ─── Worker entry point ──────────────────────────────────────────────────────

def setup_logging(logs_root: Path, model: str, task: str, mode: str) -> logging.Logger:
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


def main_one_cell(args, cfg, paths, stage2c_cfg) -> int:
    model = args.model
    task = args.task
    mode = args.mode
    layer = int(args.layer)
    concept = args.concept
    logger = setup_logging(paths["logs_root"], model, task, mode)
    logger.info("=== Stage 2c single-cell smoke: %s/%s/mode_%s/layer_%d/%s ===",
                model, task, mode, layer, concept)

    out_dir = stage2c_cell_dir(paths["results_root"], model, task, mode,
                                 layer, concept)
    if cell_complete(out_dir) and not args.force:
        logger.info("Cell already complete, skipping (use --force to overwrite).")
        return 0

    # Load activations + correct mask
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
        logger.error("Activation row count %d != answers row count %d",
                       X_all.shape[0], correct_mask.shape[0])
        return 1
    X = X_all[correct_mask].astype(np.float64)
    # Load problems CSV for concept value labels
    prob_path = paths["data_root"] / "data" / "raw" / f"{task}_problems.csv"
    if not prob_path.exists():
        logger.error("Problems file missing: %s", prob_path)
        return 1
    prob_df = pd.read_csv(prob_path)
    if concept not in prob_df.columns:
        logger.error("Concept %r not in problems CSV columns: %s",
                       concept, list(prob_df.columns))
        return 1
    labels_full = prob_df[concept].to_numpy()
    labels_correct = labels_full[correct_mask]
    # Map labels to dense 0..K_natural-1
    unique_full_values = sorted(prob_df[concept].dropna().unique().tolist())
    K_natural = len(unique_full_values)
    value_to_code = {v: i for i, v in enumerate(unique_full_values)}
    label_codes = np.array(
        [value_to_code.get(v, -1) for v in labels_correct], dtype=np.int64)
    keep = label_codes >= 0
    X = X[keep]
    label_codes = label_codes[keep]

    # Union basis
    B_u, union_meta = build_union_basis(
        paths["results_root"], model, task, mode, layer, concept)
    if union_meta["k_u"] < 1:
        logger.error("Union basis empty for %s — abort (no LDA-A or CCSVD basis)", concept)
        return 2
    logger.info("Union basis k_u=%d (LDA + CCSVD contributions: %s)",
                 union_meta["k_u"],
                 [c["source"] + f"({c['n_dims']})" for c in union_meta["contributions"]])

    # Layer mean computed from the correct-mask activations (CCSVD meta.json
    # does not store it). The mode-residualised activations already have their
    # own conditional mean structure; we still centre to be safe.
    mu_layer = X.mean(axis=0)
    Z = (X - mu_layer).dot(B_u.astype(np.float64))   # (N, k_u)

    # Stage 2a metadata (both variants if present)
    s2a_lda = read_stage2a_row(paths["results_root"], model, task, mode,
                                "lda_a", layer, concept)
    s2a_ccsvd = read_stage2a_row(paths["results_root"], model, task, mode,
                                  "ccsvd", layer, concept)
    p_spec_lda = read_stage2a_periodogram(paths["results_root"], model, task, mode,
                                            "lda_a", layer, concept)
    p_spec_ccsvd = read_stage2a_periodogram(paths["results_root"], model, task, mode,
                                              "ccsvd", layer, concept)

    result = analyze_cell(
        Z, label_codes, K_natural,
        model, task, mode, layer, concept,
        union_meta, mu_layer_source="ccsvd_meta",
        stage2a_row_lda=s2a_lda, stage2a_row_ccsvd=s2a_ccsvd,
        P_spec_lda=p_spec_lda, P_spec_ccsvd=p_spec_ccsvd,
        stage2c_cfg=stage2c_cfg, device=default_device(), logger=logger,
    )

    if result.get("status") != "ok":
        logger.warning("Cell finished with status=%s — writing minimal artefacts.",
                        result.get("status"))
        # Write a sparse summary CSV to make the cell appear in the aggregator
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_csv(pd.DataFrame([{
            "model": model, "task": task, "mode": mode, "layer": int(layer),
            "concept": concept,
            "status": result.get("status"),
            "K_natural": result.get("K_natural", K_natural),
            "K_present": result.get("K_present", 0),
            "k_u": result.get("k_u", union_meta["k_u"]),
            "N_used": result.get("N_used", 0),
            "verdict_pre_fdr": V_LOW_K if result.get("status") == V_LOW_K else V_INCONCLUSIVE,
            "verdict_post_fdr": V_LOW_K if result.get("status") == V_LOW_K else V_INCONCLUSIVE,
            "tier": "LOW",
        }]), out_dir / "gplvm_results.csv")
        atomic_json({"computation_status": "complete",
                      "status": result.get("status"),
                      "union_meta": union_meta}, out_dir / "metadata.json")
        return 0

    write_cell_artifacts(out_dir, result, logger)
    logger.info("=== Done: verdict=%s tier=%s bf_gap=%.2f nats p_2c=%.4f ===",
                 result["summary_row"]["verdict_pre_fdr"],
                 result["summary_row"]["tier"],
                 result["summary_row"]["bf_gap_nats"],
                 result["summary_row"]["p_2c"])
    return 0


def main_all_cells_for_cohort(args, cfg, paths, stage2c_cfg) -> int:
    """Walk all (mode × layer × concept) cells for one (model, task) cohort and
    write per-cell artefacts + per-model summary CSVs."""
    model = args.model
    task = args.task
    logger = setup_logging(paths["logs_root"], model, task, "all")
    modes = ["off", "answer", "norm"] if args.mode == "all" else [args.mode]
    model_cfg = next(m for m in cfg["models"] if m["key"] == model)
    layers = model_cfg["layers"]
    if args.layer != -1:
        layers = [int(args.layer)]

    # Load all Stage 2a summaries once per (model, task, mode)
    summary_rows = []
    for mode in modes:
        s2a_lda_path = stage2a_summary_path(paths["results_root"], model, task, mode, "lda_a")
        s2a_ccsvd_path = stage2a_summary_path(paths["results_root"], model, task, mode, "ccsvd")
        s2a_lda_df = pd.read_csv(s2a_lda_path) if s2a_lda_path.exists() else None
        s2a_ccsvd_df = pd.read_csv(s2a_ccsvd_path) if s2a_ccsvd_path.exists() else None

        # Load activations + correct mask + problems CSV
        ans_path = paths["data_root"] / "answers" / model / f"{task}_answers.csv"
        ans_df = pd.read_csv(ans_path)
        correct_mask = ans_df["correct"].astype(bool).to_numpy()
        prob_df = pd.read_csv(paths["data_root"] / "data" / "raw" / f"{task}_problems.csv")

        for layer in layers:
            X_path = activation_path(paths["activations_root"], paths["results_root"],
                                       model, task, layer, mode)
            if not X_path.exists():
                logger.warning("Skip %s/%s/mode_%s/L%d — no activations: %s",
                                model, task, mode, layer, X_path)
                continue
            X_all = np.load(X_path)
            X = X_all[correct_mask].astype(np.float64)

            concepts = discover_concepts_for_cell(
                paths["results_root"], model, task, mode, layer,
                s2a_lda_df, s2a_ccsvd_df)
            logger.info("[mode=%s L%d] %d eligible concepts", mode, layer, len(concepts))
            for concept in concepts:
                out_dir = stage2c_cell_dir(paths["results_root"], model, task, mode,
                                             layer, concept)
                if cell_complete(out_dir) and not args.force:
                    # Read its summary row in for the per-model summary
                    try:
                        df = pd.read_csv(out_dir / "gplvm_results.csv")
                        for _, row in df.iterrows():
                            summary_rows.append(row.to_dict())
                    except Exception:
                        pass
                    continue

                if concept not in prob_df.columns:
                    logger.warning("Concept %s not in problems columns — skip",
                                    concept)
                    continue
                labels_full = prob_df[concept].to_numpy()
                labels_correct = labels_full[correct_mask]
                unique_full_values = sorted(prob_df[concept].dropna().unique().tolist())
                K_natural = len(unique_full_values)
                value_to_code = {v: i for i, v in enumerate(unique_full_values)}
                label_codes = np.array(
                    [value_to_code.get(v, -1) for v in labels_correct], dtype=np.int64)
                keep = label_codes >= 0
                X_c = X[keep]
                label_codes = label_codes[keep]

                B_u, union_meta, mu_layer = build_union_basis(
                    paths["results_root"], model, task, mode, layer, concept)
                if union_meta["k_u"] < 2:
                    logger.info("[mode=%s L%d %s] union basis too small (k_u=%d); skip",
                                  mode, layer, concept, union_meta["k_u"])
                    continue
                Z = (X_c - mu_layer.astype(np.float64)).dot(B_u.astype(np.float64))

                s2a_lda = read_stage2a_row(paths["results_root"], model, task,
                                             mode, "lda_a", layer, concept)
                s2a_ccsvd = read_stage2a_row(paths["results_root"], model, task,
                                               mode, "ccsvd", layer, concept)
                p_spec_lda = read_stage2a_periodogram(paths["results_root"],
                                                      model, task, mode, "lda_a",
                                                      layer, concept)
                p_spec_ccsvd = read_stage2a_periodogram(paths["results_root"],
                                                        model, task, mode, "ccsvd",
                                                        layer, concept)
                try:
                    result = analyze_cell(
                        Z, label_codes, K_natural,
                        model, task, mode, layer, concept,
                        union_meta, mu_layer_source="ccsvd_meta",
                        stage2a_row_lda=s2a_lda, stage2a_row_ccsvd=s2a_ccsvd,
                        P_spec_lda=p_spec_lda, P_spec_ccsvd=p_spec_ccsvd,
                        stage2c_cfg=stage2c_cfg, device=default_device(),
                        logger=logger,
                    )
                except Exception as e:
                    logger.exception("Cell %s failed: %s", concept, e)
                    continue

                if result.get("status") != "ok":
                    out_dir.mkdir(parents=True, exist_ok=True)
                    atomic_csv(pd.DataFrame([{
                        "model": model, "task": task, "mode": mode,
                        "layer": int(layer), "concept": concept,
                        "status": result.get("status"),
                        "K_natural": result.get("K_natural", K_natural),
                        "K_present": result.get("K_present", 0),
                        "k_u": result.get("k_u", union_meta["k_u"]),
                        "N_used": result.get("N_used", 0),
                        "verdict_pre_fdr": V_LOW_K if result.get("status") == V_LOW_K else V_INCONCLUSIVE,
                        "verdict_post_fdr": V_LOW_K if result.get("status") == V_LOW_K else V_INCONCLUSIVE,
                        "tier": "LOW",
                    }]), out_dir / "gplvm_results.csv")
                    atomic_json({"computation_status": "complete",
                                  "status": result.get("status"),
                                  "union_meta": union_meta},
                                 out_dir / "metadata.json")
                    continue
                write_cell_artifacts(out_dir, result, logger)
                summary_rows.append(result["summary_row"])
        # Write the per-model summary CSV (one per mode)
        if summary_rows:
            out_path = (paths["results_root"] / "stage2c_gplvm" / model
                          / f"summary_{model}_{task}_mode_{mode}.csv")
            atomic_csv(pd.DataFrame(summary_rows), out_path)

    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--stage2c-config", default=str(STAGE2C_CONFIG_DEFAULT))
    # Single-cell / single-cohort mode
    p.add_argument("--model", default=None,
                    help="model key (e.g. gpt-j-6b); omit when using --sweep all")
    p.add_argument("--task", default=None, choices=["addition", "multiplication", None],
                    help="task; omit when using --sweep all")
    p.add_argument("--mode", default="off",
                    choices=["off", "answer", "norm", "all"])
    p.add_argument("--layer", type=int, default=-1,
                    help="-1 = all layers in config")
    p.add_argument("--concept", default="",
                    help="single concept; if empty, iterate all eligible concepts")
    p.add_argument("--force", action="store_true",
                    help="re-run cells already marked complete")
    p.add_argument("--device", default="auto")
    # Full-sweep mode (used by run_stage2c.sbatch)
    p.add_argument("--sweep", choices=["", "all"], default="",
                    help="'all' = iterate all (model, task, mode, layer, concept) cells")
    p.add_argument("--array-task", type=int, default=0,
                    help="this array task's index for striping (0..array_size-1)")
    p.add_argument("--array-size", type=int, default=1,
                    help="total array tasks for striping; cell goes to task hash(cell) % array_size")
    return p


def main():
    args = build_argparser().parse_args()
    cfg = load_config(Path(args.config))
    stage2c_cfg = {}
    s2c_path = Path(args.stage2c_config)
    if s2c_path.exists():
        with open(s2c_path) as f:
            stage2c_cfg = yaml.safe_load(f) or {}
    paths = derive_paths(cfg)
    if args.sweep == "all":
        return main_full_sweep(args, cfg, paths, stage2c_cfg)
    if not args.model or not args.task:
        raise SystemExit("Either --sweep all OR (--model X --task Y) is required.")
    if args.concept:
        return main_one_cell(args, cfg, paths, stage2c_cfg)
    return main_all_cells_for_cohort(args, cfg, paths, stage2c_cfg)


def main_full_sweep(args, cfg, paths, stage2c_cfg) -> int:
    """Walk every (model, task, mode, layer, concept) cell that has both LDA-A
    and CCSVD bases on disk; process only cells whose stable hash modulo
    array_size matches this task's array_idx (striped scheduling)."""
    array_idx = int(args.array_task)
    array_size = max(1, int(args.array_size))
    # Honour single-model / single-task / single-mode filters when given, so a
    # single sweep job can be scoped to one cohort (one model on its own
    # SLURM array, for example). Mode is walked in order off → answer → norm
    # so that headline-mode results land before the residualised modes.
    model_tag = args.model or "all"
    task_tag = args.task or "all"
    mode_tag = args.mode if args.mode != "all" else "all"
    logger = setup_logging(paths["logs_root"], model_tag, task_tag, mode_tag)
    logger.info("=== Stage 2c full sweep: model=%s task=%s mode=%s array_task=%d / %d ===",
                  model_tag, task_tag, mode_tag, array_idx, array_size)
    all_models = [m["key"] for m in cfg["models"]]
    models = [args.model] if args.model else all_models
    all_tasks = ["addition", "multiplication"]
    tasks = [args.task] if args.task else all_tasks
    all_modes = ["off", "answer", "norm"]
    modes = [args.mode] if args.mode and args.mode != "all" else all_modes

    n_done = 0
    n_skipped_other_stripe = 0
    n_skipped_complete = 0
    n_skipped_no_basis = 0
    n_errors = 0

    for model in models:
        model_cfg = next(m for m in cfg["models"] if m["key"] == model)
        layers_for_model = model_cfg["layers"]
        for task in tasks:
            # Activations cache + correctness mask + problem labels — load once
            # per (model, task), reuse across modes/layers/concepts.
            ans_path = paths["data_root"] / "answers" / model / f"{task}_answers.csv"
            if not ans_path.exists():
                logger.warning("Skip %s/%s — missing answers.csv", model, task)
                continue
            ans_df = pd.read_csv(ans_path)
            correct_mask = ans_df["correct"].astype(bool).to_numpy()
            prob_path = paths["data_root"] / "data" / "raw" / f"{task}_problems.csv"
            if not prob_path.exists():
                logger.warning("Skip %s/%s — missing problems.csv", model, task)
                continue
            prob_df = pd.read_csv(prob_path)

            for mode in modes:
                # Stage 2a summaries (both variants) — for eligibility +
                # period-regime / top-2 peak derivation.
                s2a_lda_path = stage2a_summary_path(paths["results_root"], model, task, mode, "lda_a")
                s2a_ccsvd_path = stage2a_summary_path(paths["results_root"], model, task, mode, "ccsvd")
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
                    X_all = None  # lazy-load

                    concepts = discover_concepts_for_cell(
                        paths["results_root"], model, task, mode, layer,
                        s2a_lda_df, s2a_ccsvd_df)
                    for concept in concepts:
                        # Striping by stable cell hash
                        cell_id = f"{model}|{task}|{mode}|{layer:02d}|{concept}"
                        stripe = int.from_bytes(
                            hashlib.sha256(cell_id.encode()).digest()[:4],
                            "big") % array_size
                        if stripe != array_idx:
                            n_skipped_other_stripe += 1
                            continue

                        out_dir = stage2c_cell_dir(paths["results_root"],
                                                    model, task, mode, layer, concept)
                        if cell_complete(out_dir) and not args.force:
                            n_skipped_complete += 1
                            continue
                        if concept not in prob_df.columns:
                            continue
                        # Lazy-load X on first concept for this layer
                        if X_all is None:
                            X_all = np.load(X_path)
                            X = X_all[correct_mask].astype(np.float64)
                        labels_full = prob_df[concept].to_numpy()
                        labels_correct = labels_full[correct_mask]
                        unique_full_values = sorted(prob_df[concept].dropna().unique().tolist())
                        K_natural = len(unique_full_values)
                        value_to_code = {v: i for i, v in enumerate(unique_full_values)}
                        label_codes = np.array(
                            [value_to_code.get(v, -1) for v in labels_correct],
                            dtype=np.int64)
                        keep = label_codes >= 0
                        X_c = X[keep]
                        label_codes_c = label_codes[keep]

                        B_u, union_meta = build_union_basis(
                            paths["results_root"], model, task, mode, layer, concept)
                        if union_meta["k_u"] < 1:
                            n_skipped_no_basis += 1
                            continue
                        mu_layer = X_c.mean(axis=0)
                        Z = (X_c - mu_layer).dot(B_u.astype(np.float64))

                        s2a_lda = read_stage2a_row(paths["results_root"], model, task,
                                                     mode, "lda_a", layer, concept)
                        s2a_ccsvd = read_stage2a_row(paths["results_root"], model, task,
                                                       mode, "ccsvd", layer, concept)
                        p_spec_lda = read_stage2a_periodogram(paths["results_root"],
                                                                model, task, mode, "lda_a",
                                                                layer, concept)
                        p_spec_ccsvd = read_stage2a_periodogram(paths["results_root"],
                                                                  model, task, mode, "ccsvd",
                                                                  layer, concept)

                        logger.info(">>> [stripe %d/%d] %s", array_idx, array_size, cell_id)
                        try:
                            result = analyze_cell(
                                Z, label_codes_c, K_natural,
                                model, task, mode, layer, concept,
                                union_meta, mu_layer_source="X.mean",
                                stage2a_row_lda=s2a_lda, stage2a_row_ccsvd=s2a_ccsvd,
                                P_spec_lda=p_spec_lda, P_spec_ccsvd=p_spec_ccsvd,
                                stage2c_cfg=stage2c_cfg, device=default_device(),
                                logger=logger,
                            )
                        except Exception as e:
                            logger.exception("Cell %s failed: %s", cell_id, e)
                            n_errors += 1
                            continue

                        if result.get("status") != "ok":
                            out_dir.mkdir(parents=True, exist_ok=True)
                            atomic_csv(pd.DataFrame([{
                                "model": model, "task": task, "mode": mode,
                                "layer": int(layer), "concept": concept,
                                "status": result.get("status"),
                                "K_natural": result.get("K_natural", K_natural),
                                "K_present": result.get("K_present", 0),
                                "k_u": result.get("k_u", union_meta["k_u"]),
                                "N_used": result.get("N_used", 0),
                                "verdict_pre_fdr": (V_LOW_K if result.get("status") == V_LOW_K
                                                     else V_INCONCLUSIVE),
                                "verdict_post_fdr": (V_LOW_K if result.get("status") == V_LOW_K
                                                      else V_INCONCLUSIVE),
                                "tier": "LOW",
                            }]), out_dir / "gplvm_results.csv")
                            atomic_json({"computation_status": "complete",
                                          "status": result.get("status"),
                                          "union_meta": union_meta},
                                         out_dir / "metadata.json")
                            n_done += 1
                            continue
                        write_cell_artifacts(out_dir, result, logger)
                        n_done += 1
    logger.info("=== Stripe %d done. processed=%d, other_stripe=%d, "
                  "already_complete=%d, no_basis=%d, errors=%d ===",
                  array_idx, n_done, n_skipped_other_stripe,
                  n_skipped_complete, n_skipped_no_basis, n_errors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
