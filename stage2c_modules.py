#!/usr/bin/env python3
"""
Stage 2c — Independent evidence modules for BSMI-R.

Each module returns a *score* (or richer struct) without gating any other
module. The Tier decision in stage2c_gplvm.py consumes the full vector.

Modules implemented:
    Stage 7:  intrinsic_dimension       (TwoNN + Levina-Bickel + PCA-PR)
    Stage 8:  persistent_homology       (Betti numbers with bootstrap CI)
    Stage 9:  fourier_diagnostics       (period, harmonic energy, two-axis)
    Stage 10: posterior_geometry        (curvature, torsion, helix drift test)
    Stage 11: label_alignment           (rank corr / circular corr / decoding)
    Stage 12: holdout_adequacy          (within-label + leave-value-out)
    Stage 14: permutation_test          (1000 label permutations)

Reference: docs/gplvm.md Stages 7-14.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import svd, lstsq, norm

# Silence the harmless ripser warning about transposed input. Per-label
# centroid clouds with K_present < k_u are normal here (e.g. 10 centroids in
# 18-D union basis), not a transpose mistake.
warnings.filterwarnings(
    "ignore",
    message=r".*input point cloud has more columns than rows.*")


# ============================================================================
# Stage 7 — Intrinsic dimension
# ============================================================================

def twonn_dimension(X: np.ndarray, discard_top_frac: float = 0.1
                     ) -> Tuple[float, Tuple[float, float]]:
    """TwoNN estimator (Facco et al. 2017).

    Returns (d_hat, 95% bootstrap CI). Requires N >= 4.
    """
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    if N < 4:
        return float("nan"), (float("nan"), float("nan"))
    # k=1,2 nearest distances (exclude self)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    dist, _ = nn.kneighbors(X)
    r1 = dist[:, 1]
    r2 = dist[:, 2]
    mu = r2 / np.clip(r1, 1e-12, None)
    mu = mu[mu > 1.0]
    if mu.size < 4:
        return float("nan"), (float("nan"), float("nan"))
    # Discard the top fraction (outliers in mu)
    mu_sorted = np.sort(mu)
    keep = int(np.ceil(mu_sorted.size * (1.0 - discard_top_frac)))
    mu_keep = mu_sorted[:max(keep, 4)]
    # MLE: d = N / sum log(mu)
    log_mu = np.log(mu_keep)
    if log_mu.sum() <= 0:
        return float("nan"), (float("nan"), float("nan"))
    d_hat = mu_keep.size / log_mu.sum()
    # bootstrap CI
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(200):
        idx = rng.integers(0, mu_keep.size, mu_keep.size)
        s = np.log(mu_keep[idx]).sum()
        if s > 0:
            boots.append(mu_keep.size / s)
    if boots:
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    else:
        lo = hi = float("nan")
    return float(d_hat), (lo, hi)


def levina_bickel_dimension(X: np.ndarray, k: int = 10
                              ) -> Tuple[float, Tuple[float, float]]:
    """Levina-Bickel MLE intrinsic dim using k-NN distances."""
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    k = max(2, min(k, N - 1))
    if N < k + 1:
        return float("nan"), (float("nan"), float("nan"))
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, _ = nn.kneighbors(X)
    # column 0 is self
    Tk = dist[:, k]
    # per-point estimator
    d_pts = []
    for i in range(N):
        ri = dist[i, 1:k + 1]
        if (ri > 0).all() and Tk[i] > 0:
            d_pts.append(1.0 / np.mean(np.log(Tk[i] / np.clip(ri, 1e-12, None))))
    d_pts = np.asarray([d for d in d_pts if np.isfinite(d)])
    if d_pts.size < 3:
        return float("nan"), (float("nan"), float("nan"))
    d_hat = float(np.mean(d_pts))
    lo = float(np.percentile(d_pts, 2.5))
    hi = float(np.percentile(d_pts, 97.5))
    return d_hat, (lo, hi)


def pca_participation_ratio(X: np.ndarray) -> float:
    """Effective dimension via participation ratio of PCA eigenvalues.
       PR = (sum lambda)^2 / sum(lambda^2)."""
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 2:
        return float("nan")
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2
    num = lam.sum() ** 2
    den = (lam ** 2).sum()
    return float(num / max(den, 1e-12))


def intrinsic_dimension(X: np.ndarray) -> Dict:
    """Run all dim estimators; return their values + a robust composite.

    Composite uses median of (TwoNN, Levina-Bickel). PCA-PR is reported but
    treated as a sanity check (it inflates for noisy manifolds).
    """
    d_tn, ci_tn = twonn_dimension(X)
    d_lb, ci_lb = levina_bickel_dimension(X)
    d_pr = pca_participation_ratio(X)
    # robust composite
    vals = [v for v in (d_tn, d_lb) if np.isfinite(v)]
    if not vals:
        d_hat = float("nan")
        ci_lo = ci_hi = float("nan")
        agree = False
    else:
        d_hat = float(np.median(vals))
        # CI: union of TwoNN and Levina-Bickel
        lo = min([c for c in (ci_tn[0], ci_lb[0]) if np.isfinite(c)] or [float("nan")])
        hi = max([c for c in (ci_tn[1], ci_lb[1]) if np.isfinite(c)] or [float("nan")])
        ci_lo, ci_hi = float(lo), float(hi)
        agree = (len(vals) == 2 and abs(d_tn - d_lb) / max(d_hat, 1e-6) < 0.25)
    return {
        "d_twonn": float(d_tn),
        "d_twonn_ci": [float(ci_tn[0]), float(ci_tn[1])],
        "d_levina_bickel": float(d_lb),
        "d_levina_bickel_ci": [float(ci_lb[0]), float(ci_lb[1])],
        "d_pca_pr": float(d_pr),
        "d_hat": d_hat,
        "d_ci_low": ci_lo,
        "d_ci_high": ci_hi,
        "estimators_agree": bool(agree),
    }


def dim_score_for_shape(d_hat: float, shape_latent_dim: int) -> str:
    """Map (d_hat, shape's latent dim) to support / warning / contradict / uncertain.

    Per docs/gplvm.md Stage 7:
        d_hat ≈ 1: compatible with line, circle, open helix
        d_hat ≈ 2: compatible with torus, ribbon, concentric
    """
    if not np.isfinite(d_hat):
        return "uncertain"
    diff = abs(d_hat - shape_latent_dim)
    if diff <= 0.5:
        return "supportive"
    if diff <= 1.0:
        return "warning"
    return "contradictory"


# ============================================================================
# Stage 8 — Persistent homology
# ============================================================================

def persistent_homology(X: np.ndarray, max_dim: int = 2,
                          max_edge_factor: float = 2.0
                          ) -> Dict:
    """Persistent homology of a point cloud. Falls back to a tiny placeholder
    if no PH library is installed.

    Returns:
        betti_obs: (b0, b1, b2) Betti numbers (largest persistent classes)
        betti_std: bootstrap std of betti
        diagram:   list of (dim, birth, death) tuples
        status:    "ok" | "no_ph_lib" | "too_small"
    """
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    if N < 4:
        return {"betti_obs": (1, 0, 0), "betti_std": (0.0, 0.0, 0.0),
                "diagram": [], "status": "too_small"}

    # Try ripser (preferred — fast).
    diag = None
    try:
        from ripser import ripser
        # cap max edge length to avoid huge filtrations
        from scipy.spatial.distance import pdist
        d = pdist(X)
        max_edge = float(max_edge_factor * np.median(d))
        result = ripser(X, maxdim=max_dim, thresh=max_edge)
        diag = result["dgms"]
    except Exception:
        pass
    if diag is None:
        try:
            import gudhi
            from scipy.spatial.distance import pdist
            d = pdist(X)
            max_edge = float(max_edge_factor * np.median(d))
            rips = gudhi.RipsComplex(points=X.tolist(), max_edge_length=max_edge)
            st = rips.create_simplex_tree(max_dimension=max_dim + 1)
            st.compute_persistence()
            diag_raw = st.persistence()
            diag = [[] for _ in range(max_dim + 1)]
            for dim_pt, (birth, death) in diag_raw:
                if dim_pt <= max_dim:
                    diag[dim_pt].append([birth, death])
            diag = [np.asarray(d) if d else np.zeros((0, 2)) for d in diag]
        except Exception:
            return {"betti_obs": (1, 0, 0), "betti_std": (1.0, 1.0, 1.0),
                    "diagram": [], "status": "no_ph_lib"}

    # Count persistent classes per dimension: keep features with persistence
    # > median + 2 sigma of bar lengths in that dimension.
    betti = []
    for dim in range(max_dim + 1):
        bars = diag[dim]
        if len(bars) == 0:
            betti.append(0)
            continue
        # filter infinite-death bars to a large finite for the median computation
        deaths = bars[:, 1].copy()
        finite = np.isfinite(deaths)
        if finite.sum() > 0:
            max_finite_death = deaths[finite].max()
        else:
            max_finite_death = 1.0
        deaths_filled = np.where(finite, deaths, max_finite_death + 1e-3)
        persistence = deaths_filled - bars[:, 0]
        if dim == 0:
            # in dim 0 the one infinite class is the connected component; count it
            n_inf = (~finite).sum()
            # significant finite classes (these correspond to extra components)
            thresh = persistence[finite].max() * 0.5 if finite.sum() else 0
            n_sig = (persistence[finite] > thresh).sum() if finite.sum() else 0
            betti.append(int(n_inf + n_sig if n_inf > 0 else max(0, n_sig)))
            continue
        # for dim >= 1: count classes with persistence > threshold
        if persistence.size == 0:
            betti.append(0)
            continue
        thresh = max(0.05 * persistence.max(), np.median(persistence))
        n = int((persistence > thresh).sum())
        betti.append(n)

    rng = np.random.default_rng(0)
    boots = []
    n_boot = 50
    for _ in range(n_boot):
        if N < 16:
            break
        idx = rng.choice(N, size=int(0.8 * N), replace=False)
        sub = persistent_homology_fast(X[idx], max_dim=max_dim,
                                         max_edge_factor=max_edge_factor)
        if sub is not None:
            boots.append(sub)
    if boots:
        boots = np.asarray(boots, dtype=float)
        std = tuple(float(boots[:, i].std(ddof=1)) for i in range(boots.shape[1]))
    else:
        std = (1.0, 1.0, 1.0)
    while len(betti) < 3:
        betti.append(0)
    return {
        "betti_obs": tuple(int(b) for b in betti[:3]),
        "betti_std": std,
        "diagram": [d.tolist() if hasattr(d, "tolist") else d for d in diag],
        "status": "ok",
    }


def persistent_homology_fast(X: np.ndarray, max_dim: int = 2,
                              max_edge_factor: float = 2.0
                              ) -> Optional[Tuple[int, int, int]]:
    """Returns just the Betti tuple; used inside bootstrap to keep cost down."""
    try:
        from ripser import ripser
        from scipy.spatial.distance import pdist
        d = pdist(X)
        max_edge = float(max_edge_factor * np.median(d))
        result = ripser(X, maxdim=max_dim, thresh=max_edge)
        diag = result["dgms"]
    except Exception:
        return None
    betti = []
    for dim in range(max_dim + 1):
        bars = diag[dim]
        if len(bars) == 0:
            betti.append(0)
            continue
        deaths = bars[:, 1]
        finite = np.isfinite(deaths)
        persistence = np.where(finite, deaths, deaths.max() if finite.any() else 1.0
                                ) - bars[:, 0]
        if dim == 0:
            betti.append(int((~finite).sum()))
            continue
        if persistence.size == 0:
            betti.append(0)
            continue
        thresh = max(0.05 * persistence.max(), np.median(persistence))
        betti.append(int((persistence > thresh).sum()))
    while len(betti) < 3:
        betti.append(0)
    return tuple(int(b) for b in betti[:3])


# ============================================================================
# Stage 9 — Fourier diagnostics
# ============================================================================

def fourier_diagnostics(Z_bar: np.ndarray, v_codes: np.ndarray,
                          K_natural: int) -> Dict:
    """Run a label-ordered Fourier analysis on per-value centroids.

    Returns dominant period, harmonic energy, and a two-axis periodicity flag.
    """
    Z_bar = np.asarray(Z_bar, dtype=np.float64)
    K, r = Z_bar.shape
    if K < 4:
        return {"P_hat": float("nan"), "P_hat_2": float("nan"),
                "power_top1": 0.0, "power_top2": 0.0,
                "power_median": 0.0,
                "two_axis": False, "status": "too_small"}
    # FFT along the label axis
    F = np.fft.rfft(Z_bar, axis=0)              # (K//2 + 1, r)
    power = np.abs(F) ** 2
    power_mean = power.mean(axis=1)             # avg across cols
    # exclude DC
    if power_mean.size < 3:
        return {"P_hat": float("nan"), "P_hat_2": float("nan"),
                "power_top1": 0.0, "power_top2": 0.0,
                "power_median": 0.0,
                "two_axis": False, "status": "too_small"}
    p_nonzero = power_mean[1:]
    k_top1 = int(np.argmax(p_nonzero)) + 1
    P_top1 = float(K / max(k_top1, 1))
    # Top-2: exclude k_top1 and immediate neighbours
    mask = np.ones_like(power_mean, dtype=bool)
    mask[0] = False
    for k in (k_top1 - 1, k_top1, k_top1 + 1):
        if 0 <= k < mask.size:
            mask[k] = False
    p2 = power_mean.copy()
    p2[~mask] = -np.inf
    if (p2 > -np.inf).any():
        k_top2 = int(np.argmax(p2))
        P_top2 = float(K / max(k_top2, 1))
        pw_top2 = float(power_mean[k_top2])
    else:
        P_top2 = float("nan")
        pw_top2 = 0.0
    median_floor = float(np.median(p_nonzero))
    pw_top1 = float(power_mean[k_top1])
    two_axis = (pw_top2 >= 0.5 * pw_top1) and (pw_top2 >= 3.0 * median_floor)
    return {
        "P_hat": P_top1,
        "P_hat_2": P_top2,
        "power_top1": pw_top1,
        "power_top2": pw_top2,
        "power_median": median_floor,
        "two_axis": bool(two_axis),
        "status": "ok",
    }


# ============================================================================
# Stage 10 — Posterior differential geometry
# ============================================================================

def helix_drift_test(W: np.ndarray) -> Dict:
    """For K_3 helix fit Phi = [1, cos, sin, t] -> W in R^{4 x r}, test that
    the component of W[3, :] (the linear-drift row) outside span(W[1, :], W[2, :])
    is non-trivial.

    Returns drift_norm, parallel_norm, perp_norm and a fraction perp_frac.
    The drift test passes when perp_frac > threshold (e.g. 0.2).
    """
    if W.shape[0] < 4:
        return {"perp_norm": 0.0, "parallel_norm": 0.0,
                "drift_norm": 0.0, "perp_frac": 0.0, "status": "no_drift_row"}
    span = np.vstack([W[1, :], W[2, :]])         # (2, r)
    drift = W[3, :]                                # (r,)
    drift_norm = float(np.linalg.norm(drift))
    if drift_norm < 1e-9:
        return {"perp_norm": 0.0, "parallel_norm": 0.0,
                "drift_norm": 0.0, "perp_frac": 0.0, "status": "drift_zero"}
    # Orthogonalise span
    Q, _ = np.linalg.qr(span.T)                    # (r, 2)
    proj = Q @ (Q.T @ drift)
    perp = drift - proj
    perp_norm = float(np.linalg.norm(perp))
    parallel_norm = float(np.linalg.norm(proj))
    perp_frac = perp_norm / max(drift_norm, 1e-9)
    return {
        "perp_norm": perp_norm,
        "parallel_norm": parallel_norm,
        "drift_norm": drift_norm,
        "perp_frac": perp_frac,
        "status": "ok",
    }


def curve_curvature_torsion(latent: np.ndarray) -> Dict:
    """For a 1D parameterised curve given by latent (K, d_embed), estimate
    discrete curvature (kappa) and torsion (tau) via finite differences.

    latent rows are assumed to be ordered by label index.
    """
    P = np.asarray(latent, dtype=np.float64)
    K = P.shape[0]
    if K < 5:
        return {"mean_kappa": float("nan"), "mean_abs_torsion": float("nan"),
                "status": "too_small"}
    # finite differences (centred)
    v1 = np.gradient(P, axis=0)
    v2 = np.gradient(v1, axis=0)
    if P.shape[1] < 3:
        # No torsion definable in <3D; report kappa only.
        speed = np.linalg.norm(v1, axis=1)
        # curvature: |v1 x v2| / |v1|^3 (2D cross is scalar)
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0] if P.shape[1] == 2 else np.zeros(K)
        kappa = np.abs(cross) / np.clip(speed ** 3, 1e-12, None)
        return {"mean_kappa": float(np.nanmean(kappa)),
                "mean_abs_torsion": 0.0, "status": "ok_2d"}
    v3 = np.gradient(v2, axis=0)
    if P.shape[1] == 3:
        cross = np.cross(v1, v2)
    else:
        # higher-dim latent: take the projection onto the first 3 coords for
        # the curvature/torsion estimate. This matches the parameterised
        # curves K_3, K_4 which are explicitly embedded into 3+ dims via
        # (cos, sin, [t, ...]) — the first three columns carry the geometry.
        cross = np.cross(v1[:, :3], v2[:, :3])
    speed = np.linalg.norm(v1, axis=1)
    kappa = np.linalg.norm(cross, axis=1) / np.clip(speed ** 3, 1e-12, None)
    # torsion: ((v1 x v2) . v3) / |v1 x v2|^2 — also computed in the first 3 dims.
    v3_3 = v3[:, :3] if P.shape[1] > 3 else v3
    denom = np.clip(np.sum(cross * cross, axis=1), 1e-12, None)
    tau = np.sum(cross * v3_3, axis=1) / denom
    return {
        "mean_kappa": float(np.nanmean(kappa)),
        "mean_abs_torsion": float(np.nanmean(np.abs(tau))),
        "status": "ok_3d",
    }


def posterior_geometry(shape_name: str, latent: np.ndarray, W: np.ndarray,
                        eps_kappa: float = 1e-3, eps_tau: float = 1e-3) -> Dict:
    """Compute shape-specific geometry signature and a posterior pass/fail.

    Returns the underlying numbers + a geom_status in {supportive, neutral,
    contradictory, uncertain}.
    """
    out = {"shape": shape_name}
    kt = curve_curvature_torsion(latent)
    out.update(kt)
    # Helix drift test
    drift = helix_drift_test(W) if W.shape[0] >= 4 else {"perp_frac": 0.0,
                                                            "status": "no_drift_row"}
    out["drift"] = drift

    status = "neutral"
    if shape_name == "K1_Line":
        # straight line: kappa near 0
        status = "supportive" if kt.get("mean_kappa", 1.0) <= eps_kappa else "contradictory"
    elif shape_name == "K2_Circle":
        # nonzero kappa, near-zero torsion
        ok_k = kt.get("mean_kappa", 0.0) > eps_kappa
        ok_t = kt.get("mean_abs_torsion", 1.0) <= 10 * eps_tau
        status = "supportive" if (ok_k and ok_t) else "contradictory"
    elif shape_name == "K3_OpenHelix":
        # nonzero kappa AND nonzero perp drift fraction
        ok_k = kt.get("mean_kappa", 0.0) > eps_kappa
        ok_d = drift.get("perp_frac", 0.0) > 0.2
        status = "supportive" if (ok_k and ok_d) else "uncertain"
    elif shape_name == "K4_Torus":
        # two periodic directions — checked by W rank diagnostics
        rk = 0
        if W is not None and W.shape[0] >= 5:
            s = np.linalg.svd(W[1:5], compute_uv=False)
            if s.size:
                rk = int((s > 0.1 * s[0]).sum())
        out["effective_rank_W14"] = rk
        status = "supportive" if rk >= 4 else "uncertain"
    elif shape_name in {"K5_Concentric", "K6_Ribbon", "K0_Generic"}:
        status = "neutral"
    out["geom_status"] = status
    return out


def geom_score(geom_status: str) -> float:
    return {"supportive": 1.0, "neutral": 0.0,
            "uncertain": 0.0, "contradictory": -1.0}.get(geom_status, 0.0)


# ============================================================================
# Stage 11 — Label alignment
# ============================================================================

def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho without scipy dependency."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size != b.size or a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if denom <= 0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def circular_correlation(phi: np.ndarray, target_phi: np.ndarray) -> float:
    """Jammalamadaka-Sarma circular correlation. Both inputs in radians."""
    phi = np.asarray(phi, dtype=np.float64)
    tp = np.asarray(target_phi, dtype=np.float64)
    if phi.size != tp.size or phi.size < 3:
        return float("nan")
    mu_p = math.atan2(np.sin(phi).mean(), np.cos(phi).mean())
    mu_t = math.atan2(np.sin(tp).mean(), np.cos(tp).mean())
    sp = np.sin(phi - mu_p)
    st = np.sin(tp - mu_t)
    num = (sp * st).sum()
    den = math.sqrt((sp ** 2).sum() * (st ** 2).sum())
    if den <= 0:
        return float("nan")
    return float(num / den)


def label_alignment(shape, theta: Dict[str, float],
                     latent: np.ndarray, v_codes: np.ndarray,
                     K_natural: int) -> Dict:
    """Score how well the recovered latent aligns with the true label index.

    For ordered shapes: Spearman correlation between latent[:, 0] and v_codes.
    For periodic shapes: circular correlation between the recovered angle and
    the expected angle 2 pi v / P.
    """
    out = {"shape": shape.name}
    if shape.is_periodic:
        P = float(theta.get("P", K_natural))
        # latent's first two cols are cos/sin -> angle
        if latent.shape[1] >= 2:
            phi = np.arctan2(latent[:, 1], latent[:, 0])
            target = 2.0 * np.pi * v_codes / max(P, 1e-6)
            rho = circular_correlation(phi, target)
        else:
            rho = float("nan")
        out["rho_circ"] = rho
        out["alignment_score"] = float(abs(rho)) if np.isfinite(rho) else 0.0
        if shape.is_two_periodic and latent.shape[1] >= 4:
            P2 = float(theta.get("P2", 2.0 * K_natural))
            phi2 = np.arctan2(latent[:, 3], latent[:, 2])
            target2 = 2.0 * np.pi * v_codes / max(P2, 1e-6)
            rho2 = circular_correlation(phi2, target2)
            out["rho_circ_2"] = rho2
            # combined score: minimum of the two axes
            out["alignment_score"] = float(min(abs(rho), abs(rho2))
                                              if (np.isfinite(rho) and np.isfinite(rho2))
                                              else 0.0)
    else:
        rho = rank_correlation(latent[:, 0], v_codes.astype(np.float64))
        out["rho_spearman"] = rho
        out["alignment_score"] = float(abs(rho)) if np.isfinite(rho) else 0.0
    return out


def alignment_pass(score: float, threshold: float = 0.5) -> bool:
    return bool(np.isfinite(score) and score >= threshold)


# ============================================================================
# Stage 12 — Holdout adequacy
# ============================================================================

def within_label_holdout(Z: np.ndarray, label_codes: np.ndarray,
                          K_present: int, fit_fn: Callable,
                          n_folds: int = 5, rng_seed: int = 0
                          ) -> Dict:
    """K-fold CV within each label. Every point is held out exactly once."""
    rng = np.random.default_rng(rng_seed)
    fold_assign = np.full(Z.shape[0], -1, dtype=np.int64)
    for v in range(K_present):
        idx = np.where(label_codes == v)[0]
        if idx.size == 0:
            continue
        rng.shuffle(idx)
        fold_assign[idx] = np.arange(idx.size) % n_folds
    fold_mses = []
    n_test_total = 0
    for fold in range(n_folds):
        test = fold_assign == fold
        train = ~test
        if not test.any() or not train.any():
            continue
        pred_fn = fit_fn(Z[train], label_codes[train], K_present)
        Z_pred = pred_fn(label_codes[test])
        fold_mses.append(float(np.mean((Z[test] - Z_pred) ** 2)))
        n_test_total += int(test.sum())
    if not fold_mses:
        return {"mse": float("nan"), "n_test": 0,
                "fold_mses": [], "n_folds": 0}
    return {"mse": float(np.mean(fold_mses)),
            "fold_mses": fold_mses,
            "fold_mse_std": float(np.std(fold_mses, ddof=1) if len(fold_mses) > 1 else 0.0),
            "n_test": int(n_test_total),
            "n_folds": int(len(fold_mses))}


def leave_value_out_holdout(Z: np.ndarray, label_codes: np.ndarray,
                              K_present: int, fit_fn: Callable,
                              rng_seed: int = 0) -> Dict:
    """Full leave-one-value-out CV: every label held out exactly once."""
    if K_present < 4:
        return {"mse": float("nan"), "n_folds": 0, "status": "K_too_small"}
    fold_mses = []
    for held in range(K_present):
        train_mask = label_codes != held
        test_mask = ~train_mask
        if not train_mask.any() or not test_mask.any():
            continue
        pred_fn = fit_fn(Z[train_mask], label_codes[train_mask], K_present)
        Z_pred = pred_fn(label_codes[test_mask])
        fold_mses.append(float(np.mean((Z[test_mask] - Z_pred) ** 2)))
    if not fold_mses:
        return {"mse": float("nan"), "n_folds": 0, "status": "no_folds"}
    return {"mse": float(np.mean(fold_mses)),
            "fold_mses": fold_mses,
            "fold_mse_std": float(np.std(fold_mses, ddof=1) if len(fold_mses) > 1 else 0.0),
            "n_folds": int(len(fold_mses)),
            "status": "ok"}


def holdout_pass(mse_winner: float, mse_others: List[float],
                  epsilon_factor: float = 1.10) -> bool:
    """Predictive-adequacy rule (relaxed per docs/gplvm.md Stage 12):
        winner_MSE <= min(others) * epsilon_factor
    NOT the strict 'beat runner-up by 1 SE' rule.
    """
    if not np.isfinite(mse_winner):
        return False
    finite = [m for m in mse_others if np.isfinite(m)]
    if not finite:
        return True
    return mse_winner <= epsilon_factor * min(finite)


# ============================================================================
# Stage 14 — 1000-permutation test
# ============================================================================

def permutation_pvalue(observed: float, null_samples: np.ndarray) -> float:
    """Right-tailed empirical p-value, with the +1 / B+1 correction."""
    ns = np.asarray(null_samples, dtype=np.float64)
    ns = ns[np.isfinite(ns)]
    B = int(ns.size)
    if B == 0:
        return float("nan")
    return float((1 + int((ns >= observed).sum())) / (B + 1))


def permutation_test(observed: float,
                      null_stat_fn: Callable[[np.ndarray], float],
                      v_codes: np.ndarray,
                      B: int = 1000,
                      rng_seed: int = 0,
                      early_stop_low: int = 50,
                      early_stop_high: int = 50) -> Dict:
    """Sequential label-permutation test.

    null_stat_fn(v_codes_permuted) -> statistic on the same fitting pipeline.
    Computes up to B permutations with optional early stopping on obvious
    successes/failures.
    """
    rng = np.random.default_rng(rng_seed)
    null = []
    ge = 0
    for b in range(B):
        perm = rng.permutation(v_codes)
        T_b = float(null_stat_fn(perm))
        null.append(T_b)
        if T_b >= observed:
            ge += 1
        # Optional sequential rules — only if requested counts are positive
        if early_stop_low > 0 and (b + 1) >= early_stop_low and ge >= early_stop_low:
            break
        if early_stop_high > 0 and (b + 1) >= early_stop_high and ge == 0:
            # if we have plenty of permutations and zero exceedances, keep going
            # to nail down a small p — but we record that we could stop here.
            pass
    null = np.asarray(null, dtype=np.float64)
    p = (1 + ge) / (len(null) + 1)
    return {
        "p_value": float(p),
        "n_permutations_run": int(len(null)),
        "n_exceedances": int(ge),
        "null_samples": null.tolist(),
    }
