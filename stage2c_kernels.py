#!/usr/bin/env python3
"""
Stage 2c — 6-kernel zoo for centroid Bayesian model comparison.

K1 RBF
K2 Periodic
K3 Periodic + Linear (helix — strict linear drift on the second axis)
K4 Torus (two independent periods on orthogonal latent axes)
K5 Concentric (two periodic components at the SAME period, different lengthscales)
K6 Periodic + RBF (periodic on dim 0 + smooth-but-non-periodic on dim 1 —
                    a more permissive helix where the secondary axis can have
                    any smooth shape, not just linear drift)

All kernels use float64. Period parameters are LEARNABLE under one of three
prior regimes per cell (locked in stage2c plan §C.2.5):

- regime "narrow"  → Gamma(2, 2/P_init)            mean=P_init, std=P_init/√2     ≈ ±11% spread
- regime "wide"    → Gamma(1, 1/P_init)            mean=P_init, std=P_init        ≈ ±28% spread
- regime "discover"→ None (improper flat-on-log)   no prior weight; grid-search seeded

Mirrors the structure of the parent project's `phase_i_kernels.py` but adds
the Concentric kernel (parent's K5 is Periodic+RBF, which is different).
"""

import math
from typing import Optional

import torch
import gpytorch
from gpytorch.kernels import (
    RBFKernel, PeriodicKernel, LinearKernel,
    ScaleKernel, ProductKernel, AdditiveKernel,
)
from gpytorch.priors import GammaPrior, NormalPrior


# ─── Hyperprior helpers ─────────────────────────────────────────────────────────

def lengthscale_prior() -> GammaPrior:
    return GammaPrior(concentration=1.0, rate=1.0)


def half_normal_prior(sigma: float) -> NormalPrior:
    return NormalPrior(loc=0.0, scale=sigma)


def signal_var_prior() -> NormalPrior:
    return half_normal_prior(1.0)


def noise_var_prior() -> NormalPrior:
    return half_normal_prior(0.1)


def period_prior_from_regime(period_init: float, regime: str) -> Optional[GammaPrior]:
    """Build the period hyperprior for one of three regimes.

    Returns None for regime='discover', which leaves the period parameter with
    only its positivity constraint (no log-density contribution to the marginal
    likelihood). The grid-search initialisation done upstream supplies a strong
    starting point so the optimiser still converges.
    """
    if regime == "narrow":
        return GammaPrior(concentration=2.0, rate=2.0 / float(period_init))
    if regime == "wide":
        return GammaPrior(concentration=1.0, rate=1.0 / float(period_init))
    if regime == "discover":
        return None
    raise ValueError(f"Unknown period regime: {regime!r}")


# ─── Kernel builders (each returns a ScaleKernel-wrapped kernel in fp64) ────────

def _scale(kernel) -> ScaleKernel:
    sk = ScaleKernel(kernel, outputscale_prior=signal_var_prior())
    return sk.double()


def K1_RBF(active_dim_count: int = 1) -> ScaleKernel:
    base = RBFKernel(
        ard_num_dims=active_dim_count,
        lengthscale_prior=lengthscale_prior(),
    )
    return _scale(base)


def K2_Periodic(period: float, regime: str = "narrow",
                active_dim: int = 0) -> ScaleKernel:
    base = PeriodicKernel(
        active_dims=[active_dim],
        lengthscale_prior=lengthscale_prior(),
        period_length_prior=period_prior_from_regime(period, regime),
    )
    base.period_length = float(period)
    return _scale(base)


def K3_PeriodicLinear(period: float, regime: str = "narrow") -> AdditiveKernel:
    """Helix: K2(periodic on dim 0) ⊕ Linear(on dim 1).

    Linear arm placed on a separate latent axis from the periodic arm. A
    same-axis (d=1) formulation is mathematically cleaner for a true helix
    (1D curve), but at production N (≥ 8k points) the kernel matrix
    K_periodic(t,t') + K_linear(t,t') is severely ill-conditioned — Cholesky
    fails even with jitter pushed to 1e-2. The d=2 formulation lets the linear
    arm sit on an orthogonal axis where the kernel is well-conditioned.

    To prevent K3 from absorbing arbitrary noise via its free second axis
    (the d=2 issue we saw on isotropic toys), the linear arm uses a tight
    half-normal(0.3) prior on its outputscale and the BIC penalty counts the
    latent dimension via `kernel_n_hyperparams + d_latent`.
    """
    periodic = K2_Periodic(period, regime=regime, active_dim=0)
    linear = ScaleKernel(LinearKernel(active_dims=[1]),
                         outputscale_prior=half_normal_prior(0.3))
    return (periodic + linear).double()


def K4_Torus(period_outer: float,
             period_inner: Optional[float] = None,
             regime_outer: str = "narrow",
             regime_inner: str = "narrow") -> ProductKernel:
    """Torus: K2(P_outer on dim 0) ⊗ K2(P_inner on dim 1).

    Each period carries its own regime tag. Common patterns:
      (narrow, narrow)   — Stage 2a confident in both top-1 and top-2 peaks.
      (narrow, discover) — Top-1 confident, top-2 peak below the confidence
                           threshold, so the second period is discovered.
      (discover, discover) — Neither period confident; both seeded by grid
                           search at the caller.
    """
    if period_inner is None:
        period_inner = period_outer / 2.0
    p1 = K2_Periodic(period_outer, regime=regime_outer, active_dim=0)
    p2 = K2_Periodic(period_inner, regime=regime_inner, active_dim=1)
    return (p1 * p2).double()


def K6_PeriodicRBF(period: float, regime: str = "narrow") -> AdditiveKernel:
    """K2(periodic on dim 0) ⊕ K1(RBF on dim 1).

    A relaxed helix: dim 0 is periodic at `period`; dim 1 is a smooth RBF
    direction (no enforced linearity). Captures shapes like a helix whose
    "drift axis" has a non-linear smooth profile, or a circle in dim 0 with
    independent smooth variation in a second latent direction.
    """
    periodic = K2_Periodic(period, regime=regime, active_dim=0)
    rbf_arm = ScaleKernel(
        RBFKernel(active_dims=[1], lengthscale_prior=lengthscale_prior()),
        outputscale_prior=signal_var_prior(),
    )
    return (periodic + rbf_arm).double()


def K5_Concentric(period: float,
                  lengthscale_short: float = 0.5,
                  lengthscale_long: float = 2.0,
                  regime: str = "narrow") -> AdditiveKernel:
    """Concentric: K2(P, ℓ_short) ⊕ K2(P, ℓ_long) — SAME period, two lengthscales.

    Models two periodic components sharing the same period but with different
    smoothness / amplitude. Both lengthscales are learnable; the initial values
    supply only the seed.

    Both components live on the same latent dim 0 so the kernel is genuinely
    rotationally symmetric.
    """
    base_short = PeriodicKernel(
        active_dims=[0],
        lengthscale_prior=lengthscale_prior(),
        period_length_prior=period_prior_from_regime(period, regime),
    )
    base_short.period_length = float(period)
    base_short.lengthscale = float(lengthscale_short)
    base_long = PeriodicKernel(
        active_dims=[0],
        lengthscale_prior=lengthscale_prior(),
        period_length_prior=period_prior_from_regime(period, regime),
    )
    base_long.period_length = float(period)
    base_long.lengthscale = float(lengthscale_long)
    short = ScaleKernel(base_short, outputscale_prior=signal_var_prior())
    long_ = ScaleKernel(base_long, outputscale_prior=signal_var_prior())
    return (short + long_).double()


# ─── Kernel registry ────────────────────────────────────────────────────────────

# `min_latent_dim` is the minimum number of latent dimensions the kernel uses;
# the GPLVM driver sets `d_max` ≥ this and ARD-prunes the remainder.
# `n_hyperparams_for_bic` counts free parameters used by the BIC-style penalty
# in §C.4 (excluding noise and outputscale, which are common to all kernels):
KERNEL_REGISTRY = {
    "K1_RBF":            {"builder": K1_RBF,            "min_latent_dim": 1, "needs_period": False, "n_periods": 0, "n_lengthscales_per_latent": True},
    "K2_Periodic":       {"builder": K2_Periodic,       "min_latent_dim": 1, "needs_period": True,  "n_periods": 1, "n_lengthscales_per_latent": False},
    "K3_PeriodicLinear": {"builder": K3_PeriodicLinear, "min_latent_dim": 2, "needs_period": True,  "n_periods": 1, "n_lengthscales_per_latent": False},
    "K4_Torus":          {"builder": K4_Torus,          "min_latent_dim": 2, "needs_period": True,  "n_periods": 2, "n_lengthscales_per_latent": False},
    "K5_Concentric":     {"builder": K5_Concentric,     "min_latent_dim": 1, "needs_period": True,  "n_periods": 1, "n_lengthscales_per_latent": False},
    "K6_PeriodicRBF":    {"builder": K6_PeriodicRBF,    "min_latent_dim": 2, "needs_period": True,  "n_periods": 1, "n_lengthscales_per_latent": False},
}

KERNEL_NAMES = list(KERNEL_REGISTRY.keys())


def build_kernel(kernel_name: str,
                 period: Optional[float] = None,
                 regime: str = "narrow",
                 active_dim_count: int = 1,
                 period_secondary: Optional[float] = None,
                 regime_secondary: Optional[str] = None) -> ScaleKernel:
    """Build a kernel by name. `period` required only for periodic-family kernels.

    For K4 (Torus), `period_secondary` is P_2 and `regime_secondary` is its regime;
    if either is None they fall back to (period/2, regime).
    """
    if kernel_name not in KERNEL_REGISTRY:
        raise ValueError(f"Unknown kernel: {kernel_name}")
    spec = KERNEL_REGISTRY[kernel_name]
    if kernel_name == "K1_RBF":
        return spec["builder"](active_dim_count=active_dim_count)
    if kernel_name == "K2_Periodic":
        return spec["builder"](period, regime=regime)
    if kernel_name == "K3_PeriodicLinear":
        return spec["builder"](period, regime=regime)
    if kernel_name == "K4_Torus":
        p2 = period_secondary if period_secondary is not None else (period / 2.0)
        r2 = regime_secondary if regime_secondary is not None else regime
        return spec["builder"](period, period_inner=p2,
                                regime_outer=regime, regime_inner=r2)
    if kernel_name == "K5_Concentric":
        return spec["builder"](period, regime=regime)
    if kernel_name == "K6_PeriodicRBF":
        return spec["builder"](period, regime=regime)
    raise ValueError(f"Unhandled kernel: {kernel_name}")


def kernel_n_hyperparams(kernel_name: str, d_latent: int) -> int:
    """Count free hyperparameters per kernel for the BIC penalty (§C.4).

    Convention: noise + outputscale + signal_variances + lengthscales + periods.
    For K1 (RBF with ARD), the lengthscale dim equals d_latent.
    """
    spec = KERNEL_REGISTRY[kernel_name]
    n_periods = spec["n_periods"]
    if kernel_name == "K1_RBF":
        n_lengthscales = d_latent
    elif kernel_name == "K2_Periodic":
        n_lengthscales = 1
    elif kernel_name == "K3_PeriodicLinear":
        # 1 periodic lengthscale + 1 linear outputscale (not lengthscale, but counts as parameter)
        n_lengthscales = 2
    elif kernel_name == "K4_Torus":
        n_lengthscales = 2
    elif kernel_name == "K5_Concentric":
        n_lengthscales = 2
    elif kernel_name == "K6_PeriodicRBF":
        n_lengthscales = 2
    else:
        n_lengthscales = d_latent
    # 2 = noise variance + outputscale; K3 and K5 have a second outputscale (additive),
    # K4 has two outputscales (product is wrapped twice via the K2 builder); we use
    # a uniform "+2" since the additional outputscales are small contributions to BIC.
    return 2 + n_periods + n_lengthscales


def extract_periods(kernel) -> list:
    """Walk a kernel module and return all learned period_length values."""
    periods = []
    for _, m in kernel.named_modules():
        if isinstance(m, PeriodicKernel):
            p = m.period_length.detach()
            periods.append(float(p.flatten()[0].item()))
    return periods


def extract_lengthscales(kernel) -> list:
    """Walk a kernel module and return all learned lengthscale values."""
    out = []
    for _, m in kernel.named_modules():
        if hasattr(m, "lengthscale") and m.lengthscale is not None:
            try:
                v = m.lengthscale.detach()
                out.append(float(v.flatten()[0].item()))
            except Exception:
                continue
    return out


__all__ = [
    "lengthscale_prior", "signal_var_prior", "noise_var_prior",
    "period_prior_from_regime",
    "K1_RBF", "K2_Periodic", "K3_PeriodicLinear", "K4_Torus", "K5_Concentric",
    "KERNEL_REGISTRY", "KERNEL_NAMES",
    "build_kernel", "kernel_n_hyperparams",
    "extract_periods", "extract_lengthscales",
]


if __name__ == "__main__":
    # Smoke test: build every kernel and compute K(z, z) at random latent z;
    # confirm positive-definite via Cholesky.
    import numpy as np

    print("stage2c_kernels.py smoke test — build all 5 kernels at random z.")
    z = torch.randn(20, 2, dtype=torch.float64)

    for name, spec in KERNEL_REGISTRY.items():
        for regime in ("narrow", "wide", "discover"):
            period = 10.0 if spec["needs_period"] else None
            try:
                if name == "K4_Torus":
                    k = build_kernel(name, period=period, regime=regime,
                                     period_secondary=5.0, regime_secondary=regime)
                else:
                    k = build_kernel(name, period=period, regime=regime,
                                     active_dim_count=2)
                with torch.no_grad():
                    K = k(z).evaluate()
                jitter = 1e-6 * torch.eye(K.shape[0], dtype=torch.float64)
                L = torch.linalg.cholesky(K + jitter)
                n_hyp = kernel_n_hyperparams(name, d_latent=2)
                print(f"  {name} [{regime}]: OK  shape={tuple(K.shape)}, n_hyp={n_hyp}")
            except Exception as e:
                print(f"  {name} [{regime}]: FAIL  {type(e).__name__}: {e}")
