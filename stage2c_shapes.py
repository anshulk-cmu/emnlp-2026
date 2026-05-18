#!/usr/bin/env python3
"""
Stage 2c — Shape priors for BSMI-R (Bayesian Shape Manifold Inference with Refusal).

Each candidate shape K_k is a label-conditioned basis B_k(v; theta) plus an
expected topology signature, expected geometry signature, and a parameter count.

Reference: docs/gplvm.md (BSMI-R), Stages 1-2.

The shapes are evaluated against label-collapsed activations Z_bar (per-value
centroids) under a Bayesian linear-Gaussian model
    Z_bar_v = B_k(v; theta) W + epsilon_v,    epsilon_v ~ N(0, Sigma / n_v)
with conjugate priors on W; the marginal likelihood log p(Z_bar | theta, K_k)
is therefore closed-form and gets integrated over theta by stage2c_gplvm.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ----------------------------------------------------------------------------
# Topology signatures (Betti numbers expected for the named shape)
# ----------------------------------------------------------------------------
# Per docs/gplvm.md Stage 8: PH is NEVER a hard gate. The "neutral" entries
# below are the cases where PH cannot distinguish the shape from a contractible
# manifold (line / open helix / ribbon all have beta = (1, 0, 0)).
#
# Encoding: (b0, b1, b2). None = neutral / not used as a check.
TopologySignature = Tuple[Optional[int], Optional[int], Optional[int]]


# ----------------------------------------------------------------------------
# Shape prior base class
# ----------------------------------------------------------------------------

@dataclass
class ShapeFitResult:
    """Result of fitting a single shape at a single theta value."""
    name: str
    log_marg_lik: float            # log p(Z_bar | theta, K_k) closed-form
    theta: Dict[str, float]        # learned hyperparameters at this fit
    latent: np.ndarray             # (K_present, latent_dim) latent coords
    basis: np.ndarray              # (K_present, n_basis) Phi matrix
    n_params: int                  # number of theta parameters (for BIC penalty)
    n_latent: int                  # latent dim of this shape


class ShapePrior:
    """Base class for a shape hypothesis K_k.

    Each subclass defines:
        - name
        - latent_dim: intrinsic dimension of the manifold parameterisation
        - n_params: dim of theta (excludes W which is integrated out)
        - expected_betti: (b0, b1, b2) signature, None = neutral
        - is_periodic: whether theta contains a period
        - build_basis(theta, v): returns (K_present, n_basis) design matrix
        - propose_thetas(v, Z_bar, P_hat): returns list of theta dicts to evaluate
    """
    name: str = "Shape"
    latent_dim: int = 1
    n_params: int = 1
    expected_betti: TopologySignature = (None, None, None)
    is_periodic: bool = False
    is_two_periodic: bool = False

    # Expected differential-geometry signature.
    # Each entry is a posterior-CI inclusion/exclusion rule. None = no constraint.
    expects_curvature_above_zero: Optional[bool] = None  # True: kappa > 0
    expects_torsion_above_zero: Optional[bool] = None    # True: |tau| > 0

    def n_basis(self) -> int:
        raise NotImplementedError

    def build_basis(self, theta: Dict[str, float],
                    v_codes: np.ndarray, K_natural: int) -> np.ndarray:
        """Returns (K_present, n_basis) design matrix Phi.

        v_codes: dense 0..K_present-1 codes for the K_present present values.
        K_natural: the natural span (e.g. 10 for digits 0-9) used as the
                   default cyclic period if no theta period provided.
        """
        raise NotImplementedError

    def latent_from_basis(self, theta: Dict[str, float],
                          v_codes: np.ndarray, K_natural: int) -> np.ndarray:
        """Returns (K_present, latent_dim) latent coords for label alignment."""
        raise NotImplementedError

    def column_alpha_factors(self, theta: Dict[str, float],
                              K_natural: int) -> np.ndarray:
        """Per-basis-column prior-variance multipliers (Fix 4).

        Default: all 1.0 — every column shares the same prior precision `alpha`.
        Override in shapes that want a *tighter sub-prior* on specific columns
        (e.g. K6_Ribbon shrinks its cross-term columns so it can't mimic K4).
        """
        return np.ones(self.n_basis(), dtype=np.float64)

    def propose_thetas(self, v_codes: np.ndarray, K_natural: int,
                       P_hat: Optional[float] = None,
                       P_hat_2: Optional[float] = None,
                       n_alias: int = 3) -> List[Dict[str, float]]:
        """Propose theta modes for multimodal Laplace integration.

        Default: single trivial mode (non-periodic shapes). Periodic shapes
        override to return {P, 2P, 3P, P/2} alias pockets.
        """
        return [{}]


# ----------------------------------------------------------------------------
# K_0: generic smooth manifold (fallback / catch-all)
# ----------------------------------------------------------------------------

class K0_Generic(ShapePrior):
    """Generic smooth manifold: polynomial basis in normalised label index.

    This is the 'none of the above' baseline. A high evidence here means the
    data is smooth in label-space but not one of the named shapes.
    """
    name = "K0_Generic"
    latent_dim = 1
    n_params = 1               # polynomial degree only
    expected_betti = (None, None, None)
    is_periodic = False

    def __init__(self, degree: int = 3):
        self.degree = int(degree)

    def n_basis(self) -> int:
        return self.degree + 1

    def build_basis(self, theta: Dict[str, float],
                    v_codes: np.ndarray, K_natural: int) -> np.ndarray:
        deg = int(theta.get("degree", self.degree))
        t = v_codes.astype(np.float64) / max(K_natural - 1, 1)
        Phi = np.vstack([t ** k for k in range(deg + 1)]).T
        return Phi

    def latent_from_basis(self, theta, v_codes, K_natural):
        t = v_codes.astype(np.float64) / max(K_natural - 1, 1)
        return t[:, None]

    def propose_thetas(self, v_codes, K_natural, P_hat=None,
                       P_hat_2=None, n_alias=3):
        return [{"degree": self.degree}]


# ----------------------------------------------------------------------------
# K_1: line / smooth ordered curve
# ----------------------------------------------------------------------------

class K1_Line(ShapePrior):
    """Linear basis: Phi(v) = [1, t]."""
    name = "K1_Line"
    latent_dim = 1
    n_params = 0
    # β₀ is set to None for ALL shapes because the sampled per-label centroid
    # cloud always has β₀ = K_present (each centroid is its own component at
    # small scale) — that is not the manifold's β₀ = 1. The informative
    # signature lives in β₁ and β₂.
    expected_betti = (None, 0, 0)
    is_periodic = False
    expects_curvature_above_zero = False

    def n_basis(self) -> int:
        return 2

    def build_basis(self, theta, v_codes, K_natural):
        t = v_codes.astype(np.float64) / max(K_natural - 1, 1)
        return np.vstack([np.ones_like(t), t]).T

    def latent_from_basis(self, theta, v_codes, K_natural):
        t = v_codes.astype(np.float64) / max(K_natural - 1, 1)
        return t[:, None]


# ----------------------------------------------------------------------------
# K_2: circle / closed cyclic curve
# ----------------------------------------------------------------------------

class K2_Circle(ShapePrior):
    """Circular basis: Phi(v) = [1, cos(2 pi v / P), sin(2 pi v / P)]."""
    name = "K2_Circle"
    latent_dim = 2
    n_params = 1                       # period
    expected_betti = (None, 1, 0)      # β₀ universally None (see K1_Line note)
    is_periodic = True
    expects_curvature_above_zero = True
    expects_torsion_above_zero = False

    def n_basis(self) -> int:
        return 3

    def build_basis(self, theta, v_codes, K_natural):
        P = float(theta.get("P", K_natural))
        omega = 2.0 * np.pi / max(P, 1e-6)
        v = v_codes.astype(np.float64)
        return np.vstack([np.ones_like(v), np.cos(omega * v), np.sin(omega * v)]).T

    def latent_from_basis(self, theta, v_codes, K_natural):
        P = float(theta.get("P", K_natural))
        omega = 2.0 * np.pi / max(P, 1e-6)
        v = v_codes.astype(np.float64)
        return np.vstack([np.cos(omega * v), np.sin(omega * v)]).T

    def propose_thetas(self, v_codes, K_natural, P_hat=None,
                       P_hat_2=None, n_alias=3):
        P_base = float(P_hat) if P_hat else float(K_natural)
        # Alias pockets (Stage 5 of docs/gplvm.md): P, 2P, 3P, P/2.
        modes = [P_base, 2.0 * P_base, 3.0 * P_base, 0.5 * P_base]
        return [{"P": float(P)} for P in modes[:n_alias + 1]
                if 1.5 <= P <= max(50.0, 2 * K_natural)]


# ----------------------------------------------------------------------------
# K_3: open helix (cyclic + linear drift)
# ----------------------------------------------------------------------------

class K3_OpenHelix(ShapePrior):
    """Helix basis: Phi(v) = [1, cos(2 pi v / P), sin(2 pi v / P), t].

    Distinguishable from K_2 only by the linear drift coefficient. The
    posterior helix test (docs/gplvm.md Stage 10) requires the drift
    component perpendicular to span(cos, sin) to exclude zero.
    """
    name = "K3_OpenHelix"
    latent_dim = 1                 # still a 1D curve in latent space
    n_params = 1                   # period (drift slope absorbed into W)
    # Open helix is contractible — PH cannot distinguish it from a line or
    # ribbon. Mark every dim None so PH never contradicts. The helix vs circle
    # distinction is made by the differential-geometry drift test in Stage 10.
    expected_betti = (None, None, None)
    is_periodic = True
    expects_curvature_above_zero = True
    expects_torsion_above_zero = True

    def n_basis(self) -> int:
        return 4

    def build_basis(self, theta, v_codes, K_natural):
        P = float(theta.get("P", K_natural))
        omega = 2.0 * np.pi / max(P, 1e-6)
        v = v_codes.astype(np.float64)
        t = v / max(K_natural - 1, 1)
        return np.vstack([np.ones_like(v),
                          np.cos(omega * v),
                          np.sin(omega * v),
                          t]).T

    def latent_from_basis(self, theta, v_codes, K_natural):
        P = float(theta.get("P", K_natural))
        omega = 2.0 * np.pi / max(P, 1e-6)
        v = v_codes.astype(np.float64)
        t = v / max(K_natural - 1, 1)
        return np.vstack([np.cos(omega * v), np.sin(omega * v), t]).T

    def propose_thetas(self, v_codes, K_natural, P_hat=None,
                       P_hat_2=None, n_alias=3):
        P_base = float(P_hat) if P_hat else float(K_natural)
        modes = [P_base, 2.0 * P_base, 3.0 * P_base, 0.5 * P_base]
        return [{"P": float(P)} for P in modes[:n_alias + 1]
                if 1.5 <= P <= max(50.0, 2 * K_natural)]


# ----------------------------------------------------------------------------
# K_4: torus / two-cycle manifold
# ----------------------------------------------------------------------------

class K4_Torus(ShapePrior):
    """Torus basis: Phi(v) = [1, cos phi, sin phi, cos psi, sin psi]
    with two independent cyclic coordinates phi = 2 pi v / P_1,
    psi = 2 pi v / P_2.

    For arithmetic concepts with secondary periodicity (e.g. addition has
    units period 10 and full-period 100), the two axes carry independent
    structure.
    """
    name = "K4_Torus"
    latent_dim = 2
    n_params = 2                       # two periods
    expected_betti = (None, 2, 1)      # β₀ universally None (see K1_Line note)
    is_periodic = True
    is_two_periodic = True
    expects_curvature_above_zero = True

    def n_basis(self) -> int:
        return 5

    def build_basis(self, theta, v_codes, K_natural):
        P1 = float(theta.get("P", K_natural))
        P2 = float(theta.get("P2", max(2.0 * K_natural, 4.0)))
        v = v_codes.astype(np.float64)
        omega1 = 2.0 * np.pi / max(P1, 1e-6)
        omega2 = 2.0 * np.pi / max(P2, 1e-6)
        return np.vstack([np.ones_like(v),
                          np.cos(omega1 * v), np.sin(omega1 * v),
                          np.cos(omega2 * v), np.sin(omega2 * v)]).T

    def latent_from_basis(self, theta, v_codes, K_natural):
        P1 = float(theta.get("P", K_natural))
        P2 = float(theta.get("P2", max(2.0 * K_natural, 4.0)))
        v = v_codes.astype(np.float64)
        omega1 = 2.0 * np.pi / max(P1, 1e-6)
        omega2 = 2.0 * np.pi / max(P2, 1e-6)
        return np.vstack([np.cos(omega1 * v), np.sin(omega1 * v),
                          np.cos(omega2 * v), np.sin(omega2 * v)]).T

    def propose_thetas(self, v_codes, K_natural, P_hat=None,
                       P_hat_2=None, n_alias=3):
        P1 = float(P_hat) if P_hat else float(K_natural)
        # P_hat_2 is the Stage-2a top-2 period; if missing, use 2*P1.
        P2 = float(P_hat_2) if P_hat_2 else 2.0 * P1
        thetas = []
        for p1 in [P1, 2.0 * P1, 0.5 * P1][:max(n_alias, 1)]:
            for p2 in [P2, 2.0 * P2, 0.5 * P2][:max(n_alias, 1)]:
                if 1.5 <= p1 <= max(50.0, 2 * K_natural) and \
                   1.5 <= p2 <= max(50.0, 4 * K_natural) and \
                   abs(p1 - p2) > 0.5:
                    thetas.append({"P": float(p1), "P2": float(p2)})
        return thetas[:max(n_alias * n_alias, 1)] or [{"P": P1, "P2": P2}]


# ----------------------------------------------------------------------------
# K_5: concentric / radial-periodic geometry
# ----------------------------------------------------------------------------

class K5_Concentric(ShapePrior):
    """Concentric basis: angular periodic + label-dependent radial variation.

    Phi(v) = [1, r(v) cos(2 pi v / P), r(v) sin(2 pi v / P), r(v)]
    where r(v) is a slow polynomial in v / (K-1).

    Captures geometries where the rotation is label-driven but the radius
    grows / shrinks slowly with the label.
    """
    name = "K5_Concentric"
    latent_dim = 2
    n_params = 1                       # period; radial poly absorbed
    expected_betti = (None, None, None)  # NEUTRAL (closed if outer curve closes)
    is_periodic = True
    expects_curvature_above_zero = True

    def n_basis(self) -> int:
        return 4

    def _radius(self, v, K_natural):
        t = v / max(K_natural - 1, 1)
        # quadratic radial profile centred around 1; lets W learn the scale.
        return 1.0 + 0.1 * (t - 0.5) + 0.05 * (t - 0.5) ** 2

    def build_basis(self, theta, v_codes, K_natural):
        P = float(theta.get("P", K_natural))
        omega = 2.0 * np.pi / max(P, 1e-6)
        v = v_codes.astype(np.float64)
        r = self._radius(v, K_natural)
        return np.vstack([np.ones_like(v),
                          r * np.cos(omega * v),
                          r * np.sin(omega * v),
                          r]).T

    def latent_from_basis(self, theta, v_codes, K_natural):
        P = float(theta.get("P", K_natural))
        omega = 2.0 * np.pi / max(P, 1e-6)
        v = v_codes.astype(np.float64)
        r = self._radius(v, K_natural)
        return np.vstack([r * np.cos(omega * v), r * np.sin(omega * v)]).T

    def propose_thetas(self, v_codes, K_natural, P_hat=None,
                       P_hat_2=None, n_alias=3):
        P_base = float(P_hat) if P_hat else float(K_natural)
        modes = [P_base, 2.0 * P_base, 0.5 * P_base]
        return [{"P": float(P)} for P in modes[:n_alias + 1]
                if 1.5 <= P <= max(50.0, 2 * K_natural)]


# ----------------------------------------------------------------------------
# K_6: ribbon / periodic-smooth 2D strip
# ----------------------------------------------------------------------------

class K6_Ribbon(ShapePrior):
    """Periodic-smooth surface (twisted ribbon).

    Phi(v) = [1, cos(2 pi v / P), sin(2 pi v / P), t, t cos(2 pi v / P),
              t sin(2 pi v / P)] where t = v / (K - 1).

    This is a 2D strip with one periodic coordinate and one smooth coordinate;
    the cross-term lets the strip width vary along the periodic axis (twist).
    PH is usually neutral here because a smooth open ribbon is contractible.
    """
    name = "K6_Ribbon"
    latent_dim = 2
    n_params = 1                       # period
    expected_betti = (None, None, None)  # NEUTRAL
    is_periodic = True

    def n_basis(self) -> int:
        return 6

    # K6 uses the SHARED empirical-Bayes alpha across all basis columns —
    # no hand-picked sub-prior. The K4 ↔ K6 ambiguity is correctly resolved
    # at the family level (Fix 2: 2D-periodic family Bayes factor) rather
    # than by handicapping K6 with an unjustified assumption on its
    # cross-term magnitude.

    def build_basis(self, theta, v_codes, K_natural):
        P = float(theta.get("P", K_natural))
        omega = 2.0 * np.pi / max(P, 1e-6)
        v = v_codes.astype(np.float64)
        t = v / max(K_natural - 1, 1)
        c = np.cos(omega * v)
        s = np.sin(omega * v)
        return np.vstack([np.ones_like(v), c, s, t, t * c, t * s]).T

    def latent_from_basis(self, theta, v_codes, K_natural):
        P = float(theta.get("P", K_natural))
        omega = 2.0 * np.pi / max(P, 1e-6)
        v = v_codes.astype(np.float64)
        t = v / max(K_natural - 1, 1)
        return np.vstack([np.cos(omega * v), np.sin(omega * v), t]).T

    def propose_thetas(self, v_codes, K_natural, P_hat=None,
                       P_hat_2=None, n_alias=3):
        P_base = float(P_hat) if P_hat else float(K_natural)
        modes = [P_base, 2.0 * P_base, 0.5 * P_base]
        return [{"P": float(P)} for P in modes[:n_alias + 1]
                if 1.5 <= P <= max(50.0, 2 * K_natural)]


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------

SHAPE_REGISTRY: Dict[str, ShapePrior] = {
    "K0_Generic":      K0_Generic(),
    "K1_Line":         K1_Line(),
    "K2_Circle":       K2_Circle(),
    "K3_OpenHelix":    K3_OpenHelix(),
    "K4_Torus":        K4_Torus(),
    "K5_Concentric":   K5_Concentric(),
    "K6_Ribbon":       K6_Ribbon(),
}


# Shape -> short verdict name for the summary CSV
SHAPE_TO_VERDICT = {
    "K0_Generic":      "smooth_only",
    "K1_Line":         "line",
    "K2_Circle":       "circle",
    "K3_OpenHelix":    "helix",
    "K4_Torus":        "torus",
    "K5_Concentric":   "concentric",
    "K6_Ribbon":       "ribbon",
}


# ----------------------------------------------------------------------------
# Family map — for the family-level Bayes factor (Fix 2 of docs/gplvm.md).
# When the data supports a *family* of shapes but cannot decide between
# members (e.g. K4_Torus vs K6_Ribbon), the pipeline can claim Tier A at
# the family level instead of refusing.
# ----------------------------------------------------------------------------

SHAPE_TO_FAMILY = {
    "K0_Generic":    "trivial",
    "K1_Line":       "trivial",
    "K2_Circle":     "1D_periodic",
    "K3_OpenHelix":  "1D_periodic",
    "K4_Torus":      "2D_periodic",
    "K5_Concentric": "2D_periodic",
    "K6_Ribbon":     "2D_periodic",
}

FAMILY_TO_SHAPES: Dict[str, List[str]] = {}
for _s, _f in SHAPE_TO_FAMILY.items():
    FAMILY_TO_SHAPES.setdefault(_f, []).append(_s)

FAMILY_DESCRIPTION = {
    "trivial":     "smooth / line — no periodic structure",
    "1D_periodic": "1D cyclic curve (circle or open helix)",
    "2D_periodic": "2D periodic surface (torus, concentric, or ribbon)",
}


def all_shapes() -> List[ShapePrior]:
    """Iteration order: K_0 last so it is always treated as fallback in ties."""
    return [SHAPE_REGISTRY[name] for name in
            ("K1_Line", "K2_Circle", "K3_OpenHelix", "K4_Torus",
             "K5_Concentric", "K6_Ribbon", "K0_Generic")]


def topology_consistent(shape: ShapePrior,
                         betti_obs: Tuple[float, float, float],
                         betti_std: Tuple[float, float, float],
                         tol_z: float = 2.0) -> str:
    """Per docs/gplvm.md Stage 8: PH is supportive / neutral / contradictory.

    Returns one of {'supportive', 'neutral', 'contradictory', 'uncertain'}.
    NEVER 'reject' — that's a different output handled at decision time.

    Rules (β₀ always skipped; sampled β₀ is unreliable, see K1_Line note):
        supportive    : |obs - expected| == 0 on at least one non-None dim
        contradictory : |obs - expected| >= 2 on any non-None dim, OR
                        (expected == 0 AND obs >= 1)
        neutral       : shape has expected_betti = (None, None, None)
        uncertain     : otherwise (e.g. |diff| == 1 on a single dim)
    """
    eb = shape.expected_betti
    if all(e is None for e in eb):
        return "neutral"
    contradictions = 0
    supports = 0
    for i, e in enumerate(eb):
        if e is None:
            continue
        obs = int(betti_obs[i])
        diff = abs(obs - e)
        if diff == 0:
            supports += 1
        elif diff >= 2 or (e == 0 and obs >= 1):
            contradictions += 1
    if contradictions > 0:
        return "contradictory"
    if supports > 0:
        return "supportive"
    return "uncertain"


def topology_score(ph_status: str) -> float:
    """Map PH status to evidence score in {-1, 0, +1}."""
    return {"supportive": 1.0,
            "neutral": 0.0,
            "uncertain": 0.0,
            "contradictory": -1.0}.get(ph_status, 0.0)
