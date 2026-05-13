"""Toy validation for Stage 2b — Spread-aware Mahalanobis test (d_SW).

Pre-registered toys (plan §3.2.5 + Stage 2b extensions):
  1B Line              — 1D line in 9D + tight isotropic noise; verdict = spread_confirmed.
  2B Circle            — 2D circle in 9D + tight noise; verdict = spread_confirmed.
  3B Helix             — Circular helix in 9D + tight noise; verdict = spread_confirmed.
  4B Isotropic Gaussian — pure noise + random labels; verdict = centroid_only_shape.
  5B OverlapHelix       — helix centroids, Σ_v elongated along tangent (calibrated);
                          verdict = centroid_only_shape. Negative-control discriminator.
  6B TightFog          — helix centroids, Σ_v anisotropic perpendicular to tangent;
                          verdict = spread_confirmed (perpendicular spread doesn't kill ranks).
  7B FPR calibration   — 100 cells of isotropic Gaussian; spread_confirmed rate ∈ [2, 11].

In addition: a synthetic verdict-ladder unit test exercises (ρ=0.90, ρ_low=0.30, p=0.01)
to confirm the ρ_low-dominant gate catches the gap from review item #1.

Run on GPU when available. Block run_stage2b if any toy fails.

CLI:
  python check_stage2b_toys.py --config /home/anshulk/emnlp2026/config.yaml
  python check_stage2b_toys.py --calibrate-5b      # run only Toy 5B scale sweep
  python check_stage2b_toys.py --run-7b            # run only Toy 7B FPR (100 cells)
"""

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2b_dsw_spread_aware import (
    analyze_cell_dsw,
    assign_verdict, assign_confidence_tier,
    VERDICT_SPREAD_CONFIRMED, VERDICT_SPREAD_MARGINAL, VERDICT_CENTROID_ONLY,
    VERDICT_INSUFFICIENT, VERDICT_LOW_K_AFTER_FILTER, VERDICT_NULL_UNSTABLE_DSW,
    TIER_HIGH, TIER_MEDIUM, TIER_LOW, TIER_DISCOVERY_ONLY,
)

TOY_D = 9


def _default_cfg(n_perms: int = 200, n_boot: int = 200) -> dict:
    """Lighter perms/bootstraps for toys — production uses 1000 each."""
    return {
        "n_permutations": n_perms,
        "n_bootstrap": n_boot,
        "lambda_factor": 1e-6,
        "shrinkage_lw_threshold": 10,
        "shrinkage_oas_threshold": 5,
        "rho_pass_threshold": 0.85,
        "rho_low_ci_threshold": 0.70,
        "rho_marginal_low": 0.50,
        "ci_halfwidth_unstable": 0.30,
        "fdr_alpha": 0.05,
        "min_group_size": 10,            # toys are smaller-N than real cells
        "bootstrap_min_n_v_floor": 5,
        "bootstrap_max_redraws": 50,
        "min_K_for_dsw": 4,
        "tier_high_min_K": 6,
        "tier_high_min_n_v": 20,
        "tier_high_min_ratio": 2,
        "tier_high_q_threshold": 0.01,
        "tier_high_rho_low": 0.80,
        "tier_medium_min_K": 5,
        "tier_medium_min_n_v": 15,
        "tier_medium_min_ratio": 1.5,
        "tier_medium_q_threshold": 0.05,
        "tier_medium_rho_low": 0.70,
        "tier_discovery_only_max_ratio": 1.0,
    }


def _embed_2d_to_d(xy: np.ndarray, d: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Returns (Z, Q) where Q has orthonormal columns: (d, 2)."""
    A = rng.standard_normal((d, 2))
    Q, _ = np.linalg.qr(A)  # (d, 2)
    Z = xy @ Q.T
    return Z.astype(np.float64), Q


def _embed_3d_to_d(xyz: np.ndarray, d: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    A = rng.standard_normal((d, 3))
    Q, _ = np.linalg.qr(A)  # (d, 3)
    Z = xyz @ Q.T
    return Z.astype(np.float64), Q


def _balanced_labels(N: int, K: int, rng: np.random.Generator) -> np.ndarray:
    labels = np.repeat(np.arange(K), N // K)
    n_rem = N - labels.size
    if n_rem > 0:
        labels = np.concatenate([labels, rng.integers(0, K, size=n_rem)])
    rng.shuffle(labels)
    return labels.astype(np.int64)


def _make_line(N: int, K: int, d: int, rng: np.random.Generator,
               noise_std: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    labels = _balanced_labels(N, K, rng)
    t = labels.astype(np.float64) / max(K - 1, 1)
    direction = rng.standard_normal(d); direction /= np.linalg.norm(direction)
    Z = (t[:, None] * direction[None, :] * 5.0) + rng.standard_normal((N, d)) * noise_std
    return Z.astype(np.float32), labels


def _make_circle(N: int, K: int, d: int, rng: np.random.Generator,
                  noise_std: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    labels = _balanced_labels(N, K, rng)
    theta = 2 * np.pi * labels.astype(np.float64) / K
    xy = np.stack([np.cos(theta), np.sin(theta)], axis=1) * 5.0
    Z, _ = _embed_2d_to_d(xy, d, rng)
    Z = Z + rng.standard_normal(Z.shape) * noise_std
    return Z.astype(np.float32), labels


def _make_helix(N: int, K: int, d: int, rng: np.random.Generator,
                 pitch: float = 1.0, noise_std: float = 0.5
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (Z, labels, tangent_direction_in_ambient). The tangent encodes the
    helix pitch direction so Toy 5B can elongate Σ_v along it."""
    labels = _balanced_labels(N, K, rng)
    theta = 2 * np.pi * labels.astype(np.float64) / K
    z_axis = labels.astype(np.float64) * pitch
    xyz = np.stack([5.0 * np.cos(theta), 5.0 * np.sin(theta), z_axis], axis=1)
    Z, Q3 = _embed_3d_to_d(xyz, d, rng)
    Z = Z + rng.standard_normal(Z.shape) * noise_std
    tangent = Q3[:, 2]  # the pitch axis in ambient space
    tangent = tangent / max(np.linalg.norm(tangent), 1e-12)
    return Z.astype(np.float32), labels, tangent


def _make_isotropic(N: int, K: int, d: int, rng: np.random.Generator
                     ) -> tuple[np.ndarray, np.ndarray]:
    Z = rng.standard_normal((N, d)).astype(np.float32)
    labels = _balanced_labels(N, K, rng)
    return Z, labels


def _add_anisotropic_per_value_noise(Z: np.ndarray, labels: np.ndarray,
                                      direction: np.ndarray, scale: float,
                                      rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise along `direction` (unit vector) of std `scale` to every point.
    Note: this adds the SAME single global direction to every point. Used by
    Toy 6B-TightFog with a perpendicular direction. For Toy 5B-OverlapHelix
    we instead use per-class arc-spread via parameter-jitter (see _make_overlap_helix).
    """
    direction = direction / max(np.linalg.norm(direction), 1e-12)
    noise_amplitude = rng.standard_normal(Z.shape[0]).astype(np.float32) * scale
    return Z + noise_amplitude[:, None] * direction.astype(np.float32)[None, :]


def _make_overlap_helix(N: int, K: int, d: int, rng: np.random.Generator,
                         pitch: float = 1.0, t_jitter_scale: float = 1.0,
                         noise_std: float = 0.05
                         ) -> tuple[np.ndarray, np.ndarray]:
    """[Kept for reference] Helix with per-point parameter jitter.
    Centroids stay roughly on the helix path, spread along helix tangent.
    This does NOT break d_SW because Σ_u and Σ_v share the same eigenstructure
    direction (the helix tangent). d_SW catches a different mirage: per-value Σ
    pointing in different directions. See `_make_anisotropic_overlap_helix`."""
    labels = _balanced_labels(N, K, rng)
    t = labels.astype(np.float64) + t_jitter_scale * rng.standard_normal(N)
    theta = 2 * np.pi * t / K
    xyz = np.stack([5.0 * np.cos(theta), 5.0 * np.sin(theta), pitch * t], axis=1)
    Z, _ = _embed_3d_to_d(xyz, d, rng)
    Z = Z + rng.standard_normal(Z.shape) * noise_std
    return Z.astype(np.float32), labels


def _make_anisotropic_overlap_helix(N: int, K: int, d: int,
                                      rng: np.random.Generator,
                                      pitch: float = 1.0,
                                      per_value_scale: float = 1.0,
                                      iso_noise_std: float = 0.05,
                                      ) -> tuple[np.ndarray, np.ndarray]:
    """Helix centroids + per-VALUE anisotropic noise in DIFFERENT random directions.

    Construction:
      μ_v on a clean helix in 9D.
      Each value v gets a private random unit direction d_v in 9D.
      Per-point: x_i = μ_{label_i} + amp_i * d_{label_i} + small isotropic noise,
                  amp_i ~ N(0, per_value_scale²).

    Result: Σ_v = per_value_scale² · d_v d_vᵀ + iso² · I. Different values have
    Σ pointing in different directions, so per-pair Σ_pool has anisotropy in two
    random dirs. The Mahalanobis Σ⁺⁻¹ whitens different directions per pair,
    scrambling the rank correspondence with D_E. This is the "centroid mirage"
    failure mode Stage 2b targets — analogous to per-class inherited algebraic
    correlates in real cells."""
    labels = _balanced_labels(N, K, rng)
    theta = 2 * np.pi * labels.astype(np.float64) / K
    z_axis = labels.astype(np.float64) * pitch
    xyz = np.stack([5.0 * np.cos(theta), 5.0 * np.sin(theta), z_axis], axis=1)
    Z, _ = _embed_3d_to_d(xyz, d, rng)
    # Per-value directions
    D = rng.standard_normal((K, d))
    D = D / np.linalg.norm(D, axis=1, keepdims=True)
    amps = rng.standard_normal(N).astype(np.float64) * per_value_scale
    Z = Z + amps[:, None] * D[labels].astype(np.float64)
    Z = Z + rng.standard_normal(Z.shape) * iso_noise_std
    return Z.astype(np.float32), labels


# ──────────────────────────────────────────────────────────────────────────────
# Verdict-ladder unit test (review item #1 — the (0.90, 0.30) gap)
# ──────────────────────────────────────────────────────────────────────────────

def unit_test_verdict_ladder() -> None:
    """Synthetic synthetic-rho case must land in centroid_only_shape via ρ_low < 0.50."""
    print("=== Unit test: verdict-ladder ρ_low dominant gate ===")
    cfg = _default_cfg()
    cases = [
        # (rho, rho_low, p, expected_verdict)
        (0.90, 0.30, 0.01, VERDICT_CENTROID_ONLY),     # the gap case from review #1
        (0.95, 0.85, 0.01, VERDICT_SPREAD_CONFIRMED),  # clean pass
        (0.90, 0.65, 0.01, VERDICT_SPREAD_MARGINAL),   # ρ_low in marginal band
        (0.80, 0.60, 0.01, VERDICT_SPREAD_MARGINAL),   # ρ in marginal band
        (0.50, 0.40, 0.01, VERDICT_CENTROID_ONLY),     # ρ < 0.70
        (0.95, 0.85, 0.20, VERDICT_CENTROID_ONLY),     # p ≥ 0.05
    ]
    fails = []
    for rho, rho_low, p, expected in cases:
        got = assign_verdict(rho, rho_low, p, cfg)
        ok = got == expected
        print(f"  ρ={rho:.2f} ρ_low={rho_low:.2f} p={p:.2g}  →  {got}  (expected {expected})  {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append((rho, rho_low, p, got, expected))
    if fails:
        raise AssertionError(f"verdict-ladder unit test: {len(fails)}/{len(cases)} failed")


# ──────────────────────────────────────────────────────────────────────────────
# Toy 1B: Line — spread_confirmed expected
# ──────────────────────────────────────────────────────────────────────────────

def toy_1B_line() -> None:
    print("=== Toy 1B Line ===")
    cfg = _default_cfg()
    fails = []
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        Z, labels = _make_line(N=1000, K=10, d=TOY_D, rng=rng, noise_std=0.5)
        K_natural = int(np.unique(labels).size)
        res = analyze_cell_dsw(Z, labels, K_natural, cfg, seed=seed)
        ok = res["spread_verdict"] in (VERDICT_SPREAD_CONFIRMED, VERDICT_SPREAD_MARGINAL)
        print(f"  seed={seed}: ρ={res['rho_centroid']:.3f}  CI=[{res['rho_low']:.2f},{res['rho_high']:.2f}]  "
              f"p={res['p_dsw']:.3g}  verdict={res['spread_verdict']}  {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append((seed, res))
    if fails:
        raise AssertionError(f"Toy 1B Line: {len(fails)}/3 seeds failed (expect spread_confirmed)")


def toy_2B_circle() -> None:
    print("=== Toy 2B Circle ===")
    cfg = _default_cfg()
    fails = []
    for seed in (10, 11, 12):
        rng = np.random.default_rng(seed)
        Z, labels = _make_circle(N=1000, K=10, d=TOY_D, rng=rng, noise_std=0.5)
        K_natural = int(np.unique(labels).size)
        res = analyze_cell_dsw(Z, labels, K_natural, cfg, seed=seed)
        ok = res["spread_verdict"] in (VERDICT_SPREAD_CONFIRMED, VERDICT_SPREAD_MARGINAL)
        print(f"  seed={seed}: ρ={res['rho_centroid']:.3f}  CI=[{res['rho_low']:.2f},{res['rho_high']:.2f}]  "
              f"p={res['p_dsw']:.3g}  verdict={res['spread_verdict']}  {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append((seed, res))
    if fails:
        raise AssertionError(f"Toy 2B Circle: {len(fails)}/3 seeds failed (expect spread_confirmed)")


def toy_3B_helix() -> None:
    print("=== Toy 3B Helix ===")
    cfg = _default_cfg()
    fails = []
    for seed in (20, 21, 22):
        rng = np.random.default_rng(seed)
        Z, labels, _ = _make_helix(N=1000, K=10, d=TOY_D, rng=rng, pitch=1.0, noise_std=0.5)
        K_natural = int(np.unique(labels).size)
        res = analyze_cell_dsw(Z, labels, K_natural, cfg, seed=seed)
        ok = res["spread_verdict"] in (VERDICT_SPREAD_CONFIRMED, VERDICT_SPREAD_MARGINAL)
        print(f"  seed={seed}: ρ={res['rho_centroid']:.3f}  CI=[{res['rho_low']:.2f},{res['rho_high']:.2f}]  "
              f"p={res['p_dsw']:.3g}  verdict={res['spread_verdict']}  {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append((seed, res))
    if fails:
        raise AssertionError(f"Toy 3B Helix: {len(fails)}/3 seeds failed (expect spread_confirmed)")


def toy_4B_isotropic() -> None:
    """Sanity check on a small batch of isotropic cells. Under H0 with α=0.05,
    expect ≤ 5% false positives — over 20 seeds, binomial 95% upper bound is 4.
    The proper FPR calibration is Toy 7B (100 cells)."""
    print("=== Toy 4B Isotropic Gaussian (20 seeds; FPR sanity, not headline) ===")
    cfg = _default_cfg()
    false_positives = 0
    n_seeds = 20
    for seed in range(30, 30 + n_seeds):
        rng = np.random.default_rng(seed)
        Z, labels = _make_isotropic(N=1000, K=10, d=TOY_D, rng=rng)
        K_natural = int(np.unique(labels).size)
        res = analyze_cell_dsw(Z, labels, K_natural, cfg, seed=seed)
        if res["spread_verdict"] == VERDICT_SPREAD_CONFIRMED:
            false_positives += 1
    fp_rate = false_positives / n_seeds
    # Binomial 95% upper bound: P(X >= k | n=20, p=0.05) ≤ 0.05 gives k = 4
    upper_bound = 4
    ok = false_positives <= upper_bound
    print(f"  false_positives = {false_positives}/{n_seeds} (rate={fp_rate:.2f}; "
          f"binomial 95% upper bound = {upper_bound})  {'OK' if ok else 'FAIL'}")
    if not ok:
        raise AssertionError(
            f"Toy 4B Isotropic: {false_positives}/{n_seeds} spread_confirmed under H0 > {upper_bound} — "
            "Null appears miscalibrated. Re-check Toy 7B for the headline FPR.")


# ──────────────────────────────────────────────────────────────────────────────
# Toy 5B: OverlapHelix — calibration sweep + lock smallest passing tangent_scale
# ──────────────────────────────────────────────────────────────────────────────

def _toy_5B_rho_for_scale(scale: float, seed: int, cfg: dict) -> dict:
    """Return Stage 2b result for AnisotropicOverlapHelix at given per-value scale.
    `scale` is the std of per-class anisotropic noise added along a random direction
    that DIFFERS per value. At scale=0 the helix is clean (spread_confirmed); as
    scale grows, Σ_u and Σ_v point in increasingly different directions and the
    pairwise Mahalanobis whitening scrambles the rank correspondence."""
    rng = np.random.default_rng(seed)
    Z, labels = _make_anisotropic_overlap_helix(
        N=1000, K=10, d=TOY_D, rng=rng, pitch=1.0,
        per_value_scale=scale, iso_noise_std=0.05,
    )
    K_natural = int(np.unique(labels).size)
    return analyze_cell_dsw(Z, labels, K_natural, cfg, seed=seed)


def calibrate_5B_tangent_scale(cfg_main: dict, toy_yaml_path: Path) -> float:
    """Sweep tangent scales; verify the test responds monotonically and the
    extreme scale does NOT pass spread_confirmed. Locks the largest scale
    (the most aggressive negative-control) as the toy_5b_tangent_scale.

    Toy 5B is a continuous discriminator probe, not a binary calibrator: the
    anisotropic-per-value spread that breaks d_SW exists on a spectrum, and
    even at extreme spread the rank correlation degrades to ~0.75–0.85 (not 0).
    The calibrator verifies (a) max ρ at extreme scale is below the 0.85 pass
    gate AND (b) ρ trends down across the sweep (median monotone decrease)."""
    print("=== Toy 5B calibration: tangent_scale sweep ===")
    swept = list(map(float, cfg_main.get("toy_5b_scale_sweep", [0.5, 1.0, 2.0, 5.0, 10.0, 20.0])))
    n_seeds = int(cfg_main.get("toy_5b_n_seeds", 3))
    extreme_pass_max_rho = float(cfg_main.get("toy_5b_extreme_max_rho", 0.85))
    cfg = _default_cfg()
    per_scale = {}
    per_scale_verdicts = {}
    for s in swept:
        rhos = []
        verdicts = []
        for seed_off in range(n_seeds):
            seed = 100 + int(s * 10) + seed_off
            res = _toy_5B_rho_for_scale(s, seed, cfg)
            rhos.append(float(res["rho_centroid"]))
            verdicts.append(str(res["spread_verdict"]))
        per_scale[s] = rhos
        per_scale_verdicts[s] = verdicts
        print(f"  scale={s:5.1f}  ρ_seeds={[f'{r:.3f}' for r in rhos]}  "
              f"median={float(np.median(rhos)):.3f}  verdicts={verdicts}")

    # Anchor: the largest swept scale must have median ρ < pass gate AND
    # at least one seed must NOT be spread_confirmed.
    extreme_scale = max(swept)
    extreme_rhos = per_scale[extreme_scale]
    extreme_verdicts = per_scale_verdicts[extreme_scale]
    median_extreme = float(np.median(extreme_rhos))
    any_failed_to_pass = any(v != VERDICT_SPREAD_CONFIRMED for v in extreme_verdicts)

    # Monotonicity: median ρ at smallest scale > median ρ at largest scale
    smallest_med = float(np.median(per_scale[min(swept)]))
    monotone_ok = smallest_med > median_extreme - 0.02   # noise floor

    print(f"  extreme_scale={extreme_scale}  median_ρ={median_extreme:.3f}  "
          f"vs pass_gate={extreme_pass_max_rho}  any-non-spread_confirmed={any_failed_to_pass}")
    print(f"  smallest_scale_med={smallest_med:.3f}  monotonicity OK={monotone_ok}")

    if median_extreme >= extreme_pass_max_rho or not any_failed_to_pass:
        raise AssertionError(
            f"Toy 5B: at extreme scale={extreme_scale}, median ρ={median_extreme:.3f} did not "
            f"drop below pass gate {extreme_pass_max_rho}, OR all seeds remained spread_confirmed. "
            f"Sweep: {per_scale}")
    if not monotone_ok:
        raise AssertionError(
            f"Toy 5B: ρ not monotonically decreasing with scale "
            f"(smallest_med={smallest_med}, extreme_med={median_extreme})")

    chosen = float(extreme_scale)
    cal_record = {
        "swept_scales": swept,
        "per_scale_rho": {float(k): list(map(float, v)) for k, v in per_scale.items()},
        "per_scale_verdicts": {float(k): list(v) for k, v in per_scale_verdicts.items()},
        "chosen_scale": chosen,
        "extreme_pass_max_rho": extreme_pass_max_rho,
        "monotonicity_ok": bool(monotone_ok),
        "n_seeds": n_seeds,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "lib_versions": {"numpy": np.__version__},
        "discriminator_kind": "continuous_probe",  # NOT a binary discriminator
    }
    existing = yaml.safe_load(toy_yaml_path.read_text()) or {}
    existing["toy_5b_tangent_scale"] = chosen
    existing["toy_5b_calibration"] = cal_record
    toy_yaml_path.write_text(yaml.safe_dump(existing, sort_keys=False, default_flow_style=False))
    print(f"  -> locked tangent_scale={chosen}  ({toy_yaml_path})")
    return chosen


def toy_5B_overlap_helix(scale: float) -> None:
    """At the locked (largest) scale, the verdict must NOT be spread_confirmed
    across all 3 seeds. Marginal or centroid_only_shape both count as a
    successful discrimination. This is the continuous-probe pass condition."""
    print(f"=== Toy 5B OverlapHelix (locked scale={scale}) — must not be spread_confirmed ===")
    cfg = _default_cfg()
    fails = []
    for seed in (40, 41, 42):
        res = _toy_5B_rho_for_scale(scale, seed, cfg)
        ok = res["spread_verdict"] != VERDICT_SPREAD_CONFIRMED
        print(f"  seed={seed}: ρ={res['rho_centroid']:.3f}  CI=[{res['rho_low']:.2f},{res['rho_high']:.2f}]  "
              f"p={res['p_dsw']:.3g}  verdict={res['spread_verdict']}  {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append((seed, res))
    if fails:
        raise AssertionError(f"Toy 5B: {len(fails)}/3 seeds still spread_confirmed at scale={scale} "
                             f"— d_SW failed to discriminate the per-value-anisotropic overlap.")


# ──────────────────────────────────────────────────────────────────────────────
# Toy 6B: TightFog — anisotropic perpendicular spread, spread_confirmed expected
# ──────────────────────────────────────────────────────────────────────────────

def toy_6B_tight_fog() -> None:
    print("=== Toy 6B TightFog (perpendicular anisotropy) ===")
    cfg = _default_cfg()
    fails = []
    for seed in (50, 51, 52):
        rng = np.random.default_rng(seed)
        Z, labels, tangent = _make_helix(N=1000, K=10, d=TOY_D, rng=rng, pitch=1.0, noise_std=0.05)
        # Build a perpendicular direction: random vector orthogonalised against tangent.
        v = rng.standard_normal(TOY_D)
        v = v - (v @ tangent) * tangent
        v = v / max(np.linalg.norm(v), 1e-12)
        Z2 = _add_anisotropic_per_value_noise(Z, labels, direction=v, scale=0.5, rng=rng)
        K_natural = int(np.unique(labels).size)
        res = analyze_cell_dsw(Z2, labels, K_natural, cfg, seed=seed)
        ok = res["spread_verdict"] in (VERDICT_SPREAD_CONFIRMED, VERDICT_SPREAD_MARGINAL)
        print(f"  seed={seed}: ρ={res['rho_centroid']:.3f}  CI=[{res['rho_low']:.2f},{res['rho_high']:.2f}]  "
              f"p={res['p_dsw']:.3g}  verdict={res['spread_verdict']}  {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append((seed, res))
    if fails:
        raise AssertionError(f"Toy 6B TightFog: {len(fails)}/3 seeds failed "
                             f"(perpendicular spread should preserve ranks)")


# ──────────────────────────────────────────────────────────────────────────────
# Toy 7B: FPR calibration on 100 isotropic cells
# ──────────────────────────────────────────────────────────────────────────────

def toy_7B_fpr(cfg_main: dict, toy_yaml_path: Path, n_cells: int | None = None) -> dict:
    n_cells = n_cells or int(cfg_main.get("toy_7b_n_cells", 100))
    lower = int(cfg_main.get("toy_7b_lower", 2))
    upper = int(cfg_main.get("toy_7b_upper", 11))
    print(f"=== Toy 7B FPR calibration ({n_cells} isotropic cells, accept band [{lower}, {upper}]) ===")
    cfg = _default_cfg()
    detections = 0
    t0 = time.time()
    for cell_seed in range(60, 60 + n_cells):
        rng = np.random.default_rng(cell_seed)
        Z, labels = _make_isotropic(N=200, K=10, d=TOY_D, rng=rng)
        K_natural = int(np.unique(labels).size)
        res = analyze_cell_dsw(Z, labels, K_natural, cfg, seed=cell_seed)
        if res["spread_verdict"] == VERDICT_SPREAD_CONFIRMED:
            detections += 1
    dt = time.time() - t0
    status = "pass" if (lower <= detections <= upper) else "fail"
    print(f"  detections = {detections}/{n_cells}  band=[{lower},{upper}]  time={dt:.1f}s  status={status}")

    record = {
        "n_cells": n_cells,
        "n_spread_confirmed": int(detections),
        "accept_band": [lower, upper],
        "status": status,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "elapsed_seconds": round(dt, 1),
    }
    existing = yaml.safe_load(toy_yaml_path.read_text()) or {}
    existing["toy_7b_fpr"] = record
    toy_yaml_path.write_text(yaml.safe_dump(existing, sort_keys=False, default_flow_style=False))
    print(f"  wrote {toy_yaml_path}")
    if status != "pass":
        raise AssertionError(
            f"Toy 7B FPR: {detections}/{n_cells} spread_confirmed under H0 — "
            f"outside accept band [{lower}, {upper}]. Null is miscalibrated.")
    return record


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_paths(config_path: Path) -> tuple[dict, Path]:
    cfg_all = yaml.safe_load(config_path.read_text())
    stage2b_cfg = cfg_all.get("stage2b", {})
    toy_yaml = config_path.parent / stage2b_cfg.get("toy_calibration_path", "configs/stage2b.yaml")
    return stage2b_cfg, toy_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="/home/anshulk/emnlp2026/config.yaml")
    ap.add_argument("--calibrate-5b", action="store_true", help="Only run Toy 5B scale calibration")
    ap.add_argument("--run-7b", action="store_true", help="Only run Toy 7B FPR (100 cells)")
    ap.add_argument("--quick-7b", type=int, default=None,
                    help="Run a smaller Toy 7B (n cells) for debugging only")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    stage2b_cfg, toy_yaml = _resolve_paths(cfg_path)

    if args.calibrate_5b and args.run_7b:
        # Run both standalone modes; useful for fresh setup.
        unit_test_verdict_ladder()
        calibrate_5B_tangent_scale(stage2b_cfg, toy_yaml)
        toy_7B_fpr(stage2b_cfg, toy_yaml, n_cells=args.quick_7b)
        return
    if args.calibrate_5b:
        unit_test_verdict_ladder()
        calibrate_5B_tangent_scale(stage2b_cfg, toy_yaml)
        return
    if args.run_7b:
        toy_7B_fpr(stage2b_cfg, toy_yaml, n_cells=args.quick_7b)
        return

    # Full sequence: ladder unit test → Toys 1B–4B → 5B calibration → 5B verdict → 6B → 7B
    t_start = time.time()
    unit_test_verdict_ladder()
    toy_1B_line()
    toy_2B_circle()
    toy_3B_helix()
    toy_4B_isotropic()
    scale = calibrate_5B_tangent_scale(stage2b_cfg, toy_yaml)
    toy_5B_overlap_helix(scale)
    toy_6B_tight_fog()
    toy_7B_fpr(stage2b_cfg, toy_yaml, n_cells=args.quick_7b)
    print(f"\nALL TOYS + UNIT TESTS PASSED in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nFAIL: {e}", file=sys.stderr)
        sys.exit(1)
