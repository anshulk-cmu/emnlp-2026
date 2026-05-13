"""Toy validation for Stage 2a — Centroid Fourier helix fit (discover-then-fit).

Pre-registered toys from plan.md §3.2.5 extended to verify period discovery (not
just FCR magnitude) and Whittle max-over-frequencies null calibration.

Toys:
  1B Line              — 1D line in 9D + noise; expect verdict = none.
  2B Circle            — 2D circle in 9D, 10 angles; discover P=10; verdict = circle.
  3B Helix             — Circular helix in 9D, 10 params; discover P=10; verdict = helix.
  4B Isotropic Gaussian — 9D random + random labels; verdict = none, q > 0.05.
  5B Period-7 circle   — 2D circle, 7 angles; discover P=7 (no bias toward 10).
  6B Period-13 helix   — Circular helix, 13 params; discover P=13 (prime period).
  7B Aliased noise FPR — 100 cells of pure isotropic Gaussian; ≥ 90 verdicts = none
                          (binomial 95% upper bound at p=0.05 is 9).

Run on GPU when available, ~3 min total. Block run_stage2a if any toy fails.
"""

import sys
import time
from pathlib import Path

import numpy as np

# Force re-import path so we can run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage2a_fourier_helix import (
    AMBIENT_D,
    VERDICT_HELIX, VERDICT_CIRCLE, VERDICT_NONE,
    analyze_cell,
)

# Smaller embed dimension for toys — algorithm operates in subspace, not 4096-D
TOY_D = 9
DEFAULT_CFG = {
    "n_permutations": 1000,
    "fcr_threshold": 0.30,
    "two_axis_alpha": 0.01,
    "linear_alpha": 0.01,
    "fdr_alpha": 0.05,
    "min_K_for_fft": 5,
    "min_group_size": 30,
    "redraw_rate_max": 0.10,
    "concordance_bin_tolerance": 1,
    "vote_winner_margin": 2.0,
}


def _embed_2d_to_d(xy: np.ndarray, d: int, rng: np.random.Generator,
                    noise_std: float = 0.0) -> np.ndarray:
    """Embed (N, 2) data into (N, d) via a random orthonormal projection + noise.

    Equivalent to placing a 2D shape inside a d-dimensional subspace.
    """
    A = rng.standard_normal((d, 2))
    Q, _ = np.linalg.qr(A)
    Z = xy @ Q.T  # (N, d)
    if noise_std > 0:
        Z = Z + rng.standard_normal(Z.shape) * noise_std
    return Z.astype(np.float32)


def _embed_3d_to_d(xyz: np.ndarray, d: int, rng: np.random.Generator,
                    noise_std: float = 0.0) -> np.ndarray:
    A = rng.standard_normal((d, 3))
    Q, _ = np.linalg.qr(A)
    Z = xyz @ Q.T
    if noise_std > 0:
        Z = Z + rng.standard_normal(Z.shape) * noise_std
    return Z.astype(np.float32)


def _make_line(N: int, K: int, d: int, rng: np.random.Generator, noise: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """K evenly spaced labels along a 1D line in d-D + noise."""
    labels = np.repeat(np.arange(K), N // K)
    n_remaining = N - labels.size
    if n_remaining > 0:
        labels = np.concatenate([labels, rng.integers(0, K, size=n_remaining)])
    rng.shuffle(labels)
    t = labels.astype(np.float64) / max(K - 1, 1)
    direction = rng.standard_normal(d); direction /= np.linalg.norm(direction)
    Z = t[:, None] * direction[None, :] * 5.0  # signal scale
    Z = Z + rng.standard_normal(Z.shape) * noise
    return Z.astype(np.float32), labels


def _make_circle(N: int, K: int, d: int, rng: np.random.Generator, noise: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """K evenly spaced labels on a 2D circle embedded in d-D + noise."""
    labels = np.repeat(np.arange(K), N // K)
    n_remaining = N - labels.size
    if n_remaining > 0:
        labels = np.concatenate([labels, rng.integers(0, K, size=n_remaining)])
    rng.shuffle(labels)
    theta = 2 * np.pi * labels.astype(np.float64) / K
    xy = np.stack([np.cos(theta), np.sin(theta)], axis=1) * 5.0
    Z = _embed_2d_to_d(xy, d, rng, noise_std=noise)
    return Z, labels


def _make_helix(N: int, K: int, d: int, rng: np.random.Generator,
                 pitch: float = 1.0, noise: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """K evenly spaced labels on a circular helix (circle + linear drift) in d-D."""
    labels = np.repeat(np.arange(K), N // K)
    n_remaining = N - labels.size
    if n_remaining > 0:
        labels = np.concatenate([labels, rng.integers(0, K, size=n_remaining)])
    rng.shuffle(labels)
    theta = 2 * np.pi * labels.astype(np.float64) / K
    z_axis = labels.astype(np.float64) * pitch
    xyz = np.stack([5.0 * np.cos(theta), 5.0 * np.sin(theta), z_axis], axis=1)
    Z = _embed_3d_to_d(xyz, d, rng, noise_std=noise)
    return Z, labels


def _make_isotropic(N: int, K: int, d: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """K balanced labels on N(0, I_d) — pure noise. Balanced to avoid spurious
    sparse_value_grid triggers; the test is about Z vs labels, not about group-size."""
    Z = rng.standard_normal((N, d)).astype(np.float32)
    labels = np.repeat(np.arange(K), N // K)
    n_remaining = N - labels.size
    if n_remaining > 0:
        labels = np.concatenate([labels, rng.integers(0, K, size=n_remaining)])
    rng.shuffle(labels)
    return Z, labels


# Verdicts indicating "no helix or circle found" (all valid null outcomes)
NO_DETECTION = {VERDICT_NONE, "period_inconsistent", "low_K"}


def _run(Z, labels, K_natural, seed, cfg=None):
    cfg = cfg or DEFAULT_CFG
    return analyze_cell(Z, labels, K_natural=K_natural, cfg_stage2a=cfg, seed=seed, use_gpu=True)


def toy_1B_line() -> None:
    """1D line in 9D — periodic test should fail because the data has linear ramp, not
    periodic structure. Any non-detection verdict (none / period_inconsistent /
    sparse_value_grid) passes; specifically not helix/circle."""
    print("=== Toy 1B Line ===")
    fails = []
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        Z, labels = _make_line(N=400, K=10, d=TOY_D, rng=rng, noise=0.05)
        res = _run(Z, labels, K_natural=10, seed=seed)
        ok = res["geometry_detected"] in NO_DETECTION
        print(f"  seed={seed}: geom={res['geometry_detected']}, fcr_helix={res['fcr_helix']:.3f}, "
              f"P*={res['discovered_period']:.2f}, p_helix={res['p_helix']:.3f} "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append((seed, res))
    if fails:
        raise AssertionError(f"Toy 1B Line: {len(fails)}/3 seeds reported helix/circle (no-detection expected)")


def toy_2B_circle() -> None:
    """A pure 2D circle randomly embedded in 9D distributes power across all 9 coords.
    Top-2 by Fourier power capture ~40-55% of total (not 60-80% — that would require
    the circle to be aligned with basis coords). The key checks are: verdict=circle,
    P*=10, FCR above the 0.30 threshold, linear_sig=False."""
    print("=== Toy 2B Circle ===")
    fails = []
    for seed in (10, 11, 12):
        rng = np.random.default_rng(seed)
        Z, labels = _make_circle(N=400, K=10, d=TOY_D, rng=rng, noise=0.05)
        res = _run(Z, labels, K_natural=10, seed=seed)
        ok = (res["geometry_detected"] == VERDICT_CIRCLE
              and res["fcr_helix"] >= 0.30
              and res["fcr_two_axis"] >= 0.30
              and not res["linear_significant"]
              and res["two_axis_significant"]
              and abs(res["discovered_period"] - 10.0) <= 1.0)
        print(f"  seed={seed}: geom={res['geometry_detected']}, fcr_two_axis={res['fcr_two_axis']:.3f}, "
              f"fcr_helix={res['fcr_helix']:.3f}, P*={res['discovered_period']:.2f}, "
              f"two_axis_sig={res['two_axis_significant']}, linear_sig={res['linear_significant']} "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append((seed, res))
    if fails:
        raise AssertionError(f"Toy 2B Circle: {len(fails)}/3 seeds failed (expect circle, P=10, "
                              f"FCR≥0.30, linear_sig=False)")


def toy_3B_helix() -> None:
    """A circular helix randomly embedded in 9D. The off-plane linear residual contains
    the pitch direction; linear_sig should be True. Verdict=helix, P*=10."""
    print("=== Toy 3B Helix ===")
    fails = []
    for seed in (20, 21, 22):
        rng = np.random.default_rng(seed)
        Z, labels = _make_helix(N=400, K=10, d=TOY_D, rng=rng, pitch=1.0, noise=0.05)
        res = _run(Z, labels, K_natural=10, seed=seed)
        ok = (res["geometry_detected"] == VERDICT_HELIX
              and res["fcr_helix"] >= 0.30
              and res["linear_significant"]
              and res["two_axis_significant"]
              and abs(res["discovered_period"] - 10.0) <= 1.0)
        print(f"  seed={seed}: geom={res['geometry_detected']}, fcr_two_axis={res['fcr_two_axis']:.3f}, "
              f"fcr_helix={res['fcr_helix']:.3f}, P*={res['discovered_period']:.2f}, "
              f"linear_sig={res['linear_significant']} {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append((seed, res))
    if fails:
        raise AssertionError(f"Toy 3B Helix: {len(fails)}/3 seeds failed (expect helix, P=10, "
                              f"FCR≥0.30, linear_sig=True)")


def toy_4B_isotropic() -> None:
    """Pure noise → any no-detection verdict passes (none / period_inconsistent / sparse)."""
    print("=== Toy 4B Isotropic Gaussian ===")
    fails = []
    for seed in (30, 31, 32):
        rng = np.random.default_rng(seed)
        Z, labels = _make_isotropic(N=400, K=10, d=TOY_D, rng=rng)
        res = _run(Z, labels, K_natural=10, seed=seed)
        ok = res["geometry_detected"] in NO_DETECTION
        print(f"  seed={seed}: geom={res['geometry_detected']}, fcr_helix={res['fcr_helix']:.3f}, "
              f"P*={res['discovered_period']:.2f}, p_helix={res['p_helix']:.3f} "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append((seed, res))
    if fails:
        raise AssertionError(f"Toy 4B Isotropic: {len(fails)}/3 seeds reported helix/circle (no-detection expected)")


def toy_5B_period7() -> None:
    print("=== Toy 5B Period-7 circle (verifies no bias toward P=10) ===")
    fails = []
    for seed in (40, 41, 42):
        rng = np.random.default_rng(seed)
        Z, labels = _make_circle(N=350, K=7, d=TOY_D, rng=rng, noise=0.05)
        res = _run(Z, labels, K_natural=7, seed=seed)
        ok = (res["geometry_detected"] == VERDICT_CIRCLE
              and res["fcr_helix"] >= 0.30
              and not res["linear_significant"]
              and res["two_axis_significant"]
              and abs(res["discovered_period"] - 7.0) <= 1.0)
        print(f"  seed={seed}: geom={res['geometry_detected']}, fcr_helix={res['fcr_helix']:.3f}, "
              f"P*={res['discovered_period']:.2f}, two_axis_sig={res['two_axis_significant']}, "
              f"linear_sig={res['linear_significant']} {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append((seed, res))
    if fails:
        raise AssertionError(f"Toy 5B Period-7: {len(fails)}/3 seeds failed (expect circle, P=7, FCR≥0.30)")


def toy_6B_period13() -> None:
    print("=== Toy 6B Period-13 helix (prime period) ===")
    fails = []
    for seed in (50, 51, 52):
        rng = np.random.default_rng(seed)
        Z, labels = _make_helix(N=520, K=13, d=TOY_D, rng=rng, pitch=1.0, noise=0.05)
        res = _run(Z, labels, K_natural=13, seed=seed)
        ok = (res["geometry_detected"] == VERDICT_HELIX
              and res["linear_significant"]
              and res["two_axis_significant"]
              and abs(res["discovered_period"] - 13.0) <= 1.0)
        print(f"  seed={seed}: geom={res['geometry_detected']}, fcr_helix={res['fcr_helix']:.3f}, "
              f"P*={res['discovered_period']:.2f}, linear_sig={res['linear_significant']} "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append((seed, res))
    if fails:
        raise AssertionError(f"Toy 6B Period-13: {len(fails)}/3 seeds failed (expect helix, P=13)")


def toy_7B_aliased_noise_FPR() -> None:
    """100 cells of pure isotropic Gaussian. With α=0.05, expected detection ~5/100;
    binomial 95% upper bound is 9. Threshold > 9 → fail."""
    print("=== Toy 7B Aliased-noise FPR calibration (100 cells) ===")
    n_cells = 100
    detections = 0
    t0 = time.time()
    for cell_seed in range(60, 60 + n_cells):
        rng = np.random.default_rng(cell_seed)
        Z, labels = _make_isotropic(N=400, K=10, d=TOY_D, rng=rng)
        res = _run(Z, labels, K_natural=10, seed=cell_seed)
        if res["geometry_detected"] in (VERDICT_HELIX, VERDICT_CIRCLE):
            detections += 1
    dt = time.time() - t0
    print(f"  detections = {detections}/{n_cells} (expected ≈ 5; binomial 95% upper bound = 9); time={dt:.1f}s")
    if detections > 9:
        raise AssertionError(
            f"Toy 7B FPR: {detections}/100 detections > 9 — Whittle null mis-calibrated."
        )


def main() -> None:
    t_start = time.time()
    toy_1B_line()
    toy_2B_circle()
    toy_3B_helix()
    toy_4B_isotropic()
    toy_5B_period7()
    toy_6B_period13()
    toy_7B_aliased_noise_FPR()
    print(f"\nALL 7 TOYS PASSED in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nFAIL: {e}", file=sys.stderr)
        sys.exit(1)
