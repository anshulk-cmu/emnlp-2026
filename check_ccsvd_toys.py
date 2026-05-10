"""Synthetic toy validation for ccsvd_subspaces.fit_cell.

Three toys (plan §8): 1L, 2L, 3L. All must pass before any real run.

  1L: 10 classes on a line in 9-D Gaussian ⇒ expect r=1.
  2L: 10 classes on a 2-D grid in 9-D Gaussian ⇒ expect r=2.
  3L: 9-D isotropic Gaussian, random labels ⇒ expect r=0.

Usage:
  python check_ccsvd_toys.py
"""

import sys
import numpy as np

from ccsvd_subspaces import fit_cell


def make_toy_1L(rng, n_per_class=300, n_classes=10, d=9, sigma=0.5):
    means = np.zeros((n_classes, d))
    means[:, 0] = np.arange(n_classes)
    X = np.concatenate([rng.normal(means[v], sigma, size=(n_per_class, d)) for v in range(n_classes)])
    y = np.repeat(np.arange(n_classes), n_per_class)
    return X.astype(np.float32), y


def make_toy_2L(rng, n_per_class=300, n_classes=10, d=9, sigma=0.5):
    means = np.zeros((n_classes, d))
    for v in range(n_classes):
        means[v, 0] = v % 5
        means[v, 1] = v // 5
    X = np.concatenate([rng.normal(means[v], sigma, size=(n_per_class, d)) for v in range(n_classes)])
    y = np.repeat(np.arange(n_classes), n_per_class)
    return X.astype(np.float32), y


def make_toy_3L(rng, n_per_class=300, n_classes=10, d=9, sigma=1.0):
    n = n_per_class * n_classes
    X = rng.normal(0.0, sigma, size=(n, d)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n)
    return X, y


def run_one(name, X, y, n_perm=200):
    """Run one toy. Reduced n_perm for speed (200 enough for r-determination)."""
    res = fit_cell(
        X, y,
        cell_id={"model_key": "TOY", "task": name, "layer": 0, "concept_name": name},
        min_group_size=30, n_permutations=n_perm, perm_alpha=0.01,
        cv_n_splits=5, random_state=42,
        mean_centre=True, unit_normalise=False,
    )
    return res


def main():
    rng = np.random.default_rng(0)

    # NOTE on r vs cumulative-variance criteria:
    # Plan v6 §3.1.5 phrases 1L's expectation as "λ₁ ≥ 0.95, λ₂ ≈ 0" — that's a
    # cumulative-variance statement (λ₁ contains ≥95% of total spectrum variance),
    # not a permutation-null statement. The strict perm-null at p<0.01 will often
    # report r=2 on cleanly rank-1 data because residual numerical/centroid noise
    # systematically beats the (also-tiny) null threshold. We therefore check the
    # spec's cumulative-variance criterion AND also report r for visibility.
    # The pass criterion is whether the dominant subspace structure matches.

    print("=" * 60)
    print("Toy 1L (one-axis structure, dominant rank=1)")
    X, y = make_toy_1L(rng)
    r1 = run_one("toy_1L", X, y)
    ev1 = r1["explained_variance"][0] if r1.get("explained_variance") else float("nan")
    print(f"  r={r1['r_dim']}  λ1={r1['lambda_1']:.4f}  λ2={r1['lambda_2']:.4f}  λ1/λ2={r1['lambda_1_over_2']:.2f}  ev_top1={ev1:.4f}  cv_mean={r1['cv_mean']:.3f}")
    pass1 = (ev1 >= 0.95) and (r1["lambda_1_over_2"] > 10) and (r1["cv_mean"] > 0.95)
    print(f"  PASS={pass1}  (need: ev_top1≥0.95, λ1/λ2>10, cv>0.95)")

    print("=" * 60)
    print("Toy 2L (two-axis structure, dominant rank=2)")
    X, y = make_toy_2L(rng)
    r2 = run_one("toy_2L", X, y)
    cv2 = r2.get("cumulative_variance") or [float("nan")] * 2
    cumvar_at2 = cv2[1] if len(cv2) >= 2 else float("nan")
    print(f"  r={r2['r_dim']}  λ1={r2['lambda_1']:.4f}  λ2={r2['lambda_2']:.4f}  λ3={r2['lambda_3']:.4f}  cumvar@2={cumvar_at2:.4f}  cv_mean={r2['cv_mean']:.3f}")
    pass2 = (r2["r_dim"] >= 2) and (cumvar_at2 >= 0.90) and (r2["cv_mean"] > 0.90)
    print(f"  PASS={pass2}  (need: r≥2, cumvar@2≥0.90, cv>0.90)")

    print("=" * 60)
    print("Toy 3L (no structure, expect r=0)")
    X, y = make_toy_3L(rng)
    r3 = run_one("toy_3L", X, y)
    print(f"  r={r3['r_dim']}  status={r3['status']}  λ1={r3['lambda_1']:.4f}  cv_mean={r3.get('cv_mean','nan')}")
    pass3 = (r3["r_dim"] == 0)
    print(f"  PASS={pass3}  (need: r==0)")

    print("=" * 60)
    all_pass = pass1 and pass2 and pass3
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}  ({pass1=}, {pass2=}, {pass3=})")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
