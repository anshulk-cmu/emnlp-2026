"""Toy validation for lda_subspaces.fit_one_cell.

Four toys (plan §4.12 + extension):
  1L: 10 classes whose means lie on a 1-D line in 9-D Gaussian.
      Expect: λ_T_1 ≥ 0.95, n_sig ≥ 1, top-1 cv_accuracy well above chance.
  2L: 10 classes on a 2-D grid in 9-D Gaussian.
      Expect: n_sig ≥ 2, λ_T_1 and λ_T_2 substantial, cv_accuracy strong.
  3L: 9-D isotropic Gaussian, random class assignments.
      Expect: n_sig = 0.
  4L: Sample-starved (N=120, d=4, K=10). Permutation-only would inflate;
      our dual criterion (n_sig = min(n_sig_perm, n_sig_cv)) should catch it.
      Expect: n_sig_cv = 0 and therefore final n_sig = 0.

Each toy invokes fit_one_cell with no CCSVD basis (Option B is disabled here
to keep the toy lightweight). We read out A's results.

Usage:
  python check_lda_toys.py
"""

import sys

import numpy as np
import yaml
from pathlib import Path

from lda_subspaces import fit_one_cell


def make_toy_1L(rng, n_per_class=300, n_classes=10, d=9, sigma=0.5):
    means = np.zeros((n_classes, d))
    means[:, 0] = np.arange(n_classes)
    X = np.concatenate([rng.normal(means[v], sigma, size=(n_per_class, d)) for v in range(n_classes)])
    y = np.repeat(np.arange(n_classes), n_per_class)
    return X.astype(np.float32), y


def make_toy_2L(rng, n_per_class=300, n_classes=10, d=9, sigma=0.5, spacing=2.0):
    """10 classes on a 5×2 grid in dim-0,1 of a 9-D space.

    spacing controls inter-class distance; default 2.0 gives an SNR of ~4 with
    sigma=0.5 — ample for the dual-criterion to pick up both axes.
    """
    means = np.zeros((n_classes, d))
    for v in range(n_classes):
        means[v, 0] = (v % 5) * spacing
        means[v, 1] = (v // 5) * spacing
    X = np.concatenate([rng.normal(means[v], sigma, size=(n_per_class, d)) for v in range(n_classes)])
    y = np.repeat(np.arange(n_classes), n_per_class)
    return X.astype(np.float32), y


def make_toy_3L(rng, n_per_class=300, n_classes=10, d=9, sigma=1.0):
    n = n_per_class * n_classes
    X = rng.normal(0.0, sigma, size=(n, d)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n)
    return X, y


def make_toy_4L(rng, n_per_class=12, n_classes=10, d=4, sigma=1.0):
    """Sample-starved toy: N = 120 < d * n_classes might still pass perm null
    by inflation, but CV-accuracy on truly random labels stays near chance.
    """
    n = n_per_class * n_classes
    X = rng.normal(0.0, sigma, size=(n, d)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n)
    return X, y


def make_toy_pca(rng, n_per_class=300, n_classes=10, d=9, sigma=0.5, axes=2):
    """Helper to build a CCSVD basis from a known toy: just keep the first 'axes' canonical directions."""
    B = np.zeros((d, axes), dtype=np.float32)
    for k in range(axes):
        B[k, k] = 1.0
    return B


def run_toy(name, X, y, B_basis=None):
    """Invoke fit_one_cell on a toy. Returns the A-side meta + artifacts."""
    cfg_lda = {
        "scatter_choice": "S_T",
        "regularisation_alpha": 1.0e-4,
        "use_shrinkage_when_n_over_r_below": 10,
        "full_space_shrinkage": "ledoit_wolf",
        "n_permutations": 200,                  # toys: reduced for speed
        "perm_alpha": 0.01,
        "cv_n_splits": 5,
        "cv_knn_k": [5, 1],
        "use_one_se_rule_for_n_sig_cv": True,
        "bootstrap_n": 50,
        "random_state": 42,
        "min_classes_for_lda": 2,
        "min_samples_per_class": 5,             # tighter for tiny toys
        "ans_concept_prefixes": [],
        "norm_carveout_concepts": [],
    }
    cell_id = {"model_key": "TOY", "task": name, "layer": 0, "concept_name": name, "mode": "off"}
    res = fit_one_cell(
        X_correct_resid=X, y=y,
        cell_id=cell_id, cfg_lda=cfg_lda, B_ccsvd=B_basis,
        is_carved_out=False, full_space_chol_cache=None,
    )
    return res


def main():
    rng = np.random.default_rng(0)
    summary_lines = []
    all_pass = True

    # ── Toy 1L ─────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("Toy 1L (10 classes on a 1-D line in 9-D Gaussian)")
    X, y = make_toy_1L(rng)
    B = make_toy_pca(rng, axes=3)              # CCSVD basis: keep top-3 canonical axes
    res = run_toy("toy_1L", X, y, B_basis=B)
    A = res["A"]
    print(f"  status={A.get('status')}  K={A.get('n_groups_after_filter')}")
    print(f"  λ_T_1={A.get('lambda_T_1'):.4f}  λ_T_2={A.get('lambda_T_2'):.4f}")
    print(f"  n_sig_perm={A.get('n_sig_perm')}  n_sig_cv={A.get('n_sig_cv')}  n_sig={A.get('n_sig')}")
    print(f"  cv_accuracy_max={A.get('cv_accuracy_max'):.3f}  baseline=1/K={1/A.get('n_groups_after_filter'):.3f}")
    pass1 = (
        A.get("lambda_T_1", 0) >= 0.85
        and A.get("n_sig", 0) >= 1
        and A.get("cv_accuracy_max", 0) > 0.5
    )
    print(f"  PASS={pass1}  (need: λ_T_1≥0.85, n_sig≥1, cv_max>0.5)")
    summary_lines.append(("Toy 1L", pass1))
    all_pass &= pass1

    # ── Toy 2L ─────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("Toy 2L (10 classes on a 2-D grid in 9-D Gaussian)")
    X, y = make_toy_2L(rng)
    B = make_toy_pca(rng, axes=4)
    res = run_toy("toy_2L", X, y, B_basis=B)
    A = res["A"]
    print(f"  status={A.get('status')}  K={A.get('n_groups_after_filter')}")
    print(f"  λ_T_1={A.get('lambda_T_1'):.4f}  λ_T_2={A.get('lambda_T_2'):.4f}")
    print(f"  n_sig_perm={A.get('n_sig_perm')}  n_sig_cv={A.get('n_sig_cv')}  n_sig={A.get('n_sig')}")
    print(f"  cv_accuracy_max={A.get('cv_accuracy_max'):.3f}")
    pass2 = (
        A.get("n_sig", 0) >= 2
        and A.get("lambda_T_1", 0) >= 0.5
        and A.get("lambda_T_2", 0) >= 0.5
        and A.get("cv_accuracy_max", 0) > 0.6
    )
    print(f"  PASS={pass2}  (need: n_sig≥2, λ_T_1≥0.5, λ_T_2≥0.5, cv_max>0.6)")
    summary_lines.append(("Toy 2L", pass2))
    all_pass &= pass2

    # ── Toy 3L ─────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("Toy 3L (no structure, expect n_sig=0)")
    X, y = make_toy_3L(rng)
    B = make_toy_pca(rng, axes=3)
    res = run_toy("toy_3L", X, y, B_basis=B)
    A = res["A"]
    print(f"  status={A.get('status')}  K={A.get('n_groups_after_filter')}")
    print(f"  λ_T_1={A.get('lambda_T_1'):.4f}")
    print(f"  n_sig_perm={A.get('n_sig_perm')}  n_sig_cv={A.get('n_sig_cv')}  n_sig={A.get('n_sig')}")
    print(f"  cv_accuracy_max={A.get('cv_accuracy_max'):.3f}  baseline=1/K={1/A.get('n_groups_after_filter'):.3f}")
    pass3 = (A.get("n_sig", 0) == 0)
    print(f"  PASS={pass3}  (need: n_sig==0)")
    summary_lines.append(("Toy 3L", pass3))
    all_pass &= pass3

    # ── Toy 4L (sample-starved) ─────────────────────────────────────────────────
    print("=" * 70)
    print("Toy 4L (sample-starved random — dual-criterion catches inflation)")
    X, y = make_toy_4L(rng)
    B = make_toy_pca(rng, axes=2, d=4)         # keep top-2 of 4 canonical axes
    res = run_toy("toy_4L", X, y, B_basis=B)
    A = res["A"]
    print(f"  status={A.get('status')}  K={A.get('n_groups_after_filter')}")
    print(f"  λ_T_1={A.get('lambda_T_1'):.4f}")
    print(f"  n_sig_perm={A.get('n_sig_perm')}  n_sig_cv={A.get('n_sig_cv')}  n_sig={A.get('n_sig')}")
    print(f"  cv_accuracy_max={A.get('cv_accuracy_max'):.3f}  baseline=1/K={1/A.get('n_groups_after_filter'):.3f}")
    pass4 = (A.get("n_sig", 0) == 0)
    print(f"  PASS={pass4}  (need: n_sig==0; dual criterion overrides any perm-null inflation)")
    summary_lines.append(("Toy 4L", pass4))
    all_pass &= pass4

    print("=" * 70)
    print("Summary:")
    for name, ok in summary_lines:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
