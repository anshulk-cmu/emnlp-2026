"""Step 9 — Johnson-Lindenstrauss distance preservation (port of arithmetic-geometry Phase JL).

For each (model, task, mode, layer) cell, for each Step-7 union variant
{merged, generous}:

  - Project X onto V_all: X_proj = (X @ V.T) @ V on GPU.
  - Compute pairwise distances over ALL N(N-1)/2 unordered pairs:
       d_full[i,j] = ||X_i − X_j||
       d_proj[i,j] = ||X_proj_i − X_proj_j||
  - Report Spearman ρ, Pearson r, mean and max relative error,
    distance variance explained.
  - Full-pair Pythagorean validation in float64 on GPU:
       check that ||X_i − X_j||² ≈ ||X_proj_i − X_proj_j||² + ||X_resid_i − X_resid_j||²
    aggregate max_rel_error and violations(>1e-6).

  - No subsampling. All N(N−1)/2 pairs participate in every metric.

Resume via metadata.json. Atomic writes.

Dependency: Step 7 (residual_hunting.py) must have run first — we load
results/residual_hunting/{model}/{task}/mode_{mode}/layer_LL/union_basis_{variant}.npy.

Usage:
  python jl_distance.py --config /home/anshulk/emnlp2026/config.yaml \
      --model gpt-j-6b --task addition --mode off --all-layers
"""

import argparse
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
from scipy.stats import spearmanr

try:
    import cupy as cp
    _HAS_CUPY = cp.cuda.is_available()
except Exception:
    _HAS_CUPY = False
    cp = None

from residual_hunting import atomic_json, atomic_save


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

PAIR_BATCH = 200_000              # pairs per GPU batch (balances VRAM and overhead)
SMALL_N_SAVE_FULL = 5000          # save full d_full/d_proj arrays only for N ≤ this
PLOT_SUBSAMPLE_PAIRS = 10_000     # random subsample retained for plotting on large-N cells
PYTH_TOLERANCE = 1e-6
VARIANTS = ("merged", "generous")


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(logs_root: Path, model_key: str, task: str, mode: str) -> logging.Logger:
    logs_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"jl_distance.{model_key}.{task}.{mode}")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = WatchedFileHandler(logs_root / f"jl_distance_{model_key}_{task}_{mode}.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


# ──────────────────────────────────────────────────────────────────────────────
# Pair-indices generator
# ──────────────────────────────────────────────────────────────────────────────

def all_pair_indices(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (ii, jj) for all upper-triangle unordered pairs i<j."""
    ii, jj = np.triu_indices(N, k=1)
    return ii.astype(np.int64), jj.astype(np.int64)


# ──────────────────────────────────────────────────────────────────────────────
# Pairwise distances in batches on GPU (float32 main, float64 Pythagorean)
# ──────────────────────────────────────────────────────────────────────────────

def compute_pairwise_distances_gpu(
    X: np.ndarray, X_proj: np.ndarray, ii: np.ndarray, jj: np.ndarray,
    batch: int = PAIR_BATCH,
    logger: logging.Logger | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute d_full and d_proj for all pairs (ii[k], jj[k]).

    Float32 main. GPU when available.
    Returns d_full, d_proj — float32 arrays of length len(ii).
    """
    N_pairs = len(ii)
    d_full = np.empty(N_pairs, dtype=np.float32)
    d_proj = np.empty(N_pairs, dtype=np.float32)

    if _HAS_CUPY:
        X_g = cp.asarray(X)
        Xp_g = cp.asarray(X_proj)
        ii_g = cp.asarray(ii)
        jj_g = cp.asarray(jj)
        for k0 in range(0, N_pairs, batch):
            k1 = min(k0 + batch, N_pairs)
            i_b = ii_g[k0:k1]; j_b = jj_g[k0:k1]
            dif_full = X_g[i_b] - X_g[j_b]
            dif_proj = Xp_g[i_b] - Xp_g[j_b]
            d_full[k0:k1] = cp.asnumpy(cp.linalg.norm(dif_full, axis=1))
            d_proj[k0:k1] = cp.asnumpy(cp.linalg.norm(dif_proj, axis=1))
            if logger and (k0 // batch) % 50 == 0:
                logger.debug(f"  pairwise dist: {k1}/{N_pairs}")
        del X_g, Xp_g, ii_g, jj_g
        cp.get_default_memory_pool().free_all_blocks()
    else:
        # CPU fallback (slow on large N; included for robustness)
        for k0 in range(0, N_pairs, batch):
            k1 = min(k0 + batch, N_pairs)
            i_b = ii[k0:k1]; j_b = jj[k0:k1]
            d_full[k0:k1] = np.linalg.norm(X[i_b] - X[j_b], axis=1)
            d_proj[k0:k1] = np.linalg.norm(X_proj[i_b] - X_proj[j_b], axis=1)
    return d_full, d_proj


def pythagorean_check_full_gpu(
    X: np.ndarray, X_proj: np.ndarray, X_resid: np.ndarray,
    ii: np.ndarray, jj: np.ndarray,
    batch: int = PAIR_BATCH,
    logger: logging.Logger | None = None,
) -> dict:
    """For ALL pairs (no subsampling), in float64 on GPU, verify
    ||X_i − X_j||² ≈ ||X_proj_i − X_proj_j||² + ||X_resid_i − X_resid_j||².

    Returns max/mean relative error and the count of pairs above PYTH_TOLERANCE.
    """
    N_pairs = len(ii)
    max_rel_err = 0.0
    sum_rel_err = 0.0
    n_violations = 0
    if _HAS_CUPY:
        X_g = cp.asarray(X, dtype=cp.float64)
        Xp_g = cp.asarray(X_proj, dtype=cp.float64)
        Xr_g = cp.asarray(X_resid, dtype=cp.float64)
        ii_g = cp.asarray(ii)
        jj_g = cp.asarray(jj)
        for k0 in range(0, N_pairs, batch):
            k1 = min(k0 + batch, N_pairs)
            i_b = ii_g[k0:k1]; j_b = jj_g[k0:k1]
            df = X_g[i_b] - X_g[j_b]
            dp = Xp_g[i_b] - Xp_g[j_b]
            dr = Xr_g[i_b] - Xr_g[j_b]
            d_full_sq = cp.sum(df * df, axis=1)
            d_proj_sq = cp.sum(dp * dp, axis=1)
            d_resid_sq = cp.sum(dr * dr, axis=1)
            safe_denom = cp.maximum(d_full_sq, cp.asarray(1e-20, dtype=cp.float64))
            rel_err = cp.abs(d_full_sq - d_proj_sq - d_resid_sq) / safe_denom
            rel_err = cp.where(d_full_sq > 1e-20, rel_err, cp.zeros_like(rel_err))
            chunk_max = float(cp.max(rel_err))
            chunk_sum = float(cp.sum(rel_err))
            chunk_vio = int(cp.sum(rel_err > PYTH_TOLERANCE))
            max_rel_err = max(max_rel_err, chunk_max)
            sum_rel_err += chunk_sum
            n_violations += chunk_vio
            if logger and (k0 // batch) % 50 == 0:
                logger.debug(f"  pyth check: {k1}/{N_pairs} max_rel_err={max_rel_err:.2e}")
        del X_g, Xp_g, Xr_g, ii_g, jj_g
        cp.get_default_memory_pool().free_all_blocks()
    else:
        for k0 in range(0, N_pairs, batch):
            k1 = min(k0 + batch, N_pairs)
            i_b = ii[k0:k1]; j_b = jj[k0:k1]
            df = (X[i_b] - X[j_b]).astype(np.float64)
            dp = (X_proj[i_b] - X_proj[j_b]).astype(np.float64)
            dr = (X[i_b] - X[j_b] - dp).astype(np.float64)  # construct resid from full - proj
            d_full_sq = np.sum(df * df, axis=1)
            d_proj_sq = np.sum(dp * dp, axis=1)
            d_resid_sq = np.sum(dr * dr, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_err = np.where(d_full_sq > 1e-20,
                                   np.abs(d_full_sq - d_proj_sq - d_resid_sq) / d_full_sq, 0.0)
            max_rel_err = max(max_rel_err, float(np.max(rel_err)))
            sum_rel_err += float(np.sum(rel_err))
            n_violations += int(np.sum(rel_err > PYTH_TOLERANCE))
    return {
        "pyth_max_rel_error": float(max_rel_err),
        "pyth_mean_rel_error": float(sum_rel_err / max(N_pairs, 1)),
        "pyth_n_violations": int(n_violations),
        "pyth_n_pairs": int(N_pairs),
        "pyth_tolerance": float(PYTH_TOLERANCE),
        "pyth_dtype": "float64",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Metrics on the full distance arrays
# ──────────────────────────────────────────────────────────────────────────────

def compute_jl_metrics(d_full: np.ndarray, d_proj: np.ndarray) -> dict:
    """Spearman, Pearson, mean/max relative error, distance variance explained.

    Uses scipy.spearmanr; on N_pairs > ~80M switch to chunked rank.
    """
    N_pairs = len(d_full)

    # Pearson via vectorised numpy
    df64 = d_full.astype(np.float64)
    dp64 = d_proj.astype(np.float64)
    df_mean = df64.mean(); dp_mean = dp64.mean()
    dfc = df64 - df_mean; dpc = dp64 - dp_mean
    pearson_r = float((dfc @ dpc) / (np.sqrt((dfc * dfc).sum()) * np.sqrt((dpc * dpc).sum()) + 1e-30))

    # Spearman
    if N_pairs <= 80_000_000:
        # scipy is memory-efficient enough; we have N_pairs ≤ 50M typically.
        rho_s, _ = spearmanr(df64, dp64)
        spearman_rho = float(rho_s)
    else:
        # Chunked rank fallback (defensive; not exercised at our N).
        from scipy.stats import rankdata
        r_full = rankdata(df64, method="average").astype(np.float64)
        r_proj = rankdata(dp64, method="average").astype(np.float64)
        r_full -= r_full.mean(); r_proj -= r_proj.mean()
        spearman_rho = float((r_full @ r_proj)
                             / (np.sqrt((r_full * r_full).sum()) * np.sqrt((r_proj * r_proj).sum()) + 1e-30))

    # Relative errors
    rel = np.where(df64 > 1e-20, np.abs(df64 - dp64) / df64, 0.0)
    mean_rel = float(rel.mean())
    max_rel = float(rel.max())

    # Distance variance explained: 1 - var(d_full² − d_proj²) / var(d_full²)
    sq_full = df64 * df64
    sq_proj = dp64 * dp64
    var_diff = float(np.var(sq_full - sq_proj))
    var_full = float(np.var(sq_full))
    dvar_explained = 1.0 - var_diff / max(var_full, 1e-30)

    return {
        "spearman_rho": spearman_rho,
        "pearson_r": pearson_r,
        "mean_rel_error": mean_rel,
        "max_rel_error": max_rel,
        "distance_var_explained": float(dvar_explained),
        "n_pairs": int(N_pairs),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-cell runner
# ──────────────────────────────────────────────────────────────────────────────

def load_union_basis(results_root: Path, model: str, task: str, mode: str, layer: int, variant: str) -> np.ndarray | None:
    p = (results_root / "residual_hunting" / model / task / f"mode_{mode}"
         / f"layer_{layer:02d}" / f"union_basis_{variant}.npy")
    if not p.exists():
        return None
    return np.load(p).astype(np.float32)


def project_full(X: np.ndarray, V_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_proj, X_resid) as float32 arrays."""
    if _HAS_CUPY and V_all.shape[0] > 0:
        X_g = cp.asarray(X)
        V_g = cp.asarray(V_all)
        coords = X_g @ V_g.T
        Xp = coords @ V_g
        Xr = X_g - Xp
        out_p = cp.asnumpy(Xp)
        out_r = cp.asnumpy(Xr)
        del X_g, V_g, coords, Xp, Xr
        cp.get_default_memory_pool().free_all_blocks()
        return out_p, out_r
    if V_all.shape[0] == 0:
        return np.zeros_like(X), X.copy()
    coords = X @ V_all.T
    Xp = coords @ V_all
    Xr = X - Xp
    return Xp.astype(np.float32), Xr.astype(np.float32)


def run_cell(
    cfg: dict, model: str, task: str, mode: str, layer: int,
    X_correct: np.ndarray, logger: logging.Logger,
) -> list[dict]:
    results_root = Path(cfg["paths"]["results_root"])
    cell_dir = results_root / "jl_distance" / model / task / f"mode_{mode}" / f"layer_{layer:02d}"
    meta_path = cell_dir / "metadata.json"
    if meta_path.exists():
        try:
            cached = json.loads(meta_path.read_text())
            if cached.get("computation_status") == "complete":
                logger.info(f"[skip] cached {model}/{task}/mode_{mode}/layer_{layer:02d}")
                return cached.get("summary_rows", [])
        except Exception:
            pass

    t0 = time.time()
    cell_dir.mkdir(parents=True, exist_ok=True)

    N = X_correct.shape[0]
    ii, jj = all_pair_indices(N)
    n_pairs = len(ii)
    logger.info(f"[run] {model}/{task}/mode_{mode}/layer_{layer:02d}: N={N}, n_pairs={n_pairs}")

    summary_rows = []
    rng = np.random.default_rng(seed=42)
    plot_idx = None
    if N > SMALL_N_SAVE_FULL:
        plot_idx = rng.choice(n_pairs, size=min(PLOT_SUBSAMPLE_PAIRS, n_pairs), replace=False)
        plot_idx.sort()

    for variant in VARIANTS:
        V_all = load_union_basis(results_root, model, task, mode, layer, variant)
        if V_all is None:
            logger.warning(f"  union_basis_{variant} not found — skipping")
            summary_rows.append({
                "model": model, "task": task, "mode": mode, "layer": int(layer),
                "variant": variant, "status": "missing_union",
            })
            continue
        t_v = time.time()
        X_proj, X_resid = project_full(X_correct, V_all)

        # Pairwise distances (float32 main metric)
        d_full, d_proj = compute_pairwise_distances_gpu(X_correct, X_proj, ii, jj, logger=logger)
        metrics = compute_jl_metrics(d_full, d_proj)

        # Full-pair Pythagorean check (float64)
        pyth = pythagorean_check_full_gpu(X_correct, X_proj, X_resid, ii, jj, logger=logger)

        # Persist
        out_path = cell_dir / f"jl_metrics_{variant}.json"
        atomic_json({
            "variant": variant,
            "k_union": int(V_all.shape[0]),
            "N": int(N),
            **metrics,
            **pyth,
            "runtime_seconds": round(time.time() - t_v, 2),
        }, out_path)

        # Histogram of (d_full, d_proj) 2-D for plotting later
        hist_bins = 200
        hi = float(max(d_full.max(), d_proj.max()))
        lo = 0.0
        H, xe, ye = np.histogram2d(d_full, d_proj, bins=hist_bins, range=[[lo, hi], [lo, hi]])
        np.savez_compressed(cell_dir / f"d_hist_{variant}.npz",
                            H=H.astype(np.int64), x_edges=xe.astype(np.float32), y_edges=ye.astype(np.float32))

        # Save full or subsampled distance arrays
        if N <= SMALL_N_SAVE_FULL:
            atomic_save(d_full, cell_dir / f"d_full_{variant}.npy")
            atomic_save(d_proj, cell_dir / f"d_proj_{variant}.npy")
        else:
            atomic_save(d_full[plot_idx], cell_dir / f"d_full_sample_{variant}.npy")
            atomic_save(d_proj[plot_idx], cell_dir / f"d_proj_sample_{variant}.npy")

        summary_rows.append({
            "model": model, "task": task, "mode": mode, "layer": int(layer),
            "variant": variant,
            "k_union": int(V_all.shape[0]),
            "N": int(N),
            "n_pairs": int(n_pairs),
            **metrics,
            "pyth_max_rel_error": float(pyth["pyth_max_rel_error"]),
            "pyth_mean_rel_error": float(pyth["pyth_mean_rel_error"]),
            "pyth_n_violations": int(pyth["pyth_n_violations"]),
            "runtime_seconds": round(time.time() - t_v, 2),
            "status": "ok",
        })
        logger.info(f"  {variant}: k={V_all.shape[0]} ρ_s={metrics['spearman_rho']:.4f} "
                    f"r_p={metrics['pearson_r']:.4f} mean_rel={metrics['mean_rel_error']:.3f} "
                    f"d_var_expl={metrics['distance_var_explained']:.4f} "
                    f"pyth_max_rel_err={pyth['pyth_max_rel_error']:.2e} "
                    f"in {time.time()-t_v:.1f}s")

    atomic_json({
        "model": model, "task": task, "mode": mode, "layer": int(layer),
        "computation_status": "complete",
        "summary_rows": summary_rows,
        "runtime_seconds": round(time.time() - t0, 2),
    }, meta_path)
    return summary_rows


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=["addition", "multiplication"])
    ap.add_argument("--mode", required=True, choices=["off", "answer", "norm"])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--layer", type=int)
    g.add_argument("--all-layers", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    paths = cfg["paths"]
    logs_root = Path(paths["logs_root"])
    results_root = Path(paths["results_root"])
    data_root = Path(paths["data_root"])
    model_cfg = next(m for m in cfg["models"] if m["key"] == args.model)
    layers = [args.layer] if args.layer is not None else model_cfg["layers"]

    logger = setup_logging(logs_root, args.model, args.task, args.mode)
    logger.info(f"=== Step 9 jl_distance: {args.model}/{args.task}/{args.mode} layers={layers} ===")
    logger.info(f"cupy: {'AVAILABLE' if _HAS_CUPY else 'NOT available'}")

    answers_path = data_root / "answers" / args.model / f"{args.task}_answers.csv"
    correct_mask = pd.read_csv(answers_path)["correct"].to_numpy().astype(bool)

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
        if args.mode == "off":
            X_correct = np.ascontiguousarray(X_full[correct_mask].astype(np.float32))
        else:
            X_correct = np.ascontiguousarray(X_full.astype(np.float32))
        del X_full
        rows = run_cell(cfg, args.model, args.task, args.mode, layer, X_correct, logger)
        all_rows.extend(rows)

    if all_rows:
        per_model_dir = results_root / "jl_distance" / args.model
        per_model_dir.mkdir(parents=True, exist_ok=True)
        out_csv = per_model_dir / f"summary_{args.model}_{args.task}_mode_{args.mode}.csv"
        df_new = pd.DataFrame(all_rows)
        if out_csv.exists():
            df_old = pd.read_csv(out_csv)
            key = ["model", "task", "mode", "layer", "variant"]
            df_old = df_old[~df_old.set_index(key).index.isin(df_new.set_index(key).index)]
            df_new = pd.concat([df_old, df_new], ignore_index=True)
        df_new = df_new.sort_values(["task", "mode", "layer", "variant"]).reset_index(drop=True)
        df_new.to_csv(out_csv, index=False)
        logger.info(f"wrote {out_csv} ({len(df_new)} rows)")

    logger.info(f"=== Step 9 DONE ===")


if __name__ == "__main__":
    main()
