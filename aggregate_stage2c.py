#!/usr/bin/env python3
"""
Stage 2c aggregator — concatenate every per-cell `gplvm_results.csv`, apply
global BH-FDR on the permutation-null p-values, reassign verdicts and tiers
post-FDR, and write the comparison tables.

Mirrors aggregate_stage2b_dsw.py's structure.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

try:
    from scipy.stats import false_discovery_control
except Exception:
    false_discovery_control = None


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
STAGE2C_CONFIG_DEFAULT = PROJECT_ROOT / "configs" / "stage2c.yaml"


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".csv", dir=str(path.parent))
    os.close(fd)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".json", dir=str(path.parent))
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def load_all_summaries(results_root: Path) -> pd.DataFrame:
    """Walk every per-cell gplvm_results.csv and concatenate."""
    root = results_root / "stage2c_gplvm"
    rows = []
    for p in root.rglob("gplvm_results.csv"):
        try:
            df = pd.read_csv(p)
            for _, r in df.iterrows():
                d = r.to_dict()
                d["_source_csv"] = str(p)
                rows.append(d)
        except Exception as e:
            print(f"WARN: failed to read {p}: {e}", file=sys.stderr)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def apply_bh_fdr(df: pd.DataFrame) -> pd.DataFrame:
    """Global Benjamini-Hochberg FDR on p_2c across the eligible cell family.

    Eligible = cells with finite p_2c (i.e. cells where the kernel competition
    produced a winner; low_K / numerical-failure rows get q_2c = NaN).
    """
    if df.empty or "p_2c" not in df.columns:
        df["q_2c"] = float("nan")
        return df
    p = df["p_2c"].astype(float).to_numpy()
    valid = np.isfinite(p) & (p >= 0) & (p <= 1)
    q = np.full_like(p, np.nan, dtype=float)
    if valid.any():
        p_valid = p[valid]
        if false_discovery_control is not None:
            q_valid = false_discovery_control(p_valid, method="bh")
        else:
            # Manual BH fallback
            order = np.argsort(p_valid)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, p_valid.size + 1)
            q_valid_raw = p_valid * p_valid.size / ranks
            # Enforce monotone
            sort_idx = order[::-1]
            running_min = 1.0
            q_valid = np.empty_like(q_valid_raw)
            for i in sort_idx:
                running_min = min(running_min, q_valid_raw[i])
                q_valid[i] = running_min
        q[valid] = q_valid
    df["q_2c"] = q
    return df


VERDICT_POSITIVES = {"helix", "circle", "torus", "concentric",
                      "periodic_smooth", "smooth_only"}


def downgrade_post_fdr(df: pd.DataFrame, fdr_alpha: float = 0.05) -> pd.DataFrame:
    """Cells with q_2c ≥ alpha lose their positive verdict — downgraded to
    `inconclusive`. `dim_only` cells also gated by q_2c; failure modes
    (`low_K`, `inconclusive` already) untouched."""
    if df.empty or "verdict_pre_fdr" not in df.columns:
        df["verdict_post_fdr"] = df.get("verdict_pre_fdr", "inconclusive")
        df["fdr_downgraded"] = False
        return df
    out = df.copy()
    q = out["q_2c"].astype(float)
    pre = out["verdict_pre_fdr"].astype(str)
    post = pre.copy()
    downgraded = pd.Series(False, index=out.index)
    eligible = pre.isin(list(VERDICT_POSITIVES) + ["dim_only"])
    fail_q = eligible & (q.fillna(1.0) >= fdr_alpha)
    post.loc[fail_q] = "inconclusive"
    downgraded.loc[fail_q] = True
    out["verdict_post_fdr"] = post
    out["fdr_downgraded"] = downgraded
    return out


def reassign_tiers_with_q(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Recompute the tier using post-FDR q_2c (the worker used raw p_2c as a
    placeholder). Plan §C.10 ladder, adapted for point-cloud N."""
    if df.empty:
        return df
    out = df.copy()
    q = out["q_2c"].astype(float).fillna(1.0)
    bf_gap = out.get("bf_gap_nats", 0.0).astype(float).fillna(0.0)
    bf_thr = out.get("bf_threshold_nats", 5.0).astype(float).fillna(5.0)
    seed_range = out.get("winner_seed_range_nats", 100.0).astype(float).fillna(100.0)
    N_used = out.get("N_used", 0).astype(float).fillna(0)
    min_n_v = out.get("min_n_v", 0).astype(float).fillna(0)
    k_u = out.get("k_u", 1).astype(float).fillna(1)

    tier = pd.Series("LOW", index=out.index)
    discovery_mask = (k_u / (min_n_v + 1e-9) >= 1.0) | (min_n_v / (k_u + 1e-9) < 2.0)
    high_mask = ((N_used >= 600) & (min_n_v >= 100) & (q < 0.01)
                  & (bf_gap >= 2.0 * bf_thr) & (seed_range < 1.0))
    medium_mask = ((N_used >= 300) & (min_n_v >= 50) & (q < 0.05)
                    & (bf_gap >= bf_thr) & ~high_mask)
    low_mask = (N_used >= 100) & (min_n_v >= 30)
    tier.loc[low_mask] = "LOW"
    tier.loc[medium_mask] = "MEDIUM"
    tier.loc[high_mask] = "HIGH"
    tier.loc[discovery_mask] = "DISCOVERY_ONLY"
    out["tier"] = tier
    return out


def verdict_counts_by_cell(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "verdict_post_fdr" not in df.columns:
        return pd.DataFrame()
    pivot = (df.groupby(["model", "task", "mode", "layer", "verdict_post_fdr"])
                  .size().unstack(fill_value=0).reset_index())
    return pivot


def cross_mode_kernel_survival(df: pd.DataFrame) -> pd.DataFrame:
    """For each (model, task, layer, concept), pivot mode → winner_kernel.
    Adds n_modes_agreeing on winner."""
    if df.empty or "winner_kernel" not in df.columns:
        return pd.DataFrame()
    pivot = (df.pivot_table(index=["model", "task", "layer", "concept"],
                              columns="mode", values="winner_kernel",
                              aggfunc="first")
                  .reset_index())
    if {"off", "answer", "norm"}.issubset(pivot.columns):
        modes = [c for c in ["off", "answer", "norm"] if c in pivot.columns]
        def agree_count(row):
            vals = [row.get(m) for m in modes if isinstance(row.get(m), str)]
            if not vals: return 0
            return max(vals.count(v) for v in set(vals))
        pivot["n_modes_agreeing"] = pivot.apply(agree_count, axis=1)
    return pivot


def kernel_concept_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Counts of (concept, winning kernel) — which concepts go torus, helix..."""
    if df.empty:
        return pd.DataFrame()
    sub = df[df["verdict_post_fdr"].isin(VERDICT_POSITIVES)].copy()
    if sub.empty:
        return pd.DataFrame()
    pivot = (sub.groupby(["concept", "winner_kernel"])
                  .size().unstack(fill_value=0).reset_index())
    return pivot


def dim_only_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["model", "task", "mode", "layer", "concept",
             "N_used", "K_present", "k_u",
             "ard_d_post", "ard_P_d_geq_1", "ard_P_d_geq_2", "ard_P_d_geq_3",
             "d_hat_bootstrap_median", "d_hat_bootstrap_p025", "d_hat_bootstrap_p975",
             "q_2c", "tier"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    sub = df[df["verdict_post_fdr"] == "dim_only"].copy()
    keep = [c for c in cols if c in sub.columns]
    return sub[keep]


def stage2a_2b_2c_survival(df: pd.DataFrame, results_root: Path) -> pd.DataFrame:
    """Headline transition table — joins Stage 2a verdict + Stage 2b verdict +
    Stage 2c verdict for every cell. Requires Stage 2a and 2b aggregated CSVs
    to exist."""
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    # Pull stage 2a verdict from the summary CSVs (one per model × task × mode × variant)
    s2a_dir = results_root / "stage2a_fourier_helix" / "comparison"
    s2a_path = s2a_dir / "fcr_all.csv"
    if s2a_path.exists():
        try:
            s2a = pd.read_csv(s2a_path)
            # take any variant (lda_a preferred); collapse via groupby first
            s2a_red = (s2a.sort_values(["model", "task", "mode", "layer",
                                          "concept", "variant"])
                            .groupby(["model", "task", "mode", "layer", "concept"])
                            .first().reset_index())
            out = out.merge(s2a_red[["model", "task", "mode", "layer",
                                       "concept", "geometry_detected",
                                       "discovered_period"]],
                              on=["model", "task", "mode", "layer", "concept"],
                              how="left", suffixes=("", "_2a"))
            out = out.rename(columns={
                "geometry_detected": "stage2a_verdict",
                "discovered_period": "stage2a_period_2a",
            })
        except Exception as e:
            print(f"WARN: failed to join Stage 2a: {e}", file=sys.stderr)
    s2b_dir = results_root / "stage2b_dsw" / "comparison"
    s2b_path = s2b_dir / "dsw_all.csv"
    if s2b_path.exists():
        try:
            s2b = pd.read_csv(s2b_path)
            s2b_red = (s2b.sort_values(["model", "task", "mode", "layer",
                                          "concept", "variant"])
                            .groupby(["model", "task", "mode", "layer", "concept"])
                            .first().reset_index())
            out = out.merge(s2b_red[["model", "task", "mode", "layer",
                                       "concept", "spread_verdict_post_fdr"]],
                              on=["model", "task", "mode", "layer", "concept"],
                              how="left")
            out = out.rename(columns={
                "spread_verdict_post_fdr": "stage2b_verdict",
            })
        except Exception as e:
            print(f"WARN: failed to join Stage 2b: {e}", file=sys.stderr)
    cols = ["model", "task", "mode", "layer", "concept",
             "stage2a_verdict", "stage2a_period_2a",
             "stage2b_verdict",
             "winner_kernel", "verdict_post_fdr", "tier", "q_2c",
             "N_used", "K_present", "k_u",
             "bf_gap_nats"]
    keep = [c for c in cols if c in out.columns]
    return out[keep]


def headline_tier_cells(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    sub = df[df["tier"].isin(["HIGH", "MEDIUM"])
              & df["verdict_post_fdr"].isin(VERDICT_POSITIVES)]
    return sub.copy()


def build_manifest(df: pd.DataFrame, cfg: dict, runtime_seconds: float,
                    fdr_alpha: float) -> dict:
    n_total = int(len(df))
    pre = df["verdict_pre_fdr"].value_counts().to_dict() if not df.empty else {}
    post = df["verdict_post_fdr"].value_counts().to_dict() if not df.empty else {}
    tier_counts = df["tier"].value_counts().to_dict() if not df.empty else {}
    n_downgraded = int(df.get("fdr_downgraded",
                                pd.Series(dtype=bool)).fillna(False).sum())
    return {
        "n_rows_total": n_total,
        "n_cells": int(df.drop_duplicates(subset=["model", "task", "mode",
                                                    "layer", "concept"]).shape[0])
                       if not df.empty else 0,
        "verdict_counts_pre_fdr": {str(k): int(v) for k, v in pre.items()},
        "verdict_counts_post_fdr": {str(k): int(v) for k, v in post.items()},
        "tier_counts": {str(k): int(v) for k, v in tier_counts.items()},
        "n_fdr_downgrades": n_downgraded,
        "fdr_alpha": fdr_alpha,
        "stage2c_config": cfg,
        "runtime_seconds": float(runtime_seconds),
        "lib_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
        },
    }


def setup_logging(logs_root: Path) -> logging.Logger:
    logs_root.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(logs_root / "stage2c_aggregate.log")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger = logging.getLogger("stage2c.aggregate")
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--stage2c-config", default=str(STAGE2C_CONFIG_DEFAULT))
    ap.add_argument("--fdr-alpha", type=float, default=0.05)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    s2c_cfg = (yaml.safe_load(open(args.stage2c_config))
                 if Path(args.stage2c_config).exists() else {})
    data_root = Path(cfg["paths"]["data_root"])
    results_root = Path(cfg["paths"]["results_root"])
    logs_root = Path(cfg["paths"].get("logs_root", data_root / "logs"))
    logger = setup_logging(logs_root)

    t0 = time.time()
    logger.info("Stage 2c aggregator — loading per-cell summaries")
    df_all = load_all_summaries(results_root)
    logger.info("Loaded %d rows from per-cell CSVs", len(df_all))
    if df_all.empty:
        logger.warning("No per-cell results found; aggregator exits early.")
        return 0

    df_all = apply_bh_fdr(df_all)
    df_all = downgrade_post_fdr(df_all, fdr_alpha=args.fdr_alpha)
    df_all = reassign_tiers_with_q(df_all, s2c_cfg)

    out_dir = results_root / "stage2c_gplvm" / "comparison"
    atomic_csv(df_all, out_dir / "gplvm_all.csv")
    atomic_csv(verdict_counts_by_cell(df_all),
                out_dir / "verdict_counts_by_cell.csv")
    atomic_csv(cross_mode_kernel_survival(df_all),
                out_dir / "cross_mode_kernel_survival.csv")
    atomic_csv(kernel_concept_matrix(df_all),
                out_dir / "kernel_concept_matrix.csv")
    atomic_csv(dim_only_table(df_all),
                out_dir / "dim_only_table.csv")
    atomic_csv(stage2a_2b_2c_survival(df_all, results_root),
                out_dir / "stage2a_2b_2c_survival.csv")
    atomic_csv(headline_tier_cells(df_all),
                out_dir / "headline_tier_cells.csv")

    runtime = time.time() - t0
    manifest = build_manifest(df_all, s2c_cfg, runtime, args.fdr_alpha)
    atomic_json(manifest, out_dir / "manifest.json")
    logger.info("Aggregator done in %.1fs. Manifest at %s",
                  runtime, out_dir / "manifest.json")
    logger.info("Verdict counts (post-FDR): %s",
                  manifest["verdict_counts_post_fdr"])
    logger.info("Tier counts: %s", manifest["tier_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
