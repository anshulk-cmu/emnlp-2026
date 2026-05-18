#!/usr/bin/env python3
"""
Stage 2c — BSMI-R aggregator (Stage 15 of docs/gplvm.md).

Concatenates every per-cell `gplvm_results.csv`, applies global Benjamini-Hochberg
FDR on the permutation p-values across all candidate named-shape claims, and
downgrades Tier-A cells to Tier-B when q_BH > alpha.

Outputs (under `results/stage2c_gplvm/comparison/`):
    bsmir_all.csv                 every cell, every column
    verdict_counts_by_tier.csv    per (model, task, mode, layer) histogram
    shape_winner_matrix.csv       wide: rows=concept, cols=mode, values=winner
    dim_only_cells.csv            Tier-C cells with dim + CI
    refusals.csv                  Tier-D cells with refusal reason
    cross_mode_shape_survival.csv per-cell shape consistency across modes
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


# ─── Atomic IO ────────────────────────────────────────────────────────────────

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


# ─── Load summaries ───────────────────────────────────────────────────────────

def load_all_summaries(results_root: Path) -> pd.DataFrame:
    root = results_root / "stage2c_gplvm"
    if not root.exists():
        return pd.DataFrame()
    rows = []
    for csv_path in root.glob("*/*/mode_*/layer_*/*/gplvm_results.csv"):
        try:
            df = pd.read_csv(csv_path)
            rows.append(df)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ─── BH-FDR / BY-FDR ──────────────────────────────────────────────────────────

def _stepup_fdr(p: np.ndarray, n: int, c_n: float = 1.0) -> np.ndarray:
    """Step-up FDR procedure: q_(i) = p_(i) * n * c_n / rank with monotonicity.

    c_n = 1 for BH; c_n = sum(1/i) for BY.
    """
    finite = np.isfinite(p)
    if not finite.any():
        return np.full_like(p, np.nan, dtype=float)
    order = np.argsort(np.where(finite, p, np.inf))
    ranks = np.zeros_like(p, dtype=float)
    j = 0
    for idx in order:
        if not finite[idx]:
            continue
        j += 1
        ranks[idx] = j
    q_raw = np.where(finite, p * n * c_n / np.clip(ranks, 1, None), np.nan)
    q_sorted = q_raw[order].copy()
    for i in range(len(q_sorted) - 2, -1, -1):
        if np.isfinite(q_sorted[i + 1]) and np.isfinite(q_sorted[i]):
            q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    q_final = np.empty_like(q_raw)
    q_final[order] = q_sorted
    return np.minimum(q_final, 1.0)


def apply_bh_fdr(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Benjamini-Hochberg FDR on permutation p-values for pre-FDR Tier-A cells."""
    if df.empty or "perm_p" not in df.columns:
        df["q_BH"] = float("nan")
        return df
    df = df.copy()
    df["q_BH"] = float("nan")
    mask = df["verdict_pre_fdr"].astype(str).str.startswith("tier_A")
    if not mask.any():
        return df
    sub = df[mask]
    p = sub["perm_p"].astype(float).to_numpy()
    n = int(np.isfinite(p).sum())
    if n == 0:
        return df
    q = _stepup_fdr(p, n=n, c_n=1.0)
    df.loc[mask, "q_BH"] = q
    return df


def apply_by_fdr(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Benjamini-Yekutieli FDR (dependency-robust) sensitivity analysis."""
    if df.empty or "perm_p" not in df.columns:
        df["q_BY"] = float("nan")
        return df
    df = df.copy()
    df["q_BY"] = float("nan")
    mask = df["verdict_pre_fdr"].astype(str).str.startswith("tier_A")
    if not mask.any():
        return df
    sub = df[mask]
    p = sub["perm_p"].astype(float).to_numpy()
    n = int(np.isfinite(p).sum())
    if n == 0:
        return df
    c_n = float(np.sum(1.0 / np.arange(1, n + 1)))
    q = _stepup_fdr(p, n=n, c_n=c_n)
    df.loc[mask, "q_BY"] = q
    return df


def reassign_verdicts(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Tier-A downgrades to Tier-B when q_BH > alpha. Other tiers unchanged."""
    if df.empty:
        return df
    df = df.copy()
    df["verdict_post_fdr"] = df["verdict_pre_fdr"]
    mask_fail = (df["verdict_pre_fdr"].astype(str).str.startswith("tier_A")
                  & (df["q_BH"].astype(float) > alpha))
    df.loc[mask_fail, "verdict_post_fdr"] = "tier_B_family"
    return df


# ─── Summary tables ──────────────────────────────────────────────────────────

def verdict_counts_by_tier(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "verdict_post_fdr" not in df.columns:
        return pd.DataFrame()
    grp_cols = ["model", "task", "mode", "layer", "verdict_post_fdr"]
    grp_cols = [c for c in grp_cols if c in df.columns]
    return df.groupby(grp_cols).size().reset_index(name="count")


def shape_winner_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "best_shape" not in df.columns:
        return pd.DataFrame()
    sub = df[df["verdict_post_fdr"].astype(str).str.startswith("tier_A")]
    if sub.empty:
        return pd.DataFrame()
    pivot = sub.pivot_table(index=["model", "task", "concept", "layer"],
                              columns="mode", values="best_shape",
                              aggfunc="first").reset_index()
    return pivot


def dim_only_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    sub = df[df["verdict_post_fdr"] == "tier_C_dim_only"].copy()
    cols = ["model", "task", "mode", "layer", "concept",
            "dim_hat", "dim_ci_low", "dim_ci_high",
            "dim_estimators_agree", "PH_status", "Betti"]
    keep = [c for c in cols if c in sub.columns]
    return sub[keep]


def refusals_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    sub = df[df["verdict_post_fdr"] == "tier_D_refuse"].copy()
    cols = ["model", "task", "mode", "layer", "concept",
            "tier_reason", "dim_hat", "perm_p", "q_BH"]
    keep = [c for c in cols if c in sub.columns]
    return sub[keep]


def cross_mode_shape_survival(df: pd.DataFrame) -> pd.DataFrame:
    """Within a cell (model/task/layer/concept), did the same shape win across modes?"""
    if df.empty or "best_shape" not in df.columns:
        return pd.DataFrame()
    sub = df[df["verdict_post_fdr"].astype(str).str.startswith("tier_A")].copy()
    if sub.empty:
        return pd.DataFrame()
    pivot = sub.pivot_table(index=["model", "task", "layer", "concept"],
                              columns="mode", values="best_shape",
                              aggfunc="first")
    modes = [c for c in ["off", "answer", "norm"] if c in pivot.columns]
    pivot = pivot.reset_index()
    if modes:
        pivot["consistent"] = pivot[modes].apply(
            lambda row: len(set(v for v in row if isinstance(v, str))) <= 1, axis=1)
    return pivot


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--stage2c-config", default="configs/stage2c.yaml")
    p.add_argument("--alpha", type=float, default=0.05,
                    help="overridden by stage2c config 'fdr_alpha' if present")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    s2c_cfg = {}
    if Path(args.stage2c_config).exists():
        with open(args.stage2c_config) as f:
            s2c_cfg = yaml.safe_load(f) or {}
    alpha = float(s2c_cfg.get("fdr_alpha", args.alpha))

    results_root = Path(cfg["paths"]["results_root"])
    out_dir = results_root / "stage2c_gplvm" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                          format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("aggregate_stage2c")
    log.info("Loading all per-cell summaries from %s",
              results_root / "stage2c_gplvm")
    df = load_all_summaries(results_root)
    if df.empty:
        log.warning("No cells found. Exiting.")
        return 0
    log.info("Loaded %d cell rows", len(df))

    df = apply_bh_fdr(df, alpha=alpha)
    df = apply_by_fdr(df, alpha=alpha)
    df = reassign_verdicts(df, alpha=alpha)

    log.info("Writing aggregator outputs to %s", out_dir)
    atomic_csv(df, out_dir / "bsmir_all.csv")
    atomic_csv(verdict_counts_by_tier(df),
                out_dir / "verdict_counts_by_tier.csv")
    atomic_csv(shape_winner_matrix(df),
                out_dir / "shape_winner_matrix.csv")
    atomic_csv(dim_only_table(df), out_dir / "dim_only_cells.csv")
    atomic_csv(refusals_table(df), out_dir / "refusals.csv")
    atomic_csv(cross_mode_shape_survival(df),
                out_dir / "cross_mode_shape_survival.csv")

    counts_overall = (df["verdict_post_fdr"].value_counts().to_dict()
                       if "verdict_post_fdr" in df.columns else {})
    atomic_json({
        "n_cells_total": int(len(df)),
        "alpha": float(alpha),
        "verdict_counts_post_fdr": counts_overall,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, out_dir / "aggregator_meta.json")
    log.info("Verdict counts (post-FDR): %s", counts_overall)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
