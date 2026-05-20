#!/usr/bin/env python3
"""
Stage 3 — Ownership aggregator.

Concatenates every per-cell stage3_results.csv, applies global Benjamini-Hochberg
FDR on the orth-side permutation p-values, and produces summary tables:

    stage3_all.csv                  every cell, every column, with q_BH
    verdict_counts.csv              per (model, task, mode, layer) histogram
    verdict_counts_by_shape.csv     verdicts split by raw winner shape
    verdict_counts_by_family.csv    verdicts split by raw winner family
    cross_mode_ownership.csv        per (model, task, layer, concept) ownership
                                    consistency across modes
    delta_logZ_distribution.csv     summary stats per cohort
    aggregator_meta.json            manifest
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
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


# ─── Atomic IO ─────────────────────────────────────────────────────────────

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


# ─── Load summaries ────────────────────────────────────────────────────────

def load_all_summaries(results_root: Path) -> pd.DataFrame:
    root = results_root / "stage3_ownership"
    if not root.exists():
        return pd.DataFrame()
    rows = []
    for csv_path in root.glob("*/*/mode_*/layer_*/*/stage3_results.csv"):
        try:
            df = pd.read_csv(csv_path)
            rows.append(df)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ─── FDR helpers (BH and BY) ───────────────────────────────────────────────

def _stepup_fdr(p: np.ndarray, n: int, c_n: float = 1.0) -> np.ndarray:
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


def apply_fdr(df: pd.DataFrame, alpha: float = 0.10) -> pd.DataFrame:
    """Compute BH + BY q-values on orth_perm_p as informational columns only.

    The BINARY verdict (sign of Δgap_vs_K0) is NOT modified by FDR — the
    verdict is fully determined by the Bayes-factor change which is itself
    a multiple-comparison-safe quantity (an evidence delta, not a p-value).
    q_BH and q_BY are kept as diagnostic columns for paper analysis.
    """
    if df.empty or "orth_perm_p" not in df.columns:
        return df
    df = df.copy()
    p = pd.to_numeric(df["orth_perm_p"], errors="coerce").to_numpy()
    n = int(np.isfinite(p).sum())
    if n == 0:
        df["q_BH"] = np.nan
        df["q_BY"] = np.nan
        return df
    c_BY = float(np.sum(1.0 / np.arange(1, n + 1)))
    df["q_BH"] = _stepup_fdr(p, n, c_n=1.0)
    df["q_BY"] = _stepup_fdr(p, n, c_n=c_BY)
    return df


# ─── Summary tables ────────────────────────────────────────────────────────

def verdict_counts(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return (df.groupby(group_cols + ["verdict"])
              .size()
              .reset_index(name="n")
              .pivot_table(index=group_cols, columns="verdict",
                              values="n", fill_value=0)
              .reset_index())


def cross_mode_ownership(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    pivot = df.pivot_table(
        index=["model", "task", "layer", "concept"],
        columns="mode", values="verdict", aggfunc="first")
    return pivot.reset_index()


def delta_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of Δgap_vs_K0 per (model, task, mode), split by verdict."""
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(["model", "task", "mode", "verdict"])
    out = g["delta_gap_vs_K0"].agg(
        n="count",
        mean="mean",
        std="std",
        median="median",
        p10=lambda s: float(np.nanpercentile(s, 10)),
        p90=lambda s: float(np.nanpercentile(s, 90))
    ).reset_index()
    return out


# ─── Main ──────────────────────────────────────────────────────────────────

def setup_logging(logs_root: Path) -> logging.Logger:
    logs_root.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(logs_root / "stage3_aggregator.log")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger = logging.getLogger("stage3.agg")
    logger.handlers.clear()
    logger.addHandler(fh); logger.addHandler(sh)
    logger.setLevel(logging.INFO); logger.propagate = False
    return logger


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--alpha", type=float, default=0.10)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    results_root = Path(cfg["paths"]["results_root"])
    logs_root = Path(cfg["paths"].get("logs_root",
                                          cfg["paths"]["data_root"] + "/logs"))
    logger = setup_logging(logs_root)
    t0 = time.time()
    logger.info("=== Stage 3 aggregator ===")

    df = load_all_summaries(results_root)
    if df.empty:
        logger.warning("No per-cell results found under stage3_ownership/.")
        return 0
    logger.info("Loaded %d per-cell rows", len(df))

    df = apply_fdr(df, alpha=args.alpha)
    out_dir = results_root / "stage3_ownership" / "comparison"

    atomic_csv(df, out_dir / "stage3_all.csv")

    # verdict histograms
    vc_layer = verdict_counts(df, ["model", "task", "mode", "layer"])
    atomic_csv(vc_layer, out_dir / "verdict_counts.csv")
    if "raw_winner_shape" in df.columns:
        vc_shape = verdict_counts(df, ["model", "task", "raw_winner_shape"])
        atomic_csv(vc_shape, out_dir / "verdict_counts_by_shape.csv")
    if "raw_family" in df.columns:
        vc_fam = verdict_counts(df, ["model", "task", "raw_family"])
        atomic_csv(vc_fam, out_dir / "verdict_counts_by_family.csv")

    xm = cross_mode_ownership(df)
    atomic_csv(xm, out_dir / "cross_mode_ownership.csv")

    dlz = delta_summary(df)
    atomic_csv(dlz, out_dir / "delta_gap_distribution.csv")

    # Manifest
    n_owned = int((df["verdict"] == "owned").sum())
    n_inh   = int((df["verdict"] == "inherited").sum())
    n_indet = int((df["verdict"] == "indeterminate").sum())
    n_skip  = int((df["verdict"] == "skipped").sum())
    atomic_json({
        "n_cells_total":   int(len(df)),
        "n_owned":         n_owned,
        "n_inherited":     n_inh,
        "n_indeterminate": n_indet,
        "n_skipped":       n_skip,
        "verdict_rule":    "sign(delta_gap_vs_K0)  [binary]",
        "fdr_alpha":       args.alpha,
        "wall_seconds":    round(time.time() - t0, 2),
    }, out_dir / "aggregator_meta.json")
    logger.info("done. owned=%d inherited=%d indeterminate=%d skipped=%d",
                  n_owned, n_inh, n_indet, n_skip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
