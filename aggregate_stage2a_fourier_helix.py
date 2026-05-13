"""Stage 2a aggregator — global BH-FDR + comparison tables.

Reads every per-cell `summary_<model>_<task>_mode_<mode>_variant_<variant>.csv`
under `data/results/stage2a_fourier_helix/{model}/`, concatenates them, and:

  1. Applies Benjamini-Hochberg FDR globally across the
     ELIGIBLE-verdict subset (helix, circle, none) for q_helix, q_two_axis,
     q_coord_a, q_coord_b, q_linear. Non-eligible cells (low_K,
     period_inconsistent, null_unstable) get q = NaN.
  2. Downgrades any cell whose q_helix >= 0.05 (or q_two_axis >= 0.05 for circle)
     from helix/circle to `none` post-FDR. Original verdict kept as
     `geometry_pre_fdr`.
  3. Writes `comparison/` tables: cross-cell counts, cross-mode helix survival,
     cross-variant agreement, discovered-vs-predicted match summary,
     unexpected_periods, manifest.

CLI:
  python aggregate_stage2a_fourier_helix.py --config /home/anshulk/emnlp2026/config.yaml
"""

import argparse
import glob
import hashlib
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
from scipy.stats import false_discovery_control


ELIGIBLE_VERDICTS = {"helix", "circle", "none"}


# ──────────────────────────────────────────────────────────────────────────────
# Atomic I/O
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_all_summaries(results_root: Path) -> pd.DataFrame:
    """Read every summary_<model>_<task>_mode_<mode>_variant_<variant>.csv into one frame."""
    root = results_root / "stage2a_fourier_helix"
    paths = sorted(glob.glob(str(root / "*" / "summary_*.csv")))
    if not paths:
        return pd.DataFrame()
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["__source_csv"] = str(p)
            frames.append(df)
        except Exception as exc:
            print(f"WARN: failed to read {p}: {exc}", file=sys.stderr)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# Global BH-FDR
# ──────────────────────────────────────────────────────────────────────────────

def apply_bh_fdr(df: pd.DataFrame) -> pd.DataFrame:
    """Compute q-values for the eligible-verdict subset, NaN otherwise.

    The eligible subset is `geometry_detected ∈ {helix, circle, none}` — these are
    the cells where Fourier ran and a verdict was meaningful. low_K /
    period_inconsistent / null_unstable cells are excluded from FDR (their p-values
    are 1.0 by construction).

    Adds q_helix, q_two_axis, q_coord_a, q_coord_b, q_linear columns.
    """
    eligible = df["geometry_detected"].isin(ELIGIBLE_VERDICTS)
    p_cols = ["p_helix", "p_two_axis", "p_coord_a", "p_coord_b", "p_linear"]
    for p_col in p_cols:
        q_col = "q" + p_col[1:]
        df[q_col] = np.nan
        if eligible.sum() > 0:
            sub_p = df.loc[eligible, p_col].astype(float).to_numpy()
            # Replace any NaN/Inf with 1.0 (don't drop the row from FDR)
            sub_p = np.where(np.isfinite(sub_p), sub_p, 1.0)
            sub_p = np.clip(sub_p, 1e-30, 1.0)
            sub_q = false_discovery_control(sub_p, method="bh")
            df.loc[eligible, q_col] = sub_q
    return df


def downgrade_post_fdr(df: pd.DataFrame, fdr_alpha: float) -> pd.DataFrame:
    """Cells whose q_helix >= alpha lose the helix verdict; q_two_axis >= alpha
    loses circle. Both fall back to `none`. Pre-FDR verdict preserved."""
    df["geometry_pre_fdr"] = df["geometry_detected"]
    helix_fail = (df["geometry_detected"] == "helix") & (df["q_helix"] >= fdr_alpha)
    circle_fail = (df["geometry_detected"] == "circle") & (df["q_two_axis"] >= fdr_alpha)
    df.loc[helix_fail | circle_fail, "geometry_detected"] = "none"
    df["fdr_downgraded"] = (helix_fail | circle_fail)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Comparison tables
# ──────────────────────────────────────────────────────────────────────────────

def geometry_counts_by_cell(df: pd.DataFrame) -> pd.DataFrame:
    """Counts per (model, task, mode, layer, variant) of each verdict."""
    g = (df.groupby(["model", "task", "mode", "layer", "variant", "geometry_detected"])
           .size().reset_index(name="n_concepts"))
    pivot = (g.pivot_table(index=["model", "task", "mode", "layer", "variant"],
                            columns="geometry_detected", values="n_concepts", fill_value=0)
              .reset_index())
    pivot.columns.name = None
    return pivot


def discovered_vs_predicted(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (cell, concept, variant) with discovered period vs prior table."""
    keep = ["model", "task", "mode", "layer", "variant", "concept",
            "geometry_detected", "geometry_pre_fdr", "discovered_period",
            "prior_predicted_period", "period_match",
            "fcr_helix", "fcr_two_axis",
            "p_helix", "q_helix", "p_two_axis", "q_two_axis",
            "plane_rank_ratio", "plane_2d_ok",
            "K_natural", "K_present", "r", "n_samples_used",
            "non_uniform_grid_flag", "low_K_natural_flag"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


def cross_mode_helix_survival(df: pd.DataFrame) -> pd.DataFrame:
    """For each (model, task, layer, variant, concept), wide-form verdict per mode."""
    sub = df[df["variant"].isin(["lda_a", "ccsvd"])].copy()
    pivot = (sub.pivot_table(index=["model", "task", "layer", "variant", "concept"],
                              columns="mode", values="geometry_detected", aggfunc="first")
                .reset_index())
    pivot.columns.name = None
    # Add helix-survival counts
    mode_cols = [c for c in pivot.columns if c in ("off", "answer", "norm")]
    if mode_cols:
        pivot["n_helix_modes"] = (pivot[mode_cols] == "helix").sum(axis=1)
        pivot["n_circle_modes"] = (pivot[mode_cols] == "circle").sum(axis=1)
        pivot["n_eligible_modes"] = pivot[mode_cols].apply(
            lambda r: sum(1 for v in r if v in ELIGIBLE_VERDICTS), axis=1)
    return pivot


def cross_variant_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """For each (model, task, mode, layer, concept), compare lda_a vs ccsvd verdicts."""
    pivot = (df.pivot_table(index=["model", "task", "mode", "layer", "concept"],
                             columns="variant", values="geometry_detected", aggfunc="first")
                .reset_index())
    pivot.columns.name = None
    if "lda_a" in pivot.columns and "ccsvd" in pivot.columns:
        pivot["verdict_agree"] = (pivot["lda_a"] == pivot["ccsvd"])
        pivot["both_eligible"] = (pivot["lda_a"].isin(ELIGIBLE_VERDICTS)
                                   & pivot["ccsvd"].isin(ELIGIBLE_VERDICTS))
    # Add discovered-period agreement when both ran Fourier
    dp = (df.pivot_table(index=["model", "task", "mode", "layer", "concept"],
                          columns="variant", values="discovered_period", aggfunc="first")
              .reset_index())
    dp.columns.name = None
    if "lda_a" in dp.columns and "ccsvd" in dp.columns:
        dp["period_agree_within_bin"] = (
            dp["lda_a"].astype(float).sub(dp["ccsvd"].astype(float)).abs() <= 1.0
        )
        pivot = pivot.merge(
            dp[["model", "task", "mode", "layer", "concept", "period_agree_within_bin"]],
            on=["model", "task", "mode", "layer", "concept"], how="left",
        )
    return pivot


def period_prior_for_stage2c(df: pd.DataFrame) -> pd.DataFrame:
    """For every cell that passed Stage 2a (helix or circle, post-FDR), emit the
    discovered period + top-2 coords. This feeds Stage 2c's GPLVM kernel init."""
    passed = df[df["geometry_detected"].isin({"helix", "circle"})].copy()
    keep = ["model", "task", "mode", "layer", "variant", "concept",
            "geometry_detected", "discovered_period", "k_star",
            "c_a", "c_b", "c_L", "fcr_helix", "fcr_two_axis", "plane_rank_ratio",
            "p_helix", "q_helix"]
    keep = [c for c in keep if c in passed.columns]
    return passed[keep].copy()


def unexpected_periods(df: pd.DataFrame) -> pd.DataFrame:
    """Cells where post-FDR verdict is helix/circle AND discovered ≠ predicted."""
    passed = df[df["geometry_detected"].isin({"helix", "circle"})].copy()
    if "period_match" not in passed.columns:
        return pd.DataFrame()
    unexpected = passed[~passed["period_match"].fillna(False).astype(bool)].copy()
    keep = ["model", "task", "mode", "layer", "variant", "concept",
            "geometry_detected", "discovered_period", "prior_predicted_period",
            "fcr_helix", "fcr_two_axis", "plane_rank_ratio",
            "q_helix", "q_two_axis", "non_uniform_grid_flag",
            "N_over_K", "N_over_r"]
    keep = [c for c in keep if c in unexpected.columns]
    return unexpected[keep].sort_values(
        ["model", "task", "mode", "layer", "variant", "concept"]
    ).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Manifest
# ──────────────────────────────────────────────────────────────────────────────

def build_manifest(df: pd.DataFrame, cfg: dict, runtime_seconds: float) -> dict:
    counts = df["geometry_detected"].value_counts().to_dict()
    pre = df["geometry_pre_fdr"].value_counts().to_dict() if "geometry_pre_fdr" in df.columns else {}
    return {
        "n_rows_total": int(len(df)),
        "n_cells": int(df.groupby(["model", "task", "mode", "layer", "variant"]).ngroups),
        "n_models": int(df["model"].nunique()),
        "n_tasks": int(df["task"].nunique()),
        "n_modes": int(df["mode"].nunique()),
        "n_layers": int(df["layer"].nunique()),
        "n_variants": int(df["variant"].nunique()),
        "n_concepts_unique": int(df["concept"].nunique()),
        "verdict_counts_post_fdr": counts,
        "verdict_counts_pre_fdr": pre,
        "fdr_alpha": float(cfg["stage2a"].get("fdr_alpha", 0.05)),
        "n_permutations": int(cfg["stage2a"].get("n_permutations", 1000)),
        "fcr_threshold": float(cfg["stage2a"].get("fcr_threshold", 0.30)),
        "n_fdr_downgrades": int(df["fdr_downgraded"].sum()) if "fdr_downgraded" in df.columns else 0,
        "lib_versions": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
        },
        "aggregator_runtime_seconds": round(runtime_seconds, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(logs_root: Path) -> logging.Logger:
    logs_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("aggregate_stage2a")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = WatchedFileHandler(logs_root / "aggregate_stage2a.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    paths = cfg["paths"]
    results_root = Path(paths["results_root"])
    logs_root = Path(paths["logs_root"])
    fdr_alpha = float(cfg["stage2a"].get("fdr_alpha", 0.05))

    logger = setup_logging(logs_root)
    t0 = time.time()
    logger.info(f"=== Stage 2a aggregator: results_root={results_root}, fdr_alpha={fdr_alpha} ===")

    df = load_all_summaries(results_root)
    if df.empty:
        logger.error("No per-model summary CSVs found — workers must run first.")
        sys.exit(1)
    logger.info(f"Loaded {len(df)} rows from {df['__source_csv'].nunique()} CSVs")
    logger.info(f"Pre-FDR verdict counts: {df['geometry_detected'].value_counts().to_dict()}")

    df = apply_bh_fdr(df)
    df = downgrade_post_fdr(df, fdr_alpha)
    logger.info(f"Post-FDR verdict counts: {df['geometry_detected'].value_counts().to_dict()}")
    if "fdr_downgraded" in df.columns:
        n_dg = int(df["fdr_downgraded"].sum())
        logger.info(f"FDR downgraded {n_dg} cells from helix/circle to none")

    comparison_dir = results_root / "stage2a_fourier_helix" / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    atomic_csv(df.drop(columns=["__source_csv"], errors="ignore"),
               comparison_dir / "fcr_all.csv")
    logger.info(f"wrote fcr_all.csv ({len(df)} rows)")

    cnts = geometry_counts_by_cell(df)
    atomic_csv(cnts, comparison_dir / "geometry_counts_by_cell.csv")
    logger.info(f"wrote geometry_counts_by_cell.csv ({len(cnts)} rows)")

    dvp = discovered_vs_predicted(df)
    atomic_csv(dvp, comparison_dir / "discovered_vs_predicted.csv")
    logger.info(f"wrote discovered_vs_predicted.csv ({len(dvp)} rows)")

    surv = cross_mode_helix_survival(df)
    atomic_csv(surv, comparison_dir / "cross_mode_helix_survival.csv")
    logger.info(f"wrote cross_mode_helix_survival.csv ({len(surv)} rows)")

    cva = cross_variant_agreement(df)
    atomic_csv(cva, comparison_dir / "cross_variant_agreement.csv")
    logger.info(f"wrote cross_variant_agreement.csv ({len(cva)} rows)")

    pp = period_prior_for_stage2c(df)
    atomic_csv(pp, comparison_dir / "period_prior_for_stage2c.csv")
    logger.info(f"wrote period_prior_for_stage2c.csv ({len(pp)} rows)")

    unex = unexpected_periods(df)
    atomic_csv(unex, comparison_dir / "unexpected_periods.csv")
    logger.info(f"wrote unexpected_periods.csv ({len(unex)} rows)")

    manifest = build_manifest(df, cfg, time.time() - t0)
    atomic_json(manifest, comparison_dir / "manifest.json")
    logger.info(f"wrote manifest.json: {manifest['n_rows_total']} rows across "
                f"{manifest['n_cells']} cells, {manifest['n_concepts_unique']} concepts")

    logger.info(f"=== Stage 2a aggregator DONE in {time.time() - t0:.1f}s ===")


if __name__ == "__main__":
    main()
