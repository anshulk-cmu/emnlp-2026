"""Stage 2b aggregator — global BH-FDR + comparison tables.

Reads every per-model `summary_<model>_<task>_mode_<mode>_variant_<variant>.csv`
under `data/results/stage2b_dsw/{model}/`, concatenates them, and:

  1. Applies BH-FDR globally over eligible cells (spread_confirmed, spread_marginal,
     centroid_only_shape) on p_dsw. Cells flagged insufficient / low_K_after_filter /
     null_unstable get q = NaN.
  2. Downgrades spread_confirmed → centroid_only_shape if q_dsw ≥ fdr_alpha.
     Preserves the pre-FDR verdict as `spread_verdict_pre_fdr`.
  3. Re-evaluates confidence tier using q_dsw (replaces per-cell p-based tier).
  4. Writes comparison tables: verdict counts per cell, cross-mode spread survival,
     cross-variant agreement, **stage2a_vs_stage2b_survival** (headline transition
     table), confidence tier distribution, manifest.

CLI:
  python aggregate_stage2b_dsw.py --config /home/anshulk/emnlp2026/config.yaml
"""

import argparse
import glob
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


ELIGIBLE_VERDICTS = {"spread_confirmed", "spread_marginal", "centroid_only_shape"}
STAGE2A_DETECTED = {"helix", "circle"}


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
    root = results_root / "stage2b_dsw"
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
    df["q_dsw"] = np.nan
    eligible = df["spread_verdict"].isin(ELIGIBLE_VERDICTS)
    if eligible.sum() > 0:
        p = df.loc[eligible, "p_dsw"].astype(float).to_numpy()
        p = np.where(np.isfinite(p), p, 1.0)
        p = np.clip(p, 1e-30, 1.0)
        q = false_discovery_control(p, method="bh")
        df.loc[eligible, "q_dsw"] = q
    return df


def downgrade_post_fdr(df: pd.DataFrame, fdr_alpha: float) -> pd.DataFrame:
    df["spread_verdict_pre_fdr"] = df["spread_verdict"]
    fail = (df["spread_verdict"] == "spread_confirmed") & (df["q_dsw"] >= fdr_alpha)
    df.loc[fail, "spread_verdict"] = "centroid_only_shape"
    df["fdr_downgraded"] = fail
    return df


def reassign_tiers_with_q(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per plan B.7, confidence tier uses q (not raw p). Recompute now that q is known."""
    stage2b_cfg = cfg.get("stage2b", {})
    from stage2b_dsw_spread_aware import assign_confidence_tier
    new_tiers = []
    for _, row in df.iterrows():
        K_present = int(row.get("K_present", 0) or 0)
        min_n_v = int(row.get("min_n_v", 0) or 0)
        min_ratio = float(row.get("min_ratio_v", 0.0) or 0.0)
        q = row.get("q_dsw")
        if not isinstance(q, (int, float)) or not np.isfinite(q):
            q = 1.0
        rho_low = row.get("rho_low")
        if not isinstance(rho_low, (int, float)) or not np.isfinite(rho_low):
            rho_low = 0.0
        new_tiers.append(assign_confidence_tier(K_present, min_n_v, min_ratio,
                                                  float(q), float(rho_low), stage2b_cfg))
    df["confidence_tier"] = new_tiers
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Comparison tables
# ──────────────────────────────────────────────────────────────────────────────

def verdict_counts_by_cell(df: pd.DataFrame) -> pd.DataFrame:
    g = (df.groupby(["model", "task", "mode", "layer", "variant", "spread_verdict"])
           .size().reset_index(name="n_concepts"))
    pivot = (g.pivot_table(index=["model", "task", "mode", "layer", "variant"],
                            columns="spread_verdict", values="n_concepts", fill_value=0)
              .reset_index())
    pivot.columns.name = None
    return pivot


def cross_mode_spread_survival(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["variant"].isin(["lda_a", "ccsvd"])].copy()
    pivot = (sub.pivot_table(index=["model", "task", "layer", "variant", "concept"],
                              columns="mode", values="spread_verdict", aggfunc="first")
                .reset_index())
    pivot.columns.name = None
    mode_cols = [c for c in pivot.columns if c in ("off", "answer", "norm")]
    if mode_cols:
        pivot["n_spread_confirmed_modes"] = (pivot[mode_cols] == "spread_confirmed").sum(axis=1)
        pivot["n_centroid_only_modes"] = (pivot[mode_cols] == "centroid_only_shape").sum(axis=1)
        pivot["n_eligible_modes"] = pivot[mode_cols].apply(
            lambda r: sum(1 for v in r if v in ELIGIBLE_VERDICTS), axis=1,
        )
    return pivot


def cross_variant_agreement(df: pd.DataFrame) -> pd.DataFrame:
    pivot = (df.pivot_table(index=["model", "task", "mode", "layer", "concept"],
                             columns="variant", values="spread_verdict", aggfunc="first")
                .reset_index())
    pivot.columns.name = None
    if "lda_a" in pivot.columns and "ccsvd" in pivot.columns:
        pivot["verdict_agree"] = (pivot["lda_a"] == pivot["ccsvd"])
    rho = (df.pivot_table(index=["model", "task", "mode", "layer", "concept"],
                            columns="variant", values="rho_centroid", aggfunc="first")
                .reset_index())
    rho.columns.name = None
    if "lda_a" in rho.columns and "ccsvd" in rho.columns:
        rho["rho_diff_abs"] = (rho["lda_a"].astype(float) - rho["ccsvd"].astype(float)).abs()
        pivot = pivot.merge(
            rho[["model", "task", "mode", "layer", "concept", "rho_diff_abs"]],
            on=["model", "task", "mode", "layer", "concept"], how="left",
        )
    return pivot


def stage2a_vs_stage2b_survival(df: pd.DataFrame) -> pd.DataFrame:
    """Headline transition table. For each Stage 2a helix/circle cell, report
    the Stage 2b verdict + tier."""
    if "stage2a_verdict" not in df.columns:
        return pd.DataFrame()
    sub = df[df["stage2a_verdict"].isin(list(STAGE2A_DETECTED))].copy()
    keep = ["model", "task", "mode", "layer", "variant", "concept",
            "stage2a_verdict", "stage2a_discovered_period",
            "spread_verdict_pre_fdr", "spread_verdict", "confidence_tier",
            "rho_centroid", "rho_low", "rho_high", "p_dsw", "q_dsw",
            "K_present", "min_n_v", "min_ratio_v", "gamma",
            "null_unstable_dsw", "fdr_downgraded"]
    keep = [c for c in keep if c in sub.columns]
    return sub[keep].sort_values(["model", "task", "mode", "layer", "variant", "concept"]).reset_index(drop=True)


def confidence_tier_distribution(df: pd.DataFrame) -> pd.DataFrame:
    g = (df.groupby(["model", "task", "mode", "layer", "variant",
                      "confidence_tier", "spread_verdict"])
           .size().reset_index(name="n"))
    pivot = (g.pivot_table(index=["model", "task", "mode", "layer", "variant", "confidence_tier"],
                            columns="spread_verdict", values="n", fill_value=0)
              .reset_index())
    pivot.columns.name = None
    return pivot


# ──────────────────────────────────────────────────────────────────────────────
# Manifest
# ──────────────────────────────────────────────────────────────────────────────

def build_manifest(df: pd.DataFrame, cfg: dict, runtime_seconds: float,
                    toy_calibration: dict) -> dict:
    counts = df["spread_verdict"].value_counts().to_dict()
    pre = df["spread_verdict_pre_fdr"].value_counts().to_dict() if "spread_verdict_pre_fdr" in df.columns else {}
    tier_counts = df["confidence_tier"].value_counts().to_dict() if "confidence_tier" in df.columns else {}
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
        "confidence_tier_counts": tier_counts,
        "fdr_alpha": float(cfg["stage2b"].get("fdr_alpha", 0.05)),
        "n_permutations": int(cfg["stage2b"].get("n_permutations", 1000)),
        "n_bootstrap": int(cfg["stage2b"].get("n_bootstrap", 1000)),
        "rho_pass_threshold": float(cfg["stage2b"].get("rho_pass_threshold", 0.85)),
        "rho_low_ci_threshold": float(cfg["stage2b"].get("rho_low_ci_threshold", 0.70)),
        "n_fdr_downgrades": int(df["fdr_downgraded"].sum()) if "fdr_downgraded" in df.columns else 0,
        "toy_calibration": toy_calibration,
        "lib_versions": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "sklearn": __import__("sklearn").__version__,
        },
        "aggregator_runtime_seconds": round(runtime_seconds, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(logs_root: Path) -> logging.Logger:
    logs_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("aggregate_stage2b")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = WatchedFileHandler(logs_root / "aggregate_stage2b.log")
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
    fdr_alpha = float(cfg["stage2b"].get("fdr_alpha", 0.05))

    logger = setup_logging(logs_root)
    t0 = time.time()
    logger.info(f"=== Stage 2b aggregator: results_root={results_root}, fdr_alpha={fdr_alpha} ===")

    df = load_all_summaries(results_root)
    if df.empty:
        logger.error("No per-model summary CSVs found under stage2b_dsw — workers must run first.")
        sys.exit(1)
    logger.info(f"Loaded {len(df)} rows from {df['__source_csv'].nunique()} CSVs")
    logger.info(f"Pre-FDR verdict counts: {df['spread_verdict'].value_counts().to_dict()}")

    df = apply_bh_fdr(df)
    df = downgrade_post_fdr(df, fdr_alpha)
    df = reassign_tiers_with_q(df, cfg)
    logger.info(f"Post-FDR verdict counts: {df['spread_verdict'].value_counts().to_dict()}")
    if "fdr_downgraded" in df.columns:
        logger.info(f"FDR downgraded {int(df['fdr_downgraded'].sum())} cells from spread_confirmed to centroid_only_shape")
    logger.info(f"Post-FDR tier counts: {df['confidence_tier'].value_counts().to_dict()}")

    comparison_dir = results_root / "stage2b_dsw" / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    atomic_csv(df.drop(columns=["__source_csv"], errors="ignore"),
               comparison_dir / "dsw_all.csv")
    logger.info(f"wrote dsw_all.csv ({len(df)} rows)")

    cnts = verdict_counts_by_cell(df)
    atomic_csv(cnts, comparison_dir / "spread_verdict_counts_by_cell.csv")
    logger.info(f"wrote spread_verdict_counts_by_cell.csv ({len(cnts)} rows)")

    surv = cross_mode_spread_survival(df)
    atomic_csv(surv, comparison_dir / "cross_mode_spread_survival.csv")
    logger.info(f"wrote cross_mode_spread_survival.csv ({len(surv)} rows)")

    cva = cross_variant_agreement(df)
    atomic_csv(cva, comparison_dir / "cross_variant_agreement.csv")
    logger.info(f"wrote cross_variant_agreement.csv ({len(cva)} rows)")

    trans = stage2a_vs_stage2b_survival(df)
    atomic_csv(trans, comparison_dir / "stage2a_vs_stage2b_survival.csv")
    logger.info(f"wrote stage2a_vs_stage2b_survival.csv ({len(trans)} rows)")

    tier = confidence_tier_distribution(df)
    atomic_csv(tier, comparison_dir / "confidence_tier_distribution.csv")
    logger.info(f"wrote confidence_tier_distribution.csv ({len(tier)} rows)")

    # Pull toy calibration record for the manifest.
    toy_yaml = Path(args.config).parent / cfg["stage2b"].get("toy_calibration_path", "configs/stage2b.yaml")
    toy_cal = yaml.safe_load(toy_yaml.read_text()) if toy_yaml.exists() else {}

    manifest = build_manifest(df, cfg, time.time() - t0, toy_cal)
    atomic_json(manifest, comparison_dir / "manifest.json")
    logger.info(f"wrote manifest.json: {manifest['n_rows_total']} rows across "
                f"{manifest['n_cells']} cells, {manifest['n_concepts_unique']} concepts")

    logger.info(f"=== Stage 2b aggregator DONE in {time.time() - t0:.1f}s ===")


if __name__ == "__main__":
    main()
