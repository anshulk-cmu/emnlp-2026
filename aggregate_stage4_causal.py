#!/usr/bin/env python3
"""
Stage 4 causal aggregator — concatenate every per-cell stage4_summary.csv,
join with BSMI-R metadata, and write cross-cell tables under
results/stage4_causal/comparison/.
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
    root = results_root / "stage4_causal"
    if not root.exists():
        return pd.DataFrame()
    rows = []
    for csv_p in root.glob("*/*/mode_*/layer_*/*/stage4_summary.csv"):
        try:
            rows.append(pd.read_csv(csv_p))
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def cross_layer_by_concept(df: pd.DataFrame) -> pd.DataFrame:
    """Wide table: rows=(model, task, mode, concept), cols=layer, values=causal excess."""
    if df.empty:
        return pd.DataFrame()
    out = []
    for (m, t, mode, c), grp in df.groupby(["model", "task", "mode", "concept"]):
        row = {"model": m, "task": t, "mode": mode, "concept": c}
        for _, r in grp.iterrows():
            L = int(r["layer"])
            row[f"L{L:02d}_M1_Bu"] = r.get("m1_Bu_causal_excess_logit")
            row[f"L{L:02d}_M1_geom"] = r.get("m1_geom_causal_excess_logit")
            row[f"L{L:02d}_M2_Bu"] = r.get("m2_Bu_causal_excess_donor_logit")
            row[f"L{L:02d}_M2_geom"] = r.get("m2_geom_causal_excess_donor_logit")
            row[f"L{L:02d}_shape"] = r.get("winner_shape")
        out.append(row)
    return pd.DataFrame(out)


def causal_summary_by_layer(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std of causal excess by (model, mode, layer) and by concept type."""
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["concept_type"] = "other"
    inputs = {"a", "b", "a_tens", "a_units", "b_tens", "b_units"}
    answers = ["ans_units", "ans_tens", "ans_hundreds", "answer"]
    df.loc[df["concept"].isin(inputs), "concept_type"] = "input"
    df.loc[df["concept"].isin(answers), "concept_type"] = "answer"
    g = df.groupby(["model", "task", "mode", "layer", "concept_type"])
    cols = ["m1_Bu_causal_excess_logit", "m1_geom_causal_excess_logit",
            "m2_Bu_causal_excess_donor_logit", "m2_geom_causal_excess_donor_logit"]
    return g[cols].agg(["mean", "std", "count"]).reset_index()


def tier_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "bsmir_tier" not in df.columns:
        return pd.DataFrame()
    return df.groupby(["model", "task", "mode", "layer", "bsmir_tier"]).size().reset_index(name="count")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    results_root = Path(cfg["paths"]["results_root"])

    logging.basicConfig(level=logging.INFO,
                          format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("aggregate_stage4")
    log.info("Loading per-cell stage4 summaries from %s",
              results_root / "stage4_causal")
    df = load_all_summaries(results_root)
    if df.empty:
        log.warning("No Stage 4 results found.")
        return 0
    log.info("Loaded %d cell rows", len(df))

    out_dir = results_root / "stage4_causal" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(df, out_dir / "stage4_all.csv")
    atomic_csv(cross_layer_by_concept(df), out_dir / "cross_layer_by_concept.csv")
    atomic_csv(causal_summary_by_layer(df), out_dir / "causal_summary_by_layer.csv")
    atomic_csv(tier_counts(df), out_dir / "tier_counts_by_layer.csv")

    atomic_json({
        "n_cells_total": int(len(df)),
        "models": sorted(df["model"].dropna().unique().tolist()),
        "tasks": sorted(df["task"].dropna().unique().tolist()),
        "modes": sorted(df["mode"].dropna().unique().tolist()),
        "layers": sorted(map(int, df["layer"].dropna().unique().tolist())),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, out_dir / "aggregator_meta.json")
    log.info("Wrote aggregator outputs to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
