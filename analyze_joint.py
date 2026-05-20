#!/usr/bin/env python3
"""
Joint Stage 3 (ownership) × Stage 4 (causal) analyzer.

Runs AFTER both per-stage aggregators have written:
    results/stage3_ownership/comparison/stage3_all.csv
    results/stage4_causal/comparison/stage4_all.csv

Joins them on (model, task, mode, layer, concept) and produces purely
descriptive tables — NO verdict logic, NO p-values, NO thresholds. Just
clubs the numbers we already have so reviewers can read the trajectories
directly:

  joint_all.csv
      every per-cell row with both stage3 and stage4 columns side-by-side.

  trajectory_by_concept_type.csv
      rows = (model, task, mode, layer, concept_type, verdict)
      cols = mean/std/median/count of M1 Bu Δacc, M1 geom Δacc, M2 Bu / geom donor logit.
      concept_type ∈ {input, intermediate, answer, other}
      verdict ∈ {owned, inherited, indeterminate, skipped}

  trajectory_by_concept.csv
      rows = (model, task, mode, concept, verdict)
      cols = per-layer mean Δacc (wide form: L04_M1_Bu, L04_M1_geom, ...).
      One row per concept × verdict → reviewer can plot the trajectory.

  ownership_x_causality_by_layer.csv
      2-way contingency: (model, task, mode, layer) × verdict × causal-strength bin
      where bin = {strong (Δacc < -0.3), medium (-0.3 ≤ Δacc < -0.1),
                   weak (-0.1 ≤ Δacc < -0.01), null (Δacc ≥ -0.01)}.

  delta_gap_vs_causal_scatter.csv
      every cell: x=delta_gap_vs_K0 (Stage 3), y=m1_Bu_causal_excess_acc and
      y=m1_geom_causal_excess_acc (Stage 4). For scatter plots.

  joint_meta.json
      manifest: counts of joined cells, per-stage cell counts, generated_at.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"

# Concept-type taxonomy used for trajectory grouping.
INPUTS    = {"a", "b"}
DIGITS    = {"a_units", "a_tens", "b_units", "b_tens"}
COL_SUMS  = {"column_sum_units", "column_sum_tens", "column_sum_hundreds"}
CARRIES   = {"carry_units", "carry_tens"}
RUNNING   = {"running_sum_units", "running_sum_tens", "running_sum_hundreds"}
PARTIALS  = {"partial_product_units", "partial_product_a_units_b_tens",
              "partial_product_a_tens_b_units", "partial_product_a_tens_b_tens"}
ANS_PARTS = {"ans_units", "ans_tens", "ans_hundreds"}
ANSWER    = {"answer"}
OPERAND_FUNCS = {"max_operand", "min_operand", "operand_diff", "operand_abs_diff"}


def concept_type(c: str) -> str:
    if c in INPUTS:    return "input_operand"
    if c in DIGITS:    return "input_digit"
    if c in OPERAND_FUNCS: return "operand_function"
    if c in COL_SUMS:  return "intermediate_column_sum"
    if c in CARRIES:   return "intermediate_carry"
    if c in RUNNING:   return "intermediate_running_sum"
    if c in PARTIALS:  return "intermediate_partial_product"
    if c in ANS_PARTS: return "answer_digit"
    if c in ANSWER:    return "answer_full"
    return "other"


# ---- atomic IO --------------------------------------------------------------

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


# ---- loaders ---------------------------------------------------------------

def load_stage3(results_root: Path) -> pd.DataFrame:
    p = results_root / "stage3_ownership" / "comparison" / "stage3_all.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_stage4(results_root: Path) -> pd.DataFrame:
    p = results_root / "stage4_causal" / "comparison" / "stage4_all.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


# ---- core join + slicing ---------------------------------------------------

JOIN_KEYS = ["model", "task", "mode", "layer", "concept"]


def join_stages(s3: pd.DataFrame, s4: pd.DataFrame) -> pd.DataFrame:
    if s3.empty or s4.empty:
        return pd.DataFrame()
    # Stage 4 may have a "bsmir_tier" column equal to Stage 3's "raw_tier"; we
    # take the Stage 3 verdict + ownership-only columns from s3.
    s3_cols = [c for c in s3.columns if c not in {"raw_tier"}]
    s3sub = s3[JOIN_KEYS + [c for c in s3_cols if c not in JOIN_KEYS]].copy()
    s3sub = s3sub.rename(columns={c: f"s3_{c}" for c in s3sub.columns
                                     if c not in JOIN_KEYS})
    s4sub = s4.copy()
    s4sub = s4sub.rename(columns={c: f"s4_{c}" for c in s4sub.columns
                                     if c not in JOIN_KEYS})
    joined = s3sub.merge(s4sub, on=JOIN_KEYS, how="outer", indicator=True)
    joined["join_status"] = joined["_merge"]
    joined.drop(columns=["_merge"], inplace=True)
    joined["concept_type"] = joined["concept"].astype(str).apply(concept_type)
    return joined


# ---- aggregations ----------------------------------------------------------

def _causal_cols(j: pd.DataFrame) -> List[str]:
    cols = []
    for c in ["s4_m1_Bu_causal_excess_acc",
              "s4_m1_geom_causal_excess_acc",
              "s4_m1_Bu_causal_excess_logit",
              "s4_m1_geom_causal_excess_logit",
              "s4_m2_Bu_causal_excess_donor_logit",
              "s4_m2_geom_causal_excess_donor_logit"]:
        if c in j.columns:
            cols.append(c)
    return cols


def trajectory_by_concept_type(j: pd.DataFrame) -> pd.DataFrame:
    if j.empty: return pd.DataFrame()
    j = j[j["join_status"] == "both"].copy()
    cols = _causal_cols(j)
    if not cols: return pd.DataFrame()
    g = j.groupby(["model", "task", "mode", "layer",
                    "concept_type", "s3_verdict"], dropna=False)
    agg = g[cols].agg(["mean", "std", "median", "count"]).reset_index()
    agg.columns = [
        "_".join([str(x) for x in c if str(x) != ""])
        if isinstance(c, tuple) else c
        for c in agg.columns
    ]
    return agg


def trajectory_by_concept(j: pd.DataFrame) -> pd.DataFrame:
    """Wide form: one row per (model, task, mode, concept, verdict);
    columns are per-layer mean of each causal metric."""
    if j.empty: return pd.DataFrame()
    j = j[j["join_status"] == "both"].copy()
    cols = _causal_cols(j)
    if not cols: return pd.DataFrame()
    out_rows = []
    for (m, t, md, c, v), grp in j.groupby(
            ["model", "task", "mode", "concept", "s3_verdict"], dropna=False):
        row = {"model": m, "task": t, "mode": md, "concept": c,
                "verdict": v, "n_cells": len(grp)}
        for _, r in grp.iterrows():
            try:
                L = int(r["layer"])
            except Exception:
                continue
            for col in cols:
                row[f"L{L:02d}_{col.replace('s4_','')}"] = r.get(col)
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def ownership_x_causality_by_layer(j: pd.DataFrame) -> pd.DataFrame:
    """Contingency: (model, task, mode, layer) × verdict × causal-strength bin
    (using s4_m1_Bu_causal_excess_acc as the primary signal)."""
    if j.empty: return pd.DataFrame()
    j = j[j["join_status"] == "both"].copy()
    if "s4_m1_Bu_causal_excess_acc" not in j.columns:
        return pd.DataFrame()
    def _bin(x):
        try:
            x = float(x)
        except Exception:
            return "missing"
        if not np.isfinite(x):    return "missing"
        if x < -0.30:              return "strong"
        if x < -0.10:              return "medium"
        if x < -0.01:              return "weak"
        return "null"
    j["bu_strength_bin"] = j["s4_m1_Bu_causal_excess_acc"].apply(_bin)
    # Same bin for geom
    j["geom_strength_bin"] = j["s4_m1_geom_causal_excess_acc"].apply(_bin) \
        if "s4_m1_geom_causal_excess_acc" in j.columns else "n/a"
    out_bu = (j.groupby(["model", "task", "mode", "layer",
                            "s3_verdict", "bu_strength_bin"], dropna=False)
                .size().reset_index(name="n"))
    out_bu["granularity"] = "B_u"
    out_bu = out_bu.rename(columns={"bu_strength_bin": "strength_bin"})
    if "geom_strength_bin" in j.columns:
        out_g = (j.groupby(["model", "task", "mode", "layer",
                              "s3_verdict", "geom_strength_bin"],
                              dropna=False)
                    .size().reset_index(name="n"))
        out_g["granularity"] = "Q_geom"
        out_g = out_g.rename(columns={"geom_strength_bin": "strength_bin"})
        return pd.concat([out_bu, out_g], ignore_index=True)
    return out_bu


def delta_gap_vs_causal_scatter(j: pd.DataFrame) -> pd.DataFrame:
    if j.empty: return pd.DataFrame()
    keep_cols = JOIN_KEYS + ["concept_type", "join_status",
                                "s3_verdict", "s3_delta_gap_vs_K0",
                                "s3_var_removed_frac",
                                "s4_m1_Bu_causal_excess_acc",
                                "s4_m1_geom_causal_excess_acc",
                                "s4_m1_Bu_causal_excess_logit",
                                "s4_m1_geom_causal_excess_logit",
                                "s4_m2_Bu_causal_excess_donor_logit",
                                "s4_m2_geom_causal_excess_donor_logit",
                                "s3_raw_winner_shape", "s3_raw_family",
                                "s3_raw_tier"]
    cols = [c for c in keep_cols if c in j.columns]
    return j[cols].copy()


# ---- main -------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    results_root = Path(cfg["paths"]["results_root"])

    t0 = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] joint analyzer start")
    s3 = load_stage3(results_root)
    s4 = load_stage4(results_root)
    print(f"  stage3 rows: {len(s3)}  stage4 rows: {len(s4)}")
    if s3.empty or s4.empty:
        print("  one or both aggregator outputs missing — aborting.")
        return 1

    j = join_stages(s3, s4)
    n_both    = int((j["join_status"] == "both").sum())
    n_s3only  = int((j["join_status"] == "left_only").sum())
    n_s4only  = int((j["join_status"] == "right_only").sum())
    print(f"  joined: both={n_both}  s3_only={n_s3only}  s4_only={n_s4only}")

    out_dir = results_root / "joint_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    atomic_csv(j, out_dir / "joint_all.csv")
    atomic_csv(trajectory_by_concept_type(j),
                 out_dir / "trajectory_by_concept_type.csv")
    atomic_csv(trajectory_by_concept(j),
                 out_dir / "trajectory_by_concept.csv")
    atomic_csv(ownership_x_causality_by_layer(j),
                 out_dir / "ownership_x_causality_by_layer.csv")
    atomic_csv(delta_gap_vs_causal_scatter(j),
                 out_dir / "delta_gap_vs_causal_scatter.csv")

    atomic_json({
        "n_stage3_rows": int(len(s3)),
        "n_stage4_rows": int(len(s4)),
        "n_joined":      n_both,
        "n_stage3_only": n_s3only,
        "n_stage4_only": n_s4only,
        "models": sorted(set(j["model"].dropna().tolist())),
        "tasks":  sorted(set(j["task"].dropna().tolist())),
        "modes":  sorted(set(j["mode"].dropna().tolist())),
        "layers": sorted(set(int(x) for x in j["layer"].dropna())),
        "verdict_counts": (
            j[j["join_status"] == "both"]["s3_verdict"]
              .value_counts().to_dict()
        ),
        "wall_seconds": round(time.time() - t0, 2),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "notes": "Joint Stage 3 (ownership) x Stage 4 (causal) merge. "
                  "Descriptive only; no verdicts modified.",
    }, out_dir / "joint_meta.json")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] wrote {out_dir} "
            f"({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
