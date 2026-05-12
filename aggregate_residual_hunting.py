"""Step 7 aggregator — concatenate per-model summaries and emit cross-mode/cell tables.

Inputs:
  results/residual_hunting/{model}/summary_{model}_{task}_mode_{mode}.csv

Outputs (under results/residual_hunting/comparison/):
  summary_all.csv                       # all rows, 90 cells × 2 variants
  var_explained_cross_mode.csv          # pivot, restricted to matched_population
  n_above_mp_cross_mode.csv             # pivot with mp_reliable_flag column
  residual_top_correlate_cross_mode.csv # top correlate per (model, task, layer, mode)
  variant_delta.csv                     # generous - merged per cell

Run after Step 7 finishes.
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    results_root = Path(cfg["paths"]["results_root"])
    base = results_root / "residual_hunting"
    out_dir = base / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Concatenate all summary CSVs
    summary_paths = list(base.glob("*/summary_*_mode_*.csv"))
    if not summary_paths:
        print(f"no summary CSVs found under {base}")
        return
    dfs = [pd.read_csv(p) for p in summary_paths]
    summary_all = pd.concat(dfs, ignore_index=True)
    summary_all = summary_all.sort_values(
        ["model", "task", "mode", "layer", "variant"]
    ).reset_index(drop=True)
    summary_all.to_csv(out_dir / "summary_all.csv", index=False)
    print(f"wrote summary_all.csv ({len(summary_all)} rows)")

    # 2. Cross-mode var_explained pivot
    for metric, fname in [
        ("var_explained", "var_explained_cross_mode.csv"),
        ("n_above_mp",    "n_above_mp_cross_mode.csv"),
        ("k_union",       "k_union_cross_mode.csv"),
        ("gamma",         "gamma_cross_mode.csv"),
    ]:
        if metric not in summary_all.columns:
            continue
        pivot = summary_all.pivot_table(
            index=["model", "task", "layer", "variant"],
            columns="mode", values=metric, aggfunc="first",
        ).reset_index()
        pivot.columns.name = None
        # Add mp_reliable_flag from any mode row (gamma the same up to V_all per-mode differences)
        if "mp_reliable_flag" in summary_all.columns:
            flags = summary_all.groupby(["model", "task", "layer", "variant"]).agg(
                mp_reliable_flag=("mp_reliable_flag", "max"),
            ).reset_index()
            pivot = pivot.merge(flags, on=["model", "task", "layer", "variant"], how="left")
        pivot.to_csv(out_dir / fname, index=False)
        print(f"wrote {fname}")

    # 3. Top correlate per (model, task, layer, mode) — variant=merged only (sweep is merged-only)
    if "top_corr_concept" in summary_all.columns:
        tc = summary_all[summary_all["variant"] == "merged"].copy()
        tc = tc[["model", "task", "mode", "layer", "top_corr_concept", "top_corr_rho", "top_corr_q", "n_correlation_flags"]]
        tc.to_csv(out_dir / "residual_top_correlate_cross_mode.csv", index=False)
        print("wrote residual_top_correlate_cross_mode.csv")

    # 4. variant delta: generous - merged
    merged_df = summary_all[summary_all["variant"] == "merged"].set_index(["model", "task", "mode", "layer"])
    gen_df    = summary_all[summary_all["variant"] == "generous"].set_index(["model", "task", "mode", "layer"])
    key_intersect = merged_df.index.intersection(gen_df.index)
    if len(key_intersect):
        delta = pd.DataFrame(index=key_intersect)
        for col in ("var_explained", "n_above_mp", "k_union", "gamma", "top_eigenvalue"):
            if col in merged_df.columns and col in gen_df.columns:
                delta[f"delta_{col}"] = gen_df.loc[key_intersect, col] - merged_df.loc[key_intersect, col]
        delta = delta.reset_index()
        delta.to_csv(out_dir / "variant_delta.csv", index=False)
        print(f"wrote variant_delta.csv ({len(delta)} rows)")

    # 5. Matched-population subset (cells where all 3 modes have status=fit_ok in Step 6 LDA-A)
    matched_path = results_root / "lda_subspaces" / "comparison" / "matched_population_cells.csv"
    if matched_path.exists():
        matched = pd.read_csv(matched_path)
        # matched_population cells are per-concept; we don't have a per-concept Step-7 row.
        # We summarise: for each (model, task, layer), count how many concepts are in matched_population.
        match_count = matched.groupby(["model_key", "task", "layer"]).size().reset_index(name="n_matched_concepts")
        match_count = match_count.rename(columns={"model_key": "model"})
        merge_targets = summary_all[summary_all["variant"] == "merged"]
        ann = merge_targets.merge(match_count, on=["model", "task", "layer"], how="left")
        ann.to_csv(out_dir / "summary_with_matched_count.csv", index=False)
        print("wrote summary_with_matched_count.csv")

    print("aggregator done.")


if __name__ == "__main__":
    main()
