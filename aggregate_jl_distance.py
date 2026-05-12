"""Step 9 aggregator — concatenate per-model JL summaries and emit cross-mode tables.

Inputs:
  results/jl_distance/{model}/summary_{model}_{task}_mode_{mode}.csv

Outputs (under results/jl_distance/comparison/):
  summary_all.csv                # all rows
  spearman_cross_mode.csv        # pivot per (model, task, layer, variant) across modes
  pearson_cross_mode.csv
  distance_var_explained_cross_mode.csv
  pyth_max_rel_error_cross_cell.csv  # numerical-correctness audit per cell
  variant_delta_jl.csv           # generous − merged per cell
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
    base = results_root / "jl_distance"
    out_dir = base / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_paths = list(base.glob("*/summary_*_mode_*.csv"))
    if not summary_paths:
        print(f"no summary CSVs found under {base}")
        return
    summary_all = pd.concat([pd.read_csv(p) for p in summary_paths], ignore_index=True)
    summary_all = summary_all.sort_values(
        ["model", "task", "mode", "layer", "variant"]
    ).reset_index(drop=True)
    summary_all.to_csv(out_dir / "summary_all.csv", index=False)
    print(f"wrote summary_all.csv ({len(summary_all)} rows)")

    for metric, fname in [
        ("spearman_rho",          "spearman_cross_mode.csv"),
        ("pearson_r",             "pearson_cross_mode.csv"),
        ("distance_var_explained", "distance_var_explained_cross_mode.csv"),
        ("mean_rel_error",        "mean_rel_error_cross_mode.csv"),
        ("max_rel_error",         "max_rel_error_cross_mode.csv"),
        ("pyth_max_rel_error",    "pyth_max_rel_error_cross_cell.csv"),
    ]:
        if metric not in summary_all.columns:
            continue
        pivot = summary_all.pivot_table(
            index=["model", "task", "layer", "variant"],
            columns="mode", values=metric, aggfunc="first",
        ).reset_index()
        pivot.columns.name = None
        pivot.to_csv(out_dir / fname, index=False)
        print(f"wrote {fname}")

    # Variant delta (generous − merged) per cell
    merged_df = summary_all[summary_all["variant"] == "merged"].set_index(["model", "task", "mode", "layer"])
    gen_df    = summary_all[summary_all["variant"] == "generous"].set_index(["model", "task", "mode", "layer"])
    key_intersect = merged_df.index.intersection(gen_df.index)
    if len(key_intersect):
        delta = pd.DataFrame(index=key_intersect)
        for col in ("spearman_rho", "pearson_r", "distance_var_explained",
                    "mean_rel_error", "max_rel_error", "k_union"):
            if col in merged_df.columns and col in gen_df.columns:
                delta[f"delta_{col}"] = gen_df.loc[key_intersect, col] - merged_df.loc[key_intersect, col]
        delta = delta.reset_index()
        delta.to_csv(out_dir / "variant_delta_jl.csv", index=False)
        print(f"wrote variant_delta_jl.csv ({len(delta)} rows)")

    print("aggregator done.")


if __name__ == "__main__":
    main()
