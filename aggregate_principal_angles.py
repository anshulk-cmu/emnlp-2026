"""Step 8 aggregator — concatenate per-cell pairwise-angle CSVs and emit summary tables.

Inputs:
  results/principal_angles/{model}/{task}/mode_{mode}/layer_LL/angles_pairwise.csv
  results/principal_angles/{model}/summary_{model}_{task}_mode_{mode}.csv

Outputs (under results/principal_angles/comparison/):
  pairwise_all.csv                  # every (cell, pair) row stacked
  summary_all.csv                   # cell-level summary stacked
  superposition_rate_by_cell.csv    # n_pairs and superposition fraction per cell
  superposition_by_tier_pair.csv    # superposition fraction by (tier_a, tier_b)
  cross_mode_superposition.csv      # pivot of flag count across modes
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
    base = results_root / "principal_angles"
    out_dir = base / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pairwise rows (potentially large — ~100k rows total)
    pair_paths = list(base.glob("*/*/mode_*/layer_*/angles_pairwise.csv"))
    if not pair_paths:
        print(f"no angles_pairwise.csv under {base}")
        return
    rows = []
    for p in pair_paths:
        parts = p.parts
        # .../principal_angles/{model}/{task}/mode_{mode}/layer_LL/angles_pairwise.csv
        try:
            base_idx = parts.index("principal_angles")
            model = parts[base_idx + 1]
            task = parts[base_idx + 2]
            mode = parts[base_idx + 3].replace("mode_", "")
            layer = int(parts[base_idx + 4].replace("layer_", ""))
        except (ValueError, IndexError):
            continue
        df = pd.read_csv(p)
        df["model"] = model; df["task"] = task; df["mode"] = mode; df["layer"] = layer
        rows.append(df)
    pairwise = pd.concat(rows, ignore_index=True)
    pairwise.to_csv(out_dir / "pairwise_all.csv", index=False)
    print(f"wrote pairwise_all.csv ({len(pairwise)} rows)")

    # 2. Cell-level summary
    summary_paths = list(base.glob("*/summary_*_mode_*.csv"))
    if summary_paths:
        sdf = pd.concat([pd.read_csv(p) for p in summary_paths], ignore_index=True)
        sdf = sdf.sort_values(["model", "task", "mode", "layer"]).reset_index(drop=True)
        sdf.to_csv(out_dir / "summary_all.csv", index=False)
        print(f"wrote summary_all.csv ({len(sdf)} rows)")

    # 3. Superposition rate per cell
    by_cell = pairwise.groupby(["model", "task", "mode", "layer"]).agg(
        n_pairs=("superposition_flag", "size"),
        n_flags=("superposition_flag", "sum"),
        median_angle_1=("angle_1", "median"),
        median_angle_5=("angle_5", "median"),
        n_fdr_q_below_0p05=("fdr_q", lambda x: (x < 0.05).sum()),
    ).reset_index()
    by_cell["superposition_rate"] = by_cell["n_flags"] / by_cell["n_pairs"].clip(lower=1)
    by_cell.to_csv(out_dir / "superposition_rate_by_cell.csv", index=False)
    print("wrote superposition_rate_by_cell.csv")

    # 4. Superposition by tier-pair
    if "tier_a" in pairwise.columns and "tier_b" in pairwise.columns:
        tp = pairwise.copy()
        # canonicalise tier-pair ordering
        tier_pair = pd.Series(list(zip(tp["tier_a"], tp["tier_b"]))).apply(lambda t: tuple(sorted(t)))
        tp["tier_pair"] = tier_pair.values
        by_tier = tp.groupby(["model", "task", "mode", "layer", "tier_pair"]).agg(
            n_pairs=("superposition_flag", "size"),
            n_flags=("superposition_flag", "sum"),
            median_angle_1=("angle_1", "median"),
        ).reset_index()
        by_tier["superposition_rate"] = by_tier["n_flags"] / by_tier["n_pairs"].clip(lower=1)
        by_tier.to_csv(out_dir / "superposition_by_tier_pair.csv", index=False)
        print("wrote superposition_by_tier_pair.csv")

    # 5. Cross-mode pivot
    pivot = by_cell.pivot_table(
        index=["model", "task", "layer"],
        columns="mode",
        values="superposition_rate",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    pivot.to_csv(out_dir / "cross_mode_superposition.csv", index=False)
    print("wrote cross_mode_superposition.csv")

    print("aggregator done.")


if __name__ == "__main__":
    main()
