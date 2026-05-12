"""Pre-flight check for Steps 7-9.

Asserts the on-disk state of Step 5 (CCSVD) and Step 6 (LDA + cross-mode
aggregator) is sufficient to launch the residual-hunting / principal-angles /
JL chain. Exits 0 on success, non-zero on any failure with a descriptive
message.

Usage:
  python check_step6_complete.py --config /home/anshulk/emnlp2026/config.yaml
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml


REQUIRED_COMPARISON_CSVS = [
    "cross_mode_summary.csv",
    "matched_population_cells.csv",
    "carveout_log.csv",
    "cross_mode_alignment.csv",
    "cross_mode_lambda_deltas.csv",
    "cross_mode_accuracy_deltas.csv",
    "a_vs_b_alignment.csv",
]


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"OK:   {msg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    results_root = Path(cfg["paths"]["results_root"])
    data_root = Path(cfg["paths"]["data_root"])
    models = [m["key"] for m in cfg["models"]]
    modes = cfg["lda"]["modes"]

    # 1. Step 6 per-mode LDA summary CSVs (Option A).
    for model in models:
        for mode in modes:
            p = results_root / "lda_subspaces" / "subspace_lda" / f"mode_{mode}" / model / f"summary_{model}_mode_{mode}.csv"
            if not p.exists():
                fail(f"missing LDA Option-A summary: {p}")
            df = pd.read_csv(p)
            if len(df) == 0:
                fail(f"empty LDA Option-A summary: {p}")
            ok(f"LDA-A summary present ({len(df)} rows): {p.relative_to(results_root)}")

    # 2. Step 6 per-mode LDA summary CSVs (Option B).
    for model in models:
        for mode in modes:
            p = results_root / "lda_subspaces" / "full_lda" / f"mode_{mode}" / model / f"summary_{model}_mode_{mode}.csv"
            if not p.exists():
                fail(f"missing LDA Option-B summary: {p}")
    ok(f"LDA-B summaries present ({len(models)*len(modes)} files)")

    # 3. Cross-mode comparison CSVs.
    comp_dir = results_root / "lda_subspaces" / "comparison"
    if not comp_dir.exists():
        fail(f"missing comparison directory: {comp_dir}")
    for fname in REQUIRED_COMPARISON_CSVS:
        p = comp_dir / fname
        if not p.exists():
            fail(f"missing comparison CSV: {p}")
    matched = pd.read_csv(comp_dir / "matched_population_cells.csv")
    if len(matched) < 1000:
        fail(f"matched_population_cells.csv has only {len(matched)} rows (expected ≥ 1000)")
    ok(f"comparison CSVs all present (matched_population: {len(matched)} cells)")

    # 4. Per-mode CCSVD basis directories.
    for model in models:
        # mode=off lives at the root of ccsvd_subspaces.
        off_dir = results_root / "ccsvd_subspaces" / model
        if not off_dir.exists():
            fail(f"missing mode=off CCSVD tree: {off_dir}")
        # answer / norm under mode_X.
        for mode in ("answer", "norm"):
            d = results_root / "ccsvd_subspaces" / f"mode_{mode}" / model
            if not d.exists():
                fail(f"missing mode={mode} CCSVD tree: {d}")
    ok(f"CCSVD per-mode trees present for all {len(models)} models")

    # 5. Residualized activation caches.
    for model in models:
        for mode in ("answer", "norm"):
            for task in ("addition", "multiplication"):
                # Sample one layer (headline) to check; full coverage is asserted by per-cell loads later.
                model_cfg = next(m for m in cfg["models"] if m["key"] == model)
                L = model_cfg["headline_layer"]
                p = results_root / "residualized" / model / f"{task}_layer_{L:02d}_mode_{mode}.npy"
                if not p.exists():
                    fail(f"missing residualized cache: {p}")
    ok(f"residualized activation caches present (sampled headline layer per model/task/mode)")

    # 6. Raw activations + answers CSVs (for mode=off).
    for model in models:
        for task in ("addition", "multiplication"):
            model_cfg = next(m for m in cfg["models"] if m["key"] == model)
            L = model_cfg["headline_layer"]
            p = data_root / "activations" / model / f"{task}_layer_{L:02d}.npy"
            if not p.exists():
                fail(f"missing raw activations: {p}")
            ans = data_root / "answers" / model / f"{task}_answers.csv"
            if not ans.exists():
                fail(f"missing answers CSV: {ans}")
    ok("raw activations + answer CSVs present (sampled)")

    # 7. Problems CSVs.
    for task in ("addition", "multiplication"):
        p = data_root / "data" / "raw" / f"{task}_problems.csv"
        if not p.exists():
            fail(f"missing problems CSV: {p}")
    ok("problems CSVs present")

    print("\nALL CHECKS PASSED — Steps 7/8/9 may launch.")


if __name__ == "__main__":
    main()
