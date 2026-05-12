"""Step 6 — Cross-mode and A-vs-B comparison aggregator.

Reads per-mode summary and basis files written by lda_subspaces.py.

Produces (under data/results/lda_subspaces/comparison/):
  cross_mode_summary.csv          — one row per (model, task, layer, concept) with
                                     all three modes' headline stats side by side
  cross_mode_alignment.csv        — cosine similarity of top-1 LDA direction
                                     across (mode_a, mode_b) pairs, per cell
  cross_mode_lambda_deltas.csv    — Δλ_T_1, Δn_sig, Δcv_accuracy across mode-pairs
  cross_mode_accuracy_deltas.csv  — focused on cv_accuracy deltas
  a_vs_b_alignment.csv            — concatenated A-vs-B per-mode alignments
  matched_population_cells.csv    — cells where LDA-A succeeded in ALL three modes
  carveout_log.csv                — cells carved out from any mode

Usage:
  python compare_residualization_modes.py --config /home/anshulk/emnlp2026/config.yaml
  python compare_residualization_modes.py --config ... --models gpt-j-6b
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from logging.handlers import WatchedFileHandler
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def setup_logging(logs_root: Path) -> logging.Logger:
    logs_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("compare_residualization_modes")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = WatchedFileHandler(logs_root / "compare_residualization_modes.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


def load_per_mode_summaries(A_root: Path, model: str, modes: list[str]) -> dict[str, pd.DataFrame]:
    """Load summary_*.csv per mode from the subspace_lda tree. Return {mode: df}."""
    out = {}
    for mode in modes:
        path = A_root / f"mode_{mode}" / model / f"summary_{model}_mode_{mode}.csv"
        if not path.exists():
            out[mode] = pd.DataFrame()
            continue
        out[mode] = pd.read_csv(path)
    return out


def load_top1_basis(A_root: Path, model: str, mode: str, task: str, layer: int, concept: str) -> np.ndarray | None:
    """Load LDA-A's top-1 direction lifted to 4096-D from disk. Returns None if missing."""
    path = (A_root / f"mode_{mode}" / model / task / f"layer_{layer:02d}" / concept / "lda_basis_full.npy")
    if not path.exists():
        return None
    try:
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] < 1:
            return None
        return arr[:, 0].astype(np.float64)
    except Exception:
        return None


def cos_sim(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    return float(abs(np.dot(u, v)) / (nu * nv))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", default=None,
                        help="Comma-separated model keys; default: all in config.")
    parser.add_argument("--modes", default=None,
                        help="Comma-separated modes; default: from config.lda.modes.")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    paths = cfg["paths"]
    cfg_lda = cfg["lda"]
    results_root = Path(paths["results_root"])
    logs_root = Path(paths["logs_root"])

    logger = setup_logging(logs_root)

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = [m["key"] for m in cfg["models"]]
    if args.modes:
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    else:
        modes = list(cfg_lda.get("modes", ["off", "answer", "norm"]))

    A_root = results_root / "lda_subspaces" / "subspace_lda"
    out_dir = results_root / "lda_subspaces" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 78)
    logger.info("compare_residualization_modes — models=%s modes=%s", models, modes)

    t0 = time.time()
    cross_summary_rows = []
    alignment_rows = []
    delta_rows = []
    accuracy_delta_rows = []
    matched_pop_rows = []
    carveout_rows = []
    a_vs_b_concat = []

    for model in models:
        per_mode_dfs = load_per_mode_summaries(A_root, model, modes)
        # Identify cell keys present across all modes.
        keyed = {}
        for mode, df in per_mode_dfs.items():
            if df.empty:
                logger.warning("[%s] no summary for mode=%s", model, mode)
                continue
            df_keyed = df.copy()
            df_keyed["__cell_key__"] = (
                df_keyed["task"].astype(str) + "|"
                + df_keyed["layer"].astype(str) + "|"
                + df_keyed["concept"].astype(str)
            )
            keyed[mode] = df_keyed.set_index("__cell_key__")

        if not keyed:
            logger.warning("[%s] no per-mode data; skipping model", model)
            continue

        all_keys = set()
        for d in keyed.values():
            all_keys.update(d.index)
        logger.info("[%s] cells observed across at least one mode: %d", model, len(all_keys))

        for ck in sorted(all_keys):
            task, layer_s, concept = ck.split("|", 2)
            layer = int(layer_s)
            row_xs = {"model_key": model, "task": task, "layer": layer, "concept": concept}
            statuses = {}
            n_sigs = {}
            lambdas = {}
            cv_accs = {}
            carved = {}
            for mode in modes:
                if mode in keyed and ck in keyed[mode].index:
                    r = keyed[mode].loc[ck]
                    statuses[mode] = r.get("status")
                    n_sigs[mode] = r.get("n_sig")
                    lambdas[mode] = r.get("lambda_T_1")
                    cv_accs[mode] = r.get("cv_accuracy_at_n_sig")
                    carved[mode] = bool(r.get("is_carved_out", False))
                else:
                    statuses[mode] = None
                    n_sigs[mode] = None
                    lambdas[mode] = None
                    cv_accs[mode] = None
                    carved[mode] = None

            for mode in modes:
                row_xs[f"status_{mode}"] = statuses[mode]
                row_xs[f"n_sig_{mode}"] = n_sigs[mode]
                row_xs[f"lambda_T_1_{mode}"] = lambdas[mode]
                row_xs[f"cv_accuracy_at_n_sig_{mode}"] = cv_accs[mode]
                row_xs[f"is_carved_out_{mode}"] = carved[mode]

            # Matched-population: LDA-A succeeded in all 3 modes (no carveout, no skip).
            ok_in_all = all(
                statuses.get(m) == "fit_ok" and not carved.get(m, False)
                for m in modes
            )
            row_xs["matched_population"] = ok_in_all
            cross_summary_rows.append(row_xs)
            if ok_in_all:
                matched_pop_rows.append(row_xs)
            if any(carved.values()):
                carveout_rows.append({**row_xs, "carved_modes": ",".join([m for m in modes if carved.get(m)])})

            # Pairwise mode alignment + deltas (using LDA-A top-1 basis lifted to 4096-D).
            if not ok_in_all:
                continue
            bases = {}
            for mode in modes:
                bases[mode] = load_top1_basis(A_root, model, mode, task, layer, concept)
            for i, ma in enumerate(modes):
                for mb in modes[i + 1:]:
                    if bases.get(ma) is None or bases.get(mb) is None:
                        sim = float("nan")
                    else:
                        sim = cos_sim(bases[ma], bases[mb])
                    alignment_rows.append({
                        **row_xs, "mode_a": ma, "mode_b": mb, "cos_sim_top1_AA": sim,
                    })
                    delta_rows.append({
                        **row_xs, "mode_a": ma, "mode_b": mb,
                        "delta_lambda_T_1": (lambdas[mb] - lambdas[ma]) if (lambdas[ma] is not None and lambdas[mb] is not None) else None,
                        "delta_n_sig": (n_sigs[mb] - n_sigs[ma]) if (n_sigs[ma] is not None and n_sigs[mb] is not None) else None,
                        "delta_cv_accuracy": (cv_accs[mb] - cv_accs[ma]) if (cv_accs[ma] is not None and cv_accs[mb] is not None) else None,
                    })
                    accuracy_delta_rows.append({
                        **row_xs, "mode_a": ma, "mode_b": mb,
                        "cv_accuracy_a": cv_accs[ma],
                        "cv_accuracy_b": cv_accs[mb],
                        "delta": (cv_accs[mb] - cv_accs[ma]) if (cv_accs[ma] is not None and cv_accs[mb] is not None) else None,
                    })

        # Concat A-vs-B alignments across modes for this model.
        for mode in modes:
            avb_path = out_dir / f"a_vs_b_alignment_{model}_mode_{mode}.csv"
            if avb_path.exists():
                a_vs_b_concat.append(pd.read_csv(avb_path))

    # ── Write everything ─────────────────────────────────────────────────────
    pd.DataFrame(cross_summary_rows).to_csv(out_dir / "cross_mode_summary.csv", index=False)
    pd.DataFrame(alignment_rows).to_csv(out_dir / "cross_mode_alignment.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(out_dir / "cross_mode_lambda_deltas.csv", index=False)
    pd.DataFrame(accuracy_delta_rows).to_csv(out_dir / "cross_mode_accuracy_deltas.csv", index=False)
    pd.DataFrame(matched_pop_rows).to_csv(out_dir / "matched_population_cells.csv", index=False)
    pd.DataFrame(carveout_rows).to_csv(out_dir / "carveout_log.csv", index=False)
    if a_vs_b_concat:
        pd.concat(a_vs_b_concat, ignore_index=True).to_csv(out_dir / "a_vs_b_alignment.csv", index=False)
    logger.info("Wrote outputs:")
    for fn in ["cross_mode_summary.csv", "cross_mode_alignment.csv",
               "cross_mode_lambda_deltas.csv", "cross_mode_accuracy_deltas.csv",
               "matched_population_cells.csv", "carveout_log.csv"]:
        path = out_dir / fn
        n = 0
        if path.exists() and path.stat().st_size > 0:
            try:
                n = len(pd.read_csv(path))
            except pd.errors.EmptyDataError:
                n = 0
        logger.info("  %s  (rows=%d)", path, n)

    manifest = {
        "models": models,
        "modes": modes,
        "n_cells_summary_rows": len(cross_summary_rows),
        "n_alignment_rows": len(alignment_rows),
        "n_matched_population": len(matched_pop_rows),
        "n_carveout": len(carveout_rows),
        "elapsed_s": round(time.time() - t0, 2),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(out_dir / "comparison_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info("DONE in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
