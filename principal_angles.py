"""Step 8 — Principal angles between concept subspaces (port of arithmetic-geometry Phase F).

Per (model, task, mode, layer) cell:
  - Load every concept's LDA Option A basis (lifted to 4096-D).
  - For every unordered pair (a, b) of concepts with non-empty bases:
      * principal angles via SVD of B_a @ B_b.T
      * angle_1..angle_5, angle_median, angle_max in degrees
  - 1000-trial empirical random baseline cached by (min(dim_a, dim_b), max(dim_a, dim_b)).
  - Superposition flag: angle_1 < baseline.p5 − 10°.
  - Per-pair empirical p-value and BH-FDR across the (n_pairs) grid.

Self-angle check: every concept basis vs itself; assert all top angles < 1°.

Resume logic via metadata.json. Atomic writes.

Usage:
  python principal_angles.py --config /home/anshulk/emnlp2026/config.yaml \
      --model gpt-j-6b --task addition --mode off --all-layers
"""

import argparse
import json
import logging
import os
import tempfile
import time
from logging.handlers import WatchedFileHandler
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import false_discovery_control

from ccsvd_subspaces import cell_seed
from residual_hunting import (
    atomic_json,
    atomic_save,
    eligible_concepts,
    lda_a_basis_path,
    load_basis_rows,
    load_concept_filter,
)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

N_RANDOM_BASELINE_TRIALS = 1000
SUPERPOSITION_MARGIN_DEG = 10.0
SELF_ANGLE_TOLERANCE_DEG = 1.0
AMBIENT_D = 4096
N_TOP_ANGLES_REPORT = 5
FDR_ALPHA = 0.05


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(logs_root: Path, model_key: str, task: str, mode: str) -> logging.Logger:
    logs_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"principal_angles.{model_key}.{task}.{mode}")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = WatchedFileHandler(logs_root / f"principal_angles_{model_key}_{task}_{mode}.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


# ──────────────────────────────────────────────────────────────────────────────
# Principal-angle computation
# ──────────────────────────────────────────────────────────────────────────────

def orthonormalise_basis(B: np.ndarray) -> np.ndarray:
    """Return a row-orthonormal basis spanning the same subspace as B.

    LDA Option A directions are unit-norm but not column-orthogonal (LDA solves a
    generalised eigenproblem; eigenvectors are orthogonal in the S_T metric).
    Principal-angle SVD requires row-orthonormal inputs, so we QR the transpose.
    Returns shape (rank, D) where rank ≤ B.shape[0].
    """
    if B.shape[0] == 0:
        return B
    Q, R = np.linalg.qr(B.T)        # B.T is (D, n_rows); Q is (D, n_rows)
    # Drop rank-deficient columns (numerical safety)
    diag = np.abs(np.diag(R))
    keep = diag > 1e-10 * (diag.max() if diag.size else 1.0)
    return Q[:, keep].T.astype(np.float32, copy=False)


def principal_angles_deg(B_a: np.ndarray, B_b: np.ndarray) -> np.ndarray:
    """Principal angles in degrees, ascending. Re-orthonormalises both bases on the fly."""
    Ba = orthonormalise_basis(B_a)
    Bb = orthonormalise_basis(B_b)
    if Ba.shape[0] == 0 or Bb.shape[0] == 0:
        return np.array([], dtype=np.float64)
    M = Ba @ Bb.T                                              # (m_a, m_b)
    S = np.linalg.svd(M, compute_uv=False)
    S = np.clip(S, -1.0, 1.0)
    return np.rad2deg(np.arccos(S))


# ──────────────────────────────────────────────────────────────────────────────
# Random baseline (1000 trials per (dim_a, dim_b), cached)
# ──────────────────────────────────────────────────────────────────────────────

class BaselineCache:
    """1000-trial empirical principal-angle null for (dim_a, dim_b) in R^D.

    Persists to disk; the disk cache survives across slices and runs.
    """

    def __init__(self, cache_path: Path, ambient_d: int = AMBIENT_D, n_trials: int = N_RANDOM_BASELINE_TRIALS):
        self.cache_path = cache_path
        self.ambient_d = ambient_d
        self.n_trials = n_trials
        self.mem: dict[tuple[int, int], dict] = {}
        if cache_path.exists():
            try:
                blob = np.load(cache_path, allow_pickle=True).item()
                self.mem = {tuple(k): v for k, v in blob.items()}
            except Exception:
                self.mem = {}

    def _flush(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".npy", dir=str(self.cache_path.parent))
        os.close(fd)
        # numpy expects np-arr-friendly object; we use a 0-d object array.
        np.save(tmp, np.array({k: v for k, v in self.mem.items()}, dtype=object))
        os.replace(tmp, self.cache_path)

    def get(self, dim_a: int, dim_b: int, seed: int) -> dict:
        key = (min(int(dim_a), int(dim_b)), max(int(dim_a), int(dim_b)))
        if key in self.mem:
            return self.mem[key]
        # Compute baseline: 1000 trials of θ_1 between two random orthonormal bases of dim_a, dim_b in R^D.
        rng = np.random.default_rng(seed)
        thetas = np.empty(self.n_trials, dtype=np.float64)
        D = self.ambient_d
        for t in range(self.n_trials):
            A = rng.standard_normal((D, key[0]))
            Q_A, _ = np.linalg.qr(A)
            B = rng.standard_normal((D, key[1]))
            Q_B, _ = np.linalg.qr(B)
            M = Q_A.T @ Q_B
            S = np.linalg.svd(M, compute_uv=False)
            S = np.clip(S, -1.0, 1.0)
            thetas[t] = float(np.rad2deg(np.arccos(S[0])))
        rec = {
            "n_trials": int(self.n_trials),
            "ambient_d": int(self.ambient_d),
            "theta1_mean": float(thetas.mean()),
            "theta1_std": float(thetas.std()),
            "theta1_p1": float(np.percentile(thetas, 1)),
            "theta1_p5": float(np.percentile(thetas, 5)),
            "thetas": thetas,        # keep full distribution for empirical p-values
        }
        self.mem[key] = rec
        self._flush()
        return rec


# ──────────────────────────────────────────────────────────────────────────────
# Per-cell runner
# ──────────────────────────────────────────────────────────────────────────────

def tier_of(concept: str) -> str:
    """Rough tier label (best-effort heuristic for plotting; not authoritative)."""
    if concept.startswith("ans_") or concept == "answer":
        return "tier1_answer"
    if concept in {"a", "b", "a_units", "b_units", "a_tens", "b_tens", "a_num_digits", "b_num_digits"}:
        return "tier1_operand"
    if "carry" in concept or "column_sum" in concept or "partial_product" in concept or "running_sum" in concept:
        return "tier2_column"
    if "parity" in concept or "magnitude" in concept or "zero" in concept or "ends_in_zero" in concept or "_eq_" in concept:
        return "tier3_structural"
    if concept in {"max_operand", "min_operand", "operand_diff", "operand_abs_diff", "larger_operand"} \
            or "both_" in concept or "either_" in concept:
        return "tier4_relational"
    if "__" in concept:
        return "joint"
    return "other"


def run_cell(
    cfg: dict, model: str, task: str, mode: str, layer: int,
    baseline: BaselineCache,
    logger: logging.Logger,
) -> dict:
    results_root = Path(cfg["paths"]["results_root"])
    cell_dir = results_root / "principal_angles" / model / task / f"mode_{mode}" / f"layer_{layer:02d}"
    meta_path = cell_dir / "metadata.json"
    if meta_path.exists():
        try:
            cached = json.loads(meta_path.read_text())
            if cached.get("computation_status") == "complete":
                logger.info(f"[skip] cached {model}/{task}/mode_{mode}/layer_{layer:02d}")
                return cached.get("summary_row", {})
        except Exception:
            pass

    t0 = time.time()
    cell_dir.mkdir(parents=True, exist_ok=True)
    filter_dict = load_concept_filter(results_root, model, task, mode, layer)
    eligible = eligible_concepts(filter_dict)

    # Load every eligible concept's LDA-A basis
    bases: dict[str, np.ndarray] = {}
    for c in eligible:
        B = load_basis_rows(lda_a_basis_path(results_root, model, task, layer, c, mode))
        if B.shape[0] > 0:
            bases[c] = B
    concepts = sorted(bases.keys())
    n_c = len(concepts)
    logger.info(f"[run] {model}/{task}/mode_{mode}/layer_{layer:02d}: {n_c} eligible concept bases")

    # Self-angle sanity check
    self_rows = []
    for c in concepts:
        ang = principal_angles_deg(bases[c], bases[c])
        ok = bool(np.max(ang[: min(N_TOP_ANGLES_REPORT, ang.size)]) < SELF_ANGLE_TOLERANCE_DEG)
        self_rows.append({
            "concept": c, "dim": int(bases[c].shape[0]),
            "max_angle_deg": float(np.max(ang)) if ang.size else 0.0,
            "self_angle_ok": ok,
        })
    pd.DataFrame(self_rows).to_csv(cell_dir / "self_angles.csv", index=False)
    n_self_fail = int(sum(1 for r in self_rows if not r["self_angle_ok"]))
    if n_self_fail:
        logger.warning(f"  self-angle check: {n_self_fail} concept(s) exceed {SELF_ANGLE_TOLERANCE_DEG}°")

    # Pairwise angles
    rows = []
    seed = cell_seed(model, task, layer, f"baseline_{mode}", base_seed=42)
    for i in range(n_c):
        for j in range(i + 1, n_c):
            ca, cb = concepts[i], concepts[j]
            Ba, Bb = bases[ca], bases[cb]
            ang = principal_angles_deg(Ba, Bb)
            top_k = min(N_TOP_ANGLES_REPORT, ang.size)
            base = baseline.get(Ba.shape[0], Bb.shape[0], seed=seed + i * 10007 + j)
            # Empirical p-value: fraction of null with θ1 ≤ observed
            theta1 = float(ang[0])
            null_dist = base["thetas"]
            p_perm = float((null_dist <= theta1 + 1e-12).sum() / len(null_dist))
            superposition_flag = theta1 < base["theta1_p5"] - SUPERPOSITION_MARGIN_DEG
            row = {
                "concept_a": ca, "concept_b": cb,
                "tier_a": tier_of(ca), "tier_b": tier_of(cb),
                "dim_a": int(Ba.shape[0]), "dim_b": int(Bb.shape[0]),
                "n_angles": int(ang.size),
                "angle_median": float(np.median(ang)),
                "angle_max": float(np.max(ang)),
                "baseline_theta1_mean": float(base["theta1_mean"]),
                "baseline_theta1_std": float(base["theta1_std"]),
                "baseline_theta1_p5": float(base["theta1_p5"]),
                "baseline_theta1_p1": float(base["theta1_p1"]),
                "perm_p": p_perm,
                "superposition_flag": bool(superposition_flag),
            }
            for k in range(N_TOP_ANGLES_REPORT):
                row[f"angle_{k+1}"] = float(ang[k]) if k < ang.size else float("nan")
            rows.append(row)

    df = pd.DataFrame(rows)
    # FDR-correct empirical p-values across all pairs in the cell
    if len(df):
        df["fdr_q"] = false_discovery_control(df["perm_p"].to_numpy(), method="bh")
    df.to_csv(cell_dir / "angles_pairwise.csv", index=False)

    summary = {
        "model": model, "task": task, "mode": mode, "layer": int(layer),
        "n_concepts_kept": n_c,
        "n_pairs": int(len(df)),
        "n_superposition_flags": int(df["superposition_flag"].sum()) if len(df) else 0,
        "n_fdr_q_below_alpha": int((df["fdr_q"] < FDR_ALPHA).sum()) if len(df) else 0,
        "median_angle_1": float(df["angle_1"].median()) if len(df) else float("nan"),
        "median_angle_5": float(df["angle_5"].median()) if len(df) else float("nan"),
        "self_angle_failures": n_self_fail,
        "runtime_seconds": round(time.time() - t0, 2),
        "status": "ok",
    }
    atomic_json({"computation_status": "complete", "summary_row": summary,
                 "n_unique_dim_pairs_used": len(baseline.mem)}, meta_path)
    logger.info(f"[done] pairs={len(df)} flags={summary['n_superposition_flags']} "
                f"q<{FDR_ALPHA}={summary['n_fdr_q_below_alpha']} "
                f"({time.time()-t0:.1f}s)")
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=["addition", "multiplication"])
    ap.add_argument("--mode", required=True, choices=["off", "answer", "norm"])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--layer", type=int)
    g.add_argument("--all-layers", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    paths = cfg["paths"]
    logs_root = Path(paths["logs_root"])
    results_root = Path(paths["results_root"])
    model_cfg = next(m for m in cfg["models"] if m["key"] == args.model)
    layers = [args.layer] if args.layer is not None else model_cfg["layers"]

    logger = setup_logging(logs_root, args.model, args.task, args.mode)
    logger.info(f"=== Step 8 principal_angles: {args.model}/{args.task}/{args.mode} layers={layers} ===")

    baseline = BaselineCache(results_root / "principal_angles" / "random_baseline_cache.npy",
                             ambient_d=AMBIENT_D, n_trials=N_RANDOM_BASELINE_TRIALS)

    all_summaries = []
    for layer in layers:
        s = run_cell(cfg, args.model, args.task, args.mode, layer, baseline, logger)
        if s: all_summaries.append(s)

    if all_summaries:
        per_model_dir = results_root / "principal_angles" / args.model
        per_model_dir.mkdir(parents=True, exist_ok=True)
        out_csv = per_model_dir / f"summary_{args.model}_{args.task}_mode_{args.mode}.csv"
        df_new = pd.DataFrame(all_summaries)
        if out_csv.exists():
            df_old = pd.read_csv(out_csv)
            key = ["model", "task", "mode", "layer"]
            df_old = df_old[~df_old.set_index(key).index.isin(df_new.set_index(key).index)]
            df_new = pd.concat([df_old, df_new], ignore_index=True)
        df_new = df_new.sort_values(["task", "mode", "layer"]).reset_index(drop=True)
        df_new.to_csv(out_csv, index=False)
        logger.info(f"wrote {out_csv} ({len(df_new)} rows)")

    logger.info(f"=== Step 8 DONE ===")


if __name__ == "__main__":
    main()
