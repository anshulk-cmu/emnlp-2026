"""Step 6 — Residualize activations against a scalar (answer or norm).

For each (model, task, layer), compute residualized activations under three modes:
  - off    : passthrough copy (no residualization)
  - answer : OLS-regress activations on the gold answer (a+b or a*b), keep residual
  - norm   : OLS-regress activations on the L2 norm of each activation, keep residual

OLS recipe (per dimension d_i):
  z_c = z - mean(z)
  X_c = X - X.mean(axis=0)
  beta_i = (X_c[:, i] · z_c) / (z_c · z_c)            # one slope per activation dim
  X_resid[:, i] = X[:, i] - z_c * beta_i              # mean of X preserved

Output preserves X's original mean; only the linear contribution of z is removed.
Downstream CCSVD does its own centering, so this choice is benign.

Outputs:
  data/results/residualized/{model}/{task}_layer_{LL}_mode_{mode}.npy   (N, d) float32
  data/results/residualized/{model}/{task}_layer_{LL}_residual_meta.json (one per layer)

Run on full data — no subsampling. Uses cupy when available, falls back to numpy.

Usage:
  python residualize_activations.py --config /home/anshulk/emnlp2026/config.yaml --model gpt-j-6b
  python residualize_activations.py --config /home/anshulk/emnlp2026/config.yaml --model gpt-j-6b \
      --modes off,answer,norm --tasks addition,multiplication
"""

import argparse
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from logging.handlers import WatchedFileHandler
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ───────────────────────────────────────────────────────────────────────────────
# GPU / CPU dispatch
# ───────────────────────────────────────────────────────────────────────────────

def get_xp():
    """Return (xp, on_gpu) where xp is cupy if available else numpy."""
    try:
        import cupy as cp
        # Touch a tiny op to confirm CUDA actually works on this node.
        _ = cp.asarray(np.zeros(2, dtype=np.float32))
        return cp, True
    except Exception:
        return np, False


# ───────────────────────────────────────────────────────────────────────────────
# OLS residualization (one scalar against (N, d) activation matrix)
# ───────────────────────────────────────────────────────────────────────────────

def residualize_against_scalar(X: np.ndarray, z: np.ndarray, on_gpu: bool):
    """Return residual = X - (z - mean(z)) * beta, where beta = (X_c.T @ z_c) / (z_c @ z_c).

    X: (N, d) float32 numpy array.
    z: (N,) float64 numpy array.

    Mean of X is preserved in the residual.
    """
    if on_gpu:
        import cupy as cp
        X_g = cp.asarray(X, dtype=cp.float32)
        z_g = cp.asarray(z, dtype=cp.float64)
        z_c = z_g - z_g.mean()
        z_dot_z = float((z_c * z_c).sum())
        if z_dot_z < 1e-12:
            # z is (numerically) constant; nothing to remove. Return raw copy.
            return cp.asnumpy(X_g.copy()), 0.0
        X_mean = X_g.mean(axis=0).astype(cp.float64)
        X_c = X_g.astype(cp.float64) - X_mean
        # beta_i = sum_n (X_c[n, i] * z_c[n]) / z_dot_z
        beta = (X_c.T @ z_c) / z_dot_z                      # (d,)
        # X_resid = X - outer(z_c, beta), preserving X's original mean
        residual = X_g.astype(cp.float64) - cp.outer(z_c, beta)
        return cp.asnumpy(residual.astype(cp.float32)), z_dot_z
    else:
        z_c = z - z.mean()
        z_dot_z = float((z_c * z_c).sum())
        if z_dot_z < 1e-12:
            return X.astype(np.float32, copy=True), 0.0
        X_mean = X.mean(axis=0).astype(np.float64)
        X_c = X.astype(np.float64) - X_mean
        beta = (X_c.T @ z_c) / z_dot_z                      # (d,)
        residual = X.astype(np.float64) - np.outer(z_c, beta)
        return residual.astype(np.float32), z_dot_z


# ───────────────────────────────────────────────────────────────────────────────
# Logging
# ───────────────────────────────────────────────────────────────────────────────

def setup_logging(logs_root: Path, model_key: str) -> logging.Logger:
    logs_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"residualize.{model_key}")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = WatchedFileHandler(logs_root / f"residualize_activations_{model_key}.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, help="Model key from config.")
    parser.add_argument("--modes", default="off,answer,norm",
                        help="Comma-separated subset of {off, answer, norm}.")
    parser.add_argument("--tasks", default="addition,multiplication")
    parser.add_argument("--force", action="store_true",
                        help="Re-write cache files even if they already exist.")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    paths = cfg["paths"]
    data_root = Path(paths["data_root"])
    results_root = Path(paths["results_root"])
    logs_root = Path(paths["logs_root"])
    out_root = results_root / "residualized" / args.model
    out_root.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(logs_root, args.model)
    xp, on_gpu = get_xp()
    backend = "cupy" if on_gpu else "numpy"

    model_cfg = next((m for m in cfg["models"] if m["key"] == args.model), None)
    if model_cfg is None:
        logger.error("Model %s not in config", args.model)
        sys.exit(2)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    valid_modes = {"off", "answer", "norm"}
    bad = [m for m in modes if m not in valid_modes]
    if bad:
        logger.error("Unknown modes: %s; valid: %s", bad, sorted(valid_modes))
        sys.exit(2)

    layers = model_cfg["layers"]

    logger.info("=" * 78)
    logger.info("residualize_activations — model=%s backend=%s", args.model, backend)
    logger.info("modes=%s tasks=%s layers=%s", modes, tasks, layers)

    t_run0 = time.time()
    summary_rows = []

    for task in tasks:
        problems_path = data_root / "data" / "raw" / f"{task}_problems.csv"
        if not problems_path.exists():
            logger.warning("missing problems CSV: %s — skipping task", problems_path)
            continue
        problems_df = pd.read_csv(problems_path)
        if "answer" not in problems_df.columns:
            logger.error("'answer' column missing from %s", problems_path)
            sys.exit(3)
        # Gold answer per problem (deterministic from a, b for both tasks).
        z_answer = problems_df["answer"].to_numpy().astype(np.float64)

        for layer in layers:
            act_path = data_root / "activations" / args.model / f"{task}_layer_{layer:02d}.npy"
            if not act_path.exists():
                logger.warning("missing activations: %s — skipping layer", act_path)
                continue

            t_layer = time.time()
            X = np.load(act_path)
            if X.dtype != np.float32:
                X = X.astype(np.float32)
            N, d = X.shape
            logger.info("[%s/layer_%02d] X=%s  ans_var=%.3f", task, layer, X.shape,
                        float(np.var(z_answer)))

            for mode in modes:
                out_path = out_root / f"{task}_layer_{layer:02d}_mode_{mode}.npy"
                if out_path.exists() and not args.force:
                    logger.info("  mode=%s already exists at %s — skipping (use --force to overwrite)",
                                mode, out_path)
                    summary_rows.append({
                        "model": args.model, "task": task, "layer": layer, "mode": mode,
                        "status": "exists", "out_path": str(out_path),
                    })
                    continue
                t0 = time.time()
                if mode == "off":
                    residual = X.copy()
                    z_dot_z = 0.0
                    note = "passthrough"
                elif mode == "answer":
                    residual, z_dot_z = residualize_against_scalar(X, z_answer, on_gpu=on_gpu)
                    note = f"answer (z_dot_z={z_dot_z:.4e})"
                elif mode == "norm":
                    z_norm = np.linalg.norm(X, axis=1).astype(np.float64)
                    residual, z_dot_z = residualize_against_scalar(X, z_norm, on_gpu=on_gpu)
                    note = f"L2 norm (z_dot_z={z_dot_z:.4e})"
                else:
                    raise RuntimeError(f"unreachable mode: {mode}")

                # Atomic write: tempfile in same directory, then rename.
                # NOTE: np.save appends ".npy" if absent — so tmp must already end in .npy.
                import tempfile, os
                fd, tmp_str = tempfile.mkstemp(suffix=".npy", dir=str(out_path.parent))
                os.close(fd)
                np.save(tmp_str, residual.astype(np.float32))
                os.replace(tmp_str, out_path)
                dt = time.time() - t0
                logger.info("  mode=%-7s  N=%d d=%d  %s  wrote=%s  dt=%.1fs",
                            mode, N, d, note, out_path.name, dt)
                summary_rows.append({
                    "model": args.model, "task": task, "layer": layer, "mode": mode,
                    "status": "ok", "out_path": str(out_path),
                    "N": int(N), "d": int(d), "z_dot_z": float(z_dot_z),
                    "elapsed_s": round(dt, 3),
                })
            logger.info("  layer %d done in %.1fs", layer, time.time() - t_layer)

    # Per-model manifest
    manifest = {
        "model_key": args.model,
        "modes": modes,
        "tasks": tasks,
        "layers": layers,
        "n_files_written": len(summary_rows),
        "backend": backend,
        "library_versions": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "python": platform.python_version(),
        },
        "config_path": args.config,
        "total_runtime_seconds": round(time.time() - t_run0, 2),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": summary_rows,
    }
    manifest_path = out_root / f"residualize_manifest_{args.model}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("wrote %s", manifest_path)
    logger.info("DONE in %.1fs", time.time() - t_run0)


if __name__ == "__main__":
    main()
