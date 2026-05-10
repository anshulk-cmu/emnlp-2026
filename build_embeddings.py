"""Step 4 Phase A — comprehensive UMAP + t-SNE embedding CSVs per cell.

For each of the 30 (model, task, layer) cells, computes:
  - 4 UMAP 2D embeddings (different n_neighbors / min_dist)
  - 3 t-SNE 2D embeddings (different perplexity)
  - trustworthiness score per (method, hp) on a fixed 2000-row subsample

Writes per-cell CSV (concept labels + activation_norm + 14 coord columns)
plus per-cell manifest JSON.

Usage:
  python build_embeddings.py --config /home/anshulk/emnlp2026/config.yaml

CPU-only. UMAP + t-SNE do not use the GPU.
"""

import argparse
import hashlib
import json
import logging
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.manifold import TSNE, trustworthiness

# Suppress umap-learn's noisy n_jobs warning
warnings.filterwarnings("ignore", message=".*n_jobs.*overridden.*")
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

import umap


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(logs_root: Path):
    logs_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("build_embeddings")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = RotatingFileHandler(
        logs_root / "build_embeddings.log",
        maxBytes=10_000_000, backupCount=3,
    )
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
    )
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


# ═══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER GRIDS
# ═══════════════════════════════════════════════════════════════════════════════

UMAP_HP = [
    ("umap2d_n15_md10", dict(n_neighbors=15, min_dist=0.1)),
    ("umap2d_n30_md10", dict(n_neighbors=30, min_dist=0.1)),
    ("umap2d_n50_md10", dict(n_neighbors=50, min_dist=0.1)),
    ("umap2d_n30_md30", dict(n_neighbors=30, min_dist=0.3)),
]
TSNE_HP = [
    ("tsne2d_p10", 10),
    ("tsne2d_p30", 30),
    ("tsne2d_p50", 50),
]
RANDOM_STATE = 42
TRUSTWORTHINESS_K = 30
TRUSTWORTHINESS_SUBSAMPLE = 2000


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fit_umap(X, n_neighbors, min_dist):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=RANDOM_STATE,
    )
    return reducer.fit_transform(X)


def fit_tsne(X, perplexity):
    n = X.shape[0]
    perp = min(perplexity, max(2, (n - 1) // 3))
    reducer = TSNE(
        n_components=2,
        perplexity=perp,
        init="pca",
        learning_rate="auto",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    return reducer.fit_transform(X)


def cell_already_done(csv_path: Path, manifest_path: Path) -> bool:
    """Idempotency: if both files exist and the manifest reports all 7 settings,
    skip the cell. Allows safe re-runs after interruption."""
    if not (csv_path.exists() and manifest_path.exists()):
        return False
    try:
        m = json.loads(manifest_path.read_text())
        n_done = len(m.get("trustworthiness", {}))
        return n_done == len(UMAP_HP) + len(TSNE_HP)
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# PER-CELL PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def process_cell(model_cfg, task: str, layer: int, paths: dict,
                 prompt_template: str, logger) -> dict:
    """Compute UMAP and t-SNE embeddings for one (model, task, layer) cell.
    Writes CSV and manifest. Returns the manifest dict."""
    mk = model_cfg["key"]
    act_path = Path(paths["data_root"]) / "activations" / mk / f"{task}_layer_{layer:02d}.npy"
    labels_path = Path(paths["data_root"]) / "data" / "raw" / f"{task}_problems.csv"

    out_dir = Path(paths["results_root"]) / "embeddings" / mk
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{task}_layer_{layer:02d}.csv"
    manifest_path = out_dir / f"{task}_layer_{layer:02d}_manifest.json"

    if cell_already_done(csv_path, manifest_path):
        logger.info("  [%s | %s | L%d] already done, skipping", mk, task, layer)
        return json.loads(manifest_path.read_text())

    if not act_path.exists():
        raise FileNotFoundError(act_path)
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    t0 = time.time()
    logger.info("  loading %s", act_path)
    X = np.load(act_path)
    X = np.ascontiguousarray(X.astype(np.float32))
    n, d = X.shape

    logger.info("  loading %s", labels_path)
    labels_df = pd.read_csv(labels_path)
    if len(labels_df) != n:
        raise ValueError(
            f"row mismatch: activations={n} vs labels={len(labels_df)}"
        )

    activation_norm = np.linalg.norm(X, axis=1).astype(np.float32)

    coords = {}      # name -> (n, 2) ndarray
    runtimes = {}    # name -> seconds

    # --- UMAP grid ---
    for name, hp in UMAP_HP:
        ts = time.time()
        logger.info("    fitting %s ...", name)
        emb = fit_umap(X, **hp)
        runtimes[name] = round(time.time() - ts, 2)
        coords[name] = emb.astype(np.float32)
        logger.info("    %s done in %.1fs", name, runtimes[name])

    # --- t-SNE grid ---
    for name, perp in TSNE_HP:
        ts = time.time()
        logger.info("    fitting %s (perplexity=%d) ...", name, perp)
        emb = fit_tsne(X, perp)
        runtimes[name] = round(time.time() - ts, 2)
        coords[name] = emb.astype(np.float32)
        logger.info("    %s done in %.1fs", name, runtimes[name])

    # --- Trustworthiness on a fixed subsample ---
    sub_size = min(n, TRUSTWORTHINESS_SUBSAMPLE)
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(n, sub_size, replace=False)
    X_sub = X[idx]
    scores = {}
    ts = time.time()
    for name, emb in coords.items():
        scores[name] = float(trustworthiness(
            X_sub, emb[idx], n_neighbors=TRUSTWORTHINESS_K
        ))
    logger.info("    trustworthiness scored in %.1fs", time.time() - ts)

    # --- Build output frame ---
    out_df = labels_df.copy()
    out_df["activation_norm"] = activation_norm
    for name, emb in coords.items():
        out_df[f"{name}_x"] = emb[:, 0]
        out_df[f"{name}_y"] = emb[:, 1]

    out_df.to_csv(csv_path, index=False)
    logger.info("  wrote %s (rows=%d, cols=%d)", csv_path, len(out_df), out_df.shape[1])

    # --- Manifest ---
    manifest = {
        "schema_version":              "v1",
        "model_key":                   mk,
        "model_name":                  model_cfg["name"],
        "task":                        task,
        "layer":                       layer,
        "n_problems":                  n,
        "hidden_dim":                  d,
        "activation_path":             str(act_path),
        "activation_sha256":           sha256_of(act_path),
        "labels_path":                 str(labels_path),
        "labels_sha256":               sha256_of(labels_path),
        "umap_hp_grid":                [{"name": n, **hp} for n, hp in UMAP_HP],
        "tsne_hp_grid":                [{"name": n, "perplexity": p} for n, p in TSNE_HP],
        "common_random_state":         RANDOM_STATE,
        "trustworthiness":             scores,
        "trustworthiness_n_neighbors": TRUSTWORTHINESS_K,
        "trustworthiness_subsample_size": sub_size,
        "runtime_seconds":             runtimes,
        "umap_learn_version":          umap.__version__,
        "sklearn_version":             __import__("sklearn").__version__,
        "numpy_version":               np.__version__,
        "pandas_version":              pd.__version__,
        "python_version":              platform.python_version(),
        "timestamp_utc":               datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cell_runtime_seconds":        round(time.time() - t0, 2),
        "csv_path":                    str(csv_path),
        "csv_rows":                    len(out_df),
        "csv_cols":                    out_df.shape[1],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("  wrote %s", manifest_path)
    return manifest


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    paths = cfg["paths"]
    logs_root = Path(paths["logs_root"])
    logger = setup_logging(logs_root)

    t_run0 = time.time()
    logger.info("=" * 78)
    logger.info("Step 4 Phase A: build embeddings (UMAP + t-SNE) for 30 cells")
    logger.info("config=%s", args.config)
    logger.info("UMAP grid: %s", [n for n, _ in UMAP_HP])
    logger.info("t-SNE grid: %s", [n for n, _ in TSNE_HP])

    cells = []
    for m in cfg["models"]:
        for task in ("addition", "multiplication"):
            for layer in m["layers"]:
                cells.append((m, task, layer))
    logger.info("total cells: %d", len(cells))

    results = []
    for i, (m, task, layer) in enumerate(cells, 1):
        logger.info("-" * 78)
        logger.info("[cell %d/%d] %s | %s | L%d", i, len(cells), m["key"], task, layer)
        manifest = process_cell(
            m, task, layer, paths,
            cfg["dataset"]["prompts"][task], logger
        )
        results.append(manifest)
        elapsed = time.time() - t_run0
        logger.info("  cumulative wall: %.1f min", elapsed / 60.0)

    logger.info("=" * 78)
    logger.info("HEADLINE TRUSTWORTHINESS SCORES")
    logger.info("  cell                                                | best UMAP                | best t-SNE")
    for r in results:
        umap_scores = {k: v for k, v in r["trustworthiness"].items() if k.startswith("umap")}
        tsne_scores = {k: v for k, v in r["trustworthiness"].items() if k.startswith("tsne")}
        best_u = max(umap_scores.items(), key=lambda kv: kv[1]) if umap_scores else ("-", 0.0)
        best_t = max(tsne_scores.items(), key=lambda kv: kv[1]) if tsne_scores else ("-", 0.0)
        cell_label = f"{r['model_key']:13s} | {r['task']:14s} | L{r['layer']:02d}"
        logger.info("  %s | %s=%.4f | %s=%.4f",
                    cell_label, best_u[0], best_u[1], best_t[0], best_t[1])
    logger.info("DONE  total wall=%.1f min", (time.time() - t_run0) / 60.0)


if __name__ == "__main__":
    main()
