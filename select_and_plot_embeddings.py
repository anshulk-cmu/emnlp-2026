"""Step 4 Phase B — render 45 selected UMAP/t-SNE plots.

Reads the 30 per-cell CSVs and manifests written by build_embeddings.py,
selects the best-trustworthiness UMAP and t-SNE coordinates per cell,
and produces 45 PNGs (15 per model = 10 UMAP + 5 t-SNE).

Usage:
  python select_and_plot_embeddings.py --config /home/anshulk/emnlp2026/config.yaml
"""

import argparse
import json
import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(logs_root: Path):
    logs_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("plot_embeddings")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = RotatingFileHandler(
        logs_root / "select_and_plot_embeddings.log",
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
# PLOT STYLE HELPERS (mirroring arithmetic-geometry/phase_a_embeddings.py)
# ═══════════════════════════════════════════════════════════════════════════════

def get_point_style(n: int):
    if n >= 3000: return dict(s=3, alpha=0.5)
    if n >= 1000: return dict(s=5, alpha=0.6)
    if n >= 200:  return dict(s=10, alpha=0.7)
    return dict(s=20, alpha=0.8)


def get_cmap_for_concept(values: np.ndarray):
    """Pick a colormap based on number and type of unique values."""
    uniq = pd.unique(values)
    n_uniq = len(uniq)
    is_numeric = pd.api.types.is_numeric_dtype(values)
    if n_uniq <= 10:
        return "tab10", "categorical"
    if n_uniq <= 20:
        return "tab20", "categorical"
    if is_numeric:
        return "viridis", "continuous"
    return "tab20", "categorical"


def encode_categorical_for_color(values: pd.Series):
    """Map non-numeric or sparse categorical values to consecutive ints
    so matplotlib can color them. Returns (codes_int_array, sorted_unique_values)."""
    uniq = sorted(pd.unique(values), key=lambda v: (str(type(v).__name__), v))
    mapping = {v: i for i, v in enumerate(uniq)}
    codes = np.array([mapping[v] for v in values], dtype=np.int32)
    return codes, uniq


# ═══════════════════════════════════════════════════════════════════════════════
# BEST-HP PICKER
# ═══════════════════════════════════════════════════════════════════════════════

def pick_best(manifest: dict, prefix: str):
    """Return the HP-name (e.g. 'umap2d_n30_md10') with highest
    trustworthiness among names beginning with `prefix`."""
    cands = {k: v for k, v in manifest["trustworthiness"].items()
             if k.startswith(prefix)}
    if not cands:
        raise KeyError(f"No HP setting starting with {prefix} in manifest")
    return max(cands.items(), key=lambda kv: kv[1])[0]


# ═══════════════════════════════════════════════════════════════════════════════
# RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

def render_plot(*, manifest: dict, df: pd.DataFrame, hp_name: str,
                concept: str, out_path: Path, plot_index: int, logger):
    n = len(df)
    style = get_point_style(n)
    x = df[f"{hp_name}_x"].to_numpy()
    y = df[f"{hp_name}_y"].to_numpy()
    values = df[concept]

    fig, ax = plt.subplots(figsize=(10, 8))
    if pd.api.types.is_numeric_dtype(values):
        cmap, kind = get_cmap_for_concept(values.to_numpy())
        sc = ax.scatter(x, y, c=values.to_numpy(), cmap=cmap,
                        edgecolors="none", rasterized=True, **style)
        cbar = plt.colorbar(sc, ax=ax, label=concept)
    else:
        codes, uniq = encode_categorical_for_color(values)
        cmap_name = "tab10" if len(uniq) <= 10 else "tab20"
        sc = ax.scatter(x, y, c=codes, cmap=cmap_name,
                        edgecolors="none", rasterized=True, **style)
        cbar = plt.colorbar(sc, ax=ax, label=concept,
                            ticks=range(len(uniq)))
        cbar.set_ticklabels([str(u) for u in uniq])

    method = "UMAP" if hp_name.startswith("umap") else "t-SNE"
    title = (f"{manifest['model_name']} | {manifest['task']} | "
             f"layer {manifest['layer']:02d} | colored by {concept} | "
             f"{method} ({hp_name})")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"{hp_name}_x")
    ax.set_ylabel(f"{hp_name}_y")
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  p%02d -> %s", plot_index, out_path.name)


# ═══════════════════════════════════════════════════════════════════════════════
# 45-PLOT SPEC BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_plot_spec(model_cfg: dict):
    """Return the 15 plots for a given model: 10 UMAP + 5 t-SNE."""
    mk = model_cfg["key"]
    layers = model_cfg["layers"]
    L_h = model_cfg["headline_layer"]
    L_s = layers[0]   # shallowest
    L_d = layers[-1]  # deepest

    spec = []  # list of dicts: {model, task, layer, concept, method_prefix}

    # 10 UMAP plots
    spec.extend([
        # 6 headline (3 add + 3 mult)
        dict(model=mk, task="addition",       layer=L_h, concept="a_units",            method_prefix="umap"),
        dict(model=mk, task="addition",       layer=L_h, concept="ans_units",          method_prefix="umap"),
        dict(model=mk, task="addition",       layer=L_h, concept="carry_units",        method_prefix="umap"),
        dict(model=mk, task="multiplication", layer=L_h, concept="a_units",            method_prefix="umap"),
        dict(model=mk, task="multiplication", layer=L_h, concept="ans_units",          method_prefix="umap"),
        dict(model=mk, task="multiplication", layer=L_h, concept="carry_units",        method_prefix="umap"),
        # extra concept (multiplication, headline)
        dict(model=mk, task="multiplication", layer=L_h, concept="partial_product_units", method_prefix="umap"),
        # layer progression on multiplication × ans_units
        dict(model=mk, task="multiplication", layer=L_s, concept="ans_units",          method_prefix="umap"),
        dict(model=mk, task="multiplication", layer=L_h, concept="ans_units",          method_prefix="umap"),
        dict(model=mk, task="multiplication", layer=L_d, concept="ans_units",          method_prefix="umap"),
    ])

    # 5 t-SNE plots
    spec.extend([
        dict(model=mk, task="addition",       layer=L_h, concept="a_units",      method_prefix="tsne"),
        dict(model=mk, task="addition",       layer=L_h, concept="ans_units",    method_prefix="tsne"),
        dict(model=mk, task="multiplication", layer=L_h, concept="a_units",      method_prefix="tsne"),
        dict(model=mk, task="multiplication", layer=L_h, concept="ans_units",    method_prefix="tsne"),
        dict(model=mk, task="multiplication", layer=L_h, concept="carry_units",  method_prefix="tsne"),
    ])

    return spec


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

    t0 = time.time()
    logger.info("=" * 78)
    logger.info("Step 4 Phase B: render 45 selected plots (15 per model)")

    embed_root = Path(paths["results_root"]) / "embeddings"
    fig_root = Path(paths["data_root"]) / "figures" / "embeddings"
    fig_root.mkdir(parents=True, exist_ok=True)

    # Build full plot spec across all 3 models.
    all_specs = []
    for m in cfg["models"]:
        per_model = build_plot_spec(m)
        all_specs.extend(per_model)
    if len(all_specs) != 45:
        raise RuntimeError(f"expected 45 plots, got {len(all_specs)}")
    logger.info("total plots queued: %d", len(all_specs))

    # Cache loaded CSVs and manifests.
    csv_cache = {}
    manifest_cache = {}
    plot_index_records = []

    for i, spec in enumerate(all_specs, 1):
        mk, task, layer = spec["model"], spec["task"], spec["layer"]
        concept = spec["concept"]
        method_prefix = spec["method_prefix"]

        cell_key = (mk, task, layer)
        if cell_key not in csv_cache:
            csv_path = embed_root / mk / f"{task}_layer_{layer:02d}.csv"
            manifest_path = embed_root / mk / f"{task}_layer_{layer:02d}_manifest.json"
            csv_cache[cell_key] = pd.read_csv(csv_path)
            manifest_cache[cell_key] = json.loads(manifest_path.read_text())

        df = csv_cache[cell_key]
        manifest = manifest_cache[cell_key]

        if concept not in df.columns:
            logger.warning("  [skip p%02d] concept '%s' not in CSV columns for %s",
                           i, concept, cell_key)
            continue

        hp_name = pick_best(manifest, method_prefix)
        method_label = "umap" if method_prefix == "umap" else "tsne"
        png_name = f"p{i:02d}_{mk}_{task}_L{layer:02d}_{concept}_{method_label}.png"
        out_path = fig_root / png_name

        render_plot(manifest=manifest, df=df, hp_name=hp_name,
                    concept=concept, out_path=out_path,
                    plot_index=i, logger=logger)
        plot_index_records.append({
            "plot_index": i,
            "model_key": mk,
            "task": task,
            "layer": layer,
            "concept": concept,
            "method": method_label,
            "hp_name": hp_name,
            "trustworthiness": manifest["trustworthiness"][hp_name],
            "png": str(out_path),
        })

    # Write a plot index JSON for the doc.
    index_path = fig_root / "plot_index.json"
    index_path.write_text(json.dumps({
        "n_plots": len(plot_index_records),
        "plots": plot_index_records,
    }, indent=2))
    logger.info("wrote %s", index_path)

    logger.info("=" * 78)
    logger.info("DONE  total wall=%.1f s  plots rendered=%d", time.time() - t0, len(plot_index_records))


if __name__ == "__main__":
    main()
