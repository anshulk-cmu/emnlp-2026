"""Phase 1 plots — 10 comprehensive plots per model + 3 diagnostics.

Reads merged master CSVs from results/ccsvd_subspaces/ and the per-cell .npy
artifacts. Produces, per model under data/figures/ccsvd/{model_key}/:

  01_r_dim_heatmap.png             concepts × layers, both tasks
  02_lambda_ratio_heatmap.png      log10(λ₁/λ₂) heatmap
  03_cv_mean_heatmap.png           cv_mean heatmap
  04_scree_grid_addition.png       comprehensive grid: every concept × every layer
  05_scree_grid_multiplication.png
  06_centroids_grid_addition.png   comprehensive centroid scatter grid
  07_centroids_grid_multiplication.png
  08_principal_angles.png          adjacent-layer principal angles, all concepts × both tasks
  09_r_dim_layer_trajectory.png    line plot of r_dim vs layer per concept
  10_r_dim_vs_trustworthiness.png  scatter
  diagnostics/filter_fires.png
  diagnostics/group_imbalance.png
  diagnostics/perm_null_example.png

Usage:
  python plot_ccsvd_subspaces.py --config /home/anshulk/emnlp2026/config.yaml
"""

import argparse
import hashlib
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd
import yaml
from scipy.linalg import subspace_angles


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(logs_root: Path):
    logs_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("plot_ccsvd")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = RotatingFileHandler(logs_root / "plot_ccsvd_subspaces.log", maxBytes=10_000_000, backupCount=3)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


# ═══════════════════════════════════════════════════════════════════════════════
# CONCEPT TIER ORDERING
# ═══════════════════════════════════════════════════════════════════════════════

TIER1 = {"a", "b", "answer", "a_units", "a_tens", "b_units", "b_tens",
         "ans_units", "ans_tens", "ans_hundreds", "ans_thousands",
         "a_num_digits", "b_num_digits", "ans_num_digits"}

TIER2 = {"column_sum_units", "column_sum_tens", "column_sum_hundreds", "column_sum_thousands",
         "carry_units", "carry_tens", "carry_hundreds", "carry_thousands",
         "running_sum_units", "running_sum_tens", "running_sum_hundreds", "running_sum_thousands",
         "partial_product_units", "partial_product_a_units_b_tens",
         "partial_product_a_tens_b_units", "partial_product_a_tens_b_tens"}

TIER3 = {"a_parity", "b_parity", "ans_parity", "parity_match", "parity_xor",
         "a_magnitude_tier", "b_magnitude_tier", "ans_magnitude_tier",
         "ans_ends_in_zero", "ans_is_zero", "a_is_zero", "b_is_zero", "a_eq_b"}

TIER4 = {"max_operand", "min_operand", "operand_diff", "operand_abs_diff", "larger_operand",
         "both_zero", "either_zero", "both_one", "either_one"}


def concept_tier(name: str) -> int:
    if name in TIER1: return 1
    if name in TIER2: return 2
    if name in TIER3: return 3
    if name in TIER4: return 4
    if "__" in name: return 5    # joint
    return 6


def tier_color(tier: int) -> str:
    return {1: "tab:blue", 2: "tab:orange", 3: "tab:green",
            4: "tab:purple", 5: "tab:red", 6: "tab:gray"}.get(tier, "tab:gray")


def sort_concepts(concepts: list[str]) -> list[str]:
    return sorted(concepts, key=lambda c: (concept_tier(c), c))


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_inputs(results_root: Path) -> dict:
    """Load the merged master CSVs."""
    base = results_root / "ccsvd_subspaces"
    out = {}
    for name in ("summary", "eigenvalue_spectra", "projected_centroids", "null_summary", "cv_per_fold"):
        p = base / f"{name}.csv"
        if p.exists():
            out[name] = pd.read_csv(p)
        else:
            # try per-model fallback (concatenate)
            shards = sorted(base.glob(f"*/{name}_*.csv"))
            if shards:
                out[name] = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
            else:
                out[name] = pd.DataFrame()
    return out


def cell_dir(results_root: Path, model_key: str, task: str, layer: int, concept: str) -> Path:
    return results_root / "ccsvd_subspaces" / model_key / task / f"layer_{layer:02d}" / concept


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# HEATMAP HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _heatmap_panel(ax, mat, concepts, layers, *, cmap="viridis", norm=None,
                   annot_fmt="{:.0f}", title="", xlabel="Layer", ylabel="Concept",
                   highlight_mask=None, x_outline=None):
    """Render one heatmap panel onto ax. mat is (n_concepts, n_layers)."""
    im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(L) for L in layers])
    ax.set_yticks(range(len(concepts)))
    ax.set_yticklabels(concepts, fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Cell annotations
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=6, color="dimgray")
                continue
            ax.text(j, i, annot_fmt.format(v), ha="center", va="center", fontsize=6, color="white" if _is_dark(im, v) else "black")

    # X-mark cells where status is not fit_ok
    if highlight_mask is not None:
        for i in range(highlight_mask.shape[0]):
            for j in range(highlight_mask.shape[1]):
                if highlight_mask[i, j]:
                    ax.plot([j - 0.4, j + 0.4], [i - 0.4, i + 0.4], color="red", linewidth=0.6)
                    ax.plot([j - 0.4, j + 0.4], [i + 0.4, i - 0.4], color="red", linewidth=0.6)

    # Outline cells in x_outline mask
    if x_outline is not None:
        for i in range(x_outline.shape[0]):
            for j in range(x_outline.shape[1]):
                if x_outline[i, j]:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=1.2)
                    ax.add_patch(rect)
    return im


def _is_dark(im, v):
    try:
        rgba = im.cmap(im.norm(v))
        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        return lum < 0.5
    except Exception:
        return False


def build_grid(df: pd.DataFrame, model_key: str, task: str, value_col: str, layers: list[int]) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Build (n_concepts, n_layers) matrix for one (model, task). Also return concepts (sorted), status mask, fit-ok mask."""
    sub = df[(df["model_key"] == model_key) & (df["task"] == task)]
    concepts = sort_concepts(sub["concept"].unique().tolist())
    mat = np.full((len(concepts), len(layers)), np.nan, dtype=np.float64)
    skipped = np.zeros_like(mat, dtype=bool)
    for i, c in enumerate(concepts):
        for j, L in enumerate(layers):
            row = sub[(sub["concept"] == c) & (sub["layer"] == L)]
            if len(row) == 0:
                continue
            v = row.iloc[0].get(value_col, np.nan)
            mat[i, j] = v
            stat = str(row.iloc[0].get("status", ""))
            if stat != "fit_ok":
                skipped[i, j] = True
    return mat, concepts, skipped, ~skipped


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 01-03: heatmaps
# ═══════════════════════════════════════════════════════════════════════════════

def plot_01_r_dim(model_key: str, layers: list[int], summary: pd.DataFrame, out_path: Path, logger):
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, 0.18 * 60)), sharey=True)
    panels_data = {}
    vmax = 0
    for ax_idx, task in enumerate(("addition", "multiplication")):
        mat, concepts, skipped, _ = build_grid(summary, model_key, task, "r_dim", layers)
        panels_data[task] = (mat, concepts, skipped)
        if np.any(np.isfinite(mat)):
            vmax = max(vmax, np.nanmax(mat))
    norm = Normalize(vmin=0, vmax=max(1, vmax))
    for ax_idx, task in enumerate(("addition", "multiplication")):
        mat, concepts, skipped = panels_data[task]
        _heatmap_panel(axes[ax_idx], mat, concepts, layers, cmap="viridis", norm=norm,
                       annot_fmt="{:.0f}", title=f"{model_key} — {task}", highlight_mask=skipped)
        if ax_idx == 1:
            cbar = plt.colorbar(axes[ax_idx].images[0], ax=axes[ax_idx], label="r_dim", fraction=0.046, pad=0.04)
    fig.suptitle(f"Subspace Dimensionality — {model_key}", fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  wrote %s", out_path)


def plot_02_lambda_ratio(model_key: str, layers: list[int], summary: pd.DataFrame, out_path: Path, logger):
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, 0.18 * 60)), sharey=True)
    panels = {}
    finite_max = 1
    for task in ("addition", "multiplication"):
        mat, concepts, skipped, _ = build_grid(summary, model_key, task, "lambda_1_over_2", layers)
        with np.errstate(invalid="ignore", divide="ignore"):
            log_mat = np.log10(mat)
        if np.any(np.isfinite(log_mat)):
            finite_max = max(finite_max, np.nanmax(np.abs(log_mat)))
        panels[task] = (log_mat, mat, concepts, skipped)
    norm = TwoSlopeNorm(vcenter=0, vmin=-finite_max, vmax=finite_max)
    for ax_idx, task in enumerate(("addition", "multiplication")):
        log_mat, mat, concepts, skipped = panels[task]
        # outline cells where ratio > 10
        outline = (mat > 10) & np.isfinite(mat)
        _heatmap_panel(axes[ax_idx], log_mat, concepts, layers, cmap="RdBu_r", norm=norm,
                       annot_fmt="{:.1f}", title=f"{model_key} — {task}",
                       highlight_mask=skipped, x_outline=outline)
        if ax_idx == 1:
            plt.colorbar(axes[ax_idx].images[0], ax=axes[ax_idx], label="log10(λ₁/λ₂)", fraction=0.046, pad=0.04)
    fig.suptitle(f"Single-direction dominance: log10(λ₁/λ₂) — {model_key}", fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  wrote %s", out_path)


def plot_03_cv_mean(model_key: str, layers: list[int], summary: pd.DataFrame, out_path: Path, logger):
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, 0.18 * 60)), sharey=True)
    norm = Normalize(vmin=0, vmax=1)
    for ax_idx, task in enumerate(("addition", "multiplication")):
        mat, concepts, skipped, _ = build_grid(summary, model_key, task, "cv_mean", layers)
        outline = (mat < 0.9) & np.isfinite(mat)
        _heatmap_panel(axes[ax_idx], mat, concepts, layers, cmap="viridis", norm=norm,
                       annot_fmt="{:.2f}", title=f"{model_key} — {task}",
                       highlight_mask=skipped, x_outline=outline)
        if ax_idx == 1:
            plt.colorbar(axes[ax_idx].images[0], ax=axes[ax_idx], label="cv_mean", fraction=0.046, pad=0.04)
    fig.suptitle(f"Subspace cross-validation (cv_mean) — {model_key}", fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  wrote %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 04, 05: comprehensive scree grid
# ═══════════════════════════════════════════════════════════════════════════════

def plot_scree_grid(model_key: str, task: str, layers: list[int],
                    summary: pd.DataFrame, eig: pd.DataFrame, null: pd.DataFrame,
                    out_path: Path, logger):
    sub_summary = summary[(summary["model_key"] == model_key) & (summary["task"] == task)]
    concepts = sort_concepts(sub_summary["concept"].unique().tolist())
    n_rows = len(concepts)
    n_cols = len(layers)
    if n_rows == 0:
        logger.warning("  no concepts for %s × %s — skipping", model_key, task)
        return

    fig_h = max(6, 1.4 * n_rows)
    fig_w = max(6, 2.0 * n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                             squeeze=False)

    for i, concept in enumerate(concepts):
        for j, L in enumerate(layers):
            ax = axes[i, j]
            row = sub_summary[(sub_summary["concept"] == concept) & (sub_summary["layer"] == L)]
            if len(row) == 0 or str(row.iloc[0].get("status", "")) not in ("fit_ok", "no_significant_subspace"):
                ax.set_facecolor("#f0f0f0")
                ax.text(0.5, 0.5, "—", transform=ax.transAxes, ha="center", va="center", fontsize=8, color="dimgray")
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0: ax.set_title(f"L{L}", fontsize=8)
                if j == 0: ax.set_ylabel(concept, fontsize=7, rotation=0, ha="right", va="center")
                continue

            # eigenvalue spectrum + null envelope for this cell
            eg = eig[(eig["model_key"] == model_key) & (eig["task"] == task)
                     & (eig["concept"] == concept) & (eig["layer"] == L)].sort_values("k")
            ng = null[(null["model_key"] == model_key) & (null["task"] == task)
                      & (null["concept"] == concept) & (null["layer"] == L)].sort_values("k")
            if len(eg) == 0:
                ax.set_facecolor("#f0f0f0")
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0: ax.set_title(f"L{L}", fontsize=8)
                if j == 0: ax.set_ylabel(concept, fontsize=7, rotation=0, ha="right", va="center")
                continue

            ks = eg["k"].to_numpy()
            lam = eg["lambda_k"].to_numpy()
            t99 = eg["threshold_99_k"].to_numpy()
            r_dim = int(row.iloc[0].get("r_dim", 0))
            ax.plot(ks, lam, "o-", color="tab:blue", markersize=3, linewidth=1, label="obs")
            # Null 99th percentile envelope (filled)
            ax.fill_between(ks, 1e-300, t99, color="lightgray", alpha=0.6, label="null 99%")
            # Null median if available
            if len(ng):
                ax.plot(ng["k"], ng["null_p50"], "--", color="dimgray", linewidth=0.8, label="null p50")
            # Vertical lines for r_dim (perm null) and CumVar≥95%
            if r_dim > 0:
                ax.axvline(r_dim, color="red", linestyle=":", linewidth=1.0, label=f"r={r_dim}")
            cumvar = eg["cumulative_variance_k"].to_numpy()
            cv_idx = np.argmax(cumvar >= 0.95) + 1 if np.any(cumvar >= 0.95) else None
            if cv_idx:
                ax.axvline(cv_idx, color="orange", linestyle=":", linewidth=0.7, alpha=0.6)
            ax.set_yscale("log")
            ax.set_xticks(ks[:: max(1, len(ks)//5)])
            ax.tick_params(labelsize=6)
            if i == 0: ax.set_title(f"L{L}", fontsize=8)
            if j == 0: ax.set_ylabel(concept, fontsize=7, rotation=0, ha="right", va="center")

    fig.suptitle(f"CCSVD Eigenvalue Spectra — {model_key} {task} (rows=concepts, cols=layers)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    logger.info("  wrote %s (%d × %d panels)", out_path, n_rows, n_cols)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 06, 07: comprehensive centroid grid
# ═══════════════════════════════════════════════════════════════════════════════

def plot_centroids_grid(model_key: str, task: str, layers: list[int],
                        summary: pd.DataFrame, centroids: pd.DataFrame,
                        out_path: Path, logger):
    sub_summary = summary[(summary["model_key"] == model_key) & (summary["task"] == task)]
    concepts = sort_concepts(sub_summary["concept"].unique().tolist())
    n_rows = len(concepts)
    n_cols = len(layers)
    if n_rows == 0: return

    fig_h = max(6, 1.5 * n_rows)
    fig_w = max(6, 2.0 * n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    for i, concept in enumerate(concepts):
        for j, L in enumerate(layers):
            ax = axes[i, j]
            row = sub_summary[(sub_summary["concept"] == concept) & (sub_summary["layer"] == L)]
            r_dim = int(row.iloc[0].get("r_dim", 0)) if len(row) else 0
            if r_dim < 2:
                ax.set_facecolor("#f0f0f0")
                ax.text(0.5, 0.5, f"r={r_dim}", transform=ax.transAxes, ha="center", va="center", fontsize=8, color="dimgray")
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0: ax.set_title(f"L{L}", fontsize=8)
                if j == 0: ax.set_ylabel(concept, fontsize=7, rotation=0, ha="right", va="center")
                continue

            cd = centroids[(centroids["model_key"] == model_key) & (centroids["task"] == task)
                           & (centroids["concept"] == concept) & (centroids["layer"] == L)]
            # pivot to (value, dim)
            piv = cd.pivot_table(index=["value", "n_v"], columns="dim_idx", values="dim_value")
            if piv.shape[1] < 2:
                ax.set_facecolor("#f0f0f0")
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0: ax.set_title(f"L{L}", fontsize=8)
                if j == 0: ax.set_ylabel(concept, fontsize=7, rotation=0, ha="right", va="center")
                continue
            try:
                xs = piv[1].to_numpy(); ys = piv[2].to_numpy()
            except KeyError:
                ax.set_xticks([]); ax.set_yticks([])
                continue
            values = [v for v, _ in piv.index]
            n_v_arr = np.array([n for _, n in piv.index])
            try:
                num_vals = np.array([float(v) for v in values])
                cmap, kind = ("viridis", "continuous")
            except (ValueError, TypeError):
                num_vals = np.arange(len(values))
                cmap, kind = ("tab10", "categorical")
            sizes = 6 + 30 * (n_v_arr / max(1, n_v_arr.max()))
            ax.scatter(xs, ys, c=num_vals, cmap=cmap, s=sizes, edgecolors="none", alpha=0.85)
            ax.tick_params(labelsize=6)
            if i == 0: ax.set_title(f"L{L}", fontsize=8)
            if j == 0: ax.set_ylabel(concept, fontsize=7, rotation=0, ha="right", va="center")

    fig.suptitle(f"CCSVD Projected Centroids — {model_key} {task} (rows=concepts, cols=layers, dim_1 vs dim_2)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    logger.info("  wrote %s (%d × %d panels)", out_path, n_rows, n_cols)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 08: principal angles across adjacent layers
# ═══════════════════════════════════════════════════════════════════════════════

def plot_principal_angles(model_key: str, layers: list[int], summary: pd.DataFrame,
                          results_root: Path, out_path: Path, logger):
    concepts = sort_concepts(summary[summary["model_key"] == model_key]["concept"].unique().tolist())
    if not concepts:
        return
    n_rows = len(concepts)
    fig_h = max(6, 1.5 * n_rows)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, fig_h), squeeze=False)

    transitions = [(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
    transition_labels = [f"L{a}→L{b}" for a, b in transitions]

    for i, concept in enumerate(concepts):
        for col_idx, task in enumerate(("addition", "multiplication")):
            ax = axes[i, col_idx]
            angles_per_transition = []  # list of arrays
            for a, b in transitions:
                ba = cell_dir(results_root, model_key, task, a, concept) / "basis.npy"
                bb = cell_dir(results_root, model_key, task, b, concept) / "basis.npy"
                if not (ba.exists() and bb.exists()):
                    angles_per_transition.append(np.array([np.nan, np.nan, np.nan]))
                    continue
                B1 = np.load(ba); B2 = np.load(bb)
                if B1.size == 0 or B2.size == 0 or B1.ndim != 2 or B2.ndim != 2:
                    angles_per_transition.append(np.array([np.nan, np.nan, np.nan]))
                    continue
                try:
                    ang_rad = subspace_angles(B1, B2)
                    ang_deg = np.degrees(ang_rad)
                except Exception:
                    ang_deg = np.array([np.nan, np.nan, np.nan])
                # take top 3 (largest angles); pad with nan
                top3 = np.full(3, np.nan)
                top3[: min(3, len(ang_deg))] = sorted(ang_deg, reverse=True)[:3]
                angles_per_transition.append(top3)

            mat = np.array(angles_per_transition)  # (n_trans, 3)
            xs = np.arange(len(transitions))
            for k in range(3):
                ax.plot(xs, mat[:, k], "o-", markersize=3, linewidth=0.8, label=f"angle {k+1}" if i == 0 and col_idx == 0 else None)
            ax.axhline(45, color="gray", linestyle="--", linewidth=0.5)
            ax.set_xticks(xs)
            ax.set_xticklabels(transition_labels, fontsize=6, rotation=30)
            ax.set_ylim(0, 95)
            ax.tick_params(labelsize=6)
            if i == 0: ax.set_title(task, fontsize=8)
            if col_idx == 0: ax.set_ylabel(concept, fontsize=7, rotation=0, ha="right", va="center")

    fig.suptitle(f"Principal angles across adjacent layers — {model_key}", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  wrote %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 09: r_dim vs layer trajectory
# ═══════════════════════════════════════════════════════════════════════════════

def plot_r_dim_trajectory(model_key: str, layers: list[int], summary: pd.DataFrame, out_path: Path, logger):
    sub = summary[summary["model_key"] == model_key]
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    for ax_idx, task in enumerate(("addition", "multiplication")):
        ax = axes[ax_idx]
        st = sub[sub["task"] == task]
        concepts = sort_concepts(st["concept"].unique().tolist())
        for concept in concepts:
            row_set = st[st["concept"] == concept].sort_values("layer")
            xs = row_set["layer"].to_numpy()
            ys = row_set["r_dim"].to_numpy()
            t = concept_tier(concept)
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=0.7, color=tier_color(t), alpha=0.7)
        ax.set_xlabel("Layer"); ax.set_ylabel("r_dim"); ax.set_title(task)
        ax.grid(True, alpha=0.3)
    # custom legend
    handles = [plt.Line2D([0], [0], color=tier_color(t), label=lbl) for t, lbl in
               [(1, "Tier 1 (digits)"), (2, "Tier 2 (column algebra)"),
                (3, "Tier 3 (structural)"), (4, "Tier 4 (relational)"),
                (5, "Joint")]]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(f"r_dim trajectory across layers — {model_key}", fontsize=12, y=1.04)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  wrote %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 10: r_dim vs UMAP trustworthiness
# ═══════════════════════════════════════════════════════════════════════════════

def plot_r_dim_vs_trust(model_key: str, summary: pd.DataFrame, out_path: Path, logger):
    sub = summary[summary["model_key"] == model_key]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_idx, task in enumerate(("addition", "multiplication")):
        ax = axes[ax_idx]
        st = sub[sub["task"] == task]
        if "best_umap_trustworthiness" not in st.columns:
            ax.text(0.5, 0.5, "no trustworthiness data", transform=ax.transAxes, ha="center")
            continue
        for concept in sort_concepts(st["concept"].unique().tolist()):
            cs = st[st["concept"] == concept]
            xs = cs["best_umap_trustworthiness"].to_numpy()
            ys = cs["r_dim"].to_numpy()
            t = concept_tier(concept)
            ax.scatter(xs, ys, color=tier_color(t), s=18, alpha=0.5, edgecolors="none")
        # annotate low-trustworthiness cells
        low = st[(st["best_umap_trustworthiness"] < 0.95) & st["best_umap_trustworthiness"].notna()]
        for _, row in low.iterrows():
            ax.annotate(f"{row['concept']}\nL{row['layer']}",
                        (row["best_umap_trustworthiness"], row["r_dim"]),
                        fontsize=5, alpha=0.8, color="black",
                        xytext=(2, 2), textcoords="offset points")
        ax.set_xlabel("best UMAP trustworthiness")
        ax.set_ylabel("r_dim")
        ax.set_title(task)
        ax.grid(True, alpha=0.3)
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=tier_color(t), label=lbl) for t, lbl in
               [(1, "Tier 1"), (2, "Tier 2"), (3, "Tier 3"), (4, "Tier 4"), (5, "Joint")]]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(f"r_dim vs UMAP 2-D trustworthiness — {model_key}", fontsize=12, y=1.04)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  wrote %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_filter_fires(model_key: str, layers: list[int], summary: pd.DataFrame, out_path: Path, logger):
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, 0.18 * 60)), sharey=True)
    panels = {}
    vmax = 1
    for task in ("addition", "multiplication"):
        mat, concepts, skipped, _ = build_grid(summary, model_key, task, "dropped_values_count", layers)
        panels[task] = (mat, concepts, skipped)
        if np.any(np.isfinite(mat)): vmax = max(vmax, np.nanmax(mat))
    norm = Normalize(vmin=0, vmax=max(1, vmax))
    for ax_idx, task in enumerate(("addition", "multiplication")):
        mat, concepts, skipped = panels[task]
        _heatmap_panel(axes[ax_idx], mat, concepts, layers, cmap="Greys", norm=norm,
                       annot_fmt="{:.0f}", title=f"{model_key} — {task}", highlight_mask=skipped)
        if ax_idx == 1:
            plt.colorbar(axes[ax_idx].images[0], ax=axes[ax_idx], label="dropped_values_count", fraction=0.046, pad=0.04)
    fig.suptitle(f"Filter fires (red ✕ = cell skipped, grey scale = dropped value count) — {model_key}", fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  wrote %s", out_path)


def plot_group_imbalance(model_key: str, summary: pd.DataFrame, out_path: Path, logger):
    sub = summary[summary["model_key"] == model_key]
    concepts = sort_concepts(sub["concept"].unique().tolist())
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, 0.18 * len(concepts))), sharey=True)
    for ax_idx, task in enumerate(("addition", "multiplication")):
        ax = axes[ax_idx]
        ys = []
        labels = []
        for concept in concepts:
            cs = sub[(sub["task"] == task) & (sub["concept"] == concept)]
            if "group_imbalance_ratio" in cs.columns:
                vals = cs["group_imbalance_ratio"].to_numpy()
                vals = vals[np.isfinite(vals)]
            else:
                vals = np.array([])
            ys.append(vals); labels.append(concept)
        for i, vals in enumerate(ys):
            ax.scatter(vals, [i] * len(vals), s=12, alpha=0.6, color=tier_color(concept_tier(labels[i])))
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.axvline(3, color="red", linestyle="--", linewidth=0.8, label="flag threshold (3)")
        ax.set_xscale("log")
        ax.set_xlabel("group_imbalance_ratio (max n_v / min n_v)")
        ax.set_title(task)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"Group imbalance per concept — {model_key}", fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  wrote %s", out_path)


def plot_perm_null_example(model_key: str, layers: list[int], summary: pd.DataFrame,
                           results_root: Path, out_path: Path, logger):
    # default cell: addition × headline-layer × a_units (resolve headline from layer list — middle of 5)
    headline_layer = layers[len(layers) // 2]
    cell = cell_dir(results_root, model_key, "addition", headline_layer, "a_units")
    null_path = cell / "null_eigenvalues.npy"
    eig_path = cell / "eigenvalues.npy"
    if not (null_path.exists() and eig_path.exists()):
        logger.warning("  perm_null_example: missing %s — skipping", null_path)
        return
    null_table = np.load(null_path)  # (n_perm, m-1)
    eigs = np.load(eig_path)         # (m-1,)
    n_show = min(3, null_table.shape[1])
    fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 4))
    if n_show == 1:
        axes = [axes]
    for k in range(n_show):
        ax = axes[k]
        ax.hist(null_table[:, k], bins=40, color="lightgray", edgecolor="dimgray")
        ax.axvline(eigs[k], color="red", linewidth=1.5, label=f"observed λ_{k+1}={eigs[k]:.4g}")
        p99 = np.percentile(null_table[:, k], 99)
        ax.axvline(p99, color="darkblue", linestyle="--", linewidth=1, label=f"null 99% = {p99:.4g}")
        ax.set_xlabel(f"λ_{k+1}")
        ax.set_title(f"Permutation null vs observed (k={k+1})")
        ax.legend(fontsize=8)
    fig.suptitle(f"Perm-null mechanism — {model_key} addition L{headline_layer} a_units", fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  wrote %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=None, help="restrict to one model_key; default: all 3")
    parser.add_argument("--plots", default="all",
                        help="comma-separated subset of {01..10, filter_fires, group_imbalance, perm_null_example, all}")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    paths = cfg["paths"]
    logger = setup_logging(Path(paths["logs_root"]))

    results_root = Path(paths["results_root"])
    figures_root = Path(paths["data_root"]) / "figures" / "ccsvd"
    figures_root.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    logger.info("=" * 78)
    logger.info("plot_ccsvd_subspaces — loading inputs")

    inputs = load_inputs(results_root)
    if "summary" not in inputs or len(inputs["summary"]) == 0:
        logger.error("No summary.csv found at %s/ccsvd_subspaces/. Run merge step first.", results_root)
        sys.exit(1)
    summary = inputs["summary"]
    eig = inputs.get("eigenvalue_spectra", pd.DataFrame())
    null = inputs.get("null_summary", pd.DataFrame())
    centroids = inputs.get("projected_centroids", pd.DataFrame())

    logger.info("loaded %d summary rows, %d eig rows, %d null rows, %d centroid rows",
                len(summary), len(eig), len(null), len(centroids))

    models = cfg["models"]
    if args.model:
        models = [m for m in models if m["key"] == args.model]
    plot_set = set(args.plots.split(",")) if args.plots != "all" else set(
        ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
         "filter_fires", "group_imbalance", "perm_null_example"])

    plot_index = {"plots": [], "models": [m["key"] for m in models], "plots_requested": sorted(plot_set)}

    for model_cfg in models:
        mk = model_cfg["key"]
        layers = model_cfg["layers"]
        out_dir = figures_root / mk
        diag_dir = out_dir / "diagnostics"
        out_dir.mkdir(parents=True, exist_ok=True)
        diag_dir.mkdir(parents=True, exist_ok=True)
        logger.info("-" * 78)
        logger.info("model: %s — output dir: %s", mk, out_dir)

        if "01" in plot_set:
            plot_01_r_dim(mk, layers, summary, out_dir / "01_r_dim_heatmap.png", logger)
        if "02" in plot_set:
            plot_02_lambda_ratio(mk, layers, summary, out_dir / "02_lambda_ratio_heatmap.png", logger)
        if "03" in plot_set:
            plot_03_cv_mean(mk, layers, summary, out_dir / "03_cv_mean_heatmap.png", logger)
        if "04" in plot_set:
            plot_scree_grid(mk, "addition", layers, summary, eig, null,
                            out_dir / "04_scree_grid_addition.png", logger)
        if "05" in plot_set:
            plot_scree_grid(mk, "multiplication", layers, summary, eig, null,
                            out_dir / "05_scree_grid_multiplication.png", logger)
        if "06" in plot_set:
            plot_centroids_grid(mk, "addition", layers, summary, centroids,
                                out_dir / "06_centroids_grid_addition.png", logger)
        if "07" in plot_set:
            plot_centroids_grid(mk, "multiplication", layers, summary, centroids,
                                out_dir / "07_centroids_grid_multiplication.png", logger)
        if "08" in plot_set:
            plot_principal_angles(mk, layers, summary, results_root,
                                  out_dir / "08_principal_angles.png", logger)
        if "09" in plot_set:
            plot_r_dim_trajectory(mk, layers, summary, out_dir / "09_r_dim_layer_trajectory.png", logger)
        if "10" in plot_set:
            plot_r_dim_vs_trust(mk, summary, out_dir / "10_r_dim_vs_trustworthiness.png", logger)
        if "filter_fires" in plot_set:
            plot_filter_fires(mk, layers, summary, diag_dir / "filter_fires.png", logger)
        if "group_imbalance" in plot_set:
            plot_group_imbalance(mk, summary, diag_dir / "group_imbalance.png", logger)
        if "perm_null_example" in plot_set:
            plot_perm_null_example(mk, layers, summary, results_root,
                                   diag_dir / "perm_null_example.png", logger)

    # plot_index.json
    pngs = sorted(figures_root.rglob("*.png"))
    plot_index["plots"] = [str(p) for p in pngs]
    plot_index["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    plot_index["library_versions"] = {
        "matplotlib": matplotlib.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": __import__("scipy").__version__,
        "python": platform.python_version(),
    }
    (figures_root / "plot_index.json").write_text(json.dumps(plot_index, indent=2))

    logger.info("=" * 78)
    logger.info("DONE  total wall=%.1f s  plots written=%d", time.time() - t0, len(pngs))


if __name__ == "__main__":
    main()
