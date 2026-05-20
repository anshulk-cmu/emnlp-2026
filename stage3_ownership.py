#!/usr/bin/env python3
"""
Stage 3 — Ownership test.

Per BSMI-R cell c = (model, task, mode, layer, concept) that already has a
declared shape on disk under data/results/stage2c_gplvm/..., test whether the
recovered geometry is OWNED by c or INHERITED from c's algebraic correlates.

Method
------
1.  Read the raw Stage 2c metadata from disk (winner shape, alpha_hat, P_hat,
    per-shape log_E, family info, perm_p, alignment_score).
2.  Build the union basis B_u for c (same code path as the BSMI-R worker).
3.  For each pre-registered correlate c' in C(c), load its union basis B_u(c')
    at the SAME (model, task, mode, layer). Stack and QR-orthonormalise to
    get Q ∈ R^{4096 x k_Q}.
4.  Form
        Y_orth = (X - mu)(I - Q Qᵀ) B_u                ∈ R^{N x k_u}
    where X are the same correct-only activations BSMI-R used. This is the
    probe-subspace projection of the residualised data.
5.  Re-run BSMI-R EVIDENCE on Y_orth with alpha_hat and P_hat *locked* to
    their raw-cell values (so the prior is held constant and log Z numbers
    are apples-to-apples comparable to the raw run).
6.  Decide verdict (owned / inherited / ambiguous) using the pre-registered
    thresholds in configs/stage3.yaml.

Outputs
-------
results/stage3_ownership/<model>/<task>/mode_<mode>/layer_<LL>/<concept>/
    stage3_results.csv     # one-row summary (matches aggregator schema)
    stage3_metadata.json   # full evidence vector + Q meta + correlates list
    evidence_orth.csv      # per-shape logZ / alignment under orthogonalisation
    perm_null_orth.npy     # 10,000 permutation null statistics on Y_orth
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

import stage2c_gplvm as worker
import stage2c_shapes as shapes_mod
import stage2c_modules as evidence_mod


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
DEFAULT_STAGE3_CONFIG = PROJECT_ROOT / "configs" / "stage3.yaml"

SVD_TOLERANCE_FACTOR = 1e-10


# ─── Atomic IO ──────────────────────────────────────────────────────────────

def atomic_save(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".npy", dir=str(path.parent))
    os.close(fd)
    np.save(tmp, arr)
    os.replace(tmp, path)


def atomic_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".json", dir=str(path.parent))
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".csv", dir=str(path.parent))
    os.close(fd)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# ─── Paths ─────────────────────────────────────────────────────────────────

def derive_paths(cfg: dict) -> dict:
    data_root = Path(cfg["paths"]["data_root"])
    results_root = Path(cfg["paths"]["results_root"])
    activations_root = Path(cfg["paths"].get(
        "activations_root", data_root / "activations"))
    return {
        "data_root": data_root,
        "results_root": results_root,
        "activations_root": activations_root,
        "logs_root": Path(cfg["paths"].get("logs_root", data_root / "logs")),
    }


def stage3_cell_dir(results_root: Path, model: str, task: str, mode: str,
                     layer: int, concept: str) -> Path:
    return (results_root / "stage3_ownership" / model / task / f"mode_{mode}"
            / f"layer_{layer:02d}" / concept)


def raw_metadata_path(results_root: Path, model: str, task: str, mode: str,
                       layer: int, concept: str) -> Path:
    return (results_root / "stage2c_gplvm" / model / task / f"mode_{mode}"
            / f"layer_{layer:02d}" / concept / "metadata.json")


# ─── Q-basis assembly (correlate-orthogonalisation operator) ───────────────

def build_correlate_Q(results_root: Path, model: str, task: str, mode: str,
                       layer: int, correlates: List[str]) -> Tuple[np.ndarray, Dict]:
    """Stack each correlate's union basis (LDA-A ∪ CCSVD), then QR-orthonormalise
    to get Q ∈ R^{4096 x k_Q}.

    A correlate is silently skipped if neither LDA-A nor CCSVD basis exists
    for it at this (model, task, mode, layer). The meta records every
    correlate we tried and whether it contributed.
    """
    rows = []           # each row is a (4096,) basis column, transposed
    contributions = []
    for cprime in correlates:
        B_c, meta = worker.build_union_basis(results_root, model, task, mode,
                                                layer, cprime)
        if B_c.shape[1] >= 1:
            rows.append(B_c.T.astype(np.float64))
            contributions.append({"correlate": cprime,
                                    "k_u_c": int(B_c.shape[1]),
                                    "lda_a_present": meta["contributions"][0][
                                        "source"] == "lda_a"
                                        if meta["contributions"] else False,
                                    "ccsvd_present": any(c["source"] == "ccsvd"
                                        for c in meta["contributions"]),
                                    "redundancy_removed": int(meta.get(
                                        "redundancy_removed", 0))})
        else:
            contributions.append({"correlate": cprime,
                                    "k_u_c": 0,
                                    "lda_a_present": False,
                                    "ccsvd_present": False,
                                    "missing": True})
    if not rows:
        Q = np.zeros((4096, 0), dtype=np.float64)
        return Q, {"k_Q": 0, "contributions": contributions,
                    "stacked_dim": 0, "redundancy_removed": 0}
    stacked = np.vstack(rows)                                  # (n_total, 4096)
    stacked_dim = int(stacked.shape[0])
    # QR on stacked.T (4096 x n_total) gives orthonormal columns of length 4096.
    U, S, Vt = np.linalg.svd(stacked, full_matrices=False)
    keep = S > SVD_TOLERANCE_FACTOR * S[0]
    Vt_keep = Vt[keep]                                          # (k_Q, 4096)
    Q = Vt_keep.T.astype(np.float64)                            # (4096, k_Q)
    return Q, {
        "k_Q": int(Q.shape[1]),
        "stacked_dim": stacked_dim,
        "redundancy_removed": stacked_dim - int(Q.shape[1]),
        "contributions": contributions,
        "top_singular_value": float(S[0]),
        "smallest_kept_singular_value": float(S[keep].min()
                                                if keep.any() else 0.0),
        "svd_tolerance_factor": SVD_TOLERANCE_FACTOR,
    }


def orthogonalise_to_subspace(Z_raw: np.ndarray, X_centred: np.ndarray,
                                B_u: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Apply (I - QQᵀ) in the ambient space, then project onto B_u.

    Z_raw = X_centred @ B_u  is the BSMI-R input in k_u-dim probe-coords.
    Y_orth = X_centred @ (I - QQᵀ) @ B_u = Z_raw - X_centred @ Q @ (Qᵀ B_u).

    Both forms are mathematically identical; the second is faster
    (4096 x k_Q  *  k_Q x k_u  rather than 4096 x 4096).
    """
    if Q.shape[1] == 0:
        return Z_raw.copy()
    QtB = Q.T @ B_u                              # (k_Q, k_u)
    proj_to_Q_then_to_Bu = (X_centred @ Q) @ QtB # (N, k_u)
    return Z_raw - proj_to_Q_then_to_Bu


# ─── Full BSMI-R re-run on Y_orth with locked priors ──────────────────────

def evaluate_orth_evidence(Z_orth: np.ndarray, label_codes: np.ndarray,
                            K_natural: int, alpha_locked: float,
                            P_hat: Optional[float], P_hat_2: Optional[float],
                            model: str, task: str, mode: str, layer: int,
                            concept: str, union_meta: dict, bsmir_cfg: dict,
                            logger: logging.Logger
                            ) -> Dict:
    """Re-run the FULL BSMI-R pipeline (Stages 0-17) on the orthogonalised
    point cloud, with alpha and P_hat LOCKED to the raw cell's values so
    log Z numbers are apples-to-apples comparable.

    All independent evidence modules (intrinsic dim, PH, Fourier, posterior
    geometry, label alignment, holdout, seed/prior stability, permutation)
    fire on Y_orth — orthogonalisation can change ANY of them, and each
    change is itself ownership-relevant evidence.
    """
    result = worker.analyze_cell(
        Z_orth, label_codes, K_natural, model, task, mode, layer, concept,
        union_meta=union_meta, mu_layer_source="Y_orth",
        stage2a_row_lda=None, stage2a_row_ccsvd=None,  # period override below
        bsmir_cfg=bsmir_cfg, logger=logger,
        alpha_override=alpha_locked,
        P_hat_override=P_hat, P_hat_2_override=P_hat_2,
    )
    if result.get("status") != "ok":
        return {"status": result.get("status"),
                 "K_present_used": result.get("K_present", 0),
                 "N_used": result.get("N_used", 0)}

    cr = result["cell_result"]
    # Extract just the fields Stage 3 verdict logic needs (everything else
    # stays in the metadata.json for downstream inspection).
    log_E = dict(cr.log_E)
    log_E_refined = dict(cr.log_E_refined)
    # Refined evidence (Stage 6) is what raw Stage 2c headlines on, so we use it
    # for Stage 3's logZ comparisons too.
    score_per_shape = {n: log_E_refined.get(n, log_E.get(n, float("-inf")))
                        for n in log_E}
    fam_info = worker.compute_family_evidences(score_per_shape)
    alignment = {n: float(cr.align.get(n, {}).get("alignment_score",
                                                      float("nan")))
                  for n in log_E}
    align_full = {n: dict(cr.align.get(n, {})) for n in log_E}

    return {
        "status": "ok",
        "log_E": log_E,
        "log_E_refined": log_E_refined,
        "score_per_shape": score_per_shape,
        "best_theta": dict(cr.best_theta),
        "latent": {n: cr.latent[n] for n in cr.latent},
        "W_mean": {n: cr.W_mean[n] for n in cr.W_mean},
        "alignment": alignment,
        "align_full": align_full,
        "family_log_E": fam_info.get("family_log_E", {}),
        "family_winner": fam_info.get("family_winner", ""),
        "family_runner_up": fam_info.get("family_runner_up", ""),
        "family_evidence_gap": float(fam_info.get("family_evidence_gap", 0.0)),
        # Stage 2c built-in family info (mirrors raw Stage 2c output)
        "family_winner_bsmir": cr.family_winner,
        "family_evidence_gap_bsmir": cr.family_evidence_gap,
        "perm_p": float(cr.permutation.get("p_value", float("nan"))),
        "perm_null": np.asarray(cr.permutation.get("null_samples", []),
                                  dtype=np.float64),
        "perm_observed_T": float(cr.permutation.get("observed_T", float("nan"))),
        "winner_named": cr.winner,
        "winner_tier": cr.tier,
        "winner_tier_reason": cr.tier_reason,
        "evidence_gap": cr.evidence_gap,
        # Independent modules — captured even though they don't enter the
        # primary verdict, because they're ownership-relevant evidence.
        "dim_module": dict(cr.dim),
        "ph_module": {"betti_obs": cr.ph.get("betti_obs"),
                        "betti_std": cr.ph.get("betti_std"),
                        "status": cr.ph.get("status")},
        "fourier_module": dict(cr.fourier),
        "geom_per_shape": {n: dict(cr.geom.get(n, {})) for n in cr.geom},
        "holdout_per_shape": {n: dict(cr.holdout.get(n, {}))
                                for n in cr.holdout},
        "cross_seed_stability": dict(cr.cross_seed),
        "alpha_hat_used": cr.alpha_hat,
        "K_present_used": cr.K_present,
        "N_used": cr.N_used,
    }


# ─── Verdict logic ─────────────────────────────────────────────────────────

def decide_ownership(raw_meta: Dict, orth: Dict, thresh: Dict,
                       logger: logging.Logger,
                       raw_cell_dir: Optional[Path] = None) -> Dict:
    """BINARY ownership rule (post-2026-05-19 simplification).

    Verdict is determined solely by the sign of the Bayes-factor change:
        Δgap_vs_K0 = (raw_winner_logZ - raw_K0_logZ)
                      - (orth_winner_logZ - orth_K0_logZ)

      Δgap_vs_K0 < 0  → OWNED       (shape strengthens after orthogonalising
                                       out the algebraic correlates — the
                                       residual signal genuinely belongs to
                                       the concept)
      Δgap_vs_K0 ≥ 0  → INHERITED   (shape loses evidence after orthogonalising
                                       — the geometry was borrowed from the
                                       correlates' shared subspace)

    All other signals (alignment retention, family match/flip, perm survival,
    var_removed_frac) are recorded for downstream paper analysis but do NOT
    enter the verdict. Empirically (GPT-J n=420) the distribution of
    Δgap_vs_K0 is sharply bimodal — no cell has |Δgap| ∈ (0, 5) — so the
    sign rule is the same as any Kass-Raftery threshold.
    """
    raw_winner = raw_meta.get("winner_shape", "")
    raw_family = raw_meta.get("family_winner", "")
    raw_align_score = float(raw_meta.get(
        "align_per_shape", {}).get(raw_winner, {}).get("alignment_score", 0.0))
    raw_log_E = _raw_log_E(raw_meta, cell_dir=raw_cell_dir)
    raw_logZ_winner = float(raw_log_E.get(raw_winner, float("-inf")))
    raw_logZ_K0 = float(raw_log_E.get("K0_Generic", float("-inf")))
    raw_gap_vs_K0 = (raw_logZ_winner - raw_logZ_K0
                       if math.isfinite(raw_logZ_winner)
                          and math.isfinite(raw_logZ_K0)
                       else float("nan"))
    raw_perm_p = float(raw_meta.get("permutation", {}).get("p_value", 1.0))

    orth_score = orth.get("score_per_shape") or orth.get("log_E", {})
    orth_align = orth.get("alignment", {})
    orth_family = orth.get("family_winner", "")
    orth_perm_p = float(orth.get("perm_p", float("nan")))

    logZ_orth_at_raw_winner = float(orth_score.get(raw_winner, float("-inf")))
    logZ_orth_K0 = float(orth_score.get("K0_Generic", float("-inf")))
    orth_gap_vs_K0 = (logZ_orth_at_raw_winner - logZ_orth_K0
                        if math.isfinite(logZ_orth_at_raw_winner)
                           and math.isfinite(logZ_orth_K0)
                        else float("nan"))
    align_orth_at_raw_winner = float(abs(orth_align.get(raw_winner, 0.0)))

    delta_gap_vs_K0 = (raw_gap_vs_K0 - orth_gap_vs_K0
                         if math.isfinite(raw_gap_vs_K0)
                            and math.isfinite(orth_gap_vs_K0)
                         else float("nan"))
    delta_logZ = float(raw_logZ_winner - logZ_orth_at_raw_winner)
    if abs(raw_align_score) < 1e-6:
        align_retain_ratio = 1.0
    else:
        align_retain_ratio = align_orth_at_raw_winner / abs(raw_align_score)
    family_match = (orth_family == raw_family) and bool(raw_family)
    raw_is_nontrivial = raw_family in ("1D_periodic", "2D_periodic")
    family_flip = (raw_is_nontrivial
                    and orth_family in ("trivial", "")
                    and not family_match)

    # BINARY decision on Δgap_vs_K0 sign.
    if not math.isfinite(delta_gap_vs_K0):
        verdict = "indeterminate"
    elif delta_gap_vs_K0 < 0.0:
        verdict = "owned"
    else:
        verdict = "inherited"
    return {
        "verdict": verdict,
        "verdict_rule": "sign(delta_gap_vs_K0)",
        "raw_winner_shape": raw_winner,
        "raw_family": raw_family,
        "raw_logZ_winner": raw_logZ_winner,
        "raw_logZ_K0": raw_logZ_K0,
        "raw_gap_vs_K0": raw_gap_vs_K0,
        "raw_alignment": raw_align_score,
        "raw_perm_p": raw_perm_p,
        "orth_logZ_at_raw_winner": logZ_orth_at_raw_winner,
        "orth_logZ_K0": logZ_orth_K0,
        "orth_gap_vs_K0": orth_gap_vs_K0,
        "orth_alignment_at_raw_winner": align_orth_at_raw_winner,
        "orth_family_winner": orth_family,
        "orth_perm_p": orth_perm_p,
        "delta_logZ": delta_logZ,
        "delta_gap_vs_K0": delta_gap_vs_K0,
        "alignment_retain_ratio": align_retain_ratio,
        "family_match": bool(family_match),
        "family_flip_from_nontrivial": bool(family_flip),
    }


def _raw_log_E(raw_meta: Dict, cell_dir: Optional[Path] = None
                 ) -> Dict[str, float]:
    """Pull per-shape REFINED log_E from raw Stage 2c, preferring the
    `evidence_per_shape.csv` table written next to metadata.json.

    Fallback path (older runs): cross_shape_seed_stability per-combo averages.
    """
    if cell_dir is not None:
        ev_path = cell_dir / "evidence_per_shape.csv"
        if ev_path.exists():
            try:
                df = pd.read_csv(ev_path)
                d = {}
                for _, r in df.iterrows():
                    name = r["shape"]
                    val = r["log_E_refined"] if pd.notna(
                        r["log_E_refined"]) else r["log_E"]
                    d[str(name)] = float(val)
                if d:
                    return d
            except Exception:
                pass
    css = raw_meta.get("cross_shape_seed_stability", {})
    per_combo = css.get("per_combo", []) if isinstance(css, dict) else []
    log_E: Dict[str, float] = {}
    if per_combo:
        from collections import defaultdict
        accum = defaultdict(list)
        for r in per_combo:
            for k, v in r.get("log_E_per_shape", {}).items():
                accum[k].append(float(v))
        for k, vs in accum.items():
            log_E[k] = float(np.mean(vs))
    return log_E


# ─── Cell loading + assembly ───────────────────────────────────────────────

def _load_raw_meta(results_root: Path, model: str, task: str, mode: str,
                     layer: int, concept: str) -> Optional[Dict]:
    p = raw_metadata_path(results_root, model, task, mode, layer, concept)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _load_cell_inputs_for_stage3(paths, model, task, mode, layer, concept,
                                   prob_df, correct_mask, X_all):
    """Same as worker._load_cell_inputs but returns more pieces we need."""
    labels_full = prob_df[concept].to_numpy()
    labels_correct = labels_full[correct_mask]
    unique_full_values = sorted(prob_df[concept].dropna().unique().tolist())
    K_natural = len(unique_full_values)
    value_to_code = {v: i for i, v in enumerate(unique_full_values)}
    label_codes = np.array(
        [value_to_code.get(v, -1) for v in labels_correct], dtype=np.int64)
    keep = label_codes >= 0
    X = X_all[correct_mask][keep].astype(np.float64)
    label_codes = label_codes[keep]
    B_u, union_meta = worker.build_union_basis(
        paths["results_root"], model, task, mode, layer, concept)
    if union_meta["k_u"] < 1:
        return None
    mu_layer = X.mean(axis=0)
    X_centred = X - mu_layer
    Z = X_centred @ B_u.astype(np.float64)
    return {
        "Z": Z, "X_centred": X_centred, "B_u": B_u.astype(np.float64),
        "mu_layer": mu_layer, "label_codes": label_codes,
        "K_natural": K_natural, "union_meta": union_meta,
    }


def _kept_codes_from_raw_meta(raw_meta: Dict) -> List[int]:
    kc = raw_meta.get("kept_codes", [])
    return [int(v) for v in kc]


# ─── Main cell driver ──────────────────────────────────────────────────────

def analyse_cell(model: str, task: str, mode: str, layer: int, concept: str,
                   paths: dict, stage3_cfg: dict,
                   X_all: np.ndarray, correct_mask: np.ndarray,
                   prob_df: pd.DataFrame, logger: logging.Logger,
                   force: bool = False) -> Optional[Dict]:
    out_dir = stage3_cell_dir(paths["results_root"], model, task, mode, layer,
                                 concept)
    meta_out = out_dir / "stage3_metadata.json"
    if meta_out.exists() and not force:
        try:
            with open(meta_out) as f:
                j = json.load(f)
            if j.get("computation_status") == "complete":
                return None
        except Exception:
            pass

    cell_id = f"{model}|{task}|{mode}|{layer:02d}|{concept}"

    raw_meta = _load_raw_meta(paths["results_root"], model, task, mode, layer,
                                 concept)
    if raw_meta is None or raw_meta.get("computation_status") != "complete":
        logger.info("[%s] no completed raw Stage 2c metadata — skip", cell_id)
        return {"status": "no_raw_stage2c", "cell_id": cell_id}

    raw_tier = raw_meta.get("tier", "")
    SHAPE_TIERS = {"tier_A_named_shape", "tier_A_named_family", "tier_B_family"}
    if raw_tier not in SHAPE_TIERS:
        logger.info("[%s] raw tier=%s has no shape — skip Stage 3", cell_id,
                      raw_tier)
        _write_skip(out_dir, cell_id, raw_meta,
                     reason=f"raw_tier={raw_tier}_no_shape")
        return {"status": "no_shape_tier", "cell_id": cell_id}

    correlates = (stage3_cfg.get(task, {}) or {}).get(concept, [])
    if not correlates:
        logger.info("[%s] no correlates pre-registered — skip Stage 3", cell_id)
        _write_skip(out_dir, cell_id, raw_meta,
                     reason="no_correlates_preregistered")
        return {"status": "no_correlates", "cell_id": cell_id}

    loaded = _load_cell_inputs_for_stage3(
        paths, model, task, mode, layer, concept, prob_df, correct_mask, X_all)
    if loaded is None:
        logger.warning("[%s] cell inputs missing — skip", cell_id)
        _write_skip(out_dir, cell_id, raw_meta,
                     reason="cell_inputs_missing")
        return {"status": "cell_inputs_missing", "cell_id": cell_id}

    Z = loaded["Z"]
    X_centred = loaded["X_centred"]
    B_u = loaded["B_u"]
    label_codes = loaded["label_codes"]
    K_natural = loaded["K_natural"]

    Q, Q_meta = build_correlate_Q(paths["results_root"], model, task, mode,
                                     layer, correlates)
    if Q.shape[1] == 0:
        logger.info("[%s] no correlate bases on disk — skip", cell_id)
        _write_skip(out_dir, cell_id, raw_meta,
                     reason="no_correlate_bases_on_disk")
        return {"status": "no_correlate_bases", "cell_id": cell_id}

    Z_orth = orthogonalise_to_subspace(Z, X_centred, B_u, Q)

    # Diagnostics: how much variance does orthogonalisation remove inside B_u?
    var_raw = float(np.var(Z))
    var_orth = float(np.var(Z_orth))
    var_removed_frac = 1.0 - (var_orth / max(var_raw, 1e-12))

    # Locked priors from raw metadata
    alpha_locked = float(raw_meta.get("alpha_hat", 1.0))
    P_hat = raw_meta.get("P_hat_stage9_or_stage2a")
    if isinstance(P_hat, str):
        try: P_hat = float(P_hat)
        except Exception: P_hat = None
    P_hat_2 = raw_meta.get("P_hat_2")
    if isinstance(P_hat_2, str):
        try: P_hat_2 = float(P_hat_2)
        except Exception: P_hat_2 = None

    # kept_codes is no longer used by the full-pipeline orth path —
    # analyze_cell re-derives it internally from MIN_GROUP_SIZE on Y_orth.
    # We still record raw kept_codes for the metadata.
    raw_kept_codes = _kept_codes_from_raw_meta(raw_meta)

    logger.info("[%s] Stage 3: |Q|=%d, raw_kept_codes=%d, alpha_locked=%.3g, "
                 "P_hat=%s, var_removed=%.3f",
                 cell_id, Q.shape[1], len(raw_kept_codes), alpha_locked,
                 str(P_hat), var_removed_frac)

    union_meta = loaded["union_meta"]
    bsmir_cfg_obj = stage3_cfg.get("bsmir_cfg", {})
    orth_eval = evaluate_orth_evidence(
        Z_orth, label_codes, K_natural, alpha_locked, P_hat, P_hat_2,
        model, task, mode, layer, concept, union_meta, bsmir_cfg_obj,
        logger)

    if orth_eval.get("status") != "ok":
        _write_skip(out_dir, cell_id, raw_meta,
                     reason=f"orth_eval_status={orth_eval.get('status')}",
                     Q_meta=Q_meta, var_removed_frac=var_removed_frac)
        return {"status": orth_eval.get("status"), "cell_id": cell_id}

    raw_cell_dir = (paths["results_root"] / "stage2c_gplvm" / model / task
                       / f"mode_{mode}" / f"layer_{layer:02d}" / concept)
    # Binary verdict — sign of Δgap_vs_K0; thresh dict no longer used.
    verdict_full = decide_ownership(
        raw_meta, orth_eval, thresh={}, logger=logger,
        raw_cell_dir=raw_cell_dir)

    # Persist
    _write_complete(out_dir, cell_id, model, task, mode, layer, concept,
                     raw_meta, orth_eval, verdict_full, Q_meta,
                     correlates=correlates,
                     var_removed_frac=var_removed_frac,
                     alpha_locked=alpha_locked, P_hat=P_hat, P_hat_2=P_hat_2)
    logger.info("[%s] verdict=%s  Δlogz=%.2f  align_retain=%.3f  family=%s->%s",
                 cell_id, verdict_full["verdict"], verdict_full["delta_logZ"],
                 verdict_full["alignment_retain_ratio"],
                 verdict_full["raw_family"], verdict_full["orth_family_winner"])
    return {"status": "ok", "cell_id": cell_id, **verdict_full}


# (the old _merged_verdict_cfg helper was removed when the verdict rule
# switched to binary sign-of-Δgap_vs_K0.)


def _write_skip(out_dir: Path, cell_id: str, raw_meta: Dict, reason: str,
                  Q_meta: Optional[Dict] = None,
                  var_removed_frac: Optional[float] = None) -> None:
    parts = cell_id.split("|")
    row = {
        "model": parts[0], "task": parts[1], "mode": parts[2],
        "layer": int(parts[3]) if parts[3].isdigit() else -1,
        "concept": parts[4],
        "verdict": "skipped",
        "skip_reason": reason,
        "raw_winner_shape": raw_meta.get("winner_shape", ""),
        "raw_family": raw_meta.get("family_winner", ""),
        "raw_tier": raw_meta.get("tier", ""),
        "delta_logZ": float("nan"),
        "alignment_retain_ratio": float("nan"),
        "family_match": False,
        "family_flip_from_nontrivial": False,
        "orth_perm_p": float("nan"),
        "k_Q": (Q_meta or {}).get("k_Q", 0),
        "var_removed_frac": var_removed_frac if var_removed_frac is not None
                            else float("nan"),
    }
    atomic_csv(pd.DataFrame([row]), out_dir / "stage3_results.csv")
    atomic_json({"cell_id": cell_id, "computation_status": "complete",
                  "status": "skipped", "skip_reason": reason,
                  "Q_meta": Q_meta or {}}, out_dir / "stage3_metadata.json")


def _write_complete(out_dir: Path, cell_id: str, model: str, task: str,
                      mode: str, layer: int, concept: str,
                      raw_meta: Dict, orth_eval: Dict, verdict: Dict,
                      Q_meta: Dict, correlates: List[str],
                      var_removed_frac: float, alpha_locked: float,
                      P_hat: Optional[float], P_hat_2: Optional[float]
                      ) -> None:
    # Summary row
    row = {
        "model": model, "task": task, "mode": mode, "layer": int(layer),
        "concept": concept,
        "verdict": verdict["verdict"],
        "raw_winner_shape": verdict["raw_winner_shape"],
        "raw_family": verdict["raw_family"],
        "raw_tier": raw_meta.get("tier", ""),
        "raw_logZ_winner": verdict["raw_logZ_winner"],
        "raw_logZ_K0": verdict["raw_logZ_K0"],
        "raw_gap_vs_K0": verdict["raw_gap_vs_K0"],
        "raw_alignment": verdict["raw_alignment"],
        "raw_perm_p": verdict["raw_perm_p"],
        "orth_logZ_at_raw_winner": verdict["orth_logZ_at_raw_winner"],
        "orth_logZ_K0": verdict["orth_logZ_K0"],
        "orth_gap_vs_K0": verdict["orth_gap_vs_K0"],
        "orth_alignment_at_raw_winner": verdict["orth_alignment_at_raw_winner"],
        "orth_family_winner": verdict["orth_family_winner"],
        "orth_perm_p": verdict["orth_perm_p"],
        "verdict_rule": verdict.get("verdict_rule", "sign(delta_gap_vs_K0)"),
        "delta_logZ": verdict["delta_logZ"],
        "delta_gap_vs_K0": verdict["delta_gap_vs_K0"],
        "alignment_retain_ratio": verdict["alignment_retain_ratio"],
        "family_match": verdict["family_match"],
        "family_flip_from_nontrivial": verdict["family_flip_from_nontrivial"],
        # Orth-side full-pipeline outputs (BSMI-R was re-run, so these reflect
        # the ownership-relevant changes to every independent module)
        "orth_winner_shape_bsmir": orth_eval.get("winner_named", ""),
        "orth_winner_tier_bsmir": orth_eval.get("winner_tier", ""),
        "orth_evidence_gap_bsmir": orth_eval.get("evidence_gap", float("nan")),
        "orth_family_winner_bsmir": orth_eval.get("family_winner_bsmir", ""),
        "orth_dim_hat": orth_eval.get("dim_module", {}).get(
            "d_hat", float("nan")),
        "orth_dim_estimators_agree": orth_eval.get("dim_module", {}).get(
            "estimators_agree", False),
        "orth_PH_status": orth_eval.get("ph_module", {}).get("status", "n/a"),
        "orth_winner_identity_stable": orth_eval.get(
            "cross_seed_stability", {}).get("winner_identity_stable", False),
        "orth_family_identity_stable": orth_eval.get(
            "cross_seed_stability", {}).get("family_identity_stable", False),
        "k_Q": Q_meta.get("k_Q", 0),
        "k_u": int(raw_meta.get("k_u", 0)),
        "var_removed_frac": var_removed_frac,
        "n_correlates_used": int(sum(
            1 for c in Q_meta.get("contributions", [])
            if c.get("k_u_c", 0) > 0)),
        "n_correlates_total": len(correlates),
        "alpha_locked": alpha_locked,
        "P_hat_locked": P_hat,
        "P_hat_2_locked": P_hat_2,
        "K_present_used": orth_eval.get("K_present_used", 0),
        "N_used": orth_eval.get("N_used", 0),
    }
    atomic_csv(pd.DataFrame([row]), out_dir / "stage3_results.csv")

    # Per-shape evidence under orth
    evidence_rows = []
    for shape_name, lE in orth_eval["log_E"].items():
        lE_refined = orth_eval.get("log_E_refined", {}).get(shape_name, lE)
        geom = orth_eval.get("geom_per_shape", {}).get(shape_name, {})
        ho   = orth_eval.get("holdout_per_shape", {}).get(shape_name, {})
        evidence_rows.append({
            "shape": shape_name,
            "family": shapes_mod.SHAPE_TO_FAMILY.get(shape_name, "trivial"),
            "log_E_orth": lE,
            "log_E_refined_orth": lE_refined,
            "alignment_orth": orth_eval["alignment"].get(shape_name,
                                                            float("nan")),
            "geom_status_orth": geom.get("geom_status", "n/a"),
            "mse_holdout_orth": ho.get("mse", float("nan")),
            "mse_lvo_orth": ho.get("mse_lvo", float("nan")),
            "best_theta_P": orth_eval["best_theta"].get(shape_name, {}).get(
                "P", float("nan")),
            "best_theta_P2": orth_eval["best_theta"].get(shape_name, {}).get(
                "P2", float("nan")),
        })
    atomic_csv(pd.DataFrame(evidence_rows), out_dir / "evidence_orth.csv")

    # Permutation null
    atomic_save(np.asarray(orth_eval.get("perm_null", []), dtype=np.float64),
                  out_dir / "perm_null_orth.npy")

    # Full metadata
    atomic_json({
        "cell_id": cell_id, "computation_status": "complete",
        "status": "ok",
        "verdict": verdict,
        "raw_meta_summary": {
            "winner_shape": raw_meta.get("winner_shape", ""),
            "family_winner": raw_meta.get("family_winner", ""),
            "tier": raw_meta.get("tier", ""),
            "alpha_hat": raw_meta.get("alpha_hat", 1.0),
            "P_hat": raw_meta.get("P_hat_stage9_or_stage2a"),
            "kept_codes_n": len(raw_meta.get("kept_codes", [])),
        },
        "orth_eval": {
            "log_E": orth_eval["log_E"],
            "log_E_refined": orth_eval.get("log_E_refined", {}),
            "alignment": orth_eval["alignment"],
            "family_log_E": orth_eval["family_log_E"],
            "family_winner": orth_eval["family_winner"],
            "family_runner_up": orth_eval["family_runner_up"],
            "family_evidence_gap": orth_eval["family_evidence_gap"],
            "family_winner_bsmir": orth_eval.get("family_winner_bsmir", ""),
            "family_evidence_gap_bsmir": orth_eval.get(
                "family_evidence_gap_bsmir", float("nan")),
            "perm_p": orth_eval["perm_p"],
            "perm_observed_T": orth_eval["perm_observed_T"],
            "winner_named": orth_eval["winner_named"],
            "winner_tier": orth_eval.get("winner_tier", ""),
            "winner_tier_reason": orth_eval.get("winner_tier_reason", ""),
            "evidence_gap": orth_eval.get("evidence_gap", float("nan")),
            "best_theta": orth_eval["best_theta"],
            "dim_module": orth_eval.get("dim_module", {}),
            "ph_module": orth_eval.get("ph_module", {}),
            "fourier_module": orth_eval.get("fourier_module", {}),
            "geom_per_shape": orth_eval.get("geom_per_shape", {}),
            "holdout_per_shape": orth_eval.get("holdout_per_shape", {}),
            "cross_seed_stability": orth_eval.get("cross_seed_stability", {}),
            "alpha_hat_used": orth_eval.get("alpha_hat_used",
                                                float("nan")),
            "K_present_used": orth_eval["K_present_used"],
            "N_used": orth_eval["N_used"],
        },
        "Q_meta": Q_meta,
        "correlates": correlates,
        "var_removed_frac": var_removed_frac,
        "alpha_locked": alpha_locked,
        "P_hat_locked": P_hat,
        "P_hat_2_locked": P_hat_2,
    }, out_dir / "stage3_metadata.json")


# ─── CLI ───────────────────────────────────────────────────────────────────

def setup_logging(logs_root: Path, tag: str) -> logging.Logger:
    logs_root.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(logs_root / f"stage3_{tag}.log")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger = logging.getLogger(f"stage3.{tag}")
    logger.handlers.clear()
    logger.addHandler(fh); logger.addHandler(sh)
    logger.setLevel(logging.INFO); logger.propagate = False
    return logger


def main_one_cell(args, cfg, paths, stage3_cfg) -> int:
    model = args.model; task = args.task; mode = args.mode
    layer = int(args.layer); concept = args.concept
    logger = setup_logging(paths["logs_root"],
                              f"{model}_{task}_{mode}_one")
    logger.info("=== Stage 3 single cell: %s/%s/%s/L%d/%s ===",
                  model, task, mode, layer, concept)
    X_path = (paths["activations_root"] / model / f"{task}_layer_{layer:02d}.npy"
                if mode == "off"
                else paths["results_root"] / "residualized" / model
                      / f"{task}_layer_{layer:02d}_mode_{mode}.npy")
    if not X_path.exists():
        logger.error("Activation file missing: %s", X_path); return 1
    X_all = np.load(X_path)
    ans = pd.read_csv(paths["data_root"] / "answers" / model
                         / f"{task}_answers.csv")
    correct_mask = ans["correct"].astype(bool).to_numpy()
    prob = pd.read_csv(paths["data_root"] / "data" / "raw"
                          / f"{task}_problems.csv")
    res = analyse_cell(model, task, mode, layer, concept, paths, stage3_cfg,
                          X_all, correct_mask, prob, logger, force=args.force)
    return 0 if (res and res.get("status") in ("ok",)) else 0


def main_sweep(args, cfg, paths, stage3_cfg) -> int:
    array_idx = int(args.array_task); array_size = max(1, int(args.array_size))
    logger = setup_logging(paths["logs_root"],
                              f"sweep_{args.model or 'all'}_"
                              f"{args.task or 'all'}_{args.mode or 'all'}")
    all_models = [m["key"] for m in cfg["models"]]
    models = [args.model] if args.model else all_models
    tasks = [args.task] if args.task else ["addition", "multiplication"]
    modes = [args.mode] if (args.mode and args.mode != "all") else ["off", "answer",
                                                                       "norm"]
    n_done = n_skip = n_err = 0
    for model in models:
        for task in tasks:
            ans_path = paths["data_root"] / "answers" / model / f"{task}_answers.csv"
            if not ans_path.exists():
                continue
            ans = pd.read_csv(ans_path)
            correct_mask = ans["correct"].astype(bool).to_numpy()
            prob_path = paths["data_root"] / "data" / "raw" / f"{task}_problems.csv"
            if not prob_path.exists():
                continue
            prob = pd.read_csv(prob_path)
            mcfg = next(m for m in cfg["models"] if m["key"] == model)
            for mode in modes:
                for layer in mcfg["layers"]:
                    X_path = (paths["activations_root"] / model
                                / f"{task}_layer_{layer:02d}.npy"
                                if mode == "off"
                                else paths["results_root"] / "residualized"
                                      / model
                                      / f"{task}_layer_{layer:02d}_mode_{mode}.npy")
                    if not X_path.exists():
                        continue
                    X_all = None
                    # Iterate all concepts that have a completed BSMI-R cell
                    stage2c_dir = (paths["results_root"] / "stage2c_gplvm"
                                     / model / task / f"mode_{mode}"
                                     / f"layer_{layer:02d}")
                    if not stage2c_dir.exists():
                        continue
                    for concept_dir in sorted(stage2c_dir.iterdir()):
                        if not concept_dir.is_dir():
                            continue
                        concept = concept_dir.name
                        cell_id = f"{model}|{task}|{mode}|{layer:02d}|{concept}"
                        stripe = int.from_bytes(
                            hashlib.sha256(cell_id.encode()).digest()[:4],
                            "big") % array_size
                        if stripe != array_idx:
                            continue
                        out_dir = stage3_cell_dir(paths["results_root"], model,
                                                     task, mode, layer, concept)
                        meta = out_dir / "stage3_metadata.json"
                        if meta.exists() and not args.force:
                            try:
                                j = json.load(open(meta))
                                if j.get("computation_status") == "complete":
                                    n_skip += 1; continue
                            except Exception:
                                pass
                        if X_all is None:
                            X_all = np.load(X_path)
                        if concept not in prob.columns:
                            continue
                        try:
                            res = analyse_cell(model, task, mode, layer, concept,
                                                 paths, stage3_cfg, X_all,
                                                 correct_mask, prob, logger,
                                                 force=args.force)
                            n_done += 1
                        except Exception as e:
                            logger.exception("Cell %s failed: %s", cell_id, e)
                            n_err += 1
    logger.info("=== stripe %d done. done=%d skipped=%d errors=%d ===",
                  array_idx, n_done, n_skip, n_err)
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--stage3-config", default=str(DEFAULT_STAGE3_CONFIG))
    p.add_argument("--model", default=None)
    p.add_argument("--task", default=None,
                    choices=[None, "addition", "multiplication"])
    p.add_argument("--mode", default="off",
                    choices=["off", "answer", "norm", "all"])
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--concept", default="")
    p.add_argument("--force", action="store_true")
    p.add_argument("--sweep", choices=["", "all"], default="")
    p.add_argument("--array-task", type=int, default=0)
    p.add_argument("--array-size", type=int, default=1)
    return p


def main() -> int:
    args = build_argparser().parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.stage3_config) as f:
        stage3_cfg = yaml.safe_load(f) or {}
    paths = derive_paths(cfg)
    if args.sweep == "all":
        return main_sweep(args, cfg, paths, stage3_cfg)
    if not args.model or not args.task or not args.concept:
        raise SystemExit("Either --sweep all OR "
                            "(--model X --task Y --layer L --concept C) "
                            "required.")
    return main_one_cell(args, cfg, paths, stage3_cfg)


if __name__ == "__main__":
    sys.exit(main())
