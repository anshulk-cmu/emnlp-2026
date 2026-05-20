#!/usr/bin/env python3
"""
Stage 4 causal validation — focused two-test pipeline.

Given a BSMI-R cell with a recovered geometric subspace, ask whether that
geometry is *causally* used by the model, or just correlationally present.

Two tests:
    1. Ablation        — zero out the subspace at one layer; measure Δlogit
                          on the gold first answer token. Test both:
                            (a) full union basis B_u (subspace granularity)
                            (b) recovered geometry Q_geom from BSMI-R (geometry
                                granularity)
                          Compare against random subspaces of matching rank.
    2. Activation patching — patch the geometric coordinates from a "donor"
                              prompt onto a "recipient" prompt; measure
                              Δlogit toward the donor's gold token. Same two
                              granularities.

Reads BSMI-R artifacts from results/stage2c_gplvm/<model>/<task>/mode_<mode>/
layer_<LL>/<concept>/{metadata.json, latent_winner.npy, W_winner.npy}.

Writes per-cell artifacts to results/stage4_causal/<model>/<task>/mode_<mode>/
layer_<LL>/<concept>/.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

import stage2c_gplvm as worker
import stage2c_shapes as shapes_mod


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
RNG_SEED = 42

N_ABLATION_MAX = 256
N_PATCH_PAIRS = 100
N_RANDOM_CONTROLS = 5
BATCH = 16


# ─── Atomic IO ───────────────────────────────────────────────────────────────

def atomic_save_npy(arr: np.ndarray, path: Path) -> None:
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


# ─── Paths + model loaders ───────────────────────────────────────────────────

def derive_paths(cfg: dict) -> dict:
    data_root = Path(cfg["paths"]["data_root"])
    results_root = Path(cfg["paths"]["results_root"])
    activations_root = Path(cfg["paths"].get("activations_root", data_root / "activations"))
    logs_root = Path(cfg["paths"].get("logs_root", data_root / "logs"))
    return {"data_root": data_root, "results_root": results_root,
            "activations_root": activations_root, "logs_root": logs_root}


def stage4_cell_dir(results_root: Path, model: str, task: str, mode: str,
                     layer: int, concept: str) -> Path:
    return (results_root / "stage4_causal" / model / task / f"mode_{mode}"
            / f"layer_{layer:02d}" / concept)


_LOADED_MODELS: Dict[str, Tuple] = {}


def load_lm(model_name: str, data_root: Path, logger: logging.Logger) -> Tuple:
    if model_name in _LOADED_MODELS:
        return _LOADED_MODELS[model_name]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_dir = data_root / "models" / model_name
    logger.info("Loading %s (bf16) from %s ...", model_name, model_dir)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=DTYPE, low_cpu_mem_usage=True).to(DEVICE).eval()
    n_layers = lm.config.n_layer if hasattr(lm.config, "n_layer") else lm.config.num_hidden_layers
    vram_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    logger.info("  loaded in %.1fs, VRAM=%.1f GB, n_layers=%d",
                  time.time() - t0, vram_gb, n_layers)
    _LOADED_MODELS[model_name] = (lm, tok, n_layers)
    return _LOADED_MODELS[model_name]


def get_block(lm, layer: int):
    if hasattr(lm, "transformer") and hasattr(lm.transformer, "h"):
        return lm.transformer.h[layer]
    if hasattr(lm, "model") and hasattr(lm.model, "layers"):
        return lm.model.layers[layer]
    if hasattr(lm, "gpt_neox") and hasattr(lm.gpt_neox, "layers"):
        return lm.gpt_neox.layers[layer]
    raise RuntimeError(f"Don't know how to access transformer blocks for {type(lm)}")


# ─── Cell loading: read BSMI-R artifacts ────────────────────────────────────

@dataclass
class Cell:
    model: str; task: str; mode: str; layer: int; concept: str
    X: np.ndarray             # (N, 4096) float32 correct activations
    codes: np.ndarray         # (N,) dense label codes
    B_u: np.ndarray           # (4096, k_u) union basis
    mu_layer: np.ndarray      # (4096,) layer mean of correct activations
    problems: pd.DataFrame    # per-correct-row problem metadata
    k_u: int
    # geometry-level subspace recovered from BSMI-R
    winner_shape: str         # e.g. "K6_Ribbon"
    winner_family: str        # e.g. "2D_periodic"
    d_geom: int               # column dim of latent_winner
    Q_geom: np.ndarray        # (4096, d_geom) orthonormal — the BSMI-R subspace
    z_latent: np.ndarray      # (N, d_geom) per-point latent (broadcast from per-label)
    tier: str
    geom_source: str          # "bsmir"


def _read_bsmir_cell(results_root: Path, model: str, task: str, mode: str,
                      layer: int, concept: str) -> Optional[Dict]:
    """Return BSMI-R cell info, or None if not on disk yet."""
    d = (results_root / "stage2c_gplvm" / model / task / f"mode_{mode}"
         / f"layer_{layer:02d}" / concept)
    meta_p = d / "metadata.json"
    if not meta_p.exists():
        return None
    try:
        j = json.load(open(meta_p))
    except Exception:
        return None
    if j.get("computation_status") != "complete":
        return None
    lat_p = d / "latent_winner.npy"
    W_p = d / "W_winner.npy"
    if not lat_p.exists() or not W_p.exists():
        return None
    return {
        "winner_shape": j.get("winner_shape", ""),
        "winner_family": j.get("family_winner", ""),
        "tier": j.get("tier", ""),
        "latent_winner": np.load(lat_p),
        "W_winner": np.load(W_p),
    }


def setup_cell(model: str, task: str, mode: str, layer: int, concept: str,
                paths: dict, logger: logging.Logger) -> Optional[Cell]:
    activations_root = paths["activations_root"]
    results_root = paths["results_root"]
    data_root = paths["data_root"]

    X_path = (activations_root / model / f"{task}_layer_{layer:02d}.npy"
              if mode == "off"
              else results_root / "residualized" / model
                    / f"{task}_layer_{layer:02d}_mode_{mode}.npy")
    if not X_path.exists():
        logger.warning("Missing activations: %s", X_path)
        return None

    bsmir = _read_bsmir_cell(results_root, model, task, mode, layer, concept)
    if bsmir is None:
        logger.warning("No BSMI-R artifact for this cell yet — cannot run causal.")
        return None
    # Only run causal on cells with a declared shape. Skip dim_only / refuse /
    # low_K — for those BSMI-R could not commit to a shape so there is no
    # geometry-level subspace to test.
    SHAPE_TIERS = {"tier_A_named_shape", "tier_A_named_family", "tier_B_family"}
    if bsmir["tier"] not in SHAPE_TIERS:
        logger.info("Skipping cell: BSMI-R tier=%s has no declared shape.",
                      bsmir["tier"])
        return None

    B_u, _meta = worker.build_union_basis(results_root, model, task, mode, layer, concept)
    if B_u.shape[1] < 1:
        logger.warning("Empty union basis.")
        return None

    X_all = np.load(X_path)
    ans = pd.read_csv(data_root / "answers" / model / f"{task}_answers.csv")
    mask = ans["correct"].astype(bool).to_numpy()
    if X_all.shape[0] != mask.shape[0]:
        logger.warning("Activation row count != answers row count.")
        return None
    X = X_all[mask].astype(np.float64)
    prob = pd.read_csv(data_root / "data" / "raw" / f"{task}_problems.csv")
    if concept not in prob.columns:
        logger.warning("Concept %r not in problems csv.", concept)
        return None
    labels = prob[concept].to_numpy()[mask]
    vals = sorted(prob[concept].dropna().unique().tolist())
    v2c = {v: i for i, v in enumerate(vals)}
    codes = np.array([v2c.get(v, -1) for v in labels], dtype=np.int64)
    keep = codes >= 0
    X = X[keep]
    codes = codes[keep]
    correct_probs = ans[ans["correct"] == 1].reset_index(drop=True)
    correct_probs = correct_probs.iloc[keep].reset_index(drop=True)

    mu_layer = X.mean(axis=0)

    # Geometry-level subspace from BSMI-R: use the FULL row-space of W_winner,
    # not the lower-dim latent parameterisation. BSMI-R fits a Gaussian-linear
    # regression  Z[v] = Phi(v) @ W  with W shape (n_basis, k_u). The shape
    # subspace BSMI-R actually models in probe coordinates is span(W rows) =
    # up to n_basis dims. Push it back to ambient via B_u and orthonormalise.
    #
    # Previous (broken) path used the d_latent-dim latent parameterisation
    # (e.g. 3 dims for K6_Ribbon's (cos, sin, t)) and lstsq'd through B_u —
    # that captures only a ~3-D slice of the full ~6-D shape regression,
    # leaving ~78% of the between-class scatter inside B_u untouched and
    # giving spuriously small Δacc on geometry ablation.
    lat = bsmir["latent_winner"]              # (K_present, d_latent) — kept for downstream patching
    W_winner = bsmir["W_winner"]              # (n_basis, k_u)
    K_present_bsmir = lat.shape[0]
    # codes may include values that BSMI-R filtered out via MIN_GROUP_SIZE.
    # Clamp to valid range and drop unrepresented points.
    valid = codes < K_present_bsmir
    if not valid.all():
        X = X[valid]
        codes = codes[valid]
        correct_probs = correct_probs.iloc[valid].reset_index(drop=True)
    z_latent = lat[codes].astype(np.float64)   # (N, d_latent) — still saved on Cell for M2 patching

    B_geom_amb = (B_u.astype(np.float64)
                   @ W_winner.T.astype(np.float64)).astype(np.float32)  # (4096, n_basis)
    if B_geom_amb.shape[1] == 1:
        v = B_geom_amb.reshape(-1)
        Q_geom = (v / (np.linalg.norm(v) + 1e-12)).reshape(-1, 1).astype(np.float32)
    else:
        Q_geom, _ = np.linalg.qr(B_geom_amb)
        Q_geom = Q_geom.astype(np.float32)
    d_geom = int(Q_geom.shape[1])

    logger.info("  Geometry for %s: %s (family=%s, n_basis=d_geom=%d, d_latent=%d)",
                  concept, bsmir["winner_shape"], bsmir["winner_family"],
                  d_geom, int(z_latent.shape[1]))
    return Cell(
        model=model, task=task, mode=mode, layer=layer, concept=concept,
        X=X.astype(np.float32), codes=codes,
        B_u=B_u.astype(np.float32), mu_layer=mu_layer.astype(np.float32),
        problems=correct_probs, k_u=int(B_u.shape[1]),
        winner_shape=bsmir["winner_shape"], winner_family=bsmir["winner_family"],
        d_geom=d_geom, Q_geom=Q_geom, z_latent=z_latent.astype(np.float32),
        tier=bsmir["tier"], geom_source="bsmir",
    )


# ─── Hook factories ─────────────────────────────────────────────────────────

def make_ablation_hook(P_np: np.ndarray, mu_np: np.ndarray):
    """Project the LAST-token residual out of the subspace P (orthonormal columns)
    and re-centre by adding mu back."""
    P_t = torch.tensor(P_np, dtype=DTYPE, device=DEVICE)
    mu_t = torch.tensor(mu_np, dtype=DTYPE, device=DEVICE)
    def hook(module, _input, output):
        h = output[0] if isinstance(output, tuple) else output
        last = h[:, -1, :]
        proj = (last - mu_t) @ P_t @ P_t.T
        new_last = last - proj
        h_new = h.clone()
        h_new[:, -1, :] = new_last
        return (h_new,) + output[1:] if isinstance(output, tuple) else h_new
    return hook


def make_patch_hook(donor_last_row: torch.Tensor, P_np: np.ndarray,
                     mu_np: np.ndarray):
    """Replace the recipient last-token's projection onto subspace P with the
    donor last-token's projection onto P."""
    P_t = torch.tensor(P_np, dtype=DTYPE, device=DEVICE)
    mu_t = torch.tensor(mu_np, dtype=DTYPE, device=DEVICE)
    donor_proj = (donor_last_row - mu_t) @ P_t @ P_t.T
    def hook(module, _input, output):
        h = output[0] if isinstance(output, tuple) else output
        recip_last = h[:, -1, :]
        recip_proj = (recip_last - mu_t) @ P_t @ P_t.T
        delta = donor_proj - recip_proj
        h_new = h.clone()
        h_new[:, -1, :] = recip_last + delta
        return (h_new,) + output[1:] if isinstance(output, tuple) else h_new
    return hook


# ─── Helpers: gold-token logit + random-subspace control ────────────────────

@torch.no_grad()
def batched_logits_at_token(lm, tok, prompts: List[str],
                              hook_fn=None, block=None) -> np.ndarray:
    """Returns (B, vocab) last-token logits for the given prompts; if hook_fn
    given, install it on `block` for the forward pass."""
    all_logits = []
    for i in range(0, len(prompts), BATCH):
        enc = tok(prompts[i:i + BATCH], return_tensors="pt",
                  padding=True, padding_side="left").to(DEVICE)
        if hook_fn is not None and block is not None:
            h = block.register_forward_hook(hook_fn)
            out = lm(**enc).logits[:, -1, :]
            h.remove()
        else:
            out = lm(**enc).logits[:, -1, :]
        all_logits.append(out.float().cpu().numpy())
    return np.concatenate(all_logits, axis=0)


def gold_logits(logits: np.ndarray, gold_ids: np.ndarray) -> np.ndarray:
    return logits[np.arange(len(gold_ids)), gold_ids]


def random_subspace(d_ambient: int, rank: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Random orthonormal (d_ambient, rank) subspace via QR on Gaussian noise."""
    G = rng.standard_normal((d_ambient, rank))
    Q, _ = np.linalg.qr(G)
    return Q[:, :rank].astype(np.float32)


# ─── Method 1: ablation ──────────────────────────────────────────────────────

def method_1_ablation(cell: Cell, lm, tok, block,
                       logger: logging.Logger) -> Dict:
    """Zero the subspace at the cell's layer; measure Δlogit on gold first-token.

    Two subspaces:
        (a) B_u — the full union basis (k_u dims)
        (b) Q_geom — the BSMI-R recovered geometry (d_geom dims)
    Compared against N_RANDOM_CONTROLS random subspaces of matching rank.
    """
    probs = cell.problems
    if "gold_first_token_id" not in probs.columns or "prompt" not in probs.columns:
        return {"status": "missing_columns"}
    sub = probs.head(N_ABLATION_MAX).reset_index(drop=True)
    prompts = sub["prompt"].tolist()
    gold = sub["gold_first_token_id"].to_numpy()

    base = batched_logits_at_token(lm, tok, prompts)
    base_gold = gold_logits(base, gold).mean()
    base_acc = float((base.argmax(-1) == gold).mean())

    rng = np.random.default_rng(RNG_SEED)
    out: Dict = {"status": "ok",
                 "n_test": len(sub),
                 "baseline": {"gold_logit_mean": float(base_gold),
                               "accuracy": base_acc}}

    # (a) subspace level: B_u
    P_Bu = cell.B_u                                  # (4096, k_u) already orthonormal-from-SVD
    h = make_ablation_hook(P_Bu, cell.mu_layer)
    ab = batched_logits_at_token(lm, tok, prompts, hook_fn=h, block=block)
    ab_gold = gold_logits(ab, gold).mean()
    ab_acc = float((ab.argmax(-1) == gold).mean())
    # random matched-rank controls
    rand_drops = []
    for k in range(N_RANDOM_CONTROLS):
        Q_r = random_subspace(P_Bu.shape[0], P_Bu.shape[1], rng)
        hr = make_ablation_hook(Q_r, cell.mu_layer)
        rb = batched_logits_at_token(lm, tok, prompts, hook_fn=hr, block=block)
        rand_drops.append({
            "gold_logit_mean": float(gold_logits(rb, gold).mean()),
            "accuracy": float((rb.argmax(-1) == gold).mean())})
    rand_mean = float(np.mean([r["gold_logit_mean"] for r in rand_drops]))
    rand_acc_mean = float(np.mean([r["accuracy"] for r in rand_drops]))
    out["subspace_B_u"] = {
        "rank": int(P_Bu.shape[1]),
        "gold_logit_mean": float(ab_gold),
        "Δgold_logit_vs_baseline": float(ab_gold - base_gold),
        "accuracy": ab_acc,
        "Δaccuracy_vs_baseline": float(ab_acc - base_acc),
        "random_controls": rand_drops,
        "random_mean_gold_logit": rand_mean,
        "random_mean_accuracy": rand_acc_mean,
        "causal_excess_logit": float((ab_gold - base_gold) - (rand_mean - base_gold)),
        "causal_excess_accuracy": float((ab_acc - base_acc) - (rand_acc_mean - base_acc)),
    }

    # (b) geometry level: Q_geom
    P_geom = cell.Q_geom
    h2 = make_ablation_hook(P_geom, cell.mu_layer)
    gb = batched_logits_at_token(lm, tok, prompts, hook_fn=h2, block=block)
    gb_gold = gold_logits(gb, gold).mean()
    gb_acc = float((gb.argmax(-1) == gold).mean())
    rand_drops_g = []
    for k in range(N_RANDOM_CONTROLS):
        Q_r = random_subspace(P_geom.shape[0], P_geom.shape[1], rng)
        hr = make_ablation_hook(Q_r, cell.mu_layer)
        rb = batched_logits_at_token(lm, tok, prompts, hook_fn=hr, block=block)
        rand_drops_g.append({
            "gold_logit_mean": float(gold_logits(rb, gold).mean()),
            "accuracy": float((rb.argmax(-1) == gold).mean())})
    rand_mean_g = float(np.mean([r["gold_logit_mean"] for r in rand_drops_g]))
    rand_acc_g = float(np.mean([r["accuracy"] for r in rand_drops_g]))
    out["geometry_Q_geom"] = {
        "shape": cell.winner_shape,
        "rank": int(P_geom.shape[1]),
        "gold_logit_mean": float(gb_gold),
        "Δgold_logit_vs_baseline": float(gb_gold - base_gold),
        "accuracy": gb_acc,
        "Δaccuracy_vs_baseline": float(gb_acc - base_acc),
        "random_controls": rand_drops_g,
        "random_mean_gold_logit": rand_mean_g,
        "random_mean_accuracy": rand_acc_g,
        "causal_excess_logit": float((gb_gold - base_gold) - (rand_mean_g - base_gold)),
        "causal_excess_accuracy": float((gb_acc - base_acc) - (rand_acc_g - base_acc)),
    }
    logger.info("  M1: base_acc=%.3f  | B_u Δlogit=%+.2f acc=%.3f (excess=%+.2f)  "
                  "| Q_geom Δlogit=%+.2f acc=%.3f (excess=%+.2f)",
                  base_acc,
                  out["subspace_B_u"]["Δgold_logit_vs_baseline"],
                  out["subspace_B_u"]["accuracy"],
                  out["subspace_B_u"]["causal_excess_logit"],
                  out["geometry_Q_geom"]["Δgold_logit_vs_baseline"],
                  out["geometry_Q_geom"]["accuracy"],
                  out["geometry_Q_geom"]["causal_excess_logit"])
    return out


# ─── Method 2: activation patching ──────────────────────────────────────────

def method_2_patching(cell: Cell, lm, tok, block,
                       logger: logging.Logger) -> Dict:
    """Patch the donor's last-token projection onto the subspace into the
    recipient's stream. Measure Δlogit on donor's gold token (should
    increase if the subspace causally carries the donor's identity).

    Pairs (donor, recipient) drawn from different concept-label values so
    the patch has a non-trivial effect to measure.
    """
    probs = cell.problems
    if "gold_first_token_id" not in probs.columns or "prompt" not in probs.columns:
        return {"status": "missing_columns"}
    rng = np.random.default_rng(RNG_SEED + 7)

    # Build distinct-label donor/recipient pairs.
    by_code: Dict[int, List[int]] = {}
    for i, c in enumerate(cell.codes):
        by_code.setdefault(int(c), []).append(i)
    available_codes = [c for c, idxs in by_code.items() if len(idxs) >= 2]
    if len(available_codes) < 2:
        return {"status": "too_few_label_values"}
    pairs = []
    for _ in range(N_PATCH_PAIRS):
        ca, cb = rng.choice(available_codes, size=2, replace=False)
        ia = int(rng.choice(by_code[int(ca)]))
        ib = int(rng.choice(by_code[int(cb)]))
        pairs.append((ia, ib))
    donor_idx = np.asarray([p[0] for p in pairs])
    recip_idx = np.asarray([p[1] for p in pairs])

    donor_prompts = probs["prompt"].iloc[donor_idx].tolist()
    recip_prompts = probs["prompt"].iloc[recip_idx].tolist()
    donor_gold = probs["gold_first_token_id"].iloc[donor_idx].to_numpy()
    recip_gold = probs["gold_first_token_id"].iloc[recip_idx].to_numpy()

    # Recipient baseline (no patch): donor-gold logit on recipient (should be low)
    base = batched_logits_at_token(lm, tok, recip_prompts)
    base_donor_logit = gold_logits(base, donor_gold).mean()
    base_recip_logit = gold_logits(base, recip_gold).mean()
    base_correct = float((base.argmax(-1) == recip_gold).mean())

    out: Dict = {"status": "ok", "n_pairs": len(pairs),
                 "baseline": {"recipient_correct_acc": base_correct,
                               "donor_gold_logit_on_recipient": float(base_donor_logit),
                               "recipient_gold_logit_on_recipient": float(base_recip_logit)}}

    # Capture donor last-token residuals at the target layer (one forward pass each
    # — using the same block hook to read out, no modification).
    captured: List[torch.Tensor] = [None] * len(donor_prompts)
    def make_capture_hook(slot_idx):
        def hook(module, _input, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[slot_idx] = h[:, -1, :].clone()
            return output
        return hook

    with torch.no_grad():
        for i in range(0, len(donor_prompts), BATCH):
            batch = donor_prompts[i:i + BATCH]
            enc = tok(batch, return_tensors="pt",
                       padding=True, padding_side="left").to(DEVICE)
            slot_starts = list(range(i, i + len(batch)))
            collected = [None]
            def cap(module, _input, output):
                h = output[0] if isinstance(output, tuple) else output
                collected[0] = h[:, -1, :].clone()
                return output
            hh = block.register_forward_hook(cap)
            _ = lm(**enc).logits
            hh.remove()
            for j, s in enumerate(slot_starts):
                captured[s] = collected[0][j:j+1].clone()

    def run_patch(P_np, mu_np):
        donor_logits_on_recip = np.zeros(len(pairs), dtype=np.float32)
        recip_logits_on_recip = np.zeros(len(pairs), dtype=np.float32)
        correct = 0
        with torch.no_grad():
            for i in range(0, len(pairs), BATCH):
                batch = recip_prompts[i:i + BATCH]
                enc = tok(batch, return_tensors="pt",
                          padding=True, padding_side="left").to(DEVICE)
                # use the i-th donor row for the i-th recipient
                d_row = torch.cat([captured[i + j] for j in range(len(batch))], dim=0)
                hh = block.register_forward_hook(
                    make_patch_hook(d_row, P_np, mu_np))
                lg = lm(**enc).logits[:, -1, :]
                hh.remove()
                lg_f = lg.float().cpu().numpy()
                for j in range(len(batch)):
                    donor_logits_on_recip[i + j] = lg_f[j, donor_gold[i + j]]
                    recip_logits_on_recip[i + j] = lg_f[j, recip_gold[i + j]]
                    if lg_f[j].argmax() == recip_gold[i + j]:
                        correct += 1
        return {"donor_logit_on_recip_mean": float(donor_logits_on_recip.mean()),
                "recip_logit_on_recip_mean": float(recip_logits_on_recip.mean()),
                "donor_flip_rate": float((lg_f.argmax(-1)[:1] == donor_gold[i:i+1]).mean()
                                            if False else 0.0),
                "recipient_correct_acc": float(correct / len(pairs))}

    rng2 = np.random.default_rng(RNG_SEED + 9)

    # (a) subspace level
    P_Bu = cell.B_u
    sub_res = run_patch(P_Bu, cell.mu_layer)
    # random-subspace control of matching rank
    rand_sub = []
    for _ in range(N_RANDOM_CONTROLS):
        Q_r = random_subspace(P_Bu.shape[0], P_Bu.shape[1], rng2)
        rand_sub.append(run_patch(Q_r, cell.mu_layer))
    rand_donor_mean = float(np.mean([r["donor_logit_on_recip_mean"] for r in rand_sub]))
    sub_res["random_donor_logit_mean"] = rand_donor_mean
    sub_res["causal_excess_donor_logit"] = float(
        sub_res["donor_logit_on_recip_mean"] - rand_donor_mean)
    sub_res["random_controls"] = rand_sub
    out["subspace_B_u"] = {"rank": int(P_Bu.shape[1]), **sub_res}

    # (b) geometry level
    P_geom = cell.Q_geom
    geo_res = run_patch(P_geom, cell.mu_layer)
    rand_geo = []
    for _ in range(N_RANDOM_CONTROLS):
        Q_r = random_subspace(P_geom.shape[0], P_geom.shape[1], rng2)
        rand_geo.append(run_patch(Q_r, cell.mu_layer))
    rand_donor_mean_g = float(np.mean([r["donor_logit_on_recip_mean"] for r in rand_geo]))
    geo_res["random_donor_logit_mean"] = rand_donor_mean_g
    geo_res["causal_excess_donor_logit"] = float(
        geo_res["donor_logit_on_recip_mean"] - rand_donor_mean_g)
    geo_res["random_controls"] = rand_geo
    out["geometry_Q_geom"] = {"shape": cell.winner_shape,
                                "rank": int(P_geom.shape[1]), **geo_res}

    logger.info("  M2: base_donor_logit=%+.2f base_recip_logit=%+.2f  "
                  "| B_u: donor_logit=%+.2f (excess=%+.2f)  "
                  "| Q_geom: donor_logit=%+.2f (excess=%+.2f)",
                  base_donor_logit, base_recip_logit,
                  sub_res["donor_logit_on_recip_mean"],
                  sub_res["causal_excess_donor_logit"],
                  geo_res["donor_logit_on_recip_mean"],
                  geo_res["causal_excess_donor_logit"])
    return out


# ─── Per-cell driver ────────────────────────────────────────────────────────

def run_cell(cell: Cell, lm, tok, block,
              logger: logging.Logger) -> Dict:
    t0 = time.time()
    out = {"cell": {"model": cell.model, "task": cell.task, "mode": cell.mode,
                     "layer": cell.layer, "concept": cell.concept,
                     "N": int(cell.X.shape[0]), "k_u": cell.k_u,
                     "winner_shape": cell.winner_shape,
                     "winner_family": cell.winner_family,
                     "tier": cell.tier, "d_geom": cell.d_geom,
                     "geom_source": cell.geom_source},
           "config": {"batch": BATCH, "rng_seed": RNG_SEED,
                       "n_ablation_max": N_ABLATION_MAX,
                       "n_patch_pairs": N_PATCH_PAIRS,
                       "n_random_controls": N_RANDOM_CONTROLS}}
    t = time.time()
    out["method_1_ablation"] = method_1_ablation(cell, lm, tok, block, logger)
    out["method_1_ablation"]["elapsed_seconds"] = time.time() - t
    t = time.time()
    out["method_2_patching"] = method_2_patching(cell, lm, tok, block, logger)
    out["method_2_patching"]["elapsed_seconds"] = time.time() - t
    out["total_elapsed_seconds"] = time.time() - t0
    return out


def write_cell_artifacts(out_dir: Path, results: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(results, out_dir / "stage4_results.json")
    row = {
        "model": results["cell"]["model"],
        "task": results["cell"]["task"],
        "mode": results["cell"]["mode"],
        "layer": results["cell"]["layer"],
        "concept": results["cell"]["concept"],
        "winner_shape": results["cell"]["winner_shape"],
        "winner_family": results["cell"]["winner_family"],
        "bsmir_tier": results["cell"]["tier"],
        "d_geom": results["cell"]["d_geom"],
        "k_u": results["cell"]["k_u"],
    }
    m1 = results["method_1_ablation"]
    if m1.get("status") == "ok":
        row.update({
            "m1_base_acc": m1["baseline"]["accuracy"],
            "m1_Bu_Δlogit": m1["subspace_B_u"]["Δgold_logit_vs_baseline"],
            "m1_Bu_Δacc": m1["subspace_B_u"]["Δaccuracy_vs_baseline"],
            "m1_Bu_causal_excess_logit": m1["subspace_B_u"]["causal_excess_logit"],
            "m1_Bu_causal_excess_acc": m1["subspace_B_u"]["causal_excess_accuracy"],
            "m1_geom_Δlogit": m1["geometry_Q_geom"]["Δgold_logit_vs_baseline"],
            "m1_geom_Δacc": m1["geometry_Q_geom"]["Δaccuracy_vs_baseline"],
            "m1_geom_causal_excess_logit": m1["geometry_Q_geom"]["causal_excess_logit"],
            "m1_geom_causal_excess_acc": m1["geometry_Q_geom"]["causal_excess_accuracy"],
        })
    m2 = results["method_2_patching"]
    if m2.get("status") == "ok":
        row.update({
            "m2_base_donor_logit": m2["baseline"]["donor_gold_logit_on_recipient"],
            "m2_Bu_donor_logit": m2["subspace_B_u"]["donor_logit_on_recip_mean"],
            "m2_Bu_causal_excess_donor_logit": m2["subspace_B_u"]["causal_excess_donor_logit"],
            "m2_geom_donor_logit": m2["geometry_Q_geom"]["donor_logit_on_recip_mean"],
            "m2_geom_causal_excess_donor_logit": m2["geometry_Q_geom"]["causal_excess_donor_logit"],
        })
    atomic_csv(pd.DataFrame([row]), out_dir / "stage4_summary.csv")


# ─── CLI ────────────────────────────────────────────────────────────────────

def setup_logging(logs_root: Path, model: str, task: str, mode: str
                    ) -> logging.Logger:
    logs_root.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(logs_root / f"stage4_causal_{model}_{task}_{mode}.log")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log = logging.getLogger(f"stage4.{model}.{task}.{mode}")
    log.handlers.clear()
    log.addHandler(fh)
    log.addHandler(sh)
    log.setLevel(logging.INFO)
    log.propagate = False
    return log


SHAPE_TIERS_CAUSAL = {"tier_A_named_shape", "tier_A_named_family", "tier_B_family"}


def discover_bsmir_cells(results_root: Path, model: str
                           ) -> List[Tuple[str, str, int, str, str]]:
    """Walk results/stage2c_gplvm/<model>/ and return every cell that has a
    declared shape (Tier A or Tier B family) ready for causal testing."""
    root = results_root / "stage2c_gplvm" / model
    if not root.exists():
        return []
    out = []
    for meta_p in root.glob("*/mode_*/layer_*/*/metadata.json"):
        parts = meta_p.parts
        try:
            task = parts[-5]
            mode = parts[-4].replace("mode_", "")
            layer = int(parts[-3].replace("layer_", ""))
            concept = parts[-2]
        except Exception:
            continue
        try:
            j = json.load(open(meta_p))
        except Exception:
            continue
        if j.get("computation_status") != "complete":
            continue
        if j.get("tier") not in SHAPE_TIERS_CAUSAL:
            continue
        out.append((task, mode, layer, concept, j["tier"]))
    return sorted(set(out))


def cell_stripe(model: str, task: str, mode: str, layer: int, concept: str,
                  array_size: int) -> int:
    import hashlib
    s = f"stage4|{model}|{task}|{mode}|{layer:02d}|{concept}"
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "big") % array_size


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--model", required=True)
    p.add_argument("--task", default=None, choices=["addition", "multiplication", None])
    p.add_argument("--mode", default="off", choices=["off", "answer", "norm", "all"])
    p.add_argument("--layer", default="",
                    help="Single layer (e.g. '14'), comma-separated list "
                         "('4,8,14,20,24'), or 'all' for every layer in the "
                         "model config (single-concept mode only).")
    p.add_argument("--concept", default="")
    p.add_argument("--sweep", choices=["", "all"], default="",
                    help="'all' = iterate every cell on disk with a declared shape.")
    p.add_argument("--array-task", type=int, default=0)
    p.add_argument("--array-size", type=int, default=1)
    p.add_argument("--force", action="store_true")
    return p


def _parse_layers(arg: str, cfg_layers: List[int]) -> List[int]:
    if not arg or arg == "all":
        return list(cfg_layers)
    return [int(x.strip()) for x in arg.split(",") if x.strip()]


def main_sweep(args, cfg, paths) -> int:
    log = setup_logging(paths["logs_root"], args.model, "all", args.mode)
    log.info("=== Stage 4 sweep: model=%s mode=%s array=%d/%d ===",
              args.model, args.mode, args.array_task, args.array_size)
    cells = discover_bsmir_cells(paths["results_root"], args.model)
    if args.task is not None:
        cells = [c for c in cells if c[0] == args.task]
    if args.mode != "all":
        cells = [c for c in cells if c[1] == args.mode]
    # Stripe assignment
    mine = [c for c in cells
            if cell_stripe(args.model, c[0], c[1], c[2], c[3], args.array_size)
                == args.array_task]
    log.info("Total cells discovered: %d   on this stripe: %d", len(cells), len(mine))
    if not mine:
        return 0
    lm = tok = None
    n_done = n_skipped = n_failed = 0
    for task, mode, layer, concept, tier in mine:
        out_dir = stage4_cell_dir(paths["results_root"], args.model, task, mode,
                                    layer, concept)
        if (out_dir / "stage4_results.json").exists() and not args.force:
            n_skipped += 1
            continue
        cell = setup_cell(args.model, task, mode, layer, concept, paths, log)
        if cell is None:
            n_failed += 1
            continue
        if lm is None:
            lm, tok, _ = load_lm(args.model, paths["data_root"], log)
        try:
            block = get_block(lm, layer)
            results = run_cell(cell, lm, tok, block, log)
            write_cell_artifacts(out_dir, results)
            n_done += 1
        except Exception as e:
            log.exception("Cell %s/%s/L%d/%s failed: %s",
                            task, mode, layer, concept, e)
            n_failed += 1
    log.info("=== Stripe %d done. processed=%d, skipped=%d, failed=%d ===",
              args.array_task, n_done, n_skipped, n_failed)
    return 0


def main():
    args = build_argparser().parse_args()
    cfg = yaml.safe_load(open(args.config))
    paths = derive_paths(cfg)
    if args.sweep == "all":
        return main_sweep(args, cfg, paths)
    if not args.task or not args.concept:
        raise SystemExit("Single-cell mode requires --task and --concept "
                          "(or use --sweep all).")
    mcfg = next(m for m in cfg["models"] if m["key"] == args.model)
    layers = _parse_layers(args.layer, mcfg["layers"])
    log = setup_logging(paths["logs_root"], args.model, args.task, args.mode)
    log.info("=== Stage 4 sweep across %d layer(s): %s / %s / mode_%s / %s ===",
              len(layers), args.model, args.task, args.mode, args.concept)
    lm, tok, _ = (None, None, None)
    summaries: List[Dict] = []
    for L in layers:
        log.info("--- L%02d ---", L)
        out_dir = stage4_cell_dir(paths["results_root"], args.model, args.task,
                                    args.mode, L, args.concept)
        done = out_dir / "stage4_results.json"
        if done.exists() and not args.force:
            log.info("  already done, skipping (use --force to overwrite)")
            try:
                summaries.append(json.load(open(done)))
            except Exception:
                pass
            continue
        cell = setup_cell(args.model, args.task, args.mode, L,
                           args.concept, paths, log)
        if cell is None:
            log.info("  setup_cell returned None (no BSMI-R artifact or no shape) — skipping")
            continue
        if lm is None:
            lm, tok, _ = load_lm(args.model, paths["data_root"], log)
        block = get_block(lm, L)
        results = run_cell(cell, lm, tok, block, log)
        write_cell_artifacts(out_dir, results)
        summaries.append(results)
        log.info("  L%02d done in %.1fs", L, results["total_elapsed_seconds"])
    if not summaries:
        log.warning("No layers produced results.")
        return 0
    # cross-layer summary
    rows = []
    for r in summaries:
        c = r["cell"]
        m1 = r.get("method_1_ablation", {})
        m2 = r.get("method_2_patching", {})
        row = {"layer": c["layer"], "tier": c["tier"], "winner_shape": c["winner_shape"]}
        if m1.get("status") == "ok":
            row["M1_Bu_excess_logit"] = m1["subspace_B_u"]["causal_excess_logit"]
            row["M1_Bu_excess_acc"] = m1["subspace_B_u"]["causal_excess_accuracy"]
            row["M1_geom_excess_logit"] = m1["geometry_Q_geom"]["causal_excess_logit"]
            row["M1_geom_excess_acc"] = m1["geometry_Q_geom"]["causal_excess_accuracy"]
        if m2.get("status") == "ok":
            row["M2_Bu_excess_donor"] = m2["subspace_B_u"]["causal_excess_donor_logit"]
            row["M2_geom_excess_donor"] = m2["geometry_Q_geom"]["causal_excess_donor_logit"]
        rows.append(row)
    summary_df = pd.DataFrame(rows).sort_values("layer")
    log.info("\n=== Cross-layer summary for %s/%s/mode_%s/%s ===\n%s",
              args.model, args.task, args.mode, args.concept,
              summary_df.to_string(index=False))
    cross_dir = paths["results_root"] / "stage4_causal" / args.model / args.task \
                 / f"mode_{args.mode}" / "_concept_layer_sweeps"
    atomic_csv(summary_df, cross_dir / f"{args.concept}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
