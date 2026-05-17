#!/usr/bin/env python3
"""
Causal validation of the K4_Torus geometry discovered by Stage 2c on
`gpt-j-6b/multiplication/off/L14/ans_units` (the smoke headline cell).

Four causal-inference methods, each at two granularities where applicable:
  * subspace level  — operate on the full 6-D Union(LDA-A, CCSVD) basis B_u
  * geometry level  — operate only on the 2-D K4_Torus subspace within B_u

Methods:
  1. Logit ablation (necessity)
       Zero the projection onto the target subspace at layer 14;
       measure Δlogit on the gold first answer token vs random-subspace control.

  2. Activation patching (sufficiency / directionality)
       For ordered problem pairs (P_recipient, P_donor) with different gold
       answers, transplant the target-subspace activations from donor → recipient.
       Measure how often the recipient's prediction flips toward the donor's
       answer.

  3. Steering (control-knob)
       Rotate each problem's current torus position by angle θ ∈ [0, 2π);
       record how the predicted digit shifts as a function of θ.

  4. Geodesic walk (interpretability)
       Replace the torus position with target points (cos θ, sin θ) on the
       unit circle for θ ∈ [0, 2π); plot which digit the model emits at each
       angular position.

Outputs:
  * /tmp/causal_torus/results.json  — raw measurements per method
  * Console summary table at the end
  * docs/10_causal_validation_torus_smoke.md is the human-readable write-up
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=UserWarning)

import stage2c_gplvm as worker
import stage2c_kernels as kernels


# ─── Defaults — overridable via CLI ─────────────────────────────────────────
DEFAULT_MODEL = "gpt-j-6b"
DEFAULT_TASK = "multiplication"
DEFAULT_MODE = "off"
DEFAULT_LAYER = 14
DEFAULT_CONCEPT = "ans_units"
DATA_ROOT = Path("/data/user_data/anshulk/emnlp2026")
RESULTS_ROOT = DATA_ROOT / "results"
GPTJ_DIR = "/data/user_data/anshulk/emnlp2026/models/gpt-j-6b"
OUT_ROOT = Path("/tmp/causal_torus")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Will be set by main() based on CLI args
MODEL_NAME = DEFAULT_MODEL
TASK = DEFAULT_TASK
MODE = DEFAULT_MODE
LAYER = DEFAULT_LAYER
CONCEPT = DEFAULT_CONCEPT
OUT_DIR = OUT_ROOT

BATCH = 16
DEVICE = "cuda"
DTYPE = torch.bfloat16
RNG_SEED = 42

# Sample sizes per method — chosen for ~20–30 min total wall on a free A6000
N_ABLATION_PROBLEMS = 212        # all single-digit-answer correct mults
N_PATCH_PAIRS = 100              # ordered (recipient, donor) pairs across digit classes
N_STEER_PER_DIGIT = 5            # starting problems per digit class
N_STEER_ANGLES = 18              # 20° resolution
N_GEODESIC_PROMPTS = 10          # repeats per angular bin for noise
N_GEODESIC_ANGLES = 36           # 10° resolution


# ─── Cell loading + K4_Torus refit ──────────────────────────────────────────

def setup_cell() -> Dict:
    """Load activations, codes, union basis; refit K4_Torus to recover the
    2-D latent positions; map them back into the 4096-D ambient subspace via
    OLS through the union basis."""
    print(f"Cell: {MODEL_NAME}/{TASK}/{MODE}/L{LAYER}/{CONCEPT}")
    X_all = np.load(DATA_ROOT / f"activations/{MODEL_NAME}/{TASK}_layer_{LAYER:02d}.npy")
    ans = pd.read_csv(DATA_ROOT / f"answers/{MODEL_NAME}/{TASK}_answers.csv")
    mask = ans["correct"].astype(bool).to_numpy()
    X = X_all[mask].astype(np.float64)
    prob = pd.read_csv(DATA_ROOT / f"data/raw/{TASK}_problems.csv")
    labels = prob[CONCEPT].to_numpy()[mask]
    vals = sorted(prob[CONCEPT].dropna().unique().tolist())
    codes = np.array(
        [vals.index(v) if v in vals else -1 for v in labels], dtype=np.int64
    )
    keep = codes >= 0
    X = X[keep]
    codes = codes[keep]
    B_u, _ = worker.build_union_basis(
        RESULTS_ROOT, MODEL_NAME, TASK, MODE, LAYER, CONCEPT
    )
    mu_layer = X.mean(axis=0)
    Z = (X - mu_layer) @ B_u.astype(np.float64)
    print(f"  N={X.shape[0]}, ambient=4096, k_u={B_u.shape[1]}")

    # Refit K4_Torus to recover latent positions
    ps = worker.PeriodSeed(P_init=10.0, regime="wide", source="causal_val")
    ps2 = worker.PeriodSeed(P_init=5.0, regime="wide", source="causal_val")
    t0 = time.time()
    r4 = worker.fit_kernel_one_seed(
        Z, None, "K4_Torus", ps, ps2,
        d_max=5, n_iters=100, seed=42, device=DEVICE,
    )
    print(f"  K4_Torus refit: log_lik={r4['log_marginal_likelihood']:+.1f} "
          f"in {time.time() - t0:.1f}s")
    z_torus = np.array(r4["z_final"])

    # Map z_torus (N, 2) → ambient 2-D subspace via OLS through union basis
    W, *_ = np.linalg.lstsq(Z, z_torus, rcond=None)
    B_torus_ambient = (B_u.astype(np.float64) @ W).astype(np.float32)
    Q_torus, _ = np.linalg.qr(B_torus_ambient)   # (4096, 2), orthonormal

    # Indices and prompts
    correct_problems = ans[ans["correct"] == 1].reset_index(drop=True)
    # `codes` is aligned to keep-mask-filtered correct activations; the
    # answer CSV's `correct==1` rows correspond to the same population (the
    # build_union_basis upstream loaded the same activation file).
    assert len(correct_problems) == X.shape[0], (
        f"Population mismatch: codes {X.shape[0]} vs answers {len(correct_problems)}"
    )

    return {
        "X": X,
        "codes": codes,
        "B_u": B_u.astype(np.float32),
        "Q_torus": Q_torus.astype(np.float32),
        "mu_layer": mu_layer.astype(np.float32),
        "z_torus": z_torus.astype(np.float32),
        "problems": correct_problems,
        "K4_log_lik": float(r4["log_marginal_likelihood"]),
        "k_u": int(B_u.shape[1]),
    }


def setup_gptj():
    """Load GPT-J 6B in bf16; set pad_token; return (model, tok)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Loading GPT-J 6B (bf16) ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(GPTJ_DIR)
    tok.pad_token = tok.eos_token
    model = (AutoModelForCausalLM
              .from_pretrained(GPTJ_DIR, dtype=DTYPE)
              .to(DEVICE)
              .eval())
    print(f"  loaded in {time.time() - t0:.1f}s, "
          f"VRAM={torch.cuda.memory_allocated() / 1e9:.1f} GB")
    return model, tok


# ─── Hook helpers (all live on transformer.h[LAYER]) ────────────────────────

def block_l14(model):
    return model.transformer.h[LAYER]


def make_ablation_hook(P_np: np.ndarray, mu_np: np.ndarray):
    """Subtract projection onto P from the layer-LAYER output."""
    P_t = torch.tensor(P_np, dtype=DTYPE, device=DEVICE)
    mu_t = torch.tensor(mu_np, dtype=DTYPE, device=DEVICE)

    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        proj = (h - mu_t) @ P_t
        h_new = h - proj
        return (h_new,) + output[1:] if isinstance(output, tuple) else h_new

    return hook


def make_capture_hook(slot: List):
    """Capture the layer-LAYER output of the LAST sequence position per batch."""
    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        slot.append(h[:, -1, :].detach().clone())
        return output

    return hook


def make_patch_hook(donor_last_row: torch.Tensor, B_basis_t: torch.Tensor,
                     mu_t: torch.Tensor):
    """Replace recipient's basis-projected last-row activation with donor's.

    `donor_last_row` shape (B, 4096) — bf16.
    `B_basis_t` shape (4096, k) — orthonormal columns of the target subspace.
    """
    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        recipient_last = h[:, -1, :]
        # subtract recipient's projection, add donor's
        delta = ((donor_last_row - recipient_last) @ B_basis_t) @ B_basis_t.T
        h_patched = h.clone()
        h_patched[:, -1, :] = recipient_last + delta
        return (h_patched,) + output[1:] if isinstance(output, tuple) else h_patched

    return hook


def make_steering_hook(theta: float, Q_torus_t: torch.Tensor, mu_t: torch.Tensor):
    """Rotate the last token's torus-2D position by theta radians."""
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c, -s], [s, c]], dtype=DTYPE, device=DEVICE)

    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        last = h[:, -1, :]
        z = (last - mu_t) @ Q_torus_t            # (B, 2) current torus pos
        z_new = z @ R.T                          # rotate
        delta = (z_new - z) @ Q_torus_t.T        # ambient delta
        h_patched = h.clone()
        h_patched[:, -1, :] = last + delta
        return (h_patched,) + output[1:] if isinstance(output, tuple) else h_patched

    return hook


def make_geodesic_hook(target_z: torch.Tensor, Q_torus_t: torch.Tensor,
                        mu_t: torch.Tensor, scale: float):
    """Replace torus-2D position with `target_z * scale` on last token."""
    target = (target_z * scale).to(DTYPE).to(DEVICE)        # (2,)

    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        last = h[:, -1, :]
        current_z = (last - mu_t) @ Q_torus_t                # (B, 2)
        new_z = target.expand_as(current_z)
        delta = (new_z - current_z) @ Q_torus_t.T
        h_patched = h.clone()
        h_patched[:, -1, :] = last + delta
        return (h_patched,) + output[1:] if isinstance(output, tuple) else h_patched

    return hook


# ─── Method 1 — Logit ablation ──────────────────────────────────────────────

def method_1_ablation(model, tok, cell: Dict) -> Dict:
    print("\n=== Method 1 — Logit ablation ===")
    B_u = cell["B_u"]
    Q_torus = cell["Q_torus"]
    mu_layer = cell["mu_layer"]
    P_sub = (B_u @ B_u.T).astype(np.float32)
    P_torus = (Q_torus @ Q_torus.T).astype(np.float32)
    rng = np.random.default_rng(RNG_SEED)
    R6 = rng.standard_normal((4096, B_u.shape[1])).astype(np.float32); R6, _ = np.linalg.qr(R6)
    R2 = rng.standard_normal((4096, 2)).astype(np.float32); R2, _ = np.linalg.qr(R2)
    P_rand_sub = (R6 @ R6.T).astype(np.float32)
    P_rand_torus = (R2 @ R2.T).astype(np.float32)

    # Single-digit answers so the first token IS the units digit
    probs = cell["problems"]
    single = probs[probs["answer"] < 10].reset_index(drop=True).head(N_ABLATION_PROBLEMS)
    print(f"  {len(single)} single-digit-answer problems")

    def run_condition(name, P):
        handle = None if P is None else block_l14(model).register_forward_hook(
            make_ablation_hook(P, mu_layer)
        )
        logits_gold, n_correct = [], 0
        with torch.no_grad():
            for i in range(0, len(single), BATCH):
                batch = single.iloc[i:i + BATCH]
                enc = tok(batch["prompt"].tolist(), return_tensors="pt",
                          padding=True, padding_side="left").to(DEVICE)
                out = model(**enc).logits[:, -1, :]
                preds = out.argmax(-1).cpu().tolist()
                for j, gid in enumerate(batch["gold_first_token_id"].tolist()):
                    logits_gold.append(float(out[j, gid].cpu()))
                    if preds[j] == gid:
                        n_correct += 1
        if handle is not None:
            handle.remove()
        return n_correct / len(single), float(np.mean(logits_gold))

    results = {}
    acc, lg = run_condition("baseline", None)
    results["baseline"] = {"acc": acc, "mean_gold_logit": lg, "delta_logit": 0.0}
    print(f"  baseline:               acc={acc:.3f}  mean_logit={lg:+.2f}")
    for name, P in [
        ("ablate_Bu_sub", P_sub),
        ("ablate_random_sub", P_rand_sub),
        ("ablate_K4Torus_geo", P_torus),
        ("ablate_random_geo", P_rand_torus),
    ]:
        a, l = run_condition(name, P)
        results[name] = {"acc": a, "mean_gold_logit": l, "delta_logit": l - lg}
        print(f"  {name:25s} acc={a:.3f}  Δlogit={l - lg:+.2f}")

    sub_eff = (lg - results["ablate_Bu_sub"]["mean_gold_logit"]) - \
              (lg - results["ablate_random_sub"]["mean_gold_logit"])
    geo_eff = (lg - results["ablate_K4Torus_geo"]["mean_gold_logit"]) - \
              (lg - results["ablate_random_geo"]["mean_gold_logit"])
    results["subspace_causal_excess_logit"] = sub_eff
    results["geometry_causal_excess_logit"] = geo_eff
    print(f"  → subspace causal excess Δlogit: {sub_eff:+.2f}")
    print(f"  → geometry causal excess Δlogit: {geo_eff:+.2f}")
    return results


# ─── Method 2 — Activation patching ─────────────────────────────────────────

def method_2_patching(model, tok, cell: Dict) -> Dict:
    print("\n=== Method 2 — Activation patching ===")
    B_u = cell["B_u"]
    Q_torus = cell["Q_torus"]
    mu_layer = cell["mu_layer"]
    B_u_t = torch.tensor(B_u, dtype=DTYPE, device=DEVICE)
    Q_torus_t = torch.tensor(Q_torus, dtype=DTYPE, device=DEVICE)
    mu_t = torch.tensor(mu_layer, dtype=DTYPE, device=DEVICE)

    # Build pairs: for each digit pair (a≠b), pick a problem with answer a as
    # recipient, problem with answer b as donor. Sample N_PATCH_PAIRS pairs.
    probs = cell["problems"]
    by_digit = {d: probs[probs["answer"] == d].reset_index(drop=True)
                  for d in range(10) if (probs["answer"] == d).any()}
    rng = np.random.default_rng(RNG_SEED)
    pairs = []
    digits = sorted(by_digit.keys())
    while len(pairs) < N_PATCH_PAIRS:
        a, b = rng.choice(digits, size=2, replace=False)
        rdf, ddf = by_digit[int(a)], by_digit[int(b)]
        ri = int(rng.integers(0, len(rdf)))
        di = int(rng.integers(0, len(ddf)))
        pairs.append({
            "recipient_idx": int(rdf.iloc[ri]["index"]),
            "donor_idx": int(ddf.iloc[di]["index"]),
            "recipient_prompt": str(rdf.iloc[ri]["prompt"]),
            "donor_prompt": str(ddf.iloc[di]["prompt"]),
            "recipient_gold_id": int(rdf.iloc[ri]["gold_first_token_id"]),
            "donor_gold_id": int(ddf.iloc[di]["gold_first_token_id"]),
            "recipient_answer": int(rdf.iloc[ri]["answer"]),
            "donor_answer": int(ddf.iloc[di]["answer"]),
        })
    print(f"  {len(pairs)} ordered (recipient → donor) pairs across digit classes")

    def patch_level(name, B_basis_t):
        n_flip_to_donor, n_stay_recipient, n_other = 0, 0, 0
        delta_donor_logit_total = 0.0
        delta_recipient_logit_total = 0.0
        with torch.no_grad():
            for i in range(0, len(pairs), BATCH):
                chunk = pairs[i:i + BATCH]
                # Capture donor last-row activations
                donor_prompts = [p["donor_prompt"] for p in chunk]
                enc_d = tok(donor_prompts, return_tensors="pt", padding=True,
                            padding_side="left").to(DEVICE)
                slot: List[torch.Tensor] = []
                hcap = block_l14(model).register_forward_hook(make_capture_hook(slot))
                model(**enc_d)
                hcap.remove()
                donor_last = slot[0]                                  # (B, 4096)
                # Patch into recipient run
                recip_prompts = [p["recipient_prompt"] for p in chunk]
                enc_r = tok(recip_prompts, return_tensors="pt", padding=True,
                            padding_side="left").to(DEVICE)
                hpatch = block_l14(model).register_forward_hook(
                    make_patch_hook(donor_last, B_basis_t, mu_t)
                )
                out_patch = model(**enc_r).logits[:, -1, :]
                hpatch.remove()
                # Also run a clean recipient pass for baseline comparison
                out_clean = model(**enc_r).logits[:, -1, :]
                preds_patch = out_patch.argmax(-1).cpu().tolist()
                preds_clean = out_clean.argmax(-1).cpu().tolist()
                for j, p in enumerate(chunk):
                    if preds_patch[j] == p["donor_gold_id"]:
                        n_flip_to_donor += 1
                    elif preds_patch[j] == p["recipient_gold_id"]:
                        n_stay_recipient += 1
                    else:
                        n_other += 1
                    delta_donor_logit_total += float(
                        out_patch[j, p["donor_gold_id"]].cpu()
                        - out_clean[j, p["donor_gold_id"]].cpu()
                    )
                    delta_recipient_logit_total += float(
                        out_patch[j, p["recipient_gold_id"]].cpu()
                        - out_clean[j, p["recipient_gold_id"]].cpu()
                    )
        n = len(pairs)
        out = {
            "n_pairs": n,
            "flip_to_donor_rate": n_flip_to_donor / n,
            "stay_recipient_rate": n_stay_recipient / n,
            "other_token_rate": n_other / n,
            "mean_delta_donor_logit": delta_donor_logit_total / n,
            "mean_delta_recipient_logit": delta_recipient_logit_total / n,
        }
        print(f"  {name:14s} flip→donor={out['flip_to_donor_rate']:.3f}  "
              f"stay→recipient={out['stay_recipient_rate']:.3f}  "
              f"Δ donor_logit={out['mean_delta_donor_logit']:+.2f}  "
              f"Δ recip_logit={out['mean_delta_recipient_logit']:+.2f}")
        return out

    return {
        "subspace": patch_level("subspace_Bu", B_u_t),
        "geometry": patch_level("geometry_K4", Q_torus_t),
    }


# ─── Method 3 — Steering ────────────────────────────────────────────────────

def method_3_steering(model, tok, cell: Dict) -> Dict:
    print("\n=== Method 3 — Steering (torus rotation) ===")
    Q_torus = cell["Q_torus"]
    mu_layer = cell["mu_layer"]
    Q_torus_t = torch.tensor(Q_torus, dtype=DTYPE, device=DEVICE)
    mu_t = torch.tensor(mu_layer, dtype=DTYPE, device=DEVICE)

    probs = cell["problems"]
    # 5 starting problems per source digit
    rng = np.random.default_rng(RNG_SEED)
    starters: List[Dict] = []
    for d in range(10):
        cands = probs[probs["answer"] == d].reset_index(drop=True)
        if len(cands) == 0:
            continue
        for k in range(min(N_STEER_PER_DIGIT, len(cands))):
            idx = int(rng.integers(0, len(cands)))
            row = cands.iloc[idx]
            starters.append({
                "source_digit": d,
                "prompt": str(row["prompt"]),
                "gold_first_token_id": int(row["gold_first_token_id"]),
                "answer": int(row["answer"]),
            })
    print(f"  {len(starters)} starter problems across digit classes")

    angles = np.linspace(0.0, 2 * math.pi, N_STEER_ANGLES, endpoint=False)
    # For each angle, record predicted digit distribution
    by_angle: Dict[float, List[int]] = {float(a): [] for a in angles}
    by_angle_from: Dict[Tuple[float, int], List[int]] = {}
    prompts = [s["prompt"] for s in starters]

    def predicted_digit(token_text: str) -> Optional[int]:
        s = token_text.strip()
        if len(s) >= 1 and s[0].isdigit():
            return int(s[0])
        return None

    # Baseline (no intervention) — what digit does the model emit for each
    # starter naturally? Compares against the source_digit to confirm the
    # starter set is well-aligned to the answer.
    baseline_digits: List[int] = []
    with torch.no_grad():
        for i in range(0, len(prompts), BATCH):
            pbatch = prompts[i:i + BATCH]
            enc = tok(pbatch, return_tensors="pt", padding=True,
                      padding_side="left").to(DEVICE)
            out = model(**enc).logits[:, -1, :]
            for tid in out.argmax(-1).cpu().tolist():
                d = predicted_digit(tok.decode([tid]))
                baseline_digits.append(d if d is not None else -1)
    bl_match = sum(1 for s, d in zip(starters, baseline_digits)
                    if d == s["source_digit"])
    print(f"  baseline (no intervention): {bl_match}/{len(starters)} predict source digit")

    with torch.no_grad():
        for ai, theta in enumerate(angles):
            digits_emitted: List[int] = []
            for i in range(0, len(prompts), BATCH):
                pbatch = prompts[i:i + BATCH]
                enc = tok(pbatch, return_tensors="pt", padding=True,
                          padding_side="left").to(DEVICE)
                hh = block_l14(model).register_forward_hook(
                    make_steering_hook(float(theta), Q_torus_t, mu_t)
                )
                out = model(**enc).logits[:, -1, :]
                hh.remove()
                preds = out.argmax(-1).cpu().tolist()
                for tid in preds:
                    text = tok.decode([tid])
                    d = predicted_digit(text)
                    digits_emitted.append(d if d is not None else -1)
            by_angle[float(theta)] = digits_emitted
            for s, d in zip(starters, digits_emitted):
                by_angle_from.setdefault((float(theta), s["source_digit"]),
                                          []).append(d)
            digit_hist = Counter(digits_emitted)
            top = digit_hist.most_common(1)[0]
            print(f"  θ={math.degrees(theta):5.0f}°  most common predicted digit: "
                  f"{top[0]} ({100 * top[1] / len(digits_emitted):.0f}%)")

    # Per-source-digit shift: at θ=0 should match source digit; sweep tracks
    shift_table: Dict[int, List[int]] = {d: [] for d in range(10)}
    for theta in angles:
        for src in range(10):
            vals = by_angle_from.get((float(theta), src), [])
            if not vals:
                continue
            top = Counter(vals).most_common(1)[0][0]
            shift_table[src].append((float(theta), top))

    return {
        "angles_radians": angles.tolist(),
        "predictions_by_angle": {f"{theta:.4f}": by_angle[float(theta)]
                                   for theta in angles},
        "shift_table_per_source_digit": {
            str(d): shift_table[d] for d in shift_table
        },
        "baseline_predictions": baseline_digits,
        "baseline_source_match_rate": float(bl_match) / max(len(starters), 1),
        "n_starters": len(starters),
    }


# ─── Method 4 — Geodesic walk ───────────────────────────────────────────────

def method_4_geodesic_walk(model, tok, cell: Dict) -> Dict:
    """Walk through the 10 per-digit centroids on the torus in order 0→1→…→9.

    Each digit class has a mean (cos θ_d, sin θ_d) position on the K4_Torus.
    If the torus is causally encoding the digit, patching to digit d's
    centroid should make the model emit digit d, regardless of the source
    prompt's true answer. We also do an unintervened baseline + a unit-circle
    walk for completeness.
    """
    print("\n=== Method 4 — Geodesic walk (per-digit centroids) ===")
    Q_torus = cell["Q_torus"]
    mu_layer = cell["mu_layer"]
    z_torus = cell["z_torus"]
    codes = cell["codes"]
    Q_torus_t = torch.tensor(Q_torus, dtype=DTYPE, device=DEVICE)
    mu_t = torch.tensor(mu_layer, dtype=DTYPE, device=DEVICE)

    # Per-digit centroid in z_torus space — the actual digit positions
    digit_centroids = np.zeros((10, 2), dtype=np.float32)
    for d in range(10):
        sel = (codes == d)
        if sel.any():
            digit_centroids[d] = z_torus[sel].mean(axis=0)
    print("  Per-digit centroids on torus (z_torus space):")
    for d in range(10):
        c = digit_centroids[d]
        print(f"    digit {d}: ({c[0]:+.3f}, {c[1]:+.3f})  "
              f"|c|={np.linalg.norm(c):.3f}  "
              f"angle={math.degrees(math.atan2(c[1], c[0])):+.1f}°")

    # Pick a single anchor prompt per source digit to avoid prompt-confound.
    probs = cell["problems"]
    rng = np.random.default_rng(RNG_SEED + 1)
    anchors: List[Dict] = []
    for src in range(10):
        cands = probs[probs["answer"] == src].reset_index(drop=True)
        if len(cands) == 0:
            continue
        idx = int(rng.integers(0, len(cands)))
        row = cands.iloc[idx]
        anchors.append({
            "source_digit": src,
            "prompt": str(row["prompt"]),
            "gold_first_token_id": int(row["gold_first_token_id"]),
        })
    print(f"  {len(anchors)} anchor prompts (one per source digit)")

    def predicted_digit(token_text: str) -> Optional[int]:
        s = token_text.strip()
        if len(s) >= 1 and s[0].isdigit():
            return int(s[0])
        return None

    def run_with_hook(make_hook_fn):
        digits_per_source: List[Tuple[int, int]] = []
        for i in range(0, len(anchors), BATCH):
            chunk = anchors[i:i + BATCH]
            enc = tok([a["prompt"] for a in chunk],
                      return_tensors="pt", padding=True,
                      padding_side="left").to(DEVICE)
            if make_hook_fn is not None:
                hh = block_l14(model).register_forward_hook(make_hook_fn())
            else:
                hh = None
            with torch.no_grad():
                out = model(**enc).logits[:, -1, :]
            if hh is not None:
                hh.remove()
            preds = out.argmax(-1).cpu().tolist()
            for src_dict, tid in zip(chunk, preds):
                text = tok.decode([tid])
                d = predicted_digit(text)
                digits_per_source.append(
                    (src_dict["source_digit"], d if d is not None else -1)
                )
        return digits_per_source

    # (4a) Baseline — no intervention. Per anchor, the model should emit its
    # gold digit.
    print("\n  4a. Baseline (no intervention):")
    baseline = run_with_hook(None)
    bl_correct = sum(1 for src, pred in baseline if pred == src)
    print(f"     anchors predicted = source digit: {bl_correct}/{len(baseline)}")

    # (4b) Per-digit-centroid walk: for each target digit d, replace the torus
    # position with digit d's centroid. For a fully causal torus, the model
    # should now emit digit d regardless of the source prompt.
    print("\n  4b. Per-digit-centroid walk (target → expected_digit):")
    centroid_walk: List[Dict] = []
    for target_d in range(10):
        target_z_np = digit_centroids[target_d]
        target_z = torch.tensor(target_z_np, dtype=torch.float32)
        per_src = run_with_hook(
            lambda tz=target_z: make_geodesic_hook(tz, Q_torus_t, mu_t, 1.0)
        )
        target_emitted = sum(1 for _, pred in per_src if pred == target_d)
        source_persistent = sum(1 for src, pred in per_src if pred == src)
        digit_dist = Counter(pred for _, pred in per_src)
        centroid_walk.append({
            "target_digit": target_d,
            "target_centroid": target_z_np.tolist(),
            "n_emit_target": target_emitted,
            "n_emit_source": source_persistent,
            "n_anchors": len(per_src),
            "per_source": [{"source": s, "predicted": p} for s, p in per_src],
            "digit_histogram": {str(k): int(v) for k, v in digit_dist.items()},
        })
        most_common = digit_dist.most_common(1)[0]
        print(f"     target={target_d}: emit_target={target_emitted}/{len(per_src)} "
              f"emit_source={source_persistent}/{len(per_src)}  "
              f"mode={most_common[0]} ({most_common[1]}/{len(per_src)})")

    # (4c) Unit-circle walk for completeness — at radius equal to empirical mean.
    radius = float(np.linalg.norm(z_torus, axis=1).mean())
    print(f"\n  4c. Unit-circle walk at radius={radius:.3f}:")
    angles = np.linspace(0.0, 2 * math.pi, N_GEODESIC_ANGLES, endpoint=False)
    unit_walk: List[Dict] = []
    for theta in angles:
        target_z = torch.tensor([math.cos(theta), math.sin(theta)],
                                 dtype=torch.float32)
        per_src = run_with_hook(
            lambda tz=target_z: make_geodesic_hook(tz, Q_torus_t, mu_t, radius)
        )
        dist = Counter(pred for _, pred in per_src)
        top_digit, top_count = dist.most_common(1)[0]
        unit_walk.append({
            "angle_deg": float(math.degrees(theta)),
            "mode_digit": int(top_digit),
            "mode_count": int(top_count),
            "n_anchors": len(per_src),
            "digit_histogram": {str(k): int(v) for k, v in dist.items()},
        })

    # Aggregate causal signal: how many target-digit predictions hit
    target_hit_rate = sum(w["n_emit_target"] for w in centroid_walk) / \
                       sum(w["n_anchors"] for w in centroid_walk)
    source_persist_rate = sum(w["n_emit_source"] for w in centroid_walk) / \
                          sum(w["n_anchors"] for w in centroid_walk)
    print(f"\n  → target-digit hit rate across all (source, target) pairs: "
          f"{target_hit_rate:.3f}")
    print(f"  → source-digit persistence rate: {source_persist_rate:.3f}")

    return {
        "torus_radius": radius,
        "digit_centroids": digit_centroids.tolist(),
        "baseline": {
            "anchors": [{"source": s, "predicted": p} for s, p in baseline],
            "n_correct": int(bl_correct),
            "n_total": len(baseline),
        },
        "centroid_walk": centroid_walk,
        "unit_circle_walk": unit_walk,
        "summary": {
            "target_hit_rate": target_hit_rate,
            "source_persist_rate": source_persist_rate,
        },
        "n_anchors": len(anchors),
    }


# ─── Driver ─────────────────────────────────────────────────────────────────

def main():
    global MODEL_NAME, TASK, MODE, LAYER, CONCEPT, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--mode", default=DEFAULT_MODE)
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--concept", default=DEFAULT_CONCEPT)
    args = ap.parse_args()
    MODEL_NAME, TASK, MODE, LAYER, CONCEPT = (
        args.model, args.task, args.mode, args.layer, args.concept
    )
    OUT_DIR = OUT_ROOT / f"{MODEL_NAME}__{TASK}__mode_{MODE}__L{LAYER:02d}__{CONCEPT}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    print("=" * 70)
    print(f"Causal validation: {MODEL_NAME}/{TASK}/{MODE}/L{LAYER}/{CONCEPT}")
    print("=" * 70)
    cell = setup_cell()
    model, tok = setup_gptj()

    results: Dict = {
        "cell": {
            "model": MODEL_NAME, "task": TASK, "mode": MODE,
            "layer": LAYER, "concept": CONCEPT,
            "N": int(cell["X"].shape[0]),
            "k_u": cell["k_u"],
            "K4_log_lik": cell["K4_log_lik"],
        },
        "config": {
            "batch": BATCH, "rng_seed": RNG_SEED,
            "n_ablation_problems": N_ABLATION_PROBLEMS,
            "n_patch_pairs": N_PATCH_PAIRS,
            "n_steer_per_digit": N_STEER_PER_DIGIT,
            "n_steer_angles": N_STEER_ANGLES,
            "n_geodesic_prompts": N_GEODESIC_PROMPTS,
            "n_geodesic_angles": N_GEODESIC_ANGLES,
        },
    }

    t = time.time()
    results["method_1_ablation"] = method_1_ablation(model, tok, cell)
    results["method_1_ablation"]["elapsed_seconds"] = time.time() - t
    t = time.time()
    results["method_2_patching"] = method_2_patching(model, tok, cell)
    results["method_2_patching"]["elapsed_seconds"] = time.time() - t
    t = time.time()
    results["method_3_steering"] = method_3_steering(model, tok, cell)
    results["method_3_steering"]["elapsed_seconds"] = time.time() - t
    t = time.time()
    results["method_4_geodesic_walk"] = method_4_geodesic_walk(model, tok, cell)
    results["method_4_geodesic_walk"]["elapsed_seconds"] = time.time() - t

    results["total_elapsed_seconds"] = time.time() - t_total

    out_path = OUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")
    print(f"Total wall: {results['total_elapsed_seconds']:.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
