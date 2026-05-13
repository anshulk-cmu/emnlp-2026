"""Step 10 / Stage 2a — Centroid Fourier helix fit (discover-then-fit).

For each (model, task, mode, layer, concept, variant) cell:

  1. Project activations onto the cell's Stage 1 subspace (LDA-A or CCSVD).
  2. Compute per-value centroids; precondition gate (K_natural, K < 5, sparse_grid).
  3. DC-remove centroids; compute full periodogram per coord via cupy.fft.rfft.
  4. Discover the period: top-2 coords' argmax frequencies; concordance test; vote.
  5. Compute fcr_two_axis and fcr_helix at the discovered period.
  6. Whittle max-over-frequencies null (1000 label-permutation shuffles, batched on GPU).
  7. p-values per coord (max-over-freq null) and for FCRs (max-over-period null).
  8. Hierarchical verdict: helix / circle / none / sparse_value_grid / null_unstable / period_inconsistent / n/a.
  9. Write per-cell artefacts (atomic, resume-by-metadata).

Run on full data — no subsampling. See plan §2 standing rules.

CLI:
  python stage2a_fourier_helix.py --config /home/anshulk/emnlp2026/config.yaml \
      --model gpt-j-6b --task multiplication --mode off --layer 14 --variant lda_a
  python stage2a_fourier_helix.py --config ... --model ... --task ... --mode ... --all-layers --variant lda_a
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from logging.handlers import WatchedFileHandler
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    import cupy as cp
    _HAS_CUPY = cp.cuda.is_available()
except Exception:
    _HAS_CUPY = False
    cp = None


# ──────────────────────────────────────────────────────────────────────────────
# Constants and verdict labels
# ──────────────────────────────────────────────────────────────────────────────

AMBIENT_D = 4096

VERDICT_HELIX = "helix"
VERDICT_CIRCLE = "circle"
VERDICT_NONE = "none"
VERDICT_LOW_K = "low_K"                   # K_present < MIN_K_FOR_FFT — Fourier not run
VERDICT_NULL_UNSTABLE = "null_unstable"
VERDICT_PERIOD_INCONSISTENT = "period_inconsistent"

# Concepts with K_present below this floor get no Fourier analysis (verdict = low_K).
# Reasoning: K=2 has 1 bin (k=1 = Nyquist, trivial); K=3 has 1 useful bin; K=4 has 2 bins.
# K=4 is the minimum at which period discovery (choose between P=K and P=K/2) is meaningful.
MIN_K_FOR_FFT = 4

ELIGIBLE_VERDICTS = {VERDICT_HELIX, VERDICT_CIRCLE, VERDICT_NONE}


# ──────────────────────────────────────────────────────────────────────────────
# Seeding (sha256 of cell identifier — deterministic across reruns)
# ──────────────────────────────────────────────────────────────────────────────

def stage2a_seed(model_key: str, task: str, mode: str, layer: int, variant: str, concept: str) -> tuple[int, str]:
    """Returns (int_seed, seed_input_str). Both logged in metadata.json."""
    s = f"stage2a|{model_key}|{task}|{mode}|{layer:02d}|{variant}|{concept}"
    h = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**63 - 1), s


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# Atomic I/O (mirrors residual_hunting.py)
# ──────────────────────────────────────────────────────────────────────────────

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


# ══════════════════════════════════════════════════════════════════════════════
# PURE ALGORITHM FUNCTIONS — importable by check_stage2a_toys.py
# ══════════════════════════════════════════════════════════════════════════════

def compute_centroids(Z: np.ndarray, label_codes: np.ndarray, K: int) -> np.ndarray:
    """Group Z by integer label codes 0..K-1, return (K, r) centroid matrix.

    Assumes all K groups have ≥ 1 sample (precondition gate enforced upstream).
    """
    r = Z.shape[1]
    M = np.zeros((K, r), dtype=np.float64)
    counts = np.bincount(label_codes, minlength=K)
    if (counts == 0).any():
        empty = np.where(counts == 0)[0].tolist()
        raise ValueError(f"compute_centroids: empty group(s) at codes {empty}")
    np.add.at(M, label_codes, Z.astype(np.float64))
    M /= counts[:, None]
    return M


def dc_remove(M: np.ndarray) -> np.ndarray:
    """Subtract value-axis mean. Removes DC (k=0 Fourier bin). Returns float64.

    Does NOT remove linear trend — that's captured separately by compute_linear_power.
    """
    return M - M.mean(axis=0, keepdims=True)


def periodogram_per_coord(M_centred: np.ndarray) -> np.ndarray:
    """Compute Fourier power per coord at all integer-bin frequencies.

    Returns (K//2 + 1, r) float64 power spectrum. Index 0 is DC (skip downstream).
    Period at frequency-index k is K/k for k ≥ 1.
    """
    K = M_centred.shape[0]
    F = np.fft.rfft(M_centred, axis=0)
    P_spec = (np.abs(F) ** 2) / K
    return P_spec.real.astype(np.float64)


def compute_linear_power(M_centred: np.ndarray, K: int) -> np.ndarray:
    """DOF-rescaled linear power per coord (raw, no plane-residual projection).

    Raw L_c = (Σ_v M_centred[v, c] · v_centred)² / Σ_v v_centred²
    Rescaled by K / (2 · Σ_v v_centred²) to equalise DOF with one Fourier bin.

    NOTE: this primitive is leaky for helix detection because pure circles
    `(cos 2πv/K, sin 2πv/K)` have non-zero discrete projection onto v_centred
    (Σ_{v=0..K-1} v·cos(2πv/K) = −K/2 for K > 1). Use `compute_linear_power_off_plane`
    for the helix-vs-circle test — it projects out the data plane first so that a
    pure 2D circle yields ~0 linear power.
    """
    v = np.arange(K, dtype=np.float64)
    v_c = v - v.mean()
    v_norm_sq = float((v_c ** 2).sum())
    if v_norm_sq < 1e-30:
        return np.zeros(M_centred.shape[1], dtype=np.float64)
    numer = (M_centred * v_c[:, None]).sum(axis=0) ** 2
    L_raw = numer / v_norm_sq
    rescale = K / (2.0 * v_norm_sq)
    return L_raw * rescale


def compute_linear_power_off_plane(M_centred: np.ndarray, K: int) -> np.ndarray:
    """DOF-rescaled linear power on the orthogonal complement of the rank-2 data plane.

    Method: SVD of M_centred, subtract the rank-2 approximation (top-2 right
    singular components), then compute per-coord linear power on the residual.

    Why: for a pure 2D circle embedded in d-D, the rank-2 approximation captures
    the entire data → residual ≈ 0 → linear power ≈ 0. For a helix (3D structure
    in d-D), the rank-2 approximation captures the circle plane → residual carries
    the linear-pitch drift → linear power detects it.

    Returns: per-coord linear power vector on the residual.
    """
    if M_centred.shape[1] <= 2:
        # Not enough room to have an orthogonal complement
        return np.zeros(M_centred.shape[1], dtype=np.float64)
    U, S, Vt = np.linalg.svd(M_centred.astype(np.float64), full_matrices=False)
    rank_to_keep = min(2, S.shape[0])
    M_rank2 = U[:, :rank_to_keep] @ np.diag(S[:rank_to_keep]) @ Vt[:rank_to_keep]
    M_orth = M_centred - M_rank2
    return compute_linear_power(M_orth, K)


def discover_period_for_cell(P_spec: np.ndarray, concordance_bin: int, vote_margin: float) -> dict:
    """Find c_a, c_b (top-2 coords by max-freq power) and the cell's discovered period.

    Concordance: if c_a and c_b's argmax periods agree within ±concordance_bin, use that.
    Vote: otherwise, sum periodogram power across ALL r basis coordinates per frequency;
        winner must beat runner-up by ≥ vote_margin or cell flagged period_inconsistent.

    Returns dict with c_a, c_b, k_a, k_b, k_star, period_concordant, period_inconsistent.
    """
    K_freq, r = P_spec.shape
    if r < 2:
        return {"period_inconsistent": True, "c_a": -1, "c_b": -1, "k_a": -1, "k_b": -1,
                "k_star": -1, "period_concordant": False, "reason": "r < 2"}

    # Skip DC (k=0)
    max_per_coord = P_spec[1:].max(axis=0)
    argmax_per_coord = P_spec[1:].argmax(axis=0) + 1  # +1 to compensate for skipped DC

    # Top 2 coords by max-freq power
    top2 = np.argpartition(max_per_coord, -2)[-2:]
    if max_per_coord[top2[0]] < max_per_coord[top2[1]]:
        top2 = top2[::-1]
    c_a, c_b = int(top2[0]), int(top2[1])
    k_a = int(argmax_per_coord[c_a])
    k_b = int(argmax_per_coord[c_b])

    if abs(k_a - k_b) <= concordance_bin:
        return {"c_a": c_a, "c_b": c_b, "k_a": k_a, "k_b": k_b, "k_star": k_a,
                "period_concordant": True, "period_inconsistent": False}

    # Vote across all r coords: sum periodogram power at each frequency
    period_totals = P_spec[1:].sum(axis=1)  # (K_freq - 1,)
    order = np.argsort(period_totals)[::-1]
    winner_k = int(order[0]) + 1
    runner_k = int(order[1]) + 1 if len(order) > 1 else winner_k
    if period_totals[winner_k - 1] >= vote_margin * max(period_totals[runner_k - 1], 1e-30):
        return {"c_a": c_a, "c_b": c_b, "k_a": k_a, "k_b": k_b, "k_star": winner_k,
                "period_concordant": False, "period_inconsistent": False}

    return {"c_a": c_a, "c_b": c_b, "k_a": k_a, "k_b": k_b, "k_star": -1,
            "period_concordant": False, "period_inconsistent": True,
            "reason": f"vote winner_k={winner_k} ({period_totals[winner_k-1]:.4g}) "
                      f"vs runner_k={runner_k} ({period_totals[runner_k-1]:.4g}) margin < {vote_margin}"}


def pick_linear_coord(L_rescaled: np.ndarray, c_a: int, c_b: int) -> int:
    """Argmax linear power, EXCLUDING {c_a, c_b}. Prevents double-counting helix pitch
    with periodic coords. If r ≤ 2 (degenerate), return c_a."""
    r = L_rescaled.shape[0]
    if r <= 2:
        return c_a
    order = np.argsort(L_rescaled)[::-1]
    for c in order:
        if int(c) not in {c_a, c_b}:
            return int(c)
    return c_a


def compute_fcr_metrics(P_spec: np.ndarray, k_star: int, c_a: int, c_b: int,
                         L_rescaled: np.ndarray, c_L: int) -> dict:
    """Compute fcr_two_axis and fcr_helix at the algorithm-selected (k_star, c_a, c_b, c_L)."""
    if k_star < 1:
        return {"fcr_two_axis": 0.0, "fcr_helix": 0.0}
    total_fourier = float(P_spec[1:].sum())
    p_a = float(P_spec[k_star, c_a])
    p_b = float(P_spec[k_star, c_b])
    L_at_cL = float(L_rescaled[c_L])
    fcr_two_axis = (p_a + p_b) / max(total_fourier, 1e-30)
    fcr_helix = (p_a + p_b + L_at_cL) / max(total_fourier + L_at_cL, 1e-30)
    return {"fcr_two_axis": fcr_two_axis, "fcr_helix": fcr_helix}


def compute_data_plane_rank_ratio(M_centred: np.ndarray) -> float:
    """SVD second-singular-value-squared / first-singular-value-squared.

    Discriminates 1D structures (line) from genuine 2D structures (circle, helix):
      - 1D line in d-D: S[0] >> S[1] → ratio ≈ 0
      - 2D circle: S[0] ≈ S[1] (two equal-energy orthogonal modes) → ratio ≈ 1
      - 3D helix: S[0] ≈ S[1] (circle modes dominant) → ratio ≈ 1; S[2] from pitch

    A circle/helix verdict requires this ratio above a threshold (default 0.3).
    """
    if M_centred.shape[0] < 2 or M_centred.shape[1] < 2:
        return 0.0
    S = np.linalg.svd(M_centred.astype(np.float64), compute_uv=False)
    if S.shape[0] < 2 or S[0] < 1e-30:
        return 0.0
    return float((S[1] ** 2) / (S[0] ** 2))


def run_observed_analysis(M_centred: np.ndarray, K: int,
                          concordance_bin: int, vote_margin: float) -> dict:
    """Full observed analysis: periodogram, period discovery, off-plane linear-pitch, FCRs."""
    P_spec = periodogram_per_coord(M_centred)
    # Linear power for HELIX TEST uses the off-plane (rank-2 residual) version so a
    # pure 2D circle yields ~0 linear power even when sinusoidal leakage exists in
    # individual basis coordinates.
    L_rescaled = compute_linear_power_off_plane(M_centred, K)
    # Data-plane rank ratio — discriminates 1D (line) from 2D+ (circle, helix).
    plane_rank_ratio = compute_data_plane_rank_ratio(M_centred)
    disc = discover_period_for_cell(P_spec, concordance_bin, vote_margin)
    if disc.get("period_inconsistent", False):
        return {**disc, "P_spec": P_spec, "L_rescaled": L_rescaled,
                "plane_rank_ratio": plane_rank_ratio,
                "c_L": -1, "fcr_two_axis": 0.0, "fcr_helix": 0.0,
                "discovered_period": -1.0}
    c_L = pick_linear_coord(L_rescaled, disc["c_a"], disc["c_b"])
    fcr = compute_fcr_metrics(P_spec, disc["k_star"], disc["c_a"], disc["c_b"], L_rescaled, c_L)
    discovered_period = float(K) / disc["k_star"] if disc["k_star"] >= 1 else -1.0
    return {**disc, **fcr, "P_spec": P_spec, "L_rescaled": L_rescaled,
            "plane_rank_ratio": plane_rank_ratio,
            "c_L": c_L, "discovered_period": discovered_period}


def _whittle_null_chunk_gpu(Z_gpu, label_codes_gpu, counts_gpu, perms_gpu,
                             K: int, r: int, v_c_gpu, v_norm_sq: float, linear_rescale: float):
    """One GPU chunk: returns (max_per_coord, fcr_two_axis_max, fcr_helix_max, linear_per_coord).

    perms_gpu shape (cs, N).
    """
    cs = perms_gpu.shape[0]
    shuffled_codes = label_codes_gpu[perms_gpu]  # (cs, N)
    # Scatter-mean: M_perm[t, v, c] = mean over i where shuffled_codes[t, i] == v
    M_perm = cp.zeros((cs, K, r), dtype=cp.float64)
    for t in range(cs):
        cp.add.at(M_perm[t], shuffled_codes[t], Z_gpu)
    M_perm /= counts_gpu[None, :, None]
    M_centred = M_perm - M_perm.mean(axis=1, keepdims=True)

    # Periodogram (cs, K_freq, r)
    F = cp.fft.rfft(M_centred, axis=1)
    P_spec = ((cp.abs(F) ** 2) / K).real

    # Max-over-freq per coord (skip DC)
    P_skip_dc = P_spec[:, 1:, :]  # (cs, K_freq-1, r)
    max_per_coord = P_skip_dc.max(axis=1)  # (cs, r)

    # Linear power per coord on the rank-2 SVD residual (off-plane).
    # Matches the observed-side off-plane computation so the null is calibrated for
    # the same statistic. SVD is batched on GPU via cp.linalg.svd.
    # M_centred shape: (cs, K, r); we want per-cs rank-2 residual.
    U_g, S_g, Vt_g = cp.linalg.svd(M_centred, full_matrices=False)
    # Keep top-2 singular components per cs
    rank = min(2, S_g.shape[1])
    S_diag = cp.zeros((cs, S_g.shape[1], S_g.shape[1]), dtype=cp.float64)
    idx = cp.arange(rank)
    S_diag[:, idx, idx] = S_g[:, :rank]
    M_rank2 = U_g[:, :, :rank] @ S_diag[:, :rank, :rank] @ Vt_g[:, :rank, :]
    M_orth_g = M_centred - M_rank2
    L_numer = (M_orth_g * v_c_gpu[None, :, None]).sum(axis=1) ** 2
    L_per_coord = (L_numer / v_norm_sq) * linear_rescale

    # FCR null = max over periods. At each period k:
    #   - top-2 coords by power at k → fcr_two_axis_at_k
    #   - + linear power (max over coords not in top-2) → fcr_helix_at_k
    total_power = P_skip_dc.sum(axis=(1, 2))  # (cs,)
    # top-2 power across coords per freq
    sorted_along_c = cp.sort(P_skip_dc, axis=2)
    top2_per_freq = sorted_along_c[:, :, -2:].sum(axis=2)  # (cs, K_freq-1)
    fcr_two_axis_per_freq = top2_per_freq / cp.maximum(total_power[:, None], 1e-30)
    # For helix null, take max-over-coords linear (approximation; observed uses the specific c_L
    # that's not in top-2, so max-over-coords linear is at least as large → conservative).
    L_max = L_per_coord.max(axis=1)
    fcr_helix_per_freq = ((top2_per_freq + L_max[:, None])
                          / cp.maximum(total_power[:, None] + L_max[:, None], 1e-30))
    return (cp.asnumpy(max_per_coord), cp.asnumpy(fcr_two_axis_per_freq.max(axis=1)),
            cp.asnumpy(fcr_helix_per_freq.max(axis=1)), cp.asnumpy(L_per_coord))


def run_whittle_null(
    Z: np.ndarray, label_codes: np.ndarray, K: int, r: int,
    n_perms: int, seed: int, use_gpu: bool = False,
    chunk: int = 200, logger: logging.Logger | None = None,
) -> dict:
    """1000-shuffle Whittle max-over-frequencies null.

    Position-permutation preserves group sizes exactly, so K_shuffled == K always.
    The redraw machinery in the plan is a safety net; under correct implementation it
    never triggers. We still record `redraw_rate = 0` for honesty.

    Returns dict with null_max_per_coord, null_fcr_two_axis, null_fcr_helix,
    null_linear_max, redraw_rate, null_unstable.
    """
    N = Z.shape[0]
    rng = np.random.default_rng(seed)
    null_max_per_coord = np.zeros((n_perms, r), dtype=np.float64)
    null_fcr_two_axis = np.zeros(n_perms, dtype=np.float64)
    null_fcr_helix = np.zeros(n_perms, dtype=np.float64)
    null_linear = np.zeros((n_perms, r), dtype=np.float64)
    Z_64 = Z.astype(np.float64)

    if use_gpu and _HAS_CUPY:
        Z_gpu = cp.asarray(Z_64)
        label_codes_gpu = cp.asarray(label_codes.astype(np.int32))
        counts_gpu = cp.bincount(label_codes_gpu, minlength=K).astype(cp.float64)
        v = cp.arange(K, dtype=cp.float64)
        v_c_gpu = v - v.mean()
        v_norm_sq = float((v_c_gpu ** 2).sum())
        linear_rescale = K / (2.0 * v_norm_sq)

        for chunk_start in range(0, n_perms, chunk):
            chunk_end = min(chunk_start + chunk, n_perms)
            cs = chunk_end - chunk_start
            perms = np.stack([rng.permutation(N) for _ in range(cs)]).astype(np.int64)
            perms_gpu = cp.asarray(perms)
            mpc, ft, fh, lp = _whittle_null_chunk_gpu(
                Z_gpu, label_codes_gpu, counts_gpu, perms_gpu, K, r, v_c_gpu, v_norm_sq, linear_rescale,
            )
            null_max_per_coord[chunk_start:chunk_end] = mpc
            null_fcr_two_axis[chunk_start:chunk_end] = ft
            null_fcr_helix[chunk_start:chunk_end] = fh
            null_linear[chunk_start:chunk_end] = lp
        del Z_gpu, label_codes_gpu, counts_gpu
        cp.get_default_memory_pool().free_all_blocks()
    else:
        # CPU fallback — same logic, no batching
        v = np.arange(K, dtype=np.float64)
        v_c = v - v.mean()
        v_norm_sq = float((v_c ** 2).sum())
        linear_rescale = K / (2.0 * v_norm_sq)
        for t in range(n_perms):
            perm = rng.permutation(N)
            shuffled_codes = label_codes[perm]
            M = compute_centroids(Z_64, shuffled_codes, K)
            M_c = dc_remove(M)
            P = periodogram_per_coord(M_c)
            P_skip = P[1:]
            null_max_per_coord[t] = P_skip.max(axis=0)
            # Linear power on the off-plane SVD residual — matches observed-side stat
            L = compute_linear_power_off_plane(M_c, K)
            null_linear[t] = L
            total = float(P_skip.sum())
            sorted_along_c = np.sort(P_skip, axis=1)
            top2 = sorted_along_c[:, -2:].sum(axis=1)  # (K_freq-1,)
            null_fcr_two_axis[t] = float((top2 / max(total, 1e-30)).max())
            Lmax = float(L.max())
            null_fcr_helix[t] = float(((top2 + Lmax) / max(total + Lmax, 1e-30)).max())

    return {
        "null_max_per_coord": null_max_per_coord,
        "null_fcr_two_axis": null_fcr_two_axis,
        "null_fcr_helix": null_fcr_helix,
        "null_linear_max": null_linear,
        "redraw_rate": 0.0,
        "null_unstable": False,
    }


def compute_p_values_and_verdict(
    observed: dict, null: dict,
    K: int, r: int, K_natural: int,
    fcr_threshold: float, per_coord_alpha: float, linear_alpha: float,
) -> dict:
    """Combine observed + null → p-values + provisional verdict.

    Note: final q-values are computed downstream via global BH-FDR across cells.
    This per-cell verdict uses raw p < 0.05 as a proxy; the aggregator downgrades
    to `none` if q ≥ 0.05.
    """
    if observed.get("period_inconsistent", False):
        return {
            "p_helix": 1.0, "p_two_axis": 1.0,
            "p_coord_a": 1.0, "p_coord_b": 1.0, "p_linear": 1.0,
            "two_axis_significant": False, "linear_significant": False,
            "geometry_detected": VERDICT_PERIOD_INCONSISTENT,
        }

    n_perms = null["null_fcr_two_axis"].shape[0]
    P_spec = observed["P_spec"]
    L_rescaled = observed["L_rescaled"]
    c_a, c_b, c_L = observed["c_a"], observed["c_b"], observed["c_L"]

    obs_max_a = float(P_spec[1:, c_a].max())
    obs_max_b = float(P_spec[1:, c_b].max())
    obs_L_at_cL = float(L_rescaled[c_L])

    p_coord_a = (np.sum(null["null_max_per_coord"][:, c_a] >= obs_max_a) + 1) / (n_perms + 1)
    p_coord_b = (np.sum(null["null_max_per_coord"][:, c_b] >= obs_max_b) + 1) / (n_perms + 1)
    null_linear_max_over_c = null["null_linear_max"].max(axis=1)
    p_linear = (np.sum(null_linear_max_over_c >= obs_L_at_cL) + 1) / (n_perms + 1)
    p_two_axis = (np.sum(null["null_fcr_two_axis"] >= observed["fcr_two_axis"]) + 1) / (n_perms + 1)
    p_helix = (np.sum(null["null_fcr_helix"] >= observed["fcr_helix"]) + 1) / (n_perms + 1)

    two_axis_sig = (p_coord_a < per_coord_alpha) and (p_coord_b < per_coord_alpha)
    linear_sig = p_linear < linear_alpha
    plane_rank_ratio = float(observed.get("plane_rank_ratio", 0.0))
    # Genuine 2D+ data plane: top-2 SVD singular values must be comparable.
    # Threshold 0.3 distinguishes 1D line (ratio ~ 1e-3 .. 1e-2 depending on noise)
    # from a real 2D circle (ratio ~ 1) — see compute_data_plane_rank_ratio docstring.
    plane_2d_ok = plane_rank_ratio >= 0.3

    # Per-cell provisional verdict: relies on per-coord and linear-pitch significance
    # (max-over-frequencies / max-over-coords Whittle nulls) plus a 2D-plane sanity check.
    # The FCR Whittle p-value is a max-over-3-or-more-period statistic; at small K
    # (e.g. K=7 → 3 candidate frequencies) it's weak. The per-coord tests give the
    # meaningful significance gate per cell. The aggregator applies BH-FDR on q_helix
    # and q_two_axis across all cells to control cross-cell multiple testing.
    geometry = VERDICT_NONE
    if (observed["fcr_helix"] >= fcr_threshold and two_axis_sig and linear_sig
            and plane_2d_ok):
        geometry = VERDICT_HELIX
    elif (observed["fcr_two_axis"] >= fcr_threshold and two_axis_sig and not linear_sig
            and plane_2d_ok):
        geometry = VERDICT_CIRCLE

    return {
        "p_helix": float(p_helix), "p_two_axis": float(p_two_axis),
        "p_coord_a": float(p_coord_a), "p_coord_b": float(p_coord_b),
        "p_linear": float(p_linear),
        "two_axis_significant": bool(two_axis_sig),
        "linear_significant": bool(linear_sig),
        "plane_rank_ratio": plane_rank_ratio,
        "plane_2d_ok": bool(plane_2d_ok),
        "geometry_detected": geometry,
    }


def analyze_cell(
    Z: np.ndarray, label_values: np.ndarray, K_natural: int,
    cfg_stage2a: dict, seed: int,
    use_gpu: bool = False, logger: logging.Logger | None = None,
) -> dict:
    """End-to-end analysis of one cell.

    Z: (N, r) projected activations (already in the subspace).
    label_values: (N,) raw label values (any hashable).
    K_natural: expected count of unique values (from full unmasked DataFrame).

    Per the "find them all, flag confidence" policy: no concept is skipped silently.
    A concept where K_present < K_natural still runs (with `non_uniform_grid_flag = True`).
    A concept where K_present < MIN_K_FOR_FFT (4) gets verdict = low_K (no Fourier run),
    but basic structural stats (N, K, r, N_over_K, N_over_r) are still reported.
    """
    N, r = Z.shape
    min_group_size = int(cfg_stage2a.get("min_group_size", 30))
    min_K = int(cfg_stage2a.get("min_K_for_fft", MIN_K_FOR_FFT))
    fcr_thr = float(cfg_stage2a.get("fcr_threshold", 0.30))
    per_coord_alpha = float(cfg_stage2a.get("two_axis_alpha", 0.01))
    linear_alpha = float(cfg_stage2a.get("linear_alpha", 0.01))
    n_perms = int(cfg_stage2a.get("n_permutations", 1000))
    concordance_bin = int(cfg_stage2a.get("concordance_bin_tolerance", 1))
    vote_margin = float(cfg_stage2a.get("vote_winner_margin", 2.0))
    redraw_rate_max = float(cfg_stage2a.get("redraw_rate_max", 0.10))

    s = pd.Series(label_values)
    counts = s.value_counts()
    keep_values = sorted(counts[counts >= min_group_size].index.tolist())
    dropped_values = sorted(counts[counts < min_group_size].index.tolist())
    K_present = len(keep_values)

    # Confidence flags — always reported, never used to silently skip.
    non_uniform_grid_flag = bool(K_present < K_natural)
    low_K_natural_flag = bool(K_natural < min_K)
    # Build the always-present structural fields
    structural = {
        "K_natural": int(K_natural), "K_present": int(K_present), "K": int(K_present),
        "r": int(r),
        "dropped_values": dropped_values,
        "non_uniform_grid_flag": non_uniform_grid_flag,
        "low_K_natural_flag": low_K_natural_flag,
    }

    # Base zero-result for cells where Fourier won't run
    no_fft_out = {
        **structural,
        "n_samples_used": 0, "discovered_period": -1.0,
        "fcr_two_axis": 0.0, "fcr_helix": 0.0,
        "fcr_two_axis_x_r": 0.0, "fcr_helix_x_r": 0.0,
        "c_a": -1, "c_b": -1, "c_L": -1, "k_a": -1, "k_b": -1, "k_star": -1,
        "period_concordant": False, "period_inconsistent": False,
        "plane_rank_ratio": 0.0, "plane_2d_ok": False,
        "p_helix": 1.0, "p_two_axis": 1.0,
        "p_coord_a": 1.0, "p_coord_b": 1.0, "p_linear": 1.0,
        "two_axis_significant": False, "linear_significant": False,
        "redraw_rate": 0.0, "null_unstable": False,
        "N_over_K": 0.0, "N_over_r": 0.0,
        "P_spec": None, "L_rescaled": None,
        "null_max_per_coord": None, "null_fcr_two_axis": None,
        "null_fcr_helix": None, "null_linear_max": None,
    }

    if K_present < min_K:
        # Too few values to meaningfully Fourier-test. Still report basics.
        return {**no_fft_out, "geometry_detected": VERDICT_LOW_K,
                "reason": (f"K_present={K_present} < MIN_K_FOR_FFT={min_K} "
                            f"(K_natural={K_natural}; non_uniform_grid={non_uniform_grid_flag})")}

    # Run the full analysis on the K_present surviving values. If non_uniform_grid_flag
    # is True, the DFT is treating the surviving K_present values as a uniform grid;
    # the flag warns downstream consumers that the discovered period interpretation
    # depends on which values were dropped.
    value_to_code = {v: i for i, v in enumerate(keep_values)}
    keep_mask = np.array([v in value_to_code for v in label_values])
    Z_keep = Z[keep_mask]
    label_codes = np.array([value_to_code[v] for v in label_values[keep_mask]], dtype=np.int32)
    K = K_present
    N_used = int(Z_keep.shape[0])

    M = compute_centroids(Z_keep, label_codes, K)
    M_centred = dc_remove(M)

    observed = run_observed_analysis(M_centred, K, concordance_bin, vote_margin)
    null = run_whittle_null(
        Z_keep, label_codes, K, r, n_perms=n_perms, seed=seed,
        use_gpu=use_gpu, logger=logger,
    )
    stats = compute_p_values_and_verdict(
        observed, null, K, r, K_natural,
        fcr_threshold=fcr_thr, per_coord_alpha=per_coord_alpha, linear_alpha=linear_alpha,
    )
    if null["redraw_rate"] > redraw_rate_max or null["null_unstable"]:
        stats["geometry_detected"] = VERDICT_NULL_UNSTABLE

    fcr_two_axis_x_r = float(observed["fcr_two_axis"]) * r
    fcr_helix_x_r = float(observed["fcr_helix"]) * r
    N_over_K = float(N_used) / float(max(K, 1))
    N_over_r = float(N_used) / float(max(r, 1))

    return {
        **structural,
        "n_samples_used": N_used,
        **{k: v for k, v in observed.items() if k not in ("P_spec", "L_rescaled")},
        "fcr_two_axis_x_r": fcr_two_axis_x_r, "fcr_helix_x_r": fcr_helix_x_r,
        "N_over_K": N_over_K, "N_over_r": N_over_r,
        **stats,
        "redraw_rate": float(null["redraw_rate"]),
        "null_unstable": bool(null["null_unstable"]),
        "P_spec": observed["P_spec"],
        "L_rescaled": observed["L_rescaled"],
        "null_max_per_coord": null["null_max_per_coord"],
        "null_fcr_two_axis": null["null_fcr_two_axis"],
        "null_fcr_helix": null["null_fcr_helix"],
        "null_linear_max": null["null_linear_max"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# I/O — loaders, runners, CLI
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging(logs_root: Path, model_key: str, task: str, mode: str, variant: str) -> logging.Logger:
    logs_root.mkdir(parents=True, exist_ok=True)
    name = f"stage2a.{model_key}.{task}.{mode}.{variant}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = WatchedFileHandler(logs_root / f"stage2a_{model_key}_{task}_{mode}_{variant}.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


def ccsvd_basis_path(results_root: Path, model: str, task: str, layer: int, concept: str, mode: str) -> Path:
    """Mirrors residual_hunting.ccsvd_basis_path."""
    if mode == "off":
        return results_root / "ccsvd_subspaces" / model / task / f"layer_{layer:02d}" / concept / "basis.npy"
    return results_root / "ccsvd_subspaces" / f"mode_{mode}" / model / task / f"layer_{layer:02d}" / concept / "basis.npy"


def ccsvd_meta_path(results_root: Path, model: str, task: str, layer: int, concept: str, mode: str) -> Path:
    """meta.json for the CCSVD cell — contains layer-mean and other metadata."""
    return ccsvd_basis_path(results_root, model, task, layer, concept, mode).parent / "meta.json"


def lda_a_basis_path(results_root: Path, model: str, task: str, layer: int, concept: str, mode: str) -> Path:
    """Mirrors residual_hunting.lda_a_basis_path."""
    return (results_root / "lda_subspaces" / "subspace_lda" / f"mode_{mode}"
            / model / task / f"layer_{layer:02d}" / concept / "lda_basis_full.npy")


def lda_a_meta_path(results_root: Path, model: str, task: str, layer: int, concept: str, mode: str) -> Path:
    return lda_a_basis_path(results_root, model, task, layer, concept, mode).parent / "meta.json"


def load_basis_matrix(path: Path) -> np.ndarray | None:
    """Load a basis stored as (D, r) or (r, D); return (D, r) float32 or None."""
    if not path.exists():
        return None
    B = np.load(path)
    if B.ndim != 2:
        return None
    if B.shape[0] == AMBIENT_D:
        return B.astype(np.float32, copy=False)
    if B.shape[1] == AMBIENT_D:
        return B.T.astype(np.float32, copy=False)
    return None


def load_mu_layer(meta_path: Path) -> np.ndarray | None:
    """Read μ_layer from a cell's meta.json. Returns (D,) float32 or None."""
    if not meta_path.exists():
        return None
    try:
        m = json.loads(meta_path.read_text())
    except Exception:
        return None
    for key in ("mu_layer", "global_mean", "training_mean", "mu"):
        if key in m:
            arr = np.asarray(m[key], dtype=np.float32)
            if arr.shape == (AMBIENT_D,):
                return arr
    # Fallback: look for sibling .npy artefact named global_mean.npy
    p = meta_path.parent / "global_mean.npy"
    if p.exists():
        arr = np.load(p).astype(np.float32)
        if arr.shape == (AMBIENT_D,):
            return arr
    return None


def load_concept_filter(results_root: Path, model: str, task: str, mode: str, layer: int) -> dict:
    """{concept: {status, n_sig, is_carved_out, ...}} from Step 6 LDA-A summary."""
    summary_path = (results_root / "lda_subspaces" / "subspace_lda" / f"mode_{mode}"
                    / model / f"summary_{model}_mode_{mode}.csv")
    df = pd.read_csv(summary_path)
    df = df[(df["task"] == task) & (df["layer"] == layer)]
    out = {}
    for _, row in df.iterrows():
        out[str(row["concept"])] = {
            "status": str(row.get("status", "")),
            "n_sig": int(row.get("n_sig", 0)) if not pd.isna(row.get("n_sig")) else 0,
            "is_carved_out": bool(row.get("is_carved_out", False)),
        }
    return out


def K_natural_for_concept(problems_df: pd.DataFrame, concept: str) -> int:
    """Count unique values in the FULL problems DataFrame column for this concept.

    Returns 0 if concept not in DataFrame or column is constant.
    """
    if concept not in problems_df.columns:
        return 0
    col = problems_df[concept]
    return int(col.nunique(dropna=True))


def project_to_subspace(X: np.ndarray, B_D_r: np.ndarray, mu_layer: np.ndarray) -> np.ndarray:
    """Z = (X - mu_layer) @ B  where B is (D, r). Returns (N, r) float32."""
    X_c = (X.astype(np.float32) - mu_layer.astype(np.float32, copy=False))
    return X_c @ B_D_r


def load_prior_periods(config_path: Path, stage2a_cfg: dict) -> dict:
    """Load configs/prior_periods.yaml. Returns {concept: [periods]} (empty if missing)."""
    rel = stage2a_cfg.get("prior_period_table_path", "")
    if not rel:
        return {}
    p = config_path.parent / rel
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text()) or {}
    return {k: v.get("predicted_periods", []) for k, v in raw.items() if isinstance(v, dict)}


def period_match(discovered: float, predicted: list[float], bin_tol: float = 1.0) -> bool:
    """True if discovered period is within bin_tol of any predicted period."""
    if not predicted or discovered < 0:
        return not predicted  # vacuously True when no prediction made
    for p in predicted:
        if abs(discovered - float(p)) <= bin_tol:
            return True
    return False


def cell_artifact_dir(results_root: Path, model: str, task: str, mode: str,
                     layer: int, variant: str, concept: str) -> Path:
    return (results_root / "stage2a_fourier_helix" / model / task / f"mode_{mode}"
            / f"layer_{layer:02d}" / f"variant_{variant}" / concept)


def write_cell_artifacts(out_dir: Path, result: dict, meta: dict,
                          store_full_null: bool, logger: logging.Logger) -> None:
    """Atomic-write per-cell artefacts. Big null arrays only saved on helix/circle."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV row
    csv_cols = ["K_natural", "K_present", "K", "r", "n_samples_used",
                "N_over_K", "N_over_r",
                "non_uniform_grid_flag", "low_K_natural_flag",
                "discovered_period", "k_star", "k_a", "k_b",
                "prior_predicted_period", "period_match",
                "period_concordant", "period_inconsistent",
                "c_a", "c_b", "c_L",
                "fcr_two_axis", "fcr_helix",
                "fcr_two_axis_x_r", "fcr_helix_x_r",
                "p_two_axis", "p_helix",
                "p_coord_a", "p_coord_b", "p_linear",
                "two_axis_significant", "linear_significant",
                "plane_rank_ratio", "plane_2d_ok",
                "null_unstable", "redraw_rate",
                "geometry_detected"]
    row = {c: result.get(c) for c in csv_cols}
    pd.DataFrame([row]).to_csv(out_dir / "fcr_results.csv", index=False)

    # Centroids and observed periodogram are useful for inspection regardless of verdict
    if result.get("P_spec") is not None:
        atomic_save(np.asarray(result["P_spec"], dtype=np.float64),
                    out_dir / "fourier_spectrum_observed.npy")
    if result.get("L_rescaled") is not None:
        atomic_save(np.asarray(result["L_rescaled"], dtype=np.float64),
                    out_dir / "linear_power_observed.npy")

    # Null arrays — keep null_max_per_coord always (small); keep big spectra only on detection.
    if result.get("null_max_per_coord") is not None:
        atomic_save(np.asarray(result["null_max_per_coord"], dtype=np.float32),
                    out_dir / "null_max_per_coord.npy")
    if result.get("null_linear_max") is not None:
        atomic_save(np.asarray(result["null_linear_max"], dtype=np.float32),
                    out_dir / "null_linear_max.npy")
    if result.get("null_fcr_two_axis") is not None:
        atomic_save(np.asarray(result["null_fcr_two_axis"], dtype=np.float32),
                    out_dir / "fcr_two_axis_null.npy")
    if result.get("null_fcr_helix") is not None:
        atomic_save(np.asarray(result["null_fcr_helix"], dtype=np.float32),
                    out_dir / "fcr_helix_null.npy")

    atomic_json({**meta, "computation_status": "complete"}, out_dir / "metadata.json")


def run_one_cell(
    cfg: dict, results_root: Path,
    model: str, task: str, mode: str, layer: int, variant: str, concept: str,
    problems_df: pd.DataFrame, correct_mask: np.ndarray,
    X_correct: np.ndarray, mu_layer_off: np.ndarray | None,
    prior_periods: dict, logger: logging.Logger,
) -> dict | None:
    """Run Stage 2a on a single (variant, concept) cell. Returns the summary row or None on skip."""
    out_dir = cell_artifact_dir(results_root, model, task, mode, layer, variant, concept)
    meta_path = out_dir / "metadata.json"
    if meta_path.exists():
        try:
            cached = json.loads(meta_path.read_text())
            if cached.get("computation_status") == "complete":
                logger.info(f"[skip cached] {variant}/{concept}")
                return cached.get("summary_row")
        except Exception:
            pass

    # Load basis matrix per variant (use mode-specific cell — matches plan §"μ_layer source")
    if variant == "ccsvd":
        B_path = ccsvd_basis_path(results_root, model, task, layer, concept, mode)
        meta_p = ccsvd_meta_path(results_root, model, task, layer, concept, mode)
    elif variant == "lda_a":
        B_path = lda_a_basis_path(results_root, model, task, layer, concept, mode)
        meta_p = lda_a_meta_path(results_root, model, task, layer, concept, mode)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    B = load_basis_matrix(B_path)
    if B is None or B.shape[1] == 0:
        logger.info(f"[skip no-basis] {variant}/{concept}: {B_path}")
        return None

    mu_layer = load_mu_layer(meta_p)
    if mu_layer is None:
        mu_layer = mu_layer_off  # fallback (best-effort)
    if mu_layer is None:
        logger.warning(f"[skip no-mu] {variant}/{concept}: no mu_layer for mode {mode}")
        return None

    # Project to subspace
    Z = project_to_subspace(X_correct, B, mu_layer)
    r = Z.shape[1]

    # Labels: full DataFrame for K_natural; masked for the analysis
    K_natural = K_natural_for_concept(problems_df, concept)
    if K_natural < 2:
        logger.info(f"[skip constant] {variant}/{concept}: K_natural={K_natural}")
        return None
    labels_full = problems_df[concept].to_numpy()
    labels_correct = labels_full[correct_mask]

    seed_int, seed_str = stage2a_seed(model, task, mode, layer, variant, concept)
    t0 = time.time()
    result = analyze_cell(
        Z, labels_correct, K_natural,
        cfg_stage2a=cfg.get("stage2a", {}),
        seed=seed_int, use_gpu=_HAS_CUPY, logger=logger,
    )
    runtime = time.time() - t0

    # Prior periods + match
    predicted = prior_periods.get(concept, [])
    matched = period_match(result.get("discovered_period", -1.0), predicted,
                            bin_tol=float(cfg["stage2a"].get("concordance_bin_tolerance", 1)))
    result["prior_predicted_period"] = predicted if predicted else None
    result["period_match"] = bool(matched)

    summary_row = {
        "model": model, "task": task, "mode": mode, "layer": int(layer),
        "variant": variant, "concept": concept,
        **{k: result.get(k) for k in [
            "geometry_detected", "discovered_period", "k_star",
            "fcr_two_axis", "fcr_helix",
            "p_two_axis", "p_helix", "p_coord_a", "p_coord_b", "p_linear",
            "two_axis_significant", "linear_significant",
            "plane_rank_ratio", "plane_2d_ok",
            "period_concordant", "period_inconsistent",
            "K_natural", "K_present", "K", "r", "n_samples_used",
            "N_over_K", "N_over_r",
            "non_uniform_grid_flag", "low_K_natural_flag",
            "prior_predicted_period", "period_match",
            "fcr_two_axis_x_r", "fcr_helix_x_r",
            "null_unstable", "redraw_rate",
        ]},
        "runtime_seconds": round(runtime, 2),
    }

    meta = {
        "model": model, "task": task, "mode": mode, "layer": int(layer),
        "variant": variant, "concept": concept,
        "random_seed_input": seed_str, "random_seed": int(seed_int),
        "n_permutations": int(cfg["stage2a"].get("n_permutations", 1000)),
        "gpu_used": _HAS_CUPY,
        "runtime_seconds": round(runtime, 2),
        "lib_versions": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "cupy": cp.__version__ if _HAS_CUPY else None,
        },
        "summary_row": summary_row,
        "K_natural": int(K_natural),
        "K_present": int(result.get("K_present", 0)),
        "dropped_values": result.get("dropped_values", []),
        "B_sha256": sha256_of(B_path) if B_path.exists() else None,
        "B_path": str(B_path),
        "mu_layer_source": str(meta_p),
    }
    store_full_null = bool(cfg["stage2a"].get("store_full_null_only_for_detections", True))
    write_cell_artifacts(out_dir, result, meta, store_full_null, logger)

    return summary_row


# ──────────────────────────────────────────────────────────────────────────────
# Layer-level + main
# ──────────────────────────────────────────────────────────────────────────────

def discover_concepts_for_cell(results_root: Path, model: str, task: str, mode: str,
                                layer: int, variant: str, problems_df: pd.DataFrame) -> list[str]:
    """Return list of concept names with available basis files for the given variant.

    For LDA-A: read Step 6 LDA-A summary, take fit_ok rows with n_sig >= 1.
    For CCSVD: scan filesystem for cells with basis.npy.
    """
    if variant == "lda_a":
        try:
            flt = load_concept_filter(results_root, model, task, mode, layer)
        except FileNotFoundError:
            return []
        return [c for c, s in flt.items()
                if s["status"] == "fit_ok" and s["n_sig"] >= 1 and not s["is_carved_out"]]
    elif variant == "ccsvd":
        base_dir = (results_root / "ccsvd_subspaces" / (model if mode == "off"
                                                          else f"mode_{mode}/{model}")
                    / task / f"layer_{layer:02d}")
        if not base_dir.exists():
            base_dir2 = (results_root / "ccsvd_subspaces"
                         / (f"mode_{mode}" if mode != "off" else "") / model / task
                         / f"layer_{layer:02d}")
            base_dir = base_dir2 if base_dir2.exists() else base_dir
        if not base_dir.exists():
            return []
        return sorted([p.name for p in base_dir.iterdir()
                        if p.is_dir() and (p / "basis.npy").exists()])
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=["addition", "multiplication"])
    ap.add_argument("--mode", required=True, choices=["off", "answer", "norm"])
    ap.add_argument("--variant", required=True, choices=["lda_a", "ccsvd"])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--layer", type=int)
    g.add_argument("--all-layers", action="store_true")
    ap.add_argument("--single-concept", type=str, default=None,
                    help="Process only this concept (smoke-test convenience).")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    paths = cfg["paths"]
    logs_root = Path(paths["logs_root"])
    results_root = Path(paths["results_root"])
    data_root = Path(paths["data_root"])
    model_cfg = next(m for m in cfg["models"] if m["key"] == args.model)
    layers = [args.layer] if args.layer is not None else model_cfg["layers"]

    logger = setup_logging(logs_root, args.model, args.task, args.mode, args.variant)
    logger.info(f"=== Stage 2a: model={args.model} task={args.task} mode={args.mode} "
                f"variant={args.variant} layers={layers} ===")
    logger.info(f"cupy: {'AVAILABLE' if _HAS_CUPY else 'NOT available — CPU only'}")

    problems_df = pd.read_csv(data_root / "data" / "raw" / f"{args.task}_problems.csv")
    answers_df = pd.read_csv(data_root / "answers" / args.model / f"{args.task}_answers.csv")
    if len(answers_df) != len(problems_df):
        raise RuntimeError("answers/problems row count mismatch")
    correct_mask = answers_df["correct"].to_numpy().astype(bool)
    logger.info(f"N_correct={int(correct_mask.sum())} / N_total={len(problems_df)}")

    prior_periods = load_prior_periods(Path(args.config).absolute(), cfg.get("stage2a", {}))

    all_rows = []
    for layer in layers:
        if args.mode == "off":
            act_path = data_root / "activations" / args.model / f"{args.task}_layer_{layer:02d}.npy"
        else:
            act_path = results_root / "residualized" / args.model / f"{args.task}_layer_{layer:02d}_mode_{args.mode}.npy"
        if not act_path.exists():
            logger.warning(f"missing activations: {act_path} — skipping layer {layer}")
            continue
        X_full = np.load(act_path)
        X_correct = np.ascontiguousarray(X_full[correct_mask].astype(np.float32))
        del X_full

        # μ_layer fallback (raw-activation mean) — used if a per-concept meta.json lacks it
        mu_layer_fallback = X_correct.mean(axis=0).astype(np.float32)

        concepts = discover_concepts_for_cell(results_root, args.model, args.task,
                                                args.mode, layer, args.variant, problems_df)
        if args.single_concept:
            concepts = [c for c in concepts if c == args.single_concept]
        logger.info(f"layer={layer:02d} variant={args.variant}: {len(concepts)} concepts to analyze")

        for concept in concepts:
            try:
                row = run_one_cell(
                    cfg, results_root,
                    args.model, args.task, args.mode, layer, args.variant, concept,
                    problems_df, correct_mask, X_correct, mu_layer_fallback,
                    prior_periods, logger,
                )
                if row is not None:
                    all_rows.append(row)
                    logger.info(f"  {concept:35s}  K={row.get('K')}  geom={row.get('geometry_detected'):<22s} "
                                f"P*={row.get('discovered_period'):.2f}  fcr_helix={row.get('fcr_helix'):.3f}  "
                                f"p_h={row.get('p_helix'):.3g}")
            except Exception as exc:
                logger.exception(f"[FAIL] {args.variant}/{concept}: {exc}")

    if all_rows:
        per_model_dir = results_root / "stage2a_fourier_helix" / args.model
        per_model_dir.mkdir(parents=True, exist_ok=True)
        out_csv = per_model_dir / f"summary_{args.model}_{args.task}_mode_{args.mode}_variant_{args.variant}.csv"
        df_new = pd.DataFrame(all_rows)
        if out_csv.exists():
            df_old = pd.read_csv(out_csv)
            key = ["model", "task", "mode", "layer", "variant", "concept"]
            df_old = df_old[~df_old.set_index(key).index.isin(df_new.set_index(key).index)]
            df_new = pd.concat([df_old, df_new], ignore_index=True)
        df_new = df_new.sort_values(["task", "mode", "layer", "variant", "concept"]).reset_index(drop=True)
        df_new.to_csv(out_csv, index=False)
        logger.info(f"wrote {out_csv} ({len(df_new)} rows)")

    logger.info(f"=== Stage 2a DONE: {args.model}/{args.task}/{args.mode}/{args.variant} ===")


if __name__ == "__main__":
    main()
