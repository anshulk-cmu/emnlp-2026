"""Build the addition and multiplication ground-truth datasets.

Step 2 of the emnlp2026 pipeline. No model runs — pure Python ground-truth
generation with the comprehensive Tier 1-5 concept schema (atomic, algebraic,
structural, relational, tokenization).

Inputs:
  - config.yaml (paths, prompts, operand range)
  - data/results/tokenization_limits/multiplication_per_pair.csv (from Step 1)

Outputs (under data/data/raw/):
  - addition_problems.json / .csv (10,000 problems)
  - multiplication_problems.json / .csv (cross-model single-token intersection)
  - multiplication_intersection_mask.csv (audit trail, 10,000 rows)
  - build_manifest.json (reproducibility metadata)
  - coverage_report.md (per-concept value counts)

Usage:
  python generate_datasets.py --config /home/anshulk/emnlp2026/config.yaml
"""

import argparse
import csv
import hashlib
import json
import logging
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import yaml


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(logs_root: Path):
    logs_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("generate_datasets")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = RotatingFileHandler(logs_root / "generate_datasets.log",
                             maxBytes=10_000_000, backupCount=3)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s",
                            datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ═══════════════════════════════════════════════════════════════════════════════
# TOKENIZATION DATA LOAD (from Step 1's per-pair CSV)
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_KEYS = ["gpt-j-6b", "llama-3.1-8b", "pythia-6.9b"]
MODEL_KEY_TO_PYKEY = {
    "gpt-j-6b": "gpt_j",
    "llama-3.1-8b": "llama",
    "pythia-6.9b": "pythia",
}


def load_per_pair_tokenization(csv_path: Path) -> dict:
    """Read multiplication_per_pair.csv (or addition) and return:

      { (a, b): { model_key: {n_tokens, is_single_token, first_token_id,
                              first_token_text} } }
    """
    out = {}
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            a = int(row["a"]); b = int(row["b"])
            mk = row["model_key"]
            ftid = row["first_token_id"]
            entry = {
                "n_tokens": int(row["n_tokens"]) if row["n_tokens"] else 0,
                "is_single_token": int(row["is_single_token"]),
                "first_token_id": int(ftid) if ftid else -1,
                "first_token_text": row["first_token_text"],
            }
            out.setdefault((a, b), {})[mk] = entry
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# LABEL COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def magnitude_tier(n: int) -> str:
    if n == 0: return "zero"
    if n < 10: return "single_digit"
    if n < 100: return "two_digit"
    if n < 1000: return "three_digit"
    if n < 10000: return "four_digit"
    return "five_plus_digit"


def digits_lsf(n: int, max_digits: int) -> list:
    out = []
    for _ in range(max_digits):
        out.append(n % 10)
        n //= 10
    return out


def shared_operand_labels(a: int, b: int) -> dict:
    """Tier 1 atomic operand labels common to both tasks."""
    a_units, a_tens = a % 10, (a // 10) % 10
    b_units, b_tens = b % 10, (b // 10) % 10
    return {
        "a": a, "b": b,
        "a_units": a_units, "a_tens": a_tens,
        "a_num_digits": 1 if a < 10 else 2,
        "a_digits_lsf": [a_units, a_tens],
        "b_units": b_units, "b_tens": b_tens,
        "b_num_digits": 1 if b < 10 else 2,
        "b_digits_lsf": [b_units, b_tens],
    }


def shared_structural_labels(a: int, b: int, answer: int) -> dict:
    """Tier 3 structural / distributional labels."""
    a_par, b_par, ans_par = a % 2, b % 2, answer % 2
    return {
        "a_parity": a_par, "b_parity": b_par, "ans_parity": ans_par,
        "parity_match": (a_par == b_par),
        "parity_xor": (a_par + b_par) % 2,
        "a_magnitude_tier": magnitude_tier(a),
        "b_magnitude_tier": magnitude_tier(b),
        "ans_magnitude_tier": magnitude_tier(answer),
        "ans_ends_in_zero": (answer % 10 == 0),
        "ans_is_zero": (answer == 0),
        "a_is_zero": (a == 0),
        "b_is_zero": (b == 0),
    }


def shared_relational_labels(a: int, b: int) -> dict:
    """Tier 4 relational labels (operand-pair comparisons)."""
    if a > b:    larger = "a"
    elif a < b:  larger = "b"
    else:        larger = "equal"
    return {
        "a_eq_b": (a == b),
        "max_operand": max(a, b),
        "min_operand": min(a, b),
        "operand_diff": a - b,
        "operand_abs_diff": abs(a - b),
        "larger_operand": larger,
        "both_zero": (a == 0 and b == 0),
        "either_zero": (a == 0 or b == 0),
        "both_one": (a == 1 and b == 1),
        "either_one": (a == 1 or b == 1),
    }


def addition_labels(a: int, b: int) -> dict:
    """Build the full Tier 1-4 label dict for an addition problem."""
    answer = a + b
    a_u, a_t = a % 10, (a // 10) % 10
    b_u, b_t = b % 10, (b // 10) % 10

    # Tier 1 atomic answer
    ans_u  = answer % 10
    ans_t  = (answer // 10) % 10
    ans_h  = (answer // 100) % 10
    if answer == 0:
        n_dig = 1
    else:
        n_dig = 1 + int(np.floor(np.log10(answer)))
    answer_lsf = digits_lsf(answer, max(3, n_dig))[:n_dig]
    answer_msf = list(reversed(answer_lsf))

    # Tier 2 algebraic intermediates (addition)
    col_sum_u = a_u + b_u
    col_sum_t = a_t + b_t
    carry_u   = 1 if col_sum_u >= 10 else 0
    carry_t   = 1 if (col_sum_t + carry_u) >= 10 else 0
    run_sum_u = col_sum_u
    run_sum_t = col_sum_t + carry_u

    rec = {}
    rec.update(shared_operand_labels(a, b))
    rec.update({
        "answer": answer,
        "ans_units": ans_u, "ans_tens": ans_t, "ans_hundreds": ans_h,
        "ans_num_digits": n_dig,
        "answer_digits_lsf": answer_lsf,
        "answer_digits_msf": answer_msf,
    })
    rec.update({
        # Tier 2: addition algebraic intermediates
        "column_sum_units": col_sum_u,
        "column_sum_tens":  col_sum_t,
        "carry_units":      carry_u,
        "carry_tens":       carry_t,
        "running_sum_units": run_sum_u,
        "running_sum_tens":  run_sum_t,
        "column_sums":      [col_sum_u, col_sum_t],
        "carries":          [carry_u, carry_t],
        "running_sums":     [run_sum_u, run_sum_t],
    })
    rec.update(shared_structural_labels(a, b, answer))
    rec.update(shared_relational_labels(a, b))
    return rec


def multiplication_labels(a: int, b: int) -> dict:
    """Build the full Tier 1-4 label dict for a multiplication problem."""
    answer = a * b
    a_u, a_t = a % 10, (a // 10) % 10
    b_u, b_t = b % 10, (b // 10) % 10

    # Tier 1 atomic answer
    ans_u  = answer % 10
    ans_t  = (answer // 10) % 10
    ans_h  = (answer // 100) % 10
    ans_th = (answer // 1000) % 10
    if answer == 0:
        n_dig = 1
    else:
        n_dig = 1 + int(np.floor(np.log10(answer)))
    answer_lsf = digits_lsf(answer, max(4, n_dig))[:n_dig]
    answer_msf = list(reversed(answer_lsf))

    # Tier 2 algebraic intermediates (multiplication, schoolbook 2x2)
    pp_uu = a_u * b_u
    pp_ut = a_u * b_t
    pp_tu = a_t * b_u
    pp_tt = a_t * b_t
    partial_products = {
        "a0_x_b0": pp_uu, "a0_x_b1": pp_ut,
        "a1_x_b0": pp_tu, "a1_x_b1": pp_tt,
    }
    col_sum_u  = pp_uu
    col_sum_t  = pp_ut + pp_tu
    col_sum_h  = pp_tt
    col_sum_th = 0
    carry_u    = col_sum_u // 10
    run_sum_u  = col_sum_u
    carry_t    = (col_sum_t + carry_u) // 10
    run_sum_t  = col_sum_t + carry_u
    carry_h    = (col_sum_h + carry_t) // 10
    run_sum_h  = col_sum_h + carry_t
    carry_th   = (col_sum_th + carry_h) // 10
    run_sum_th = col_sum_th + carry_h

    column_products = {
        "0": ["a0_x_b0"],
        "1": ["a0_x_b1", "a1_x_b0"],
        "2": ["a1_x_b1"],
        "3": [],
    }

    rec = {}
    rec.update(shared_operand_labels(a, b))
    rec.update({
        "answer": answer,
        "ans_units": ans_u, "ans_tens": ans_t,
        "ans_hundreds": ans_h, "ans_thousands": ans_th,
        "ans_num_digits": n_dig,
        "answer_digits_lsf": answer_lsf,
        "answer_digits_msf": answer_msf,
    })
    rec.update({
        # Tier 2: multiplication algebraic intermediates
        "partial_products": partial_products,
        "partial_product_units":           pp_uu,
        "partial_product_a_units_b_tens":  pp_ut,
        "partial_product_a_tens_b_units":  pp_tu,
        "partial_product_a_tens_b_tens":   pp_tt,
        "column_sum_units":     col_sum_u,
        "column_sum_tens":      col_sum_t,
        "column_sum_hundreds":  col_sum_h,
        "column_sum_thousands": col_sum_th,
        "column_sums":          [col_sum_u, col_sum_t, col_sum_h, col_sum_th],
        "column_products":      column_products,
        "carry_units":          carry_u,
        "carry_tens":           carry_t,
        "carry_hundreds":       carry_h,
        "carry_thousands":      carry_th,
        "carries":              [carry_u, carry_t, carry_h, carry_th],
        "running_sum_units":    run_sum_u,
        "running_sum_tens":     run_sum_t,
        "running_sum_hundreds": run_sum_h,
        "running_sum_thousands": run_sum_th,
        "running_sums":         [run_sum_u, run_sum_t, run_sum_h, run_sum_th],
    })
    rec.update(shared_structural_labels(a, b, answer))
    rec.update(shared_relational_labels(a, b))
    return rec


def attach_tokenization(rec: dict, tok_for_pair: dict, intersection: bool) -> dict:
    """Append Tier 5 tokenization metadata (per-model fields)."""
    flat = {"is_intersection": intersection}
    for mk, py in MODEL_KEY_TO_PYKEY.items():
        if mk not in tok_for_pair:
            raise KeyError(f"Tokenization data missing for model {mk}")
        info = tok_for_pair[mk]
        flat[f"is_single_token_{py}"]   = info["is_single_token"]
        flat[f"first_token_id_{py}"]    = info["first_token_id"]
        flat[f"first_token_text_{py}"]  = info["first_token_text"]
        flat[f"n_tokens_{py}"]          = info["n_tokens"]
    rec.update(flat)
    return rec


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate(rec: dict, task: str):
    a, b, ans = rec["a"], rec["b"], rec["answer"]
    assert 0 <= a <= 99, f"a out of range: {a}"
    assert 0 <= b <= 99, f"b out of range: {b}"
    if task == "addition":
        assert ans == a + b, f"answer mismatch (add): {ans} vs {a+b}"
        assert 0 <= ans <= 198, f"addition answer out of range: {ans}"
    else:
        assert ans == a * b, f"answer mismatch (mul): {ans} vs {a*b}"
        assert 0 <= ans <= 9801, f"multiplication answer out of range: {ans}"

    # Digit reconstruction
    recon = sum(d * 10 ** i for i, d in enumerate(rec["answer_digits_lsf"]))
    assert recon == ans, f"digit reconstruction mismatch: {recon} vs {ans}"

    # Carry/running-sum self-consistency
    if task == "multiplication":
        cs = rec["column_sums"]
        carries = rec["carries"]
        run = rec["running_sums"]
        carry_in = 0
        for k in range(len(cs)):
            assert run[k] == cs[k] + carry_in, (
                f"running-sum mismatch col {k}: {run[k]} vs {cs[k]}+{carry_in}")
            assert carries[k] == run[k] // 10, (
                f"carry mismatch col {k}: {carries[k]} vs {run[k]//10}")
            carry_in = carries[k]
    else:  # addition
        cs = rec["column_sums"]
        carries = rec["carries"]
        run = rec["running_sums"]
        # col 0: cs[0], no carry in
        assert run[0] == cs[0]
        assert carries[0] == (1 if cs[0] >= 10 else 0)
        # col 1: cs[1] + carries[0]
        assert run[1] == cs[1] + carries[0]
        assert carries[1] == (1 if run[1] >= 10 else 0)


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT WRITERS
# ═══════════════════════════════════════════════════════════════════════════════

def csv_cell(v):
    """Serialize a Python value into a single CSV cell."""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (list, tuple)):
        return " ".join(str(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, separators=(",", ":"))
    return v


def write_csv_from_records(path: Path, records: list, extra_first: list[str] = None):
    """Write a CSV with one row per record, columns = union of keys
    (preserving insertion order from the first record).
    """
    if not records:
        path.write_text("")
        return
    # Build column list from the first record (Python dicts preserve insertion order).
    ordered_cols = []
    seen = set()
    base_cols = list(records[0].keys())
    for k in (extra_first or []):
        if k in records[0]:
            if k not in seen:
                ordered_cols.append(k); seen.add(k)
    for k in base_cols:
        if k not in seen:
            ordered_cols.append(k); seen.add(k)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(ordered_cols)
        for r in records:
            w.writerow([csv_cell(r.get(c, "")) for c in ordered_cols])


def sha256_of_path(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# COVERAGE REPORT
# ═══════════════════════════════════════════════════════════════════════════════

# Concepts whose value distribution we audit, per task.
# Tier-1 atomic, Tier-2 algebraic, Tier-3 structural, Tier-4 relational.
# Tier-5 tokenization metadata is deliberately excluded — it's a join artefact.
ADDITION_AUDIT_CONCEPTS = [
    # Tier 1 atomic
    "a", "b", "a_units", "a_tens", "a_num_digits",
    "b_units", "b_tens", "b_num_digits",
    "answer", "ans_units", "ans_tens", "ans_hundreds", "ans_num_digits",
    # Tier 2 algebraic (addition)
    "column_sum_units", "column_sum_tens",
    "carry_units", "carry_tens",
    "running_sum_units", "running_sum_tens",
    # Tier 3 structural
    "a_parity", "b_parity", "ans_parity",
    "parity_match", "parity_xor",
    "a_magnitude_tier", "b_magnitude_tier", "ans_magnitude_tier",
    "ans_ends_in_zero", "ans_is_zero", "a_is_zero", "b_is_zero",
    # Tier 4 relational
    "a_eq_b", "max_operand", "min_operand",
    "operand_diff", "operand_abs_diff", "larger_operand",
    "both_zero", "either_zero", "both_one", "either_one",
]

MULT_AUDIT_CONCEPTS = [
    # Tier 1 atomic
    "a", "b", "a_units", "a_tens", "a_num_digits",
    "b_units", "b_tens", "b_num_digits",
    "answer", "ans_units", "ans_tens", "ans_hundreds", "ans_thousands",
    "ans_num_digits",
    # Tier 2 algebraic (multiplication)
    "partial_product_units", "partial_product_a_units_b_tens",
    "partial_product_a_tens_b_units", "partial_product_a_tens_b_tens",
    "column_sum_units", "column_sum_tens",
    "column_sum_hundreds", "column_sum_thousands",
    "carry_units", "carry_tens", "carry_hundreds", "carry_thousands",
    "running_sum_units", "running_sum_tens",
    "running_sum_hundreds", "running_sum_thousands",
    # Tier 3 structural
    "a_parity", "b_parity", "ans_parity",
    "parity_match", "parity_xor",
    "a_magnitude_tier", "b_magnitude_tier", "ans_magnitude_tier",
    "ans_ends_in_zero", "ans_is_zero", "a_is_zero", "b_is_zero",
    # Tier 4 relational
    "a_eq_b", "max_operand", "min_operand",
    "operand_diff", "operand_abs_diff", "larger_operand",
    "both_zero", "either_zero", "both_one", "either_one",
]

CONCEPT_FLOOR = 30


def per_concept_counts(records: list, concept: str) -> Counter:
    c = Counter()
    for r in records:
        v = r.get(concept)
        if isinstance(v, (list, dict)):
            v = json.dumps(v, separators=(",", ":"))
        c[v] += 1
    return c


def write_coverage_report(path: Path, addition_records: list,
                          multiplication_records: list, logger):
    lines = []
    lines.append("# Coverage report — per-concept value counts\n")
    lines.append(f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}_\n")
    lines.append(f"_Concept floor: {CONCEPT_FLOOR} examples per value (warning if below)._\n\n")

    for task_name, recs, registry in [
        ("Addition", addition_records, ADDITION_AUDIT_CONCEPTS),
        ("Multiplication", multiplication_records, MULT_AUDIT_CONCEPTS),
    ]:
        lines.append(f"\n## {task_name} ({len(recs)} problems)\n\n")
        for concept in registry:
            counts = per_concept_counts(recs, concept)
            n_values = len(counts)
            n_below_floor = sum(1 for v in counts.values() if v < CONCEPT_FLOOR)
            min_count = min(counts.values()) if counts else 0
            max_count = max(counts.values()) if counts else 0
            flag = " :warning:" if n_below_floor > 0 else ""
            lines.append(
                f"### `{concept}` — {n_values} unique values "
                f"(min={min_count}, max={max_count}, "
                f"below_floor={n_below_floor}){flag}\n\n"
            )
            # Tabulate: sort by value if numeric, else by descending count.
            try:
                items = sorted(counts.items(),
                               key=lambda kv: (kv[0] is None, kv[0]))
            except TypeError:
                items = sorted(counts.items(),
                               key=lambda kv: (-kv[1], str(kv[0])))
            lines.append("| value | count | flag |\n|---|---:|---|\n")
            for val, cnt in items:
                marker = "⚠ <30" if cnt < CONCEPT_FLOOR else ""
                lines.append(f"| `{val}` | {cnt} | {marker} |\n")
            lines.append("\n")
            if n_below_floor > 0:
                logger.warning(
                    "  [%s] concept '%s' has %d/%d values below floor of %d",
                    task_name.lower(), concept, n_below_floor, n_values,
                    CONCEPT_FLOOR
                )

    path.write_text("".join(lines))
    logger.info("wrote coverage report -> %s", path)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    logs_root = Path(paths["logs_root"])
    raw_root  = Path(paths["data_root"]) / "data" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(logs_root)
    t_start = time.time()
    logger.info("=" * 78)
    logger.info("Step 2: dataset generation (addition + multiplication)")
    logger.info("config=%s", config_path)
    logger.info("raw output -> %s", raw_root)

    # ───────────────────────────────────────────────────────────────────────────
    # Load Step 1 tokenization data
    # ───────────────────────────────────────────────────────────────────────────
    results_root = Path(paths["results_root"]) / "tokenization_limits"
    add_csv  = results_root / "addition_per_pair.csv"
    mult_csv = results_root / "multiplication_per_pair.csv"
    if not add_csv.exists():  raise FileNotFoundError(add_csv)
    if not mult_csv.exists(): raise FileNotFoundError(mult_csv)
    logger.info("loading addition tokenization data %s", add_csv)
    add_tok  = load_per_pair_tokenization(add_csv)
    logger.info("loading multiplication tokenization data %s", mult_csv)
    mult_tok = load_per_pair_tokenization(mult_csv)
    logger.info("  addition pairs:       %d", len(add_tok))
    logger.info("  multiplication pairs: %d", len(mult_tok))

    a_lo, a_hi = cfg["dataset"]["operand_range"]
    b_lo, b_hi = cfg["dataset"]["operand_range"]
    add_prompt  = cfg["dataset"]["prompts"]["addition"]
    mult_prompt = cfg["dataset"]["prompts"]["multiplication"]

    # ───────────────────────────────────────────────────────────────────────────
    # Build addition dataset (full Cartesian product)
    # ───────────────────────────────────────────────────────────────────────────
    logger.info("-" * 78)
    logger.info("Building addition dataset...")
    addition_records = []
    addition_problems = []  # full problem objects for JSON
    idx = 0
    for a in range(a_lo, a_hi + 1):
        for b in range(b_lo, b_hi + 1):
            tok_for_pair = add_tok.get((a, b))
            if tok_for_pair is None:
                raise RuntimeError(f"missing addition tok for ({a},{b})")
            labels = addition_labels(a, b)
            attach_tokenization(labels, tok_for_pair, intersection=True)
            validate(labels, "addition")
            addition_records.append(labels)
            addition_problems.append({
                "index": idx,
                "task": "addition",
                "prompt": add_prompt.format(a=a, b=b),
                "labels": labels,
            })
            idx += 1
    logger.info("  addition problems: %d", len(addition_problems))

    # ───────────────────────────────────────────────────────────────────────────
    # Compute the cross-model intersection mask for multiplication
    # ───────────────────────────────────────────────────────────────────────────
    logger.info("-" * 78)
    logger.info("Computing multiplication cross-model intersection mask...")
    mask_records = []
    intersection_pairs = []
    for a in range(a_lo, a_hi + 1):
        for b in range(b_lo, b_hi + 1):
            tok_for_pair = mult_tok.get((a, b))
            if tok_for_pair is None:
                raise RuntimeError(f"missing mult tok for ({a},{b})")
            flags = {mk: tok_for_pair[mk]["is_single_token"]
                     for mk in MODEL_KEYS}
            is_inter = all(flags[mk] == 1 for mk in MODEL_KEYS)
            mask_records.append({
                "a": a, "b": b, "answer": a * b,
                "is_single_token_gpt_j":  flags["gpt-j-6b"],
                "is_single_token_llama":  flags["llama-3.1-8b"],
                "is_single_token_pythia": flags["pythia-6.9b"],
                "is_intersection": int(is_inter),
            })
            if is_inter:
                intersection_pairs.append((a, b))
    n_inter = len(intersection_pairs)
    logger.info("  intersection size: %d / %d (%.2f%%)",
                n_inter, len(mult_tok), 100.0 * n_inter / len(mult_tok))

    # Write the mask CSV (10,000 rows)
    mask_path = raw_root / "multiplication_intersection_mask.csv"
    write_csv_from_records(mask_path, mask_records,
                           extra_first=["a", "b", "answer"])
    logger.info("wrote intersection mask -> %s (%d rows)",
                mask_path, len(mask_records))

    # ───────────────────────────────────────────────────────────────────────────
    # Build multiplication dataset (intersection only)
    # ───────────────────────────────────────────────────────────────────────────
    logger.info("-" * 78)
    logger.info("Building multiplication dataset (intersection only)...")
    multiplication_records = []
    multiplication_problems = []
    for new_idx, (a, b) in enumerate(intersection_pairs):
        tok_for_pair = mult_tok[(a, b)]
        labels = multiplication_labels(a, b)
        attach_tokenization(labels, tok_for_pair, intersection=True)
        validate(labels, "multiplication")
        # Invariant: every record in this set has all three single-token flags = 1
        for py in MODEL_KEY_TO_PYKEY.values():
            assert labels[f"is_single_token_{py}"] == 1, (
                f"intersection invariant broken for ({a},{b})")
        multiplication_records.append(labels)
        multiplication_problems.append({
            "index": new_idx,
            "task": "multiplication",
            "prompt": mult_prompt.format(a=a, b=b),
            "labels": labels,
        })
    logger.info("  multiplication problems: %d", len(multiplication_problems))

    # ───────────────────────────────────────────────────────────────────────────
    # Write JSON + CSV outputs
    # ───────────────────────────────────────────────────────────────────────────
    add_json  = raw_root / "addition_problems.json"
    mult_json = raw_root / "multiplication_problems.json"
    add_csv_p  = raw_root / "addition_problems.csv"
    mult_csv_p = raw_root / "multiplication_problems.csv"

    with open(add_json, "w") as f:
        json.dump({
            "task": "addition",
            "n_problems": len(addition_problems),
            "operand_range": [a_lo, a_hi],
            "prompt_template": add_prompt,
            "problems": addition_problems,
        }, f, indent=2)
    logger.info("wrote %s (%d problems)", add_json, len(addition_problems))

    with open(mult_json, "w") as f:
        json.dump({
            "task": "multiplication",
            "n_problems": len(multiplication_problems),
            "operand_range": [a_lo, a_hi],
            "prompt_template": mult_prompt,
            "intersection_models": MODEL_KEYS,
            "problems": multiplication_problems,
        }, f, indent=2)
    logger.info("wrote %s (%d problems)", mult_json,
                len(multiplication_problems))

    write_csv_from_records(add_csv_p, addition_records,
                           extra_first=["a", "b", "answer"])
    logger.info("wrote %s", add_csv_p)
    write_csv_from_records(mult_csv_p, multiplication_records,
                           extra_first=["a", "b", "answer"])
    logger.info("wrote %s", mult_csv_p)

    # ───────────────────────────────────────────────────────────────────────────
    # Coverage report
    # ───────────────────────────────────────────────────────────────────────────
    cov_path = raw_root / "coverage_report.md"
    write_coverage_report(cov_path, addition_records, multiplication_records,
                          logger)

    # ───────────────────────────────────────────────────────────────────────────
    # Manifest
    # ───────────────────────────────────────────────────────────────────────────
    manifest = {
        "schema_version":   "v1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed":             cfg.get("seed", 42),
        "config_path":      str(config_path),
        "config_sha256":    sha256_of_path(config_path),
        "build_script":     "generate_datasets.py",
        "build_script_sha256": sha256_of_path(Path(__file__)),
        "tokenization_csv_addition":      str(add_csv),
        "tokenization_csv_addition_sha256":      sha256_of_path(add_csv),
        "tokenization_csv_multiplication": str(mult_csv),
        "tokenization_csv_multiplication_sha256": sha256_of_path(mult_csv),
        "operand_range":    [a_lo, a_hi],
        "addition_count":   len(addition_problems),
        "multiplication_count": len(multiplication_problems),
        "intersection_models": MODEL_KEYS,
        "outputs": {
            "addition_json":  str(add_json),
            "addition_csv":   str(add_csv_p),
            "multiplication_json": str(mult_json),
            "multiplication_csv":  str(mult_csv_p),
            "intersection_mask_csv": str(mask_path),
            "coverage_report": str(cov_path),
        },
        "numpy_version":    np.__version__,
        "python_version":   platform.python_version(),
        "log_path":         str(logs_root / "generate_datasets.log"),
        "concept_floor":    CONCEPT_FLOOR,
        "addition_audit_concepts":      ADDITION_AUDIT_CONCEPTS,
        "multiplication_audit_concepts": MULT_AUDIT_CONCEPTS,
        "runtime_seconds":  None,  # filled below
    }
    runtime = time.time() - t_start
    manifest["runtime_seconds"] = round(runtime, 3)
    manifest_path = raw_root / "build_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("wrote manifest -> %s", manifest_path)

    # ───────────────────────────────────────────────────────────────────────────
    # Headlines
    # ───────────────────────────────────────────────────────────────────────────
    logger.info("=" * 78)
    logger.info("HEADLINE NUMBERS")
    logger.info("  addition_count       = %d", len(addition_problems))
    logger.info("  multiplication_count = %d (intersection of %d / 10000 pairs)",
                len(multiplication_problems), len(multiplication_problems))
    logger.info("  runtime              = %.2f s", runtime)
    logger.info("DONE")


if __name__ == "__main__":
    main()
