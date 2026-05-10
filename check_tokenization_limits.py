"""Single-token integer limit sweep across GPT-J 6B, Llama 3.1 8B, and Pythia 6.9B.

Tokenizer-only preflight (no model weights loaded). Verifies plan v6 §4.2
(addition single-token assumption) and §4.3 / §21.5 (multiplication multi-token
rate). Writes per-pair CSVs that downstream stages can join against.

Usage:
    python check_tokenization_limits.py --config /home/anshulk/emnlp2026/config.yaml
"""

import argparse
import csv
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import yaml
from transformers import AutoTokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(logs_root: Path):
    logs_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tokenization_limits")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = RotatingFileHandler(logs_root / "tokenization_limits.log",
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
# TOKENIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def tokens_for_n_in_context(tok, n: int, prefix: str) -> dict:
    """Tokenize `prefix + str(n)` and report what str(n) contributes.

    Returns a dict with:
      prefix_stable (bool): True iff tok(prefix) is a token-id prefix of
        tok(prefix + str(n)). When False, the prefix's last token merged with
        the leading char of str(n) (classic BPE behavior, e.g. ' ' + '0' ->
        single token ' 0' in GPT-style tokenizers).
      ans_ids (list[int] or None): the trailing token ids attributable to
        str(n) when prefix_stable, else None.
      n_tokens (int or None): len(ans_ids) when prefix_stable, else None.
      delta_len (int): len(full) - len(prefix). Robust regardless of merging,
        but loses meaning when the merge eats a prefix token.
      full_ids (list[int]): tok(prefix + str(n)).
      prefix_ids (list[int]): tok(prefix).
    """
    prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    full_ids = tok(prefix + str(n), add_special_tokens=False)["input_ids"]
    stable = full_ids[: len(prefix_ids)] == prefix_ids
    ans_ids = full_ids[len(prefix_ids):] if stable else None
    return {
        "prefix_stable": stable,
        "ans_ids": ans_ids,
        "n_tokens": len(ans_ids) if stable else None,
        "delta_len": len(full_ids) - len(prefix_ids),
        "full_ids": full_ids,
        "prefix_ids": prefix_ids,
    }


CONTEXT_PREFIXES = {
    "bare":           "",
    "leading_space":  " ",
    "post_equals":    "=",
    "post_plus":      "+",
    "post_star":      "*",
}


# ═══════════════════════════════════════════════════════════════════════════════
# PER-MODEL SWEEPS
# ═══════════════════════════════════════════════════════════════════════════════

def sweep_integer_contexts(tok, sweep_max: int, logger) -> dict:
    """For each fixed-prefix context, record per-integer info for n in [0, sweep_max].

    Returns: {context_name: {n: result_dict_from_tokens_for_n_in_context}}.
    """
    out = {ctx: {} for ctx in CONTEXT_PREFIXES}
    for ctx, prefix in CONTEXT_PREFIXES.items():
        for n in range(sweep_max + 1):
            out[ctx][n] = tokens_for_n_in_context(tok, n, prefix)
        logger.debug("  context=%s done (%d integers)", ctx, sweep_max + 1)
    return out


def per_pair_sweep(tok, prompt_template: str, op, a_lo: int, a_hi: int,
                   b_lo: int, b_hi: int) -> list[dict]:
    """For each (a, b) in the operand box, encode the full KT prompt + answer
    and record how many tokens the answer occupies in that real context.
    """
    rows = []
    for a in range(a_lo, a_hi + 1):
        for b in range(b_lo, b_hi + 1):
            ans = op(a, b)
            prefix = prompt_template.format(a=a, b=b)
            res = tokens_for_n_in_context(tok, ans, prefix)
            ans_ids = res["ans_ids"] or []
            first = ans_ids[0] if ans_ids else None
            rows.append({
                "a": a, "b": b, "answer": ans,
                "prefix_stable": int(res["prefix_stable"]),
                "n_tokens": res["n_tokens"] if res["n_tokens"] is not None else "",
                "is_single_token": int(res["prefix_stable"] and res["n_tokens"] == 1),
                "first_token_id": first if first is not None else "",
                "first_token_text": tok.decode([first]) if first is not None else "",
                "all_token_ids": " ".join(str(x) for x in ans_ids),
                "delta_len": res["delta_len"],
            })
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

def summarize_context(tok, ctx_name: str, ctx_results: dict, sweep_max: int) -> dict:
    """Compute summary stats for a context's per-integer sweep.

    A "single-token" hit requires prefix_stable AND n_tokens == 1. When the
    prefix is BPE-merged with str(n) (e.g. " " + "0" -> single fused token in
    GPT/Pythia tokenizers), the integer is reported as merged_count, not as
    single-token, since str(n) does not have its own tokens in that context.
    """
    n_single = sum(1 for r in ctx_results.values()
                   if r["prefix_stable"] and r["n_tokens"] == 1)
    n_merged = sum(1 for r in ctx_results.values() if not r["prefix_stable"])

    # Maximum N such that every integer in [0, N] is prefix-stable single-token.
    max_contig = -1
    for n in range(sweep_max + 1):
        r = ctx_results[n]
        if r["prefix_stable"] and r["n_tokens"] == 1:
            max_contig = n
        else:
            break

    # First integer that is NOT prefix-stable single-token (multi or merged).
    first_bad = None
    first_bad_kind = ""
    for n in range(sweep_max + 1):
        r = ctx_results[n]
        if not (r["prefix_stable"] and r["n_tokens"] == 1):
            first_bad = n
            first_bad_kind = "merged" if not r["prefix_stable"] else "multi"
            break
    example = ""
    if first_bad is not None:
        r = ctx_results[first_bad]
        if r["prefix_stable"]:
            decoded = [tok.decode([i]) for i in r["ans_ids"]]
            example = f"n={first_bad} ans_ids={r['ans_ids']} decoded={decoded}"
        else:
            decoded = [tok.decode([i]) for i in r["full_ids"]]
            example = (f"n={first_bad} MERGED prefix_ids={r['prefix_ids']} "
                       f"full_ids={r['full_ids']} decoded={decoded}")
    return {
        "context": ctx_name,
        "max_contiguous_single_token_N": max_contig,
        "total_single_token_count_in_0_sweep_max": n_single,
        "merged_count_in_0_sweep_max": n_merged,
        "first_failing_n": first_bad if first_bad is not None else "",
        "first_failing_kind": first_bad_kind,
        "example_failure": example,
    }


def coverage_for_pairs(rows: list[dict]) -> dict:
    n_total = len(rows)
    n_single = sum(r["is_single_token"] for r in rows)
    multi_answers = [r["answer"] for r in rows if not r["is_single_token"]]
    single_answers = [r["answer"] for r in rows if r["is_single_token"]]
    return {
        "n_pairs_total": n_total,
        "n_pairs_single_token": n_single,
        "frac_single_token": round(n_single / n_total, 6),
        "min_answer_multi_token": min(multi_answers) if multi_answers else "",
        "max_answer_single_token": max(single_answers) if single_answers else "",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    logs_root = Path(paths["logs_root"])
    results_root = Path(paths["results_root"]) / "tokenization_limits"
    results_root.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(logs_root)
    logger.info("=" * 78)
    logger.info("Tokenization-limits preflight")
    logger.info("config=%s", args.config)
    logger.info("results -> %s", results_root)
    logger.info("logs    -> %s", logs_root / "tokenization_limits.log")

    sweep_max = cfg["tokenization"]["sweep_max"]
    a_lo, a_hi = cfg["dataset"]["operand_range"]
    b_lo, b_hi = cfg["dataset"]["operand_range"]
    prompts = cfg["dataset"]["prompts"]

    summary_rows = []   # (model, context, ...)
    coverage_rows = []  # (model, task, ...)

    addition_per_pair = []        # rows tagged with model_key
    multiplication_per_pair = []

    for m in cfg["models"]:
        logger.info("-" * 78)
        logger.info("Model: %s  (%s)", m["name"], m["local_path"])
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(m["local_path"])
        logger.info("  tokenizer loaded in %.1fs | vocab_size=%d",
                    time.time() - t0, tok.vocab_size)

        # 1. Per-integer sweeps over fixed-prefix contexts (0..sweep_max)
        logger.info("  sweeping integer contexts 0..%d ...", sweep_max)
        t1 = time.time()
        ctx_results = sweep_integer_contexts(tok, sweep_max, logger)
        logger.info("  context sweeps done in %.1fs", time.time() - t1)

        for ctx_name, ctx_data in ctx_results.items():
            row = summarize_context(tok, ctx_name, ctx_data, sweep_max)
            row = {"model_key": m["key"], "model_name": m["name"], **row}
            summary_rows.append(row)
            logger.info("    [%s | %-15s] max_contig_N=%s  first_failing=%s (%s)  merged=%d",
                        m["key"], ctx_name,
                        row["max_contiguous_single_token_N"],
                        row["first_failing_n"],
                        row["first_failing_kind"] or "-",
                        row["merged_count_in_0_sweep_max"])

        # 2. Per-pair sweep in the *real* KT prompt context.
        for task_name, op, prompt_key in [
            ("addition", lambda a, b: a + b, "addition"),
            ("multiplication", lambda a, b: a * b, "multiplication"),
        ]:
            logger.info("  per-pair sweep: task=%s prompt=%r",
                        task_name, prompts[prompt_key])
            t2 = time.time()
            pair_rows = per_pair_sweep(tok, prompts[prompt_key], op,
                                       a_lo, a_hi, b_lo, b_hi)
            logger.info("    %d pairs in %.1fs", len(pair_rows),
                        time.time() - t2)

            # Tag with model and append to global per-task sink.
            for r in pair_rows:
                r2 = {"model_key": m["key"], "model_name": m["name"], **r}
                if task_name == "addition":
                    addition_per_pair.append(r2)
                else:
                    multiplication_per_pair.append(r2)

            # Also synthesize a kt_*_full summary row for this model+task,
            # mirroring what summarize_context produces for fixed contexts.
            cov = coverage_for_pairs(pair_rows)
            coverage_rows.append({
                "model_key": m["key"], "model_name": m["name"],
                "task": task_name,
                "operand_range": f"[{a_lo},{a_hi}]",
                "prompt": prompts[prompt_key],
                **cov,
            })
            ctx_label = "kt_addition_full" if task_name == "addition" else "kt_multiplication_full"
            single_answers = sorted({r["answer"] for r in pair_rows if r["is_single_token"]})
            multi_answers = sorted({r["answer"] for r in pair_rows if not r["is_single_token"]})
            n_merged = sum(1 for r in pair_rows if not r["prefix_stable"])
            example_split = ""
            if multi_answers:
                first_multi_ans = multi_answers[0]
                example_row = next(r for r in pair_rows if r["answer"] == first_multi_ans)
                example_split = (f"a={example_row['a']} b={example_row['b']} "
                                 f"answer={first_multi_ans} ids=[{example_row['all_token_ids']}]")
            # Max contiguous: largest answer A such that all integers in [0, A]
            # that ACTUALLY appear as answers are single-token in the KT context.
            max_contig = -1
            answers_set = {r["answer"]: r["is_single_token"] for r in pair_rows}
            for n_ans in sorted(answers_set.keys()):
                if answers_set[n_ans]:
                    max_contig = n_ans
                else:
                    break
            summary_rows.append({
                "model_key": m["key"], "model_name": m["name"],
                "context": ctx_label,
                "max_contiguous_single_token_N": max_contig,
                "total_single_token_count_in_0_sweep_max": len(single_answers),
                "merged_count_in_0_sweep_max": n_merged,
                "first_failing_n": multi_answers[0] if multi_answers else "",
                "first_failing_kind": "multi" if multi_answers else "",
                "example_failure": example_split,
            })

    # ───────────────────────────────────────────────────────────────────────────
    # Write CSVs
    # ───────────────────────────────────────────────────────────────────────────

    def write_csv(path, rows, fieldnames):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    summary_path = results_root / "summary.csv"
    write_csv(summary_path, summary_rows,
              fieldnames=["model_key", "model_name", "context",
                          "max_contiguous_single_token_N",
                          "total_single_token_count_in_0_sweep_max",
                          "merged_count_in_0_sweep_max",
                          "first_failing_n", "first_failing_kind",
                          "example_failure"])
    logger.info("wrote %s (%d rows)", summary_path, len(summary_rows))

    coverage_path = results_root / "coverage.csv"
    write_csv(coverage_path, coverage_rows,
              fieldnames=["model_key", "model_name", "task", "operand_range",
                          "prompt", "n_pairs_total", "n_pairs_single_token",
                          "frac_single_token", "min_answer_multi_token",
                          "max_answer_single_token"])
    logger.info("wrote %s (%d rows)", coverage_path, len(coverage_rows))

    pair_fieldnames = ["model_key", "model_name", "a", "b", "answer",
                       "prefix_stable", "n_tokens", "is_single_token",
                       "first_token_id", "first_token_text",
                       "all_token_ids", "delta_len"]

    add_path = results_root / "addition_per_pair.csv"
    write_csv(add_path, addition_per_pair, fieldnames=pair_fieldnames)
    logger.info("wrote %s (%d rows)", add_path, len(addition_per_pair))

    mult_path = results_root / "multiplication_per_pair.csv"
    write_csv(mult_path, multiplication_per_pair, fieldnames=pair_fieldnames)
    logger.info("wrote %s (%d rows)", mult_path, len(multiplication_per_pair))

    # ───────────────────────────────────────────────────────────────────────────
    # Headline numbers to stderr
    # ───────────────────────────────────────────────────────────────────────────
    logger.info("=" * 78)
    logger.info("HEADLINE NUMBERS")
    logger.info("=" * 78)
    for row in coverage_rows:
        logger.info("  %-13s | %-15s | %5d/%d single-token (%.1f%%)  "
                    "max_single_ans=%s  min_multi_ans=%s",
                    row["model_key"], row["task"],
                    row["n_pairs_single_token"], row["n_pairs_total"],
                    100.0 * row["frac_single_token"],
                    row["max_answer_single_token"],
                    row["min_answer_multi_token"])
    logger.info("DONE")


if __name__ == "__main__":
    main()
