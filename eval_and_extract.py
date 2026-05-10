"""Step 3: per-model behavioral evaluation + activation extraction.

Loads one model in bfloat16 onto one GPU, runs both datasets (addition + the
multiplication intersection), captures residual-stream activations at the `=`
token (last input token, position -1) for the configured layers via forward
hooks, runs `model.generate(max_new_tokens=4)` for first-token-correctness
behavioral evaluation, and writes JSON + CSV + .npy + manifest.

This script is per-model. Three SLURM array tasks (one per model) provide the
cross-model coverage. The current GPU node is used for `--smoke-test` runs.

Usage:
  # Smoke test (4 problems per task) on the current node:
  python eval_and_extract.py --config config.yaml --model gpt-j-6b --smoke-test

  # Full run (called from SLURM):
  python eval_and_extract.py --config config.yaml --model llama-3.1-8b
"""

import argparse
import csv
import gc
import hashlib
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(logs_root: Path, model_key: str):
    logs_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"eval_{model_key}")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = RotatingFileHandler(
        logs_root / f"eval_and_extract_{model_key}.log",
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
# CONFIG / PER-MODEL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_KEY_TO_PYKEY = {
    "gpt-j-6b": "gpt_j",
    "llama-3.1-8b": "llama",
    "pythia-6.9b": "pythia",
}


def get_model_cfg(cfg: dict, model_key: str) -> dict:
    for m in cfg["models"]:
        if m["key"] == model_key:
            return m
    raise ValueError(f"Unknown model_key: {model_key}")


def get_layer_modules(model, model_key: str):
    """Return the list of transformer-block modules per architecture."""
    if model_key == "gpt-j-6b":
        return model.transformer.h            # GPTJForCausalLM
    if model_key == "llama-3.1-8b":
        return model.model.layers             # LlamaForCausalLM
    if model_key == "pythia-6.9b":
        return model.gpt_neox.layers          # GPTNeoXForCausalLM
    raise ValueError(f"Unknown model_key for layer access: {model_key}")


def make_hook(storage: dict, layer_idx: int):
    """Capture the residual-stream output of a transformer block at position -1."""
    def hook_fn(module, inp, output):
        # HF block forward returns either a Tensor or a tuple-with-tensor[0].
        hidden = output if isinstance(output, torch.Tensor) else output[0]
        storage[layer_idx].append(hidden[:, -1, :].detach().float().cpu())
    return hook_fn


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# CORE: PER-TASK PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def run_task(
    *, model, tokenizer, model_key: str, py_key: str,
    problems: list, layers: list, batch_size: int, min_batch_size: int,
    max_new_tokens: int, hidden_dim: int, logger,
):
    """Process one task: extract activations for `layers` and generate answers.

    Returns (predictions: list[dict], activations: dict[layer_idx -> np.ndarray]).
    """
    layer_modules = get_layer_modules(model, model_key)
    n = len(problems)
    captured_act = {L: [] for L in layers}
    predictions = []

    cur_bs = batch_size
    i = 0
    t_extract = 0.0
    t_generate = 0.0
    while i < n:
        batch = problems[i: i + cur_bs]
        prompts = [p["prompt"] for p in batch]
        try:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

            # ── pass 1: activation extraction (forward, no generation) ──
            t0 = time.time()
            handles = []
            tmp = {L: [] for L in layers}
            for L in layers:
                h = layer_modules[L].register_forward_hook(make_hook(tmp, L))
                handles.append(h)
            with torch.inference_mode():
                _ = model(**inputs)
            for h in handles:
                h.remove()
            for L in layers:
                # tmp[L] is a list of one tensor of shape (cur_bs, hidden)
                captured_act[L].append(tmp[L][0])
            t_extract += time.time() - t0

            # ── pass 2: generation (greedy, max_new_tokens) ──
            t0 = time.time()
            with torch.inference_mode():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            t_generate += time.time() - t0
            input_len = inputs.input_ids.shape[1]
            new_tokens = gen[:, input_len:]
            for j, prob in enumerate(batch):
                first_id = int(new_tokens[j, 0])
                first_text = tokenizer.decode([first_id])
                gold_id = prob["labels"][f"first_token_id_{py_key}"]
                gold_text = prob["labels"][f"first_token_text_{py_key}"]
                raw_text = tokenizer.decode(
                    new_tokens[j].tolist(), skip_special_tokens=True
                )
                predictions.append({
                    "index": prob["index"],
                    "a": prob["labels"]["a"],
                    "b": prob["labels"]["b"],
                    "answer": prob["labels"]["answer"],
                    "prompt": prob["prompt"],
                    "gold_first_token_id": gold_id,
                    "gold_first_token_text": gold_text,
                    "pred_first_token_id": first_id,
                    "pred_first_token_text": first_text,
                    "raw_text": raw_text,
                    "correct": int(first_id == gold_id),
                })

            i += cur_bs
            if i % (cur_bs * 8) == 0 or i >= n:
                logger.info(
                    "    [%s] %d / %d (extract %.1fs, generate %.1fs, batch=%d)",
                    py_key, i, n, t_extract, t_generate, cur_bs
                )

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            new_bs = max(min_batch_size, cur_bs // 2)
            if new_bs == cur_bs:
                logger.error("OOM at min batch size %d; aborting", cur_bs)
                raise
            logger.warning(
                "  CUDA OOM at batch=%d; halving to %d and retrying", cur_bs, new_bs
            )
            cur_bs = new_bs
            # Drop any partial captures from the failed batch (no progress made).
            continue

    # Stack captured activations per layer.
    activations = {}
    for L in layers:
        arr = torch.cat(captured_act[L], dim=0).numpy().astype(np.float32)
        if arr.shape[0] != n:
            raise RuntimeError(
                f"Activation row count mismatch for layer {L}: {arr.shape[0]} vs {n}"
            )
        if arr.shape[1] != hidden_dim:
            raise RuntimeError(
                f"Activation hidden_dim mismatch for layer {L}: {arr.shape[1]} vs {hidden_dim}"
            )
        activations[L] = arr

    return predictions, activations, {
        "t_extract_s": round(t_extract, 3),
        "t_generate_s": round(t_generate, 3),
        "final_batch_size": cur_bs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_activations(
    activations: dict, problems: list, hidden_dim: int, logger
) -> dict:
    """Mirror arithmetic-geometry's post_extraction_checks."""
    report = {}
    n = len(problems)
    # Find first two distinct prompts for the closure-bug check.
    distinct_idx = None
    for k in range(1, n):
        if problems[k]["prompt"] != problems[0]["prompt"]:
            distinct_idx = k
            break
    for L, arr in activations.items():
        rep = {"layer": L, "shape": list(arr.shape)}
        if arr.shape != (n, hidden_dim):
            rep["error"] = f"shape mismatch: {arr.shape} vs ({n},{hidden_dim})"
            logger.error("  layer %d: %s", L, rep["error"])
        rep["any_nan"] = bool(np.any(np.isnan(arr)))
        rep["any_inf"] = bool(np.any(np.isinf(arr)))
        if rep["any_nan"] or rep["any_inf"]:
            logger.error("  layer %d: NaN=%s Inf=%s", L,
                         rep["any_nan"], rep["any_inf"])
        if distinct_idx is not None:
            same = bool(np.allclose(arr[0], arr[distinct_idx]))
            rep["distinctness_ok"] = (not same)
            if same:
                logger.error(
                    "  layer %d: distinct prompts produced identical activations",
                    L
                )
        else:
            rep["distinctness_ok"] = True  # only one unique prompt; vacuous
        norms = np.linalg.norm(arr, axis=1)
        rep["norm_min"]  = float(norms.min())
        rep["norm_mean"] = float(norms.mean())
        rep["norm_max"]  = float(norms.max())
        logger.debug(
            "  layer %d: shape=%s norms[min/mean/max]=%.1f/%.1f/%.1f",
            L, arr.shape, rep["norm_min"], rep["norm_mean"], rep["norm_max"]
        )
        report[str(L)] = rep
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT WRITERS
# ═══════════════════════════════════════════════════════════════════════════════

def write_predictions(path_json: Path, path_csv: Path, predictions: list,
                      header: dict, logger):
    payload = {**header, "results": predictions}
    with open(path_json, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("  wrote %s", path_json)

    if predictions:
        cols = list(predictions[0].keys())
        with open(path_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(predictions)
    else:
        path_csv.write_text("")
    logger.info("  wrote %s", path_csv)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True,
                        choices=list(MODEL_KEY_TO_PYKEY.keys()))
    parser.add_argument("--task", choices=["addition", "multiplication"],
                        default=None,
                        help="if not given, runs both tasks")
    parser.add_argument("--smoke-test", action="store_true",
                        help="process 4 problems per task, write to _smoke/")
    parser.add_argument("--util-probe", action="store_true",
                        help="run full addition only, write to _smoke/, "
                             "for measuring GPU utilization")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    paths = cfg["paths"]
    data_root = Path(paths["data_root"])
    logs_root = Path(paths["logs_root"])

    model_key = args.model
    py_key    = MODEL_KEY_TO_PYKEY[model_key]
    mcfg      = get_model_cfg(cfg, model_key)

    smoke = args.smoke_test or args.util_probe
    if smoke:
        ans_root = data_root / "answers" / "_smoke" / model_key
        act_root = data_root / "activations" / "_smoke" / model_key
        log_key  = f"{model_key}_smoke"
    else:
        ans_root = data_root / "answers" / model_key
        act_root = data_root / "activations" / model_key
        log_key  = model_key
    ans_root.mkdir(parents=True, exist_ok=True)
    act_root.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(logs_root, log_key)
    t_run0 = time.time()

    logger.info("=" * 78)
    logger.info("Step 3: eval + activation extraction")
    logger.info("model=%s  py_key=%s  smoke=%s", model_key, py_key, smoke)
    logger.info("config=%s", config_path)
    logger.info("answers -> %s", ans_root)
    logger.info("activations -> %s", act_root)

    # ───────────────────────────────────────────────────────────────────────────
    # Load model + tokenizer
    # ───────────────────────────────────────────────────────────────────────────
    eval_cfg = cfg.get("eval", {})
    bs       = int(eval_cfg.get("batch_size", 512))
    min_bs   = int(eval_cfg.get("min_batch_size", 64))
    max_nt   = int(eval_cfg.get("max_new_tokens", 4))
    dtype_s  = eval_cfg.get("inference_dtype", "bfloat16")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[dtype_s]

    logger.info("loading tokenizer %s", mcfg["local_path"])
    tokenizer = AutoTokenizer.from_pretrained(mcfg["local_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logger.info("loading model %s in %s", mcfg["local_path"], dtype_s)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["local_path"], torch_dtype=dtype, device_map="cuda:0"
    )
    model.eval()
    hidden_dim = model.config.hidden_size
    logger.info(
        "  loaded in %.1fs | params=%.2fB | hidden=%d | dtype=%s",
        time.time() - t0,
        sum(p.numel() for p in model.parameters()) / 1e9,
        hidden_dim, str(next(model.parameters()).dtype),
    )

    layers = mcfg["layers"]
    logger.info("layers to extract: %s", layers)

    # ───────────────────────────────────────────────────────────────────────────
    # Decide tasks
    # ───────────────────────────────────────────────────────────────────────────
    raw_root = data_root / "data" / "raw"
    task_specs = []
    if args.util_probe:
        task_specs = [("addition", raw_root / "addition_problems.json")]
    else:
        if args.task in (None, "addition"):
            task_specs.append(("addition", raw_root / "addition_problems.json"))
        if args.task in (None, "multiplication"):
            task_specs.append(("multiplication",
                               raw_root / "multiplication_problems.json"))

    summary = {
        "model_key": model_key, "model_name": mcfg["name"],
        "smoke": smoke, "tasks": {},
    }
    validation_report = {}

    for task_name, ds_path in task_specs:
        logger.info("-" * 78)
        logger.info("Task: %s  (dataset %s)", task_name, ds_path)
        ds = load_dataset(ds_path)
        problems = ds["problems"]
        if smoke and not args.util_probe:
            problems = problems[:4]
        logger.info("  n_problems=%d", len(problems))

        t0 = time.time()
        predictions, activations, timings = run_task(
            model=model, tokenizer=tokenizer, model_key=model_key, py_key=py_key,
            problems=problems, layers=layers, batch_size=bs,
            min_batch_size=min_bs, max_new_tokens=max_nt,
            hidden_dim=hidden_dim, logger=logger,
        )
        task_runtime = time.time() - t0
        n_correct = sum(p["correct"] for p in predictions)
        accuracy = n_correct / len(predictions) if predictions else 0.0
        logger.info(
            "  done in %.1fs | accuracy = %d/%d (%.2f%%)  final_batch=%d",
            task_runtime, n_correct, len(predictions),
            100 * accuracy, timings["final_batch_size"],
        )

        # Write activations as float32 .npy
        for L, arr in activations.items():
            path = act_root / f"{task_name}_layer_{L:02d}.npy"
            np.save(path, arr)
            logger.info(
                "  wrote activation %s shape=%s dtype=%s",
                path, arr.shape, arr.dtype,
            )

        # Validate
        rep = validate_activations(activations, problems, hidden_dim, logger)
        validation_report[task_name] = rep

        # Write predictions JSON + CSV
        header = {
            "task": task_name,
            "model_key": model_key,
            "model_name": mcfg["name"],
            "n_problems": len(predictions),
            "n_correct": n_correct,
            "accuracy": accuracy,
            "operand_range": ds.get("operand_range"),
            "prompt_template": ds.get("prompt_template"),
            "smoke": smoke,
        }
        write_predictions(
            ans_root / f"{task_name}_answers.json",
            ans_root / f"{task_name}_answers.csv",
            predictions, header, logger,
        )

        # Print smoke records inline
        if smoke and not args.util_probe:
            logger.info("  smoke results:")
            for r in predictions:
                logger.info(
                    "    a=%d b=%d ans=%d gold='%s'(%d) pred='%s'(%d) ok=%s raw=%r",
                    r["a"], r["b"], r["answer"],
                    r["gold_first_token_text"], r["gold_first_token_id"],
                    r["pred_first_token_text"], r["pred_first_token_id"],
                    r["correct"], r["raw_text"],
                )

        summary["tasks"][task_name] = {
            "n_problems":   len(predictions),
            "n_correct":    n_correct,
            "accuracy":     accuracy,
            "task_runtime_seconds": round(task_runtime, 3),
            **timings,
        }

    # ───────────────────────────────────────────────────────────────────────────
    # Manifest + summary
    # ───────────────────────────────────────────────────────────────────────────
    summary["total_runtime_seconds"] = round(time.time() - t_run0, 3)
    summary["timestamp_utc"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    with open(ans_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("wrote %s", ans_root / "summary.json")

    manifest = {
        "schema_version": "v1",
        "model_key":  model_key,
        "model_name": mcfg["name"],
        "model_local_path": mcfg["local_path"],
        "hidden_dim":  hidden_dim,
        "layers":      layers,
        "inference_dtype": dtype_s,
        "activation_dtype": "float32",
        "batch_size_initial": bs,
        "batch_size_min":     min_bs,
        "max_new_tokens":  max_nt,
        "operand_range":   cfg["dataset"]["operand_range"],
        "prompt_addition":      cfg["dataset"]["prompts"]["addition"],
        "prompt_multiplication": cfg["dataset"]["prompts"]["multiplication"],
        "config_path":      str(config_path),
        "config_sha256":    sha256_of(config_path),
        "addition_dataset_path":   str(raw_root / "addition_problems.json"),
        "addition_dataset_sha256": sha256_of(raw_root / "addition_problems.json"),
        "multiplication_dataset_path":   str(raw_root / "multiplication_problems.json"),
        "multiplication_dataset_sha256": sha256_of(raw_root / "multiplication_problems.json"),
        "validation_report": validation_report,
        "tasks_run": [t for t, _ in task_specs],
        "smoke": smoke,
        "timestamp_utc": summary["timestamp_utc"],
        "total_runtime_seconds": summary["total_runtime_seconds"],
        "torch_version":  torch.__version__,
        "cuda_version":   torch.version.cuda,
        "transformers_version": __import__("transformers").__version__,
        "numpy_version":  np.__version__,
        "python_version": platform.python_version(),
        "log_path": str(logs_root / f"eval_and_extract_{log_key}.log"),
        "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": __import__("os").environ.get("SLURM_ARRAY_TASK_ID"),
    }
    with open(act_root / "extraction_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("wrote %s", act_root / "extraction_manifest.json")

    # ───────────────────────────────────────────────────────────────────────────
    # Headlines
    # ───────────────────────────────────────────────────────────────────────────
    logger.info("=" * 78)
    logger.info("HEADLINE NUMBERS")
    for task_name, info in summary["tasks"].items():
        logger.info(
            "  %-15s | n=%5d  correct=%5d  acc=%6.2f%%  runtime=%.1fs",
            task_name, info["n_problems"], info["n_correct"],
            100 * info["accuracy"], info["task_runtime_seconds"],
        )
    logger.info("DONE  total_runtime=%.1fs", summary["total_runtime_seconds"])

    # Free the model so subsequent smoke-test invocations have headroom.
    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
