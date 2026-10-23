"""
Controlled three-mode OLMoE prefetch benchmark.

Modes:
  A: baseline forward
  B: transfer-only (stage weights concurrently, do not install them)
  C: staged weights (stage and install predicted expert weights)

This is a single-token forward diagnostic, not a full generation benchmark.
Writes per-trial results to results/real_prefetch_three_mode.csv.
"""

import csv
import statistics
import time
from pathlib import Path

import torch

from src.modeling.olmoe_loader import load_olmoe_model, load_tokenizer
from src.prefetch.layer_aware_prefetcher import LayerAwareHistoryPrefetcher
from scripts.diagnose_async_prefetch import (
    prime_predictor,
    run_forward,
    run_async_trial,
    start_staging,
    sync,
)

LAYER_ID = 2
PREFETCH_START_LAYER = 1
PROMPT = "The future of artificial intelligence is"
TRIALS = 7
LOGIT_ATOL = 1e-5
RESULTS_PATH = Path("results") / "real_prefetch_three_mode.csv"

LABELS = {
    "A": "Baseline",
    "B": "Transfer only",
    "C": "Staged weights",
}

# Seven rounds; each mode appears first, second, and third at least twice.
ORDERS = [
    ("A", "B", "C"),
    ("B", "C", "A"),
    ("C", "A", "B"),
    ("A", "C", "B"),
    ("C", "B", "A"),
    ("B", "A", "C"),
    ("A", "B", "C"),
]


def timed_baseline(model, input_ids):
    sync()
    start = time.perf_counter()
    logits = run_forward(model, input_ids)
    sync()
    return logits, (time.perf_counter() - start) * 1000


def timed_transfer_only(model, input_ids, expert_id):
    """Stage the candidate during layer 1 without installing its weight maps."""
    layer1 = model.model.layers[PREFETCH_START_LAYER]
    state = {"staging": None}

    def start_hook(module, inputs):
        state["staging"] = start_staging(model, expert_id)

    handle = layer1.register_forward_pre_hook(start_hook)
    sync()
    start = time.perf_counter()
    try:
        logits = run_forward(model, input_ids)
        sync()
        elapsed_ms = (time.perf_counter() - start) * 1000
    finally:
        handle.remove()
        staging = state["staging"]
        if staging is not None:
            # Do not release pinned sources/tensors before asynchronous copies end.
            staging["copy_end"].synchronize()

    return logits, elapsed_ms, state["staging"]


def run_mode(model, input_ids, expert_id, mode):
    if mode == "A":
        logits, elapsed_ms = timed_baseline(model, input_ids)
        return logits, elapsed_ms, None
    if mode == "B":
        return timed_transfer_only(model, input_ids, expert_id)
    if mode == "C":
        return run_async_trial(model, input_ids, expert_id)
    raise ValueError(f"Unknown mode: {mode}")


def staging_metrics(staging):
    if staging is None:
        return {
            "cpu_prepare_ms": "",
            "enqueue_ms": "",
            "gpu_copy_ms": "",
            "wait_ms": "",
            "staged_hits": "",
            "fallback_reads": "",
        }

    copy_ms = staging.get("copy_ms")
    if copy_ms is None:
        copy_ms = staging["copy_start"].elapsed_time(staging["copy_end"])

    wrappers = staging.get("wrappers", {})
    return {
        "cpu_prepare_ms": staging.get("cpu_prepare_ms", ""),
        "enqueue_ms": staging.get("enqueue_ms", ""),
        "gpu_copy_ms": copy_ms,
        "wait_ms": staging.get("wait_ms", ""),
        "staged_hits": sum(w.staged_hits for w in wrappers.values()),
        "fallback_reads": sum(w.fallback_reads for w in wrappers.values()),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    print("Loading tokenizer and OLMoE...")
    tokenizer = load_tokenizer()
    model = load_olmoe_model()
    model.eval()

    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to("cuda")
    # Preserve the original diagnostic: evaluate one token, not full generation.
    target_ids = input_ids[:, -1:]

    predictor = LayerAwareHistoryPrefetcher(history_size=8, prefetch_size=1)
    print("Priming predictor on the fixed prompt...")
    prime_predictor(model, predictor, input_ids)
    candidates = predictor.predict(LAYER_ID)
    if not candidates:
        raise RuntimeError("Predictor produced no expert candidate.")
    expert_id = candidates[0]
    print(f"Frozen candidate: layer={LAYER_ID}, expert={expert_id}")

    print("Warming up all three modes...")
    warmup = {}
    warmup_staging = {}
    for mode in ("A", "B", "C"):
        logits, _, staging = run_mode(model, target_ids, expert_id, mode)
        warmup[mode] = logits
        warmup_staging[mode] = staging
        sync()

    for mode in ("B", "C"):
        diff = (warmup["A"] - warmup[mode]).abs().max().item()
        print(f"Warm-up max |logit difference| A vs {mode}: {diff:.8g}")

    warmup_c_metrics = staging_metrics(warmup_staging["C"])
    print(f"Warm-up staged-weight metrics: {warmup_c_metrics}")
    if not warmup_c_metrics["staged_hits"]:
        raise RuntimeError(
            "The staged expert was not used during the warm-up forward "
            "(zero staged projection hits). Do not interpret latency results. "
            "Choose a fixed input where the candidate expert is actually routed, "
            "then rerun."
        )

    measurements = {"A": [], "B": [], "C": []}
    raw_rows = []
    max_diffs = []
    all_staged_hits = []

    for trial, order in enumerate(ORDERS, start=1):
        print(f"\nTrial {trial}/{TRIALS}: " +
              " -> ".join(f"{m} ({LABELS[m]})" for m in order))
        trial_logits = {}
        trial_metrics = {}
        trial_elapsed = {}

        for position, mode in enumerate(order, start=1):
            logits, elapsed_ms, staging = run_mode(
                model, target_ids, expert_id, mode
            )
            sync()
            measurements[mode].append(elapsed_ms)
            trial_elapsed[mode] = elapsed_ms
            trial_logits[mode] = logits
            metrics = staging_metrics(staging)
            trial_metrics[mode] = metrics

            print(f"  {LABELS[mode]}: {elapsed_ms / 1000:.3f} s")
            if staging is not None:
                print(
                    f"    CPU prep={metrics['cpu_prepare_ms']} ms; "
                    f"enqueue={metrics['enqueue_ms']} ms; "
                    f"GPU copy={metrics['gpu_copy_ms']} ms; "
                    f"wait={metrics['wait_ms']} ms; "
                    f"staged hits={metrics['staged_hits']}; "
                    f"fallbacks={metrics['fallback_reads']}"
                )

        baseline_logits = trial_logits["A"]
        for mode in ("B", "C"):
            diff = (baseline_logits - trial_logits[mode]).abs().max().item()
            max_diffs.append(diff)
            passed = diff <= LOGIT_ATOL
            print(
                f"  A vs {mode}: max logit difference={diff:.8g}; "
                f"within atol={LOGIT_ATOL}: {passed}"
            )
            if mode == "C":
                hits = trial_metrics["C"]["staged_hits"]
                all_staged_hits.append(hits)
                if not hits:
                    raise RuntimeError(
                        f"Trial {trial}: staged weights were not used. "
                        "Latency result is invalid for evaluating weight staging."
                    )

        for mode in order:
            metrics = trial_metrics[mode]
            diff = "" if mode == "A" else (
                trial_logits["A"] - trial_logits[mode]
            ).abs().max().item()
            raw_rows.append({
                "trial": trial,
                "order_position": order.index(mode) + 1,
                "mode": mode,
                "mode_label": LABELS[mode],
                "layer_id": LAYER_ID,
                "expert_id": expert_id,
                "prompt": PROMPT,
                "input_token_count": int(target_ids.shape[1]),
                "elapsed_ms": trial_elapsed[mode],
                **metrics,
                "max_abs_logit_diff_vs_baseline": diff,
                "logit_atol": LOGIT_ATOL,
                "logits_within_tolerance": "" if mode == "A" else diff <= LOGIT_ATOL,
            })

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(raw_rows[0].keys())
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)

    medians = {m: statistics.median(measurements[m]) for m in ("A", "B", "C")}
    print("\n=== THREE-MODE PREFETCH BENCHMARK ===")
    print(f"Trials per mode: {TRIALS}")
    print(f"Fixed input token count: {target_ids.shape[1]}")
    print(f"Frozen candidate: layer={LAYER_ID}, expert={expert_id}")
    for mode in ("A", "B", "C"):
        vals = measurements[mode]
        mad = statistics.median(abs(x - medians[mode]) for x in vals)
        print(
            f"{mode} — {LABELS[mode]}: median={medians[mode] / 1000:.3f} s; "
            f"range={min(vals) / 1000:.3f}–{max(vals) / 1000:.3f} s; "
            f"MAD={mad / 1000:.3f} s"
        )

    for mode in ("B", "C"):
        change_pct = (medians[mode] / medians["A"] - 1) * 100
        print(f"{LABELS[mode]} vs baseline median: {change_pct:+.2f}%")

    print(f"Maximum observed logit difference: {max(max_diffs):.8g}")
    print(f"Staged projection hits per measured C trial: {all_staged_hits}")
    print(f"Raw measurements saved to: {RESULTS_PATH}")
    print(
        "Caveat: this is a single-token forward diagnostic. It does not "
        "establish a multi-token generation speedup or prove general overlap."
    )


if __name__ == "__main__":
    main()
