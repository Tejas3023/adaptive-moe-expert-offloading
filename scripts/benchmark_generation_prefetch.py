"""
Real OLMoE multi-token generation benchmark for three prefetch modes.

A: baseline generation
B: stage a fixed predicted expert concurrently, but do not install its weights
C: stage and install the fixed predicted expert's weights

This benchmark intentionally uses greedy decoding and a fixed prompt. It records
end-to-end generation latency, generated token IDs, score-logit differences,
staging metrics, staged projection hits, and peak CUDA memory.

Run from the project root:
    python -m scripts.benchmark_generation_prefetch

The first run is expensive on a small GPU with CPU/disk offloading.
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
    start_staging,
    install_staged_maps,
    restore_staged_maps,
    sync,
)

LAYER_ID = 2
PREFETCH_START_LAYER = 1
PROMPT = "The future of artificial intelligence is"
MAX_NEW_TOKENS = 3
TRIALS = 7
LOGIT_ATOL = 1e-5
RESULTS_PATH = Path("results") / "real_prefetch_generation.csv"

LABELS = {
    "A": "Baseline",
    "B": "Transfer only",
    "C": "Staged weights",
}

ORDERS = [
    ("A", "B", "C"),
    ("B", "C", "A"),
    ("C", "A", "B"),
    ("A", "C", "B"),
    ("C", "B", "A"),
    ("B", "A", "C"),
    ("A", "B", "C"),
]


def generate_once(model, input_ids, max_new_tokens):
    with torch.inference_mode():
        return model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=None,
        )


def timed_baseline(model, input_ids):
    sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    output = generate_once(model, input_ids, MAX_NEW_TOKENS)
    sync()
    elapsed_ms = (time.perf_counter() - start) * 1000
    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if torch.cuda.is_available() else 0.0
    )
    return output, elapsed_ms, peak_memory_mb, None


def timed_prefetch_mode(model, input_ids, expert_id, install):
    """
    Stage on entry to layer 1. At layer 2 either install the staged maps (C),
    or wait for the copy to finish without installing (B). Restore maps after
    layer 2 so no staged map leaks into the next forward.
    """
    layer1 = model.model.layers[PREFETCH_START_LAYER]
    layer2 = model.model.layers[LAYER_ID]
    modules = {
        "gate_proj": model.model.layers[LAYER_ID].mlp.experts[expert_id].gate_proj,
        "up_proj": model.model.layers[LAYER_ID].mlp.experts[expert_id].up_proj,
        "down_proj": model.model.layers[LAYER_ID].mlp.experts[expert_id].down_proj,
    }
    state = {"staging": None, "all_staging": []}

    def start_hook(module, inputs):
        # Each generation forward may have a different sequence length/cache state.
        # The prior copy must be complete before its buffers can be released.
        previous = state["staging"]
        if previous is not None:
            previous["copy_end"].synchronize()
            restore_staged_maps(previous)
        staging = start_staging(model, expert_id)
        staging["module_by_name"] = modules
        state["staging"] = staging
        state["all_staging"].append(staging)

    def target_pre_hook(module, inputs):
        staging = state["staging"]
        if staging is None:
            raise RuntimeError("Layer 2 reached without staging being started.")
        if install:
            install_staged_maps(staging)
        else:
            wait_start = time.perf_counter()
            staging["copy_end"].synchronize()
            staging["wait_ms"] = (time.perf_counter() - wait_start) * 1000
            staging["copy_ms"] = staging["copy_start"].elapsed_time(
                staging["copy_end"]
            )

    def target_post_hook(module, inputs, output):
        staging = state["staging"]
        if staging is not None:
            restore_staged_maps(staging)

    h1 = layer1.register_forward_pre_hook(start_hook)
    h2 = layer2.register_forward_pre_hook(target_pre_hook)
    h3 = layer2.register_forward_hook(target_post_hook)

    sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    output = None
    try:
        output = generate_once(model, input_ids, MAX_NEW_TOKENS)
        sync()
        elapsed_ms = (time.perf_counter() - start) * 1000
    finally:
        h1.remove()
        h2.remove()
        h3.remove()
        for staging in state["all_staging"]:
            staging["copy_end"].synchronize()
            restore_staged_maps(staging)

    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if torch.cuda.is_available() else 0.0
    )
    return output, elapsed_ms, peak_memory_mb, state["all_staging"]


def run_mode(model, input_ids, expert_id, mode):
    if mode == "A":
        return timed_baseline(model, input_ids)
    if mode == "B":
        output, elapsed, peak, staging = timed_prefetch_mode(
            model, input_ids, expert_id, install=False
        )
        return output, elapsed, peak, staging
    if mode == "C":
        output, elapsed, peak, staging = timed_prefetch_mode(
            model, input_ids, expert_id, install=True
        )
        return output, elapsed, peak, staging
    raise ValueError(f"Unknown mode: {mode}")


def output_tokens(output, input_len):
    return output.sequences[0, input_len:].detach().cpu().tolist()


def max_score_diff(reference, candidate):
    if len(reference.scores) != len(candidate.scores):
        return float("inf")
    if not reference.scores:
        return 0.0
    diffs = []
    for ref_score, cand_score in zip(reference.scores, candidate.scores):
        # Scores are on the same device for each comparison; synchronize before
        # extracting the scalar result.
        diffs.append((ref_score.float() - cand_score.float()).abs().max().item())
    return max(diffs)


def staging_summary(staging_list):
    if not staging_list:
        return {
            "forward_count": 0,
            "cpu_prepare_ms_sum": 0.0,
            "gpu_copy_ms_sum": 0.0,
            "wait_ms_sum": 0.0,
            "staged_hits": 0,
            "fallback_reads": 0,
        }

    cpu_prep = sum(s.get("cpu_prepare_ms", 0.0) for s in staging_list)
    gpu_copy = 0.0
    wait = 0.0
    hits = 0
    fallbacks = 0
    for staging in staging_list:
        copy_ms = staging.get("copy_ms")
        if copy_ms is None:
            copy_ms = staging["copy_start"].elapsed_time(staging["copy_end"])
        gpu_copy += copy_ms
        wait += staging.get("wait_ms") or 0.0
        for wrapper in staging.get("wrappers", {}).values():
            hits += wrapper.staged_hits
            fallbacks += wrapper.fallback_reads

    return {
        "forward_count": len(staging_list),
        "cpu_prepare_ms_sum": cpu_prep,
        "gpu_copy_ms_sum": gpu_copy,
        "wait_ms_sum": wait,
        "staged_hits": hits,
        "fallback_reads": fallbacks,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    print("Loading tokenizer and OLMoE...")
    tokenizer = load_tokenizer()
    model = load_olmoe_model()
    model.eval()

    encoded = tokenizer(PROMPT, return_tensors="pt")
    input_ids = encoded["input_ids"].to("cuda")
    input_len = input_ids.shape[1]
    print(f"Prompt tokens: {input_len}; max new tokens: {MAX_NEW_TOKENS}")
    print(f"Trials per mode: {TRIALS}")

    predictor = LayerAwareHistoryPrefetcher(history_size=8, prefetch_size=1)
    print("Priming predictor from the fixed prompt...")
    prime_predictor(model, predictor, input_ids)
    candidates = predictor.predict(LAYER_ID)
    if not candidates:
        raise RuntimeError("Predictor produced no expert candidate.")
    expert_id = candidates[0]
    print(f"Frozen candidate: layer={LAYER_ID}, expert={expert_id}")

    print("\nWarming up all modes...")
    warmups = {}
    for mode in ("A", "B", "C"):
        output, elapsed, peak, staging = run_mode(
            model, input_ids, expert_id, mode
        )
        warmups[mode] = output
        print(
            f"  {LABELS[mode]}: {elapsed / 1000:.3f} s; "
            f"tokens={output_tokens(output, input_len)}; peak={peak:.1f} MiB"
        )

    base_warm_tokens = output_tokens(warmups["A"], input_len)
    for mode in ("B", "C"):
        diff = max_score_diff(warmups["A"], warmups[mode])
        tokens_match = base_warm_tokens == output_tokens(warmups[mode], input_len)
        print(
            f"Warm-up A vs {mode}: generated tokens match={tokens_match}; "
            f"max score difference={diff:.8g}"
        )

    measurements = {m: [] for m in ("A", "B", "C")}
    rows = []
    max_diffs = []
    token_match_all = {"B": True, "C": True}
    all_c_hits = []

    for trial, order in enumerate(ORDERS, start=1):
        print(f"\nTrial {trial}/{TRIALS}: " +
              " -> ".join(f"{m} ({LABELS[m]})" for m in order))
        trial_results = {}
        trial_meta = {}

        for position, mode in enumerate(order, start=1):
            output, elapsed_ms, peak_mb, staging = run_mode(
                model, input_ids, expert_id, mode
            )
            measurements[mode].append(elapsed_ms)
            trial_results[mode] = output
            summary = staging_summary(staging)
            trial_meta[mode] = {
                "elapsed_ms": elapsed_ms,
                "peak_gpu_memory_mb": peak_mb,
                **summary,
            }
            print(
                f"  {LABELS[mode]}: {elapsed_ms / 1000:.3f} s; "
                f"generated={output_tokens(output, input_len)}; "
                f"peak GPU memory={peak_mb:.1f} MiB"
            )
            if mode in ("B", "C"):
                print(
                    f"    staging forwards={summary['forward_count']}; "
                    f"CPU prep sum={summary['cpu_prepare_ms_sum']:.2f} ms; "
                    f"GPU copy sum={summary['gpu_copy_ms_sum']:.2f} ms; "
                    f"wait sum={summary['wait_ms_sum']:.2f} ms; "
                    f"staged hits={summary['staged_hits']}; "
                    f"fallbacks={summary['fallback_reads']}"
                )

        base = trial_results["A"]
        base_tokens = output_tokens(base, input_len)
        for mode in ("B", "C"):
            candidate = trial_results[mode]
            candidate_tokens = output_tokens(candidate, input_len)
            tokens_match = base_tokens == candidate_tokens
            score_diff = max_score_diff(base, candidate)
            within_tol = score_diff <= LOGIT_ATOL
            token_match_all[mode] = token_match_all[mode] and tokens_match
            max_diffs.append(score_diff)
            print(
                f"  A vs {mode}: tokens match={tokens_match}; "
                f"max generation-score difference={score_diff:.8g}; "
                f"within atol={LOGIT_ATOL}: {within_tol}"
            )

        c_hits = trial_meta["C"]["staged_hits"]
        all_c_hits.append(c_hits)
        if c_hits == 0:
            print("  WARNING: no staged projection hits in this C trial.")

        for mode in ("A", "B", "C"):
            output = trial_results[mode]
            base = trial_results["A"]
            score_diff = "" if mode == "A" else max_score_diff(base, output)
            tokens_match = "" if mode == "A" else (
                output_tokens(base, input_len) == output_tokens(output, input_len)
            )
            row = {
                "trial": trial,
                "order_position": order.index(mode) + 1,
                "mode": mode,
                "mode_label": LABELS[mode],
                "layer_id": LAYER_ID,
                "expert_id": expert_id,
                "prompt": PROMPT,
                "input_tokens": input_len,
                "max_new_tokens": MAX_NEW_TOKENS,
                "elapsed_ms": trial_meta[mode]["elapsed_ms"],
                "peak_gpu_memory_mb": trial_meta[mode]["peak_gpu_memory_mb"],
                "generated_token_ids": " ".join(
                    map(str, output_tokens(output, input_len))
                ),
                "forward_count": trial_meta[mode]["forward_count"],
                "cpu_prepare_ms_sum": trial_meta[mode]["cpu_prepare_ms_sum"],
                "gpu_copy_ms_sum": trial_meta[mode]["gpu_copy_ms_sum"],
                "wait_ms_sum": trial_meta[mode]["wait_ms_sum"],
                "staged_hits": trial_meta[mode]["staged_hits"],
                "fallback_reads": trial_meta[mode]["fallback_reads"],
                "max_score_diff_vs_baseline": score_diff,
                "generated_tokens_match_baseline": tokens_match,
                "score_atol": LOGIT_ATOL,
                "scores_within_tolerance": "" if mode == "A" else score_diff <= LOGIT_ATOL,
            }
            rows.append(row)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    medians = {m: statistics.median(measurements[m]) for m in ("A", "B", "C")}
    print("\n=== MULTI-TOKEN GENERATION BENCHMARK SUMMARY ===")
    print(f"Trials per mode: {TRIALS}; max new tokens: {MAX_NEW_TOKENS}")
    for mode in ("A", "B", "C"):
        vals = measurements[mode]
        mad = statistics.median(abs(x - medians[mode]) for x in vals)
        print(
            f"{mode} — {LABELS[mode]}: median={medians[mode] / 1000:.3f} s; "
            f"range={min(vals) / 1000:.3f}–{max(vals) / 1000:.3f} s; "
            f"MAD={mad / 1000:.3f} s"
        )
    for mode in ("B", "C"):
        print(
            f"{LABELS[mode]} vs baseline median: "
            f"{(medians[mode] / medians['A'] - 1) * 100:+.2f}%"
        )
        print(f"Generated token IDs match baseline in all trials: {token_match_all[mode]}")
    print(f"Maximum observed score difference: {max(max_diffs):.8g}")
    print(f"Staged projection hits per C trial: {all_c_hits}")
    print(f"Raw measurements saved to: {RESULTS_PATH}")
    print(
        "Caveat: fixed prompt and greedy decoding; this is a small-scale "
        "generation benchmark, not a broad performance claim."
    )


if __name__ == "__main__":
    main()
