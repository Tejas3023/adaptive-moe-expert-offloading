
import time
import statistics
import torch

from src.modeling.olmoe_loader import load_olmoe_model, load_tokenizer
from src.prefetch.layer_aware_prefetcher import LayerAwareHistoryPrefetcher
from scripts.diagnose_async_prefetch import (
    prime_predictor,
    run_forward,
    run_async_trial,
    sync,
)

LAYER_ID = 2
PROMPT = "The future of artificial intelligence is"
TRIALS = 7


def timed_baseline(model, input_ids):
    sync()
    start = time.perf_counter()
    logits = run_forward(model, input_ids)
    sync()
    elapsed_ms = (time.perf_counter() - start) * 1000
    return logits, elapsed_ms


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    print("Loading OLMoE...")
    tokenizer = load_tokenizer()
    model = load_olmoe_model()
    model.eval()

    input_ids = tokenizer(
        PROMPT, return_tensors="pt"
    )["input_ids"].to("cuda")
    target_ids = input_ids[:, -1:]

    predictor = LayerAwareHistoryPrefetcher(
        history_size=8, prefetch_size=1
    )
    print("Priming predictor...")
    prime_predictor(model, predictor, input_ids)

    candidates = predictor.predict(LAYER_ID)
    if not candidates:
        raise RuntimeError("No predicted expert available.")

    expert_id = candidates[0]
    print(f"Predicted expert: {expert_id}")

    # Warm up both paths before collecting measurements.
    print("Warming up baseline...")
    baseline_logits, _ = timed_baseline(model, target_ids)

    print("Warming up async path...")
    async_logits, _, _ = run_async_trial(model, target_ids, expert_id)

    warmup_diff = (baseline_logits - async_logits).abs().max().item()
    print(f"Warm-up max logit difference: {warmup_diff:.8g}")

    results = []
    all_max_diffs = [warmup_diff]

    for trial in range(1, TRIALS + 1):
        # Alternate order to reduce systematic order bias.
        order = ["baseline", "async"] if trial % 2 else ["async", "baseline"]
        trial_results = {}

        print(f"\nTrial {trial}/{TRIALS} — order: {' -> '.join(order)}")

        for path in order:
            if path == "baseline":
                logits, elapsed_ms = timed_baseline(model, target_ids)
                trial_results["baseline_logits"] = logits
                trial_results["baseline_ms"] = elapsed_ms
            else:
                logits, elapsed_ms, staging = run_async_trial(
                    model, target_ids, expert_id
                )
                trial_results["async_logits"] = logits
                trial_results["async_ms"] = elapsed_ms

        diff = (
            trial_results["baseline_logits"]
            - trial_results["async_logits"]
        ).abs().max().item()
        all_max_diffs.append(diff)

        baseline_ms = trial_results["baseline_ms"]
        async_ms = trial_results["async_ms"]
        results.append((baseline_ms, async_ms))

        print(f"  Baseline: {baseline_ms / 1000:.3f} s")
        print(f"  Async:    {async_ms / 1000:.3f} s")
        print(f"  Async latency change: {(async_ms / baseline_ms - 1) * 100:+.2f}%")
        print(f"  Max logit difference: {diff:.8g}")
        print(
            "  Staged projection usage: "
            + ", ".join(
                f"{name}={staging['wrappers'][name].staged_hits} hit(s)/"
                f"{staging['wrappers'][name].fallback_reads} fallback(s)"
                for name in ("gate_proj", "up_proj", "down_proj")
            )
        )

    baseline_values = [r[0] for r in results]
    async_values = [r[1] for r in results]

    baseline_median = statistics.median(baseline_values)
    async_median = statistics.median(async_values)
    change_pct = (async_median / baseline_median - 1) * 100

    print("\n=== CONTROLLED ASYNC BENCHMARK ===")
    print(f"Trials: {TRIALS}")
    print(f"Predicted expert: {expert_id}")
    print(f"Baseline median: {baseline_median / 1000:.3f} s")
    print(f"Async median:    {async_median / 1000:.3f} s")
    print(f"Median latency change: {change_pct:+.2f}%")
    print(f"Maximum logit difference: {max(all_max_diffs):.8g}")

    baseline_mad = statistics.median(
        [abs(value - baseline_median) for value in baseline_values]
    )
    async_mad = statistics.median(
        [abs(value - async_median) for value in async_values]
    )
    print(f"Baseline median absolute deviation: {baseline_mad / 1000:.3f} s")
    print(f"Async median absolute deviation:    {async_mad / 1000:.3f} s")

    print(
        "\nNote: this is a small, single-machine benchmark. "
        "It tests the full async path, including CPU preparation and staging. "
        "It does not by itself prove that overlap caused any latency change."
    )


if __name__ == "__main__":
    main()
