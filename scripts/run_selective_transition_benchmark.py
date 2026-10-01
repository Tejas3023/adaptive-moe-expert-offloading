import json
import sys
from pathlib import Path

# ---------------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.prefetch.selective_prefetcher import (
    SelectiveTransitionPrefetcher
)

from src.cache.multi_tier_cache import (
    MultiTierExpertCache
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TRACE_PATH = (
    PROJECT_ROOT
    / "results"
    / "traces"
    / "wikitext_10_samples.jsonl"
)

GPU_CAPACITY = 256
CPU_CAPACITY = 170

MAX_PREFETCH_SIZE = 8

CONFIDENCE_THRESHOLDS = [
    0.000,
    0.025,
    0.050,
    0.075,
    0.100,
    0.105,
]


# ---------------------------------------------------------
# Load trace
# ---------------------------------------------------------

def load_trace(path):

    records = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


# ---------------------------------------------------------
# Benchmark one threshold
# ---------------------------------------------------------

def run_benchmark(records, threshold):

    predictor = SelectiveTransitionPrefetcher(
        num_experts=64,
        top_k=8,
        max_prefetch_size=MAX_PREFETCH_SIZE,
        confidence_threshold=threshold,
    )

    cache = MultiTierExpertCache(
        gpu_capacity=GPU_CAPACITY,
        cpu_capacity=CPU_CAPACITY,
    )

    # -----------------------------------------------------
    # Predictor-level metrics
    # -----------------------------------------------------

    total_actual = 0
    total_predictions = 0
    correct_predictions = 0

    # -----------------------------------------------------
    # Cache-level metrics
    # -----------------------------------------------------

    prefetches_issued = 0
    useful_prefetches = 0
    wasted_prefetches = 0

    # A prefetched expert remains in this set until it is
    # actually consumed or becomes unavailable.
    outstanding_prefetches = set()

    # -----------------------------------------------------
    # Track previous token separately for each layer
    # -----------------------------------------------------

    previous_experts = {}

    current_token = None

    # -----------------------------------------------------
    # Process trace
    # -----------------------------------------------------

    for record in records:

        layer_id = record["layer_id"]
        sample_id = record["sample_id"]
        token_position = record["token_position"]

        actual_experts = set(
            record["selected_experts"]
        )

        total_actual += len(
            actual_experts
        )

        token_key = (
            sample_id,
            token_position
        )

        # -------------------------------------------------
        # Before processing the current layer/token,
        # predict using the previous token's experts.
        # -------------------------------------------------

        previous = previous_experts.get(
            layer_id,
            []
        )

        predictions = predictor.predict(
            layer_id,
            previous
        )

        # -------------------------------------------------
        # Predictor metrics
        #
        # This measures whether the prediction itself was
        # correct for the current routing decision.
        # -------------------------------------------------

        for expert_id in predictions:

            total_predictions += 1

            if expert_id in actual_experts:

                correct_predictions += 1

        # -------------------------------------------------
        # Issue prefetches
        # -------------------------------------------------

        for expert_id in predictions:

            key = (
                layer_id,
                expert_id
            )

            # If already in GPU, this is not a new
            # prefetch opportunity.
            if key in cache.cached_gpu():
                continue

            prefetches_issued += 1

            cache.prefetch(
                layer_id,
                expert_id
            )

            # The prediction is now tracked as an
            # outstanding prefetch.
            if key in cache.cached_gpu():

                outstanding_prefetches.add(
                    key
                )

        # -------------------------------------------------
        # Request actual experts
        # -------------------------------------------------

        for expert_id in actual_experts:

            key = (
                layer_id,
                expert_id
            )

            # -------------------------------------------------
            # If this expert was prefetched and is still
            # available, count it as a useful prefetch.
            # -------------------------------------------------

            if key in outstanding_prefetches:

                useful_prefetches += 1

                outstanding_prefetches.remove(
                    key
                )

            cache.request(
                layer_id,
                expert_id
            )

        # -------------------------------------------------
        # Update previous experts AFTER observing current
        # experts.
        #
        # This maintains online/no-leakage behavior.
        # -------------------------------------------------

        previous_experts[layer_id] = list(
            actual_experts
        )

        predictor.update(
            layer_id,
            list(actual_experts)
        )

        # -------------------------------------------------
        # At token boundaries, identify prefetched experts
        # that disappeared from both cache tiers.
        #
        # These are counted as wasted.
        # -------------------------------------------------

        next_token = token_key

        if current_token is None:

            current_token = next_token

        elif next_token != current_token:

            current_token = next_token

    # -----------------------------------------------------
    # Remaining outstanding prefetches were never consumed.
    # -----------------------------------------------------

    wasted_prefetches = len(
        outstanding_prefetches
    )

    stats = cache.statistics()

    # -----------------------------------------------------
    # Predictor metrics
    # -----------------------------------------------------

    precision = (
        correct_predictions / total_predictions
        if total_predictions > 0
        else 0.0
    )

    coverage = (
        correct_predictions / total_actual
        if total_actual > 0
        else 0.0
    )

    # -----------------------------------------------------
    # Cache metrics
    # -----------------------------------------------------

    prefetch_utilization = (
        useful_prefetches / prefetches_issued
        if prefetches_issued > 0
        else 0.0
    )

    wasted_rate = (
        wasted_prefetches / prefetches_issued
        if prefetches_issued > 0
        else 0.0
    )

    return {
        "threshold": threshold,

        "predictions": total_predictions,
        "correct": correct_predictions,

        "precision": precision,
        "coverage": coverage,

        "prefetches_issued": prefetches_issued,
        "useful_prefetches": useful_prefetches,
        "wasted_prefetches": wasted_prefetches,

        "prefetch_utilization": prefetch_utilization,
        "wasted_rate": wasted_rate,

        "gpu_hits": stats["gpu_hits"],
        "cpu_hits": stats["cpu_hits"],
        "disk_fetches": stats["disk_fetches"],

        "gpu_evictions": stats[
            "gpu_evictions"
        ],

        "cpu_evictions": stats[
            "cpu_evictions"
        ],

        "total_latency_ms": stats[
            "total_latency_ms"
        ],

        "average_latency_ms": stats[
            "average_latency_ms"
        ],
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 110)
    print("Selective Transition Prefetch Benchmark")
    print("=" * 110)

    print()
    print(f"Trace: {TRACE_PATH}")
    print(f"GPU capacity: {GPU_CAPACITY}")
    print(f"CPU capacity: {CPU_CAPACITY}")
    print(
        f"Maximum prefetch size: "
        f"{MAX_PREFETCH_SIZE}"
    )

    print()

    records = load_trace(
        TRACE_PATH
    )

    print(
        f"Loaded {len(records):,} routing records."
    )

    print()

    results = []

    for threshold in CONFIDENCE_THRESHOLDS:

        print(
            f"Running threshold = "
            f"{threshold:.3f} ..."
        )

        result = run_benchmark(
            records,
            threshold
        )

        results.append(
            result
        )

        print(
            f"  Predictions: "
            f"{result['predictions']:,}"
        )

        print(
            f"  Correct: "
            f"{result['correct']:,}"
        )

        print(
            f"  Precision: "
            f"{result['precision'] * 100:.2f}%"
        )

        print(
            f"  Coverage: "
            f"{result['coverage'] * 100:.2f}%"
        )

        print(
            f"  Prefetches issued: "
            f"{result['prefetches_issued']:,}"
        )

        print(
            f"  Useful prefetches: "
            f"{result['useful_prefetches']:,}"
        )

        print(
            f"  Wasted prefetches: "
            f"{result['wasted_prefetches']:,}"
        )

        print(
            f"  Prefetch utilization: "
            f"{result['prefetch_utilization'] * 100:.2f}%"
        )

        print(
            f"  Disk fetches: "
            f"{result['disk_fetches']:,}"
        )

        print(
            f"  Average latency: "
            f"{result['average_latency_ms']:.4f} ms"
        )

        print()

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)

    print(
        f"{'Threshold':>10} "
        f"{'Pred':>10} "
        f"{'Correct':>10} "
        f"{'Precision':>11} "
        f"{'Coverage':>11} "
        f"{'Prefetch':>11} "
        f"{'Useful':>10} "
        f"{'Wasted':>10} "
        f"{'Util.':>9} "
        f"{'Disk':>10} "
        f"{'Avg Latency':>14}"
    )

    print("-" * 120)

    for r in results:

        print(
            f"{r['threshold']:>10.3f} "
            f"{r['predictions']:>10,} "
            f"{r['correct']:>10,} "
            f"{r['precision'] * 100:>10.2f}% "
            f"{r['coverage'] * 100:>10.2f}% "
            f"{r['prefetches_issued']:>11,} "
            f"{r['useful_prefetches']:>10,} "
            f"{r['wasted_prefetches']:>10,} "
            f"{r['prefetch_utilization'] * 100:>8.2f}% "
            f"{r['disk_fetches']:>10,} "
            f"{r['average_latency_ms']:>13.4f}"
        )

    print("=" * 120)


if __name__ == "__main__":
    main()