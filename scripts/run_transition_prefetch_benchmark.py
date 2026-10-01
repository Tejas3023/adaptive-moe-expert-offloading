import os
import sys
import json
from collections import defaultdict


# ============================================================
# Make project root importable
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.append(PROJECT_ROOT)


# ============================================================
# Imports
# ============================================================

from src.prefetch.transition_prefetcher import (
    TransitionPrefetcher
)

from src.cache.multi_tier_cache import (
    MultiTierExpertCache
)


# ============================================================
# Configuration
# ============================================================

TRACE_FILE = (
    "results/traces/wikitext_10_samples.jsonl"
)

GPU_CAPACITY = 256
CPU_CAPACITY = 170

PREFETCH_SIZES = [1, 2, 4, 8]


# ============================================================
# Load trace
# ============================================================

def load_trace():

    records = []

    with open(
        TRACE_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


# ============================================================
# Group trace
# ============================================================

def group_trace(records):

    grouped = defaultdict(
        lambda: defaultdict(list)
    )

    for record in records:

        sample_id = record["sample_id"]
        layer_id = record["layer_id"]

        grouped[sample_id][layer_id].append(
            record
        )

    return grouped


# ============================================================
# Run benchmark
# ============================================================

def run_benchmark(
    grouped,
    prefetch_size
):

    # --------------------------------------------------------
    # Predictor
    # --------------------------------------------------------

    predictor = TransitionPrefetcher(
        num_experts=64,
        top_k=8,
        prefetch_size=prefetch_size
    )

    # --------------------------------------------------------
    # Multi-tier cache
    # --------------------------------------------------------

    cache = MultiTierExpertCache(
        gpu_capacity=GPU_CAPACITY,
        cpu_capacity=CPU_CAPACITY
    )

    # ========================================================
    # Prediction metrics
    # ========================================================

    correct_predictions = 0
    total_predictions = 0

    prediction_events = 0

    # ========================================================
    # Request metrics
    # ========================================================

    total_requests = 0

    # ========================================================
    # Prefetch usefulness metrics
    # ========================================================

    # Every key that has been prefetched and has not yet
    # been consumed by an actual request.
    outstanding_prefetches = set()

    useful_prefetches = 0
    wasted_prefetches = 0

    # A prefetch can be correct but still fail to help if
    # the expert gets evicted before it is requested.
    correct_prefetches = 0

    # ========================================================
    # Process samples
    # ========================================================

    for sample_id in sorted(grouped):

        # ----------------------------------------------------
        # Reset predictor at sample boundary.
        # ----------------------------------------------------

        predictor.reset()

        layers = grouped[sample_id]

        if not layers:
            continue

        # ----------------------------------------------------
        # Determine token count.
        # ----------------------------------------------------

        first_layer = min(layers.keys())

        num_tokens = len(
            layers[first_layer]
        )

        # ====================================================
        # Process tokens
        # ====================================================

        for token_index in range(num_tokens):

            # ------------------------------------------------
            # Process every layer
            # ------------------------------------------------

            for layer_id in sorted(layers):

                current_record = layers[
                    layer_id
                ][token_index]

                actual_experts = current_record[
                    "selected_experts"
                ]

                # =================================================
                # STEP 1 — PREDICT
                # =================================================

                predicted_experts = []

                if token_index > 0:

                    previous_record = layers[
                        layer_id
                    ][token_index - 1]

                    previous_experts = (
                        previous_record[
                            "selected_experts"
                        ]
                    )

                    predicted_experts = (
                        predictor.predict(
                            layer_id,
                            previous_experts
                        )
                    )

                    prediction_events += 1

                    # ---------------------------------------------
                    # Prediction accuracy
                    # ---------------------------------------------

                    predicted_set = set(
                        predicted_experts
                    )

                    actual_set = set(
                        actual_experts
                    )

                    correct_set = (
                        predicted_set
                        & actual_set
                    )

                    correct_predictions += len(
                        correct_set
                    )

                    total_predictions += len(
                        predicted_experts
                    )

                # =================================================
                # STEP 2 — PREFETCH
                # =================================================

                for expert_id in predicted_experts:

                    key = (
                        layer_id,
                        expert_id
                    )

                    # ------------------------------------------------
                    # If it was already outstanding from an earlier
                    # prediction, don't count another independent
                    # outstanding prefetch.
                    # ------------------------------------------------

                    if key not in outstanding_prefetches:

                        outstanding_prefetches.add(
                            key
                        )

                    cache.prefetch(
                        layer_id,
                        expert_id
                    )

                # =================================================
                # STEP 3 — REQUEST ACTUAL EXPERTS
                # =================================================

                for expert_id in actual_experts:

                    key = (
                        layer_id,
                        expert_id
                    )

                    # ------------------------------------------------
                    # Was this expert predicted?
                    # ------------------------------------------------

                    if key in outstanding_prefetches:

                        useful_prefetches += 1

                        # Remove it because the predicted expert
                        # has now been consumed.
                        outstanding_prefetches.remove(
                            key
                        )

                    # ------------------------------------------------
                    # Check whether this was a prediction match.
                    # ------------------------------------------------

                    if expert_id in predicted_experts:

                        correct_prefetches += 1

                    # ------------------------------------------------
                    # Actual cache request
                    # ------------------------------------------------

                    cache.request(
                        layer_id,
                        expert_id
                    )

                    total_requests += 1

                # =================================================
                # STEP 4 — REMOVE PREFETCHES THAT WERE EVICTED
                # =================================================
                #
                # A prediction is considered wasted if:
                #
                #     predicted
                #        ↓
                #     prefetched
                #        ↓
                #     no longer in GPU/CPU
                #        ↓
                #     never actually used
                #
                # We check outstanding keys after actual requests.
                # -------------------------------------------------

                still_outstanding = set()

                for key in outstanding_prefetches:

                    layer_key, expert_key = key

                    gpu_keys = cache.cached_gpu()
                    cpu_keys = cache.cached_cpu()

                    if (
                        key in gpu_keys
                        or key in cpu_keys
                    ):

                        still_outstanding.add(
                            key
                        )

                    else:

                        wasted_prefetches += 1

                outstanding_prefetches = (
                    still_outstanding
                )

                # =================================================
                # STEP 5 — UPDATE PREDICTOR
                # =================================================

                predictor.update(
                    layer_id,
                    actual_experts
                )

        # ====================================================
        # End of sample
        #
        # Anything still outstanding at the end of a sample
        # was never used by that sample.
        # ====================================================

        wasted_prefetches += len(
            outstanding_prefetches
        )

        outstanding_prefetches.clear()

    # ========================================================
    # Cache statistics
    # ========================================================

    stats = cache.statistics()

    # ========================================================
    # Prediction metrics
    # ========================================================

    if total_predictions > 0:

        precision = (
            correct_predictions
            / total_predictions
        ) * 100

    else:

        precision = 0.0

    if total_requests > 0:

        coverage = (
            correct_predictions
            / total_requests
        ) * 100

    else:

        coverage = 0.0

    # ========================================================
    # Prefetch utilization
    # ========================================================

    total_prefetches = stats[
        "prefetches"
    ]

    if total_prefetches > 0:

        prefetch_utilization = (
            useful_prefetches
            / total_prefetches
        ) * 100

    else:

        prefetch_utilization = 0.0

    # ========================================================
    # Wasted percentage
    # ========================================================

    if total_prefetches > 0:

        wasted_percentage = (
            wasted_prefetches
            / total_prefetches
        ) * 100

    else:

        wasted_percentage = 0.0

    # ========================================================
    # Print
    # ========================================================

    print(
        f"Prefetch {prefetch_size}: "
        f"GPU={stats['gpu_hits']}, "
        f"CPU={stats['cpu_hits']}, "
        f"Disk={stats['disk_fetches']}, "
        f"Prefetches={total_prefetches}, "
        f"Useful={useful_prefetches}, "
        f"Wasted={wasted_prefetches}, "
        f"Avg={stats['average_latency_ms']:.4f} ms, "
        f"Precision={precision:.2f}%, "
        f"Coverage={coverage:.2f}%, "
        f"Utilization={prefetch_utilization:.2f}%"
    )

    # ========================================================
    # Return
    # ========================================================

    return {

        "prefetch_size":
            prefetch_size,

        "total_requests":
            total_requests,

        "prediction_events":
            prediction_events,

        "correct_predictions":
            correct_predictions,

        "total_predictions":
            total_predictions,

        "precision":
            precision,

        "coverage":
            coverage,

        "gpu_hits":
            stats["gpu_hits"],

        "cpu_hits":
            stats["cpu_hits"],

        "disk_fetches":
            stats["disk_fetches"],

        "gpu_evictions":
            stats["gpu_evictions"],

        "cpu_evictions":
            stats["cpu_evictions"],

        "prefetches":
            total_prefetches,

        "prefetch_hits":
            stats["prefetch_hits"],

        "useful_prefetches":
            useful_prefetches,

        "wasted_prefetches":
            wasted_prefetches,

        "prefetch_utilization":
            prefetch_utilization,

        "wasted_percentage":
            wasted_percentage,

        "total_latency_ms":
            stats["total_latency_ms"],

        "average_latency_ms":
            stats["average_latency_ms"]
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 120)

    print(
        "Transition Prefetch + Multi-Tier Cache "
        "with Prefetch Usefulness Analysis"
    )

    print("=" * 120)

    print()

    print(
        f"Trace: {TRACE_FILE}"
    )

    print(
        f"GPU capacity: {GPU_CAPACITY} experts"
    )

    print(
        f"CPU capacity: {CPU_CAPACITY} experts"
    )

    print(
        f"Prefetch sizes: {PREFETCH_SIZES}"
    )

    print()

    # ========================================================
    # Load
    # ========================================================

    print("Loading trace...")

    records = load_trace()

    print(
        f"Loaded {len(records)} routing records."
    )

    grouped = group_trace(records)

    print(
        f"Loaded {len(grouped)} samples."
    )

    print()

    # ========================================================
    # Run
    # ========================================================

    results = []

    for prefetch_size in PREFETCH_SIZES:

        result = run_benchmark(
            grouped,
            prefetch_size
        )

        results.append(result)

    # ========================================================
    # Summary
    # ========================================================

    print()

    print("=" * 120)
    print("Summary")
    print("=" * 120)

    print(
        f"{'Prefetch':>10} "
        f"{'GPU':>10} "
        f"{'CPU':>10} "
        f"{'Disk':>10} "
        f"{'Prefetches':>12} "
        f"{'Useful':>10} "
        f"{'Wasted':>10} "
        f"{'Utilization':>13} "
        f"{'Avg Latency':>15}"
    )

    print("-" * 120)

    for result in results:

        print(
            f"{result['prefetch_size']:>10} "
            f"{result['gpu_hits']:>10} "
            f"{result['cpu_hits']:>10} "
            f"{result['disk_fetches']:>10} "
            f"{result['prefetches']:>12} "
            f"{result['useful_prefetches']:>10} "
            f"{result['wasted_prefetches']:>10} "
            f"{result['prefetch_utilization']:>12.2f}% "
            f"{result['average_latency_ms']:>14.4f} ms"
        )

    # ========================================================
    # Prediction summary
    # ========================================================

    print()

    print("=" * 120)
    print("Prediction Metrics")
    print("=" * 120)

    print(
        f"{'Prefetch':>10} "
        f"{'Correct':>12} "
        f"{'Predicted':>12} "
        f"{'Precision':>12} "
        f"{'Coverage':>12}"
    )

    print("-" * 120)

    for result in results:

        print(
            f"{result['prefetch_size']:>10} "
            f"{result['correct_predictions']:>12} "
            f"{result['total_predictions']:>12} "
            f"{result['precision']:>11.2f}% "
            f"{result['coverage']:>11.2f}%"
        )

    # ========================================================
    # Latency / cache summary
    # ========================================================

    print()

    print("=" * 120)
    print("Latency and Cache Metrics")
    print("=" * 120)

    for result in results:

        print()

        print(
            f"Prefetch size = "
            f"{result['prefetch_size']}"
        )

        print(
            f"  GPU hits:             "
            f"{result['gpu_hits']}"
        )

        print(
            f"  CPU hits:             "
            f"{result['cpu_hits']}"
        )

        print(
            f"  Disk fetches:         "
            f"{result['disk_fetches']}"
        )

        print(
            f"  GPU evictions:        "
            f"{result['gpu_evictions']}"
        )

        print(
            f"  CPU evictions:        "
            f"{result['cpu_evictions']}"
        )

        print(
            f"  Total prefetches:     "
            f"{result['prefetches']}"
        )

        print(
            f"  Useful prefetches:    "
            f"{result['useful_prefetches']}"
        )

        print(
            f"  Wasted prefetches:    "
            f"{result['wasted_prefetches']}"
        )

        print(
            f"  Prefetch utilization: "
            f"{result['prefetch_utilization']:.2f}%"
        )

        print(
            f"  Wasted percentage:    "
            f"{result['wasted_percentage']:.2f}%"
        )

        print(
            f"  Total latency:        "
            f"{result['total_latency_ms']:.2f} ms"
        )

        print(
            f"  Average latency:      "
            f"{result['average_latency_ms']:.4f} ms"
        )


if __name__ == "__main__":
    main()