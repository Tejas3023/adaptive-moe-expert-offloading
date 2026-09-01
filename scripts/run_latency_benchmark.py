import json
from collections import defaultdict

from src.cache.latency_cache import LatencyAwareCache
from src.cache.cost_model import CacheCostModel
from src.prefetch.history_prefetcher import HistoryPrefetcher


TRACE_FILE = (
    "results/traces/wikitext_10_samples.jsonl"
)

CACHE_CAPACITIES = [
    8,
    16,
    32,
    64,
]

HISTORY_SIZE = 8
PREFETCH_SIZE = 1


def load_traces():

    data = defaultdict(
        lambda: defaultdict(dict)
    )

    with open(
        TRACE_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            trace = json.loads(line)

            sample = trace["sample_id"]
            layer = trace["layer_id"]
            token = trace["token_position"]

            data[sample][layer][token] = set(
                trace["selected_experts"]
            )

    return data


def run_baseline(
    data,
    capacity,
    cost_model,
):

    cache = LatencyAwareCache(
        capacity=capacity,
        cost_model=cost_model,
    )

    for sample_layers in data.values():

        for layer_id, token_layers in sample_layers.items():

            for token in sorted(token_layers):

                actual = token_layers[token]

                for expert_id in actual:

                    cache.request(
                        layer_id,
                        expert_id,
                    )

    return cache.statistics()


def run_prefetch(
    data,
    capacity,
    cost_model,
):

    cache = LatencyAwareCache(
        capacity=capacity,
        cost_model=cost_model,
    )

    total_correct = 0
    total_predictions = 0
    total_actual = 0

    for sample_layers in data.values():

        for layer_id, token_layers in sample_layers.items():

            predictor = HistoryPrefetcher(
                history_size=HISTORY_SIZE,
                prefetch_size=PREFETCH_SIZE,
            )

            for token in sorted(token_layers):

                actual = token_layers[token]

                # Predict before seeing current token.
                predictions = set(
                    predictor.predict()
                )

                # Prediction metrics.
                total_correct += len(
                    predictions & actual
                )

                total_predictions += len(
                    predictions
                )

                total_actual += len(
                    actual
                )

                # Prefetch predicted experts.
                for expert_id in predictions:

                    cache.prefetch(
                        layer_id,
                        expert_id,
                    )

                # Actual requests.
                for expert_id in actual:

                    cache.request(
                        layer_id,
                        expert_id,
                    )

                # Reveal routing.
                predictor.observe(
                    actual
                )

    stats = cache.statistics()

    stats["prediction_precision"] = (
        total_correct /
        total_predictions *
        100
        if total_predictions
        else 0.0
    )

    stats["prediction_coverage"] = (
        total_correct /
        total_actual *
        100
        if total_actual
        else 0.0
    )

    return stats


def print_result(
    result,
):

    print(
        f"{result['capacity']:<10}"
        f"{result['hits']:<10}"
        f"{result['misses']:<10}"
        f"{result['hit_rate_percent']:<12.2f}"
        f"{result['evictions']:<12}"
        f"{result['prefetches']:<12}"
        f"{result['total_latency_ms']:<15.2f}"
    )


def main():

    print("=" * 100)
    print("LATENCY-AWARE CACHE BENCHMARK")
    print("=" * 100)

    print()
    print("Trace file:")
    print(TRACE_FILE)

    print()
    print("Cost model:")
    print("GPU hit:       0.05 ms")
    print("CPU → GPU:     1.00 ms")
    print("Prefetch:      1.00 ms")

    data = load_traces()

    cost_model = CacheCostModel(
        gpu_hit_ms=0.05,
        cpu_to_gpu_ms=1.0,
        disk_to_gpu_ms=10.0,
        prefetch_ms=1.0,
    )

    # ================================================================
    # BASELINE
    # ================================================================

    print()
    print("=" * 100)
    print("BASELINE LRU")
    print("=" * 100)

    print()

    print(
        f"{'Cache':<10}"
        f"{'Hits':<10}"
        f"{'Misses':<10}"
        f"{'Hit Rate':<12}"
        f"{'Evictions':<12}"
        f"{'Prefetches':<12}"
        f"{'Latency ms':<15}"
    )

    print("-" * 90)

    for capacity in CACHE_CAPACITIES:

        result = run_baseline(
            data,
            capacity,
            cost_model,
        )

        print_result(result)

    # ================================================================
    # PREFETCH
    # ================================================================

    print()
    print("=" * 100)
    print(
        f"HISTORY PREFETCH "
        f"(window={HISTORY_SIZE}, "
        f"prefetch={PREFETCH_SIZE})"
    )
    print("=" * 100)

    print()

    print(
        f"{'Cache':<10}"
        f"{'Hits':<10}"
        f"{'Misses':<10}"
        f"{'Hit Rate':<12}"
        f"{'Evictions':<12}"
        f"{'Prefetches':<12}"
        f"{'Latency ms':<15}"
        f"{'Precision':<12}"
        f"{'Coverage':<12}"
    )

    print("-" * 110)

    for capacity in CACHE_CAPACITIES:

        result = run_prefetch(
            data,
            capacity,
            cost_model,
        )

        print(
            f"{result['capacity']:<10}"
            f"{result['hits']:<10}"
            f"{result['misses']:<10}"
            f"{result['hit_rate_percent']:<12.2f}"
            f"{result['evictions']:<12}"
            f"{result['prefetches']:<12}"
            f"{result['total_latency_ms']:<15.2f}"
            f"{result['prediction_precision']:<12.2f}"
            f"{result['prediction_coverage']:<12.2f}"
        )

    print()
    print("=" * 100)
    print("LATENCY BENCHMARK COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()