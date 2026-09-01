import json
from collections import defaultdict

from src.cache.multi_tier_cache import MultiTierExpertCache
from src.prefetch.layer_aware_prefetcher import LayerAwareHistoryPrefetcher


TRACE_FILE = "results/traces/wikitext_10_samples.jsonl"

GPU_CAPACITY = 256
CPU_CAPACITY = 170

HISTORY_SIZE = 8
PREFETCH_SIZES = [1, 2, 4, 8]


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


def run_experiment(
    data,
    prefetch_size,
):

    cache = MultiTierExpertCache(
        gpu_capacity=GPU_CAPACITY,
        cpu_capacity=CPU_CAPACITY,
    )

    total_correct = 0
    total_predictions = 0
    total_actual = 0

    for sample_layers in data.values():

        for layer_id, token_layers in sample_layers.items():

            predictor = LayerAwareHistoryPrefetcher(
                history_size=HISTORY_SIZE,
                prefetch_size=prefetch_size,
            )

            for token in sorted(token_layers):

                actual = token_layers[token]

                # Predict before seeing the current token.
                predictions = set(
                    predictor.predict(layer_id)
                )

                total_correct += len(
                    predictions & actual
                )

                total_predictions += len(
                    predictions
                )

                total_actual += len(actual)

                # Prefetch predicted experts.
                for expert_id in predictions:

                    cache.prefetch(
                        layer_id,
                        expert_id,
                    )

                # Process actual expert requests.
                for expert_id in actual:

                    cache.request(
                        layer_id,
                        expert_id,
                    )

                # Reveal current routing.
                predictor.observe(
                    layer_id,
                    actual,
                )

    stats = cache.statistics()

    precision = (
        total_correct /
        total_predictions *
        100
        if total_predictions
        else 0.0
    )

    coverage = (
        total_correct /
        total_actual *
        100
        if total_actual
        else 0.0
    )

    stats["prediction_precision"] = precision
    stats["prediction_coverage"] = coverage
    stats["correct_predictions"] = total_correct

    return stats


def main():

    print("=" * 100)
    print("LAYER-AWARE PREFETCH BENCHMARK")
    print("=" * 100)

    print()
    print("Trace file:")
    print(TRACE_FILE)

    print()
    print("GPU capacity:", GPU_CAPACITY)
    print("CPU capacity:", CPU_CAPACITY)
    print("History window:", HISTORY_SIZE)
    print("Prefetch sizes:", PREFETCH_SIZES)

    print()
    print("Loading traces...")

    data = load_traces()

    print("Traces loaded.")

    print()
    print("=" * 100)
    print("RESULTS")
    print("=" * 100)

    print()

    print(
        f"{'Prefetch':<12}"
        f"{'GPU Hits':<12}"
        f"{'CPU Hits':<12}"
        f"{'Disk':<12}"
        f"{'Avg Latency':<15}"
        f"{'Precision':<12}"
        f"{'Coverage':<12}"
    )

    print("-" * 95)

    for prefetch_size in PREFETCH_SIZES:

        result = run_experiment(
            data,
            prefetch_size,
        )

        print(
            f"{prefetch_size:<12}"
            f"{result['gpu_hits']:<12}"
            f"{result['cpu_hits']:<12}"
            f"{result['disk_fetches']:<12}"
            f"{result['average_latency_ms']:<15.4f}"
            f"{result['prediction_precision']:<12.2f}"
            f"{result['prediction_coverage']:<12.2f}"
        )

    print()
    print("=" * 100)
    print("LAYER-AWARE PREFETCH BENCHMARK COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()