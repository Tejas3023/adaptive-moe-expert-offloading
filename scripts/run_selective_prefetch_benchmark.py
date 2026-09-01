import json
from collections import defaultdict

from src.cache.expert_cache import ExpertCache
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

PREFETCH_SIZES = [
    1,
    2,
    4,
    8,
]


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
    cache_capacity,
    prefetch_size,
):

    cache = ExpertCache(
        cache_capacity
    )

    total_correct = 0
    total_predictions = 0
    total_actual = 0

    for sample_layers in data.values():

        for layer_id, token_layers in sample_layers.items():

            predictor = HistoryPrefetcher(
                history_size=HISTORY_SIZE,
                prefetch_size=prefetch_size,
            )

            for token in sorted(token_layers):

                actual = token_layers[token]

                # --------------------------------------------------
                # Predict before seeing current routing
                # --------------------------------------------------

                predictions = set(
                    predictor.predict()
                )

                # --------------------------------------------------
                # Measure prediction quality
                # --------------------------------------------------

                total_correct += len(
                    predictions & actual
                )

                total_predictions += len(
                    predictions
                )

                total_actual += len(
                    actual
                )

                # --------------------------------------------------
                # Prefetch predicted experts
                # --------------------------------------------------

                for expert_id in predictions:

                    cache.prefetch(
                        layer_id,
                        expert_id,
                    )

                # --------------------------------------------------
                # Process actual expert requests
                # --------------------------------------------------

                for expert_id in actual:

                    cache.request(
                        layer_id,
                        expert_id,
                    )

                # --------------------------------------------------
                # Reveal current routing
                # --------------------------------------------------

                predictor.observe(
                    actual
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
    print("SELECTIVE HISTORY PREFETCH BENCHMARK")
    print("=" * 100)

    print()
    print("Trace file:")
    print(TRACE_FILE)

    print()
    print(
        f"History window: {HISTORY_SIZE}"
    )

    print(
        f"Prefetch sizes: {PREFETCH_SIZES}"
    )

    print(
        f"Cache capacities: {CACHE_CAPACITIES}"
    )

    data = load_traces()

    for prefetch_size in PREFETCH_SIZES:

        print()
        print("=" * 100)

        print(
            f"PREFETCH SIZE = {prefetch_size}"
        )

        print("=" * 100)

        print()

        print(
            f"{'Cache':<10}"
            f"{'Hits':<10}"
            f"{'Misses':<10}"
            f"{'Hit Rate':<12}"
            f"{'Prefetches':<12}"
            f"{'Precision':<12}"
            f"{'Coverage':<12}"
        )

        print("-" * 90)

        for capacity in CACHE_CAPACITIES:

            result = run_experiment(
                data,
                capacity,
                prefetch_size,
            )

            print(
                f"{capacity:<10}"
                f"{result['hits']:<10}"
                f"{result['misses']:<10}"
                f"{result['hit_rate_percent']:<12.2f}"
                f"{result['prefetches']:<12}"
                f"{result['prediction_precision']:<12.2f}"
                f"{result['prediction_coverage']:<12.2f}"
            )

    print()
    print("=" * 100)
    print(
        "SELECTIVE HISTORY PREFETCH BENCHMARK COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()