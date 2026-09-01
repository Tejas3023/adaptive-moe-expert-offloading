import json
from collections import defaultdict

from src.cache.expert_cache import ExpertCache
from src.prefetch.history_prefetcher import HistoryPrefetcher


TRACE_FILE = "results/traces/wikitext_10_samples.jsonl"

CACHE_CAPACITIES = [8, 16, 32, 64]
HISTORY_WINDOWS = [1, 4, 8, 16, 32]


def load_traces():
    traces = defaultdict(lambda: defaultdict(dict))

    with open(TRACE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            trace = json.loads(line)

            sample = trace["sample_id"]
            layer = trace["layer_id"]
            token = trace["token_position"]

            traces[sample][layer][token] = set(
                trace["selected_experts"]
            )

    return traces


def run_baseline(data, capacity):
    cache = ExpertCache(capacity)

    for sample_layers in data.values():
        for token_layers in sample_layers.values():
            for token in sorted(token_layers):
                for expert in token_layers[token]:
                    cache.request(expert)

    return cache.statistics()


def run_prefetch(data, capacity, history_window):
    cache = ExpertCache(capacity)

    for sample_layers in data.values():
        for token_layers in sample_layers.values():

            predictor = HistoryPrefetcher(
                history_size=history_window,
                prefetch_size=8,
            )

            for token in sorted(token_layers):

                actual = token_layers[token]

                # Predict BEFORE seeing the current token.
                predictions = predictor.predict()

                for expert in predictions:
                    cache.prefetch(expert)

                # Now process the experts actually required.
                for expert in actual:
                    cache.request(expert)

                # Only after the request is processed do we
                # reveal the current routing decision.
                predictor.observe(actual)

    return cache.statistics()


def main():
    print("=" * 90)
    print("PREFETCH BENCHMARK")
    print("=" * 90)

    print()
    print("Trace file:")
    print(TRACE_FILE)

    data = load_traces()

    print()
    print("Running baseline LRU experiments...")
    print()

    print(
        f"{'Capacity':<12}"
        f"{'Hits':<10}"
        f"{'Misses':<10}"
        f"{'Hit Rate':<12}"
    )

    print("-" * 50)

    for capacity in CACHE_CAPACITIES:

        result = run_baseline(
            data,
            capacity,
        )

        print(
            f"{capacity:<12}"
            f"{result['hits']:<10}"
            f"{result['misses']:<10}"
            f"{result['hit_rate_percent']:.2f}%"
        )

    print()
    print("=" * 90)
    print("HISTORY PREFETCH RESULTS")
    print("=" * 90)

    for window in HISTORY_WINDOWS:

        print()
        print(f"HISTORY WINDOW = {window}")
        print("-" * 70)

        print(
            f"{'Capacity':<12}"
            f"{'Hits':<10}"
            f"{'Misses':<10}"
            f"{'Hit Rate':<12}"
            f"{'Prefetches':<12}"
        )

        print("-" * 70)

        for capacity in CACHE_CAPACITIES:

            result = run_prefetch(
                data,
                capacity,
                window,
            )

            print(
                f"{capacity:<12}"
                f"{result['hits']:<10}"
                f"{result['misses']:<10}"
                f"{result['hit_rate_percent']:.2f}%"
                f"{result['prefetches']:<12}"
            )

    print()
    print("=" * 90)
    print("PREFETCH BENCHMARK COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()