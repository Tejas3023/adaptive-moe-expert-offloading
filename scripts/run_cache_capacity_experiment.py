from src.benchmark.cache_benchmark import CacheBenchmark
from src.benchmark.results_writer import ResultsWriter


def main():
    print("=" * 90)
    print("CACHE CAPACITY EXPERIMENT")
    print("=" * 90)

    trace_path = "results/traces/test_trace.jsonl"

    cache_capacities = [
        2,
        4,
        8,
        16,
        32,
        64,
    ]

    output_path = (
        "results/experiments/"
        "cache_capacity_results.csv"
    )

    print("\nTrace file:")
    print(trace_path)

    print("\nRunning experiments...\n")

    results = []

    for capacity in cache_capacities:

        print("-" * 90)

        print(
            f"Running experiment with "
            f"cache capacity = {capacity}"
        )

        benchmark = CacheBenchmark(
            trace_path=trace_path,
            cache_capacity=capacity,
        )

        statistics = benchmark.run()

        result = {
            "capacity": capacity,
            "hits": statistics["hits"],
            "misses": statistics["misses"],
            "evictions": statistics["evictions"],
            "hit_rate_percent": (
                statistics["hit_rate_percent"]
            ),
        }

        results.append(result)

        print(
            f"Hits: {result['hits']}"
        )

        print(
            f"Misses: {result['misses']}"
        )

        print(
            f"Evictions: {result['evictions']}"
        )

        print(
            "Hit rate: "
            f"{result['hit_rate_percent']:.2f}%"
        )

    print("\n" + "=" * 90)
    print("EXPERIMENT SUMMARY")
    print("=" * 90)

    print(
        f"{'Capacity':<12}"
        f"{'Hits':<10}"
        f"{'Misses':<10}"
        f"{'Evictions':<12}"
        f"{'Hit Rate'}"
    )

    print("-" * 90)

    for result in results:

        print(
            f"{result['capacity']:<12}"
            f"{result['hits']:<10}"
            f"{result['misses']:<10}"
            f"{result['evictions']:<12}"
            f"{result['hit_rate_percent']:.2f}%"
        )

    # ----------------------------------------------------------
    # SAVE RESULTS
    # ----------------------------------------------------------

    writer = ResultsWriter(
        output_path
    )

    writer.write(results)

    print("\nResults saved successfully.")

    print(
        f"Output file: {output_path}"
    )

    print("\n" + "=" * 90)
    print("CACHE CAPACITY EXPERIMENT COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()