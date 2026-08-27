from src.benchmark.cache_benchmark import CacheBenchmark


def main():
    print("=" * 80)
    print("EXPERT CACHE BENCHMARK")
    print("=" * 80)

    trace_path = (
        "results/traces/test_trace.jsonl"
    )

    cache_capacity = 8

    benchmark = CacheBenchmark(
        trace_path=trace_path,
        cache_capacity=cache_capacity,
    )

    print("\nBENCHMARK CONFIGURATION")
    print("-" * 80)

    print(
        "Trace file:",
        trace_path,
    )

    print(
        "Cache capacity:",
        cache_capacity,
        "experts",
    )

    print(
        "Routing events:",
        benchmark.total_routing_events(),
    )

    print(
        "Total expert requests:",
        benchmark.total_expert_requests(),
    )

    print("\nRUNNING BENCHMARK...")
    print("-" * 80)

    statistics = benchmark.run()

    print("\nCACHE RESULTS")
    print("-" * 80)

    for key, value in statistics.items():
        print(
            f"{key}: {value}"
        )

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()