from src.benchmark.trace_analyzer import TraceAnalyzer


def main():
    print("=" * 80)
    print("TRACE ANALYZER TEST")
    print("=" * 80)

    trace_path = (
        "results/traces/test_trace.jsonl"
    )

    analyzer = TraceAnalyzer(
        trace_path
    )

    print("\nTRACE FILE")
    print("-" * 80)
    print(trace_path)

    print("\nBASIC STATISTICS")
    print("-" * 80)

    print(
        "Total routing events:",
        analyzer.total_routing_events(),
    )

    print(
        "Total expert selections:",
        analyzer.total_expert_selections(),
    )

    print(
        "Unique experts used:",
        analyzer.unique_experts_used(),
    )

    print("\nTOP 10 MOST USED EXPERTS")
    print("-" * 80)

    for expert_id, count in analyzer.most_used_experts(
        top_n=10
    ):
        print(
            f"Expert {expert_id}: "
            f"{count} selections"
        )

    print("\nEXPERT USAGE PERCENTAGES")
    print("-" * 80)

    percentages = analyzer.expert_usage_percentage()

    for expert_id, percentage in sorted(
        percentages.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:10]:
        print(
            f"Expert {expert_id}: "
            f"{percentage:.2f}%"
        )

    print("\nFULL SUMMARY")
    print("-" * 80)

    summary = analyzer.summary()

    for key, value in summary.items():
        if key != "expert_usage_counts":
            print(f"{key}: {value}")

    print("\n" + "=" * 80)
    print("TRACE ANALYZER TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()