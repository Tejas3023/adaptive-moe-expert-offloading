from src.prefetch.history_prefetcher import HistoryPrefetcher


def main():
    print("=" * 80)
    print("HISTORY PREFETCHER TEST")
    print("=" * 80)

    prefetcher = HistoryPrefetcher(
        history_size=8,
        prefetch_size=8,
    )

    test_events = [
        [5, 12, 20],
        [5, 8, 12],
        [5, 20, 8],
    ]

    print("\nOBSERVING EXPERT ROUTING")
    print("-" * 80)

    for i, experts in enumerate(test_events, start=1):
        print(f"Event {i}: {experts}")

        prefetcher.observe(experts)

        print(
            f"History: "
            f"{prefetcher.history_contents()}"
        )

        print(
            f"Prediction: "
            f"{prefetcher.predict()}"
        )

    print("\nFINAL PREDICTION")
    print("-" * 80)

    print(
        "Recent history:",
        prefetcher.history_contents(),
    )

    print(
        "Predicted experts:",
        prefetcher.predict(),
    )

    print("\n" + "=" * 80)
    print("HISTORY PREFETCHER TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()