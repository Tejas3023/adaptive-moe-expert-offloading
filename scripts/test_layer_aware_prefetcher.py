from src.prefetch.layer_aware_prefetcher import (
    LayerAwareHistoryPrefetcher,
)


def main():

    print("=" * 80)
    print("LAYER-AWARE HISTORY PREFETCHER TEST")
    print("=" * 80)

    predictor = LayerAwareHistoryPrefetcher(
        history_size=3,
        prefetch_size=4,
    )

    print()
    print("Observing routing:")
    print("-" * 80)

    predictor.observe(
        0,
        [5, 10, 20, 30],
    )

    predictor.observe(
        0,
        [5, 10, 15, 20],
    )

    predictor.observe(
        1,
        [40, 50, 60, 70],
    )

    predictor.observe(
        1,
        [40, 50, 80, 90],
    )

    print("Layer 0 history:")
    print(
        predictor.history_contents(0)
    )

    print()

    print("Layer 1 history:")
    print(
        predictor.history_contents(1)
    )

    print()
    print("Predictions:")
    print("-" * 80)

    print(
        "Layer 0:",
        predictor.predict(0),
    )

    print(
        "Layer 1:",
        predictor.predict(1),
    )

    print()

    print("=" * 80)
    print("LAYER-AWARE PREFETCHER TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()