import os
import sys


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


from src.prefetch.selective_prefetcher import (
    SelectiveTransitionPrefetcher
)


def main():

    print("=" * 70)
    print("Testing Selective Transition Prefetcher")
    print("=" * 70)

    # --------------------------------------------------------
    # Use a relatively high threshold for an easy-to-observe
    # selective prediction.
    # --------------------------------------------------------

    prefetcher = SelectiveTransitionPrefetcher(
        num_experts=64,
        top_k=8,
        max_prefetch_size=8,
        confidence_threshold=0.20
    )

    # --------------------------------------------------------
    # Build transition history.
    # --------------------------------------------------------

    token_1 = [5, 10, 20]

    token_2 = [5, 10, 30]

    token_3 = [10, 20, 30]

    print("\nToken 1:")
    print(token_1)

    prefetcher.update(
        0,
        token_1
    )

    print("\nPrediction before Token 2:")

    prediction = prefetcher.predict(
        0,
        token_1
    )

    print(prediction)

    prefetcher.update(
        0,
        token_2
    )

    print("\nPrediction before Token 3:")

    prediction = prefetcher.predict(
        0,
        token_2
    )

    print(prediction)

    # --------------------------------------------------------
    # Show scores/confidence.
    # --------------------------------------------------------

    print("\nPrediction with confidence:")

    results = prefetcher.predict_with_scores(
        0,
        token_2
    )

    for expert_id, score, confidence in results:

        print(
            f"Expert {expert_id}: "
            f"score={score}, "
            f"confidence={confidence:.4f} "
            f"({confidence * 100:.2f}%)"
        )

    # --------------------------------------------------------
    # Layer awareness.
    # --------------------------------------------------------

    print("\nLayer awareness test:")

    layer_1_token_1 = [40, 50, 60]

    layer_1_token_2 = [40, 50, 70]

    prefetcher.update(
        1,
        layer_1_token_1
    )

    prediction = prefetcher.predict(
        1,
        layer_1_token_1
    )

    print(
        "Layer 1 prediction before learning:",
        prediction
    )

    prefetcher.update(
        1,
        layer_1_token_2
    )

    prediction = prefetcher.predict(
        1,
        layer_1_token_2
    )

    print(
        "Layer 1 prediction after learning:",
        prediction
    )

    # --------------------------------------------------------
    # Statistics.
    # --------------------------------------------------------

    print("\nStatistics:")

    print(
        prefetcher.get_stats()
    )

    print("\n" + "=" * 70)
    print(
        "Selective Transition Prefetcher test completed"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()