import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.prefetch.transition_prefetcher import TransitionPrefetcher

def main():

    print("=" * 60)
    print("Testing Transition Prefetcher")
    print("=" * 60)

    prefetcher = TransitionPrefetcher(
        num_experts=64,
        top_k=8,
        prefetch_size=3
    )

    # ---------------------------------------------------------
    # Layer 0
    # ---------------------------------------------------------

    print("\n--- Training transitions ---")

    token_1 = [5, 10, 20]
    token_2 = [5, 10, 30]
    token_3 = [10, 20, 30]

    print("Token 1:", token_1)
    prefetcher.update(0, token_1)

    print("Token 2:", token_2)
    print("Prediction before Token 2:")

    prediction = prefetcher.predict(0, token_1)
    print("  ", prediction)

    prefetcher.update(0, token_2)

    print("Token 3:", token_3)
    print("Prediction before Token 3:")

    prediction = prefetcher.predict(0, token_2)
    print("  ", prediction)

    prefetcher.update(0, token_3)

    # ---------------------------------------------------------
    # Show transition counts
    # ---------------------------------------------------------

    print("\n--- Transition counts ---")

    transitions = prefetcher.get_layer_transitions(0)

    for current_expert in sorted(transitions):

        print(
            f"Expert {current_expert} -> "
            f"{dict(transitions[current_expert])}"
        )

    # ---------------------------------------------------------
    # Prediction with scores
    # ---------------------------------------------------------

    print("\n--- Prediction with scores ---")

    scores = prefetcher.predict_with_scores(
        0,
        token_3
    )

    print(scores)

    # ---------------------------------------------------------
    # Verify specific transition
    # ---------------------------------------------------------

    print("\n--- Specific transition check ---")

    count = prefetcher.get_transition_count(
        layer_id=0,
        current_expert=5,
        next_expert=10
    )

    print(
        "Layer 0: Expert 5 -> Expert 10 =",
        count
    )

    # ---------------------------------------------------------
    # Test layer awareness
    # ---------------------------------------------------------

    print("\n--- Layer awareness test ---")

    layer_1_token_1 = [40, 50, 60]
    layer_1_token_2 = [40, 50, 70]

    prefetcher.update(1, layer_1_token_1)

    prediction_layer_1 = prefetcher.predict(
        1,
        layer_1_token_1
    )

    print(
        "Layer 1 prediction before learning:",
        prediction_layer_1
    )

    prefetcher.update(1, layer_1_token_2)

    prediction_layer_1 = prefetcher.predict(
        1,
        layer_1_token_2
    )

    print(
        "Layer 1 prediction after learning:",
        prediction_layer_1
    )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    print("\n--- Statistics ---")

    print(prefetcher.get_stats())

    print("\n" + "=" * 60)
    print("Transition Prefetcher test completed")
    print("=" * 60)


if __name__ == "__main__":
    main()