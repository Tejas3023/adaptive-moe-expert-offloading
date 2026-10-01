import json
import os
import sys
from collections import defaultdict

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.prefetch.transition_prefetcher import TransitionPrefetcher


TRACE_FILE = "results/traces/wikitext_10_samples.jsonl"

PREFETCH_SIZES = [1, 2, 4, 8]


def load_trace():

    records = []

    with open(
        TRACE_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def group_by_sample_and_layer(records):

    grouped = defaultdict(
        lambda: defaultdict(list)
    )

    for record in records:

        sample_id = record["sample_id"]
        layer_id = record["layer_id"]

        grouped[sample_id][layer_id].append(
            record
        )

    return grouped


def evaluate(prefetch_size):

    records = load_trace()

    grouped = group_by_sample_and_layer(records)

    prefetcher = TransitionPrefetcher(
        num_experts=64,
        top_k=8,
        prefetch_size=prefetch_size
    )

    total_predicted = 0
    total_correct = 0
    total_actual = 0

    prediction_events = 0

    # ---------------------------------------------------------
    # Process each sample separately
    # ---------------------------------------------------------

    for sample_id in sorted(grouped):

        # A new sample means there should not be a
        # transition from the previous sample.
        prefetcher.reset()

        layers = grouped[sample_id]

        # Number of tokens can be determined from layer 0.
        if 0 not in layers:
            continue

        num_tokens = len(layers[0])

        # -----------------------------------------------------
        # Process token by token
        # -----------------------------------------------------

        for token_index in range(num_tokens):

            for layer_id in sorted(layers):

                token_record = layers[layer_id][token_index]

                actual_experts = token_record[
                    "selected_experts"
                ]

                # -------------------------------------------------
                # Predict BEFORE updating with the current token.
                #
                # We use the previous token's expert set.
                # -------------------------------------------------

                if token_index > 0:

                    previous_record = layers[layer_id][
                        token_index - 1
                    ]

                    previous_experts = previous_record[
                        "selected_experts"
                    ]

                    predicted = prefetcher.predict(
                        layer_id,
                        previous_experts
                    )

                    actual = set(actual_experts)

                    predicted_set = set(predicted)

                    correct = (
                        predicted_set & actual
                    )

                    total_correct += len(correct)
                    total_predicted += len(predicted)
                    total_actual += len(actual)

                    prediction_events += 1

                # -------------------------------------------------
                # Only AFTER prediction do we update the model
                # with the current token.
                # -------------------------------------------------

                prefetcher.update(
                    layer_id,
                    actual_experts
                )

    if total_predicted > 0:

        precision = (
            total_correct /
            total_predicted
        ) * 100

    else:

        precision = 0.0

    if total_actual > 0:

        coverage = (
            total_correct /
            total_actual
        ) * 100

    else:

        coverage = 0.0

    print(
        f"Prefetch size {prefetch_size}: "
        f"correct={total_correct}, "
        f"predicted={total_predicted}, "
        f"actual={total_actual}, "
        f"precision={precision:.2f}%, "
        f"coverage={coverage:.2f}%"
    )

    return {
        "prefetch_size": prefetch_size,
        "correct": total_correct,
        "predicted": total_predicted,
        "actual": total_actual,
        "precision": precision,
        "coverage": coverage,
        "prediction_events": prediction_events
    }


def main():

    print("=" * 70)
    print("Transition Prefetcher Evaluation")
    print("=" * 70)

    print(
        f"\nTrace: {TRACE_FILE}"
    )

    print(
        f"Prefetch sizes: {PREFETCH_SIZES}\n"
    )

    results = []

    for size in PREFETCH_SIZES:

        result = evaluate(size)

        results.append(result)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print(
        f"{'Prefetch':>10} "
        f"{'Correct':>10} "
        f"{'Predicted':>10} "
        f"{'Actual':>10} "
        f"{'Precision':>12} "
        f"{'Coverage':>12}"
    )

    print("-" * 70)

    for result in results:

        print(
            f"{result['prefetch_size']:>10} "
            f"{result['correct']:>10} "
            f"{result['predicted']:>10} "
            f"{result['actual']:>10} "
            f"{result['precision']:>11.2f}% "
            f"{result['coverage']:>11.2f}%"
        )


if __name__ == "__main__":
    main()