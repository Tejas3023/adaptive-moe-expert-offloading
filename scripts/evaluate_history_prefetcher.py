import json
from collections import defaultdict

from src.prefetch.history_prefetcher import HistoryPrefetcher


TRACE_FILE = "results/traces/wikitext_10_samples.jsonl"
WINDOWS = [1, 2, 4, 8, 16, 32]


def load_tokens():
    """
    Organize traces as:

    sample -> layer -> token -> set(experts)
    """

    data = defaultdict(lambda: defaultdict(dict))

    with open(TRACE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            trace = json.loads(line)

            sample = trace["sample_id"]
            layer = trace["layer_id"]
            token = trace["token_position"]

            data[sample][layer][token] = set(
                trace["selected_experts"]
            )

    return data


def evaluate(data, window):
    """
    Evaluate history-based expert prediction
    for one history window size.
    """

    total_correct = 0
    total_actual = 0
    total_predicted = 0
    total_possible = 0

    for sample_layers in data.values():

        for token_layers in sample_layers.values():

            tokens = sorted(token_layers)

            # Separate predictor for each layer.
            predictor = HistoryPrefetcher(
                history_size=window,
                prefetch_size=8,
            )

            for token in tokens:

                actual = token_layers[token]

                # Predict BEFORE observing the current token.
                predicted = set(
                    predictor.predict()
                )

                # First token has no history,
                # so there is nothing valid to evaluate.
                if predicted:

                    correct = len(
                        actual & predicted
                    )

                    total_correct += correct
                    total_actual += len(actual)
                    total_predicted += len(predicted)
                    total_possible += 1

                # Observe current token only AFTER prediction.
                predictor.observe(actual)

    coverage = (
        total_correct / total_actual * 100
        if total_actual
        else 0.0
    )

    precision = (
        total_correct / total_predicted * 100
        if total_predicted
        else 0.0
    )

    return {
        "correct": total_correct,
        "actual": total_actual,
        "predicted": total_predicted,
        "coverage": coverage,
        "precision": precision,
        "tokens": total_possible,
    }


def main():

    print("=" * 80)
    print("HISTORY PREFETCHER EVALUATION")
    print("=" * 80)

    print()
    print("Trace file:")
    print(TRACE_FILE)

    data = load_tokens()

    samples = len(data)

    print()
    print(f"Samples: {samples}")

    print()
    print("-" * 80)
    print("WINDOW COMPARISON")
    print("-" * 80)

    print(
        f"{'Window':<10}"
        f"{'Correct':<12}"
        f"{'Predicted':<12}"
        f"{'Actual':<12}"
        f"{'Coverage':<12}"
        f"{'Precision':<12}"
    )

    for window in WINDOWS:

        result = evaluate(
            data,
            window,
        )

        print(
            f"{window:<10}"
            f"{result['correct']:<12}"
            f"{result['predicted']:<12}"
            f"{result['actual']:<12}"
            f"{result['coverage']:<12.2f}%"
            f"{result['precision']:<12.2f}%"
        )

    print()
    print("=" * 80)
    print("HISTORY PREFETCHER EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()