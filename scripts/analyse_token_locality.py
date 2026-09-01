from collections import defaultdict
from pathlib import Path

from src.tracing.trace_writer import TraceWriter


TRACE_FILE = Path(
    "results/traces/wikitext_10_samples.jsonl"
)

WINDOWS = [1, 2, 4, 8]


def group_traces_by_sample_and_layer(traces):
    """
    Organize traces as:

        sample_id
            └── layer_id
                    └── token_position -> experts

    This prevents us from accidentally comparing experts
    across different samples or different transformer layers.
    """

    grouped = defaultdict(lambda: defaultdict(dict))

    for trace in traces:
        grouped[trace["sample_id"]][trace["layer_id"]][
            trace["token_position"]
        ] = set(trace["selected_experts"])

    return grouped


def overlap(a, b):
    """
    Number of experts shared by two token routing decisions.
    """

    return len(a & b)


def analyze(groups):
    """
    Calculate token-level locality statistics.
    """

    consecutive_overlaps = []
    window_results = {
        window: []
        for window in WINDOWS
    }

    for sample_id, layers in groups.items():

        for layer_id, tokens in layers.items():

            positions = sorted(tokens)

            for index in range(1, len(positions)):

                current_position = positions[index]
                current_experts = tokens[current_position]

                # --------------------------------------------------
                # Consecutive token overlap
                # --------------------------------------------------

                previous_position = positions[index - 1]

                previous_experts = tokens[
                    previous_position
                ]

                consecutive_overlaps.append(
                    overlap(
                        current_experts,
                        previous_experts,
                    )
                )

                # --------------------------------------------------
                # Lookback windows
                # --------------------------------------------------

                for window in WINDOWS:

                    start = max(
                        0,
                        index - window,
                    )

                    previous_experts_union = set()

                    for previous_index in range(
                        start,
                        index,
                    ):

                        previous_position = positions[
                            previous_index
                        ]

                        previous_experts_union.update(
                            tokens[previous_position]
                        )

                    hits = len(
                        current_experts
                        & previous_experts_union
                    )

                    window_results[window].append(
                        hits
                    )

    return (
        consecutive_overlaps,
        window_results,
    )


def print_distribution(values):
    """
    Print the distribution of overlap counts.
    """

    counts = {
        value: values.count(value)
        for value in range(9)
    }

    total = len(values)

    for value in range(9):

        percentage = (
            counts[value] / total * 100
            if total
            else 0.0
        )

        print(
            f"{value} experts: "
            f"{counts[value]:6d} "
            f"({percentage:6.2f}%)"
        )


def main():

    print("=" * 80)
    print("TOKEN-LEVEL EXPERT LOCALITY ANALYSIS")
    print("=" * 80)

    print()
    print("Trace file:")
    print(TRACE_FILE)

    # --------------------------------------------------------------
    # Load traces
    # --------------------------------------------------------------

    writer = TraceWriter(
        TRACE_FILE
    )

    traces = writer.read()

    print()
    print("-" * 80)
    print("TRACE INFORMATION")
    print("-" * 80)

    print(
        f"Routing events: {len(traces)}"
    )

    print(
        f"Expert selections: "
        f"{sum(len(t['selected_experts']) for t in traces)}"
    )

    # --------------------------------------------------------------
    # Organize traces
    # --------------------------------------------------------------

    groups = group_traces_by_sample_and_layer(
        traces
    )

    print(
        f"Samples: {len(groups)}"
    )

    # --------------------------------------------------------------
    # Analyze
    # --------------------------------------------------------------

    (
        consecutive_overlaps,
        window_results,
    ) = analyze(groups)

    # --------------------------------------------------------------
    # Consecutive overlap
    # --------------------------------------------------------------

    print()
    print("-" * 80)
    print("CONSECUTIVE TOKEN EXPERT OVERLAP")
    print("-" * 80)

    total_comparisons = len(
        consecutive_overlaps
    )

    average_overlap = (
        sum(consecutive_overlaps)
        / total_comparisons
        if total_comparisons
        else 0.0
    )

    average_percentage = (
        average_overlap / 8 * 100
    )

    print(
        f"Token comparisons: "
        f"{total_comparisons}"
    )

    print(
        f"Average experts reused: "
        f"{average_overlap:.2f} / 8"
    )

    print(
        f"Average overlap: "
        f"{average_percentage:.2f}%"
    )

    print()
    print("Overlap distribution:")
    print_distribution(
        consecutive_overlaps
    )

    # --------------------------------------------------------------
    # Lookback windows
    # --------------------------------------------------------------

    print()
    print("-" * 80)
    print("TOKEN LOOKBACK LOCALITY")
    print("-" * 80)

    print(
        "Window    Avg Experts Found    Coverage"
    )

    print("-" * 80)

    for window in WINDOWS:

        values = window_results[window]

        average = (
            sum(values) / len(values)
            if values
            else 0.0
        )

        coverage = (
            average / 8 * 100
        )

        print(
            f"{window:<10}"
            f"{average:<21.2f}"
            f"{coverage:.2f}%"
        )

    # --------------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------------

    print()
    print("-" * 80)
    print("PREFETCH OPPORTUNITY")
    print("-" * 80)

    best_window = max(
        WINDOWS,
        key=lambda window: (
            sum(window_results[window])
            / len(window_results[window])
            if window_results[window]
            else 0
        ),
    )

    best_values = window_results[
        best_window
    ]

    best_average = (
        sum(best_values)
        / len(best_values)
        if best_values
        else 0.0
    )

    print(
        f"Using the previous {best_window} tokens, "
        f"an average of "
        f"{best_average:.2f} / 8 "
        f"current experts were seen recently."
    )

    print(
        f"Potential coverage: "
        f"{best_average / 8 * 100:.2f}%"
    )

    print()
    print("=" * 80)
    print("TOKEN LOCALITY ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()