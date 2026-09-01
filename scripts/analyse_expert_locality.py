from collections import Counter, defaultdict
from pathlib import Path

from src.tracing.trace_writer import TraceWriter


TRACE_FILE = Path(
    "results/traces/wikitext_10_samples.jsonl"
)


def flatten_expert_requests(traces):
    """
    Convert routing traces into a chronological sequence
    of expert requests.

    Each routing event contains 8 selected experts.
    We preserve the order in which the traces appear.
    """

    requests = []

    for trace in traces:
        for expert_id in trace["selected_experts"]:
            requests.append(
                {
                    "sample_id": trace["sample_id"],
                    "layer_id": trace["layer_id"],
                    "token_position": trace["token_position"],
                    "expert_id": expert_id,
                }
            )

    return requests


def calculate_reuse_distance(requests):
    """
    Measure how many expert requests occur between
    two consecutive uses of the same expert.

    Smaller reuse distance = stronger locality.
    """

    last_seen = {}
    distances = []

    for index, request in enumerate(requests):

        expert_id = request["expert_id"]

        if expert_id in last_seen:
            distance = index - last_seen[expert_id] - 1
            distances.append(distance)

        last_seen[expert_id] = index

    return distances


def calculate_window_hit_rates(requests, windows):
    """
    For each window size, calculate how often the next
    expert request is already present in the previous
    N expert requests.
    """

    results = {}

    for window in windows:

        hits = 0
        total = 0

        for index in range(1, len(requests)):

            start = max(0, index - window)

            previous_experts = {
                requests[i]["expert_id"]
                for i in range(start, index)
            }

            current_expert = requests[index]["expert_id"]

            if current_expert in previous_experts:
                hits += 1

            total += 1

        hit_rate = (
            hits / total * 100
            if total > 0
            else 0.0
        )

        results[window] = {
            "hits": hits,
            "total": total,
            "hit_rate": hit_rate,
        }

    return results


def calculate_expert_transitions(requests):
    """
    Count which expert tends to follow another expert.
    """

    transitions = defaultdict(Counter)

    for previous, current in zip(
        requests,
        requests[1:],
    ):

        previous_expert = previous["expert_id"]
        current_expert = current["expert_id"]

        transitions[previous_expert][current_expert] += 1

    return transitions


def calculate_layer_locality(traces):
    """
    Calculate locality independently for each MoE layer.

    This is important because different layers may exhibit
    different expert-routing behavior.
    """

    layer_requests = defaultdict(list)

    for trace in traces:

        layer_id = trace["layer_id"]

        for expert_id in trace["selected_experts"]:
            layer_requests[layer_id].append(expert_id)

    results = {}

    for layer_id, experts in sorted(
        layer_requests.items()
    ):

        hits = 0
        total = 0

        window = 8

        for index in range(1, len(experts)):

            start = max(0, index - window)

            previous = set(
                experts[start:index]
            )

            if experts[index] in previous:
                hits += 1

            total += 1

        hit_rate = (
            hits / total * 100
            if total > 0
            else 0.0
        )

        results[layer_id] = {
            "requests": len(experts),
            "hit_rate": hit_rate,
        }

    return results


def main():

    print("=" * 80)
    print("EXPERT LOCALITY ANALYSIS")
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

    total_selections = sum(
        len(trace["selected_experts"])
        for trace in traces
    )

    print(
        f"Expert selections: {total_selections}"
    )

    # --------------------------------------------------------------
    # Flatten requests
    # --------------------------------------------------------------

    requests = flatten_expert_requests(
        traces
    )

    print()
    print("-" * 80)
    print("EXPERT REQUEST SEQUENCE")
    print("-" * 80)

    print(
        f"Total expert requests: {len(requests)}"
    )

    print(
        "First 30 expert requests:"
    )

    print(
        [
            request["expert_id"]
            for request in requests[:30]
        ]
    )

    # --------------------------------------------------------------
    # Consecutive reuse
    # --------------------------------------------------------------

    print()
    print("-" * 80)
    print("CONSECUTIVE EXPERT REUSE")
    print("-" * 80)

    consecutive_hits = 0

    for previous, current in zip(
        requests,
        requests[1:],
    ):

        if (
            previous["expert_id"]
            == current["expert_id"]
        ):
            consecutive_hits += 1

    comparisons = max(
        len(requests) - 1,
        0,
    )

    consecutive_rate = (
        consecutive_hits / comparisons * 100
        if comparisons > 0
        else 0.0
    )

    print(
        f"Consecutive reuse hits: "
        f"{consecutive_hits}"
    )

    print(
        f"Consecutive comparisons: "
        f"{comparisons}"
    )

    print(
        f"Consecutive reuse rate: "
        f"{consecutive_rate:.2f}%"
    )

    # --------------------------------------------------------------
    # Window locality
    # --------------------------------------------------------------

    windows = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
    ]

    window_results = calculate_window_hit_rates(
        requests,
        windows,
    )

    print()
    print("-" * 80)
    print("SHORT-WINDOW LOCALITY")
    print("-" * 80)

    print(
        "Window    Hits      Total     Hit Rate"
    )

    print("-" * 80)

    for window in windows:

        result = window_results[window]

        print(
            f"{window:<9}"
            f"{result['hits']:<10}"
            f"{result['total']:<10}"
            f"{result['hit_rate']:.2f}%"
        )

    # --------------------------------------------------------------
    # Reuse distance
    # --------------------------------------------------------------

    distances = calculate_reuse_distance(
        requests
    )

    print()
    print("-" * 80)
    print("REUSE DISTANCE")
    print("-" * 80)

    if distances:

        average_distance = (
            sum(distances)
            / len(distances)
        )

        print(
            f"Repeated expert references: "
            f"{len(distances)}"
        )

        print(
            f"Average reuse distance: "
            f"{average_distance:.2f}"
        )

        print(
            f"Minimum reuse distance: "
            f"{min(distances)}"
        )

        print(
            f"Maximum reuse distance: "
            f"{max(distances)}"
        )

    else:

        print(
            "No repeated expert references found."
        )

    # --------------------------------------------------------------
    # Most common expert transitions
    # --------------------------------------------------------------

    transitions = calculate_expert_transitions(
        requests
    )

    transition_counts = Counter()

    for previous_expert, next_experts in transitions.items():

        for next_expert, count in next_experts.items():

            transition_counts[
                (
                    previous_expert,
                    next_expert,
                )
            ] += count

    print()
    print("-" * 80)
    print("TOP 20 EXPERT TRANSITIONS")
    print("-" * 80)

    for (
        (previous_expert, next_expert),
        count,
    ) in transition_counts.most_common(20):

        print(
            f"Expert {previous_expert}"
            f" -> "
            f"Expert {next_expert}: "
            f"{count}"
        )

    # --------------------------------------------------------------
    # Layer-specific locality
    # --------------------------------------------------------------

    layer_results = calculate_layer_locality(
        traces
    )

    print()
    print("-" * 80)
    print("LAYER-SPECIFIC LOCALITY")
    print("-" * 80)

    print(
        "Layer    Requests    8-Request Hit Rate"
    )

    print("-" * 80)

    for layer_id, result in layer_results.items():

        print(
            f"{layer_id:<9}"
            f"{result['requests']:<12}"
            f"{result['hit_rate']:.2f}%"
        )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    print()
    print("=" * 80)
    print("LOCALITY ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()