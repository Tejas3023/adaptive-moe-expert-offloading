import json
from pathlib import Path
from collections import Counter


TRACE_FILES = {
    "WikiText-103": Path("results/traces/wikitext_100_samples.jsonl"),
    "GSM8K": Path("results/traces/gsm8k_100_samples.jsonl"),
    "MBPP": Path("results/traces/mbpp_100_samples.jsonl"),
}

EXPECTED_SAMPLES = 100
EXPECTED_LAYERS = 16
EXPECTED_EXPERTS = 64
EXPECTED_TOP_K = 8


def validate_trace(name, path):
    print("=" * 80)
    print(f"VALIDATING {name}")
    print("=" * 80)

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return False

    traces = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                trace = json.loads(line)
                traces.append(trace)
            except json.JSONDecodeError as e:
                print(f"ERROR: Invalid JSON at line {line_number}")
                print(e)
                return False

    print(f"Trace records: {len(traces):,}")

    if not traces:
        print("ERROR: No traces found.")
        return False

    # ------------------------------------------------------------------
    # Collect sample IDs
    # ------------------------------------------------------------------

    sample_ids = set()

    for trace in traces:
        if "sample_id" not in trace:
            print("ERROR: Missing sample_id")
            return False

        sample_ids.add(trace["sample_id"])

    print(f"Unique sample IDs: {len(sample_ids)}")
    print(f"Sample IDs: {sorted(sample_ids)[:10]} ...")

    # ------------------------------------------------------------------
    # Basic field validation
    # ------------------------------------------------------------------

    required_fields = [
        "sample_id",
        "layer_id",
        "token_position",
        "selected_experts",
        "routing_weights",
    ]

    for i, trace in enumerate(traces):

        for field in required_fields:
            if field not in trace:
                print(
                    f"ERROR: Record {i} is missing field: {field}"
                )
                return False

        layer = trace["layer_id"]

        if not (0 <= layer < EXPECTED_LAYERS):
            print(
                f"ERROR: Invalid layer_id {layer} "
                f"in record {i}"
            )
            return False

        experts = trace["selected_experts"]
        weights = trace["routing_weights"]

        # --------------------------------------------------------------
        # Exactly top-k experts?
        # --------------------------------------------------------------

        if len(experts) != EXPECTED_TOP_K:
            print(
                f"ERROR: Record {i} has {len(experts)} experts "
                f"instead of {EXPECTED_TOP_K}"
            )
            return False

        if len(weights) != EXPECTED_TOP_K:
            print(
                f"ERROR: Record {i} has {len(weights)} weights "
                f"instead of {EXPECTED_TOP_K}"
            )
            return False

        # --------------------------------------------------------------
        # Valid expert IDs?
        # --------------------------------------------------------------

        for expert_id in experts:
            if not (0 <= expert_id < EXPECTED_EXPERTS):
                print(
                    f"ERROR: Invalid expert ID {expert_id} "
                    f"in record {i}"
                )
                return False

        # --------------------------------------------------------------
        # Expert IDs should not repeat within one routing event
        # --------------------------------------------------------------

        if len(set(experts)) != EXPECTED_TOP_K:
            print(
                f"ERROR: Duplicate expert IDs in record {i}: "
                f"{experts}"
            )
            return False

        # --------------------------------------------------------------
        # Routing weights should be valid
        # --------------------------------------------------------------

        for weight in weights:
            if not isinstance(weight, (int, float)):
                print(
                    f"ERROR: Invalid routing weight "
                    f"in record {i}"
                )
                return False

            if weight < 0:
                print(
                    f"ERROR: Negative routing weight "
                    f"in record {i}"
                )
                return False

    # ------------------------------------------------------------------
    # Token / layer structure
    # ------------------------------------------------------------------

    sample_token_layers = {}

    for trace in traces:

        sample_id = trace["sample_id"]
        token_position = trace["token_position"]
        layer_id = trace["layer_id"]

        key = (sample_id, token_position)

        if key not in sample_token_layers:
            sample_token_layers[key] = set()

        sample_token_layers[key].add(layer_id)

    incomplete_tokens = []

    for key, layers in sample_token_layers.items():

        if len(layers) != EXPECTED_LAYERS:

            incomplete_tokens.append(
                (key, sorted(layers))
            )

    if incomplete_tokens:

        print(
            f"ERROR: Found {len(incomplete_tokens)} "
            f"tokens without all {EXPECTED_LAYERS} layers."
        )

        print("First few incomplete tokens:")

        for item in incomplete_tokens[:5]:
            print(item)

        return False

    print(
        f"Complete token-layer groups: "
        f"{len(sample_token_layers):,}"
    )

    # ------------------------------------------------------------------
    # Expected routing-event count
    # ------------------------------------------------------------------

    expected_records = (
        len(sample_token_layers) * EXPECTED_LAYERS
    )

    if len(traces) != expected_records:

        print(
            "ERROR: Routing-event count mismatch."
        )

        print(
            f"Expected: {expected_records:,}"
        )

        print(
            f"Actual:   {len(traces):,}"
        )

        return False

    print(
        f"Routing-event formula verified:"
    )

    print(
        f"{len(sample_token_layers):,} tokens × "
        f"{EXPECTED_LAYERS} layers = "
        f"{expected_records:,} records"
    )

    # ------------------------------------------------------------------
    # Token counts per sample
    # ------------------------------------------------------------------

    sample_tokens = Counter()

    for sample_id, token_position in sample_token_layers:

        sample_tokens[sample_id] = max(
            sample_tokens[sample_id],
            token_position + 1
        )

    print("\nTokens per sample:")

    token_values = list(sample_tokens.values())

    print(
        f"Minimum: {min(token_values)}"
    )

    print(
        f"Maximum: {max(token_values)}"
    )

    print(
        f"Average: {sum(token_values) / len(token_values):.2f}"
    )

    # ------------------------------------------------------------------
    # Final status
    # ------------------------------------------------------------------

    passed = True

    if len(sample_ids) != EXPECTED_SAMPLES:
        print(
            f"WARNING: Expected {EXPECTED_SAMPLES} samples "
            f"but found {len(sample_ids)}."
        )
        passed = False

    print("\n" + "-" * 80)

    if passed:
        print(f"{name}: VALIDATION PASSED")
    else:
        print(f"{name}: VALIDATION COMPLETED WITH WARNINGS")

    print("-" * 80)

    return passed


def main():

    print("=" * 80)
    print("LARGE OLMOE TRACE VALIDATION")
    print("=" * 80)

    results = {}

    for name, path in TRACE_FILES.items():

        results[name] = validate_trace(name, path)

        print()

    print("=" * 80)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 80)

    for name, result in results.items():

        status = "PASS" if result else "CHECK"

        print(
            f"{name:<15} : {status}"
        )

    if all(results.values()):
        print("\nALL TRACE FILES PASSED VALIDATION.")
    else:
        print(
            "\nSome trace files need inspection."
        )


if __name__ == "__main__":
    main()