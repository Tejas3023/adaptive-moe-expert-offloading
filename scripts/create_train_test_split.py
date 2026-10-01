import json
from pathlib import Path

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

TRACE_FILES = {
    "wikitext": Path("results/traces/wikitext_100_samples.jsonl"),
    "gsm8k": Path("results/traces/gsm8k_100_samples.jsonl"),
    "mbpp": Path("results/traces/mbpp_100_samples.jsonl"),
}

OUTPUT_DIR = Path("results/dataset")

TEST_SIZE = 0.20
RANDOM_SEED = 42


# ============================================================
# LOAD JSONL TRACE FILE
# ============================================================

def load_traces(path):
    traces = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                traces.append(json.loads(line))

    return traces


# ============================================================
# SAVE JSONL TRACE FILE
# ============================================================

def save_traces(path, traces):
    with path.open("w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace) + "\n")


# ============================================================
# SPLIT BY SAMPLE ID
# ============================================================

def split_by_sample(traces):

    # Get unique sample IDs
    sample_ids = sorted({
        trace["sample_id"]
        for trace in traces
    })

    print(f"Unique samples: {len(sample_ids)}")

    # Split SAMPLE IDs, not individual routing records
    train_ids, test_ids = train_test_split(
        sample_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )

    # Convert to sets for fast lookup
    train_ids = set(train_ids)
    test_ids = set(test_ids)

    # Put complete samples into train/test
    train_traces = [
        trace
        for trace in traces
        if trace["sample_id"] in train_ids
    ]

    test_traces = [
        trace
        for trace in traces
        if trace["sample_id"] in test_ids
    ]

    return train_traces, test_traces, train_ids, test_ids


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("OLMoE ROUTING TRACE TRAIN / TEST SPLIT")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, trace_path in TRACE_FILES.items():

        print("\n" + "=" * 80)
        print(f"PROCESSING: {name.upper()}")
        print("=" * 80)

        # ----------------------------------------------------
        # Check input file
        # ----------------------------------------------------

        if not trace_path.exists():
            print(f"ERROR: File not found: {trace_path}")
            continue

        # ----------------------------------------------------
        # Load traces
        # ----------------------------------------------------

        traces = load_traces(trace_path)

        print(f"Total routing records: {len(traces):,}")

        # ----------------------------------------------------
        # Split
        # ----------------------------------------------------

        train_traces, test_traces, train_ids, test_ids = split_by_sample(
            traces
        )

        # ----------------------------------------------------
        # Check for leakage
        # ----------------------------------------------------

        overlap = train_ids.intersection(test_ids)

        if overlap:
            raise RuntimeError(
                f"DATA LEAKAGE DETECTED! "
                f"Overlapping sample IDs: {overlap}"
            )

        # ----------------------------------------------------
        # Output paths
        # ----------------------------------------------------

        train_path = OUTPUT_DIR / f"{name}_train.jsonl"
        test_path = OUTPUT_DIR / f"{name}_test.jsonl"

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_traces(train_path, train_traces)
        save_traces(test_path, test_traces)

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        total_samples = len(train_ids) + len(test_ids)

        print()
        print(f"Total samples:        {total_samples}")
        print(f"Training samples:     {len(train_ids)}")
        print(f"Testing samples:      {len(test_ids)}")

        print()
        print(f"Training records:     {len(train_traces):,}")
        print(f"Testing records:      {len(test_traces):,}")

        print()
        print(f"Sample overlap:       {len(overlap)}")

        print()
        print(f"Training file: {train_path}")
        print(f"Testing file:  {test_path}")

        # ----------------------------------------------------
        # Assertions
        # ----------------------------------------------------

        if len(train_ids) != 80:
            raise RuntimeError(
                f"Expected 80 training samples, "
                f"got {len(train_ids)}"
            )

        if len(test_ids) != 20:
            raise RuntimeError(
                f"Expected 20 testing samples, "
                f"got {len(test_ids)}"
            )

        if overlap:
            raise RuntimeError(
                "Train/test sample overlap detected!"
            )

        if len(train_traces) + len(test_traces) != len(traces):
            raise RuntimeError(
                "Routing record count changed during split!"
            )

        # ----------------------------------------------------
        # Verify every sample is entirely in one split
        # ----------------------------------------------------

        train_samples_from_records = {
            trace["sample_id"]
            for trace in train_traces
        }

        test_samples_from_records = {
            trace["sample_id"]
            for trace in test_traces
        }

        if train_samples_from_records != train_ids:
            raise RuntimeError(
                "Training sample IDs do not match training records!"
            )

        if test_samples_from_records != test_ids:
            raise RuntimeError(
                "Testing sample IDs do not match testing records!"
            )

        print()
        print(f"{name.upper()}: SPLIT PASSED")


    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 80)
    print("TRAIN / TEST SPLIT COMPLETE")
    print("=" * 80)

    print("\nCreated files:")

    for name in TRACE_FILES:
        print(f"  results/dataset/{name}_train.jsonl")
        print(f"  results/dataset/{name}_test.jsonl")


if __name__ == "__main__":
    main()