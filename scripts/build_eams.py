import json
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILES = {
    "wikitext": Path("results/dataset/wikitext_train.jsonl"),
    "gsm8k": Path("results/dataset/gsm8k_train.jsonl"),
    "mbpp": Path("results/dataset/mbpp_train.jsonl"),
}

OUTPUT_DIR = Path("results/dataset/eams")

NUM_LAYERS = 16
NUM_EXPERTS = 64
TOP_K = 8


# ============================================================
# LOAD JSONL
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
# BUILD EAMS
# ============================================================

def build_eams(traces):

    # Group routing records by sample
    samples = {}

    for trace in traces:
        sample_id = trace["sample_id"]

        if sample_id not in samples:
            samples[sample_id] = []

        samples[sample_id].append(trace)

    eams = []

    # --------------------------------------------------------
    # Build one EAM per sample
    # --------------------------------------------------------

    for sample_id, sample_traces in samples.items():

        # EAM = 16 × 64
        eam = [
            [0 for _ in range(NUM_EXPERTS)]
            for _ in range(NUM_LAYERS)
        ]

        token_positions = set()

        # ----------------------------------------------------
        # Count expert activations
        # ----------------------------------------------------

        for trace in sample_traces:

            layer_id = trace["layer_id"]
            token_position = trace["token_position"]
            expert_ids = trace["selected_experts"]

            token_positions.add(token_position)

            # Basic validation
            if not 0 <= layer_id < NUM_LAYERS:
                raise ValueError(
                    f"Invalid layer_id {layer_id} "
                    f"in sample {sample_id}"
                )

            if len(expert_ids) != TOP_K:
                raise ValueError(
                    f"Expected {TOP_K} experts, "
                    f"got {len(expert_ids)} "
                    f"in sample {sample_id}"
                )

            if len(set(expert_ids)) != TOP_K:
                raise ValueError(
                    f"Duplicate experts found in "
                    f"sample {sample_id}, "
                    f"token {token_position}, "
                    f"layer {layer_id}"
                )

            for expert_id in expert_ids:

                if not 0 <= expert_id < NUM_EXPERTS:
                    raise ValueError(
                        f"Invalid expert_id {expert_id} "
                        f"in sample {sample_id}"
                    )

                eam[layer_id][expert_id] += 1

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        num_tokens = len(token_positions)

        expected_activations = (
            num_tokens *
            NUM_LAYERS *
            TOP_K
        )

        actual_activations = sum(
            sum(row)
            for row in eam
        )

        if actual_activations != expected_activations:
            raise ValueError(
                f"EAM activation mismatch for "
                f"sample {sample_id}: "
                f"expected {expected_activations}, "
                f"got {actual_activations}"
            )

        # ----------------------------------------------------
        # Store EAM
        # ----------------------------------------------------

        eams.append({
            "sample_id": sample_id,
            "num_tokens": num_tokens,
            "num_activations": actual_activations,
            "shape": [NUM_LAYERS, NUM_EXPERTS],
            "eam": eam
        })

    return eams


# ============================================================
# SAVE EAMS
# ============================================================

def save_eams(path, eams):

    with path.open("w", encoding="utf-8") as f:

        for eam in eams:
            f.write(
                json.dumps(eam) + "\n"
            )


# ============================================================
# PROCESS ONE WORKLOAD
# ============================================================

def process_workload(name, input_path):

    print()
    print("=" * 80)
    print(f"PROCESSING: {name.upper()}")
    print("=" * 80)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    traces = load_traces(input_path)

    print(f"Routing records: {len(traces):,}")

    # Build EAMs
    eams = build_eams(traces)

    print(f"Training samples: {len(eams)}")

    # --------------------------------------------------------
    # Validate number of samples
    # --------------------------------------------------------

    if len(eams) != 80:
        raise ValueError(
            f"Expected 80 training samples, "
            f"got {len(eams)}"
        )

    # --------------------------------------------------------
    # Validate every EAM
    # --------------------------------------------------------

    for item in eams:

        eam = item["eam"]

        if len(eam) != NUM_LAYERS:
            raise ValueError(
                f"Wrong number of layers "
                f"for sample {item['sample_id']}"
            )

        for row in eam:

            if len(row) != NUM_EXPERTS:
                raise ValueError(
                    f"Wrong number of experts "
                    f"for sample {item['sample_id']}"
                )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR /
        f"{name}_train_eams.jsonl"
    )

    save_eams(output_path, eams)

    print(f"EAM shape: {NUM_LAYERS} × {NUM_EXPERTS}")
    print(f"Output: {output_path}")

    # --------------------------------------------------------
    # Display a few examples
    # --------------------------------------------------------

    print()
    print("First 3 samples:")

    for item in eams[:3]:

        print(
            f"  sample={item['sample_id']} "
            f"tokens={item['num_tokens']} "
            f"activations={item['num_activations']}"
        )

    print()
    print(f"{name.upper()}: EAM BUILD PASSED")

    return eams


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("OLMoE EXPERT ACTIVATION MATRIX BUILDER")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for name, input_path in INPUT_FILES.items():

        process_workload(
            name,
            input_path
        )

    print()
    print("=" * 80)
    print("ALL EAMS BUILT SUCCESSFULLY")
    print("=" * 80)

    print()
    print("Created files:")

    for name in INPUT_FILES:

        print(
            f"  results/dataset/eams/"
            f"{name}_train_eams.jsonl"
        )


if __name__ == "__main__":
    main()