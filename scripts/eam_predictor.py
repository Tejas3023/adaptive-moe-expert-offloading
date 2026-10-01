import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIG
# ============================================================

TRAIN_EAM_FILE = Path(
    "results/dataset/eams/wikitext_train_eams.jsonl"
)

TRAIN_TRACE_FILE = Path(
    "results/dataset/wikitext_train.jsonl"
)

TEST_TRACE_FILE = Path(
    "results/dataset/wikitext_test.jsonl"
)

NUM_LAYERS = 16
NUM_EXPERTS = 64
TOP_K = 8

NUM_MATCHES = 5

# Test prediction point
OBSERVED_TOKENS = 10


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path):

    with path.open("r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


# ============================================================
# GROUP ROUTING TRACES BY SAMPLE
# ============================================================

def group_by_sample(traces):

    samples = defaultdict(list)

    for trace in traces:
        samples[trace["sample_id"]].append(trace)

    for sample_id in samples:
        samples[sample_id].sort(
            key=lambda x: (
                x["token_position"],
                x["layer_id"]
            )
        )

    return dict(samples)


# ============================================================
# BUILD EAM
# ============================================================

def build_eam(traces):

    eam = np.zeros(
        (NUM_LAYERS, NUM_EXPERTS),
        dtype=np.float32
    )

    for trace in traces:

        layer = trace["layer_id"]

        for expert in trace["selected_experts"]:
            eam[layer, expert] += 1

    return eam


# ============================================================
# BUILD PARTIAL EAM
# ============================================================

def build_partial_eam(
    sample_traces,
    observed_tokens
):

    observed = [
        trace
        for trace in sample_traces
        if trace["token_position"] < observed_tokens
    ]

    return build_eam(observed)


# ============================================================
# FIND SIMILAR HISTORICAL EAMS
# ============================================================

def find_similar_eams(
    query_eam,
    train_eams,
    train_sample_ids,
    top_k
):

    query = query_eam.reshape(1, -1)

    database = np.asarray(
        [
            eam.reshape(-1)
            for eam in train_eams
        ],
        dtype=np.float32
    )

    similarities = cosine_similarity(
        query,
        database
    )[0]

    # Highest similarity first
    indices = np.argsort(
        similarities
    )[::-1][:top_k]

    return [
        {
            "sample_id": train_sample_ids[i],
            "similarity": float(similarities[i])
        }
        for i in indices
    ]


# ============================================================
# PREDICT FUTURE EXPERTS
# ============================================================

def predict_future_experts(
    matches,
    train_samples,
    current_token,
    current_layer
):

    scores = np.zeros(
        NUM_EXPERTS,
        dtype=np.float64
    )

    for match in matches:

        sample_id = match["sample_id"]
        similarity = match["similarity"]

        for trace in train_samples[sample_id]:

            token = trace["token_position"]
            layer = trace["layer_id"]

            # Only information that would occur AFTER
            # the current prediction point.
            if token > current_token:

                for expert in trace["selected_experts"]:
                    scores[expert] += similarity

            elif (
                token == current_token
                and layer > current_layer
            ):

                for expert in trace["selected_experts"]:
                    scores[expert] += similarity

    # Highest-scoring experts first
    ranking = np.argsort(scores)[::-1]

    return [
        {
            "expert_id": int(expert),
            "score": float(scores[expert])
        }
        for expert in ranking
        if scores[expert] > 0
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("MoE-INFINITY-STYLE EAM PREDICTOR")
    print("=" * 80)

    # --------------------------------------------------------
    # Load training EAMs
    # --------------------------------------------------------

    eam_data = load_jsonl(
        TRAIN_EAM_FILE
    )

    if len(eam_data) != 80:
        raise ValueError(
            f"Expected 80 training EAMs, "
            f"found {len(eam_data)}"
        )

    train_eams = [
        np.asarray(
            item["eam"],
            dtype=np.float32
        )
        for item in eam_data
    ]

    train_sample_ids = [
        item["sample_id"]
        for item in eam_data
    ]

    # Validate EAM shape
    for eam in train_eams:

        if eam.shape != (
            NUM_LAYERS,
            NUM_EXPERTS
        ):
            raise ValueError(
                f"Invalid EAM shape: {eam.shape}"
            )

    print(
        f"Training EAMs: {len(train_eams)}"
    )

    print(
        f"EAM shape: "
        f"{NUM_LAYERS} × {NUM_EXPERTS}"
    )

    # --------------------------------------------------------
    # Load routing traces
    # --------------------------------------------------------

    train_traces = load_jsonl(
        TRAIN_TRACE_FILE
    )

    test_traces = load_jsonl(
        TEST_TRACE_FILE
    )

    train_samples = group_by_sample(
        train_traces
    )

    test_samples = group_by_sample(
        test_traces
    )

    print(
        f"Training samples: "
        f"{len(train_samples)}"
    )

    print(
        f"Test samples: "
        f"{len(test_samples)}"
    )

    # --------------------------------------------------------
    # Select first TEST sample
    # --------------------------------------------------------

    test_sample_id = sorted(
        test_samples.keys()
    )[0]

    sample_traces = test_samples[
        test_sample_id
    ]

    num_tokens = len({
        trace["token_position"]
        for trace in sample_traces
    })

    print()
    print(
        f"Test sample: {test_sample_id}"
    )

    print(
        f"Test tokens: {num_tokens}"
    )

    # --------------------------------------------------------
    # Prediction point
    # --------------------------------------------------------

    observed_tokens = min(
        OBSERVED_TOKENS,
        num_tokens - 1
    )

    current_token = observed_tokens
    current_layer = 0

    print()
    print(
        f"Observed tokens: "
        f"0 → {observed_tokens - 1}"
    )

    print(
        f"Prediction point: "
        f"token {current_token}, "
        f"layer {current_layer}"
    )

    # --------------------------------------------------------
    # Build partial EAM
    # --------------------------------------------------------

    partial_eam = build_partial_eam(
        sample_traces,
        observed_tokens
    )

    observed_activations = int(
        partial_eam.sum()
    )

    expected_activations = (
        observed_tokens
        * NUM_LAYERS
        * TOP_K
    )

    print()
    print(
        f"Observed EAM activations: "
        f"{observed_activations}"
    )

    print(
        f"Expected activations: "
        f"{expected_activations}"
    )

    if observed_activations != expected_activations:
        raise ValueError(
            "Partial EAM validation failed."
        )

    print(
        "Partial EAM validation: PASSED"
    )

    # --------------------------------------------------------
    # Similarity search
    # --------------------------------------------------------

    matches = find_similar_eams(
        partial_eam,
        train_eams,
        train_sample_ids,
        NUM_MATCHES
    )

    print()
    print(
        f"Top {NUM_MATCHES} historical matches:"
    )

    for rank, match in enumerate(
        matches,
        start=1
    ):

        print(
            f"  {rank}. "
            f"sample={match['sample_id']} "
            f"similarity="
            f"{match['similarity']:.6f}"
        )

    # --------------------------------------------------------
    # Predict future experts
    # --------------------------------------------------------

    predictions = predict_future_experts(
        matches,
        train_samples,
        current_token,
        current_layer
    )

    print()
    print("Top 10 predicted future experts:")

    for rank, prediction in enumerate(
        predictions[:10],
        start=1
    ):

        print(
            f"  {rank}. "
            f"expert={prediction['expert_id']} "
            f"score="
            f"{prediction['score']:.6f}"
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if not predictions:

        raise RuntimeError(
            "No future expert predictions generated."
        )

    for prediction in predictions:

        if not (
            0 <= prediction["expert_id"]
            < NUM_EXPERTS
        ):
            raise RuntimeError(
                "Invalid expert ID in prediction."
            )

    print()
    print("=" * 80)
    print("EAM PREDICTOR TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()