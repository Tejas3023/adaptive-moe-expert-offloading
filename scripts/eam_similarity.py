import json
from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

EAM_FILE = Path(
    "results/dataset/eams/wikitext_train_eams.jsonl"
)

NUM_LAYERS = 16
NUM_EXPERTS = 64


# ============================================================
# LOAD EAMS
# ============================================================

def load_eams(path):

    eams = []

    with path.open("r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if line:
                eams.append(json.loads(line))

    return eams


# ============================================================
# EAM → VECTOR
# ============================================================

def eam_to_vector(eam):

    matrix = np.array(
        eam["eam"],
        dtype=np.float32
    )

    # Validate shape
    if matrix.shape != (
        NUM_LAYERS,
        NUM_EXPERTS
    ):
        raise ValueError(
            f"Expected EAM shape "
            f"{NUM_LAYERS}x{NUM_EXPERTS}, "
            f"got {matrix.shape}"
        )

    # Flatten:
    # 16 × 64 → 1024
    return matrix.reshape(-1)


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(
        np.dot(a, b)
        / (norm_a * norm_b)
    )


# ============================================================
# EAMC CLASS
# ============================================================

class EAMC:

    def __init__(self, eams):

        self.eams = eams

        # Pre-compute flattened vectors
        self.vectors = []

        for eam in eams:

            vector = eam_to_vector(eam)

            self.vectors.append(vector)

        self.vectors = np.array(
            self.vectors,
            dtype=np.float32
        )

    # --------------------------------------------------------
    # Find most similar historical EAMs
    # --------------------------------------------------------

    def search(
        self,
        query_eam,
        top_k=5
    ):

        query_vector = eam_to_vector(
            query_eam
        )

        query_norm = np.linalg.norm(
            query_vector
        )

        if query_norm == 0:
            raise ValueError(
                "Query EAM contains no activations."
            )

        # Vectorized cosine similarity
        database_norms = np.linalg.norm(
            self.vectors,
            axis=1
        )

        similarities = (
            self.vectors @ query_vector
        ) / (
            database_norms * query_norm
        )

        # Sort highest similarity first
        indices = np.argsort(
            similarities
        )[::-1]

        indices = indices[:top_k]

        results = []

        for index in indices:

            results.append({
                "sample_id": self.eams[index]["sample_id"],
                "similarity": float(
                    similarities[index]
                )
            })

        return results


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 80)
    print("EAMC COSINE-SIMILARITY TEST")
    print("=" * 80)

    # --------------------------------------------------------
    # Load training EAMs
    # --------------------------------------------------------

    eams = load_eams(EAM_FILE)

    print(
        f"Loaded EAMs: {len(eams)}"
    )

    # --------------------------------------------------------
    # Build EAMC
    # --------------------------------------------------------

    database = EAMC(eams)

    print(
        f"EAMC matrix shape: "
        f"{database.vectors.shape}"
    )

    # Expected:
    # 80 EAMs × 1024 values

    expected_shape = (
        80,
        NUM_LAYERS * NUM_EXPERTS
    )

    if database.vectors.shape != expected_shape:

        raise ValueError(
            f"Expected database shape "
            f"{expected_shape}, "
            f"got {database.vectors.shape}"
        )

    print(
        f"Expected shape: "
        f"{expected_shape}"
    )

    # --------------------------------------------------------
    # Use first training EAM as a test query
    # --------------------------------------------------------
    #
    # This is ONLY a sanity test right now.
    #
    # A real evaluation will use the 20 held-out
    # test samples later.
    # --------------------------------------------------------

    query = eams[0]

    results = database.search(
        query,
        top_k=5
    )

    print()
    print(
        f"Query sample: "
        f"{query['sample_id']}"
    )

    print()
    print("Top 5 matches:")

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"  {rank}. "
            f"sample={result['sample_id']} "
            f"similarity="
            f"{result['similarity']:.6f}"
        )

    # --------------------------------------------------------
    # Self-similarity sanity check
    # --------------------------------------------------------

    top_result = results[0]

    if top_result["sample_id"] != query["sample_id"]:

        print()
        print(
            "WARNING: Query's own EAM "
            "was not the closest match."
        )

    else:

        print()
        print(
            "Self-similarity check: PASSED"
        )

        if abs(
            top_result["similarity"] - 1.0
        ) > 1e-5:

            raise ValueError(
                "Expected self-similarity "
                "to be approximately 1.0"
            )

    print()
    print("=" * 80)
    print("EAMC SIMILARITY TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()