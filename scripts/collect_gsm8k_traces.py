from pathlib import Path
import sys

import torch
from datasets import load_dataset


# ==============================================================
# PROJECT ROOT
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==============================================================
# PROJECT IMPORTS
# ==============================================================

from src.modeling.olmoe_loader import (
    load_olmoe_model,
    load_tokenizer,
)

from src.tracing.expert_trace_logger import ExpertTraceLogger
from src.tracing.trace_writer import TraceWriter


# ==============================================================
# CONFIGURATION
# ==============================================================

MODEL_NAME = "allenai/OLMoE-1B-7B-0924"

DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"
DATASET_SPLIT = "test"

NUM_SAMPLES = 100
MAX_LENGTH = 256

OUTPUT_FILE = Path(
    "results/traces/gsm8k_100_samples.jsonl"
)


# ==============================================================
# MAIN
# ==============================================================

def main():

    print("=" * 80)
    print("GSM8K → REAL OLMOE EXPERT TRACE COLLECTION")
    print("=" * 80)

    # ==========================================================
    # LOAD GSM8K
    # ==========================================================

    print()
    print("=" * 80)
    print("LOADING GSM8K")
    print("=" * 80)

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
    )

    print()
    print(f"Dataset: {DATASET_NAME}")
    print(f"Configuration: {DATASET_CONFIG}")
    print(f"Split: {DATASET_SPLIT}")
    print(f"Total dataset examples: {len(dataset)}")

    # ==========================================================
    # SELECT QUESTIONS
    #
    # IMPORTANT:
    # Only the question is passed to OLMoE.
    # The answer/solution is NOT passed to the model.
    # ==========================================================

    samples = []

    for example in dataset:

        question = example["question"].strip()

        if question:
            samples.append(question)

        if len(samples) >= NUM_SAMPLES:
            break

    print()
    print(
        f"Non-empty questions selected: "
        f"{len(samples)}"
    )

    # ==========================================================
    # LOAD TOKENIZER
    # ==========================================================

    print()
    print("=" * 80)
    print("LOADING OLMOE TOKENIZER")
    print("=" * 80)

    tokenizer = load_tokenizer()

    # ==========================================================
    # LOAD REAL OLMOE MODEL
    # ==========================================================

    print()
    print("=" * 80)
    print("LOADING REAL OLMOE MODEL")
    print("=" * 80)

    model = load_olmoe_model()

    model.eval()

    # ==========================================================
    # CREATE TRACE LOGGER
    # ==========================================================

    logger = ExpertTraceLogger(
        num_experts=64,
        top_k=8,
        norm_topk_prob=False,
    )

    total_tokens = 0

    # ==========================================================
    # OUTPUT DIRECTORY
    # ==========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==========================================================
    # PROCESS QUESTIONS ONE AT A TIME
    # ==========================================================

    for sample_id, question in enumerate(samples):

        print()
        print("-" * 80)
        print(
            f"PROCESSING GSM8K QUESTION "
            f"{sample_id + 1}/{len(samples)}"
        )
        print("-" * 80)

        # ------------------------------------------------------
        # Keep workload size consistent with WikiText
        # ------------------------------------------------------

        question = question[:2000]

        print()
        print("Question preview:")
        print(
            question[:500]
            .replace("\n", " ")
        )

        # ------------------------------------------------------
        # Tokenize
        # ------------------------------------------------------

        inputs = tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        input_ids = inputs["input_ids"]

        token_count = input_ids.shape[1]

        print()
        print(
            f"Token count: {token_count}"
        )

        total_tokens += token_count

        # ------------------------------------------------------
        # Real OLMoE forward pass
        # ------------------------------------------------------

        print()
        print("Running real OLMoE forward pass...")

        with torch.no_grad():

            outputs = model(
                **inputs,
                output_router_logits=True,
            )

        router_logits = outputs.router_logits

        print(
            f"Router outputs: "
            f"{len(router_logits)} layers"
        )

        # ------------------------------------------------------
        # Extract expert routing
        # ------------------------------------------------------

        for layer_id, layer_router_logits in enumerate(
            router_logits
        ):

            logger.process_router_logits(
                router_logits=layer_router_logits,
                layer_id=layer_id,
                sample_id=sample_id,
            )

        print(
            f"Routing events collected so far: "
            f"{len(logger)}"
        )

        # ======================================================
        # INCREMENTAL SAVE
        # ======================================================
        #
        # Save after every successfully processed question.
        # This protects the experiment from losing all
        # previously collected traces if a later question
        # causes an OLMoE/CUDA/Accelerate failure.
        # ======================================================

        current_traces = logger.get_traces()

        progress_writer = TraceWriter(
            OUTPUT_FILE
        )

        written = progress_writer.write(
            current_traces
        )

        print(
            f"Progress saved: "
            f"{written} traces"
        )

        # ------------------------------------------------------
        # Release tensors before next question
        # ------------------------------------------------------

        del outputs
        del router_logits
        del inputs
        del input_ids

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ==========================================================
    # FINAL TRACE COLLECTION
    # ==========================================================

    traces = logger.get_traces()

    # ==========================================================
    # FINAL WRITE
    # ==========================================================

    print()
    print("=" * 80)
    print("WRITING FINAL GSM8K TRACES")
    print("=" * 80)

    writer = TraceWriter(
        OUTPUT_FILE
    )

    written = writer.write(
        traces
    )

    # ==========================================================
    # SUMMARY
    # ==========================================================

    print()
    print("=" * 80)
    print("GSM8K TRACE COLLECTION SUMMARY")
    print("=" * 80)

    print()
    print(
        f"Dataset samples processed: "
        f"{len(samples)}"
    )

    print(
        f"Total input tokens: "
        f"{total_tokens}"
    )

    print(
        "Transformer layers: 16"
    )

    print(
        "Experts per MoE layer: 64"
    )

    print(
        "Experts selected per token: 8"
    )

    print(
        f"Total routing events: "
        f"{len(traces)}"
    )

    print(
        f"Total expert selections: "
        f"{len(traces) * 8}"
    )

    print(
        f"Traces written: "
        f"{written}"
    )

    print()
    print(
        f"Output file: "
        f"{OUTPUT_FILE}"
    )

    # ==========================================================
    # FIRST 5 ROUTING TRACES
    # ==========================================================

    print()
    print("=" * 80)
    print("FIRST 5 GSM8K ROUTING TRACES")
    print("=" * 80)

    for trace in traces[:5]:

        print()

        print(
            f"Layer: {trace.layer_id}"
        )

        print(
            f"Sample ID: {trace.sample_id}"
        )

        print(
            f"Token position: "
            f"{trace.token_position}"
        )

        print(
            f"Selected experts: "
            f"{trace.selected_experts}"
        )

        print(
            f"Routing weights: "
            f"{[round(w, 6) for w in trace.routing_weights]}"
        )

    # ==========================================================
    # COMPLETE
    # ==========================================================

    print()
    print("=" * 80)
    print("GSM8K 100-SAMPLE TRACE COLLECTION COMPLETE")
    print("=" * 80)


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    main()