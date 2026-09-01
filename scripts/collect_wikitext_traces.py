from pathlib import Path

import torch
from datasets import load_dataset

from src.modeling.olmoe_loader import (
    load_olmoe_model,
    load_tokenizer,
)
from src.tracing.expert_trace_logger import ExpertTraceLogger
from src.tracing.trace_writer import TraceWriter


MODEL_NAME = "allenai/OLMoE-1B-7B-0924"

DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_SPLIT = "test"

NUM_SAMPLES = 10

OUTPUT_FILE = Path(
    "results/traces/wikitext_10_samples.jsonl"
)


def main():

    print("=" * 80)
    print("WIKITEXT → REAL OLMOE EXPERT TRACE COLLECTION")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING WIKITEXT")
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

    # Remove empty WikiText records.
    samples = []

    for example in dataset:

        text = example["text"].strip()

        if text:
            samples.append(text)

        if len(samples) >= NUM_SAMPLES:
            break

    print(f"Non-empty samples selected: {len(samples)}")

    # ------------------------------------------------------------------
    # Load tokenizer
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING OLMOE TOKENIZER")
    print("=" * 80)

    tokenizer = load_tokenizer()

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING REAL OLMOE MODEL")
    print("=" * 80)

    model = load_olmoe_model()

    model.eval()

    # ------------------------------------------------------------------
    # Create trace logger
    # ------------------------------------------------------------------

    logger = ExpertTraceLogger(
        num_experts=64,
        top_k=8,
        norm_topk_prob=False,
    )

    total_tokens = 0

    # ------------------------------------------------------------------
    # Process samples one at a time
    # ------------------------------------------------------------------

    for sample_id, text in enumerate(samples):

        print()
        print("-" * 80)
        print(f"PROCESSING SAMPLE {sample_id + 1}/{len(samples)}")
        print("-" * 80)

        # Keep the prompt reasonably small.
        text = text[:2000]

        print()
        print("Text preview:")
        print(text[:300].replace("\n", " "))

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )

        input_ids = inputs["input_ids"]

        print()
        print(
            f"Token count: "
            f"{input_ids.shape[1]}"
        )

        total_tokens += input_ids.shape[1]

        # --------------------------------------------------------------
        # Real OLMoE forward pass
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Extract expert routing
        # --------------------------------------------------------------

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

        # Explicitly release the output tensors before next sample.
        del outputs
        del router_logits
        del inputs

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Write traces
    # ------------------------------------------------------------------

    traces = logger.get_traces()

    print()
    print("=" * 80)
    print("WRITING TRACES")
    print("=" * 80)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = TraceWriter(
        OUTPUT_FILE
    )

    written = writer.write(traces)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("TRACE COLLECTION SUMMARY")
    print("=" * 80)

    print()
    print(f"Dataset samples processed: {len(samples)}")
    print(f"Total input tokens: {total_tokens}")
    print(f"Transformer layers: 16")
    print(f"Experts per MoE layer: 64")
    print(f"Experts selected per token: 8")
    print(f"Total routing events: {len(traces)}")
    print(f"Total expert selections: {len(traces) * 8}")
    print(f"Traces written: {written}")

    print()
    print(f"Output file: {OUTPUT_FILE}")

    print()
    print("=" * 80)
    print("WIKITEXT TRACE COLLECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()