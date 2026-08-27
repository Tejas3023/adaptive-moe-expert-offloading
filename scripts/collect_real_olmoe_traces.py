import json
from pathlib import Path

import torch

from src.modeling.olmoe_loader import (
    load_olmoe_model,
    load_tokenizer,
)
from src.tracing.expert_trace_logger import ExpertTraceLogger
from src.tracing.trace_writer import TraceWriter


MODEL_NAME = "allenai/OLMoE-1B-7B-0924"

PROMPT = (
    "The future of artificial intelligence is closely connected "
    "to efficient machine learning systems."
)

OUTPUT_FILE = Path(
    "results/traces/real_olmoe_trace.jsonl"
)


def main():

    print("=" * 80)
    print("REAL OLMOE EXPERT TRACE COLLECTION")
    print("=" * 80)

    print()
    print("Loading tokenizer...")

    tokenizer = load_tokenizer()

    print()
    print("Loading real OLMoE model...")

    model = load_olmoe_model()

    model.eval()

    print()
    print("=" * 80)
    print("INPUT")
    print("=" * 80)

    print()
    print(f"Prompt: {PROMPT}")

    inputs = tokenizer(
        PROMPT,
        return_tensors="pt",
    )

    print(
        f"Input IDs shape: "
        f"{inputs['input_ids'].shape}"
    )

    print()

    print("=" * 80)
    print("RUNNING REAL OLMOE FORWARD PASS")
    print("=" * 80)

    with torch.no_grad():

        outputs = model(
            **inputs,
            output_router_logits=True,
        )

    router_logits = outputs.router_logits

    print()
    print("Forward pass completed.")

    print(
        f"Number of router outputs: "
        f"{len(router_logits)}"
    )

    print()

    print("=" * 80)
    print("PROCESSING EXPERT ROUTING")
    print("=" * 80)

    logger = ExpertTraceLogger(
        num_experts=64,
        top_k=8,
        norm_topk_prob=False,
    )

    for layer_id, layer_router_logits in enumerate(
        router_logits
    ):

        print(
            f"Processing layer {layer_id} "
            f"with shape "
            f"{tuple(layer_router_logits.shape)}"
        )

        logger.process_router_logits(
            router_logits=layer_router_logits,
            layer_id=layer_id,
        )

    traces = logger.get_traces()

    print()

    print(
        f"Total routing events collected: "
        f"{len(traces)}"
    )

    print()

    print("=" * 80)
    print("WRITING TRACES TO DISK")
    print("=" * 80)

    writer = TraceWriter(
        OUTPUT_FILE
    )

    writer.write(
        traces
    )

    print()
    print(
        f"Trace file written to: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Total traces written: "
        f"{len(traces)}"
    )

    print()

    print("=" * 80)
    print("TRACE SUMMARY")
    print("=" * 80)

    print()

    print(
        f"Layers processed: "
        f"{len(router_logits)}"
    )

    print(
        f"Tokens per layer: "
        f"{router_logits[0].shape[0]}"
    )

    print(
        f"Experts selected per token: "
        f"{logger.top_k}"
    )

    expected_events = (
        len(router_logits)
        * router_logits[0].shape[0]
    )

    print(
        f"Expected routing events: "
        f"{expected_events}"
    )

    print(
        f"Actual routing events: "
        f"{len(traces)}"
    )

    print()

    print("=" * 80)
    print("FIRST 5 ROUTING TRACES")
    print("=" * 80)

    for trace in traces[:5]:

        print()

        print(
            f"Layer: {trace.layer_id}"
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

    print()

    print("=" * 80)
    print("REAL OLMOE TRACE COLLECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()