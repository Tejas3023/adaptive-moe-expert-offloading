import torch

from src.modeling.olmoe_loader import load_tokenizer, load_olmoe_model
from src.tracing.expert_trace_logger import ExpertTraceLogger
from src.tracing.trace_writer import TraceWriter


def main():

    print("=" * 80)
    print("REAL OLMOE EXPERT ROUTING TRACE COLLECTION")
    print("=" * 80)

    # ------------------------------------------------------------
    # Load tokenizer
    # ------------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = load_tokenizer()

    # ------------------------------------------------------------
    # Load real model
    # ------------------------------------------------------------

    print("\nLoading real OLMoE model...")

    model = load_olmoe_model()

    model.eval()

    # ------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------

    prompt = "The future of artificial intelligence is"

    print("\n" + "=" * 80)
    print("INPUT PROMPT")
    print("=" * 80)

    print(prompt)

    # ------------------------------------------------------------
    # Tokenize input
    # ------------------------------------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    print("\nInput IDs:")

    for position, token_id in enumerate(inputs["input_ids"][0]):

        token_text = tokenizer.decode(
            [token_id.item()]
        )

        print(
            f"Position {position}: "
            f"Token ID = {token_id.item()}, "
            f"Token = {repr(token_text)}"
        )

    # ------------------------------------------------------------
    # Move inputs to embedding device
    # ------------------------------------------------------------

    embedding_device = model.hf_device_map[
        "model.embed_tokens"
    ]

    if isinstance(embedding_device, int):
        embedding_device = torch.device(
            f"cuda:{embedding_device}"
        )

    elif isinstance(embedding_device, str):
        embedding_device = torch.device(
            embedding_device
        )

    inputs = {
        key: value.to(embedding_device)
        for key, value in inputs.items()
    }

    # ------------------------------------------------------------
    # Run forward pass
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("RUNNING REAL OLMOE FORWARD PASS")
    print("=" * 80)

    with torch.no_grad():

        outputs = model(
            **inputs,
            output_router_logits=True
        )

    router_logits = outputs.router_logits

    print("\nForward pass complete.")

    print(
        f"Number of MoE layers: "
        f"{len(router_logits)}"
    )

    # ------------------------------------------------------------
    # Create trace logger
    # ------------------------------------------------------------

    trace_logger = ExpertTraceLogger(
        num_experts=64,
        top_k=8
    )

    all_traces = []

    # ------------------------------------------------------------
    # Convert router logits to expert routing traces
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("EXTRACTING EXPERT ROUTING TRACES")
    print("=" * 80)

    for layer_id, layer_router_logits in enumerate(
        router_logits
    ):

        layer_traces = trace_logger.log_routing(
            router_logits=layer_router_logits,
            layer_id=layer_id
        )

        all_traces.extend(
            layer_traces
        )

        print(
            f"Layer {layer_id}: "
            f"{len(layer_traces)} routing events collected"
        )

    # ------------------------------------------------------------
    # Write traces to disk
    # ------------------------------------------------------------

    output_file = (
        "results/traces/"
        "real_olmoe_single_prompt.jsonl"
    )

    trace_writer = TraceWriter()

    trace_writer.write_traces(
        traces=all_traces,
        output_file=output_file
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("TRACE COLLECTION SUMMARY")
    print("=" * 80)

    print(f"Prompt: {prompt}")
    print(f"Token count: {inputs['input_ids'].shape[1]}")
    print(f"MoE layers: {len(router_logits)}")
    print(f"Total routing events: {len(all_traces)}")

    print(
        f"Total expert selections: "
        f"{len(all_traces) * 8}"
    )

    print(f"\nTrace file saved to:")

    print(output_file)

    # ------------------------------------------------------------
    # Show example trace
    # ------------------------------------------------------------

    if all_traces:

        print("\n" + "=" * 80)
        print("EXAMPLE REAL ROUTING TRACE")
        print("=" * 80)

        print(all_traces[0])

    print("\n" + "=" * 80)
    print("REAL OLMOE TRACE COLLECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()