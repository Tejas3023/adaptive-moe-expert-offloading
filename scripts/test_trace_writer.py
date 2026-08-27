from pathlib import Path

import torch

from src.tracing.expert_trace_logger import ExpertTraceLogger
from src.tracing.trace_writer import TraceWriter


def main():
    print("=" * 80)
    print("TRACE WRITER TEST")
    print("=" * 80)

    # Use a fixed random seed so that the test is reproducible.
    torch.manual_seed(42)

    # Simulate router output:
    #
    # 5 tokens
    # 64 experts
    router_logits = torch.randn(5, 64)

    print("\nSynthetic router logits created.")
    print(f"Shape: {tuple(router_logits.shape)}")

    # Create the expert trace logger.
    logger = ExpertTraceLogger(
        num_experts=64,
        top_k=8,
        norm_topk_prob=False,
    )

    # Convert synthetic router logits into traces.
    traces = logger.process_router_logits(
        router_logits=router_logits,
        layer_id=0,
    )

    print("\nRouting traces created.")
    print(f"Number of traces: {len(traces)}")

    # Define where the trace file will be saved.
    output_path = Path(
        "results/traces/test_trace.jsonl"
    )

    # Create the writer.
    writer = TraceWriter(output_path)

    # Write traces to disk.
    written_count = writer.write(
        traces,
        append=False,
    )

    print("\nTraces written to disk.")
    print(f"Output file: {output_path}")
    print(f"Traces written: {written_count}")

    # Read the traces back from disk.
    loaded_traces = writer.read()

    print("\nTraces read back from disk.")
    print(f"Traces loaded: {len(loaded_traces)}")

    # Verify that what we saved matches what we loaded.
    if len(loaded_traces) != len(traces):
        raise RuntimeError(
            "Number of loaded traces does not match "
            "number of written traces."
        )

    print("\n" + "-" * 80)
    print("FIRST LOADED TRACE")
    print("-" * 80)

    print(loaded_traces[0])

    print("\n" + "=" * 80)
    print("TRACE WRITER TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()