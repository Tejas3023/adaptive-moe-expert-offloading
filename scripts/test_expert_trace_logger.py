import torch

from src.tracing.expert_trace_logger import ExpertTraceLogger


def main():
    print("=" * 80)
    print("EXPERT TRACE LOGGER TEST")
    print("=" * 80)

    torch.manual_seed(42)

    # Simulate OLMoE router output.
    #
    # 5 tokens
    # 64 experts
    router_logits = torch.randn(5, 64)

    print("\nSynthetic router logits created.")
    print(f"Shape: {tuple(router_logits.shape)}")

    logger = ExpertTraceLogger(
        num_experts=64,
        top_k=8,
        norm_topk_prob=False,
    )

    traces = logger.process_router_logits(
        router_logits=router_logits,
        layer_id=0,
        sample_id=0,
    )

    print("\nRouting traces created.")
    print(f"Number of traces: {len(traces)}")

    print("\n" + "-" * 80)
    print("TOKEN ROUTING RESULTS")
    print("-" * 80)

    for trace in traces:
        print(f"\nLayer: {trace.layer_id}")
        print(f"Token position: {trace.token_position}")
        print(f"Selected experts: {trace.selected_experts}")

        formatted_weights = [
            round(weight, 6)
            for weight in trace.routing_weights
        ]

        print(f"Routing weights: {formatted_weights}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()