import torch

from src.modeling.olmoe_loader import load_olmoe_model


def main():

    print("=" * 80)
    print("OLMOE EXPERT SIZE INSPECTION")
    print("=" * 80)

    model = load_olmoe_model()

    model.eval()

    print()
    print("Inspecting MoE layers...")

    total_expert_params = 0

    for layer_id, layer in enumerate(
        model.model.layers
    ):

        moe = layer.mlp

        experts = moe.experts

        print()
        print(
            f"Layer {layer_id}:"
        )

        print(
            f"Number of experts: "
            f"{len(experts)}"
        )

        # Inspect first expert.
        expert = experts[0]

        expert_params = sum(
            parameter.numel()
            for parameter in expert.parameters()
        )

        expert_bytes = sum(
            parameter.numel()
            * parameter.element_size()
            for parameter in expert.parameters()
        )

        expert_mb = (
            expert_bytes /
            (1024 ** 2)
        )

        print(
            f"Expert 0 parameters: "
            f"{expert_params:,}"
        )

        print(
            f"Expert 0 size: "
            f"{expert_mb:.2f} MB"
        )

        total_expert_params += (
            expert_params *
            len(experts)
        )

        # Only inspect the first layer.
        # All layers should use the same expert
        # architecture for this model.
        break

    print()
    print("-" * 80)
    print("EXPERT SUMMARY")
    print("-" * 80)

    print(
        f"Parameters per expert: "
        f"{expert_params:,}"
    )

    print(
        f"Expert size: "
        f"{expert_mb:.2f} MB"
    )

    print(
        f"Experts per layer: "
        f"{len(experts)}"
    )

    print(
        f"Estimated expert memory per layer: "
        f"{expert_mb * len(experts):.2f} MB"
    )

    print()
    print("=" * 80)
    print("EXPERT SIZE INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()