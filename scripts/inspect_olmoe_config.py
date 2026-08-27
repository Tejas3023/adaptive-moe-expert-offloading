from transformers import AutoConfig


MODEL_NAME = "allenai/OLMoE-1B-7B-0924"


def print_separator():
    print("=" * 70)


def main():
    print_separator()
    print("OLMoE CONFIGURATION INSPECTION")
    print_separator()

    print(f"\nModel: {MODEL_NAME}")
    print("\nDownloading model configuration only...")

    config = AutoConfig.from_pretrained(MODEL_NAME)

    print("\nConfiguration downloaded successfully.")

    print_separator()
    print("GENERAL MODEL INFORMATION")
    print_separator()

    print(f"Model type: {config.model_type}")
    print(f"Architecture: {config.architectures}")
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Intermediate size: {config.intermediate_size}")
    print(f"Number of layers: {config.num_hidden_layers}")
    print(f"Number of attention heads: {config.num_attention_heads}")
    print(f"Max position embeddings: {config.max_position_embeddings}")

    print_separator()
    print("MIXTURE OF EXPERTS CONFIGURATION")
    print_separator()

    print(f"Total number of experts: {config.num_experts}")
    print(f"Experts selected per token: {config.num_experts_per_tok}")
    print(f"Router auxiliary loss coefficient: {config.router_aux_loss_coef}")
    print(f"Normalize top-k probabilities: {config.norm_topk_prob}")
    print(f"Output router logits: {config.output_router_logits}")

    print_separator()
    print("ROUTING INTERPRETATION")
    print_separator()

    print(
        f"\nFor every token entering an MoE layer, "
        f"the router scores {config.num_experts} experts."
    )

    print(
        f"The top {config.num_experts_per_tok} experts are selected "
        f"for that token."
    )

    print(
        "\nExample conceptual routing:"
    )

    print(
        f"Token -> Router -> Top-{config.num_experts_per_tok} Experts "
        f"from {config.num_experts}"
    )

    print_separator()
    print("INSPECTION COMPLETE")
    print_separator()


if __name__ == "__main__":
    main()