import torch

from src.modeling.olmoe_loader import (
    load_tokenizer,
    load_olmoe_model,
)


def print_gpu_memory():
    """
    Print current GPU memory usage.
    """

    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)

    print(f"Allocated GPU memory: {allocated:.2f} GB")
    print(f"Reserved GPU memory: {reserved:.2f} GB")


def main():

    print("=" * 80)
    print("REAL OLMOE FORWARD PASS TEST")
    print("=" * 80)

    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer()

    print("\nLoading model...")
    model = load_olmoe_model()

    prompt = "The future of artificial intelligence is"

    print("\n" + "=" * 80)
    print("INPUT")
    print("=" * 80)

    print(f"\nPrompt: {prompt}")

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    print(f"Input IDs shape: {inputs['input_ids'].shape}")

    # The input must initially go to the same device as the
    # embedding layer.
    embedding_device = model.hf_device_map.get(
        "model.embed_tokens",
        "cpu",
    )

    if isinstance(embedding_device, int):
        embedding_device = f"cuda:{embedding_device}"

    print(f"Embedding device: {embedding_device}")

    inputs = {
        key: value.to(embedding_device)
        for key, value in inputs.items()
    }

    print("\n" + "=" * 80)
    print("BEFORE FORWARD PASS")
    print("=" * 80)

    print_gpu_memory()

    print("\nRunning real OLMoE forward pass...")

    with torch.no_grad():

        outputs = model(
            **inputs,
            output_router_logits=True,
            use_cache=False,
        )

    print("\nForward pass completed successfully.")

    print("\n" + "=" * 80)
    print("MODEL OUTPUT")
    print("=" * 80)

    print(f"\nOutput logits shape: {outputs.logits.shape}")

    print("\n" + "=" * 80)
    print("ROUTER LOGITS")
    print("=" * 80)

    router_logits = outputs.router_logits

    if router_logits is None:
        print("\nRouter logits were not returned.")

    else:

        print(f"\nNumber of MoE router outputs: {len(router_logits)}")

        for layer_id, layer_router_logits in enumerate(router_logits):

            print(
                f"Layer {layer_id}: "
                f"shape = {tuple(layer_router_logits.shape)}"
            )

    print("\n" + "=" * 80)
    print("AFTER FORWARD PASS")
    print("=" * 80)

    print_gpu_memory()

    print("\n" + "=" * 80)
    print("REAL OLMOE FORWARD PASS TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()