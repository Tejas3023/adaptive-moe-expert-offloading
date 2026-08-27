import torch

from src.modeling.olmoe_loader import (
    load_tokenizer,
    load_olmoe_model,
)


def print_device_map(model):
    """
    Print how Accelerate distributed the model.
    """

    print("\n" + "=" * 80)
    print("MODEL DEVICE MAP")
    print("=" * 80)

    if not hasattr(model, "hf_device_map"):
        print("\nNo Hugging Face device map found.")
        return

    device_map = model.hf_device_map

    for module_name, device in device_map.items():
        print(f"{module_name}: {device}")


def print_memory_information():
    """
    Print GPU memory information.
    """

    print("\n" + "=" * 80)
    print("GPU MEMORY INFORMATION")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("\nCUDA is not available.")
        return

    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Allocated GPU memory: {allocated:.2f} GB")
    print(f"Reserved GPU memory: {reserved:.2f} GB")


def main():

    print("=" * 80)
    print("REAL OLMOE MODEL LOADING TEST")
    print("=" * 80)

    print("\nLoading tokenizer...")

    tokenizer = load_tokenizer()

    print("\nTokenizer test:")

    test_text = "The future of artificial intelligence is"

    encoded = tokenizer(test_text)

    print(f"Input text: {test_text}")
    print(f"Token count: {len(encoded['input_ids'])}")

    print("\n" + "-" * 80)
    print("LOADING REAL MODEL")
    print("-" * 80)

    model = load_olmoe_model()

    print_device_map(model)

    print_memory_information()

    print("\n" + "=" * 80)
    print("REAL OLMOE LOADING TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()