from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import (
    MODEL_NAME,
    OFFLOAD_DIR,
    GPU_MEMORY_LIMIT,
    CPU_MEMORY_LIMIT,
)


def get_model_dtype():
    """
    Select the PyTorch dtype used for model loading.
    """

    if torch.cuda.is_available():
        return torch.float16

    return torch.float32


def create_offload_directory() -> Path:
    """
    Create the directory used for disk offloading.
    """

    OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

    return OFFLOAD_DIR


def load_tokenizer():
    """
    Load the tokenizer for the real OLMoE model.
    """

    print("=" * 80)
    print("LOADING OLMOE TOKENIZER")
    print("=" * 80)

    print(f"\nModel: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
    )

    print("Tokenizer loaded successfully.")

    return tokenizer


def load_olmoe_model():
    """
    Load the real OLMoE model using automatic device placement.

    The model may be split across:
        - GPU
        - CPU
        - disk offloading
    """

    print("=" * 80)
    print("LOADING REAL OLMOE MODEL")
    print("=" * 80)

    print(f"\nModel: {MODEL_NAME}")

    dtype = get_model_dtype()

    print(f"PyTorch dtype: {dtype}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory limit: {GPU_MEMORY_LIMIT}")
    else:
        print("CUDA is not available.")
        print("Model will not use GPU acceleration.")

    print(f"CPU memory limit: {CPU_MEMORY_LIMIT}")

    offload_dir = create_offload_directory()

    print(f"Disk offload directory: {offload_dir}")

    max_memory = {
        0: GPU_MEMORY_LIMIT,
        "cpu": CPU_MEMORY_LIMIT,
    }

    print("\nLoading model...")
    print("This may take time during the first run.")
    print("Model files may need to be downloaded.")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto",
        max_memory=max_memory,
        offload_folder=str(offload_dir),
        low_cpu_mem_usage=True,
    )

    model.eval()

    print("\nModel loaded successfully.")

    return model


def load_olmoe():
    """
    Load both the tokenizer and model.

    Returns:
        tokenizer, model
    """

    tokenizer = load_tokenizer()
    model = load_olmoe_model()

    return tokenizer, model
