import torch


def main():
    print("=" * 60)
    print("PYTORCH AND GPU INFORMATION")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")

        device_count = torch.cuda.device_count()
        print(f"Number of GPUs: {device_count}")

        for i in range(device_count):
            properties = torch.cuda.get_device_properties(i)

            print("\nGPU INFORMATION")
            print("-" * 60)
            print(f"GPU ID: {i}")
            print(f"Name: {properties.name}")

            total_memory_gb = properties.total_memory / (1024 ** 3)

            print(f"Total VRAM: {total_memory_gb:.2f} GB")
            print(f"Compute capability: "
                  f"{properties.major}.{properties.minor}")

            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)

            print(f"Allocated memory: {allocated:.2f} GB")
            print(f"Reserved memory: {reserved:.2f} GB")

    else:
        print("\nCUDA GPU NOT DETECTED.")
        print("PyTorch is currently using CPU.")


if __name__ == "__main__":
    main()