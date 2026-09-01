import time

import torch


def main():

    print("=" * 80)
    print("GPU TRANSFER BENCHMARK")
    print("=" * 80)

    if not torch.cuda.is_available():

        print("\nCUDA is not available.")
        return

    device = torch.device("cuda")

    print()
    print("GPU:")
    print(torch.cuda.get_device_name(0))

    sizes_mb = [
        1,
        10,
        50,
        100,
        250,
    ]

    repetitions = 50

    print()
    print(
        f"{'Size (MB)':<15}"
        f"{'Time (ms)':<15}"
        f"{'Bandwidth (GB/s)':<20}"
    )

    print("-" * 50)

    for size_mb in sizes_mb:

        num_elements = (
            size_mb * 1024 * 1024
        ) // 4

        cpu_tensor = torch.empty(
            num_elements,
            dtype=torch.float32,
            pin_memory=True,
        )

        gpu_tensor = torch.empty(
            num_elements,
            dtype=torch.float32,
            device=device,
        )

        # Warm-up
        gpu_tensor.copy_(
            cpu_tensor,
            non_blocking=True,
        )

        torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(repetitions):

            gpu_tensor.copy_(
                cpu_tensor,
                non_blocking=True,
            )

        torch.cuda.synchronize()

        elapsed = (
            time.perf_counter() -
            start
        )

        average_seconds = (
            elapsed /
            repetitions
        )

        average_ms = (
            average_seconds * 1000
        )

        bandwidth_gbs = (
            size_mb /
            1024 /
            average_seconds
        )

        print(
            f"{size_mb:<15}"
            f"{average_ms:<15.3f}"
            f"{bandwidth_gbs:<20.3f}"
        )

        del cpu_tensor
        del gpu_tensor

        torch.cuda.empty_cache()

    print()
    print("=" * 80)
    print("GPU TRANSFER BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()