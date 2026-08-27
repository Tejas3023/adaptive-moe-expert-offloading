import shutil
import psutil
import torch


def bytes_to_gb(value):
    return value / (1024 ** 3)


def main():

    print("=" * 80)
    print("SYSTEM RESOURCE CHECK")
    print("=" * 80)

    # ------------------------------------------------------------
    # RAM
    # ------------------------------------------------------------

    memory = psutil.virtual_memory()

    print("\nSYSTEM MEMORY")
    print("-" * 80)

    print(
        f"Total RAM: "
        f"{bytes_to_gb(memory.total):.2f} GB"
    )

    print(
        f"Available RAM: "
        f"{bytes_to_gb(memory.available):.2f} GB"
    )

    print(
        f"Used RAM: "
        f"{bytes_to_gb(memory.used):.2f} GB"
    )

    print(
        f"RAM usage: "
        f"{memory.percent}%"
    )

    # ------------------------------------------------------------
    # DISK SPACE
    # ------------------------------------------------------------

    disk = shutil.disk_usage("C:\\")

    print("\nDISK SPACE")
    print("-" * 80)

    print(
        f"Total disk space: "
        f"{bytes_to_gb(disk.total):.2f} GB"
    )

    print(
        f"Free disk space: "
        f"{bytes_to_gb(disk.free):.2f} GB"
    )

    print(
        f"Used disk space: "
        f"{bytes_to_gb(disk.used):.2f} GB"
    )

    # ------------------------------------------------------------
    # GPU
    # ------------------------------------------------------------

    print("\nGPU")
    print("-" * 80)

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        device = torch.cuda.get_device_properties(0)

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"Total VRAM: "
            f"{bytes_to_gb(device.total_memory):.2f} GB"
        )

    # ------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("RESOURCE SUMMARY")
    print("=" * 80)

    available_ram = bytes_to_gb(memory.available)
    free_disk = bytes_to_gb(disk.free)

    print(
        f"Available RAM: {available_ram:.2f} GB"
    )

    print(
        f"Free disk space: {free_disk:.2f} GB"
    )

    if available_ram >= 16:
        print(
            "\nRAM status: GOOD"
        )
    elif available_ram >= 10:
        print(
            "\nRAM status: USABLE, "
            "but close other applications before loading."
        )
    else:
        print(
            "\nRAM status: LIMITED"
        )

    if free_disk >= 25:
        print(
            "Disk status: GOOD"
        )
    else:
        print(
            "Disk status: LOW"
        )

    print("\n" + "=" * 80)
    print("SYSTEM RESOURCE CHECK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()