from src.cache.latency_cache import LatencyAwareCache
from src.cache.cost_model import CacheCostModel


def main():

    print("=" * 80)
    print("LATENCY-AWARE CACHE TEST")
    print("=" * 80)

    cost_model = CacheCostModel(
        gpu_hit_ms=0.05,
        cpu_to_gpu_ms=1.0,
        disk_to_gpu_ms=10.0,
        prefetch_ms=1.0,
    )

    cache = LatencyAwareCache(
        capacity=2,
        cost_model=cost_model,
    )

    print("\nRequest Layer 0 Expert 5")

    print(
        "Hit:",
        cache.request(0, 5)
    )

    print("\nRequest Layer 0 Expert 5 again")

    print(
        "Hit:",
        cache.request(0, 5)
    )

    print("\nRequest Layer 1 Expert 5")

    print(
        "Hit:",
        cache.request(1, 5)
    )

    print("\nRequest Layer 2 Expert 5")

    print(
        "Hit:",
        cache.request(2, 5)
    )

    print("\nStatistics:")

    for key, value in cache.statistics().items():

        print(
            f"{key}: {value}"
        )

    print()
    print("=" * 80)
    print("LATENCY CACHE TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()