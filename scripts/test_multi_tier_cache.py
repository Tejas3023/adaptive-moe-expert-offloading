from src.cache.multi_tier_cache import MultiTierExpertCache


def main():

    print("=" * 80)
    print("MULTI-TIER EXPERT CACHE TEST")
    print("=" * 80)

    cache = MultiTierExpertCache(
        gpu_capacity=2,
        cpu_capacity=2,
    )

    print()
    print("Initial GPU cache:")
    print(cache.cached_gpu())

    print()
    print("Initial CPU cache:")
    print(cache.cached_cpu())

    print()
    print("Request Layer 0 Expert 5")
    print("Result:", cache.request(0, 5))

    print()
    print("Request Layer 0 Expert 10")
    print("Result:", cache.request(0, 10))

    print()
    print("GPU cache:")
    print(cache.cached_gpu())

    print()
    print("Request Layer 0 Expert 20")
    print("Result:", cache.request(0, 20))

    print()
    print("GPU cache:")
    print(cache.cached_gpu())

    print()
    print("CPU cache:")
    print(cache.cached_cpu())

    print()
    print("Request Layer 0 Expert 5 again")
    print("Result:", cache.request(0, 5))

    print()
    print("GPU cache:")
    print(cache.cached_gpu())

    print()
    print("CPU cache:")
    print(cache.cached_cpu())

    print()
    print("Statistics:")
    for key, value in cache.statistics().items():
        print(f"{key}: {value}")

    print()
    print("=" * 80)
    print("MULTI-TIER CACHE TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()