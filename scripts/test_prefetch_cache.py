from src.cache.expert_cache import ExpertCache


def main():
    print("=" * 80)
    print("PREFETCH-AWARE CACHE TEST")
    print("=" * 80)

    cache = ExpertCache(capacity=3)

    print("\nInitial cache:")
    print(cache.cached_experts())

    print("\nPrefetch experts: 5, 10, 20")

    for expert in [5, 10, 20]:
        cache.prefetch(expert)

    print("Cache:", cache.cached_experts())

    print("\nRequest expert 10")
    print("Hit:", cache.request(10))

    print("\nRequest expert 30")
    print("Hit:", cache.request(30))

    print("\nFinal cache:")
    print(cache.cached_experts())

    print("\nStatistics:")
    for key, value in cache.statistics().items():
        print(f"{key}: {value}")

    print("\n" + "=" * 80)
    print("PREFETCH-AWARE CACHE TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()