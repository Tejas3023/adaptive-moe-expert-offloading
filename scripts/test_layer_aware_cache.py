from src.cache.expert_cache import ExpertCache


def main():

    print("=" * 80)
    print("LAYER-AWARE EXPERT CACHE TEST")
    print("=" * 80)

    cache = ExpertCache(
        capacity=3
    )

    print("\nInitial cache:")
    print(cache.cached_experts())

    print("\nAdding experts:")

    cache.request(0, 5)
    cache.request(0, 10)
    cache.request(1, 5)

    print(
        "Cache:",
        cache.cached_experts()
    )

    print("\nRequesting Layer 0 Expert 5")

    hit = cache.request(
        0,
        5
    )

    print("Hit:", hit)

    print("\nRequesting Layer 1 Expert 5")

    hit = cache.request(
        1,
        5
    )

    print("Hit:", hit)

    print("\nRequesting Layer 2 Expert 5")

    hit = cache.request(
        2,
        5
    )

    print("Hit:", hit)

    print("\nFinal cache:")
    print(cache.cached_experts())

    print("\nStatistics:")

    for key, value in cache.statistics().items():
        print(
            f"{key}: {value}"
        )

    print()
    print("=" * 80)
    print("LAYER-AWARE CACHE TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()