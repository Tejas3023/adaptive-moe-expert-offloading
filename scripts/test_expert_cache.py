from src.cache.expert_cache import ExpertCache


def main():
    print("=" * 80)
    print("EXPERT CACHE TEST")
    print("=" * 80)

    cache = ExpertCache(
        capacity=3
    )

    print("\nCache capacity:", cache.capacity)

    # This request sequence is designed to demonstrate:
    #
    # - Cache misses
    # - Cache hits
    # - LRU updates
    # - Cache evictions
    #
    request_sequence = [
        1,
        2,
        3,
        1,
        4,
        2,
        5,
        1,
    ]

    print("\nREQUEST SEQUENCE")
    print("-" * 80)
    print(request_sequence)

    print("\nPROCESSING REQUESTS")
    print("-" * 80)

    for expert_id in request_sequence:
        hit = cache.request(expert_id)

        result = "HIT" if hit else "MISS"

        print(
            f"Expert {expert_id}: "
            f"{result}"
        )

        print(
            "Cached experts "
            f"(LRU -> MRU): "
            f"{cache.cached_experts()}"
        )

    print("\nCACHE STATISTICS")
    print("-" * 80)

    statistics = cache.statistics()

    for key, value in statistics.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 80)
    print("EXPERT CACHE TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()