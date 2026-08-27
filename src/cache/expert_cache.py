from collections import OrderedDict


class ExpertCache:
    """
    A simple LRU cache for MoE experts.

    The cache stores expert IDs and simulates whether an expert
    is currently resident in GPU memory.

    When the cache is full, the least recently used expert is evicted.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(
                "Cache capacity must be greater than 0."
            )

        self.capacity = capacity

        # OrderedDict allows us to implement LRU behavior.
        #
        # Left side  -> least recently used
        # Right side -> most recently used
        self.cache = OrderedDict()

        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def request(self, expert_id: int) -> bool:
        """
        Request an expert from the cache.

        Returns
        -------
        bool
            True  -> cache hit
            False -> cache miss
        """

        # Cache hit
        if expert_id in self.cache:
            self.hits += 1

            # Move the expert to the most recently used position.
            self.cache.move_to_end(expert_id)

            return True

        # Cache miss
        self.misses += 1

        # Cache is full: evict least recently used expert.
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
            self.evictions += 1

        # Add new expert as most recently used.
        self.cache[expert_id] = True

        return False

    def contains(self, expert_id: int) -> bool:
        """
        Check whether an expert is currently cached.
        """

        return expert_id in self.cache

    def cached_experts(self) -> list[int]:
        """
        Return cached experts from least recently used
        to most recently used.
        """

        return list(self.cache.keys())

    def size(self) -> int:
        """
        Return current number of cached experts.
        """

        return len(self.cache)

    def hit_rate(self) -> float:
        """
        Return cache hit rate as a percentage.
        """

        total_requests = self.hits + self.misses

        if total_requests == 0:
            return 0.0

        return (
            self.hits / total_requests
        ) * 100

    def statistics(self) -> dict:
        """
        Return cache statistics.
        """

        return {
            "capacity": self.capacity,
            "current_size": self.size(),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate_percent": self.hit_rate(),
        }

    def clear(self) -> None:
        """
        Clear the cache and reset statistics.
        """

        self.cache.clear()

        self.hits = 0
        self.misses = 0
        self.evictions = 0