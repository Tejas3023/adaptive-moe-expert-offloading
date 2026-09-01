from collections import OrderedDict


class ExpertCache:
    """
    LRU cache for OLMoE experts.

    Each cached item is identified by:

        (layer_id, expert_id)

    because Expert 18 in Layer 0 is different from
    Expert 18 in Layer 7.
    """

    def __init__(self, capacity: int):

        if capacity <= 0:
            raise ValueError(
                "Cache capacity must be greater than 0."
            )

        self.capacity = capacity

        # Key:
        #     (layer_id, expert_id)
        #
        # Left  = least recently used
        # Right = most recently used
        self.cache = OrderedDict()

        self.hits = 0
        self.misses = 0
        self.evictions = 0

        self.prefetches = 0
        self.prefetch_hits = 0
        self.prefetch_evictions = 0

    def _key(self, layer_id, expert_id):
        return (layer_id, expert_id)

    def request(self, layer_id, expert_id):
        """
        Request an expert during actual model execution.

        Returns:
            True  -> cache hit
            False -> cache miss
        """

        key = self._key(layer_id, expert_id)

        if key in self.cache:

            self.hits += 1
            self.cache.move_to_end(key)

            return True

        self.misses += 1

        self._insert(
            key,
            prefetch=False
        )

        return False

    def prefetch(self, layer_id, expert_id):
        """
        Load an expert into the cache before it is requested.

        Prefetches do not count as normal cache requests.
        """

        key = self._key(layer_id, expert_id)

        self.prefetches += 1

        # Already resident.
        if key in self.cache:
            self.prefetch_hits += 1
            self.cache.move_to_end(key)
            return

        self._insert(
            key,
            prefetch=True
        )

    def _insert(self, key, prefetch=False):

        if len(self.cache) >= self.capacity:

            self.cache.popitem(
                last=False
            )

            self.evictions += 1

            if prefetch:
                self.prefetch_evictions += 1

        self.cache[key] = True

    def contains(self, layer_id, expert_id):

        return self._key(
            layer_id,
            expert_id
        ) in self.cache

    def cached_experts(self):

        return list(self.cache.keys())

    def size(self):

        return len(self.cache)

    def hit_rate(self):

        total = self.hits + self.misses

        if total == 0:
            return 0.0

        return (
            self.hits / total
        ) * 100

    def statistics(self):

        return {
            "capacity": self.capacity,
            "current_size": self.size(),

            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,

            "hit_rate_percent":
                self.hit_rate(),

            "prefetches":
                self.prefetches,

            "prefetch_hits":
                self.prefetch_hits,

            "prefetch_evictions":
                self.prefetch_evictions,
        }

    def clear(self):

        self.cache.clear()

        self.hits = 0
        self.misses = 0
        self.evictions = 0

        self.prefetches = 0
        self.prefetch_hits = 0
        self.prefetch_evictions = 0