from collections import OrderedDict

from src.cache.cost_model import CacheCostModel


class LatencyAwareCache:
    """
    LRU expert cache that also tracks simulated latency.

    Each expert is identified by:

        (layer_id, expert_id)
    """

    def __init__(
        self,
        capacity,
        cost_model=None,
    ):

        if capacity <= 0:
            raise ValueError(
                "Cache capacity must be greater than 0."
            )

        self.capacity = capacity

        self.cost_model = (
            cost_model
            if cost_model is not None
            else CacheCostModel()
        )

        self.cache = OrderedDict()

        self.hits = 0
        self.misses = 0
        self.evictions = 0

        self.prefetches = 0
        self.prefetch_hits = 0

        self.total_latency_ms = 0.0

    def _key(
        self,
        layer_id,
        expert_id,
    ):

        return (
            layer_id,
            expert_id,
        )

    def request(
        self,
        layer_id,
        expert_id,
    ):

        key = self._key(
            layer_id,
            expert_id,
        )

        # ---------------------------------------------
        # CACHE HIT
        # ---------------------------------------------

        if key in self.cache:

            self.hits += 1

            self.cache.move_to_end(
                key
            )

            self.total_latency_ms += (
                self.cost_model.hit_cost()
            )

            return True

        # ---------------------------------------------
        # CACHE MISS
        # ---------------------------------------------

        self.misses += 1

        self.total_latency_ms += (
            self.cost_model.cpu_fetch_cost()
        )

        self._insert(key)

        return False

    def prefetch(
        self,
        layer_id,
        expert_id,
    ):

        key = self._key(
            layer_id,
            expert_id,
        )

        self.prefetches += 1

        # Already cached.
        if key in self.cache:

            self.prefetch_hits += 1

            self.cache.move_to_end(
                key
            )

            return

        self.total_latency_ms += (
            self.cost_model.prefetch_cost()
        )

        self._insert(key)

    def _insert(self, key):

        if len(self.cache) >= self.capacity:

            self.cache.popitem(
                last=False
            )

            self.evictions += 1

        self.cache[key] = True

    def statistics(self):

        total_requests = (
            self.hits +
            self.misses
        )

        hit_rate = (
            self.hits /
            total_requests *
            100
            if total_requests
            else 0.0
        )

        return {
            "capacity":
                self.capacity,

            "hits":
                self.hits,

            "misses":
                self.misses,

            "evictions":
                self.evictions,

            "hit_rate_percent":
                hit_rate,

            "prefetches":
                self.prefetches,

            "prefetch_hits":
                self.prefetch_hits,

            "total_latency_ms":
                self.total_latency_ms,
        }