
from collections import OrderedDict

from src.cache.cost_model import CacheCostModel


class MultiTierExpertCache:
    """
    Simulates a multi-tier OLMoE expert cache.

    Tier 1: GPU
    Tier 2: CPU/RAM
    Tier 3: Disk

    Each expert is identified by (layer_id, expert_id).

    Cost accounting:
        demand_latency_ms:
            Cost of servicing actual expert requests.

        prefetch_transfer_cost_ms:
            Cost of transferring predicted experts into GPU memory.

        total_latency_ms:
            Combined simulated demand and prefetch transfer cost.

    These values are simulation estimates, not measured
    end-to-end OLMoE inference latency.
    """

    def __init__(
        self,
        gpu_capacity,
        cpu_capacity,
        cost_model=None,
    ):
        if gpu_capacity <= 0:
            raise ValueError("gpu_capacity must be greater than 0.")

        if cpu_capacity <= 0:
            raise ValueError("cpu_capacity must be greater than 0.")

        self.gpu_capacity = gpu_capacity
        self.cpu_capacity = cpu_capacity

        self.cost_model = (
            cost_model
            if cost_model is not None
            else CacheCostModel()
        )

        self.gpu = OrderedDict()
        self.cpu = OrderedDict()

        # Demand request statistics.
        self.gpu_hits = 0
        self.cpu_hits = 0
        self.disk_fetches = 0

        # Cache eviction statistics.
        self.gpu_evictions = 0
        self.cpu_evictions = 0

        # Prefetch statistics.
        self.prefetches = 0
        self.prefetch_hits = 0

        # Separate simulated cost accounting.
        self.total_latency_ms = 0.0
        self.demand_latency_ms = 0.0
        self.prefetch_transfer_cost_ms = 0.0

    def _key(self, layer_id, expert_id):
        return (layer_id, expert_id)

    def _insert_gpu(self, key):
        """Insert an expert into GPU, evicting LRU if necessary."""
        if key in self.gpu:
            self.gpu.move_to_end(key)
            return

        if len(self.gpu) >= self.gpu_capacity:
            evicted_key = self.gpu.popitem(last=False)[0]
            self.gpu_evictions += 1
            self._insert_cpu(evicted_key)

        # An expert should not remain in both tiers.
        self.cpu.pop(key, None)
        self.gpu[key] = True

    def _insert_cpu(self, key):
        """Insert an expert into CPU, evicting LRU if necessary."""
        if key in self.cpu:
            self.cpu.move_to_end(key)
            return

        if len(self.cpu) >= self.cpu_capacity:
            self.cpu.popitem(last=False)
            self.cpu_evictions += 1

        self.cpu[key] = True

    def _record_demand_cost(self, cost):
        """Record the simulated cost of an actual request."""
        self.demand_latency_ms += cost
        self.total_latency_ms += cost

    def _record_prefetch_cost(self, cost):
        """Record the simulated cost of a prefetch transfer."""
        self.prefetch_transfer_cost_ms += cost
        self.total_latency_ms += cost

    def request(self, layer_id, expert_id):
        """
        Request an expert during model execution.

        Returns:
            "gpu": GPU hit
            "cpu": CPU hit; transfer to GPU
            "disk": Disk fetch; load into GPU
        """
        key = self._key(layer_id, expert_id)

        # GPU hit.
        if key in self.gpu:
            self.gpu_hits += 1
            self.gpu.move_to_end(key)

            self._record_demand_cost(
                self.cost_model.hit_cost()
            )
            return "gpu"

        # CPU hit.
        if key in self.cpu:
            self.cpu_hits += 1
            self.cpu.move_to_end(key)

            self._record_demand_cost(
                self.cost_model.cpu_fetch_cost()
            )

            self._insert_gpu(key)
            self.cpu.pop(key, None)
            return "cpu"

        # Disk miss.
        self.disk_fetches += 1

        self._record_demand_cost(
            self.cost_model.disk_fetch_cost()
        )

        self._insert_gpu(key)
        return "disk"

    def prefetch(self, layer_id, expert_id):
        """
        Prefetch an expert into GPU memory.

        Returns:
            "gpu": Already resident in GPU
            "cpu": Transferred from CPU to GPU
            "disk": Loaded from disk into GPU

        Prefetch transfers contribute to total simulated cost,
        but do not count as normal demand requests.
        """
        key = self._key(layer_id, expert_id)
        self.prefetches += 1

        # Already resident in GPU: no transfer cost.
        if key in self.gpu:
            self.prefetch_hits += 1
            self.gpu.move_to_end(key)
            return "gpu"

        # Resident in CPU: CPU-to-GPU transfer.
        if key in self.cpu:
            self.cpu.pop(key)

            self._record_prefetch_cost(
                self.cost_model.cpu_fetch_cost()
            )

            self._insert_gpu(key)
            return "cpu"

        # Not resident in either tier: disk-to-GPU transfer.
        self._record_prefetch_cost(
            self.cost_model.disk_fetch_cost()
        )

        self._insert_gpu(key)
        return "disk"

    def statistics(self):
        """Return cache counters and separated cost metrics."""
        total_requests = (
            self.gpu_hits
            + self.cpu_hits
            + self.disk_fetches
        )

        average_demand_cost = (
            self.demand_latency_ms / total_requests
            if total_requests
            else 0.0
        )

        prefetch_cost_per_request = (
            self.prefetch_transfer_cost_ms / total_requests
            if total_requests
            else 0.0
        )

        return {
            "gpu_capacity": self.gpu_capacity,
            "cpu_capacity": self.cpu_capacity,

            "gpu_hits": self.gpu_hits,
            "cpu_hits": self.cpu_hits,
            "disk_fetches": self.disk_fetches,

            "gpu_evictions": self.gpu_evictions,
            "cpu_evictions": self.cpu_evictions,

            "prefetches": self.prefetches,
            "prefetch_hits": self.prefetch_hits,

            "total_requests": total_requests,

            # Backward-compatible combined cost metrics.
            "total_latency_ms": self.total_latency_ms,
            "average_latency_ms": (
                self.total_latency_ms / total_requests
                if total_requests
                else 0.0
            ),

            # New separated cost metrics.
            "demand_latency_ms": self.demand_latency_ms,
            "prefetch_transfer_cost_ms": (
                self.prefetch_transfer_cost_ms
            ),
            "average_demand_latency_ms": average_demand_cost,
            "prefetch_transfer_cost_per_demand_request_ms": (
                prefetch_cost_per_request
            ),
        }

    def cached_gpu(self):
        """Return the keys currently resident in GPU."""
        return list(self.gpu.keys())

    def cached_cpu(self):
        """Return the keys currently resident in CPU."""
        return list(self.cpu.keys())

    def clear(self):
        """Clear cache contents and reset all statistics."""
        self.gpu.clear()
        self.cpu.clear()

        self.gpu_hits = 0
        self.cpu_hits = 0
        self.disk_fetches = 0

        self.gpu_evictions = 0
        self.cpu_evictions = 0

        self.prefetches = 0
        self.prefetch_hits = 0

        self.total_latency_ms = 0.0
        self.demand_latency_ms = 0.0
        self.prefetch_transfer_cost_ms = 0.0
