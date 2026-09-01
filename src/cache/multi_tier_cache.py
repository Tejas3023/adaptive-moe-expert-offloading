from collections import OrderedDict

from src.cache.cost_model import CacheCostModel


class MultiTierExpertCache:
    """
    Simulates a multi-tier OLMoE expert cache.

    Tier 1: GPU
    Tier 2: CPU/RAM
    Tier 3: Disk

    Each expert is identified by:

        (layer_id, expert_id)
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

        self.gpu_hits = 0
        self.cpu_hits = 0
        self.disk_fetches = 0

        self.gpu_evictions = 0
        self.cpu_evictions = 0

        self.prefetches = 0
        self.prefetch_hits = 0

        self.total_latency_ms = 0.0

    def _key(self, layer_id, expert_id):
        return (layer_id, expert_id)

    def _insert_gpu(self, key):
        if len(self.gpu) >= self.gpu_capacity:
            evicted = self.gpu.popitem(last=False)[0]
            self.gpu_evictions += 1

            # Evicted GPU expert moves to CPU.
            self._insert_cpu(evicted)

        self.gpu[key] = True

    def _insert_cpu(self, key):
        if len(self.cpu) >= self.cpu_capacity:
            self.cpu.popitem(last=False)
            self.cpu_evictions += 1

        self.cpu[key] = True

    def request(self, layer_id, expert_id):
        """
        Request an expert during model execution.

        Returns:
            "gpu"  -> GPU hit
            "cpu"  -> CPU hit
            "disk" -> disk fetch
        """

        key = self._key(layer_id, expert_id)

        # --------------------------------------------------
        # GPU HIT
        # --------------------------------------------------

        if key in self.gpu:

            self.gpu_hits += 1

            self.gpu.move_to_end(key)

            self.total_latency_ms += (
                self.cost_model.hit_cost()
            )

            return "gpu"

        # --------------------------------------------------
        # CPU HIT
        # --------------------------------------------------

        if key in self.cpu:

            self.cpu_hits += 1

            self.cpu.move_to_end(key)

            self.total_latency_ms += (
                self.cost_model.cpu_fetch_cost()
            )

            # Move CPU expert into GPU.
            self._insert_gpu(key)

            # Remove old CPU copy.
            self.cpu.pop(key, None)

            return "cpu"

        # --------------------------------------------------
        # DISK FETCH
        # --------------------------------------------------

        self.disk_fetches += 1

        self.total_latency_ms += (
            self.cost_model.disk_fetch_cost()
        )

        self._insert_gpu(key)

        return "disk"

    def prefetch(self, layer_id, expert_id):
        """
        Prefetch an expert into GPU memory.

        Returns:
            "gpu"  -> already in GPU
            "cpu"  -> moved from CPU
            "disk" -> loaded from disk
        """

        key = self._key(
            layer_id,
            expert_id,
        )

        self.prefetches += 1

        # Already in GPU.
        if key in self.gpu:

            self.prefetch_hits += 1

            self.gpu.move_to_end(key)

            return "gpu"

        # Expert exists in CPU.
        if key in self.cpu:

            self.cpu_hits += 0  # no normal request

            self.cpu.pop(key)

            self.total_latency_ms += (
                self.cost_model.cpu_fetch_cost()
            )

            self._insert_gpu(key)

            return "cpu"

        # Otherwise it must come from disk.
        self.total_latency_ms += (
            self.cost_model.disk_fetch_cost()
        )

        self._insert_gpu(key)

        return "disk"

    def statistics(self):

        total_requests = (
            self.gpu_hits
            + self.cpu_hits
            + self.disk_fetches
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

            "total_latency_ms": self.total_latency_ms,

            "average_latency_ms": (
                self.total_latency_ms / total_requests
                if total_requests
                else 0.0
            ),
        }

    def cached_gpu(self):
        return list(self.gpu.keys())

    def cached_cpu(self):
        return list(self.cpu.keys())

    def clear(self):

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