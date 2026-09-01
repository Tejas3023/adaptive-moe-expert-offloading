class CacheCostModel:
    """
    Simple latency model for expert movement.

    All costs are expressed in milliseconds.

    These are simulation parameters, not measurements
    of the actual hardware.
    """

    def __init__(
        self,
        gpu_hit_ms=0.05,
        cpu_to_gpu_ms=1.0,
        disk_to_gpu_ms=10.0,
        prefetch_ms=1.0,
    ):

        self.gpu_hit_ms = gpu_hit_ms
        self.cpu_to_gpu_ms = cpu_to_gpu_ms
        self.disk_to_gpu_ms = disk_to_gpu_ms
        self.prefetch_ms = prefetch_ms

    def hit_cost(self):
        """
        Cost when the requested expert is already
        resident in GPU memory.
        """

        return self.gpu_hit_ms

    def cpu_fetch_cost(self):
        """
        Cost of fetching an expert from CPU/RAM
        into GPU memory.
        """

        return self.cpu_to_gpu_ms

    def disk_fetch_cost(self):
        """
        Cost of fetching an expert from disk
        into GPU memory.
        """

        return self.disk_to_gpu_ms

    def prefetch_cost(self):
        """
        Cost associated with moving a predicted expert
        into GPU memory.
        """

        return self.prefetch_ms