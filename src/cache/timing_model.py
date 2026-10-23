"""Capacity-aware event-based timing model for expert transfers.

This is a separate simulator; it does not modify MultiTierExpertCache.

Model assumptions:
- One serialized transfer channel shared by demand loads and prefetches.
- Fixed transfer durations supplied by CacheCostModel.
- Transfers may overlap with simulated computation.
- GPU and CPU tiers use LRU eviction and layer-aware (layer_id, expert_id) keys.
- Completed transfers enter GPU; GPU evictions move to CPU; CPU evictions go to disk.
- The model estimates simulated timing, not measured hardware inference latency.
"""

from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class TransferEvent:
    key: tuple
    source: str
    kind: str
    request_ms: float
    start_ms: float
    finish_ms: float

    @property
    def queue_wait_ms(self):
        return self.start_ms - self.request_ms

    @property
    def transfer_duration_ms(self):
        return self.finish_ms - self.start_ms


class TimingAwareSimulator:
    """Discrete-event transfer simulator with bounded GPU and CPU tiers."""

    def __init__(
        self,
        cost_model,
        gpu_capacity=256,
        cpu_capacity=170,
        initially_resident=None,
        initially_cpu=None,
    ):
        if gpu_capacity <= 0:
            raise ValueError("gpu_capacity must be greater than 0.")
        if cpu_capacity <= 0:
            raise ValueError("cpu_capacity must be greater than 0.")

        self.cost_model = cost_model
        self.gpu_capacity = gpu_capacity
        self.cpu_capacity = cpu_capacity
        self.current_time_ms = 0.0
        self.transfer_channel_free_ms = 0.0
        self.demand_waiting_on_prefetch_ms = 0.0
        self.demand_waiting_on_prefetch_count = 0

        self.demand_waiting_on_demand_transfer_ms = 0.0
        self.demand_waiting_on_demand_transfer_count = 0

        self.gpu = OrderedDict()
        self.cpu = OrderedDict()

        for key in initially_cpu or []:
            self._insert_cpu(tuple(key))
        for key in initially_resident or []:
            self._insert_gpu(tuple(key))

        self.in_flight = {}
        self.events = []

        self.prefetch_count = 0
        self.prefetch_deduplicated = 0

        self.demand_requests = 0
        self.demand_gpu_hits = 0
        self.demand_cpu_hits = 0
        self.demand_disk_misses = 0
        self.demand_transfer_count = 0

        self.gpu_evictions = 0
        self.cpu_evictions = 0

        self.total_stall_ms = 0.0
        self.total_transfer_ms = 0.0
        self.prefetch_transfer_ms = 0.0
        self.demand_transfer_ms = 0.0

        # Queue-wait diagnostics.
        self.prefetch_queue_wait_ms = 0.0
        self.demand_queue_wait_ms = 0.0
        self.prefetch_queue_wait_count = 0
        self.demand_queue_wait_count = 0

    def _duration(self, source):
        if source == "cpu":
            return float(self.cost_model.cpu_fetch_cost())
        if source == "disk":
            return float(self.cost_model.disk_fetch_cost())
        raise ValueError("source must be 'cpu' or 'disk'")

    def _insert_cpu(self, key):
        if key in self.gpu:
            return

        if key in self.cpu:
            self.cpu.move_to_end(key)
            return

        if len(self.cpu) >= self.cpu_capacity:
            self.cpu.popitem(last=False)
            self.cpu_evictions += 1

        self.cpu[key] = True

    def _insert_gpu(self, key):
        if key in self.gpu:
            self.gpu.move_to_end(key)
            return

        if len(self.gpu) >= self.gpu_capacity:
            evicted_key = self.gpu.popitem(last=False)[0]
            self.gpu_evictions += 1
            self._insert_cpu(evicted_key)

        self.cpu.pop(key, None)
        self.gpu[key] = True

    def _complete_transfers_until(self, time_ms):
        completed = sorted(
            (
                event
                for event in self.in_flight.values()
                if event.finish_ms <= time_ms
            ),
            key=lambda event: (
                event.finish_ms,
                event.start_ms,
                event.key,
            ),
        )

        for event in completed:
            self.in_flight.pop(event.key, None)
            self._insert_gpu(event.key)

    def advance_compute(self, duration_ms):
        """Advance simulated wall-clock time while transfers progress."""
        if duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")

        self.current_time_ms += float(duration_ms)
        self._complete_transfers_until(self.current_time_ms)

        return self.current_time_ms

    def prefetch(self, layer_id, expert_id, source=None):
        """Schedule a non-blocking prefetch from the current tier or disk."""
        key = (layer_id, expert_id)
        self.prefetch_count += 1

        self._complete_transfers_until(self.current_time_ms)

        if key in self.gpu:
            self.gpu.move_to_end(key)
            return {
                "status": "already_resident",
                "key": key,
                "event": None,
            }

        if key in self.in_flight:
            self.prefetch_deduplicated += 1
            return {
                "status": "already_in_flight",
                "key": key,
                "event": self.in_flight[key],
            }

        if source is None:
            source = "cpu" if key in self.cpu else "disk"

        if source not in ("cpu", "disk"):
            raise ValueError("source must be 'cpu' or 'disk'")

        if source == "cpu":
            if key not in self.cpu:
                raise ValueError(
                    f"Cannot prefetch {key} from CPU: key is absent."
                )
            self.cpu.move_to_end(key)

        duration = self._duration(source)
        request_time = self.current_time_ms
        start = max(request_time, self.transfer_channel_free_ms)

        event = TransferEvent(
            key=key,
            source=source,
            kind="prefetch",
            request_ms=request_time,
            start_ms=start,
            finish_ms=start + duration,
        )

        self.prefetch_queue_wait_ms += event.queue_wait_ms
        self.prefetch_queue_wait_count += 1

        self.in_flight[key] = event
        self.events.append(event)

        self.transfer_channel_free_ms = event.finish_ms
        self.total_transfer_ms += duration
        self.prefetch_transfer_ms += duration

        return {
            "status": "scheduled",
            "key": key,
            "event": event,
        }

    def request(self, layer_id, expert_id, source_on_miss="disk"):
        """Serve a demand request, waiting only when the expert is not ready."""
        key = (layer_id, expert_id)
        self.demand_requests += 1

        self._complete_transfers_until(self.current_time_ms)

        if key in self.gpu:
            self.demand_gpu_hits += 1
            self.gpu.move_to_end(key)

            access_ms = float(self.cost_model.hit_cost())
            start_time = self.current_time_ms

            self.current_time_ms += access_ms
            self._complete_transfers_until(self.current_time_ms)

            return {
                "status": "gpu",
                "key": key,
                "request_time_ms": start_time,
                "stall_ms": 0.0,
                "access_ms": access_ms,
                "finish_time_ms": self.current_time_ms,
            }

        request_time = self.current_time_ms
        event = self.in_flight.get(key)

        # Diagnose demand requests that join a transfer already in progress.
        # A newly scheduled demand transfer is not counted as waiting on an
        # existing transfer; its channel queue wait is recorded separately.
        if event is not None:
            wait_ms = max(0.0, event.finish_ms - request_time)
            if event.kind == "prefetch":
                self.demand_waiting_on_prefetch_ms += wait_ms
                if wait_ms > 0.0:
                    self.demand_waiting_on_prefetch_count += 1
            elif event.kind == "demand":
                self.demand_waiting_on_demand_transfer_ms += wait_ms
                if wait_ms > 0.0:
                    self.demand_waiting_on_demand_transfer_count += 1

        if event is None:
            if key in self.cpu:
                source = "cpu"
                self.cpu.move_to_end(key)
                self.demand_cpu_hits += 1
            else:
                source = source_on_miss
                if source != "disk":
                    raise ValueError(
                        "source_on_miss must be 'disk' for a disk miss"
                    )
                self.demand_disk_misses += 1

            duration = self._duration(source)
            start = max(
                self.current_time_ms,
                self.transfer_channel_free_ms,
            )

            event = TransferEvent(
                key=key,
                source=source,
                kind="demand",
                request_ms=request_time,
                start_ms=start,
                finish_ms=start + duration,
            )

            self.demand_queue_wait_ms += event.queue_wait_ms
            self.demand_queue_wait_count += 1

            self.in_flight[key] = event
            self.events.append(event)

            self.transfer_channel_free_ms = event.finish_ms
            self.total_transfer_ms += duration
            self.demand_transfer_ms += duration
            self.demand_transfer_count += 1

        stall = max(0.0, event.finish_ms - request_time)
        self.total_stall_ms += stall

        self.current_time_ms = max(
            self.current_time_ms,
            event.finish_ms,
        )
        self._complete_transfers_until(self.current_time_ms)

        return {
            "status": (
                "prefetch_wait"
                if event.kind == "prefetch"
                else "miss"
            ),
            "key": key,
            "request_time_ms": request_time,
            "stall_ms": stall,
            "access_ms": 0.0,
            "finish_time_ms": self.current_time_ms,
        }

    def statistics(self):
        return {
            "simulated_time_ms": self.current_time_ms,
            "gpu_capacity": self.gpu_capacity,
            "cpu_capacity": self.cpu_capacity,
            "gpu_resident": len(self.gpu),
            "cpu_resident": len(self.cpu),

            "demand_requests": self.demand_requests,
            "demand_gpu_hits": self.demand_gpu_hits,
            "demand_cpu_hits": self.demand_cpu_hits,
            "demand_disk_misses": self.demand_disk_misses,
            "demand_transfer_count": self.demand_transfer_count,

            "prefetch_count": self.prefetch_count,
            "prefetch_deduplicated": self.prefetch_deduplicated,

            "gpu_evictions": self.gpu_evictions,
            "cpu_evictions": self.cpu_evictions,

            "total_stall_ms": self.total_stall_ms,
            "total_transfer_ms": self.total_transfer_ms,
            "prefetch_transfer_ms": self.prefetch_transfer_ms,
            "demand_transfer_ms": self.demand_transfer_ms,

            "transfer_events": len(self.events),
            "in_flight_transfers": len(self.in_flight),

            # Queue-wait diagnostics.
            "prefetch_queue_wait_ms": self.prefetch_queue_wait_ms,
            "demand_queue_wait_ms": self.demand_queue_wait_ms,
            "prefetch_queue_wait_count": self.prefetch_queue_wait_count,
            "demand_queue_wait_count": self.demand_queue_wait_count,

            "avg_prefetch_queue_wait_ms": (
                self.prefetch_queue_wait_ms
                / self.prefetch_queue_wait_count
                if self.prefetch_queue_wait_count
                else 0.0
            ),
            "avg_demand_queue_wait_ms": (
                self.demand_queue_wait_ms
                / self.demand_queue_wait_count
                if self.demand_queue_wait_count
                else 0.0
            ),

            # Demand requests that joined transfers already in progress.
            "demand_waiting_on_prefetch_ms": (
                self.demand_waiting_on_prefetch_ms
            ),
            "demand_waiting_on_prefetch_count": (
                self.demand_waiting_on_prefetch_count
            ),
            "demand_waiting_on_demand_transfer_ms": (
                self.demand_waiting_on_demand_transfer_ms
            ),
            "demand_waiting_on_demand_transfer_count": (
                self.demand_waiting_on_demand_transfer_count
            ),
        }

    def cached_gpu(self):
        return list(self.gpu.keys())

    def cached_cpu(self):
        return list(self.cpu.keys())
