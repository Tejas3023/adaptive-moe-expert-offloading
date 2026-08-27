from pathlib import Path

from src.cache.expert_cache import ExpertCache
from src.tracing.trace_writer import TraceWriter


class CacheBenchmark:
    """
    Replays expert routing traces through an expert cache.

    Each expert selected by a routing event is treated as
    a cache request.

    This allows us to measure:

    - Cache hits
    - Cache misses
    - Cache evictions
    - Cache hit rate
    """

    def __init__(
        self,
        trace_path: str | Path,
        cache_capacity: int,
    ):
        self.trace_path = Path(trace_path)

        self.cache = ExpertCache(
            capacity=cache_capacity
        )

        writer = TraceWriter(
            self.trace_path
        )

        self.traces = writer.read()

    def run(self) -> dict:
        """
        Replay all expert requests from the trace file.
        """

        for trace in self.traces:

            selected_experts = (
                trace["selected_experts"]
            )

            for expert_id in selected_experts:
                self.cache.request(
                    expert_id
                )

        return self.cache.statistics()

    def total_routing_events(self) -> int:
        """
        Return the number of routing events replayed.
        """

        return len(self.traces)

    def total_expert_requests(self) -> int:
        """
        Return the total number of expert cache requests.
        """

        return sum(
            len(trace["selected_experts"])
            for trace in self.traces
        )