from collections import Counter
from pathlib import Path
from typing import Any

from src.tracing.trace_writer import TraceWriter


class TraceAnalyzer:
    """
    Analyzes expert routing traces produced by the OLMoE model.

    The analyzer calculates statistics such as:

    - Total routing events
    - Total expert selections
    - Number of unique experts used
    - Expert usage frequency
    - Most frequently selected experts
    """

    def __init__(self, trace_path: str | Path):
        self.trace_path = Path(trace_path)

        writer = TraceWriter(self.trace_path)

        self.traces = writer.read()

    def total_routing_events(self) -> int:
        """
        Return the number of routing events.

        One routing event corresponds to one token
        processed by one MoE layer.
        """

        return len(self.traces)

    def total_expert_selections(self) -> int:
        """
        Return the total number of expert selections.

        Each routing event in OLMoE selects top-k experts.
        For this model, top-k = 8.
        """

        return sum(
            len(trace["selected_experts"])
            for trace in self.traces
        )

    def expert_usage_counts(self) -> Counter:
        """
        Count how many times each expert was selected.
        """

        counter = Counter()

        for trace in self.traces:
            counter.update(
                trace["selected_experts"]
            )

        return counter

    def unique_experts_used(self) -> int:
        """
        Return the number of unique experts selected.
        """

        return len(
            self.expert_usage_counts()
        )

    def most_used_experts(
        self,
        top_n: int = 10,
    ) -> list[tuple[int, int]]:
        """
        Return the most frequently selected experts.

        Returns
        -------
        List of:
            (expert_id, selection_count)
        """

        return self.expert_usage_counts().most_common(
            top_n
        )

    def expert_usage_percentage(self) -> dict[int, float]:
        """
        Calculate the percentage of total expert selections
        accounted for by each expert.
        """

        total = self.total_expert_selections()

        if total == 0:
            return {}

        counts = self.expert_usage_counts()

        return {
            expert_id: (count / total) * 100
            for expert_id, count in counts.items()
        }

    def summary(self) -> dict[str, Any]:
        """
        Return a complete summary of the trace.
        """

        counts = self.expert_usage_counts()

        return {
            "trace_file": str(self.trace_path),
            "total_routing_events": (
                self.total_routing_events()
            ),
            "total_expert_selections": (
                self.total_expert_selections()
            ),
            "unique_experts_used": (
                self.unique_experts_used()
            ),
            "most_used_experts": (
                self.most_used_experts()
            ),
            "expert_usage_counts": dict(counts),
        }