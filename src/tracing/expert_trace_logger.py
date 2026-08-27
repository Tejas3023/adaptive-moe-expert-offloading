from dataclasses import dataclass
from typing import List

import torch


@dataclass
class ExpertRoutingTrace:
    """
    Routing information for one token at one MoE layer.
    """

    layer_id: int
    token_position: int
    selected_experts: List[int]
    routing_weights: List[float]


class ExpertTraceLogger:
    """
    Converts OLMoE router logits into expert-routing traces.

    The logic matches the routing logic used in
    OlmoeSparseMoeBlock.forward().
    """

    def __init__(
        self,
        num_experts: int = 64,
        top_k: int = 8,
        norm_topk_prob: bool = False,
    ):
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob

        self.traces: List[ExpertRoutingTrace] = []

    def process_router_logits(
        self,
        router_logits: torch.Tensor,
        layer_id: int,
    ) -> List[ExpertRoutingTrace]:
        """
        Convert router logits into expert IDs and routing weights.

        Expected shape:
            (number_of_tokens, number_of_experts)

        For OLMoE:
            (batch_size * sequence_length, 64)
        """

        if router_logits.ndim != 2:
            raise ValueError(
                "router_logits must have shape "
                "(number_of_tokens, number_of_experts)"
            )

        if router_logits.shape[1] != self.num_experts:
            raise ValueError(
                f"Expected {self.num_experts} experts, "
                f"but received {router_logits.shape[1]}"
            )

        routing_weights = torch.softmax(
            router_logits,
            dim=1,
            dtype=torch.float,
        )

        routing_weights, selected_experts = torch.topk(
            routing_weights,
            self.top_k,
            dim=-1,
        )

        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1,
                keepdim=True,
            )

        layer_traces = []

        for token_position in range(router_logits.shape[0]):
            trace = ExpertRoutingTrace(
                layer_id=layer_id,
                token_position=token_position,
                selected_experts=selected_experts[
                    token_position
                ].tolist(),
                routing_weights=[
                    float(weight)
                    for weight in routing_weights[token_position].tolist()
                ],
            )

            self.traces.append(trace)
            layer_traces.append(trace)

        return layer_traces

    def get_traces(self) -> List[ExpertRoutingTrace]:
        """
        Return all collected traces.
        """

        return self.traces

    def clear(self) -> None:
        """
        Remove all stored traces.
        """

        self.traces.clear()

    def __len__(self) -> int:
        return len(self.traces)