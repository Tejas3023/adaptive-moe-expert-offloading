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

    The routing logic matches OLMoE's OlmoeSparseMoeBlock.forward():

        router_logits
            -> softmax
            -> top-k expert selection
            -> optional top-k probability normalization

    For OLMoE-1B-7B-0924:
        - 64 total experts
        - top 8 experts selected per token
        - 16 MoE layers
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
        Convert router logits from one OLMoE layer into routing traces.

        Expected shape:
            (number_of_tokens, number_of_experts)

        For our real OLMoE forward pass:
            (batch_size * sequence_length, 64)

        Example:
            (6, 64)

        means:
            6 tokens were processed by this layer,
            and the router produced scores for all
            64 experts for each token.
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

        # Move router logits to CPU before processing.
        #
        # The real model may return tensors located on different
        # devices because we are using GPU/CPU/disk offloading.
        # The trace logger only needs the numerical values, so
        # keeping these calculations on CPU is appropriate.
        router_logits = router_logits.detach().float().cpu()

        # Same routing logic used by OLMoE:
        #
        # Router scores -> probabilities
        routing_weights = torch.softmax(
            router_logits,
            dim=1,
        )

        # Select the top-k experts for every token.
        #
        # selected_experts:
        #     shape = (number_of_tokens, top_k)
        #
        # routing_weights:
        #     shape = (number_of_tokens, top_k)
        routing_weights, selected_experts = torch.topk(
            routing_weights,
            self.top_k,
            dim=-1,
        )

        # OLMoE configuration currently has:
        # norm_topk_prob = False
        #
        # We still keep this here so the logger correctly supports
        # models/configurations that normalize only the selected
        # top-k routing probabilities.
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
                    for weight in routing_weights[
                        token_position
                    ].tolist()
                ],
            )

            self.traces.append(trace)
            layer_traces.append(trace)

        return layer_traces

    def get_traces(self) -> List[ExpertRoutingTrace]:
        """
        Return all collected routing traces.
        """

        return self.traces

    def clear(self) -> None:
        """
        Remove all stored traces.
        """

        self.traces.clear()

    def __len__(self) -> int:
        """
        Return the number of routing events stored.
        """

        return len(self.traces)