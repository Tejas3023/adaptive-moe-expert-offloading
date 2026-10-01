from collections import defaultdict, Counter, deque


class TransitionPrefetcher:
    """
    First-order transition-based expert prefetcher.

    For each layer, learns which experts tend to appear
    after the experts selected for the previous token.

    Example:

        Previous token:
            [5, 10, 20]

        Next token:
            [5, 15, 20]

    This creates transitions such as:

        5  -> 5
        5  -> 15
        5  -> 20
        10 -> 5
        10 -> 15
        10 -> 20
        20 -> 5
        20 -> 15
        20 -> 20
    """

    def __init__(self, num_experts=64, top_k=8, prefetch_size=8):
        self.num_experts = num_experts
        self.top_k = top_k
        self.prefetch_size = prefetch_size

        # transition_counts[layer][current_expert][next_expert]
        #
        # Example:
        #
        # transition_counts[0][5][10] = 17
        #
        # means that in layer 0,
        # expert 10 appeared after expert 5
        # 17 times.
        self.transition_counts = defaultdict(
            lambda: defaultdict(Counter)
        )

        # Most recently observed token's experts for each layer.
        #
        # previous_experts[layer] = set/list of experts
        self.previous_experts = {}

        # Number of transitions learned.
        self.total_transitions = 0

    def reset(self):
        """
        Reset the learned transition model and history.
        """

        self.transition_counts.clear()
        self.previous_experts.clear()
        self.total_transitions = 0

    def update(self, layer_id, current_experts):
        """
        Observe the current token's selected experts.

        The previous token's experts are used to update
        the transition model.

        Parameters
        ----------
        layer_id : int
            Transformer layer ID.

        current_experts : iterable
            Experts selected for the current token.
        """

        current_experts = list(current_experts)

        # Remove duplicates while preserving order.
        current_experts = list(dict.fromkeys(current_experts))

        # If this is the first token for this layer,
        # there is no transition to learn yet.
        if layer_id in self.previous_experts:

            previous_experts = self.previous_experts[layer_id]

            for previous_expert in previous_experts:

                for current_expert in current_experts:

                    self.transition_counts[
                        layer_id
                    ][previous_expert][current_expert] += 1

                    self.total_transitions += 1

        # Current token becomes previous token
        # for the next prediction.
        self.previous_experts[layer_id] = current_experts

    def predict(self, layer_id, current_experts):
        """
        Predict the experts likely to be selected for
        the next token.

        IMPORTANT:
        This method does not update the model.

        Prediction must happen before observing the
        actual next token.

        Parameters
        ----------
        layer_id : int
            Transformer layer ID.

        current_experts : iterable
            Experts selected for the current token.

        Returns
        -------
        list
            Predicted expert IDs.
        """

        current_experts = list(current_experts)

        if not current_experts:
            return []

        scores = Counter()

        layer_transitions = self.transition_counts.get(layer_id, {})

        for current_expert in current_experts:

            next_experts = layer_transitions.get(
                current_expert,
                {}
            )

            for next_expert, count in next_experts.items():

                scores[next_expert] += count

        # No learned transitions yet.
        if not scores:
            return []

        # Rank:
        #
        # 1. Highest transition score
        # 2. Smaller expert ID as deterministic tie breaker
        ranked_experts = sorted(
            scores.items(),
            key=lambda x: (-x[1], x[0])
        )

        predictions = [
            expert_id
            for expert_id, _ in ranked_experts[:self.prefetch_size]
        ]

        return predictions

    def get_all_transition_scores(self, layer_id, current_experts):
        """
        Return transition scores for all experts observed after the
        current expert set.

        The score of a candidate expert is the sum of transition
        counts from all currently active experts.

        Returns:
            List of (expert_id, score), sorted by:
            1. descending score
            2. ascending expert_id
        """

        if layer_id not in self.transition_counts:
            return []

        scores = {}

        for current_expert in current_experts:

            if current_expert not in self.transition_counts[layer_id]:
                continue

            next_counts = self.transition_counts[layer_id][current_expert]

            for next_expert, count in next_counts.items():
                scores[next_expert] = (
                    scores.get(next_expert, 0) + count
                )

        ranked = sorted(
            scores.items(),
            key=lambda x: (-x[1], x[0])
        )

        return ranked

    def predict_with_scores(self, layer_id, current_experts):
        """
        Same as predict(), but returns transition scores.

        Useful for debugging and analysis.
        """

        current_experts = list(current_experts)

        scores = Counter()

        layer_transitions = self.transition_counts.get(layer_id, {})

        for current_expert in current_experts:

            next_experts = layer_transitions.get(
                current_expert,
                {}
            )

            for next_expert, count in next_experts.items():

                scores[next_expert] += count

        ranked = sorted(
            scores.items(),
            key=lambda x: (-x[1], x[0])
        )

        return ranked[:self.prefetch_size]

    def get_transition_count(
        self,
        layer_id,
        current_expert,
        next_expert
    ):
        """
        Return how many times a particular transition
        has been observed.
        """

        return self.transition_counts[
            layer_id
        ][current_expert][next_expert]

    def get_layer_transitions(self, layer_id):
        """
        Return the complete transition table for a layer.
        """

        return self.transition_counts.get(
            layer_id,
            {}
        )

    def get_stats(self):
        """
        Return basic predictor statistics.
        """

        return {
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "prefetch_size": self.prefetch_size,
            "layers_tracked": len(self.transition_counts),
            "total_transitions": self.total_transitions
        }