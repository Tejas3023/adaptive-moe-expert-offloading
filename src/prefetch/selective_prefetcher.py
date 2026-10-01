from src.prefetch.transition_prefetcher import TransitionPrefetcher


class SelectiveTransitionPrefetcher:
    """
    Adaptive transition-based prefetcher.

    Uses the transition predictor to generate candidate experts,
    calculates relative transition confidence across all observed
    candidates, and only prefetches candidates whose confidence
    exceeds the configured threshold.
    """

    def __init__(
        self,
        num_experts=64,
        top_k=8,
        max_prefetch_size=8,
        confidence_threshold=0.0,
    ):
        self.num_experts = num_experts
        self.top_k = top_k
        self.max_prefetch_size = max_prefetch_size
        self.confidence_threshold = confidence_threshold

        self.transition_prefetcher = TransitionPrefetcher(
            num_experts=num_experts,
            top_k=top_k,
            prefetch_size=max_prefetch_size,
        )

    def reset(self):
        """Reset the transition model and history."""
        self.transition_prefetcher.reset()

    def update(self, layer_id, current_experts):
        """
        Update the transition model using the experts activated
        by the current token.
        """
        self.transition_prefetcher.update(
            layer_id,
            current_experts
        )

    def predict(self, layer_id, current_experts):
        """
        Predict experts using transition confidence filtering.

        Returns:
            List of expert IDs passing the confidence threshold.
        """

        ranked = self.transition_prefetcher.get_all_transition_scores(
            layer_id,
            current_experts
        )

        if not ranked:
            return []

        total_score = sum(
            score for _, score in ranked
        )

        if total_score <= 0:
            return []

        selected = []

        for expert_id, score in ranked:

            confidence = score / total_score

            if confidence >= self.confidence_threshold:
                selected.append(expert_id)

            if len(selected) >= self.max_prefetch_size:
                break

        return selected

    def predict_with_scores(self, layer_id, current_experts):
        """
        Return selected predictions with their transition scores
        and relative confidence.

        Returns:
            List of:
                (expert_id, score, confidence)
        """

        ranked = self.transition_prefetcher.get_all_transition_scores(
            layer_id,
            current_experts
        )

        if not ranked:
            return []

        total_score = sum(
            score for _, score in ranked
        )

        if total_score <= 0:
            return []

        selected = []

        for expert_id, score in ranked:

            confidence = score / total_score

            if confidence >= self.confidence_threshold:
                selected.append(
                    (
                        expert_id,
                        score,
                        confidence
                    )
                )

            if len(selected) >= self.max_prefetch_size:
                break

        return selected

    def get_stats(self):
        """
        Return statistics from the underlying transition predictor.
        """

        stats = self.transition_prefetcher.get_stats()

        stats.update({
            "max_prefetch_size": self.max_prefetch_size,
            "confidence_threshold": self.confidence_threshold,
        })

        return stats