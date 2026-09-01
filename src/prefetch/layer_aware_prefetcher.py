from collections import Counter, defaultdict, deque


class LayerAwareHistoryPrefetcher:
    """
    Predicts experts separately for each transformer layer
    using recent expert-routing history.
    """

    def __init__(self, history_size=8, prefetch_size=4):
        if history_size <= 0:
            raise ValueError("history_size must be greater than 0.")

        if prefetch_size <= 0:
            raise ValueError("prefetch_size must be greater than 0.")

        self.history_size = history_size
        self.prefetch_size = prefetch_size

        # layer_id -> recent token expert sets
        self.history = defaultdict(
            lambda: deque(maxlen=history_size)
        )

    def observe(self, layer_id, experts):
        """Record the experts used by one token in one layer."""

        self.history[layer_id].append(
            tuple(experts)
        )

    def predict(self, layer_id):
        """
        Predict experts for the next token in this layer.

        Experts are ranked by frequency in recent history.
        """

        history = self.history[layer_id]

        if not history:
            return []

        counts = Counter()

        for experts in history:
            counts.update(experts)

        predictions = sorted(
            counts,
            key=lambda expert: (
                -counts[expert],
                expert,
            ),
        )

        return predictions[:self.prefetch_size]

    def history_contents(self, layer_id):
        """Return recent history for one layer."""

        return [
            list(experts)
            for experts in self.history[layer_id]
        ]

    def clear(self):
        """Clear all layer histories."""

        self.history.clear()

    def __len__(self):
        """Return total number of history entries."""

        return sum(
            len(history)
            for history in self.history.values()
        )