from collections import Counter, deque


class HistoryPrefetcher:
    """
    Predicts the next token's experts using recent token-level
    expert-routing history.
    """

    def __init__(self, history_size=8, prefetch_size=8):
        if history_size <= 0:
            raise ValueError("history_size must be greater than 0.")

        if prefetch_size <= 0:
            raise ValueError("prefetch_size must be greater than 0.")

        self.history_size = history_size
        self.prefetch_size = prefetch_size

        # Each entry represents ONE token and contains
        # the experts selected for that token.
        self.history = deque(maxlen=history_size)

    def observe(self, experts):
        """
        Record one token's expert selections.
        """

        self.history.append(
            tuple(experts)
        )

    def predict(self):
        """
        Predict experts for the next token.

        Experts are ranked by how frequently they appeared
        across the recent token history.
        """

        if not self.history:
            return []

        counts = Counter()

        for experts in self.history:
            counts.update(experts)

        predictions = sorted(
            counts,
            key=lambda expert: (
                -counts[expert],
                expert,
            ),
        )

        return predictions[:self.prefetch_size]

    def history_contents(self):
        """
        Return recent token-level expert history.
        """

        return [
            list(experts)
            for experts in self.history
        ]

    def clear(self):
        """Clear routing history."""
        self.history.clear()

    def __len__(self):
        """Return number of tokens currently in history."""
        return len(self.history)