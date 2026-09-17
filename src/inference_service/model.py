"""A tiny deterministic model so the project can focus on serving infrastructure."""

import math
import time
from typing import Dict, List, Sequence


class ToyModel:
    """Scores numeric feature vectors with a fixed logistic regression model."""

    version = "toy-logistic-v1"

    def __init__(self, latency_ms: int = 20) -> None:
        self.latency_seconds = latency_ms / 1000

    def predict_batch(self, batch: Sequence[Sequence[float]]) -> List[Dict[str, object]]:
        # Sleeping once per batch mimics a fixed accelerator/kernel launch cost.
        # More requests in the batch therefore improve throughput.
        if self.latency_seconds:
            time.sleep(self.latency_seconds)

        predictions = []
        for values in batch:
            weighted_sum = sum((index + 1) * value for index, value in enumerate(values))
            logit = weighted_sum / max(len(values), 1)
            probability = _stable_sigmoid(logit)
            predictions.append(
                {
                    "label": "positive" if probability >= 0.5 else "negative",
                    "score": round(probability, 6),
                    "model_version": self.version,
                }
            )
        return predictions


def _stable_sigmoid(value: float) -> float:
    if value >= 0:
        negative_exp = math.exp(-value)
        return 1 / (1 + negative_exp)
    positive_exp = math.exp(value)
    return positive_exp / (1 + positive_exp)
