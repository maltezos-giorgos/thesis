"""Cross-encoder retrieval evaluator backed by sentence-transformers.

The model outputs a raw logit; sigmoid converts it to a probability in [0, 1].
Default thresholds (tunable after training):
    score >= 0.5          → CORRECT
    0.2 <= score < 0.5    → AMBIGUOUS
    score < 0.2           → INCORRECT
"""

from __future__ import annotations

import math

from tqdm import tqdm

from thesis_crag.evaluators.base import Action, RetrievalEvaluator


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class CrossEncoderEvaluator(RetrievalEvaluator):
    """Retrieval evaluator backed by a fine-tuned cross-encoder model."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        batch_size: int = 32,
        correct_threshold: float = 0.5,
        incorrect_threshold: float = 0.2,
    ) -> None:
        super().__init__(correct_threshold=correct_threshold, incorrect_threshold=incorrect_threshold)
        self.batch_size = batch_size
        from sentence_transformers import CrossEncoder  # lazy: optional at import time
        self._model = CrossEncoder(model_path, device=device, num_labels=1)

    def score(self, query: str, passage: str) -> float:
        """Return sigmoid probability in [0, 1] for a single query-passage pair."""
        raw = self._model.predict([[query, passage]])
        logit = float(raw[0]) if hasattr(raw, "__len__") else float(raw)
        return _sigmoid(logit)

    def score_batch(self, queries: list[str], passages: list[str]) -> list[float]:
        """Score query-passage pairs in batches; returns sigmoid probabilities."""
        if not queries:
            return []
        pairs = list(zip(queries, passages, strict=True))
        results: list[float] = []
        for i in tqdm(range(0, len(pairs), self.batch_size), desc="Scoring batches"):
            batch = pairs[i : i + self.batch_size]
            raw = self._model.predict(batch)
            results.extend(_sigmoid(float(r)) for r in raw)
        return results

    def classify_action(self, scores: list[float]) -> Action:
        return super().classify_action(scores)
