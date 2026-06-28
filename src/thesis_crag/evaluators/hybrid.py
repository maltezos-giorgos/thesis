"""Hybrid Two-Stage Retrieval Evaluator (Architecture C).

Stage 1 — Cross-Encoder (fast, local):
  score >= high_threshold (0.8)  → CORRECT immediately, no LLM call
  score <= low_threshold  (0.2)  → INCORRECT immediately, no LLM call

Stage 2 — LLM Judge (slow, accurate) invoked only for the ambiguous band:
  0.2 < score < 0.8              → escalate to LLM, return its judgment

Routing rationale (from Phase 4A ablation):
  Cross-encoder: 91.9% trap detection, 78.7% gold recall — strong at rejecting traps.
  LLM judge:     85.3% trap detection, 91.9% gold recall — strong at confirming gold.
  The ambiguous band covers passages the CE is unsure about; LLM handles those.
"""

from __future__ import annotations

from thesis_crag.evaluators.base import Action, RetrievalEvaluator
from thesis_crag.evaluators.cross_encoder import CrossEncoderEvaluator
from thesis_crag.evaluators.llm_judge import LLMJudgeEvaluator, _judgment_to_float


class HybridEvaluator(RetrievalEvaluator):
    """Two-stage evaluator: cross-encoder for clear cases, LLM for ambiguous ones."""

    def __init__(
        self,
        cross_encoder_path: str,
        llm_prompt_variant: str = "WITH_NEGATIVE_EXAMPLES",
        high_threshold: float = 0.8,
        low_threshold: float = 0.2,
        device: str = "cpu",
        cache_db: str = "data/cache/llm_judge.db",
    ) -> None:
        # Use 0.5 / 0.2 for classify_action — the hybrid output is already
        # well-separated (CE >= 0.8 or <= 0.2; LLM confidence typically >= 0.7)
        super().__init__(correct_threshold=0.5, incorrect_threshold=0.2)
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self._ce  = CrossEncoderEvaluator(cross_encoder_path, device=device)
        self._llm = LLMJudgeEvaluator(prompt_variant=llm_prompt_variant, cache_db=cache_db)
        # Routing statistics
        self.total_queries:      int = 0
        self.cross_encoder_only: int = 0
        self.llm_called:         int = 0
        self._last_router:       str = "ce"  # updated after each score() call

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def score(self, query: str, passage: str) -> float:
        """Return a relevance score in [0, 1], routing to LLM only when ambiguous.

        Values >= high_threshold or <= low_threshold are returned directly from
        the cross-encoder. Scores in the ambiguous band are resolved by the LLM
        judge (using its judgment confidence as a probability proxy).
        """
        self.total_queries += 1
        ce_score = self._ce.score(query, passage)
        if ce_score >= self.high_threshold or ce_score <= self.low_threshold:
            self.cross_encoder_only += 1
            self._last_router = "ce"
            return ce_score
        self.llm_called += 1
        self._last_router = "llm"
        judgment = self._llm.score(query, passage)
        return _judgment_to_float(judgment)

    def classify_action(self, scores: list[float]) -> Action:
        return super().classify_action(scores)

    # ------------------------------------------------------------------
    # Routing statistics
    # ------------------------------------------------------------------

    @property
    def llm_route_rate(self) -> float:
        """Fraction of score() calls that were escalated to the LLM."""
        return self.llm_called / self.total_queries if self.total_queries else 0.0

    @property
    def estimated_cost_usd(self) -> float:
        """Cumulative LLM cost (cache hits count as $0)."""
        return self._llm.estimated_cost_usd

    @property
    def total_llm_api_calls(self) -> int:
        """New (non-cached) LLM API calls made during this run."""
        return self._llm.total_api_calls

    def close(self) -> None:
        self._llm.close()
