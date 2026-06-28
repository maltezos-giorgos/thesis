"""LLM-as-a-Judge retrieval evaluator (Architecture B).

Uses Claude Haiku to make two independent judgments per (query, passage) pair:
  - topic_match:         passage discusses the correct entity
  - answer_containment:  passage states or implies the answer

relevance = topic_match AND answer_containment

The evaluator maps the structured judgment to a scalar relevance score for
CRAG action classification:
  score = confidence          if relevant=True
  score = 1.0 - confidence   if relevant=False

CRAG action thresholds:
  score >= 0.7  →  CORRECT
  score  < 0.3  →  INCORRECT
  otherwise     →  AMBIGUOUS

All LLM calls are cached in SQLite, keyed by SHA-256(variant_name|query|passage).
Supports five prompt variants defined in prompts/llm_judge_variants.py.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from pathlib import Path

from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from thesis_crag.evaluators.base import Action, EvaluatorJudgment, RetrievalEvaluator
from thesis_crag.prompts.llm_judge_variants import VARIANTS
from thesis_crag.utils.llm_clients import call_llm_with_validation

logger = logging.getLogger(__name__)

# Haiku pricing (April 2025)
_INPUT_COST_PER_TOKEN = 0.80 / 1_000_000
_OUTPUT_COST_PER_TOKEN = 4.00 / 1_000_000
_EST_INPUT_TOKENS = 350   # system prompt + query + passage
_EST_OUTPUT_TOKENS = 80   # JSON response

# CRAG action thresholds (applied to relevance score in [0, 1])
_CORRECT_THRESHOLD = 0.7
_INCORRECT_THRESHOLD = 0.3


class _JudgeCache:
    """SQLite cache for LLM judgments.

    Key: SHA-256(variant_name + "|" + query + "|" + passage[:800])
    Cache hits avoid API calls across runs and retries.
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS judgments (key TEXT PRIMARY KEY, result TEXT)"
        )
        self._conn.commit()

    def _key(self, variant: str, query: str, passage: str) -> str:
        raw = variant + "|" + query + "|" + passage[:800]
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, variant: str, query: str, passage: str) -> dict | None:
        key = self._key(variant, query, passage)
        row = self._conn.execute(
            "SELECT result FROM judgments WHERE key=?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, variant: str, query: str, passage: str, result: dict) -> None:
        key = self._key(variant, query, passage)
        self._conn.execute(
            "INSERT OR REPLACE INTO judgments (key, result) VALUES (?, ?)",
            (key, json.dumps(result)),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def _judgment_to_float(judgment: EvaluatorJudgment) -> float:
    """Convert a structured judgment to a scalar relevance score in [0, 1].

    Interprets confidence as P(the model's determination is correct):
      relevant=True,  confidence=c → score = c       (confident it IS relevant)
      relevant=False, confidence=c → score = 1 - c   (confident it is NOT relevant)
    """
    return judgment.confidence if judgment.relevant else (1.0 - judgment.confidence)


class LLMJudgeEvaluator(RetrievalEvaluator):
    """Retrieval evaluator using Claude Haiku as a structured judge.

    Implements Architecture B from the thesis: dual topic_match +
    answer_containment judgments, with five prompt variants for ablation.
    """

    def __init__(
        self,
        prompt_variant: str = "BASE",
        model: str = "claude-haiku-4-5-20251001",
        cache_db: str = "data/cache/llm_judge.db",
    ) -> None:
        super().__init__(
            correct_threshold=_CORRECT_THRESHOLD,
            incorrect_threshold=_INCORRECT_THRESHOLD,
        )
        if prompt_variant not in VARIANTS:
            raise ValueError(
                f"Unknown prompt variant {prompt_variant!r}. "
                f"Valid options: {sorted(VARIANTS)}"
            )
        self.prompt_variant = prompt_variant
        self.model = model
        self._system_prompt = VARIANTS[prompt_variant]
        self._cache = _JudgeCache(Path(cache_db))
        self._api_calls: int = 0
        self._estimated_cost: float = 0.0

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((ValueError, ValidationError)),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def score(self, query: str, passage: str) -> EvaluatorJudgment:  # type: ignore[override]
        """Return a structured dual judgment for one query-passage pair.

        Retries up to 3 times on malformed JSON (ValueError) or schema
        validation errors (ValidationError) with exponential backoff.
        Results are cached on disk to avoid redundant API calls.
        """
        cached = self._cache.get(self.prompt_variant, query, passage)
        if cached is not None:
            logger.debug("Cache hit for query %r", query[:60])
            return EvaluatorJudgment(**cached)

        user = f"Question: {query}\nPassage: {passage[:800]}"
        raw = call_llm_with_validation(
            self._system_prompt, user, model=self.model
        )
        judgment = EvaluatorJudgment(**raw)  # raises ValidationError if schema wrong

        self._cache.put(self.prompt_variant, query, passage, raw)
        self._api_calls += 1
        self._estimated_cost += (
            _EST_INPUT_TOKENS * _INPUT_COST_PER_TOKEN
            + _EST_OUTPUT_TOKENS * _OUTPUT_COST_PER_TOKEN
        )
        logger.debug(
            "Judged %r | relevant=%s conf=%.2f",
            query[:60], judgment.relevant, judgment.confidence,
        )
        return judgment

    def score_batch(
        self, queries: list[str], passages: list[str]
    ) -> list[EvaluatorJudgment]:
        """Score a list of (query, passage) pairs with a tqdm progress bar."""
        if not queries:
            return []
        results: list[EvaluatorJudgment] = []
        for query, passage in tqdm(
            zip(queries, passages, strict=True),
            total=len(queries),
            desc=f"LLM judging [{self.prompt_variant}]",
        ):
            results.append(self.score(query, passage))
        return results

    # ------------------------------------------------------------------
    # CRAG action classification
    # ------------------------------------------------------------------

    def classify_action(  # type: ignore[override]
        self, scores: list[EvaluatorJudgment | float]
    ) -> Action:
        """Map a list of judgments (or floats) to a CRAG Action.

        Converts EvaluatorJudgment instances to scalar relevance scores,
        then takes the max across passages (same logic as the T5 baseline).
        """
        if not scores:
            return Action.INCORRECT
        float_scores = [
            _judgment_to_float(s) if isinstance(s, EvaluatorJudgment) else float(s)
            for s in scores
        ]
        best = max(float_scores)
        if best >= _CORRECT_THRESHOLD:
            return Action.CORRECT
        if best < _INCORRECT_THRESHOLD:
            return Action.INCORRECT
        return Action.AMBIGUOUS

    # ------------------------------------------------------------------
    # Cost tracking
    # ------------------------------------------------------------------

    @property
    def estimated_cost_usd(self) -> float:
        """Cumulative estimated API cost in USD (based on fixed token estimates)."""
        return self._estimated_cost

    @property
    def total_api_calls(self) -> int:
        """Number of new API calls made (cache hits not counted)."""
        return self._api_calls

    def close(self) -> None:
        self._cache.close()
