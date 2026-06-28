"""Abstract base classes and shared data models for retrieval evaluators."""

from __future__ import annotations

import abc
from enum import StrEnum

from pydantic import BaseModel, field_validator, model_validator


class Action(StrEnum):
    CORRECT = "CORRECT"
    AMBIGUOUS = "AMBIGUOUS"
    INCORRECT = "INCORRECT"


class EvaluatorJudgment(BaseModel):
    """Structured output from any retrieval evaluator."""

    topic_match: bool
    answer_containment: bool
    relevant: bool
    confidence: float  # in [0, 1]
    reasoning: str

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @model_validator(mode="after")
    def check_relevance_consistency(self) -> EvaluatorJudgment:
        expected = self.topic_match and self.answer_containment
        if self.relevant != expected:
            raise ValueError(
                f"relevant={self.relevant} is inconsistent with "
                f"topic_match={self.topic_match} and answer_containment={self.answer_containment}. "
                f"relevant must equal (topic_match AND answer_containment)."
            )
        return self


class RetrievalEvaluator(abc.ABC):
    """Abstract retrieval evaluator.

    Subclasses implement score() for a single query-passage pair.
    classify_action() maps a list of per-passage scores to a CRAG Action.
    """

    def __init__(self, correct_threshold: float = 0.5, incorrect_threshold: float = 0.5) -> None:
        self.correct_threshold = correct_threshold
        self.incorrect_threshold = incorrect_threshold

    @abc.abstractmethod
    def score(self, query: str, passage: str) -> float:
        """Return a relevance score in [0, 1] for a single query-passage pair."""

    def classify_action(self, scores: list[float]) -> Action:
        """Map a list of passage scores to a CRAG Action.

        Uses the max score across passages. Scores above correct_threshold → CORRECT,
        below incorrect_threshold → INCORRECT, otherwise AMBIGUOUS.
        """
        if not scores:
            return Action.INCORRECT
        best = max(scores)
        if best >= self.correct_threshold:
            return Action.CORRECT
        if best < self.incorrect_threshold:
            return Action.INCORRECT
        return Action.AMBIGUOUS
