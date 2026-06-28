"""Tests for EvaluatorJudgment and the Action enum."""


import pytest
from pydantic import ValidationError

from thesis_crag.evaluators.base import Action, EvaluatorJudgment

# ---------------------------------------------------------------------------
# 1. Valid construction
# ---------------------------------------------------------------------------

def test_valid_judgment_relevant():
    j = EvaluatorJudgment(
        topic_match=True,
        answer_containment=True,
        relevant=True,
        confidence=0.9,
        reasoning="The passage directly answers the question.",
    )
    assert j.relevant is True
    assert j.confidence == 0.9


def test_valid_judgment_irrelevant():
    j = EvaluatorJudgment(
        topic_match=False,
        answer_containment=False,
        relevant=False,
        confidence=0.1,
        reasoning="The passage is off-topic.",
    )
    assert j.relevant is False


# ---------------------------------------------------------------------------
# 2. Relevance-consistency validator
# ---------------------------------------------------------------------------

def test_inconsistent_relevant_raises():
    """relevant=True is invalid when topic_match=True but answer_containment=False."""
    with pytest.raises(ValidationError, match="inconsistent"):
        EvaluatorJudgment(
            topic_match=True,
            answer_containment=False,
            relevant=True,  # wrong: True AND False = False
            confidence=0.8,
            reasoning="Should fail.",
        )


def test_inconsistent_not_relevant_raises():
    """relevant=False is invalid when both topic_match and answer_containment are True."""
    with pytest.raises(ValidationError, match="inconsistent"):
        EvaluatorJudgment(
            topic_match=True,
            answer_containment=True,
            relevant=False,  # wrong: True AND True = True
            confidence=0.5,
            reasoning="Should also fail.",
        )


# ---------------------------------------------------------------------------
# 3. JSON serialisation roundtrip
# ---------------------------------------------------------------------------

def test_json_roundtrip():
    j = EvaluatorJudgment(
        topic_match=True,
        answer_containment=True,
        relevant=True,
        confidence=0.75,
        reasoning="Roundtrip check.",
    )
    raw = j.model_dump_json()
    restored = EvaluatorJudgment.model_validate_json(raw)
    assert restored == j


def test_json_roundtrip_dict():
    j = EvaluatorJudgment(
        topic_match=False,
        answer_containment=False,
        relevant=False,
        confidence=0.2,
        reasoning="Dict roundtrip.",
    )
    d = j.model_dump()
    restored = EvaluatorJudgment(**d)
    assert restored == j


# ---------------------------------------------------------------------------
# 4. Malformed / missing fields raise ValidationError
# ---------------------------------------------------------------------------

def test_missing_required_field_raises():
    with pytest.raises(ValidationError):
        EvaluatorJudgment(
            topic_match=True,
            # answer_containment missing
            relevant=True,
            confidence=0.9,
            reasoning="Missing field.",
        )


def test_wrong_type_raises():
    # Pydantic v2 coerces strings like "yes" to bool, but cannot coerce a dict.
    with pytest.raises(ValidationError):
        EvaluatorJudgment(
            topic_match={"not": "a bool"},  # dict cannot be coerced to bool
            answer_containment=True,
            relevant=True,
            confidence=0.9,
            reasoning="Wrong type.",
        )


# ---------------------------------------------------------------------------
# 5. Confidence clamping to [0, 1]
# ---------------------------------------------------------------------------

def test_confidence_clamped_above_one():
    j = EvaluatorJudgment(
        topic_match=True,
        answer_containment=True,
        relevant=True,
        confidence=1.5,  # should clamp to 1.0
        reasoning="Over-confident.",
    )
    assert j.confidence == 1.0


def test_confidence_clamped_below_zero():
    j = EvaluatorJudgment(
        topic_match=False,
        answer_containment=False,
        relevant=False,
        confidence=-0.3,  # should clamp to 0.0
        reasoning="Negative confidence.",
    )
    assert j.confidence == 0.0


# ---------------------------------------------------------------------------
# Action enum smoke test
# ---------------------------------------------------------------------------

def test_action_enum_values():
    assert Action.CORRECT == "CORRECT"
    assert Action.AMBIGUOUS == "AMBIGUOUS"
    assert Action.INCORRECT == "INCORRECT"
