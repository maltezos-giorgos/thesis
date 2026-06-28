"""Unit tests for LLMJudgeEvaluator — all API calls are mocked."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from thesis_crag.evaluators.base import Action, EvaluatorJudgment
from thesis_crag.evaluators.llm_judge import LLMJudgeEvaluator, _judgment_to_float
from thesis_crag.prompts.llm_judge_variants import VARIANTS

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

_RELEVANT_RESPONSE = {
    "topic_match": True,
    "answer_containment": True,
    "relevant": True,
    "confidence": 0.9,
    "reasoning": "The passage directly states the answer.",
}

_IRRELEVANT_RESPONSE = {
    "topic_match": False,
    "answer_containment": False,
    "relevant": False,
    "confidence": 0.85,
    "reasoning": "The passage is about a different entity with the same name.",
}


@pytest.fixture
def evaluator(tmp_path):
    """LLMJudgeEvaluator with a fresh temp-dir cache; no real API calls."""
    return LLMJudgeEvaluator(
        prompt_variant="BASE",
        cache_db=str(tmp_path / "test_judge.db"),
    )


# ---------------------------------------------------------------------------
# Test 1: score() returns EvaluatorJudgment
# ---------------------------------------------------------------------------

@patch("thesis_crag.evaluators.llm_judge.call_llm_with_validation")
def test_score_returns_judgment(mock_llm, evaluator):
    mock_llm.return_value = _RELEVANT_RESPONSE
    result = evaluator.score("What is X's occupation?", "X is a politician.")
    assert isinstance(result, EvaluatorJudgment)
    assert result.relevant is True
    assert result.topic_match is True
    assert result.answer_containment is True
    assert result.confidence == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Test 2: cache hit skips API call
# ---------------------------------------------------------------------------

@patch("thesis_crag.evaluators.llm_judge.call_llm_with_validation")
def test_cache_hit_skips_api_call(mock_llm, evaluator):
    mock_llm.return_value = _RELEVANT_RESPONSE
    evaluator.score("q?", "passage text")
    evaluator.score("q?", "passage text")  # identical — should hit cache
    assert mock_llm.call_count == 1
    assert evaluator.total_api_calls == 1


# ---------------------------------------------------------------------------
# Test 3: malformed JSON triggers retry via tenacity
# ---------------------------------------------------------------------------

@patch("thesis_crag.evaluators.llm_judge.call_llm_with_validation")
def test_malformed_json_triggers_retry(mock_llm, evaluator):
    mock_llm.side_effect = [ValueError("no JSON found in response"), _RELEVANT_RESPONSE]
    result = evaluator.score("q?", "passage")
    assert mock_llm.call_count == 2
    assert isinstance(result, EvaluatorJudgment)


# ---------------------------------------------------------------------------
# Test 4–7: classify_action boundaries at 0.69, 0.70, 0.30, 0.29
# ---------------------------------------------------------------------------

def _make_judgment(relevant: bool, confidence: float) -> EvaluatorJudgment:
    tm = relevant
    ac = relevant
    return EvaluatorJudgment(
        topic_match=tm,
        answer_containment=ac,
        relevant=relevant,
        confidence=confidence,
        reasoning="test",
    )


def test_classify_action_correct_at_threshold(evaluator):
    # score = confidence = 0.70 → exactly at threshold → CORRECT
    j = _make_judgment(relevant=True, confidence=0.70)
    assert evaluator.classify_action([j]) == Action.CORRECT


def test_classify_action_ambiguous_just_below(evaluator):
    # score = confidence = 0.69 → just below CORRECT threshold → AMBIGUOUS
    j = _make_judgment(relevant=True, confidence=0.69)
    assert evaluator.classify_action([j]) == Action.AMBIGUOUS


def test_classify_action_ambiguous_just_above_incorrect(evaluator):
    # relevant=False, confidence=0.70 → score = 1-0.70 = 0.30 → AMBIGUOUS (>= 0.3)
    j = _make_judgment(relevant=False, confidence=0.70)
    assert _judgment_to_float(j) == pytest.approx(0.30)
    assert evaluator.classify_action([j]) == Action.AMBIGUOUS


def test_classify_action_incorrect_below_threshold(evaluator):
    # relevant=False, confidence=0.71 → score = 1-0.71 = 0.29 → INCORRECT (< 0.3)
    j = _make_judgment(relevant=False, confidence=0.71)
    assert _judgment_to_float(j) == pytest.approx(0.29)
    assert evaluator.classify_action([j]) == Action.INCORRECT


# ---------------------------------------------------------------------------
# Test 8: score_batch returns empty list for empty input
# ---------------------------------------------------------------------------

def test_score_batch_empty_input(evaluator):
    result = evaluator.score_batch([], [])
    assert result == []


# ---------------------------------------------------------------------------
# Test 9: prompt variant selection changes the system prompt sent to the API
# ---------------------------------------------------------------------------

@patch("thesis_crag.evaluators.llm_judge.call_llm_with_validation")
def test_prompt_variant_changes_system_prompt(mock_llm, tmp_path):
    mock_llm.return_value = _RELEVANT_RESPONSE

    base_eval = LLMJudgeEvaluator(
        prompt_variant="BASE", cache_db=str(tmp_path / "base.db")
    )
    minimal_eval = LLMJudgeEvaluator(
        prompt_variant="MINIMAL", cache_db=str(tmp_path / "minimal.db")
    )

    base_eval.score("q?", "passage")
    minimal_eval.score("q?", "passage")

    assert mock_llm.call_count == 2
    base_call_system = mock_llm.call_args_list[0][0][0]
    minimal_call_system = mock_llm.call_args_list[1][0][0]
    assert base_call_system != minimal_call_system
    assert base_call_system == VARIANTS["BASE"]
    assert minimal_call_system == VARIANTS["MINIMAL"]


# ---------------------------------------------------------------------------
# Test 10: unknown variant raises ValueError at construction
# ---------------------------------------------------------------------------

def test_unknown_variant_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown prompt variant"):
        LLMJudgeEvaluator(
            prompt_variant="DOES_NOT_EXIST",
            cache_db=str(tmp_path / "x.db"),
        )


# ---------------------------------------------------------------------------
# Test 11: EvaluatorJudgment consistency validator rejects relevant != tm AND ac
# ---------------------------------------------------------------------------

def test_judgment_inconsistency_raises():
    with pytest.raises(ValidationError):
        EvaluatorJudgment(
            topic_match=True,
            answer_containment=True,
            relevant=False,  # inconsistent: should be True
            confidence=0.8,
            reasoning="test",
        )
