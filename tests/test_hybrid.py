"""Unit tests for HybridEvaluator routing logic."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thesis_crag.evaluators.base import Action, EvaluatorJudgment


@pytest.fixture(autouse=True)
def mock_st(monkeypatch):
    """Prevent the real sentence_transformers from loading during each test.

    Function-scoped (monkeypatch.setitem) so this module cannot clobber another
    test module's mock via a shared module-level sys.modules assignment, which
    previously made test ordering affect results (see test_cross_encoder.py).
    The hybrid tests mock CrossEncoderEvaluator/LLMJudgeEvaluator directly, so
    this is a defensive guard against any accidental real import.
    """
    monkeypatch.setitem(sys.modules, "sentence_transformers", MagicMock())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _judgment(relevant: bool, confidence: float = 0.9) -> EvaluatorJudgment:
    return EvaluatorJudgment(
        topic_match=relevant,
        answer_containment=relevant,
        relevant=relevant,
        confidence=confidence,
        reasoning="test",
    )


def _make_hybrid(ce_score: float, llm_relevant: bool = True, llm_confidence: float = 0.9):
    """Create a HybridEvaluator with both sub-evaluators fully mocked.

    Returns (hybrid, mock_ce, mock_llm) so tests can assert call counts.
    """
    with patch("thesis_crag.evaluators.hybrid.CrossEncoderEvaluator") as mock_ce_cls, \
         patch("thesis_crag.evaluators.hybrid.LLMJudgeEvaluator") as mock_llm_cls:

        mock_ce = MagicMock()
        mock_ce.score.return_value = ce_score
        mock_ce_cls.return_value = mock_ce

        mock_llm = MagicMock()
        mock_llm.score.return_value = _judgment(llm_relevant, llm_confidence)
        mock_llm_cls.return_value = mock_llm

        from thesis_crag.evaluators.hybrid import HybridEvaluator
        hybrid = HybridEvaluator("dummy-model")

    return hybrid, mock_ce, mock_llm


# ---------------------------------------------------------------------------
# 1. High CE score → LLM not called
# ---------------------------------------------------------------------------

def test_high_score_skips_llm():
    hybrid, mock_ce, mock_llm = _make_hybrid(ce_score=0.95)
    hybrid.score("q", "p")
    mock_llm.score.assert_not_called()
    assert hybrid.cross_encoder_only == 1
    assert hybrid.llm_called == 0


# ---------------------------------------------------------------------------
# 2. Low CE score → LLM not called
# ---------------------------------------------------------------------------

def test_low_score_skips_llm():
    hybrid, mock_ce, mock_llm = _make_hybrid(ce_score=0.05)
    hybrid.score("q", "p")
    mock_llm.score.assert_not_called()
    assert hybrid.cross_encoder_only == 1
    assert hybrid.llm_called == 0


# ---------------------------------------------------------------------------
# 3. Ambiguous CE score → LLM is called
# ---------------------------------------------------------------------------

def test_ambiguous_score_calls_llm():
    hybrid, mock_ce, mock_llm = _make_hybrid(ce_score=0.5)
    hybrid.score("q", "p")
    mock_llm.score.assert_called_once_with("q", "p")
    assert hybrid.llm_called == 1
    assert hybrid.cross_encoder_only == 0


# ---------------------------------------------------------------------------
# 4. Routing statistics are tracked correctly across multiple calls
# ---------------------------------------------------------------------------

def test_routing_statistics():
    hybrid, mock_ce, mock_llm = _make_hybrid(ce_score=0.95)

    # Manually set return values for a sequence of scores
    mock_ce.score.side_effect = [0.95, 0.05, 0.50, 0.85, 0.30]

    for _ in range(5):
        hybrid.score("q", "p")

    # 0.95 → ce, 0.05 → ce, 0.50 → llm, 0.85 → ce, 0.30 → llm
    assert hybrid.total_queries == 5
    assert hybrid.cross_encoder_only == 3
    assert hybrid.llm_called == 2
    assert abs(hybrid.llm_route_rate - 2 / 5) < 1e-9


# ---------------------------------------------------------------------------
# 5. Boundary: exactly high_threshold (0.8) → CE decisive (inclusive)
# ---------------------------------------------------------------------------

def test_boundary_high_threshold_is_ce():
    hybrid, mock_ce, mock_llm = _make_hybrid(ce_score=0.8)
    score = hybrid.score("q", "p")
    mock_llm.score.assert_not_called()
    assert score == pytest.approx(0.8)
    assert hybrid._last_router == "ce"


# ---------------------------------------------------------------------------
# 6. Boundary: exactly low_threshold (0.2) → CE decisive (inclusive)
# ---------------------------------------------------------------------------

def test_boundary_low_threshold_is_ce():
    hybrid, mock_ce, mock_llm = _make_hybrid(ce_score=0.2)
    score = hybrid.score("q", "p")
    mock_llm.score.assert_not_called()
    assert score == pytest.approx(0.2)
    assert hybrid._last_router == "ce"


# ---------------------------------------------------------------------------
# Bonus 7: LLM result mapped through _judgment_to_float correctly
# ---------------------------------------------------------------------------

def test_llm_score_converted_to_float():
    # relevant=True, conf=0.8 → score should be 0.8
    hybrid, _, mock_llm = _make_hybrid(ce_score=0.5, llm_relevant=True, llm_confidence=0.8)
    result = hybrid.score("q", "p")
    assert result == pytest.approx(0.8)

    # relevant=False, conf=0.8 → score should be 1 - 0.8 = 0.2
    hybrid2, _, mock_llm2 = _make_hybrid(ce_score=0.5, llm_relevant=False, llm_confidence=0.8)
    result2 = hybrid2.score("q", "p")
    assert result2 == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Bonus 8: classify_action uses correct thresholds
# ---------------------------------------------------------------------------

def test_classify_action_correct_threshold():
    hybrid, _, _ = _make_hybrid(ce_score=0.95)
    assert hybrid.classify_action([0.5]) == Action.CORRECT
    assert hybrid.classify_action([0.49]) == Action.AMBIGUOUS
    assert hybrid.classify_action([0.19]) == Action.INCORRECT
    assert hybrid.classify_action([]) == Action.INCORRECT
