"""Unit tests for CrossEncoderEvaluator and training data preparation."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from prepare_training_data import (
    ENTITY_ALIAS_MULTIPLIER,
    LABEL_NEG,
    LABEL_POS,
    make_pairs,
)

from thesis_crag.evaluators.base import Action
from thesis_crag.evaluators.cross_encoder import CrossEncoderEvaluator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_st(monkeypatch):
    """Mock sentence_transformers for the duration of each test.

    The real sentence_transformers fails in this environment due to a
    torchvision NMS registration conflict (same issue handled in
    eval_llm_judge_full.py). Tests only need to verify our wrapper logic, not
    the upstream library. Using monkeypatch.setitem (function-scoped) keeps each
    test isolated — unlike a module-level sys.modules assignment, two test
    modules cannot clobber each other's mock regardless of collection order.
    """
    mock = MagicMock()
    monkeypatch.setitem(sys.modules, "sentence_transformers", mock)
    return mock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_case(trap_type: str = "topic_overlap") -> dict:
    return {
        "question":            "Who wrote Hamlet?",
        "gold_passage":        "Hamlet was written by William Shakespeare.",
        "trap_passage":        "Hamlet is a famous play set in Denmark.",
        "irrelevant_passages": ["Cats are mammals.", "Rome is in Italy.", "Extra passage."],
        "trap_type":           trap_type,
    }


@pytest.fixture
def ev(mock_st):
    """CrossEncoderEvaluator with a fresh mock model for each test."""
    mock_model = MagicMock()
    mock_model.predict.return_value = [2.0]   # sigmoid(2.0) ≈ 0.88
    mock_st.CrossEncoder.return_value = mock_model
    return CrossEncoderEvaluator("dummy-model")


# ---------------------------------------------------------------------------
# 1. Model loads — CrossEncoder called with the right arguments
# ---------------------------------------------------------------------------

def test_model_loads(mock_st):
    CrossEncoderEvaluator("my-model-path", device="cpu")
    mock_st.CrossEncoder.assert_called_once_with("my-model-path", device="cpu", num_labels=1)


# ---------------------------------------------------------------------------
# 2. score() returns float in [0, 1]
# ---------------------------------------------------------------------------

def test_score_returns_float_in_range(ev):
    ev._model.predict.return_value = [2.0]
    s = ev.score("Who wrote Hamlet?", "Shakespeare wrote Hamlet.")
    assert isinstance(s, float)
    assert 0.0 <= s <= 1.0


def test_score_is_sigmoid_of_logit(ev):
    ev._model.predict.return_value = [0.0]
    assert abs(ev.score("q", "p") - 0.5) < 1e-6

    ev._model.predict.return_value = [100.0]
    assert ev.score("q", "p") > 0.99

    ev._model.predict.return_value = [-100.0]
    assert ev.score("q", "p") < 0.01


# ---------------------------------------------------------------------------
# 3. score_batch() handles empty input
# ---------------------------------------------------------------------------

def test_score_batch_empty(ev):
    assert ev.score_batch([], []) == []


# ---------------------------------------------------------------------------
# 4. classify_action boundary cases
# ---------------------------------------------------------------------------

def test_classify_action_boundaries(ev):
    assert ev.classify_action([0.50]) == Action.CORRECT     # >= 0.5
    assert ev.classify_action([0.49]) == Action.AMBIGUOUS   # < 0.5, >= 0.2
    assert ev.classify_action([0.20]) == Action.AMBIGUOUS   # exactly at lower bound
    assert ev.classify_action([0.19]) == Action.INCORRECT   # < 0.2


# ---------------------------------------------------------------------------
# 5. Training data — correct pair counts
# ---------------------------------------------------------------------------

def test_make_pairs_count_no_oversample():
    cases = [_make_case("topic_overlap")] * 10
    assert len(make_pairs(cases, oversample_entity_alias=False)) == 40  # 10 × 4


def test_make_pairs_count_with_oversample():
    pairs = make_pairs(
        [_make_case("entity_alias")] * 5 + [_make_case("topic_overlap")] * 3,
        oversample_entity_alias=True,
    )
    # entity_alias: 5 × 4 × 3 = 60;  topic_overlap: 3 × 4 = 12
    assert len(pairs) == 72


# ---------------------------------------------------------------------------
# 6. Label smoothing — constants are 0.9 / 0.1, not 1.0 / 0.0
# ---------------------------------------------------------------------------

def test_label_smoothing_constants():
    assert LABEL_POS == pytest.approx(0.9)
    assert LABEL_NEG == pytest.approx(0.1)


def test_label_smoothing_in_pairs():
    pairs  = make_pairs([_make_case()], oversample_entity_alias=False)
    labels = {p["label"] for p in pairs}
    assert 1.0 not in labels
    assert 0.0 not in labels
    assert LABEL_POS in labels
    assert LABEL_NEG in labels


# ---------------------------------------------------------------------------
# 7. entity_alias oversampling increases count by the correct multiplier
# ---------------------------------------------------------------------------

def test_entity_alias_oversampling_multiplier():
    alias_cases = [_make_case("entity_alias")]  * 4
    topic_cases = [_make_case("topic_overlap")] * 4

    no_over   = make_pairs(alias_cases + topic_cases, oversample_entity_alias=False)
    with_over = make_pairs(alias_cases + topic_cases, oversample_entity_alias=True)

    assert len(with_over) > len(no_over)
    alias_no   = [p for p in no_over   if p["trap_type"] == "entity_alias"]
    alias_with = [p for p in with_over if p["trap_type"] == "entity_alias"]
    assert len(alias_with) == len(alias_no) * ENTITY_ALIAS_MULTIPLIER


# ---------------------------------------------------------------------------
# 8. make_pairs uses only the first 2 irrelevant passages
# ---------------------------------------------------------------------------

def test_make_pairs_uses_exactly_two_irrelevants():
    case = _make_case()
    case["irrelevant_passages"] = ["irr0", "irr1", "irr2_unused"]
    pairs    = make_pairs([case], oversample_entity_alias=False)
    passages = [p["passage"] for p in pairs]
    assert "irr0"        in passages
    assert "irr1"        in passages
    assert "irr2_unused" not in passages
    assert len(pairs) == 4
