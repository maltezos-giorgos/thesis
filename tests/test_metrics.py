"""Tests for thesis_crag.metrics.retrieval — pure functions, no model required."""

import pytest

from thesis_crag.metrics.retrieval import (
    evaluator_accuracy,
    false_positive_rate_at_1,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)

# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------


def test_precision_at_k_perfect():
    preds = ["a", "b", "c"]
    gt = {"a", "b", "c"}
    assert precision_at_k(preds, gt, k=3) == pytest.approx(1.0)


def test_precision_at_k_zero():
    preds = ["x", "y", "z"]
    gt = {"a", "b"}
    assert precision_at_k(preds, gt, k=3) == pytest.approx(0.0)


def test_precision_at_k_partial():
    preds = ["a", "x", "b"]
    gt = {"a", "b"}
    assert precision_at_k(preds, gt, k=3) == pytest.approx(2 / 3)


def test_precision_at_k_truncates():
    preds = ["a", "b", "c", "d"]
    gt = {"a", "b", "c", "d"}
    assert precision_at_k(preds, gt, k=2) == pytest.approx(1.0)


def test_precision_at_k_zero_k():
    assert precision_at_k(["a"], {"a"}, k=0) == 0.0


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


def test_recall_at_k_perfect():
    preds = ["a", "b"]
    gt = {"a", "b"}
    assert recall_at_k(preds, gt, k=2) == pytest.approx(1.0)


def test_recall_at_k_partial():
    preds = ["a", "x", "b"]
    gt = {"a", "b", "c"}
    assert recall_at_k(preds, gt, k=3) == pytest.approx(2 / 3)


def test_recall_at_k_empty_gt():
    assert recall_at_k(["a"], set(), k=1) == 0.0


def test_recall_at_k_zero_k():
    assert recall_at_k(["a"], {"a"}, k=0) == 0.0


def test_recall_at_k_none_retrieved():
    preds = ["x", "y"]
    gt = {"a", "b"}
    assert recall_at_k(preds, gt, k=2) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# mean_reciprocal_rank
# ---------------------------------------------------------------------------


def test_mrr_first_position():
    assert mean_reciprocal_rank(["a", "b"], {"a"}) == pytest.approx(1.0)


def test_mrr_second_position():
    assert mean_reciprocal_rank(["x", "a", "b"], {"a"}) == pytest.approx(0.5)


def test_mrr_third_position():
    assert mean_reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)


def test_mrr_not_found():
    assert mean_reciprocal_rank(["x", "y"], {"a"}) == pytest.approx(0.0)


def test_mrr_empty_predictions():
    assert mean_reciprocal_rank([], {"a"}) == pytest.approx(0.0)


def test_mrr_multiple_relevant_first_wins():
    # Should return 1/rank of FIRST relevant item, regardless of how many exist
    assert mean_reciprocal_rank(["x", "a", "b"], {"a", "b"}) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# false_positive_rate_at_1
# ---------------------------------------------------------------------------


def test_fpr_all_correct():
    # Predict irrelevant for all truly irrelevant → FPR = 0
    predicted = [False, False, False]
    gt = [False, False, False]
    assert false_positive_rate_at_1(predicted, gt) == pytest.approx(0.0)


def test_fpr_all_false_positives():
    predicted = [True, True]
    gt = [False, False]
    assert false_positive_rate_at_1(predicted, gt) == pytest.approx(1.0)


def test_fpr_partial():
    # 2 truly irrelevant; 1 correctly rejected, 1 false positive → FPR = 0.5
    predicted = [True, False, True]
    gt = [True, False, False]  # first is truly relevant, last two are not
    assert false_positive_rate_at_1(predicted, gt) == pytest.approx(0.5)


def test_fpr_no_true_negatives():
    # All items are truly relevant — FPR is undefined, return 0.0
    predicted = [True, True]
    gt = [True, True]
    assert false_positive_rate_at_1(predicted, gt) == pytest.approx(0.0)


def test_fpr_mixed():
    predicted = [True, False, True, False]
    gt = [True, True, False, False]
    # True negatives: indices 2, 3 (gt=False) → 2
    # False positives: index 2 (pred=True, gt=False) → 1
    # FPR = 1/2
    assert false_positive_rate_at_1(predicted, gt) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# evaluator_accuracy
# ---------------------------------------------------------------------------


def test_evaluator_accuracy_perfect():
    preds = ["CORRECT", "AMBIGUOUS", "INCORRECT"]
    truth = ["CORRECT", "AMBIGUOUS", "INCORRECT"]
    assert evaluator_accuracy(preds, truth) == pytest.approx(1.0)


def test_evaluator_accuracy_zero():
    preds = ["CORRECT", "CORRECT"]
    truth = ["INCORRECT", "AMBIGUOUS"]
    assert evaluator_accuracy(preds, truth) == pytest.approx(0.0)


def test_evaluator_accuracy_partial():
    preds = ["CORRECT", "AMBIGUOUS", "INCORRECT"]
    truth = ["CORRECT", "INCORRECT", "INCORRECT"]
    assert evaluator_accuracy(preds, truth) == pytest.approx(2 / 3)


def test_evaluator_accuracy_empty():
    assert evaluator_accuracy([], []) == pytest.approx(0.0)
