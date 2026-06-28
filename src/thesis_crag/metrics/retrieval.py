"""Retrieval and evaluator quality metrics — pure functions, no side-effects."""

from __future__ import annotations

from collections.abc import Sequence


def precision_at_k(predictions: Sequence[str], ground_truth: set[str], k: int) -> float:
    """Fraction of top-k predictions that are in ground_truth."""
    if k <= 0:
        return 0.0
    top_k = list(predictions)[:k]
    return sum(1 for p in top_k if p in ground_truth) / k


def recall_at_k(predictions: Sequence[str], ground_truth: set[str], k: int) -> float:
    """Fraction of ground_truth items that appear in the top-k predictions."""
    if not ground_truth or k <= 0:
        return 0.0
    top_k = list(predictions)[:k]
    return sum(1 for p in top_k if p in ground_truth) / len(ground_truth)


def mean_reciprocal_rank(predictions: Sequence[str], ground_truth: set[str]) -> float:
    """1 / rank of the first relevant item, or 0.0 if no relevant item is found."""
    for rank, p in enumerate(predictions, 1):
        if p in ground_truth:
            return 1.0 / rank
    return 0.0


def false_positive_rate_at_1(
    predicted_relevant: Sequence[bool],
    ground_truth_relevant: Sequence[bool],
) -> float:
    """False-positive rate at position 1 (our flagship metric).

    Among items that are truly NOT relevant, what fraction does the evaluator
    incorrectly label as relevant?

        FPR = FP / (FP + TN)

    This measures how often the CRAG evaluator triggers a spurious CORRECT action
    on an irrelevant passage — directly connected to RQ1 in the thesis.

    Args:
        predicted_relevant: model's binary relevance prediction per item.
        ground_truth_relevant: ground-truth binary label per item.

    Returns:
        FPR in [0, 1], or 0.0 if there are no true negatives.
    """
    true_negatives = sum(1 for g in ground_truth_relevant if not g)
    false_positives = sum(
        1 for p, g in zip(predicted_relevant, ground_truth_relevant, strict=True)
        if p and not g
    )
    return false_positives / true_negatives if true_negatives > 0 else 0.0


def evaluator_accuracy(
    predicted_actions: Sequence[str],
    true_actions: Sequence[str],
) -> float:
    """Fraction of questions where predicted action == true action."""
    if not predicted_actions:
        return 0.0
    return sum(
        1 for p, t in zip(predicted_actions, true_actions, strict=True) if p == t
    ) / len(predicted_actions)
