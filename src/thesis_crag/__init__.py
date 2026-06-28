"""thesis_crag: Improving the CRAG retrieval evaluator beyond the similarity trap."""

__version__ = "0.1.0"

from thesis_crag.evaluators.base import Action, EvaluatorJudgment, RetrievalEvaluator

__all__ = ["Action", "EvaluatorJudgment", "RetrievalEvaluator", "__version__"]
