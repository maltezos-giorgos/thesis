"""T5-large retrieval evaluator — faithful reimplementation of Yan et al. 2024.

The original CRAG evaluator is a T5-large model fine-tuned as a single-label
regressor (T5ForSequenceClassification, num_labels=1). It outputs a raw logit
that is compared against two thresholds to assign a CRAG action.

Input format (from external/CRAG/scripts/data_process.py, line 98):
    "question [SEP] passage_text"

Thresholds (from external/CRAG/run_crag_inference.sh):
    --upper_threshold 0.592   → score >= 0.592  → CORRECT
    --lower_threshold 0.995   → negated in CRAG_Inference.py L221 → -0.995
                              → score >= -0.995  → AMBIGUOUS
                              → score <  -0.995  → INCORRECT
"""

from __future__ import annotations

import os

import torch

# -------------------------------------------------------------------------
# Fix: transformers 4.49 + CPU torchvision incompatibility
#
# transformers 4.49 imports image_utils → torchvision._meta_registrations,
# which calls @torch.library.register_fake("torchvision::nms"). CPU torchvision
# builds don't register this op as a native kernel, so _dispatch_has_kernel_for
# _dispatch_key raises RuntimeError("operator torchvision::nms does not exist").
# Defining the op schema via a FRAGMENT library BEFORE torchvision is imported
# satisfies the check. Must happen before any `from transformers import ...`.
# -------------------------------------------------------------------------
try:
    _tv_lib = torch.library.Library("torchvision", "FRAGMENT")  # type: ignore[attr-defined]
    _tv_lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    pass

# transformers imported AFTER the stub above
from transformers import T5Config, T5ForSequenceClassification, T5Tokenizer  # noqa: E402

from .base import RetrievalEvaluator  # noqa: E402


def _load_model(model_path: str, quantize: bool = False) -> T5ForSequenceClassification:
    """Load T5ForSequenceClassification, handling the model-001.safetensors naming.

    The downloaded CRAG checkpoint uses model-001.safetensors instead of the
    standard model.safetensors, so from_pretrained fails. We initialise from
    config and inject weights via safetensors directly.

    quantize=True applies dynamic INT8 quantization to Linear layers (~2x CPU speedup).
    Measured INT8-vs-FP32 logit delta on a 240-passage sample of the test set:
    mean |Δ| ≈ 0.016, median ≈ 0.004, p95 ≈ 0.054, max ≈ 1.02. Most passages barely
    move, but ~3.3% flip their action assignment vs FP32 — mostly INCORRECT<->AMBIGUOUS
    at LOWER_THRESHOLD (-0.995), with the occasional crossing of UPPER_THRESHOLD
    (0.592) — because many trap/irrelevant logits sit near the thresholds. About
    0.4% of the sampled passages flipped across the relevance boundary (0.592) used
    for trap-detection/gold-recall/FPR, so those T5 metrics are mildly
    quantization-dependent. All reported T5 results are produced under INT8
    quantization (quantize=True); pass quantize=False / --no-quantize for exact FP32.
    """
    config = T5Config.from_pretrained(model_path)
    config.num_labels = 1
    config.problem_type = "regression"
    model = T5ForSequenceClassification(config)

    weight_file = os.path.join(model_path, "model-001.safetensors")
    if os.path.exists(weight_file):
        from safetensors.torch import load_file

        state_dict = load_file(weight_file)
        model.load_state_dict(state_dict, strict=False)
    else:
        model = T5ForSequenceClassification.from_pretrained(model_path, num_labels=1)

    model.eval()
    if quantize:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    return model


class T5Evaluator(RetrievalEvaluator):
    """Retrieval evaluator using the fine-tuned T5-large from Yan et al. 2024.

    score() returns a raw logit (not normalised to [0, 1]); the parent class
    classify_action() compares against UPPER/LOWER_THRESHOLD in logit space.
    """

    # Yan et al. 2024 — from external/CRAG/run_crag_inference.sh
    # --upper_threshold 0.592 --lower_threshold 0.995
    # lower is negated in CRAG_Inference.py:221 (args.lower_threshold = -args.lower_threshold)
    UPPER_THRESHOLD: float = 0.592   # logit >= this → CORRECT
    LOWER_THRESHOLD: float = -0.995  # logit >= this → AMBIGUOUS; < this → INCORRECT

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        batch_size: int = 10,
        quantize: bool = True,
        num_threads: int | None = None,
    ) -> None:
        # Pass thresholds to base class; classify_action uses max(scores) against them
        super().__init__(
            correct_threshold=self.UPPER_THRESHOLD,
            incorrect_threshold=self.LOWER_THRESHOLD,
        )
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = _load_model(model_path, quantize=quantize)
        if not quantize:
            self.model.to(self.device)

    def _build_input(self, query: str, passage: str) -> str:
        return f"{query} [SEP] {passage}"

    def _tokenize(self, texts: list[str]) -> dict:
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,       # pad to batch-max length, not model-max (faster on CPU)
            truncation=True,
            max_length=512,
        )

    def score(self, query: str, passage: str) -> float:
        """Return raw T5 logit for one query-passage pair.

        Note: result is in logit space comparable to UPPER/LOWER_THRESHOLD,
        not normalised to [0, 1] as the base-class docstring implies.
        """
        enc = self._tokenize([self._build_input(query, passage)])
        with torch.inference_mode():
            out = self.model(
                enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device),
            )
        return float(out.logits.squeeze())

    def score_batch(self, queries: list[str], passages: list[str]) -> list[float]:
        """Batch-score query-passage pairs; returns logits in input order."""
        texts = [self._build_input(q, p) for q, p in zip(queries, passages, strict=True)]
        all_scores: list[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            enc = self._tokenize(batch)
            with torch.inference_mode():
                out = self.model(
                    enc["input_ids"].to(self.device),
                    attention_mask=enc["attention_mask"].to(self.device),
                )
            logits = out.logits.squeeze(-1)
            if logits.dim() == 0:
                all_scores.append(float(logits))
            else:
                all_scores.extend(logits.tolist())
        return all_scores
