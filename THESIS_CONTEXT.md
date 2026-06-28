# Thesis Context: Improving the CRAG Retrieval Evaluator

## Research Problem

### What is CRAG?

Corrective Retrieval-Augmented Generation (CRAG), introduced by Yan et al. (2024, arXiv:2401.15884),
extends the standard RAG pipeline with an explicit *retrieval evaluator* that inspects retrieved
passages before they reach the language model. Instead of blindly passing every retrieved document
to the generator, CRAG classifies each passage as **CORRECT**, **AMBIGUOUS**, or **INCORRECT**
relative to the query. Based on this classification, the pipeline takes one of three actions:
use the passage directly, supplement it with a live web search, or discard it entirely and fall
back to web search. This corrective loop is meant to prevent hallucination caused by irrelevant
context. The evaluator in the original implementation is a T5 model fine-tuned on relevance
judgments from the KILT and MS-MARCO benchmarks.

### The Similarity Trap

A reproduction study (arXiv:2603.16169) revealed a critical failure mode in the T5 evaluator:
it behaves as an **entity alignment detector** rather than a genuine *semantic relevance judge*.
The model assigns high relevance scores whenever the retrieved passage shares surface-level named
entities (persons, dates, locations) with the query, regardless of whether the passage actually
contains information needed to answer it. This is the *similarity trap*: passages that are
entity-similar but answer-irrelevant receive inflated scores and are incorrectly labelled CORRECT,
causing the pipeline to generate answers from useless context. Conversely, semantically relevant
passages that rephrase or paraphrase the query entities may be under-scored. The net result is
elevated false-positive rates and degraded downstream generation quality — a structural flaw that
benchmark numbers alone fail to expose.

### Proposed Evaluator Architectures

This thesis proposes and compares three evaluator architectures designed to overcome the
similarity trap:

1. **Cross-Encoder with Hard Negatives** — A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-12-v2`)
   fine-tuned on contrastive pairs where negatives are specifically mined to be entity-similar but
   answer-irrelevant. Hard-negative training forces the model to look beyond surface entity
   overlap and learn deeper semantic compatibility.

2. **LLM-as-Judge with Structured Prompting** — Claude Haiku prompted with a structured rubric
   that decomposes relevance into two explicit sub-criteria: *topic match* (does the passage
   address the query topic?) and *answer containment* (does the passage contain the information
   needed to answer the question?). The model returns a structured JSON judgment, making its
   reasoning transparent and auditable.

3. **Hybrid Two-Stage Evaluator** — A cascade that first applies the cross-encoder for efficient
   bulk scoring and then routes only *ambiguous* passages (those in an uncertainty band around the
   decision threshold) to the LLM judge. This design targets the best accuracy-efficiency tradeoff:
   cheap decisions are made cheaply, and costly LLM calls are reserved for genuinely hard cases.

### Research Questions

- **RQ1**: Does the original T5 CRAG evaluator suffer measurably from the similarity trap?
  Specifically, what is its false-positive rate on a curated set of entity-similar,
  answer-irrelevant hard-negative passages, and how does this compare to an ideal judge?

- **RQ2**: Can a cross-encoder trained with hard negatives significantly reduce the false-positive
  rate relative to the T5 baseline while maintaining comparable true-positive recall?

- **RQ3**: Does structured LLM-as-Judge prompting with explicit sub-criteria outperform both the
  T5 baseline and the cross-encoder on the hard-negative benchmark, and at what inference cost?

- **RQ4**: Does the hybrid two-stage evaluator achieve a Pareto improvement over the individual
  components — i.e., does it match or exceed the LLM judge's accuracy at a fraction of the
  LLM call volume?
