# Phase 4: 4-Way Evaluator Comparison

**Test set:** 136 cases × 4 passages = 544 total | **Date:** Phase 4A complete

---

## Aggregate Metrics

| Metric | T5 (baseline) | LLM Judge | Cross-Encoder | **Hybrid** |
|--------|:-------------:|:---------:|:-------------:|:----------:|
| Trap Detection  | 95.6% | 85.3% | 91.9% | **91.2%** |
| Gold Recall     | 58.1% | 91.9% | 78.7% | **92.7%** |
| FPR@1           | 4.4% | 14.7% | 8.1% | **8.8%** |
| Precision@1     | — | 90.4% | 86.8% | 87.5% |
| MRR             | — | 94.3% | 92.5% | 93.0% |
| Latency (med.)  | ~2.66s/case | 1.212s/call | 0.026s/call | 0.026s/call |
| Est. cost (544) | $0 | $0.3264 | $0 | ~$0.0972 |

---

## Per-Trap-Type Trap Detection

| Trap type | T5 | LLM Judge | Cross-Encoder | Hybrid |
|-----------|:--:|:---------:|:-------------:|:------:|
| entity_alias | 24/26 (92.3%) | 17/26 (65.4%) | 17/26 (65.4%) | 19/26 (73.1%) |
| topic_overlap | 62/66 (93.9%) | 55/66 (83.3%) | 64/66 (97.0%) | 61/66 (92.4%) |
| unknown | 44/44 (100.0%) | 44/44 (100.0%) | 44/44 (100.0%) | 44/44 (100.0%) |

---

## Hybrid Routing Statistics

| | Count | % of 544 passages |
|--|------:|:-----------------:|
| CE decisive (fast path) | 382 | 70.2% |
| LLM escalated (slow path) | 162 | 29.8% |
| New LLM API calls | 0 | — |
| Est. cost (this run) | $0.0000 | — |
| Projected cost (fresh, no cache) | ~$0.0972 | — |

---

## Case Studies: Where Hybrid Outperforms Individual Evaluators

### Case 1: What genre is Unknown?

- **Trap type:** topic_overlap
- **Gold passage:** Unknown (1988 anthology)
- **Trap passage:** The Unknown (novel)

| | Gold recall | Trap detection |
|--|:-----------:|:--------------:|
| T5            | ✗  | ✓  |
| Cross-Encoder | ✗  | ✓  |
| **Hybrid**    | **✓**  | **✓**  |

Hybrid routing: gold→**llm**, trap→**ce**

### Case 2: Who was the composer of Chasing?

- **Trap type:** unknown
- **Gold passage:** Chasing (song)
- **Trap passage:** George S. Chase

| | Gold recall | Trap detection |
|--|:-----------:|:--------------:|
| T5            | ✓  | ✓  |
| Cross-Encoder | ✗  | ✓  |
| **Hybrid**    | **✓**  | **✓**  |

Hybrid routing: gold→**llm**, trap→**llm**

### Case 3: What genre is Heaven?

- **Trap type:** topic_overlap
- **Gold passage:** Heaven (Cosmic Baby album)
- **Trap passage:** Heaven (1987 film)

| | Gold recall | Trap detection |
|--|:-----------:|:--------------:|
| T5            | ✗  | ✓  |
| Cross-Encoder | ✗  | ✓  |
| **Hybrid**    | **✓**  | **✓**  |

Hybrid routing: gold→**llm**, trap→**ce**

---

## Summary

The Hybrid evaluator achieves the best overall balance:
- **Trap detection** 91.2% vs CE 91.9% and LLM 85.3%
- **Gold recall** 92.7% vs CE 78.7% and LLM 91.9%
- Only 29.8% of passages need LLM calls → 70.2% served at CE speed (0.026s/call)
- Projected cost: ~$0.0972 per full test run vs $0.3264 for LLM-only

_Entity-alias trap detection remains the structural weak point across all evaluators._
