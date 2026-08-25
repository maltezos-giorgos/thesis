# Phase 5 — Corrected Ranking Metrics

**Test set:** splits_v2 test, n=212 cases. Re-aggregated from the existing `results/phase5/*.jsonl` score files (no model or API calls).

**Two corrections applied (audit C1 + C2):**

- **C1 — expected-rank ties:** ranking metrics are the expected value under uniformly-random tie-breaking (replaces the gold-favouring `rank = 1 + #(score > gold)`).
- **C2 — contamination exclusion:** 28 'irrelevant' distractors across 23 cases were LLM-verified by the miner to contain the answer; they are dropped from the candidate set before ranking.

_As-reported reproduction check vs `comparison.json`: max |Δ| = 0.000000 (0.0 = exact match)._

## As-reported vs Corrected (95% bootstrap CI, n=1000, seed=0)

| Evaluator | Metric | As-reported | Corrected (point [95% CI]) | Δ |
|---|---|---:|:---:|---:|
| T5 | Precision@1 | 0.7972 | 0.8160 [0.7594, 0.8679] | +0.0189 |
|  | Recall@2 | 0.9009 | 0.9104 [0.8679, 0.9481] | +0.0094 |
|  | MRR | 0.8785 | 0.8903 [0.8561, 0.9218] | +0.0118 |
| LLM Judge | Precision@1 | 0.9104 | 0.9006 [0.8616, 0.9363] | -0.0098 |
|  | Recall@2 | 0.9764 | 0.9693 [0.9481, 0.9882] | -0.0071 |
|  | MRR | 0.9509 | 0.9440 [0.9228, 0.9643] | -0.0068 |
| CE v2 | Precision@1 | 0.9387 | 0.9481 [0.9151, 0.9764] | +0.0094 |
|  | Recall@2 | 0.9858 | 0.9906 [0.9764, 1.0000] | +0.0047 |
|  | MRR | 0.9670 | 0.9725 [0.9552, 0.9874] | +0.0055 |
| Hybrid v2 | Precision@1 | 0.9057 | 0.9269 [0.8892, 0.9575] | +0.0212 |
|  | Recall@2 | 0.9858 | 0.9906 [0.9764, 1.0000] | +0.0047 |
|  | MRR | 0.9501 | 0.9615 [0.9414, 0.9788] | +0.0114 |

## Corrected ranking metrics (appendix form)

| Metric | T5 | LLM Judge | CE v2 | Hybrid v2 |
|---|---:|---:|---:|---:|
| Precision@1 | 0.8160 [0.7594–0.8679] | 0.9006 [0.8616–0.9363] | 0.9481 [0.9151–0.9764] | 0.9269 [0.8892–0.9575] |
| Recall@2 | 0.9104 [0.8679–0.9481] | 0.9693 [0.9481–0.9882] | 0.9906 [0.9764–1.0000] | 0.9906 [0.9764–1.0000] |
| MRR | 0.8903 [0.8561–0.9218] | 0.9440 [0.9228–0.9643] | 0.9725 [0.9552–0.9874] | 0.9615 [0.9414–0.9788] |

