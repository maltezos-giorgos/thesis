# Thesis: Improving the CRAG Retrieval Evaluator

**Beyond the Similarity Trap: Towards a Semantic Relevance Judge for Corrective RAG**

Master's thesis — University of Ioannina, Department of Computer Science and
Engineering, 2025–2026.

## Motivation

Corrective RAG (Yan et al. 2024) improves RAG pipelines by classifying retrieved
passages as CORRECT, AMBIGUOUS, or INCORRECT before generation. However, a
reproduction study (arXiv:2603.16169) shows the original T5 evaluator acts as an
*entity alignment detector*: it rewards surface entity overlap rather than true
answer relevance — the *similarity trap*.

This thesis provides the first quantitative diagnosis of the CRAG evaluator and
proposes three alternative architectures that overcome the trap. The hybrid
evaluator is Pareto-optimal: it matches the accuracy of a pure LLM judge while
calling the expensive LLM on only ~10% of cases.

## Key Results

| Metric (212 hard cases) | T5 | Cross-Encoder | LLM-as-Judge | Hybrid |
|---|---|---|---|---|
| Trap Detection | **96.7%** | 89.2% | 90.1% | 90.1% |
| Gold Recall | 58.0% | 91.0% | 93.4% | **94.3%** |
| Median latency/passage | 0.485s | **0.025s** | 1.204s | 0.025s |
| % LLM calls | — | 0% | 100% | 10.5% |

The T5 evaluator is pathologically conservative: it detects 96.7% of traps but
rejects 42% of genuinely correct passages. The hybrid recovers the most correct
passages at cross-encoder latency.

## Project Layout

```
configs/         YAML configs documenting each evaluator variant's hyperparameters
data/            Datasets: mined hard cases, stratified splits, CE training pairs,
                 and SQLite caches of LLM/Wikipedia calls (for reproducibility)
src/thesis_crag/ Installable Python package — the four evaluator architectures,
                 hard-case miner, loaders, metrics, prompts, utils
scripts/         CLI entry points for the full reproduction pipeline (mine, split,
                 train, evaluate, aggregate)
results/         Final result files that produce the thesis tables
tests/           Pytest test suite (66 tests)
external/CRAG/   Original CRAG repository (git submodule; provides the T5 baseline)
```

> Note: this is a clean "showcase" copy of the repository. Earlier-phase and
> exploratory scripts/intermediates were moved to a local `_archive/` (not tracked).
> The thesis text itself is maintained separately in Overleaf.

## Setup

**Prerequisites:** Python 3.11+, a CUDA-capable GPU recommended for cross-encoder
training (evaluation runs on CPU).

```bash
# 1. Clone with the CRAG submodule
git clone --recursive <repo-url>
cd thesis

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install in editable mode with dev dependencies
make install-dev

# 4. Copy the env template and add your API key
cp .env.example .env
# edit .env — add ANTHROPIC_API_KEY (required for LLM-as-Judge and mining)

# 5. Download datasets
make download-data
```

> The fine-tuned cross-encoder weights (`models/cross_encoder_v2/`) are not
> included in the repository due to size. Re-train them with
> `scripts/train_cross_encoder.py`, or contact the author for the checkpoint.

## Reproducing the Results

The pipeline runs in phases, mirroring the thesis methodology:

```bash
# Phase 1 — T5 baseline diagnosis on full PopQA
python scripts/reproduce_baseline.py

# Phase 2 — mine hard cases and build stratified splits
python scripts/mine_hard_cases.py
python scripts/mine_entity_questions.py
python scripts/build_splits_v2.py

# Phase 3 — LLM-as-Judge prompt A/B test
python scripts/eval_llm_judge_prompts.py

# Phase 4 — train the cross-encoder
python scripts/prepare_training_data.py
python scripts/train_cross_encoder.py

# Phase 5 — final 4-way comparison + full-PopQA evaluation
python scripts/tune_hybrid.py
python scripts/eval_phase5.py
python scripts/eval_full_popqa.py
python scripts/aggregate_full_popqa.py
```

All LLM and Wikipedia calls are cached in `data/cache/*.db`, so re-runs are fast
and do not re-incur API costs.

## Running Tests

```bash
make test     # 66 tests
```

## Linting and Formatting

```bash
make lint     # ruff + mypy
make format   # black + ruff --fix
```

## Evaluator Architectures

| Variant | Model | Notes |
|---|---|---|
| T5 Baseline | Yan et al. 2024 fine-tuned T5-large (`external/CRAG`) | Original CRAG evaluator; run under INT8 quantization |
| Cross-Encoder | `cross-encoder/ms-marco-MiniLM-L-12-v2`, fine-tuned | Trained with entity-similar hard negatives |
| LLM-as-Judge | Claude Haiku | Structured JSON rubric (topic_match ∧ answer_containment) |
| Hybrid | Cross-Encoder → Haiku | LLM called only for ambiguous (0.20–0.80) cases |

## Research Questions

- **RQ1 — Diagnosis:** Quantify the T5 evaluator's trade-off between trap
  detection and gold recall. *(Finding: 96.7% trap detection but only 58.0%
  gold recall — pathologically conservative.)*
- **RQ2 — Alternatives:** Can a cross-encoder and an LLM-as-Judge improve gold
  recall over T5? *(Both reach 91–93%.)*
- **RQ3 — Comparative analysis:** Where do the evaluators differ by trap type,
  and how does prompting affect the LLM judge? *(entity_alias is the structural
  ceiling; Chain-of-Thought backfires.)*
- **RQ4 — Cost–accuracy trade-off:** Can a hybrid reach LLM-level accuracy at a
  fraction of the cost? *(Yes — Pareto-optimal at 10.5% LLM routing.)*

## Citation

If you reference this work, please cite the thesis and the underlying CRAG paper (Yan et al. 2024, arXiv:2401.15884).
