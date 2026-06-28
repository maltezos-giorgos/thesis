"""Phase 3B: A/B comparison of 5 LLM judge prompt variants on a validation sample.

Samples 20 cases from data/hard_cases/splits/val.jsonl (seed=42), builds
4 (query, passage, expected_relevant) pairs per case, then runs all 5 prompt
variants and reports Trap Detection Rate, Gold Recall, FPR@1, confidence, and cost.

Usage:
    python scripts/eval_llm_judge_prompts.py

Output:
    results/phase3/prompt_comparison.md
    results/phase3/prompt_comparison.json   (raw numbers for programmatic use)
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

try:
    _tv = torch.library.Library("torchvision", "FRAGMENT")
    _tv.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    pass

from thesis_crag.evaluators.base import EvaluatorJudgment
from thesis_crag.evaluators.llm_judge import LLMJudgeEvaluator
from thesis_crag.metrics.retrieval import false_positive_rate_at_1
from thesis_crag.utils.logging import get_logger

logger = get_logger("eval_llm_judge_prompts")

REPO_ROOT = Path(__file__).parent.parent
VAL_PATH = REPO_ROOT / "data/hard_cases/splits/val.jsonl"
RESULTS_DIR = REPO_ROOT / "results/phase3"
CACHE_DIR = REPO_ROOT / "data/cache"
N_SAMPLE = 20
SEED = 42
VARIANTS = ["BASE", "WITH_COT", "WITH_NEGATIVE_EXAMPLES", "WITH_STRICT_CONTAINMENT", "MINIMAL"]


class Pair(NamedTuple):
    case_idx: int        # index in the 20-sample list (0–19)
    pair_type: str       # "gold" | "trap" | "irr1" | "irr2"
    query: str
    passage: str
    expected_relevant: bool


def build_pairs(cases: list[dict]) -> list[Pair]:
    """Expand each hard case into 4 (query, passage, expected_relevant) pairs."""
    pairs: list[Pair] = []
    for i, c in enumerate(cases):
        q = c["question"]
        pairs.append(Pair(i, "gold", q, c["gold_passage"], True))
        pairs.append(Pair(i, "trap", q, c["trap_passage"], False))
        pairs.append(Pair(i, "irr1", q, c["irrelevant_passages"][0], False))
        pairs.append(Pair(i, "irr2", q, c["irrelevant_passages"][1], False))
    return pairs


def run_variant(
    variant: str, pairs: list[Pair]
) -> tuple[list[EvaluatorJudgment], float, float]:
    """Run one variant on all pairs. Returns (judgments, elapsed_s, est_cost_usd)."""
    ev = LLMJudgeEvaluator(
        prompt_variant=variant,
        cache_db=str(CACHE_DIR / f"llm_judge_{variant.lower()}.db"),
    )
    queries = [p.query for p in pairs]
    passages = [p.passage for p in pairs]

    t0 = time.time()
    judgments = ev.score_batch(queries, passages)
    elapsed = time.time() - t0
    cost = ev.estimated_cost_usd
    ev.close()
    return judgments, elapsed, cost


def compute_metrics(
    pairs: list[Pair], judgments: list[EvaluatorJudgment]
) -> dict:
    """Compute trap detection, gold recall, FPR@1, confidence, and token usage metrics."""
    trap_correct = 0
    trap_total = 0
    gold_correct = 0
    gold_total = 0

    predicted_relevant: list[bool] = []
    ground_truth_relevant: list[bool] = []
    confidences: list[float] = []
    output_token_estimates: list[int] = []

    for pair, j in zip(pairs, judgments, strict=True):
        predicted_relevant.append(j.relevant)
        ground_truth_relevant.append(pair.expected_relevant)
        confidences.append(j.confidence)
        # Estimate output tokens from response JSON length
        raw_len = len(json.dumps({
            "topic_match": j.topic_match, "answer_containment": j.answer_containment,
            "relevant": j.relevant, "confidence": j.confidence, "reasoning": j.reasoning,
        }))
        output_token_estimates.append(max(1, raw_len // 4))

        if pair.pair_type == "trap":
            trap_total += 1
            if not j.relevant:
                trap_correct += 1
        if pair.pair_type == "gold":
            gold_total += 1
            if j.relevant:
                gold_correct += 1

    return {
        "trap_detection": trap_correct / trap_total if trap_total else 0.0,
        "gold_recall": gold_correct / gold_total if gold_total else 0.0,
        "fpr_at_1": false_positive_rate_at_1(predicted_relevant, ground_truth_relevant),
        "mean_confidence": sum(confidences) / len(confidences),
        "mean_output_tokens": sum(output_token_estimates) / len(output_token_estimates),
    }


def find_disagreements(
    pairs: list[Pair],
    all_judgments: dict[str, list[EvaluatorJudgment]],
    n: int = 3,
) -> list[dict]:
    """Find the n most interesting disagreements (prefer trap passages)."""
    disagreements = []
    for i, pair in enumerate(pairs):
        verdicts = {v: all_judgments[v][i].relevant for v in VARIANTS}
        if len(set(verdicts.values())) == 1:
            continue  # unanimous — skip
        entry = {
            "pair_idx": i,
            "case_idx": pair.case_idx,
            "pair_type": pair.pair_type,
            "query": pair.query,
            "passage": pair.passage[:350],
            "expected_relevant": pair.expected_relevant,
            "verdicts": verdicts,
            "reasonings": {v: all_judgments[v][i].reasoning for v in VARIANTS},
            "confidences": {v: round(all_judgments[v][i].confidence, 3) for v in VARIANTS},
        }
        disagreements.append(entry)

    # Sort: trap passages first (most thesis-relevant), then gold, then others
    order = {"trap": 0, "gold": 1, "irr1": 2, "irr2": 3}
    disagreements.sort(key=lambda d: order.get(d["pair_type"], 9))
    return disagreements[:n]


def write_markdown(
    sample: list[dict],
    pairs: list[Pair],
    metrics_table: dict[str, dict],
    times: dict[str, float],
    costs: dict[str, float],
    disagreements: list[dict],
    total_cost: float,
    total_time: float,
) -> str:
    """Render the A/B variant comparison as a Markdown report string."""
    lines: list[str] = []
    lines.append("# Phase 3B — LLM Judge Prompt Variant Comparison\n")
    lines.append(f"**Sample:** {N_SAMPLE} cases from `val.jsonl` (seed={SEED})  ")
    lines.append("**Pairs per case:** 4 (gold × 1, trap × 1, irrelevant × 2)  ")
    lines.append(f"**Total LLM calls:** {N_SAMPLE * 4 * len(VARIANTS)}  ")
    lines.append(f"**Total cost:** ${total_cost:.4f}  ")
    lines.append(f"**Total wall-clock time:** {total_time:.1f}s  \n")

    # --- Main comparison table ---
    lines.append("## Variant Comparison\n")
    lines.append(
        "| Variant | Trap Detection ↑ | Gold Recall ↑ | FPR@1 ↓ | Confidence | Avg Out Tokens | Time (s) |"
    )
    lines.append("|---------|-----------------|---------------|---------|------------|----------------|----------|")
    for v in VARIANTS:
        m = metrics_table[v]
        lines.append(
            f"| {v} "
            f"| {m['trap_detection']:.0%} "
            f"| {m['gold_recall']:.0%} "
            f"| {m['fpr_at_1']:.4f} "
            f"| {m['mean_confidence']:.3f} "
            f"| {m['mean_output_tokens']:.1f} "
            f"| {times[v]:.1f} |"
        )
    lines.append("")

    # --- Winner ---
    lines.append("## Winner\n")
    ranked = sorted(
        VARIANTS,
        key=lambda v: (
            -metrics_table[v]["trap_detection"],
            -metrics_table[v]["gold_recall"],
            metrics_table[v]["mean_output_tokens"],
        ),
    )
    winner = ranked[0]
    runner_up = ranked[1]
    wm = metrics_table[winner]
    rm = metrics_table[runner_up]
    lines.append(f"**{winner}**\n")
    lines.append(
        f"- Trap Detection: {wm['trap_detection']:.0%} "
        f"(runner-up {runner_up}: {rm['trap_detection']:.0%})"
    )
    lines.append(f"- Gold Recall: {wm['gold_recall']:.0%}")
    lines.append(f"- FPR@1: {wm['fpr_at_1']:.4f}")
    lines.append(f"- Avg output tokens: {wm['mean_output_tokens']:.1f}")
    lines.append("")

    # --- Disagreement analysis ---
    lines.append("## Disagreement Analysis (3 cases where variants split)\n")
    for di, d in enumerate(disagreements, 1):
        expected_label = "RELEVANT" if d["expected_relevant"] else "NOT RELEVANT"
        lines.append(
            f"### Case {di} — {d['pair_type'].upper()} passage (expected: {expected_label})\n"
        )
        lines.append(f"**Query:** {d['query']}  ")
        lines.append(f"**Passage (first 350 chars):** {d['passage']}…\n")
        lines.append("| Variant | Verdict | Confidence | Reasoning |")
        lines.append("|---------|---------|------------|-----------|")
        for v in VARIANTS:
            verdict = "RELEVANT" if d["verdicts"][v] else "NOT RELEVANT"
            conf = d["confidences"][v]
            reasoning = d["reasonings"][v].replace("|", "\\|")
            lines.append(f"| {v} | **{verdict}** | {conf} | {reasoning} |")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    """A/B test 5 prompt variants on 20 val cases and write results/phase3/prompt_comparison.md."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # --- Sample ---
    random.seed(SEED)
    cases: list[dict] = []
    with open(VAL_PATH) as f:
        for line in f:
            cases.append(json.loads(line))
    sample = random.sample(cases, N_SAMPLE)
    pairs = build_pairs(sample)

    logger.info(
        "Running %d variants × %d pairs = %d LLM calls",
        len(VARIANTS), len(pairs), len(VARIANTS) * len(pairs),
    )

    # --- Run all variants ---
    all_judgments: dict[str, list[EvaluatorJudgment]] = {}
    metrics_table: dict[str, dict] = {}
    times: dict[str, float] = {}
    costs: dict[str, float] = {}

    total_wall_start = time.time()
    for variant in VARIANTS:
        logger.info("=== Variant: %s ===", variant)
        judgments, elapsed, cost = run_variant(variant, pairs)
        all_judgments[variant] = judgments
        metrics_table[variant] = compute_metrics(pairs, judgments)
        times[variant] = elapsed
        costs[variant] = cost
        logger.info(
            "%s → trap=%s gold=%s fpr=%.4f cost=$%.4f time=%.1fs",
            variant,
            f"{metrics_table[variant]['trap_detection']:.0%}",
            f"{metrics_table[variant]['gold_recall']:.0%}",
            metrics_table[variant]["fpr_at_1"],
            cost, elapsed,
        )

    total_time = time.time() - total_wall_start
    total_cost = sum(costs.values())

    # --- Disagreements ---
    disagreements = find_disagreements(pairs, all_judgments)

    # --- Write results ---
    md = write_markdown(
        sample, pairs, metrics_table, times, costs,
        disagreements, total_cost, total_time,
    )
    out_md = RESULTS_DIR / "prompt_comparison.md"
    out_md.write_text(md)

    out_json = RESULTS_DIR / "prompt_comparison.json"
    out_json.write_text(json.dumps({
        "metrics": metrics_table,
        "times": times,
        "costs": costs,
        "total_cost": total_cost,
        "total_time": total_time,
        "disagreements": disagreements,
    }, indent=2))

    logger.info("Results written to %s", out_md)

    # --- Console summary ---
    print("\n" + "=" * 64)
    print("PROMPT VARIANT COMPARISON — SUMMARY")
    print("=" * 64)
    print(f"{'Variant':<28} {'Trap Det':>9} {'Gold Rec':>9} {'FPR@1':>7} {'Cost':>8}")
    print("-" * 64)
    for v in VARIANTS:
        m = metrics_table[v]
        print(
            f"{v:<28} {m['trap_detection']:>8.0%} {m['gold_recall']:>8.0%} "
            f"{m['fpr_at_1']:>7.4f} ${costs[v]:>6.4f}"
        )
    print("-" * 64)
    print(f"Total cost: ${total_cost:.4f}   Total time: {total_time:.1f}s")
    print(f"\nReport: {out_md}")


if __name__ == "__main__":
    main()
