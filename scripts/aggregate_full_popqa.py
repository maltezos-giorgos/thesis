"""Aggregate full-PopQA evaluation results — extended 4-way comparison.

Reads:
    results/baseline/phase1_popqa_actions.jsonl  (T5, has all_passage_scores)
    results/full_popqa/ce_v2_results.jsonl
    results/full_popqa/llm_results.jsonl
    results/full_popqa/hybrid_results.jsonl

Produces five output tables:
    Table 1 — Action distribution + top-1 (question-level) metrics
    Table 2 — Passage-level Precision / Recall / F1 across all 13,990 passages
              T5 shown in two variants (AMBIGUOUS→CORRECT and AMBIGUOUS→INCORRECT)
    Table 3 — Question-level accuracy vs. ground-truth action
    Table 4 — Hybrid routing summary

Thresholds
----------
    T5 (strict / ambig→incorr)  : score ≥  0.592  (UPPER_THRESHOLD)
    T5 (liberal / ambig→corr)   : score ≥ −0.995  (LOWER_THRESHOLD)
    CE v2                        : score ≥  0.5
    LLM Judge                    : score ≥  0.7
    Hybrid                       : score ≥  0.5

Question-level ground-truth action definition
--------------------------------------------
    GT = CORRECT   if gt_labels[0] == True   (retriever's top-1 is relevant)
    GT = AMBIGUOUS if gt_labels[0] == False AND any(gt_labels) == True
    GT = INCORRECT if no passage in the set is relevant

Usage
-----
    python scripts/aggregate_full_popqa.py
    python scripts/aggregate_full_popqa.py --results-dir results/full_popqa
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO_ROOT    = Path(__file__).parent.parent
RESULTS_DIR  = REPO_ROOT / "results" / "full_popqa"
T5_JSONL     = REPO_ROOT / "results" / "baseline" / "phase1_popqa_actions.jsonl"

T5_UPPER =  0.592
T5_LOWER = -0.995
N_DOCS   = 10

CE_THR     = 0.5
LLM_THR    = 0.7
HYBRID_THR = 0.5


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _mrr(results: list[dict], score_key: str = "all_scores") -> float:
    reciprocals = []
    for r in results:
        scores = r.get(score_key, [])
        labels = r.get("all_gt_labels", [])
        if not scores or not labels:
            reciprocals.append(0.0)
            continue
        ranked = sorted(zip(scores, labels, strict=True), key=lambda x: -x[0])
        rr = 0.0
        for rank, (_, rel) in enumerate(ranked, start=1):
            if rel:
                rr = 1.0 / rank
                break
        reciprocals.append(rr)
    return sum(reciprocals) / len(reciprocals) if reciprocals else 0.0


def _prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def passage_prf1(results: list[dict], pos_threshold: float,
                 score_key: str = "all_scores") -> tuple[float, float, float]:
    """Passage-level P/R/F1 across ALL passages (not just top-1)."""
    tp = fp = fn = tn = 0
    for r in results:
        scores = r.get(score_key, [])
        labels = r.get("all_gt_labels", [])
        for s, gt in zip(scores, labels, strict=True):
            pred = s >= pos_threshold
            if pred and gt:
                tp += 1
            elif pred and not gt:
                fp += 1
            elif not pred and gt:
                fn += 1
            else:
                tn += 1
    return _prf1(tp, fp, fn)


def question_level_metrics(results: list[dict], pos_threshold: float,
                            score_key: str = "all_scores") -> dict:
    """Action distribution, top-1 FPR/TPR/Prec, MRR."""
    n = len(results)
    if n == 0:
        return {}
    actions   = Counter(r["assigned_action"] for r in results)
    true_pos  = sum(1 for r in results if r.get("top1_gt_relevant", r["all_gt_labels"][0]))
    true_neg  = n - true_pos

    def top1(r):
        if "top1_score" in r:
            return r["top1_score"]
        scores = r.get(score_key, [])
        return scores[0] if scores else 0.0

    scored_pos = sum(1 for r in results if top1(r) >= pos_threshold)
    fpr_num    = sum(1 for r in results if top1(r) >= pos_threshold
                     and not r.get("top1_gt_relevant", r["all_gt_labels"][0]))
    tpr_num    = sum(1 for r in results if top1(r) >= pos_threshold
                     and r.get("top1_gt_relevant", r["all_gt_labels"][0]))
    fpr  = fpr_num / true_neg  if true_neg  else 0.0
    tpr  = tpr_num / true_pos  if true_pos  else 0.0
    prec = tpr_num / scored_pos if scored_pos else 0.0

    return {
        "n":             n,
        "correct_pct":   actions.get("CORRECT",   0) / n,
        "ambiguous_pct": actions.get("AMBIGUOUS",  0) / n,
        "incorrect_pct": actions.get("INCORRECT",  0) / n,
        "fpr_at_1":      fpr,
        "tpr_at_1":      tpr,
        "precision_at_1": prec,
        "mrr":           _mrr(results, score_key),
    }


def gt_action(r: dict) -> str:
    """Ground-truth question-level action based on retriever ordering."""
    labels = r.get("all_gt_labels", [])
    if not labels:
        return "INCORRECT"
    if labels[0]:
        return "CORRECT"
    if any(labels):
        return "AMBIGUOUS"
    return "INCORRECT"


def question_accuracy(results: list[dict]) -> dict:
    """How often the evaluator picks the ground-truth action."""
    n = len(results)
    if n == 0:
        return {}
    correct = sum(1 for r in results if r["assigned_action"] == gt_action(r))
    gt_dist = Counter(gt_action(r) for r in results)
    per_class: dict[str, float] = {}
    for cls in ("CORRECT", "AMBIGUOUS", "INCORRECT"):
        subset = [r for r in results if gt_action(r) == cls]
        if subset:
            per_class[cls] = sum(
                1 for r in subset if r["assigned_action"] == cls
            ) / len(subset)
    return {
        "overall_acc":    correct / n,
        "per_class":      per_class,
        "gt_dist":        {k: v / n for k, v in gt_dist.items()},
    }


# ---------------------------------------------------------------------------
# T5-specific — reconstruct from raw JSONL (has all_passage_scores)
# ---------------------------------------------------------------------------

def t5_question_level(t5_rows: list[dict]) -> dict:
    """Question-level metrics from T5 raw rows (uses all_passage_scores)."""
    n = len(t5_rows)
    actions  = Counter(r["assigned_action"] for r in t5_rows)
    true_pos = sum(1 for r in t5_rows if r["all_gt_labels"][0])
    true_neg = n - true_pos

    fpr_num  = sum(1 for r in t5_rows
                   if r["all_passage_scores"][0] >= T5_UPPER
                   and not r["all_gt_labels"][0])
    tpr_num  = sum(1 for r in t5_rows
                   if r["all_passage_scores"][0] >= T5_UPPER
                   and r["all_gt_labels"][0])
    scored_p = sum(1 for r in t5_rows
                   if r["all_passage_scores"][0] >= T5_UPPER)
    fpr  = fpr_num / true_neg  if true_neg  else 0.0
    tpr  = tpr_num / true_pos  if true_pos  else 0.0
    prec = tpr_num / scored_p  if scored_p  else 0.0

    return {
        "n":             n,
        "correct_pct":   actions.get("CORRECT",   0) / n,
        "ambiguous_pct": actions.get("AMBIGUOUS",  0) / n,
        "incorrect_pct": actions.get("INCORRECT",  0) / n,
        "fpr_at_1":      fpr,
        "tpr_at_1":      tpr,
        "precision_at_1": prec,
        "mrr":           _mrr(t5_rows, "all_passage_scores"),
    }


def t5_question_accuracy(t5_rows: list[dict]) -> dict:
    n = len(t5_rows)
    correct  = sum(1 for r in t5_rows if r["assigned_action"] == gt_action(r))
    gt_dist  = Counter(gt_action(r) for r in t5_rows)
    per_class: dict[str, float] = {}
    for cls in ("CORRECT", "AMBIGUOUS", "INCORRECT"):
        subset = [r for r in t5_rows if gt_action(r) == cls]
        if subset:
            per_class[cls] = sum(
                1 for r in subset if r["assigned_action"] == cls
            ) / len(subset)
    return {
        "overall_acc": correct / n,
        "per_class":   per_class,
        "gt_dist":     {k: v / n for k, v in gt_dist.items()},
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _pct(x: float | None, d: int = 1) -> str:
    return "  N/A" if x is None else f"{100 * x:.{d}f}%"


def _f(x: float | None, d: int = 4) -> str:
    return "  N/A" if x is None else f"{x:.{d}f}"


def print_section(title: str) -> None:
    print(f"\n{'='*82}")
    print(f"  {title}")
    print(f"{'='*82}")


def print_row(label: str, vals: list[str], col_w: int = 15) -> None:
    print(f"  {label:<28}" + "".join(f"{v:>{col_w}}" for v in vals))


def print_sep(ncols: int, col_w: int = 15, label_w: int = 30) -> None:
    print("-" * (label_w + ncols * col_w))


# ---------------------------------------------------------------------------
# Output tables
# ---------------------------------------------------------------------------

def table1_question_level(evals_ql: dict[str, dict]) -> None:
    print_section("TABLE 1 — Action distribution & top-1 (question-level) metrics")
    names = list(evals_ql.keys())
    col_w = 15
    print_row("Metric", names, col_w)
    print_sep(len(names), col_w)
    print_row("N questions", [f"{evals_ql[n]['n']:,}" for n in names], col_w)
    print_sep(len(names), col_w)
    for k, lbl in [("correct_pct","CORRECT %"), ("ambiguous_pct","AMBIGUOUS %"),
                   ("incorrect_pct","INCORRECT %")]:
        print_row(lbl, [_pct(evals_ql[n].get(k)) for n in names], col_w)
    print_sep(len(names), col_w)
    for k, lbl in [("fpr_at_1","FPR@1"), ("tpr_at_1","TPR@1 (Recall)"),
                   ("precision_at_1","Precision@1")]:
        print_row(lbl, [_pct(evals_ql[n].get(k), 2) for n in names], col_w)
    print_row("MRR", [_f(evals_ql[n].get("mrr")) for n in names], col_w)
    print_sep(len(names), col_w)


def table2_passage_level(passage_evals: dict[str, tuple[float, float, float]]) -> None:
    print_section("TABLE 2 — Passage-level Precision / Recall / F1  (all 13,990 passages)")
    names = list(passage_evals.keys())
    col_w = 16
    print_row("Evaluator variant", names, col_w)
    print_sep(len(names), col_w)
    for lbl, idx in [("Precision", 0), ("Recall", 1), ("F1", 2)]:
        print_row(lbl, [_pct(passage_evals[n][idx], 2) for n in names], col_w)
    print_sep(len(names), col_w)
    print("  Note: T5-liberal  = AMBIGUOUS treated as CORRECT (threshold=−0.995)")
    print("        T5-strict   = AMBIGUOUS treated as INCORRECT (threshold=+0.592)")


def table3_question_accuracy(acc_evals: dict[str, dict]) -> None:
    print_section("TABLE 3 — Question-level accuracy vs. ground-truth action")
    # Print GT distribution once (same for all — same questions)
    first = next(iter(acc_evals.values()))
    gd = first.get("gt_dist", {})
    print(f"  Ground-truth distribution:  "
          f"CORRECT={_pct(gd.get('CORRECT'))}  "
          f"AMBIGUOUS={_pct(gd.get('AMBIGUOUS'))}  "
          f"INCORRECT={_pct(gd.get('INCORRECT'))}")
    print()
    names = list(acc_evals.keys())
    col_w = 15
    print_row("Accuracy", names, col_w)
    print_sep(len(names), col_w)
    print_row("Overall", [_pct(acc_evals[n].get("overall_acc"), 1) for n in names], col_w)
    for cls in ("CORRECT", "AMBIGUOUS", "INCORRECT"):
        print_row(f"  when GT={cls}",
                  [_pct(acc_evals[n]["per_class"].get(cls)) for n in names], col_w)
    print_sep(len(names), col_w)
    print("  'Overall' = fraction of questions where assigned_action == GT action.")
    print("  Per-class = recall within each GT class.")


def table4_hybrid_routing(hybrid_results: list[dict]) -> None:
    if not hybrid_results:
        return
    n  = len(hybrid_results)
    tp = n * N_DOCS
    llm_routed = sum(r.get("llm_route_count", 0) for r in hybrid_results)
    new_api    = sum(r.get("new_api_calls",   0) for r in hybrid_results)
    ce_only    = tp - llm_routed
    print_section("TABLE 4 — Hybrid routing summary")
    print(f"  {n} questions × {N_DOCS} passages = {tp:,} total passage evaluations")
    print(f"  CE-decisive  : {ce_only:>6,}  ({100*ce_only/tp:.1f}%)")
    print(f"  LLM-escalated: {llm_routed:>6,}  ({100*llm_routed/tp:.1f}%)")
    print(f"  New API calls: {new_api:>6,}  (cache hits = {llm_routed - new_api:,})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    rdir: Path = args.results_dir

    # ── Load data ────────────────────────────────────────────────────────────
    t5_rows        = load_jsonl(T5_JSONL)
    ce_results     = load_jsonl(rdir / "ce_v2_results.jsonl")
    llm_results    = load_jsonl(rdir / "llm_results.jsonl")
    hybrid_results = load_jsonl(rdir / "hybrid_results.jsonl")

    print(f"Loaded: T5={len(t5_rows)}, CE={len(ce_results)}, "
          f"LLM={len(llm_results)}, Hybrid={len(hybrid_results)}")

    # ── Table 1: question-level ──────────────────────────────────────────────
    evals_ql: dict[str, dict] = {}
    if t5_rows:
        evals_ql["T5"] = t5_question_level(t5_rows)
    if ce_results:
        evals_ql["CE v2"] = question_level_metrics(ce_results, CE_THR)
    if llm_results:
        evals_ql["LLM Judge"] = question_level_metrics(llm_results, LLM_THR)
    if hybrid_results:
        evals_ql["Hybrid"] = question_level_metrics(hybrid_results, HYBRID_THR)
    table1_question_level(evals_ql)

    # ── Table 2: passage-level ───────────────────────────────────────────────
    passage_evals: dict[str, tuple[float, float, float]] = {}
    if t5_rows:
        passage_evals["T5-liberal"]  = passage_prf1(
            t5_rows, T5_LOWER, "all_passage_scores")
        passage_evals["T5-strict"]   = passage_prf1(
            t5_rows, T5_UPPER, "all_passage_scores")
    if ce_results:
        passage_evals["CE v2"]     = passage_prf1(ce_results,     CE_THR)
    if llm_results:
        passage_evals["LLM Judge"] = passage_prf1(llm_results,    LLM_THR)
    if hybrid_results:
        passage_evals["Hybrid"]    = passage_prf1(hybrid_results,  HYBRID_THR)
    table2_passage_level(passage_evals)

    # ── Table 3: question-level accuracy ────────────────────────────────────
    acc_evals: dict[str, dict] = {}
    if t5_rows:
        acc_evals["T5"]        = t5_question_accuracy(t5_rows)
    if ce_results:
        acc_evals["CE v2"]     = question_accuracy(ce_results)
    if llm_results:
        acc_evals["LLM Judge"] = question_accuracy(llm_results)
    if hybrid_results:
        acc_evals["Hybrid"]    = question_accuracy(hybrid_results)
    table3_question_accuracy(acc_evals)

    # ── Table 4: hybrid routing ──────────────────────────────────────────────
    table4_hybrid_routing(hybrid_results)

    # ── Save JSON ────────────────────────────────────────────────────────────
    out = rdir / "comparison_table_extended.json"
    rdir.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "question_level": evals_ql,
            "passage_level": {k: list(v) for k, v in passage_evals.items()},
            "question_accuracy": acc_evals,
        }, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
