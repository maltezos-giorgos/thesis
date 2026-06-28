"""Tune hybrid routing thresholds on the val set.

Strategy (two-pass, inference-free sweep):
  Pass 1 — Score all val passages with the CE (fast, local).
  Pass 2 — For passages whose CE score falls in the widest ambiguous band
            (low_min < score < high_max = 0.1 < score < 0.9), call the LLM
            judge once and store the result. Subsequent threshold sweeps reuse
            these precomputed scores — no additional inference.

Output:
    results/phase4/hybrid_threshold_tuning.md   (table + top-5)
    results/phase4/hybrid_threshold_tuning.json (full grid)
"""

from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path

import torch

try:
    _tv = torch.library.Library("torchvision", "FRAGMENT")
    _tv.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thesis_crag.evaluators.cross_encoder import CrossEncoderEvaluator
from thesis_crag.evaluators.llm_judge import LLMJudgeEvaluator, _judgment_to_float
from thesis_crag.utils.logging import get_logger

logger = get_logger("tune_hybrid")

REPO_ROOT   = Path(__file__).parent.parent
VAL_PATH    = REPO_ROOT / "data/hard_cases/splits_v2/val.jsonl"
MODEL_PATH  = REPO_ROOT / "models/cross_encoder_v2"
CACHE_DB    = REPO_ROOT / "data/cache/llm_judge_val_tuning.db"
RESULTS_DIR = REPO_ROOT / "results/phase4"

HIGH_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
LOW_THRESHOLDS  = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

# The widest possible ambiguous band across all combos we'll test
WIDE_LOW  = min(LOW_THRESHOLDS)   # 0.10
WIDE_HIGH = max(HIGH_THRESHOLDS)  # 0.90

CORRECT_THR   = 0.5   # classify final score as "relevant" for metric computation
INCORRECT_THR = 0.2


def load_val_cases() -> list[dict]:
    with open(VAL_PATH) as f:
        return [json.loads(l) for l in f]


def _pct(x: float) -> str:
    return f"{x:.1%}"


# ---------------------------------------------------------------------------
# Pass 1: CE scores for all val passages
# ---------------------------------------------------------------------------

def run_ce_pass(cases: list[dict], ce: CrossEncoderEvaluator) -> list[dict]:
    """Returns a flat list of passage records with CE scores."""
    records = []
    for i, case in enumerate(cases):
        if i % 30 == 0:
            logger.info("CE pass: case %d/%d", i, len(cases))
        q = case["question"]
        passages = [
            ("gold",  case["gold_passage"]),
            ("trap",  case["trap_passage"]),
            ("irr1",  case["irrelevant_passages"][0]),
            ("irr2",  case["irrelevant_passages"][1]),
        ]
        for role, passage in passages:
            ce_score = ce.score(q, passage)
            records.append({
                "question_id": case["question_id"],
                "question":    q,
                "trap_type":   case["trap_type"],
                "role":        role,
                "passage":     passage,
                "ce_score":    ce_score,
                "llm_score":   None,  # filled in Pass 2
            })
    return records


# ---------------------------------------------------------------------------
# Pass 2: LLM scores for ambiguous passages only
# ---------------------------------------------------------------------------

def run_llm_pass(records: list[dict], llm: LLMJudgeEvaluator) -> int:
    """Fill llm_score in-place for passages in the widest ambiguous band.

    Returns the number of passages scored by LLM.
    """
    ambiguous = [r for r in records if WIDE_LOW < r["ce_score"] < WIDE_HIGH]
    logger.info(
        "LLM pass: %d/%d passages in ambiguous band (%.1f%%)  "
        "[CE < %.2f or > %.2f → CE-decisive]",
        len(ambiguous), len(records),
        100 * len(ambiguous) / len(records),
        WIDE_LOW, WIDE_HIGH,
    )
    for j, r in enumerate(ambiguous):
        if j % 20 == 0 and j:
            logger.info("LLM pass: %d/%d scored", j, len(ambiguous))
        t0 = time.time()
        judgment = llm.score(r["question"], r["passage"])
        r["llm_score"] = _judgment_to_float(judgment)
        elapsed = time.time() - t0
        if elapsed > 0.5:
            logger.debug("  slow call (%.2fs) — likely new API call", elapsed)
    return len(ambiguous)


# ---------------------------------------------------------------------------
# Metric computation for a given threshold pair
# ---------------------------------------------------------------------------

def _final_score(r: dict, high: float, low: float) -> tuple[float, str]:
    """Return (score, router) for a record given thresholds."""
    cs = r["ce_score"]
    if cs >= high or cs <= low:
        return cs, "ce"
    # Should always have llm_score if ce_score is in band; fallback to ce
    if r["llm_score"] is None:
        return cs, "ce"
    return r["llm_score"], "llm"


def compute_metrics(records: list[dict], high: float, low: float) -> dict:
    # Group by question_id and role
    by_qid: dict[str, dict] = {}
    for r in records:
        qid = r["question_id"]
        if qid not in by_qid:
            by_qid[qid] = {"trap_type": r["trap_type"]}
        score, router = _final_score(r, high, low)
        by_qid[qid][r["role"] + "_score"]  = score
        by_qid[qid][r["role"] + "_router"] = router

    n = len(by_qid)
    gold_recall = trap_fp = llm_count = total_passages = 0
    trap_types: dict[str, dict] = {}

    for qid, d in by_qid.items():
        gold_ok  = d["gold_score"] >= CORRECT_THR
        trap_bad = d["trap_score"] >= CORRECT_THR

        if gold_ok:
            gold_recall += 1
        if trap_bad:
            trap_fp += 1

        tt = d["trap_type"]
        if tt not in trap_types:
            trap_types[tt] = {"total": 0, "detected": 0}
        trap_types[tt]["total"] += 1
        if not trap_bad:
            trap_types[tt]["detected"] += 1

        for role in ("gold", "trap", "irr1", "irr2"):
            if d.get(role + "_router") == "llm":
                llm_count += 1
            total_passages += 1

    trap_det = (n - trap_fp) / n
    gold_rec = gold_recall / n
    return {
        "high_threshold":    high,
        "low_threshold":     low,
        "trap_detection":    round(trap_det, 4),
        "gold_recall":       round(gold_rec, 4),
        "combined":          round((trap_det + gold_rec) / 2, 4),
        "fpr_at_1":          round(trap_fp / n, 4),
        "llm_pct":           round(llm_count / total_passages, 4),
        "per_trap_type":     {
            tt: {
                "detected": v["detected"],
                "total":    v["total"],
                "rate":     round(v["detected"] / v["total"], 4),
            }
            for tt, v in trap_types.items()
        },
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _row(m: dict, bold: bool = False) -> str:
    b = "**" if bold else ""
    return (
        f"| {b}{m['high_threshold']:.2f}{b} | {b}{m['low_threshold']:.2f}{b} "
        f"| {b}{_pct(m['trap_detection'])}{b} | {b}{_pct(m['gold_recall'])}{b} "
        f"| {b}{_pct(m['combined'])}{b} | {b}{_pct(m['fpr_at_1'])}{b} "
        f"| {b}{_pct(m['llm_pct'])}{b} |"
    )


def generate_report(grid: list[dict], n_cases: int) -> str:
    sorted_all = sorted(grid, key=lambda m: -m["combined"])
    top5 = sorted_all[:5]
    default = next(m for m in grid if m["high_threshold"] == 0.80 and m["low_threshold"] == 0.20)

    header = (
        f"# Hybrid Routing Threshold Tuning (Val Set, {n_cases} cases)\n\n"
        f"**Val set:** {n_cases} cases × 4 passages = {n_cases * 4} passages  \n"
        f"**Metric: combined** = (trap_detection + gold_recall) / 2  \n"
        f"**Sweep:** high ∈ {HIGH_THRESHOLDS}, low ∈ {LOW_THRESHOLDS}  \n"
        f"**Total combinations:** {len(grid)}\n\n"
        "---\n\n"
        "## Top 5 Combinations (sorted by combined score)\n\n"
        "| high | low | trap_det | gold_rec | **combined** | FPR@1 | % LLM |\n"
        "|-----:|----:|:--------:|:--------:|:------------:|:-----:|:------:|\n"
    )
    top5_rows = "\n".join(_row(m, bold=(i == 0)) for i, m in enumerate(top5))

    default_section = (
        "\n\n---\n\n"
        "## Current Default (high=0.80, low=0.20)\n\n"
        "| high | low | trap_det | gold_rec | **combined** | FPR@1 | % LLM |\n"
        "|-----:|----:|:--------:|:--------:|:------------:|:-----:|:------:|\n"
        + _row(default)
    )

    # Full grid
    sorted_grid = sorted(grid, key=lambda m: (-m["high_threshold"], m["low_threshold"]))
    full_rows = "\n".join(_row(m) for m in sorted_grid)
    full_section = (
        "\n\n---\n\n"
        "## Full Grid\n\n"
        "| high | low | trap_det | gold_rec | **combined** | FPR@1 | % LLM |\n"
        "|-----:|----:|:--------:|:--------:|:------------:|:-----:|:------:|\n"
        + full_rows
    )

    # Winner analysis
    winner = top5[0]
    delta_trap = winner["trap_detection"] - default["trap_detection"]
    delta_gold = winner["gold_recall"] - default["gold_recall"]
    delta_comb = winner["combined"] - default["combined"]
    delta_llm  = winner["llm_pct"] - default["llm_pct"]

    analysis = (
        "\n\n---\n\n"
        "## Winner Analysis\n\n"
        f"Best combination: **high={winner['high_threshold']:.2f}, low={winner['low_threshold']:.2f}**\n\n"
        f"| | Default (0.80/0.20) | Winner ({winner['high_threshold']:.2f}/{winner['low_threshold']:.2f}) | Δ |\n"
        "|---|:---:|:---:|:---:|\n"
        f"| Trap detection | {_pct(default['trap_detection'])} | {_pct(winner['trap_detection'])} | {delta_trap:+.1%} |\n"
        f"| Gold recall    | {_pct(default['gold_recall'])} | {_pct(winner['gold_recall'])} | {delta_gold:+.1%} |\n"
        f"| Combined       | {_pct(default['combined'])} | {_pct(winner['combined'])} | {delta_comb:+.1%} |\n"
        f"| FPR@1          | {_pct(default['fpr_at_1'])} | {_pct(winner['fpr_at_1'])} | {winner['fpr_at_1'] - default['fpr_at_1']:+.1%} |\n"
        f"| % LLM routed   | {_pct(default['llm_pct'])} | {_pct(winner['llm_pct'])} | {delta_llm:+.1%} |\n"
    )

    return header + top5_rows + default_section + full_section + analysis + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cases = load_val_cases()
    logger.info("Loaded %d val cases", len(cases))

    logger.info("Loading cross-encoder from %s", MODEL_PATH)
    ce = CrossEncoderEvaluator(str(MODEL_PATH), device="cpu")

    logger.info("Pass 1: scoring all %d passages with CE...", len(cases) * 4)
    t0 = time.time()
    records = run_ce_pass(cases, ce)
    logger.info("CE pass done in %.1fs", time.time() - t0)

    logger.info("Loading LLM judge (cache: %s)", CACHE_DB)
    llm = LLMJudgeEvaluator(
        prompt_variant="WITH_NEGATIVE_EXAMPLES",
        cache_db=str(CACHE_DB),
    )

    logger.info("Pass 2: LLM scoring for ambiguous passages...")
    t1 = time.time()
    run_llm_pass(records, llm)
    logger.info(
        "LLM pass done in %.1fs  |  new API calls: %d  |  est. cost: $%.4f",
        time.time() - t1, llm.total_api_calls, llm.estimated_cost_usd,
    )
    llm.close()

    logger.info("Sweeping %d threshold combinations...", len(HIGH_THRESHOLDS) * len(LOW_THRESHOLDS))
    grid: list[dict] = []
    for high, low in product(HIGH_THRESHOLDS, LOW_THRESHOLDS):
        if low >= high:
            continue  # degenerate: no CE-decisive region at all
        grid.append(compute_metrics(records, high, low))

    sorted_grid = sorted(grid, key=lambda m: -m["combined"])
    top5 = sorted_grid[:5]

    md = generate_report(grid, len(cases))
    out_md   = RESULTS_DIR / "hybrid_threshold_tuning.md"
    out_json = RESULTS_DIR / "hybrid_threshold_tuning.json"
    out_md.write_text(md)
    out_json.write_text(json.dumps(sorted_grid, indent=2))

    print("\n" + "=" * 65)
    print(f"HYBRID THRESHOLD TUNING — VAL SET ({len(cases)} cases, {len(cases) * 4} passages)")
    print("=" * 65)
    print(f"\n{'high':>6} {'low':>6} {'trap_det':>10} {'gold_rec':>10} {'combined':>10} {'FPR@1':>7} {'%LLM':>7}")
    print("-" * 65)
    for m in top5:
        print(
            f"  {m['high_threshold']:.2f}   {m['low_threshold']:.2f}"
            f"   {_pct(m['trap_detection']):>9}"
            f"   {_pct(m['gold_recall']):>9}"
            f"   {_pct(m['combined']):>9}"
            f"   {_pct(m['fpr_at_1']):>6}"
            f"   {_pct(m['llm_pct']):>6}"
        )

    default = next(m for m in grid if m["high_threshold"] == 0.80 and m["low_threshold"] == 0.20)
    print("\n--- default (0.80/0.20) ---")
    print(
        f"  {default['high_threshold']:.2f}   {default['low_threshold']:.2f}"
        f"   {_pct(default['trap_detection']):>9}"
        f"   {_pct(default['gold_recall']):>9}"
        f"   {_pct(default['combined']):>9}"
        f"   {_pct(default['fpr_at_1']):>6}"
        f"   {_pct(default['llm_pct']):>6}"
    )
    print(f"\nMD:   {out_md}")
    print(f"JSON: {out_json}")
    print(f"\nNew LLM API calls: {llm.total_api_calls}  est. cost: ${llm.estimated_cost_usd:.4f}")


if __name__ == "__main__":
    main()
