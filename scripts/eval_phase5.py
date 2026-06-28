"""Phase 5: 4-way evaluator comparison on splits_v2/test.jsonl (212 cases).

Evaluators
----------
  A  T5 baseline          external/CRAG/models/evaluator
  B  LLM Judge            WITH_NEGATIVE_EXAMPLES (Haiku, cached)
  C  Cross-Encoder v2     models/cross_encoder_v2
  D  Hybrid v2            CE v2 fast path + LLM for ambiguous band (0.8/0.2)

Each evaluator scores 212 × 4 passages (gold, trap, irr1, irr2).
Results are saved to results/phase5/ as JSONL (per-case) + JSON (aggregate).
Bootstrap CIs (1000 iterations, case-level resampling) are added to every metric.

Usage
-----
    python scripts/eval_phase5.py [--skip-t5] [--skip-llm] [--skip-ce] [--skip-hybrid]
    python scripts/eval_phase5.py --only-table    # recompute table from saved JSONLs

Output
------
    results/phase5/t5_eval.jsonl
    results/phase5/llm_eval.jsonl
    results/phase5/ce_v2_eval.jsonl
    results/phase5/hybrid_v2_eval.jsonl
    results/phase5/comparison.json
    results/phase5/comparison.md
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from statistics import median, quantiles
from typing import NamedTuple

import torch

try:
    _tv = torch.library.Library("torchvision", "FRAGMENT")
    _tv.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    pass

REPO_ROOT   = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_crag.utils.logging import get_logger

logger = get_logger("eval_phase5")

TEST_PATH   = REPO_ROOT / "data/hard_cases/splits_v2/test.jsonl"
RESULTS_DIR = REPO_ROOT / "results/phase5"
CACHE_DIR   = REPO_ROOT / "data/cache"

T5_MODEL   = str(REPO_ROOT / "external/CRAG/models/evaluator")
CE_MODEL   = str(REPO_ROOT / "models/cross_encoder_v2")
LLM_CACHE  = str(CACHE_DIR / "llm_judge_with_negative_examples.db")
LLM_VARIANT = "WITH_NEGATIVE_EXAMPLES"

HIGH_THR = 0.8
LOW_THR  = 0.2
CE_CORRECT_THR   = 0.5
CE_INCORRECT_THR = 0.2

N_BOOT = 1000
BOOT_SEED = 0


# ──────────────────────────────────────────────────────────────────────────────
# Unified record schema (one per case)
# ──────────────────────────────────────────────────────────────────────────────

class Record(NamedTuple):
    question_id:   str
    trap_type:     str
    gold_relevant: bool
    trap_relevant: bool
    irr1_relevant: bool
    irr2_relevant: bool
    gold_score:    float
    trap_score:    float
    irr1_score:    float
    irr2_score:    float
    gold_latency:  float
    trap_latency:  float
    irr1_latency:  float
    irr2_latency:  float
    routing:       dict  # {"gold": "ce"|"llm", ...} — empty for non-hybrid


# ──────────────────────────────────────────────────────────────────────────────
# Metrics (pure functions over list[Record])
# ──────────────────────────────────────────────────────────────────────────────

def _compute_metrics(records: list[Record]) -> dict[str, float]:
    n = len(records)
    prec_hits = recall2_hits = mrr_sum = 0.0
    trap_fp = gold_cnt = 0
    tt_totals: dict[str, int] = {}
    tt_detected: dict[str, int] = {}

    for r in records:
        scores = [r.gold_score, r.trap_score, r.irr1_score, r.irr2_score]
        rank = 1 + sum(1 for s in scores[1:] if s > scores[0])
        if rank == 1:
            prec_hits += 1
        if rank <= 2:
            recall2_hits += 1
        mrr_sum += 1.0 / rank

        if r.gold_relevant:
            gold_cnt += 1
        if r.trap_relevant:
            trap_fp += 1

        tt = r.trap_type
        tt_totals[tt]   = tt_totals.get(tt, 0) + 1
        tt_detected[tt] = tt_detected.get(tt, 0) + (0 if r.trap_relevant else 1)

    out: dict[str, float] = {
        "precision_at_1":      prec_hits / n,
        "recall_at_2":         recall2_hits / n,
        "mrr":                 mrr_sum / n,
        "fpr_at_1":            trap_fp / n,
        "trap_detection_rate": (n - trap_fp) / n,
        "gold_recall":         gold_cnt / n,
    }
    for tt in tt_totals:
        out[f"tt_{tt}_trap_det"] = tt_detected[tt] / tt_totals[tt]
    return out


def _latency_stats(records: list[Record]) -> dict:
    all_lat = []
    for r in records:
        all_lat += [r.gold_latency, r.trap_latency, r.irr1_latency, r.irr2_latency]
    return {
        "latency_median_s": round(median(all_lat), 3),
        "latency_p95_s":    round(quantiles(all_lat, n=20)[18], 3),
    }


def _routing_stats(records: list[Record]) -> dict:
    total = len(records) * 4
    llm_cnt = sum(
        sum(1 for v in r.routing.values() if v == "llm")
        for r in records
        if r.routing
    )
    if total == 0:
        return {}
    return {
        "total_passages": total,
        "ce_decisive":    total - llm_cnt,
        "llm_escalated":  llm_cnt,
        "llm_pct":        round(llm_cnt / total, 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    records: list[Record],
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict[str, tuple[float, float, float]]:
    """Return {metric: (point_est, ci_lo, ci_hi)} with 95% CI."""
    rng = random.Random(seed)
    n = len(records)
    boot: list[dict[str, float]] = []
    for _ in range(n_boot):
        sample = [records[rng.randint(0, n - 1)] for _ in range(n)]
        boot.append(_compute_metrics(sample))

    point = _compute_metrics(records)
    out: dict[str, tuple[float, float, float]] = {}
    for key in point:
        vals = sorted(b[key] for b in boot)
        lo = vals[int(0.025 * n_boot)]
        hi = vals[int(0.975 * n_boot)]
        out[key] = (point[key], lo, hi)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# JSONL helpers
# ──────────────────────────────────────────────────────────────────────────────

def _records_from_jsonl(path: Path) -> list[Record]:
    records = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            records.append(Record(**d))
    return records


def _save_records(records: list[Record], path: Path) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r._asdict()) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Evaluator A: T5
# ──────────────────────────────────────────────────────────────────────────────

def run_t5(cases: list[dict]) -> list[Record]:
    from thesis_crag.evaluators.t5_baseline import T5Evaluator

    T5_UPPER = T5Evaluator.UPPER_THRESHOLD
    logger.info("Loading T5 model from %s", T5_MODEL)
    ev = T5Evaluator(T5_MODEL, quantize=True)
    logger.info("T5 loaded. Scoring %d cases…", len(cases))

    records = []
    t_start = time.time()
    for i, case in enumerate(cases):
        if i % 40 == 0:
            logger.info("T5 case %d/%d  (%.0fs)", i, len(cases), time.time() - t_start)
        q = case["question"]
        passages = [
            case["gold_passage"],
            case["trap_passage"],
            case["irrelevant_passages"][0],
            case["irrelevant_passages"][1],
        ]
        t0 = time.time()
        scores = ev.score_batch([q] * 4, passages)
        elapsed = time.time() - t0
        lat_each = elapsed / 4

        records.append(Record(
            question_id   = str(case["question_id"]),
            trap_type     = case["trap_type"],
            gold_relevant = scores[0] >= T5_UPPER,
            trap_relevant = scores[1] >= T5_UPPER,
            irr1_relevant = scores[2] >= T5_UPPER,
            irr2_relevant = scores[3] >= T5_UPPER,
            gold_score    = float(scores[0]),
            trap_score    = float(scores[1]),
            irr1_score    = float(scores[2]),
            irr2_score    = float(scores[3]),
            gold_latency  = lat_each,
            trap_latency  = lat_each,
            irr1_latency  = lat_each,
            irr2_latency  = lat_each,
            routing       = {},
        ))
    logger.info("T5 done. Total: %.0fs", time.time() - t_start)
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Evaluator B: LLM Judge
# ──────────────────────────────────────────────────────────────────────────────

def run_llm(cases: list[dict]) -> list[Record]:
    from thesis_crag.evaluators.llm_judge import LLMJudgeEvaluator, _judgment_to_float

    ev = LLMJudgeEvaluator(
        prompt_variant=LLM_VARIANT,
        cache_db=LLM_CACHE,
    )
    logger.info("LLM Judge: scoring %d cases…", len(cases))

    records = []
    t_start = time.time()
    for i, case in enumerate(cases):
        if i % 20 == 0:
            logger.info(
                "LLM case %d/%d  api_calls=%d  cost=$%.4f",
                i, len(cases), ev.total_api_calls, ev.estimated_cost_usd,
            )
        q = case["question"]
        passages = [
            case["gold_passage"],
            case["trap_passage"],
            case["irrelevant_passages"][0],
            case["irrelevant_passages"][1],
        ]
        scores, relevants, lats = [], [], []
        for p in passages:
            t0 = time.time()
            j = ev.score(q, p)
            lats.append(time.time() - t0)
            scores.append(_judgment_to_float(j))
            relevants.append(j.relevant)

        records.append(Record(
            question_id   = str(case["question_id"]),
            trap_type     = case["trap_type"],
            gold_relevant = relevants[0],
            trap_relevant = relevants[1],
            irr1_relevant = relevants[2],
            irr2_relevant = relevants[3],
            gold_score    = scores[0],
            trap_score    = scores[1],
            irr1_score    = scores[2],
            irr2_score    = scores[3],
            gold_latency  = lats[0],
            trap_latency  = lats[1],
            irr1_latency  = lats[2],
            irr2_latency  = lats[3],
            routing       = {},
        ))

    ev.close()
    logger.info(
        "LLM done. API calls=%d  cost=$%.4f  time=%.0fs",
        ev.total_api_calls, ev.estimated_cost_usd, time.time() - t_start,
    )
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Evaluator C: Cross-Encoder v2
# ──────────────────────────────────────────────────────────────────────────────

def run_ce(cases: list[dict]) -> list[Record]:
    from thesis_crag.evaluators.cross_encoder import CrossEncoderEvaluator

    ev = CrossEncoderEvaluator(model_path=CE_MODEL, device="cpu", batch_size=32)
    logger.info("CE v2: scoring %d cases…", len(cases))

    records = []
    t_start = time.time()
    for i, case in enumerate(cases):
        if i % 40 == 0:
            logger.info("CE case %d/%d", i, len(cases))
        q = case["question"]
        passages = [
            case["gold_passage"],
            case["trap_passage"],
            case["irrelevant_passages"][0],
            case["irrelevant_passages"][1],
        ]
        scores, lats = [], []
        for p in passages:
            t0 = time.time()
            s = ev.score(q, p)
            lats.append(time.time() - t0)
            scores.append(s)

        records.append(Record(
            question_id   = str(case["question_id"]),
            trap_type     = case["trap_type"],
            gold_relevant = scores[0] >= CE_CORRECT_THR,
            trap_relevant = scores[1] >= CE_CORRECT_THR,
            irr1_relevant = scores[2] >= CE_CORRECT_THR,
            irr2_relevant = scores[3] >= CE_CORRECT_THR,
            gold_score    = scores[0],
            trap_score    = scores[1],
            irr1_score    = scores[2],
            irr2_score    = scores[3],
            gold_latency  = lats[0],
            trap_latency  = lats[1],
            irr1_latency  = lats[2],
            irr2_latency  = lats[3],
            routing       = {},
        ))
    logger.info("CE done. Total: %.0fs", time.time() - t_start)
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Evaluator D: Hybrid v2
# ──────────────────────────────────────────────────────────────────────────────

def run_hybrid(cases: list[dict]) -> list[Record]:
    from thesis_crag.evaluators.hybrid import HybridEvaluator

    ev = HybridEvaluator(
        cross_encoder_path=CE_MODEL,
        llm_prompt_variant=LLM_VARIANT,
        high_threshold=HIGH_THR,
        low_threshold=LOW_THR,
        device="cpu",
        cache_db=LLM_CACHE,
    )
    logger.info("Hybrid v2: scoring %d cases…", len(cases))

    records = []
    t_start = time.time()
    for i, case in enumerate(cases):
        if i % 40 == 0:
            logger.info(
                "Hybrid case %d/%d  ce_only=%d  llm=%d",
                i, len(cases), ev.cross_encoder_only, ev.llm_called,
            )
        q = case["question"]
        passages = [
            case["gold_passage"],
            case["trap_passage"],
            case["irrelevant_passages"][0],
            case["irrelevant_passages"][1],
        ]
        labels = ["gold", "trap", "irr1", "irr2"]
        scores, routers, lats = [], [], []
        for p in passages:
            t0 = time.time()
            s = ev.score(q, p)
            lats.append(time.time() - t0)
            scores.append(s)
            routers.append(ev._last_router)

        records.append(Record(
            question_id   = str(case["question_id"]),
            trap_type     = case["trap_type"],
            gold_relevant = scores[0] >= ev.correct_threshold,
            trap_relevant = scores[1] >= ev.correct_threshold,
            irr1_relevant = scores[2] >= ev.correct_threshold,
            irr2_relevant = scores[3] >= ev.correct_threshold,
            gold_score    = scores[0],
            trap_score    = scores[1],
            irr1_score    = scores[2],
            irr2_score    = scores[3],
            gold_latency  = lats[0],
            trap_latency  = lats[1],
            irr1_latency  = lats[2],
            irr2_latency  = lats[3],
            routing       = dict(zip(labels, routers, strict=True)),
        ))

    ev.close()
    logger.info(
        "Hybrid done. ce_only=%d  llm=%d  new_api=%d  cost=$%.4f  time=%.0fs",
        ev.cross_encoder_only, ev.llm_called,
        ev.total_llm_api_calls, ev.estimated_cost_usd,
        time.time() - t_start,
    )
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Comparison table
# ──────────────────────────────────────────────────────────────────────────────

def _fmt(val: float, lo: float, hi: float, pct: bool = True) -> str:
    if pct:
        return f"{val:.1%} [{lo:.1%}–{hi:.1%}]"
    return f"{val:.4f} [{lo:.4f}–{hi:.4f}]"


def print_comparison(
    label_records: dict[str, list[Record]],
    cis: dict[str, dict[str, tuple[float, float, float]]],
) -> str:
    lines = []
    col_labels = {"T5": "T5", "LLM": "LLM Judge", "CE": "CE v2", "Hybrid": "Hybrid v2"}
    keys = [k for k in ["T5", "LLM", "CE", "Hybrid"] if k in cis]
    header = f"{'Metric':<30}" + "".join(f" {col_labels[k]:>30}" for k in keys)
    sep = "─" * len(header)
    lines.append("")
    lines.append("=" * len(header))
    lines.append("PHASE 5  —  4-WAY EVALUATOR COMPARISON  (splits_v2 test, n=212)")
    lines.append("  Bootstrap 95% CI in brackets [n=1000 case-level resamples]")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append(sep)
    metrics_to_show = [
        ("trap_detection_rate", "Trap Detection",  True),
        ("gold_recall",         "Gold Recall",      True),
        ("fpr_at_1",            "FPR@1",            True),
        ("precision_at_1",      "Precision@1",      True),
        ("recall_at_2",         "Recall@2",         True),
        ("mrr",                 "MRR",              True),
    ]
    for metric_key, label, pct in metrics_to_show:
        row = f"{label:<30}"
        for k in keys:
            v, lo, hi = cis[k][metric_key]
            row += f" {_fmt(v, lo, hi, pct):>30}"
        lines.append(row)

    lines.append(sep)

    # Per-trap-type breakdown (point estimates only)
    all_tt = set()
    for recs in label_records.values():
        for r in recs:
            all_tt.add(r.trap_type)

    lines.append("Per-trap-type trap detection (point estimates, no CI):")
    for tt in sorted(all_tt):
        row = f"  {tt:<28}"
        for k in keys:
            recs = label_records[k]
            sub = [r for r in recs if r.trap_type == tt]
            if not sub:
                row += f" {'—':>30}"
                continue
            det = sum(1 for r in sub if not r.trap_relevant)
            rate = det / len(sub)
            row += f" {rate:.1%} ({det}/{len(sub)}){' ':>19}"
        lines.append(row)

    lines.append(sep)

    # Latency
    lines.append("Latency (median per passage):")
    for k, recs in label_records.items():
        stats = _latency_stats(recs)
        lines.append(f"  {k:<12} med={stats['latency_median_s']:.3f}s  p95={stats['latency_p95_s']:.3f}s")

    # Routing stats for hybrid
    lines.append("")
    lines.append("Hybrid routing stats:")
    hybrid_recs = label_records.get("Hybrid", [])
    rst = _routing_stats(hybrid_recs)
    if rst:
        lines.append(f"  Total passages : {rst['total_passages']}")
        lines.append(f"  CE decisive    : {rst['ce_decisive']} ({1-rst['llm_pct']:.1%})")
        lines.append(f"  LLM escalated  : {rst['llm_escalated']} ({rst['llm_pct']:.1%})")

    lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 full evaluation pipeline.")
    parser.add_argument("--skip-t5",     action="store_true")
    parser.add_argument("--skip-llm",    action="store_true")
    parser.add_argument("--skip-ce",     action="store_true")
    parser.add_argument("--skip-hybrid", action="store_true")
    parser.add_argument("--only-table",  action="store_true",
                        help="Skip all inference, recompute table from saved JSONLs.")
    parser.add_argument("--n-boot",      type=int, default=N_BOOT)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cases: list[dict] = []
    with open(TEST_PATH) as f:
        for line in f:
            cases.append(json.loads(line))
    logger.info("Loaded %d test cases from %s", len(cases), TEST_PATH)

    JSONL = {
        "T5":     RESULTS_DIR / "t5_eval.jsonl",
        "LLM":    RESULTS_DIR / "llm_eval.jsonl",
        "CE":     RESULTS_DIR / "ce_v2_eval.jsonl",
        "Hybrid": RESULTS_DIR / "hybrid_v2_eval.jsonl",
    }

    label_records: dict[str, list[Record]] = {}

    if not args.only_table:

        # ── T5 ────────────────────────────────────────────────────────────
        if not args.skip_t5 and not JSONL["T5"].exists():
            logger.info("═" * 50)
            logger.info("EVALUATOR A: T5 BASELINE")
            logger.info("═" * 50)
            recs = run_t5(cases)
            _save_records(recs, JSONL["T5"])
            logger.info("T5 saved → %s", JSONL["T5"])
        elif JSONL["T5"].exists():
            logger.info("T5 JSONL exists, loading from %s", JSONL["T5"])

        # ── LLM Judge ─────────────────────────────────────────────────────
        if not args.skip_llm and not JSONL["LLM"].exists():
            logger.info("═" * 50)
            logger.info("EVALUATOR B: LLM JUDGE")
            logger.info("═" * 50)
            recs = run_llm(cases)
            _save_records(recs, JSONL["LLM"])
            logger.info("LLM saved → %s", JSONL["LLM"])
        elif JSONL["LLM"].exists():
            logger.info("LLM JSONL exists, loading from %s", JSONL["LLM"])

        # ── Cross-Encoder v2 ──────────────────────────────────────────────
        if not args.skip_ce and not JSONL["CE"].exists():
            logger.info("═" * 50)
            logger.info("EVALUATOR C: CROSS-ENCODER v2")
            logger.info("═" * 50)
            recs = run_ce(cases)
            _save_records(recs, JSONL["CE"])
            logger.info("CE saved → %s", JSONL["CE"])
        elif JSONL["CE"].exists():
            logger.info("CE JSONL exists, loading from %s", JSONL["CE"])

        # ── Hybrid v2 ─────────────────────────────────────────────────────
        if not args.skip_hybrid and not JSONL["Hybrid"].exists():
            logger.info("═" * 50)
            logger.info("EVALUATOR D: HYBRID v2")
            logger.info("═" * 50)
            recs = run_hybrid(cases)
            _save_records(recs, JSONL["Hybrid"])
            logger.info("Hybrid saved → %s", JSONL["Hybrid"])
        elif JSONL["Hybrid"].exists():
            logger.info("Hybrid JSONL exists, loading from %s", JSONL["Hybrid"])

    # ── Load all saved JSONLs ─────────────────────────────────────────────
    for key, path in JSONL.items():
        if path.exists():
            label_records[key] = _records_from_jsonl(path)
            logger.info("Loaded %d records for %s", len(label_records[key]), key)
        else:
            logger.warning("No JSONL for %s at %s — skipped in table", key, path)

    if not label_records:
        logger.error("No evaluator results available. Run without --only-table first.")
        sys.exit(1)

    # ── Bootstrap CIs ────────────────────────────────────────────────────
    logger.info("Computing bootstrap CIs (n=%d) for %d evaluators…",
                args.n_boot, len(label_records))
    cis: dict[str, dict[str, tuple[float, float, float]]] = {}
    for key, recs in label_records.items():
        logger.info("  Bootstrapping %s…", key)
        cis[key] = bootstrap_ci(recs, n_boot=args.n_boot)

    # ── Print + save table ───────────────────────────────────────────────
    table = print_comparison(label_records, cis)
    print(table)

    md_path = RESULTS_DIR / "comparison.md"
    md_path.write_text(f"```\n{table}\n```\n")
    logger.info("Markdown saved → %s", md_path)

    # ── Save full comparison JSON ─────────────────────────────────────────
    comparison = {}
    for key in label_records:
        recs = label_records[key]
        lat   = _latency_stats(recs)
        ci    = cis[key]
        comparison[key] = {
            "n_cases": len(recs),
            "metrics": {
                m: {
                    "point": round(ci[m][0], 4),
                    "ci_lo": round(ci[m][1], 4),
                    "ci_hi": round(ci[m][2], 4),
                }
                for m in ci
            },
            "latency": lat,
        }
        if key == "Hybrid":
            comparison[key]["routing"] = _routing_stats(recs)

    (RESULTS_DIR / "comparison.json").write_text(json.dumps(comparison, indent=2))
    logger.info("Full JSON saved → %s", RESULTS_DIR / "comparison.json")


if __name__ == "__main__":
    main()
