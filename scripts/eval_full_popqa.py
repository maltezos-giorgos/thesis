"""Evaluate CE v2, LLM-as-Judge, and Hybrid on the full PopQA dataset.

Mirrors the Phase 1 T5 baseline (reproduce_baseline.py) but for the three
proposed evaluators. Crash-safe: each evaluator writes one JSONL line per
question immediately after scoring and resumes from the checkpoint on restart.

Usage
-----
    # Dry-run: first 10 questions only
    python scripts/eval_full_popqa.py --limit 10

    # Full run (sequential, ~6 hours)
    python scripts/eval_full_popqa.py

    # Run a single evaluator
    python scripts/eval_full_popqa.py --skip-llm --skip-hybrid

Output
------
    results/full_popqa/ce_v2_results.jsonl
    results/full_popqa/llm_results.jsonl
    results/full_popqa/hybrid_results.jsonl

Each line (per question):
    {
        "question_id": int,
        "question": str,
        "assigned_action": "CORRECT" | "AMBIGUOUS" | "INCORRECT",
        "top1_score": float,
        "top1_gt_relevant": bool,
        "all_scores": [float × n_docs],
        "all_gt_labels": [bool × n_docs],
        "elapsed_s": float,
        // hybrid only:
        "llm_route_count": int,
        "new_api_calls": int
    }

Cost estimate (full run)
------------------------
    CE v2    : free, ~2-3 h on CPU
    LLM Judge: ~13 990 API calls × $0.000480 ≈ $6.72
    Hybrid   : ~15-20% escalation → ~2 100-2 800 calls × $0.000480 ≈ $1.00-1.34
    Total    : ~$7.70-8.06  (cache hits from prior runs cost $0)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Torchvision stub — must precede any sentence-transformers / transformers import
import torch
from tqdm import tqdm

try:
    _tv = torch.library.Library("torchvision", "FRAGMENT")
    _tv.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    pass

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_crag.evaluators.cross_encoder import CrossEncoderEvaluator
from thesis_crag.evaluators.hybrid import HybridEvaluator
from thesis_crag.evaluators.llm_judge import LLMJudgeEvaluator, _judgment_to_float
from thesis_crag.utils.logging import get_logger

logger = get_logger("eval_full_popqa")

POPQA_PATH  = REPO_ROOT / "external/CRAG/eval_data/popqa_longtail_w_gs.jsonl"
CE_MODEL    = str(REPO_ROOT / "models/cross_encoder_v2")
CACHE_DIR   = REPO_ROOT / "data/cache"
RESULTS_DIR = REPO_ROOT / "results/full_popqa"

LLM_VARIANT     = "WITH_NEGATIVE_EXAMPLES"
LLM_CACHE_DB    = str(CACHE_DIR / "llm_judge_full_popqa.db")
HYBRID_CACHE_DB = str(CACHE_DIR / "llm_judge_full_popqa.db")  # shared cache
HIGH_THR        = 0.8
LOW_THR         = 0.2
N_DOCS          = 10   # match Phase 1


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_popqa(path: Path) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_checkpoint(path: Path) -> tuple[set[int], list[dict]]:
    done: set[int] = set()
    saved: list[dict] = []
    if not path.exists():
        return done, saved
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                done.add(r["question_id"])
                saved.append(r)
    return done, saved


def build_passage_text(ctx: dict) -> str:
    return ctx["text"].strip().replace("\n", " ").replace("\t", " ")


def is_relevant(ctx: dict, s_wiki_title: str) -> bool:
    return ctx["title"].strip() == s_wiki_title.strip()


# ---------------------------------------------------------------------------
# Per-evaluator run functions
# ---------------------------------------------------------------------------

def run_ce(items: list[dict], output_path: Path, done_ids: set[int]) -> list[dict]:
    logger.info("Loading CE v2 from %s", CE_MODEL)
    ev = CrossEncoderEvaluator(CE_MODEL, device="cpu", batch_size=32)
    results: list[dict] = []
    t0 = time.time()

    with open(output_path, "a") as f:
        for item in tqdm(items, desc="CE v2"):
            qid = int(item["id"])
            if qid in done_ids:
                continue
            q        = item["question"]
            ctxs     = item["ctxs"][:N_DOCS]
            s_wiki   = item.get("s_wiki_title", "")
            passages = [build_passage_text(c) for c in ctxs]
            gt       = [is_relevant(c, s_wiki) for c in ctxs]

            t1       = time.time()
            scores   = [ev.score(q, p) for p in passages]
            action   = ev.classify_action(scores)

            row = {
                "question_id":      qid,
                "question":         q,
                "assigned_action":  str(action),
                "top1_score":       round(scores[0], 6),
                "top1_gt_relevant": gt[0],
                "all_scores":       [round(s, 6) for s in scores],
                "all_gt_labels":    gt,
                "elapsed_s":        round(time.time() - t1, 3),
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            results.append(row)

    logger.info("CE v2 done in %.0fs — %d questions", time.time() - t0, len(results))
    return results


def run_llm(items: list[dict], output_path: Path, done_ids: set[int]) -> list[dict]:
    logger.info("Loading LLM Judge (%s, cache: %s)", LLM_VARIANT, LLM_CACHE_DB)
    ev = LLMJudgeEvaluator(prompt_variant=LLM_VARIANT, cache_db=LLM_CACHE_DB)
    results: list[dict] = []
    t0 = time.time()

    with open(output_path, "a") as f:
        for i, item in enumerate(tqdm(items, desc="LLM Judge")):
            qid = int(item["id"])
            if qid in done_ids:
                continue
            q        = item["question"]
            ctxs     = item["ctxs"][:N_DOCS]
            s_wiki   = item.get("s_wiki_title", "")
            passages = [build_passage_text(c) for c in ctxs]
            gt       = [is_relevant(c, s_wiki) for c in ctxs]

            t1 = time.time()
            judgments = [ev.score(q, p) for p in passages]
            scores    = [_judgment_to_float(j) for j in judgments]
            action    = ev.classify_action(judgments)

            row = {
                "question_id":      qid,
                "question":         q,
                "assigned_action":  str(action),
                "top1_score":       round(scores[0], 6),
                "top1_gt_relevant": gt[0],
                "all_scores":       [round(s, 6) for s in scores],
                "all_gt_labels":    gt,
                "elapsed_s":        round(time.time() - t1, 3),
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            results.append(row)

            if (i + 1) % 100 == 0:
                logger.info(
                    "LLM %d/%d | api_calls=%d cost=$%.4f",
                    i + 1, len(items), ev.total_api_calls, ev.estimated_cost_usd,
                )

    ev.close()
    logger.info(
        "LLM done in %.0fs — api_calls=%d cost=$%.4f",
        time.time() - t0, ev.total_api_calls, ev.estimated_cost_usd,
    )
    return results


def run_hybrid(items: list[dict], output_path: Path, done_ids: set[int]) -> list[dict]:
    logger.info("Loading Hybrid (CE v2 + LLM, thresholds %.2f/%.2f, cache: %s)",
                HIGH_THR, LOW_THR, HYBRID_CACHE_DB)
    ev = HybridEvaluator(
        cross_encoder_path=CE_MODEL,
        llm_prompt_variant=LLM_VARIANT,
        high_threshold=HIGH_THR,
        low_threshold=LOW_THR,
        device="cpu",
        cache_db=HYBRID_CACHE_DB,
    )
    results: list[dict] = []
    t0 = time.time()

    with open(output_path, "a") as f:
        for i, item in enumerate(tqdm(items, desc="Hybrid")):
            qid = int(item["id"])
            if qid in done_ids:
                continue
            q        = item["question"]
            ctxs     = item["ctxs"][:N_DOCS]
            s_wiki   = item.get("s_wiki_title", "")
            passages = [build_passage_text(c) for c in ctxs]
            gt       = [is_relevant(c, s_wiki) for c in ctxs]

            t1             = time.time()
            llm_before     = ev.llm_called
            api_before     = ev.total_llm_api_calls
            scores         = [ev.score(q, p) for p in passages]
            action         = ev.classify_action(scores)
            llm_this_q     = ev.llm_called - llm_before
            api_this_q     = ev.total_llm_api_calls - api_before

            row = {
                "question_id":      qid,
                "question":         q,
                "assigned_action":  str(action),
                "top1_score":       round(scores[0], 6),
                "top1_gt_relevant": gt[0],
                "all_scores":       [round(s, 6) for s in scores],
                "all_gt_labels":    gt,
                "elapsed_s":        round(time.time() - t1, 3),
                "llm_route_count":  llm_this_q,
                "new_api_calls":    api_this_q,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            results.append(row)

            if (i + 1) % 100 == 0:
                logger.info(
                    "Hybrid %d/%d | ce_only=%d llm=%d new_api=%d cost=$%.4f",
                    i + 1, len(items),
                    ev.cross_encoder_only, ev.llm_called,
                    ev.total_llm_api_calls, ev.estimated_cost_usd,
                )

    ev.close()
    logger.info(
        "Hybrid done in %.0fs — ce_only=%d llm=%d new_api=%d cost=$%.4f",
        time.time() - t0,
        ev.cross_encoder_only, ev.llm_called,
        ev.total_llm_api_calls, ev.estimated_cost_usd,
    )
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(label: str, results: list[dict]) -> None:
    n = len(results)
    if n == 0:
        return
    from collections import Counter
    actions = Counter(r["assigned_action"] for r in results)
    fpr_num = sum(
        1 for r in results
        if r["top1_score"] >= 0.5 and not r["top1_gt_relevant"]
    )
    # For LLM judge the CORRECT threshold is 0.7, but top1_gt FPR uses >0.5
    # as a conservative proxy; aggregate script computes exact per-evaluator FPR.
    fpr = fpr_num / n

    print(f"\n{'='*55}")
    print(f"{label} — {n} questions")
    print(f"{'='*55}")
    for a in ("CORRECT", "AMBIGUOUS", "INCORRECT"):
        cnt = actions.get(a, 0)
        print(f"  {a:<12} {cnt:>5}  ({100*cnt/n:.1f}%)")
    print(f"  FPR@1 (proxy): {fpr:.4f}")

    if any("llm_route_count" in r for r in results):
        total_passages = n * N_DOCS
        llm_routed = sum(r.get("llm_route_count", 0) for r in results)
        print(f"  LLM routed:    {llm_routed}/{total_passages} ({100*llm_routed/total_passages:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CE v2, LLM Judge, and Hybrid on full PopQA."
    )
    parser.add_argument("--limit",        type=int,  default=None,
                        help="Process only first N questions (dry-run).")
    parser.add_argument("--skip-ce",      action="store_true")
    parser.add_argument("--skip-llm",     action="store_true")
    parser.add_argument("--skip-hybrid",  action="store_true")
    parser.add_argument("--no-resume",    action="store_true",
                        help="Ignore existing checkpoints and start from scratch.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading PopQA from %s", POPQA_PATH)
    items = load_popqa(POPQA_PATH)
    if args.limit:
        items = items[: args.limit]
        logger.info("DRY RUN — processing %d questions", len(items))
    else:
        logger.info("Full run — %d questions", len(items))

    CE_PATH     = RESULTS_DIR / "ce_v2_results.jsonl"
    LLM_PATH    = RESULTS_DIR / "llm_results.jsonl"
    HYBRID_PATH = RESULTS_DIR / "hybrid_results.jsonl"

    # ── Cross-Encoder v2 ──────────────────────────────────────────────────
    if not args.skip_ce:
        if args.no_resume and CE_PATH.exists():
            CE_PATH.unlink()
        done_ids, prior = load_checkpoint(CE_PATH)
        if done_ids:
            logger.info("CE: resuming — %d already done", len(done_ids))
        new = run_ce(items, CE_PATH, done_ids)
        print_summary("CE v2", prior + new)

    # ── LLM Judge ─────────────────────────────────────────────────────────
    if not args.skip_llm:
        if args.no_resume and LLM_PATH.exists():
            LLM_PATH.unlink()
        done_ids, prior = load_checkpoint(LLM_PATH)
        if done_ids:
            logger.info("LLM: resuming — %d already done", len(done_ids))
        new = run_llm(items, LLM_PATH, done_ids)
        print_summary("LLM Judge", prior + new)

    # ── Hybrid ────────────────────────────────────────────────────────────
    if not args.skip_hybrid:
        if args.no_resume and HYBRID_PATH.exists():
            HYBRID_PATH.unlink()
        done_ids, prior = load_checkpoint(HYBRID_PATH)
        if done_ids:
            logger.info("Hybrid: resuming — %d already done", len(done_ids))
        new = run_hybrid(items, HYBRID_PATH, done_ids)
        print_summary("Hybrid", prior + new)


if __name__ == "__main__":
    main()
