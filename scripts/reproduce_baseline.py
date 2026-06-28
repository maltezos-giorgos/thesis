"""Phase 1: reproduce CRAG baseline action assignment on PopQA long-tail subset.

Supports crash-safe checkpointing: results are written one line per question
immediately after scoring. On restart, already-processed question IDs are
read from the checkpoint file and skipped, so the run resumes from where it
stopped.

Usage:
    python scripts/reproduce_baseline.py [--evaluator-path PATH] [--eval-data PATH]
                                         [--output-dir DIR] [--batch-size N]
                                         [--n-docs N] [--limit N] [--no-resume]
"""

from __future__ import annotations

import argparse
import json
import statistics

# Must come before any transformers import — patches CPU torchvision conflict
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thesis_crag.evaluators.base import Action
from thesis_crag.evaluators.t5_baseline import T5Evaluator
from thesis_crag.metrics.retrieval import false_positive_rate_at_1
from thesis_crag.utils.logging import get_logger

logger = get_logger("reproduce_baseline")

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_EVALUATOR = REPO_ROOT / "external/CRAG/models/evaluator"
DEFAULT_EVAL_DATA = REPO_ROOT / "external/CRAG/eval_data/popqa_longtail_w_gs.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/baseline"


def load_popqa(path: Path) -> list[dict]:
    """Load non-empty JSONL lines from a PopQA file."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_checkpoint(path: Path) -> tuple[set[int], list[dict]]:
    """Return (set of already-processed question IDs, list of saved results)."""
    done_ids: set[int] = set()
    saved: list[dict] = []
    if not path.exists():
        return done_ids, saved
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                done_ids.add(r["question_id"])
                saved.append(r)
    return done_ids, saved


def build_passage_text(ctx: dict) -> str:
    """Replicate data_process.py L95: strip newlines and tabs."""
    return ctx["text"].strip().replace("\n", " ").replace("\t", " ")


def is_relevant(ctx: dict, s_wiki_title: str) -> bool:
    """Ground-truth relevance label from data_process.py L96."""
    return ctx["title"].strip() == s_wiki_title.strip()


def print_summary(results: list[dict]) -> None:
    """Print action distribution, score statistics, and FPR@1 for all processed questions."""
    total = len(results)
    if total == 0:
        print("No results to summarise.")
        return

    action_counts: dict[str, int] = {a.value: 0 for a in Action}
    scores_by_action: dict[str, list[float]] = {a.value: [] for a in Action}
    top1_predicted: list[bool] = []
    top1_gt: list[bool] = []

    for r in results:
        action_counts[r["assigned_action"]] += 1
        scores_by_action[r["assigned_action"]].append(r["t5_score"])
        top1_predicted.append(r["t5_score"] >= T5Evaluator.UPPER_THRESHOLD)
        top1_gt.append(r["ground_truth_relevant"])

    fpr = false_positive_rate_at_1(top1_predicted, top1_gt)

    print("\n" + "=" * 60)
    print("PHASE 1 RESULTS — CRAG T5 BASELINE ON POPQA LONG-TAIL")
    print("=" * 60)
    print(f"Questions processed : {total}")
    print()
    print("--- Action distribution ---")
    for action in Action:
        n = action_counts[action.value]
        pct = 100 * n / total if total else 0
        print(f"  {action.value:<12} {n:>5}  ({pct:.1f}%)")
    print()
    print("--- T5 score of top passage (rank-1 Contriever result) by action ---")
    for action in Action:
        s = scores_by_action[action.value]
        if s:
            print(
                f"  {action.value:<12}  mean={statistics.mean(s):+.4f}  "
                f"median={statistics.median(s):+.4f}  n={len(s)}"
            )
    print()
    n_top1_rel = sum(top1_gt)
    print("--- Retrieval quality ---")
    print(f"  Top-1 passage relevant : {n_top1_rel}/{total} ({100*n_top1_rel/total:.1f}%)")
    print(f"  FPR@1 (top passage)    : {fpr:.4f}")
    print()
    print("--- Paper reference (Yan et al. 2024) ---")
    print("  Expected EM (with generation) : ~54.9%")
    print("  Expected action dist          : CORRECT ~40-50%, AMBIGUOUS ~30-40%, INCORRECT ~15-25%")
    print("=" * 60)

    print("\n--- First 5 output records ---")
    for r in results[:5]:
        print(json.dumps(
            {k: v for k, v in r.items() if k not in {"all_passage_scores", "all_gt_labels"}},
            indent=2,
        ))


def main() -> None:
    """Run T5 CRAG baseline on PopQA long-tail with checkpointing and print Phase 1 summary."""
    parser = argparse.ArgumentParser(description="Reproduce CRAG baseline on PopQA")
    parser.add_argument("--evaluator-path", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--eval-data", type=Path, default=DEFAULT_EVAL_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--num-threads", type=int, default=8,
                        help="CPU threads for PyTorch (default 8, matches nproc)")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Disable dynamic INT8 quantization (slower but exact FP32)")
    parser.add_argument("--n-docs", type=int, default=10,
                        help="Number of retrieved passages per question (paper uses 10)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N questions (for debugging)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing checkpoint and start from scratch")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / "phase1_popqa_actions.jsonl"

    # ------------------------------------------------------------------
    # Checkpoint resume
    # ------------------------------------------------------------------
    if args.no_resume and output_file.exists():
        output_file.unlink()
        logger.info("--no-resume: cleared existing checkpoint")

    done_ids, all_results = load_checkpoint(output_file)
    if done_ids:
        logger.info("Resuming: %d questions already in checkpoint, skipping them", len(done_ids))

    # ------------------------------------------------------------------
    logger.info("Loading T5 evaluator from %s", args.evaluator_path)
    evaluator = T5Evaluator(
        model_path=str(args.evaluator_path),
        device="cpu",
        batch_size=args.batch_size,
        quantize=not args.no_quantize,
        num_threads=args.num_threads,
    )
    logger.info(
        "Thresholds: CORRECT >= %.3f, AMBIGUOUS >= %.3f, INCORRECT < %.3f",
        T5Evaluator.UPPER_THRESHOLD,
        T5Evaluator.LOWER_THRESHOLD,
        T5Evaluator.LOWER_THRESHOLD,
    )

    # ------------------------------------------------------------------
    logger.info("Loading eval data from %s", args.eval_data)
    items = load_popqa(args.eval_data)
    if args.limit:
        items = items[: args.limit]

    remaining = [it for it in items if it["id"] not in done_ids]
    logger.info(
        "Total questions: %d  |  Already done: %d  |  To process: %d",
        len(items), len(done_ids), len(remaining),
    )

    if not remaining:
        logger.info("All questions already processed. Printing summary.")
        print_summary(all_results)
        return

    # ------------------------------------------------------------------
    # Open checkpoint file in append mode — one line written per question
    # ------------------------------------------------------------------
    t_start = time.time()
    with open(output_file, "a") as ckpt_f:
        for i, item in enumerate(tqdm(remaining, desc="Scoring passages")):
            ctxs = item["ctxs"][: args.n_docs]
            s_wiki = item["s_wiki_title"]

            queries = [item["question"]] * len(ctxs)
            passages = [build_passage_text(c) for c in ctxs]
            gt_labels = [is_relevant(c, s_wiki) for c in ctxs]

            passage_scores = evaluator.score_batch(queries, passages)
            action = evaluator.classify_action(passage_scores)

            top_score = passage_scores[0]
            top_gt = gt_labels[0]

            result = {
                "question_id": item["id"],
                "question": item["question"],
                "top_passage_text": passages[0],
                "t5_score": round(top_score, 6),
                "assigned_action": action.value,
                "ground_truth_relevant": top_gt,
                "all_passage_scores": [round(s, 6) for s in passage_scores],
                "all_gt_labels": gt_labels,
            }

            # Write and flush immediately — crash-safe checkpoint
            ckpt_f.write(json.dumps(result) + "\n")
            ckpt_f.flush()

            all_results.append(result)

            n_done = len(done_ids) + i + 1
            if n_done % 100 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                remaining_q = len(remaining) - i - 1
                logger.info(
                    "Processed %d/%d total  |  %.2f q/s  |  ETA %.0f min",
                    n_done, len(items), rate, remaining_q / rate / 60,
                )

    elapsed_total = time.time() - t_start
    logger.info("Finished in %.1f min. Results in %s", elapsed_total / 60, output_file)
    print_summary(all_results)


if __name__ == "__main__":
    main()
