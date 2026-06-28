"""Phase 2: Mine hard cases from PopQA long-tail retrieved passages.

Usage:
    # Dry run (default): process up to 100 hard cases, then stop and show samples
    python scripts/mine_hard_cases.py

    # Full run: mine all available questions
    python scripts/mine_hard_cases.py --no-dry-run

    # Custom limit
    python scripts/mine_hard_cases.py --limit 500

Output:
    data/hard_cases/hard_cases_YYYYMMDD_HHMMSS.jsonl   (all hard cases)
    data/hard_cases/splits/train.jsonl
    data/hard_cases/splits/val.jsonl
    data/hard_cases/splits/test.jsonl

STOP CONDITION: after a dry run completes, this script prints 5 sample hard cases
and exits without writing splits. Inspect the samples, then rerun with --no-dry-run.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Torchvision stub must be first
import torch

try:
    _tv = torch.library.Library("torchvision", "FRAGMENT")
    _tv.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    pass

from thesis_crag.data.hard_case_miner import HardCase, HardCaseMiner
from thesis_crag.utils.logging import get_logger

logger = get_logger("mine_hard_cases")

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_EVAL_DATA = REPO_ROOT / "external/CRAG/eval_data/popqa_longtail_w_gs.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/hard_cases"
DEFAULT_CACHE = REPO_ROOT / "data/hard_cases/llm_cache.db"

DRY_RUN_LIMIT = 100
TRAIN_FRAC, VAL_FRAC = 0.7, 0.15  # test gets the rest


def load_popqa_raw(path: Path) -> list[dict]:
    """Load all non-empty JSONL lines from the PopQA long-tail file."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_splits(hard_cases: list[HardCase], output_dir: Path) -> None:
    """Shuffle and split hard cases 70/15/15 into train/val/test JSONL files."""
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    random.shuffle(hard_cases)
    n = len(hard_cases)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    splits = {
        "train": hard_cases[:n_train],
        "val": hard_cases[n_train : n_train + n_val],
        "test": hard_cases[n_train + n_val :],
    }
    for name, subset in splits.items():
        out = splits_dir / f"{name}.jsonl"
        with open(out, "w") as f:
            for hc in subset:
                f.write(hc.model_dump_json() + "\n")
        logger.info("Split %s: %d examples -> %s", name, len(subset), out)


def print_samples(hard_cases: list[HardCase], n: int = 5) -> None:
    """Print n hard case examples for human inspection after a dry run."""
    print("\n" + "=" * 70)
    print(f"PHASE 2 DRY RUN COMPLETE — {len(hard_cases)} hard cases mined")
    print("=" * 70)
    print(f"\n--- Sample {n} hard cases ---\n")
    for hc in hard_cases[:n]:
        print(f"Q: {hc.question}")
        print(f"  Answers: {', '.join(hc.answers[:3])}")
        print(f"  Gold title: {hc.gold_passage_title}")
        print(f"  Gold passage: {hc.gold_passage[:120]}...")
        print(
            f"  Trap title: {hc.trap_passage_title}"
            f"  [score={hc.trap_score:.3f}, type={hc.trap_type}]"
        )
        print(f"  Trap passage: {hc.trap_passage[:120]}...")
        print()


def print_summary(hard_cases: list[HardCase], elapsed: float, llm_calls: int) -> None:
    """Print mining statistics: count, elapsed time, cost, and trap-type distribution."""
    trap_type_counts: dict[str, int] = {}
    for hc in hard_cases:
        trap_type_counts[hc.trap_type] = trap_type_counts.get(hc.trap_type, 0) + 1

    print("\n--- Statistics ---")
    print(f"  Hard cases mined  : {len(hard_cases)}")
    print(f"  Elapsed           : {elapsed:.1f}s")
    print(f"  LLM API calls     : {llm_calls} (new, excluding cache hits)")
    # Haiku pricing: ~$0.80/M input + $4/M output tokens. ~200 in + 80 out per call.
    cost_usd = llm_calls * (200 * 0.80 + 80 * 4.0) / 1_000_000
    print(f"  Estimated cost    : ${cost_usd:.4f}")
    print("\n  Trap type distribution:")
    for ttype, cnt in sorted(trap_type_counts.items(), key=lambda x: -x[1]):
        print(f"    {ttype:<20} {cnt:>5}  ({100*cnt/len(hard_cases):.1f}%)")


def main() -> None:
    """Mine hard cases from PopQA long-tail, optionally writing train/val/test splits."""
    parser = argparse.ArgumentParser(description="Mine hard cases from PopQA long-tail")
    parser.add_argument(
        "--no-dry-run", action="store_true",
        help="Run the full pipeline (default: dry run with --limit 100)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max hard cases to mine (overrides dry-run default of 100)",
    )
    parser.add_argument("--eval-data", type=Path, default=DEFAULT_EVAL_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dry_run = not args.no_dry_run
    limit = args.limit if args.limit is not None else (DRY_RUN_LIMIT if dry_run else None)
    random.seed(args.seed)

    mode_str = f"DRY RUN (limit={limit})" if dry_run else f"FULL RUN (limit={limit or 'all'})"
    logger.info("=== Phase 2: Hard Case Mining — %s ===", mode_str)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output_dir / f"hard_cases_{timestamp}.jsonl"

    logger.info("Loading PopQA long-tail data from %s", args.eval_data)
    items = load_popqa_raw(args.eval_data)
    logger.info("Loaded %d questions", len(items))

    miner = HardCaseMiner(cache_path=args.cache)
    hard_cases: list[HardCase] = []

    t_start = time.time()
    try:
        with open(output_file, "w") as out_f:
            for hc in tqdm(
                miner.mine_from_popqa_longtail(items, limit=limit),
                desc="Mining hard cases",
                total=limit,
            ):
                out_f.write(hc.model_dump_json() + "\n")
                out_f.flush()
                hard_cases.append(hc)
    finally:
        miner.close()

    elapsed = time.time() - t_start
    logger.info("Mining complete in %.1fs. Output: %s", elapsed, output_file)

    print_summary(hard_cases, elapsed, miner.total_llm_calls)

    if dry_run:
        # STOP: show samples for human inspection before committing to full run
        print_samples(hard_cases)
        print(
            "\n[DRY RUN COMPLETE]\n"
            "Inspect the 5 samples above.\n"
            "If quality looks good, run the full pipeline:\n"
            "  python scripts/mine_hard_cases.py --no-dry-run\n"
        )
        return

    # Full run: write train/val/test splits
    write_splits(hard_cases, args.output_dir)
    logger.info("All done. Hard cases at %s", output_file)


if __name__ == "__main__":
    main()
