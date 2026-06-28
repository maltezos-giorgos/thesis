"""Prepare cross-encoder training pairs from hard-cases splits.

Each hard case produces 4 pairs: (query, gold) → 0.9, (query, trap) → 0.1,
(query, irr1) → 0.1, (query, irr2) → 0.1. entity_alias cases are oversampled ×3
because they represent the hardest failure mode (only ~20% of data, 65% detection
by the LLM judge vs 85% for other types).

Usage:
    python scripts/prepare_training_data.py [--train PATH] [--val PATH] [--out-dir DIR]

Output:
    data/training_v2/cross_encoder_train.jsonl
    data/training_v2/cross_encoder_val.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

LABEL_POS: float = 0.9
LABEL_NEG: float = 0.1
ENTITY_ALIAS_MULTIPLIER: int = 3


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def make_pairs(
    cases: list[dict],
    oversample_entity_alias: bool = True,
    entity_alias_multiplier: int = ENTITY_ALIAS_MULTIPLIER,
) -> list[dict]:
    """Convert hard cases to (query, passage, label) training pairs.

    Uses only the first two irrelevant passages per case (irr1, irr2).
    entity_alias cases are duplicated entity_alias_multiplier times when
    oversample_entity_alias=True.
    """
    pairs: list[dict] = []
    for case in cases:
        q = case["question"]
        trap_type = case.get("trap_type", "unknown")
        base = [
            {"query": q, "passage": case["gold_passage"],             "label": LABEL_POS, "trap_type": trap_type},
            {"query": q, "passage": case["trap_passage"],             "label": LABEL_NEG, "trap_type": trap_type},
            {"query": q, "passage": case["irrelevant_passages"][0],   "label": LABEL_NEG, "trap_type": trap_type},
            {"query": q, "passage": case["irrelevant_passages"][1],   "label": LABEL_NEG, "trap_type": trap_type},
        ]
        multiplier = entity_alias_multiplier if (oversample_entity_alias and trap_type == "entity_alias") else 1
        pairs.extend(base * multiplier)
    return pairs


def print_stats(pairs: list[dict], split_name: str) -> None:
    total = len(pairs)
    pos = sum(1 for p in pairs if p["label"] > 0.5)
    neg = total - pos
    trap_counts: Counter = Counter(p["trap_type"] for p in pairs)
    print(f"\n=== {split_name} ===")
    print(f"Total pairs : {total}")
    print(f"Positive    : {pos}  ({pos / total:.1%})")
    print(f"Negative    : {neg}  ({neg / total:.1%})")
    print(f"Pos:Neg     : 1:{neg / pos:.2f}")
    print("Per trap type (all pairs, including oversampled):")
    for tt, count in sorted(trap_counts.items()):
        print(f"  {tt:<24} {count}")


def save_jsonl(pairs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cross-encoder training data from hard-cases splits.")
    parser.add_argument("--train",            default="data/hard_cases/splits_v2/train.jsonl")
    parser.add_argument("--val",              default="data/hard_cases/splits_v2/val.jsonl")
    parser.add_argument("--out-dir",          default="data/training_v2")
    parser.add_argument("--oversample-factor", type=int, default=ENTITY_ALIAS_MULTIPLIER,
                        help="Oversampling multiplier for entity_alias cases (default: 3)")
    parser.add_argument("--seed",             type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    train_cases = load_jsonl(Path(args.train))
    val_cases   = load_jsonl(Path(args.val))

    train_pairs = make_pairs(train_cases, oversample_entity_alias=True,
                             entity_alias_multiplier=args.oversample_factor)
    random.shuffle(train_pairs)
    val_pairs = make_pairs(val_cases, oversample_entity_alias=False)

    out_dir = Path(args.out_dir)
    save_jsonl(train_pairs, out_dir / "cross_encoder_train.jsonl")
    save_jsonl(val_pairs,   out_dir / "cross_encoder_val.jsonl")

    print_stats(train_pairs, "Train")
    print_stats(val_pairs,   "Val")
    print(f"\nSaved to {out_dir}/")
    print(f"  cross_encoder_train.jsonl  ({len(train_pairs)} pairs)")
    print(f"  cross_encoder_val.jsonl    ({len(val_pairs)} pairs)")


if __name__ == "__main__":
    main()
