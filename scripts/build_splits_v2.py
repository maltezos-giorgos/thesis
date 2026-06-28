"""Merge PopQA + EntityQuestions hard cases and create fresh stratified splits.

Stratification: by source × trap_type (6 strata) so every split has
proportional representation of both sources and all trap types.

Output:
    data/hard_cases/all_hard_cases.jsonl       (merged, 1 414 cases)
    data/hard_cases/splits_v2/train.jsonl      (70%)
    data/hard_cases/splits_v2/val.jsonl        (15%)
    data/hard_cases/splits_v2/test.jsonl       (15%)
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT  = Path(__file__).parent.parent
# NOTE: this raw PopQA mining output was archived to _archive/data_intermediate/
# (it is not needed to reproduce results, since splits_v2/ and all_hard_cases.jsonl
# are already produced). To re-run splitting, restore the file to the path below.
POPQA_PATH = REPO_ROOT / "data/hard_cases/hard_cases_20260424_223019.jsonl"
EQ_PATH    = REPO_ROOT / "data/hard_cases/entity_questions_mined.jsonl"
ALL_PATH   = REPO_ROOT / "data/hard_cases/all_hard_cases.jsonl"
SPLITS_DIR = REPO_ROOT / "data/hard_cases/splits_v2"

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# TEST_FRAC  = 0.15  (remainder)

SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f]


def _pct(x: float) -> str:
    return f"{x:.1%}"


def stratified_split(
    cases: list[dict],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split cases stratified by (source, trap_type)."""
    rng = random.Random(seed)

    # Group by stratum
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for c in cases:
        key = (c["source"], c["trap_type"])
        strata[key].append(c)

    train, val, test = [], [], []
    for key, group in sorted(strata.items()):
        rng.shuffle(group)
        n      = len(group)
        n_val  = max(1, round(n * val_frac))
        n_test = max(1, round(n * (1.0 - train_frac - val_frac)))
        n_train = n - n_val - n_test

        # Guard: if group is tiny, put at least 1 in train
        if n_train < 1:
            n_train = 1
            # Reduce val or test
            if n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1

        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:n_train + n_val + n_test])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def save_jsonl(cases: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")


def print_split_summary(split: list[dict], name: str) -> None:
    n = len(split)
    src  = Counter(c["source"]    for c in split)
    tt   = Counter(c["trap_type"] for c in split)
    print(f"  {name} ({n} cases):")
    for s, cnt in sorted(src.items()):
        print(f"    source={s:<20} {cnt:>4}  ({_pct(cnt/n)})")
    for t, cnt in sorted(tt.items()):
        print(f"    trap_type={t:<15} {cnt:>4}  ({_pct(cnt/n)})")


def main() -> None:
    # ── 1. Load both sources ──────────────────────────────────────────
    popqa_cases = load_jsonl(POPQA_PATH)
    eq_cases    = load_jsonl(EQ_PATH)

    # Normalise source labels
    for c in popqa_cases:
        c["source"] = "popqa"        # was "popqa_longtail"
    for c in eq_cases:
        c["source"] = "entity_questions"   # already set; ensure consistent

    print("=" * 60)
    print("BUILD SPLITS V2")
    print("=" * 60)
    print(f"\nPopQA cases      : {len(popqa_cases)}")
    popqa_tt = Counter(c["trap_type"] for c in popqa_cases)
    for t, n in sorted(popqa_tt.items()):
        print(f"  {t:<20} {n:>4}  ({_pct(n/len(popqa_cases))})")

    print(f"\nEntityQuestions  : {len(eq_cases)}")
    eq_tt = Counter(c["trap_type"] for c in eq_cases)
    for t, n in sorted(eq_tt.items()):
        print(f"  {t:<20} {n:>4}  ({_pct(n/len(eq_cases))})")

    # ── 2. Dedup check ────────────────────────────────────────────────
    all_qids = [c["question_id"] for c in popqa_cases + eq_cases]
    dups = len(all_qids) - len(set(all_qids))
    if dups:
        print(f"\nWARNING: {dups} duplicate question_ids across sources")

    all_questions = [c["question"].lower().strip() for c in popqa_cases + eq_cases]
    dup_qs = len(all_questions) - len(set(all_questions))
    if dup_qs:
        print(f"WARNING: {dup_qs} duplicate question strings across sources")
    else:
        print(f"\nDedup check: 0 duplicate questions — clean merge")

    # ── 3. Merge ──────────────────────────────────────────────────────
    all_cases = popqa_cases + eq_cases
    print(f"\nMerged total     : {len(all_cases)}")
    save_jsonl(all_cases, ALL_PATH)
    print(f"Written: {ALL_PATH}")

    # ── 4. Stratified splits ──────────────────────────────────────────
    train, val, test = stratified_split(all_cases, TRAIN_FRAC, VAL_FRAC, SEED)

    save_jsonl(train, SPLITS_DIR / "train.jsonl")
    save_jsonl(val,   SPLITS_DIR / "val.jsonl")
    save_jsonl(test,  SPLITS_DIR / "test.jsonl")

    total = len(train) + len(val) + len(test)
    print(f"\n── Splits (seed={SEED}) ──────────────────────────────────────")
    print(f"  Total assigned  : {total}  (of {len(all_cases)})")
    print_split_summary(train, "train")
    print_split_summary(val,   "val  ")
    print_split_summary(test,  "test ")
    print(f"\nWritten: {SPLITS_DIR}/")


if __name__ == "__main__":
    main()
