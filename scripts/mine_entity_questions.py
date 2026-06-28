"""Mine hard retrieval cases from EntityQuestions TEST split.

Pipeline for each sampled entry:
  1. Extract subject entity from question using relation template.
  2. Fetch Wikipedia gold + trap candidate passages (cached, no repeat HTTP).
  3. Verify with Claude Haiku:
       - gold_check:  does gold passage contain the answer?
       - trap_checks: does each trap candidate NOT contain the answer?
  4. Pick the best verified trap (most entity-name similarity to gold → hardest).
  5. Classify trap_type: entity_alias | topic_overlap | unknown.
  6. Assign 2 irrelevant passages from the pool of other mined gold passages.
  7. Emit one hard case in the same JSONL schema as data/hard_cases/*.jsonl.

Usage:
    python scripts/mine_entity_questions.py           # dry-run: 20 entries
    python scripts/mine_entity_questions.py --no-dry-run --sample 2000

Output:
    data/hard_cases/entity_questions_mined.jsonl          (--no-dry-run)
    data/hard_cases/entity_questions_mined_dryrun.jsonl   (default dry-run; never
                                                           clobbers the production file)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from fetch_entity_passages import EntityPassageFetcher

from thesis_crag.utils.llm_clients import call_llm_with_validation
from thesis_crag.utils.logging import get_logger

logger = get_logger("mine_entity_questions")

REPO_ROOT  = Path(__file__).parent.parent
EQ_TEST    = REPO_ROOT / "data/raw/entity_questions/dataset/test"
OUT_PATH   = REPO_ROOT / "data/hard_cases/entity_questions_mined.jsonl"
DRYRUN_OUT_PATH = REPO_ROOT / "data/hard_cases/entity_questions_mined_dryrun.jsonl"
WIKI_CACHE = str(REPO_ROOT / "data/cache/wikipedia_passages.db")
LLM_CACHE  = str(REPO_ROOT / "data/cache/mining_verification.db")

HAIKU_MODEL = "claude-haiku-4-5-20251001"
# Haiku pricing (April 2025)
_IN_CPT  = 0.80 / 1_000_000
_OUT_CPT = 4.00 / 1_000_000
_EST_IN  = 300   # tokens per verification call
_EST_OUT = 60
COST_PER_CALL = _EST_IN * _IN_CPT + _EST_OUT * _OUT_CPT

VERIFY_SYSTEM = """\
You are a fact-verification assistant.

Given a question, its expected answer(s), and a Wikipedia passage, determine
whether the passage contains information that directly and correctly answers
the question.

Return ONLY valid JSON in exactly this format:
{"contains_answer": true, "confidence": 0.85, "reasoning": "one sentence"}

Rules:
- contains_answer: true only if the passage explicitly states or strongly implies
  the correct answer. A passage about a related but different entity does NOT count.
- confidence: 0.0 (completely uncertain) to 1.0 (absolutely certain).
- reasoning: one sentence explaining your decision."""


# ---------------------------------------------------------------------------
# SQLite cache for verification calls
# ---------------------------------------------------------------------------

class _VerifyCache:
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS verifications "
            "(key TEXT PRIMARY KEY, result TEXT)"
        )
        self._conn.commit()
        self.api_calls = 0
        self.cost_usd  = 0.0

    def _key(self, question: str, answers: list[str], passage: str) -> str:
        raw = question + "|" + ",".join(sorted(answers)) + "|" + passage[:600]
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, question: str, answers: list[str], passage: str) -> dict | None:
        k = self._key(question, answers, passage)
        row = self._conn.execute(
            "SELECT result FROM verifications WHERE key=?", (k,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, question: str, answers: list[str], passage: str, result: dict) -> None:
        k = self._key(question, answers, passage)
        self._conn.execute(
            "INSERT OR REPLACE INTO verifications VALUES (?,?)",
            (k, json.dumps(result)),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# LLM verification
# ---------------------------------------------------------------------------

def _verify(
    question: str,
    answers: list[str],
    passage: str,
    cache: _VerifyCache,
) -> dict:
    """Return {contains_answer, confidence, reasoning}. Cached."""
    hit = cache.get(question, answers, passage)
    if hit is not None:
        return hit

    user = (
        f"Question: {question}\n"
        f"Expected answer(s): {', '.join(answers)}\n"
        f"Passage: {passage[:700]}"
    )
    try:
        result = call_llm_with_validation(
            VERIFY_SYSTEM, user, model=HAIKU_MODEL, max_tokens=200
        )
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return {"contains_answer": False, "confidence": 0.0, "reasoning": str(exc)}

    cache.api_calls += 1
    cache.cost_usd  += COST_PER_CALL
    cache.put(question, answers, passage, result)
    return result


# ---------------------------------------------------------------------------
# Trap-type classification (token-overlap heuristic)
# ---------------------------------------------------------------------------

_STOPS = {
    "the", "a", "an", "of", "in", "at", "on", "by", "for", "and", "or",
    "its", "is", "are", "was", "were", "it", "this", "that",
}


def _tokens(s: str) -> set[str]:
    return {w for w in re.sub(r"[^\w\s]", " ", s.lower()).split() if w not in _STOPS}


def classify_trap_type(entity: str, trap_title: str) -> str:
    """entity_alias if names share ≥2 meaningful tokens or Jaccard ≥ 0.25."""
    et = _tokens(entity)
    tt = _tokens(trap_title)
    if not et or not tt:
        return "unknown"
    overlap  = et & tt
    jaccard  = len(overlap) / len(et | tt)
    if len(overlap) >= 2 or jaccard >= 0.25:
        return "entity_alias"
    return "topic_overlap"


# ---------------------------------------------------------------------------
# Question-ID generator
# ---------------------------------------------------------------------------

def _qid(question: str, relation: str) -> str:
    raw = f"eq|{relation}|{question}"
    return str(int(hashlib.md5(raw.encode()).hexdigest()[:8], 16))


# ---------------------------------------------------------------------------
# Stratified sampler
# ---------------------------------------------------------------------------

def stratified_sample(n: int, seed: int = 42) -> list[tuple[str, dict]]:
    """Sample n entries proportionally from each relation in the TEST split.

    Returns list of (relation, entry) pairs, shuffled for processing order.
    """
    rng = random.Random(seed)
    relations: dict[str, list[dict]] = {}
    for f in sorted(EQ_TEST.glob("*.test.json")):
        rel = f.stem.split(".")[0]
        data = json.loads(f.read_text())
        rng.shuffle(data)
        relations[rel] = data

    total = sum(len(v) for v in relations.values())
    samples: list[tuple[str, dict]] = []
    for rel, data in relations.items():
        k = max(1, round(n * len(data) / total))
        samples.extend((rel, e) for e in data[:k])

    rng.shuffle(samples)
    return samples[:n]


# ---------------------------------------------------------------------------
# Main mining loop
# ---------------------------------------------------------------------------

def mine(
    sample_size: int,
    dry_run: bool,
    seed: int,
) -> None:
    effective_n = 20 if dry_run else sample_size
    entries = stratified_sample(effective_n, seed=seed)

    # Dry runs write to a SEPARATE file so a no-arg invocation can never
    # clobber the production entity_questions_mined.jsonl used to build splits.
    out_path = DRYRUN_OUT_PATH if dry_run else OUT_PATH

    # Cost estimate
    avg_llm_per_entry = 3.5   # 1 gold_check + ~2.5 trap_checks on average
    est_calls = effective_n * avg_llm_per_entry
    est_cost  = est_calls * COST_PER_CALL
    logger.info("=" * 60)
    logger.info("MINE ENTITY QUESTIONS%s", " [DRY RUN]" if dry_run else "")
    logger.info("Entries to process : %d", len(entries))
    logger.info("Est. LLM calls     : ~%d  (~$%.4f)", int(est_calls), est_cost)
    logger.info("Wikipedia cache    : %s", WIKI_CACHE)
    logger.info("LLM verify cache   : %s", LLM_CACHE)
    logger.info("Output             : %s", out_path)
    logger.info("=" * 60)

    fetcher = EntityPassageFetcher(cache_db=WIKI_CACHE)
    vcache  = _VerifyCache(LLM_CACHE)

    mined:  list[dict] = []   # successfully mined hard cases (without irr. passages)
    gold_pool: list[str] = [] # gold passages for cross-pollinating irrelevant passages

    stats = {
        "processed":      0,
        "fetch_error":    0,
        "gold_fail":      0,
        "no_valid_trap":  0,
        "mined":          0,
        "trap_types":     {"entity_alias": 0, "topic_overlap": 0, "unknown": 0},
    }

    for i, (rel, entry) in enumerate(entries):
        if i > 0 and i % 50 == 0:
            logger.info(
                "Progress %d/%d | mined=%d  fetch_err=%d  gold_fail=%d  no_trap=%d",
                i, len(entries),
                stats["mined"], stats["fetch_error"],
                stats["gold_fail"], stats["no_valid_trap"],
            )

        stats["processed"] += 1
        question = entry["question"]
        answers  = entry["answers"]

        # ── Step 1: fetch Wikipedia passages ────────────────────────────
        result = fetcher.fetch(entry, rel)
        if result.error or not result.gold_passage:
            stats["fetch_error"] += 1
            logger.debug("Fetch error [%s] %r: %s", rel, question[:60], result.error)
            continue

        # ── Step 2: verify gold passage contains the answer ─────────────
        gold_v = _verify(question, answers, result.gold_passage, vcache)
        if not gold_v.get("contains_answer") or gold_v.get("confidence", 0) < 0.6:
            stats["gold_fail"] += 1
            logger.debug("Gold fail [%s] %r: %s", rel, question[:60], gold_v.get("reasoning", ""))
            continue

        # ── Step 3: find best verified trap ─────────────────────────────
        best_trap_title: str | None     = None
        best_trap_passage: str | None   = None
        best_trap_type: str             = "unknown"
        best_trap_score: float          = -1.0   # higher = more similar = harder trap

        for tc in result.trap_candidates:
            if not tc.passage:
                continue
            trap_v = _verify(question, answers, tc.passage, vcache)
            if trap_v.get("contains_answer") or trap_v.get("confidence", 0) < 0.5:
                # Trap candidate actually contains the answer → bad trap, skip
                continue

            # Score by entity-name similarity (more similar = harder trap)
            et = _tokens(result.entity)
            tt = _tokens(tc.title)
            union = et | tt
            score = len(et & tt) / len(union) if union else 0.0

            if score > best_trap_score:
                best_trap_score   = score
                best_trap_title   = tc.title
                best_trap_passage = tc.passage
                best_trap_type    = classify_trap_type(result.entity, tc.title)

        if best_trap_title is None:
            stats["no_valid_trap"] += 1
            logger.debug("No valid trap [%s] %r", rel, question[:60])
            continue

        # ── Step 4: record the mined case ───────────────────────────────
        stats["mined"] += 1
        stats["trap_types"][best_trap_type] += 1
        gold_pool.append(result.gold_passage)

        mined.append({
            "question_id":       _qid(question, rel),
            "question":          question,
            "answers":           answers,
            "gold_passage":      result.gold_passage,
            "gold_passage_title": result.gold_title,
            "trap_passage":      best_trap_passage,
            "trap_passage_title": best_trap_title,
            "trap_score":        round(best_trap_score, 4),
            "trap_type":         best_trap_type,
            "irrelevant_passages": [],   # filled below
            "source":            "entity_questions",
            "relation":          rel,
            "llm_calls":         vcache.api_calls,  # cumulative; individual count in post
        })

    # ── Step 5: assign irrelevant passages from cross-pool ───────────────
    # Build pool excluding the case's own gold passage.
    rng = random.Random(seed + 1)
    for case in mined:
        pool = [p for p in gold_pool if p != case["gold_passage"]]
        if len(pool) < 2:
            # Fallback: duplicate with slight offset if pool is tiny (dry-run edge case)
            pool = pool + pool
        irr = rng.sample(pool, min(2, len(pool)))
        case["irrelevant_passages"] = irr

    # ── Write output ─────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w"  # always overwrite — re-runs are idempotent
    with open(out_path, mode) as f:
        for case in mined:
            f.write(json.dumps(case) + "\n")

    fetcher.close()
    vcache.close()

    # ── Final report ──────────────────────────────────────────────────────
    logger.info("")
    logger.info("─" * 60)
    logger.info("RESULTS%s", " [DRY RUN]" if dry_run else "")
    logger.info("  Processed       : %d", stats["processed"])
    logger.info("  Fetch errors    : %d  (%.0f%%)", stats["fetch_error"],
                100 * stats["fetch_error"] / max(1, stats["processed"]))
    logger.info("  Gold failed     : %d  (%.0f%%)", stats["gold_fail"],
                100 * stats["gold_fail"] / max(1, stats["processed"]))
    logger.info("  No valid trap   : %d  (%.0f%%)", stats["no_valid_trap"],
                100 * stats["no_valid_trap"] / max(1, stats["processed"]))
    logger.info("  ─────────────────────────────────")
    logger.info("  Mined           : %d  (%.0f%% yield)", stats["mined"],
                100 * stats["mined"] / max(1, stats["processed"]))
    logger.info("  Trap types      : entity_alias=%d  topic_overlap=%d  unknown=%d",
                stats["trap_types"]["entity_alias"],
                stats["trap_types"]["topic_overlap"],
                stats["trap_types"]["unknown"])
    logger.info("  LLM API calls   : %d  (est. $%.4f)",
                vcache.api_calls, vcache.cost_usd)
    logger.info("  Wikipedia calls : %d", fetcher.api_calls)
    logger.info("  Output          : %s  (%d lines)", out_path, stats["mined"])
    logger.info("─" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Mine hard cases from EntityQuestions.")
    parser.add_argument(
        "--dry-run", dest="dry_run", action="store_true", default=True,
        help="Process only 20 entries (default).",
    )
    parser.add_argument(
        "--no-dry-run", dest="dry_run", action="store_false",
        help="Run full pipeline (--sample N entries).",
    )
    parser.add_argument(
        "--sample", "--sample-size", dest="sample", type=int, default=2000,
        help="Number of entries to sample when not in dry-run mode.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    mine(sample_size=args.sample, dry_run=args.dry_run, seed=args.seed)


if __name__ == "__main__":
    main()
