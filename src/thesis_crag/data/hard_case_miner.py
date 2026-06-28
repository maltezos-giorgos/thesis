"""Phase 2: mine hard cases (gold, trap, irrelevant) from PopQA long-tail.

A "hard case" is a triple where:
  - gold_passage:  LLM-verified to contain the answer
  - trap_passage:  highest-Contriever-similarity passage that does NOT contain the answer
  - irrelevant_passages: remaining non-answer passages (up to 3)

The trap passage is the type of false positive the T5 baseline emits — a passage
that is semantically close to the question (entity name overlap) but answers a
different fact about a different entity.

LLM calls are cached in an SQLite database to avoid redundant API hits across runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, Field

from thesis_crag.utils.llm_clients import call_llm_with_validation

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts/answer_containment.txt"
_SYSTEM_PROMPT: str | None = None


def _get_system_prompt() -> str:
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        _SYSTEM_PROMPT = _PROMPT_PATH.read_text().strip()
    return _SYSTEM_PROMPT


class HardCase(BaseModel):
    question_id: str
    question: str
    answers: list[str]
    gold_passage: str
    gold_passage_title: str
    trap_passage: str
    trap_passage_title: str
    trap_score: float  # Contriever similarity of trap passage
    trap_type: str     # heuristic label: entity_alias | topic_overlap | unknown
    irrelevant_passages: list[str] = Field(default_factory=list)
    source: str
    llm_calls: int = 0  # count of LLM calls made for this example


class LLMCache:
    """Disk-backed SQLite cache for LLM containment calls.

    Key: SHA-256 of (system_prompt + user_message). Value: JSON result dict.
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, result TEXT)"
        )
        self._conn.commit()

    def _key(self, system: str, user: str) -> str:
        return hashlib.sha256((system + "\n" + user).encode()).hexdigest()

    def get(self, system: str, user: str) -> dict | None:
        key = self._key(system, user)
        row = self._conn.execute("SELECT result FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, system: str, user: str, result: dict) -> None:
        key = self._key(system, user)
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, result) VALUES (?, ?)",
            (key, json.dumps(result)),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def _classify_trap_type(gold_title: str, trap_title: str, question: str) -> str:
    """Heuristic trap-type label based on title similarity."""
    gold_tokens = set(gold_title.lower().split())
    trap_tokens = set(trap_title.lower().split())
    overlap = gold_tokens & trap_tokens
    # Remove generic stop words before measuring overlap
    stops = {"the", "a", "an", "of", "and", "or", "in", "at", "by", "for"}
    meaningful_overlap = overlap - stops
    if len(meaningful_overlap) >= 2:
        return "entity_alias"
    if len(meaningful_overlap) == 1:
        return "topic_overlap"
    return "unknown"


class HardCaseMiner:
    """Mine hard-case triples from PopQA long-tail retrieved passages.

    Uses Claude Haiku to determine whether each passage contains the answer.
    All LLM calls are cached on disk so interrupted runs can be resumed cheaply.
    """

    MAX_IRRELEVANT = 3  # max irrelevant passages to store per hard case

    def __init__(self, cache_path: Path) -> None:
        self._cache = LLMCache(cache_path)
        self._llm_calls_total = 0

    def _check_containment(self, question: str, answers: list[str], passage: str) -> bool:
        """Return True if the passage contains one of the acceptable answers.

        The underlying API call (call_llm_with_validation) retries network/rate-limit
        errors with exponential backoff (up to 8 attempts). If it still fails, the
        exception is caught here and the passage is treated as non-containing
        (returns False) so mining can proceed; the result is cached on success only.
        """
        system = _get_system_prompt()
        answer_str = ", ".join(answers[:10])  # cap to avoid excessively long prompts
        user = (
            f"Question: {question}\n"
            f"Acceptable answers: {answer_str}\n"
            f"Passage: {passage[:800]}"  # truncate very long passages
        )
        result = self._cache.get(system, user)
        if result is None:
            try:
                result = call_llm_with_validation(system, user)
            except Exception:
                # Log with full traceback; return False to skip this passage
                # (caller treats False as "does not contain answer").
                logger.warning(
                    "LLM call failed for question %r — treating as non-containing",
                    question[:60],
                    exc_info=True,
                )
                return False
            self._cache.put(system, user, result)
            self._llm_calls_total += 1
        else:
            logger.debug("Cache hit for question %r", question[:60])
        return bool(result.get("contains_answer", False))

    def mine_from_popqa_longtail(
        self,
        items: list[dict],
        limit: int | None = None,
    ) -> Iterator[HardCase]:
        """Yield HardCase triples mined from PopQA long-tail examples.

        items: list of raw dicts from the popqa_longtail_w_gs.jsonl file.
        limit: stop after yielding this many hard cases (None = no limit).
        """
        yielded = 0
        for q_processed, item in enumerate(items):
            if limit is not None and yielded >= limit:
                break

            if q_processed > 0 and q_processed % 200 == 0:
                cost_est = self._llm_calls_total * (200 * 0.80 + 80 * 4.0) / 1_000_000
                logger.info(
                    "Progress: %d/%d questions | %d hard cases | %d LLM calls | est. $%.3f",
                    q_processed, len(items), yielded, self._llm_calls_total, cost_est,
                )

            qid = str(item["id"])
            question = item["question"]
            answers = item["answers"]
            s_wiki_title = item.get("s_wiki_title", "")
            ctxs = item.get("ctxs", [])

            # Separate gold candidates (title match) from trap candidates
            gold_ctxs = [c for c in ctxs if c.get("title", "").strip() == s_wiki_title.strip()]
            trap_ctxs = [c for c in ctxs if c.get("title", "").strip() != s_wiki_title.strip()
                         and c.get("score") is not None]

            if not gold_ctxs or not trap_ctxs:
                logger.debug("Q%s: skipped (no gold or no scored trap candidates)", qid)
                continue

            # Pick highest-scored gold passage and verify it contains the answer
            gold_sorted = sorted(gold_ctxs, key=lambda c: float(c.get("score", 0)), reverse=True)
            gold = gold_sorted[0]
            llm_calls_this = 0

            if not self._check_containment(question, answers, gold["text"]):
                llm_calls_this += 1
                logger.debug("Q%s: gold passage does not contain answer — skipping", qid)
                continue
            llm_calls_this += 1

            # Sort trap candidates by Contriever score descending; find first non-answer passage
            trap_sorted = sorted(trap_ctxs, key=lambda c: float(c.get("score", 0)), reverse=True)
            trap = None
            for cand in trap_sorted:
                if not self._check_containment(question, answers, cand["text"]):
                    trap = cand
                    llm_calls_this += 1
                    break
                llm_calls_this += 1

            if trap is None:
                logger.debug("Q%s: all trap candidates contain answer — skipping", qid)
                continue

            # Remaining non-answer passages (up to MAX_IRRELEVANT), skip the chosen trap
            irrelevant = [
                c["text"] for c in trap_sorted
                if c is not trap
            ][: self.MAX_IRRELEVANT]

            hard_case = HardCase(
                question_id=qid,
                question=question,
                answers=answers,
                gold_passage=gold["text"],
                gold_passage_title=gold.get("title", ""),
                trap_passage=trap["text"],
                trap_passage_title=trap.get("title", ""),
                trap_score=float(trap.get("score", 0)),
                trap_type=_classify_trap_type(
                    gold.get("title", ""), trap.get("title", ""), question
                ),
                irrelevant_passages=irrelevant,
                source="popqa_longtail",
                llm_calls=llm_calls_this,
            )
            logger.info(
                "Hard case Q%s | trap_type=%s | trap_score=%.3f | llm_calls=%d",
                qid, hard_case.trap_type, hard_case.trap_score, llm_calls_this,
            )
            yield hard_case
            yielded += 1

    @property
    def total_llm_calls(self) -> int:
        return self._llm_calls_total

    def close(self) -> None:
        self._cache.close()
