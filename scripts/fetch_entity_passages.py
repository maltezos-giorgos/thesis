"""Wikipedia passage fetcher for EntityQuestions entries.

Given an EntityQuestions entry (question + answers + relation ID):
  1. Extracts the subject entity [X] from the question using the Wikidata
     relation template (e.g. "Where was [X] born?" → regex to find [X]).
  2. Searches Wikipedia for the entity → first result is the GOLD PAGE.
  3. Uses the remaining search results as TRAP CANDIDATE pages.
  4. Returns the first ~500 chars of each page's intro paragraph.

All Wikipedia API calls (search + page fetch) are cached in SQLite so
re-running costs zero extra HTTP requests.

Disambiguation pages are handled gracefully: we catch DisambiguationError
and try the first suggested title instead. Pages that still fail are skipped.

Usage (as a module):
    from scripts.fetch_entity_passages import EntityPassageFetcher
    fetcher = EntityPassageFetcher()
    result = fetcher.fetch(entry, relation="P19")
    # result: FetchResult(entity, gold_title, gold_passage, trap_candidates)
    fetcher.close()

Usage (as a script — 5 demo examples):
    python scripts/fetch_entity_passages.py
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import wikipedia

warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")
wikipedia.set_lang("en")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from thesis_crag.utils.logging import get_logger

logger = get_logger("fetch_entity_passages")

REPO_ROOT      = Path(__file__).parent.parent
TEMPLATES_PATH = REPO_ROOT / "data/raw/entity_questions/relation_query_templates.json"
DEFAULT_CACHE  = str(REPO_ROOT / "data/cache/wikipedia_passages.db")

PASSAGE_CHARS     = 500   # chars to take from each page intro
N_SEARCH_RESULTS  = 6     # search results to fetch; [0]=gold, [1..]=trap candidates
MAX_TRAP_CANDS    = 4     # max trap candidates to return
WIKIPEDIA_DELAY_S = 0.2   # polite delay between uncached API calls


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrapCandidate:
    title:   str
    passage: str


@dataclass
class FetchResult:
    entity:          str
    relation:        Optional[str]
    gold_title:      Optional[str]   # None if Wikipedia search returned nothing
    gold_passage:    Optional[str]
    trap_candidates: list[TrapCandidate] = field(default_factory=list)
    error:           Optional[str] = None   # set if gold fetch failed completely


# ---------------------------------------------------------------------------
# Template → regex
# ---------------------------------------------------------------------------

def _load_templates() -> dict[str, re.Pattern]:
    """Load Wikidata relation templates and compile each to a capture regex."""
    raw = json.loads(TEMPLATES_PATH.read_text())
    patterns: dict[str, re.Pattern] = {}
    for pid, tmpl in raw.items():
        parts = tmpl.split("[X]")
        if len(parts) != 2:
            continue
        # Escape literal text on both sides; put a capture group where [X] was
        pattern = re.escape(parts[0]) + r"(.+?)" + re.escape(parts[1])
        # Anchor end with optional punctuation
        patterns[pid] = re.compile(pattern.rstrip(r"\?") + r"\??$", re.IGNORECASE)
    return patterns


_TEMPLATES: dict[str, re.Pattern] = {}


def _get_templates() -> dict[str, re.Pattern]:
    global _TEMPLATES
    if not _TEMPLATES:
        _TEMPLATES = _load_templates()
    return _TEMPLATES


def extract_entity(question: str, relation: str) -> Optional[str]:
    """Extract the [X] subject entity from a question given its relation ID.

    Returns None if the template doesn't match (shouldn't happen for valid
    EntityQuestions entries).
    """
    pat = _get_templates().get(relation)
    if pat is None:
        return None
    m = pat.search(question)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS wikipedia_search (
    query      TEXT PRIMARY KEY,
    titles_json TEXT NOT NULL,
    fetched_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS wikipedia_page (
    title          TEXT PRIMARY KEY,
    passage        TEXT,          -- first PASSAGE_CHARS of intro, NULL=disambiguation/error
    is_disambiguation INTEGER NOT NULL DEFAULT 0,
    error          TEXT,          -- set if PageError or other failure
    fetched_at     REAL NOT NULL
);
"""


class WikipediaCache:
    """SQLite-backed cache for Wikipedia search and page fetches."""

    def __init__(self, db_path: str = DEFAULT_CACHE) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._api_calls = 0

    # ---- search ----

    def get_search(self, query: str) -> Optional[list[str]]:
        row = self._conn.execute(
            "SELECT titles_json FROM wikipedia_search WHERE query=?", (query,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put_search(self, query: str, titles: list[str]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO wikipedia_search VALUES (?,?,?)",
            (query, json.dumps(titles), time.time()),
        )
        self._conn.commit()

    # ---- page ----

    def get_page(self, title: str) -> Optional[dict]:
        """Returns dict with keys: passage, is_disambiguation, error. Or None if not cached."""
        row = self._conn.execute(
            "SELECT passage, is_disambiguation, error FROM wikipedia_page WHERE title=?",
            (title,),
        ).fetchone()
        if row is None:
            return None
        return {"passage": row[0], "is_disambiguation": bool(row[1]), "error": row[2]}

    def put_page(
        self,
        title: str,
        passage: Optional[str],
        is_disambiguation: bool = False,
        error: Optional[str] = None,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO wikipedia_page VALUES (?,?,?,?,?)",
            (title, passage, int(is_disambiguation), error, time.time()),
        )
        self._conn.commit()

    @property
    def api_calls(self) -> int:
        return self._api_calls

    def inc_api(self) -> None:
        self._api_calls += 1

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Wikipedia helpers
# ---------------------------------------------------------------------------

def _search_cached(query: str, cache: WikipediaCache, n: int = N_SEARCH_RESULTS) -> list[str]:
    """Return Wikipedia search results (list of page titles), using cache."""
    cached = cache.get_search(query)
    if cached is not None:
        return cached
    # Retry once on empty result — Wikipedia search is occasionally flaky
    for attempt in range(2):
        time.sleep(WIKIPEDIA_DELAY_S * (attempt + 1))
        cache.inc_api()
        try:
            titles = wikipedia.search(query, results=n)
        except Exception as exc:
            logger.debug("Search error for %r (attempt %d): %s", query, attempt, exc)
            titles = []
        if titles:
            cache.put_search(query, titles)
            return titles
    return []


def _fetch_page_cached(title: str, cache: WikipediaCache) -> Optional[str]:
    """Return first PASSAGE_CHARS of the page intro, or None if unavailable.

    Handles DisambiguationError by trying the first suggested alternative.
    Stores result (including errors) in the cache.
    """
    cached = cache.get_page(title)
    if cached is not None:
        if cached["is_disambiguation"] or cached["error"]:
            return None
        return cached["passage"]

    time.sleep(WIKIPEDIA_DELAY_S)
    cache.inc_api()
    try:
        page = wikipedia.page(title, auto_suggest=False)
        passage = (page.summary or page.content)[:PASSAGE_CHARS].strip()
        cache.put_page(title, passage)
        return passage

    except wikipedia.exceptions.DisambiguationError as exc:
        cache.put_page(title, None, is_disambiguation=True)
        # Try the first non-disambiguation suggestion
        alts = exc.options[:3]
        for alt in alts:
            alt_cached = cache.get_page(alt)
            if alt_cached is not None:
                if alt_cached["passage"]:
                    return alt_cached["passage"]
                continue
            time.sleep(WIKIPEDIA_DELAY_S)
            cache.inc_api()
            try:
                page = wikipedia.page(alt, auto_suggest=False)
                passage = (page.summary or page.content)[:PASSAGE_CHARS].strip()
                cache.put_page(alt, passage)
                return passage
            except Exception:
                cache.put_page(alt, None, error="nested_error")
        return None

    except wikipedia.exceptions.PageError:
        cache.put_page(title, None, error="page_not_found")
        return None

    except Exception as exc:
        msg = str(exc)[:120]
        cache.put_page(title, None, error=msg)
        logger.debug("Page error for %r: %s", title, msg)
        return None


# ---------------------------------------------------------------------------
# Main fetcher class
# ---------------------------------------------------------------------------

class EntityPassageFetcher:
    """Fetch gold + trap passage candidates for an EntityQuestions entry."""

    def __init__(self, cache_db: str = DEFAULT_CACHE) -> None:
        self._cache = WikipediaCache(cache_db)

    def fetch(self, entry: dict, relation: str) -> FetchResult:
        """Fetch passages for one EntityQuestions entry.

        Args:
            entry:    dict with at least 'question' and 'answers' keys.
            relation: Wikidata property ID (e.g. 'P19').

        Returns:
            FetchResult with entity, gold_title/passage, trap_candidates.
        """
        question = entry["question"]

        # 1. Extract the subject entity
        entity = extract_entity(question, relation)
        if entity is None:
            return FetchResult(
                entity=question,
                relation=relation,
                gold_title=None,
                gold_passage=None,
                error=f"entity_extraction_failed for relation {relation}",
            )

        # 2. Search Wikipedia for the entity
        titles = _search_cached(entity, self._cache)
        if not titles:
            return FetchResult(
                entity=entity,
                relation=relation,
                gold_title=None,
                gold_passage=None,
                error="no_search_results",
            )

        # 3. Fetch the gold page (first search result)
        gold_title = titles[0]
        gold_passage = _fetch_page_cached(gold_title, self._cache)

        if gold_passage is None:
            # Gold page failed; try the second result
            if len(titles) > 1:
                gold_title = titles[1]
                gold_passage = _fetch_page_cached(gold_title, self._cache)
                trap_titles = titles[2:]
            else:
                trap_titles = []

            if gold_passage is None:
                return FetchResult(
                    entity=entity,
                    relation=relation,
                    gold_title=gold_title,
                    gold_passage=None,
                    error="gold_page_unavailable",
                )
        else:
            trap_titles = titles[1:]

        # 4. Fetch trap candidates from remaining search results
        trap_candidates: list[TrapCandidate] = []
        for t in trap_titles:
            if t == gold_title:
                continue
            passage = _fetch_page_cached(t, self._cache)
            if passage:
                trap_candidates.append(TrapCandidate(title=t, passage=passage))
            if len(trap_candidates) >= MAX_TRAP_CANDS:
                break

        return FetchResult(
            entity=entity,
            relation=relation,
            gold_title=gold_title,
            gold_passage=gold_passage,
            trap_candidates=trap_candidates,
        )

    @property
    def api_calls(self) -> int:
        return self._cache.api_calls

    def close(self) -> None:
        self._cache.close()


# ---------------------------------------------------------------------------
# Demo: test on 5 EntityQuestions entries across different relations
# ---------------------------------------------------------------------------

def _demo() -> None:
    import random

    random.seed(42)
    eq_base = REPO_ROOT / "data/raw/entity_questions/dataset/test"

    # Pick 5 entries from 5 different relations
    demo_entries = []
    for rel in ["P19", "P50", "P106", "P170", "P175"]:
        path = eq_base / f"{rel}.test.json"
        data = json.loads(path.read_text())
        entry = random.choice(data[:100])
        demo_entries.append((rel, entry))

    fetcher = EntityPassageFetcher()
    print("\n" + "=" * 70)
    print("DEMO: fetch_entity_passages.py — 5 test entries")
    print("=" * 70)

    for rel, entry in demo_entries:
        entity = extract_entity(entry["question"], rel)
        result = fetcher.fetch(entry, rel)

        print(f"\n{'─'*70}")
        print(f"Relation : {rel}")
        print(f"Question : {entry['question']}")
        print(f"Answers  : {entry['answers']}")
        print(f"Entity   : {entity!r}")
        if result.error:
            print(f"ERROR    : {result.error}")
            continue
        print(f"\nGold page: {result.gold_title!r}")
        print(f"  {result.gold_passage[:200]}...")
        print(f"\nTrap candidates ({len(result.trap_candidates)}):")
        for tc in result.trap_candidates[:2]:
            print(f"  [{tc.title}] {tc.passage[:120]}...")

    print(f"\n{'='*70}")
    print(f"Wikipedia API calls made: {fetcher.api_calls}")
    fetcher.close()


if __name__ == "__main__":
    _demo()
