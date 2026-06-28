"""Data loaders for Phase 2 hard-case mining.

Reusable public loader API: import these to read each dataset into uniform
QAExample instances (kept as a stable convenience layer even though the final
pipeline scripts read their JSONL inputs directly).

All loaders return QAExample instances. The primary source for mining is
load_popqa_longtail() since it includes pre-retrieved passages from Contriever.
The other loaders return QA pairs without passages (for future retrieval).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).parent.parent.parent.parent  # src/thesis_crag/data -> thesis root


class QAExample(BaseModel):
    id: str
    question: str
    answers: list[str]
    source: str
    # Only present for popqa_longtail — 10 Contriever-retrieved passages per question
    ctxs: list[dict] = Field(default_factory=list)
    # Wikidata subject title (popqa sources only)
    s_wiki_title: str = ""
    metadata: dict = Field(default_factory=dict)


def load_popqa_longtail(path: Path | None = None) -> Iterator[QAExample]:
    """Load the 1,399-question PopQA long-tail split used in Phase 1.

    Each example includes up to 10 Contriever-retrieved passages in ctxs.
    """
    if path is None:
        path = REPO_ROOT / "external/CRAG/eval_data/popqa_longtail_w_gs.jsonl"
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            yield QAExample(
                id=str(item["id"]),
                question=item["question"],
                answers=item["answers"],
                source="popqa_longtail",
                ctxs=item.get("ctxs", []),
                s_wiki_title=item.get("s_wiki_title", ""),
                metadata={"prop": item.get("prop", ""), "pop": item.get("pop", 0)},
            )


def load_entity_questions(data_dir: Path | None = None) -> Iterator[QAExample]:
    """Load EntityQuestions test set (all 24 relation types).

    Each example has a question and one or more gold answers.
    No retrieved passages — these require separate retrieval for mining.
    """
    if data_dir is None:
        data_dir = REPO_ROOT / "data/raw/entity_questions/dataset/test"
    for json_file in sorted(data_dir.glob("*.test.json")):
        relation = json_file.stem.replace(".test", "")
        with open(json_file) as f:
            items = json.load(f)
        for i, item in enumerate(items):
            yield QAExample(
                id=f"eq_{relation}_{i}",
                question=item["question"],
                answers=item["answers"],
                source="entity_questions",
                metadata={"relation": relation},
            )


def load_popqa_dev(path: Path | None = None) -> Iterator[QAExample]:
    """Load the PopQA HuggingFace test split (14,267 examples).

    No retrieved passages. 'possible_answers' is a JSON-encoded string list.
    """
    if path is None:
        path = REPO_ROOT / "data/raw/popqa/dev.jsonl"
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            possible_answers_raw = item.get("possible_answers", "[]")
            if isinstance(possible_answers_raw, str):
                answers = json.loads(possible_answers_raw)
            else:
                answers = possible_answers_raw
            yield QAExample(
                id=str(item["id"]),
                question=item["question"],
                answers=answers,
                source="popqa_dev",
                s_wiki_title=item.get("s_wiki_title", ""),
                metadata={
                    "prop": item.get("prop", ""),
                    "subj": item.get("subj", ""),
                    "obj": item.get("obj", ""),
                },
            )


def load_beir_nq_queries(path: Path | None = None) -> Iterator[QAExample]:
    """Load 2,000 BEIR-NQ queries. Note: no answer strings, only questions.

    Suitable for retrieval experiments but not directly for answer-containment
    checking. Included for completeness and future retrieval benchmarking.
    """
    if path is None:
        path = REPO_ROOT / "data/raw/beir_nq/queries.jsonl"
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            yield QAExample(
                id=item.get("_id", str(i)),
                question=item.get("text", item.get("question", "")),
                answers=[],  # BEIR queries don't include answer strings
                source="beir_nq",
                metadata=item.get("metadata", {}),
            )
