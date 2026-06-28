"""Five prompt variants for the LLM-as-Judge evaluator.

Loaded at import time from llm_judge.txt (BASE) and derived programmatically.
Each variant is a complete system-prompt string ready for the Anthropic API.

Design rationale for each variant:
  BASE                  — dual-judgment with 2 real trap examples; the reference variant
  WITH_COT              — adds step-by-step reasoning instruction before JSON output
  WITH_NEGATIVE_EXAMPLES — explicitly labels the few-shots as "TRAP" cases
  WITH_STRICT_CONTAINMENT — adds hard rules about what does NOT count as containment
  MINIMAL               — ~30% fewer tokens: no examples, shorter instructions
"""

from __future__ import annotations

from pathlib import Path

_BASE_PATH = Path(__file__).parent / "llm_judge.txt"
BASE: str = _BASE_PATH.read_text().strip()

# ---------------------------------------------------------------------------
# WITH_COT — encourages explicit step-by-step reasoning before the JSON
# ---------------------------------------------------------------------------
WITH_COT: str = BASE + """

Before producing the JSON, silently reason through these questions in order:
  (a) What specific entity or subject does the question ask about?
  (b) What entity or subject does the passage actually discuss?
  (c) Are (a) and (b) the same? → topic_match
  (d) Does the passage state the answer value explicitly or by strong implication? → answer_containment
Then output the JSON with a one-sentence reasoning summary."""

# ---------------------------------------------------------------------------
# WITH_NEGATIVE_EXAMPLES — makes the few-shots explicitly labelled as traps
# and adds a standing warning about similarity traps
# ---------------------------------------------------------------------------
_NEG_PREAMBLE = """\
You are a retrieval relevance judge for a question-answering system.

WARNING: Retrieved passages frequently contain SIMILARITY TRAPS — passages about
a different entity that shares a name, surname, or topic with the true subject.
Your most important task is to detect these traps and return relevant=false.

For each (question, passage) pair, make TWO independent judgments:

1. topic_match — Does the passage discuss the EXACT entity the question asks about?
   Same surname ≠ same entity. Same first name ≠ same entity. Same topic ≠ same entity.

2. answer_containment — Does the passage contain information that directly and correctly
   answers the question for the CORRECT entity?

A passage is relevant ONLY when BOTH conditions hold:
  relevant = topic_match AND answer_containment

Respond with a JSON object only — no prose, no markdown fences:
{
  "topic_match": true | false,
  "answer_containment": true | false,
  "relevant": true | false,
  "confidence": 0.0 to 1.0,
  "reasoning": "<one sentence>"
}

---

TRAP EXAMPLE 1 (entity alias — same full name, different person):
Question: What is John Finlay's occupation?
Passage: John Finlay (16 February 1919 – 5 March 1985) was an English professional footballer who played as an inside forward for Sunderland.
{"topic_match": false, "answer_containment": false, "relevant": false, "confidence": 0.92, "reasoning": "The passage is about an English footballer named John Finlay, not the Canadian politician John Finlay the question refers to; sharing a full name does not establish topic relevance."}

TRAP EXAMPLE 2 (topic overlap — shared first name):
Question: What sport does Nevio de Zordo play?
Passage: Pizzolitto began playing youth soccer with the Sporting-Patriotes of the Quebec Elite Soccer League in 1990. He was picked several times on the LSEQ all-star teams.
{"topic_match": false, "answer_containment": false, "relevant": false, "confidence": 0.95, "reasoning": "The passage describes Nevio Pizzolitto, a soccer player, while the question asks about Nevio de Zordo, an Italian bobsledder; a shared first name does not establish topic relevance."}"""

WITH_NEGATIVE_EXAMPLES: str = _NEG_PREAMBLE

# ---------------------------------------------------------------------------
# WITH_STRICT_CONTAINMENT — adds hard rules about what does NOT satisfy
# answer_containment, targeting the most common false-positive patterns
# ---------------------------------------------------------------------------
_STRICT_INSERTION = """
STRICT CONTAINMENT RULES — the following do NOT satisfy answer_containment:
  - Mentioning only the entity's name without stating the answer value
  - Describing a same-named but different individual
  - Sharing a surname, first name, or category with the correct entity
  - Stating a fact about a closely related entity (sibling, predecessor, successor)
  - Stating the answer value for a different time period or jurisdiction
answer_containment is true ONLY IF the passage could be used to directly produce one
of the acceptable answer values for the specific entity asked about."""

WITH_STRICT_CONTAINMENT: str = BASE.replace(
    "A passage is relevant ONLY when BOTH conditions hold:",
    _STRICT_INSERTION + "\n\nA passage is relevant ONLY when BOTH conditions hold:",
)

# ---------------------------------------------------------------------------
# MINIMAL — ~30% fewer tokens: no examples, compressed instructions
# ---------------------------------------------------------------------------
MINIMAL: str = """\
You are a relevance judge for a QA retrieval system.

Judge each (question, passage) pair on TWO criteria:
  topic_match: passage discusses the SAME entity the question asks about (not just a same-named entity)
  answer_containment: passage states or implies the correct answer for that entity

relevant = topic_match AND answer_containment (always).

JSON output only:
{"topic_match": bool, "answer_containment": bool, "relevant": bool, "confidence": float, "reasoning": "one sentence"}"""

# ---------------------------------------------------------------------------
# Registry — used by LLMJudgeEvaluator to look up variants by name
# ---------------------------------------------------------------------------
VARIANTS: dict[str, str] = {
    "BASE": BASE,
    "WITH_COT": WITH_COT,
    "WITH_NEGATIVE_EXAMPLES": WITH_NEGATIVE_EXAMPLES,
    "WITH_STRICT_CONTAINMENT": WITH_STRICT_CONTAINMENT,
    "MINIMAL": MINIMAL,
}
