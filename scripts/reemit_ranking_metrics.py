"""Re-emit Phase 5 ranking metrics with two audit corrections (no model/API calls).

This script re-aggregates the EXISTING per-case score files in results/phase5/
(t5_eval.jsonl, llm_eval.jsonl, ce_v2_eval.jsonl, hybrid_v2_eval.jsonl). It does
NOT run any evaluator and makes no API calls — it only re-derives the ranking
metrics (Precision@1, Recall@2, MRR) under two corrections identified by the
forensic audit:

C1 — Expected-rank tie handling
-------------------------------
The original eval_phase5.py computed   rank = 1 + sum(s > gold_score)   which
silently breaks score ties in gold's favour. The LLM judge emits quantised
confidences, so exact ties between gold and a distractor are common; T5/CE emit
continuous scores and (almost) never tie, so the original method inflates the
LLM/Hybrid ranking metrics relative to T5/CE.

We replace it with the EXPECTED VALUE of each metric under uniformly-random
tie-breaking. For a case, let
    g = number of distractors scoring strictly GREATER than gold
    t = number of distractors scoring EXACTLY EQUAL to gold
Gold then occupies a uniformly-random position in the tied block of size (t+1)
that sits at integer ranks [g+1, g+2, ..., g+t+1]. The per-case contributions
are the expectations over that uniform placement:
    P@1 = P(rank == 1)        = (# block ranks equal to 1) / (t+1)
    R@2 = P(rank <= 2)        = (# block ranks <= 2)        / (t+1)
    MRR = E[1/rank]           = mean(1/r for r in block ranks)
With no ties (t=0) these reduce to the usual indicators on the single rank g+1.
Worked example (one tie at the top, g=0, t=1 → ranks {1,2}):
    P@1 = 1/2 = 0.5,  R@2 = 2/2 = 1.0,  MRR = (1/1 + 1/2)/2 = 0.75.
For reference we also report the fractional E[rank] = 1 + g + t/2.

C2 — Contamination exclusion
----------------------------
The hard-case miner sometimes placed passages it had itself LLM-verified to
CONTAIN the answer into the "irrelevant_passages" slots (a bug in candidate
selection: candidates checked BEFORE the chosen trap were answer-containing but
still kept as distractors). We replay the miner's SQLite verification cache
(data/hard_cases/llm_cache.db) to find these specific distractors and drop them
from the candidate set of their case before ranking (the case itself stays; only
the contaminated distractor is removed). Only PopQA cases can be affected — the
EntityQuestions miner used a different, clean pipeline.

Output (both written to results/phase5/):
    ranking_metrics_corrected.json   — machine-readable, all 4 evaluators
    ranking_metrics_corrected.md     — side-by-side table for the thesis appendix

Usage:
    python scripts/reemit_ranking_metrics.py
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

TEST_PATH    = REPO_ROOT / "data/hard_cases/splits_v2/test.jsonl"
MINING_CACHE = REPO_ROOT / "data/hard_cases/llm_cache.db"
PROMPT_PATH  = REPO_ROOT / "src/thesis_crag/prompts/answer_containment.txt"
RESULTS_DIR  = REPO_ROOT / "results/phase5"
COMPARISON_JSON = RESULTS_DIR / "comparison.json"

# Per-evaluator score files (same keys/labels as eval_phase5.py)
EVAL_FILES = {
    "T5":     RESULTS_DIR / "t5_eval.jsonl",
    "LLM":    RESULTS_DIR / "llm_eval.jsonl",
    "CE":     RESULTS_DIR / "ce_v2_eval.jsonl",
    "Hybrid": RESULTS_DIR / "hybrid_v2_eval.jsonl",
}
EVAL_ORDER = ["T5", "LLM", "CE", "Hybrid"]
COL_LABELS = {"T5": "T5", "LLM": "LLM Judge", "CE": "CE v2", "Hybrid": "Hybrid v2"}

# Bootstrap convention — identical to eval_phase5.py
N_BOOT    = 1000
BOOT_SEED = 0

# The two distractor roles that can be contaminated (gold/trap are never dropped)
IRR_ROLES = ("irr1", "irr2")


# ──────────────────────────────────────────────────────────────────────────────
# Contamination mask (C2): replay the miner's verification cache
# ──────────────────────────────────────────────────────────────────────────────

def load_contamination_mask() -> dict[str, set[str]]:
    """Map question_id -> subset of {"irr1","irr2"} that is LLM-verified answer-containing.

    Reconstructs the exact cache key the hard-case miner used
    (thesis_crag.data.hard_case_miner.HardCaseMiner._check_containment /
    LLMCache._key) and looks each irrelevant passage up in the mining cache.
    A cache hit with contains_answer=True marks that distractor as contaminated.
    Cache misses (passages the miner never verified) are left untouched.
    """
    system = PROMPT_PATH.read_text().strip()
    conn = sqlite3.connect(str(MINING_CACHE))

    def verdict(question: str, answers: list[str], passage: str) -> bool | None:
        answer_str = ", ".join(answers[:10])              # miner caps answers to 10
        user = (
            f"Question: {question}\n"
            f"Acceptable answers: {answer_str}\n"
            f"Passage: {passage[:800]}"                    # miner truncates to 800 chars
        )
        key = hashlib.sha256((system + "\n" + user).encode()).hexdigest()
        row = conn.execute("SELECT result FROM cache WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        return bool(json.loads(row[0]).get("contains_answer", False))

    mask: dict[str, set[str]] = {}
    with open(TEST_PATH) as f:
        for line in f:
            case = json.loads(line)
            if case.get("source") != "popqa":
                continue   # only PopQA-mined cases can carry this bug
            irr = case["irrelevant_passages"]
            contaminated: set[str] = set()
            for idx, role in enumerate(IRR_ROLES):
                if idx < len(irr) and verdict(case["question"], case["answers"], irr[idx]) is True:
                    contaminated.add(role)
            if contaminated:
                mask[str(case["question_id"])] = contaminated
    conn.close()
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Per-case ranking contributions
# ──────────────────────────────────────────────────────────────────────────────

def as_reported_contrib(rec: dict) -> tuple[float, float, float]:
    """Reproduce eval_phase5.py exactly: rank = 1 + #(distractor strictly > gold)."""
    gold = rec["gold_score"]
    distractors = [rec["trap_score"], rec["irr1_score"], rec["irr2_score"]]
    rank = 1 + sum(1 for s in distractors if s > gold)
    return float(rank == 1), float(rank <= 2), 1.0 / rank


def corrected_contrib(rec: dict, contaminated_roles: set[str]) -> tuple[float, float, float]:
    """Expected-rank tie handling (C1) + contamination exclusion (C2).

    Returns (P@1, R@2, MRR) expected-value contributions for one case.
    """
    gold = rec["gold_score"]
    distractors = [rec["trap_score"]]                      # trap is never excluded
    for role in IRR_ROLES:
        if role not in contaminated_roles:
            distractors.append(rec[f"{role}_score"])

    g = sum(1 for s in distractors if s > gold)            # strictly greater
    t = sum(1 for s in distractors if s == gold)           # exact ties with gold
    block_ranks = list(range(g + 1, g + t + 2))            # gold's possible ranks
    m = len(block_ranks)                                   # == t + 1
    p1  = sum(1 for r in block_ranks if r == 1) / m
    r2  = sum(1 for r in block_ranks if r <= 2) / m
    mrr = sum(1.0 / r for r in block_ranks) / m
    return p1, r2, mrr


def aggregate(contribs: list[tuple[float, float, float]]) -> dict[str, float]:
    n = len(contribs)
    return {
        "precision_at_1": sum(c[0] for c in contribs) / n,
        "recall_at_2":    sum(c[1] for c in contribs) / n,
        "mrr":            sum(c[2] for c in contribs) / n,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap (case-level, identical convention to eval_phase5.py)
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    contribs: list[tuple[float, float, float]],
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict[str, tuple[float, float, float]]:
    """Return {metric: (point, ci_lo, ci_hi)} with a 95% percentile CI."""
    rng = random.Random(seed)
    n = len(contribs)
    boot: list[dict[str, float]] = []
    for _ in range(n_boot):
        sample = [contribs[rng.randint(0, n - 1)] for _ in range(n)]
        boot.append(aggregate(sample))

    point = aggregate(contribs)
    out: dict[str, tuple[float, float, float]] = {}
    for key in point:
        vals = sorted(b[key] for b in boot)
        lo = vals[int(0.025 * n_boot)]
        hi = vals[int(0.975 * n_boot)]
        out[key] = (point[key], lo, hi)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

METRICS = [("precision_at_1", "Precision@1"), ("recall_at_2", "Recall@2"), ("mrr", "MRR")]


def _fmt_ci(point: float, lo: float, hi: float) -> str:
    return f"{point:.4f} [{lo:.4f}, {hi:.4f}]"


def main() -> None:
    mask = load_contamination_mask()
    n_contaminated_distractors = sum(len(v) for v in mask.values())
    print(
        f"Contamination mask: {n_contaminated_distractors} distractors across "
        f"{len(mask)} test cases flagged as LLM-verified answer-containing."
    )

    # Load the published as-reported CIs to validate our reproduction.
    published = json.loads(COMPARISON_JSON.read_text()) if COMPARISON_JSON.exists() else {}

    out: dict[str, dict] = {
        "meta": {
            "n_cases": None,
            "n_boot": N_BOOT,
            "boot_seed": BOOT_SEED,
            "contamination_distractors": n_contaminated_distractors,
            "contamination_cases": len(mask),
            "corrections": {
                "C1": "expected-value ranking metrics under uniform random tie-breaking",
                "C2": "drop LLM-verified answer-containing 'irrelevant' distractors before ranking",
            },
        },
        "evaluators": {},
    }

    repro_max_diff = 0.0

    for key in EVAL_ORDER:
        path = EVAL_FILES[key]
        records = [json.loads(line) for line in open(path)]
        out["meta"]["n_cases"] = len(records)

        rep_contribs = [as_reported_contrib(r) for r in records]
        cor_contribs = [
            corrected_contrib(r, mask.get(str(r["question_id"]), set())) for r in records
        ]

        rep_point = aggregate(rep_contribs)
        cor_ci = bootstrap_ci(cor_contribs)

        # Validate as-reported point estimates against the published comparison.json
        if key in published:
            pub_metrics = published[key]["metrics"]
            for m, _ in METRICS:
                repro_max_diff = max(
                    repro_max_diff, abs(round(rep_point[m], 4) - pub_metrics[m]["point"])
                )

        out["evaluators"][key] = {
            "label": COL_LABELS[key],
            "as_reported": {m: round(rep_point[m], 4) for m, _ in METRICS},
            "corrected": {
                m: {
                    "point": round(cor_ci[m][0], 4),
                    "ci_lo": round(cor_ci[m][1], 4),
                    "ci_hi": round(cor_ci[m][2], 4),
                }
                for m, _ in METRICS
            },
            "delta": {m: round(cor_ci[m][0] - rep_point[m], 4) for m, _ in METRICS},
        }

    print(
        f"Reproduction check: max |as-reported − comparison.json| over "
        f"P@1/R@2/MRR = {repro_max_diff:.6f}  (expect 0.0)"
    )

    # ── Write JSON ────────────────────────────────────────────────────────────
    json_path = RESULTS_DIR / "ranking_metrics_corrected.json"
    json_path.write_text(json.dumps(out, indent=2))

    # ── Write Markdown ────────────────────────────────────────────────────────
    md = _render_markdown(out, repro_max_diff)
    md_path = RESULTS_DIR / "ranking_metrics_corrected.md"
    md_path.write_text(md)

    print(md)
    print(f"\nSaved: {json_path}")
    print(f"Saved: {md_path}")


def _render_markdown(out: dict, repro_max_diff: float) -> str:
    meta = out["meta"]
    ev = out["evaluators"]
    lines: list[str] = []
    lines.append("# Phase 5 — Corrected Ranking Metrics\n")
    lines.append(
        f"**Test set:** splits_v2 test, n={meta['n_cases']} cases. "
        f"Re-aggregated from the existing `results/phase5/*.jsonl` score files "
        f"(no model or API calls).\n"
    )
    lines.append("**Two corrections applied (audit C1 + C2):**\n")
    lines.append(
        "- **C1 — expected-rank ties:** ranking metrics are the expected value under "
        "uniformly-random tie-breaking (replaces the gold-favouring "
        "`rank = 1 + #(score > gold)`).\n"
        "- **C2 — contamination exclusion:** "
        f"{meta['contamination_distractors']} 'irrelevant' distractors across "
        f"{meta['contamination_cases']} cases were LLM-verified by the miner to contain "
        "the answer; they are dropped from the candidate set before ranking.\n"
    )
    lines.append(
        f"_As-reported reproduction check vs `comparison.json`: "
        f"max |Δ| = {repro_max_diff:.6f} (0.0 = exact match)._\n"
    )

    # Side-by-side table
    lines.append("## As-reported vs Corrected (95% bootstrap CI, n=1000, seed=0)\n")
    header = "| Evaluator | Metric | As-reported | Corrected (point [95% CI]) | Δ |"
    lines.append(header)
    lines.append("|---|---|---:|:---:|---:|")
    for key in EVAL_ORDER:
        e = ev[key]
        for i, (m, mlabel) in enumerate(METRICS):
            label = e["label"] if i == 0 else ""
            rep = e["as_reported"][m]
            cor = e["corrected"][m]
            d = e["delta"][m]
            lines.append(
                f"| {label} | {mlabel} | {rep:.4f} | "
                f"{_fmt_ci(cor['point'], cor['ci_lo'], cor['ci_hi'])} | {d:+.4f} |"
            )
    lines.append("")

    # Compact corrected-only table (thesis appendix form)
    lines.append("## Corrected ranking metrics (appendix form)\n")
    cols = " | ".join(COL_LABELS[k] for k in EVAL_ORDER)
    lines.append(f"| Metric | {cols} |")
    lines.append("|---|" + "---:|" * len(EVAL_ORDER))
    for m, mlabel in METRICS:
        cells = []
        for key in EVAL_ORDER:
            c = ev[key]["corrected"][m]
            cells.append(f"{c['point']:.4f} [{c['ci_lo']:.4f}–{c['ci_hi']:.4f}]")
        lines.append(f"| {mlabel} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
