# Phase 3B — LLM Judge Prompt Variant Comparison

**Sample:** 20 cases from `val.jsonl` (seed=42)  
**Pairs per case:** 4 (gold × 1, trap × 1, irrelevant × 2)  
**Total LLM calls:** 400  
**Total cost:** $0.2400  
**Total wall-clock time:** 583.9s  

## Variant Comparison

| Variant | Trap Detection ↑ | Gold Recall ↑ | FPR@1 ↓ | Confidence | Avg Out Tokens | Time (s) |
|---------|-----------------|---------------|---------|------------|----------------|----------|
| BASE | 85% | 95% | 0.1667 | 0.936 | 70.6 | 98.8 |
| WITH_COT | 80% | 100% | 0.2333 | 0.939 | 67.0 | 196.3 |
| WITH_NEGATIVE_EXAMPLES | 85% | 95% | 0.1333 | 0.938 | 72.2 | 97.4 |
| WITH_STRICT_CONTAINMENT | 85% | 95% | 0.1667 | 0.935 | 70.4 | 94.5 |
| MINIMAL | 85% | 95% | 0.1833 | 0.958 | 63.2 | 96.9 |

## Winner

**MINIMAL**

- Trap Detection: 85% (runner-up WITH_STRICT_CONTAINMENT: 85%)
- Gold Recall: 95%
- FPR@1: 0.1833
- Avg output tokens: 63.2

## Disagreement Analysis (3 cases where variants split)

### Case 1 — TRAP passage (expected: NOT RELEVANT)

**Query:** What sport does Miroslav Milutinović play?  
**Passage (first 350 chars):**  Milutinov represented the senior men's Serbian national basketball team for the first time, at the EuroBasket 2015, under the team's head coach Aleksandar Đorđević. In the first phase of the tournament, Serbia dominated in the toughest group of the tournament, Group B, with a 5–0 record, and they then eliminated Finland and the Czech Republic, in …

| Variant | Verdict | Confidence | Reasoning |
|---------|---------|------------|-----------|
| BASE | **NOT RELEVANT** | 0.85 | The passage discusses Milutinov (a Serbian basketball player), not Miroslav Milutinović; while they share a similar surname, they appear to be different individuals, and the question's specific name does not match the passage's subject. |
| WITH_COT | **RELEVANT** | 0.85 | The passage discusses a Serbian basketball player with the surname Milutinov at EuroBasket 2015, which very likely refers to Miroslav Milutinović, and clearly states the sport as basketball. |
| WITH_NEGATIVE_EXAMPLES | **NOT RELEVANT** | 0.85 | The passage discusses Milutinov (a basketball player at EuroBasket 2015), but the question asks about Miroslav Milutinović; while these names are similar, they appear to refer to different individuals, and without confirmation of shared identity, this represents a potential surname similarity trap. |
| WITH_STRICT_CONTAINMENT | **NOT RELEVANT** | 0.85 | The passage discusses Milutinov (a Serbian basketball player), not Miroslav Milutinović; while they share a surname and similar names, they appear to be different individuals, and without confirmation this is the same person, topic match cannot be established. |
| MINIMAL | **NOT RELEVANT** | 0.85 | The passage discusses 'Milutinov' at EuroBasket 2015, but the question asks about 'Miroslav Milutinović' - these appear to be different individuals (different name spellings/forms), so there is no confirmed topic match despite both being basketball players. |

### Case 2 — TRAP passage (expected: NOT RELEVANT)

**Query:** Who was the director of The Return?  
**Passage (first 350 chars):**  Return is a 2011 independent film about an American reservist, wife and mother returning home from her tour of duty in the Middle East. The film was written and directed by Liza Johnson, and stars Linda Cardellini, Michael Shannon and John Slattery. It is Johnson's first feature-length film, and received good reviews at its premiere at the Cannes …

| Variant | Verdict | Confidence | Reasoning |
|---------|---------|------------|-----------|
| BASE | **NOT RELEVANT** | 0.85 | The passage discusses a film titled 'Return' (2011) directed by Liza Johnson, but the question asks about 'The Return,' which is a different film title and may refer to a different work entirely. |
| WITH_COT | **RELEVANT** | 0.88 | The passage explicitly identifies Liza Johnson as the director of the 2011 film 'Return' about an American reservist, which matches the question's subject despite the minor title variation. |
| WITH_NEGATIVE_EXAMPLES | **NOT RELEVANT** | 0.85 | The passage discusses a film titled 'Return' (2011) directed by Liza Johnson, but the question asks about 'The Return,' which is a different film; the similar titles create a similarity trap where the passage may not address the specific film in question. |
| WITH_STRICT_CONTAINMENT | **NOT RELEVANT** | 0.85 | The passage discusses a film titled 'Return' (2011) directed by Liza Johnson, but the question asks about 'The Return,' which is a different film title and may refer to a different work. |
| MINIMAL | **NOT RELEVANT** | 0.85 | The passage discusses a film titled 'Return' (2011), not 'The Return', which appears to be a different film entity. |

### Case 3 — TRAP passage (expected: NOT RELEVANT)

**Query:** What genre is Tempting The Gods: The Selected Stories of Tanith Lee, Volume 1?  
**Passage (first 350 chars):**  Hunting The Shadows: The Selected Stories of Tanith Lee, Volume Two is a 2009 collection of 12 fantasy and science fiction short stories by author Tanith Lee, published by Wildside Press. Only one story, "Queens in Crimson", is a previously unpublished work. The book includes an introduction by Donald Wollheim. Most of the stories of the volume we…

| Variant | Verdict | Confidence | Reasoning |
|---------|---------|------------|-----------|
| BASE | **RELEVANT** | 0.95 | The passage discusses 'Hunting The Shadows: The Selected Stories of Tanith Lee, Volume Two' and explicitly states it is a 'collection of 12 fantasy and science fiction short stories,' while also confirming that 'Volume one, titled Tempting The Gods' exists, establishing that Volume 1 is the same type of collection with the same genres (fantasy and science fiction). |
| WITH_COT | **NOT RELEVANT** | 0.88 | The passage mentions Tempting The Gods, Volume 1 by name but only states the genre of Volume Two (fantasy and science fiction), not explicitly the genre of Volume One. |
| WITH_NEGATIVE_EXAMPLES | **RELEVANT** | 0.92 | The passage discusses 'Hunting The Shadows: The Selected Stories of Tanith Lee, Volume Two' and explicitly mentions that 'Volume one, titled Tempting The Gods' contains fantasy and science fiction stories, directly answering the genre question for the correct work. |
| WITH_STRICT_CONTAINMENT | **RELEVANT** | 0.85 | The passage explicitly states that Volume one is 'Tempting The Gods' and describes Volume Two as a 'collection of fantasy and science fiction short stories,' establishing the genre for the series. |
| MINIMAL | **RELEVANT** | 0.85 | The passage discusses Volume Two of the series and explicitly mentions that Volume One is titled 'Tempting The Gods,' and indicates it is part of a fantasy and science fiction short story collection series, allowing inference of the genre. |

