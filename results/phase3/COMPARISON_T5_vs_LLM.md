# Phase 3 — T5 Baseline vs LLM Judge: Fair Head-to-Head Comparison

**Test set:** 136 cases from `data/hard_cases/splits/test.jsonl`  
**LLM variant:** WITH_NEGATIVE_EXAMPLES  
**T5 thresholds:** UPPER=0.592 (CORRECT), LOWER=-0.995 (INCORRECT)  
**Evaluation design:** Both evaluators scored the SAME four passages per case  
(gold_passage, trap_passage, irrelevant[0], irrelevant[1]).  
T5 was run live via `scripts/eval_t5_on_hard_cases.py` — no retroactive score lookup.  

## Metric Comparison

All metrics computed on the same n=136 cases; denominators are identical.

| Metric | T5 Baseline | LLM Judge (WITH_NEG_EX) | Winner |
|--------|------------|------------------------|--------|
| Trap Detection Rate ↑ | 95.6% | 85.3% | **T5** |
| Gold Recall ↑ | 58.1% | 91.9% | **LLM** |
| FPR@1 on Traps ↓ | 4.4% | 14.7% | **T5** |

> **Interpretation:** T5 achieves 95.6% trap detection but only 58.1% gold recall — it is systematically over-conservative, rejecting many correct gold passages. The LLM reaches a better precision/recall balance: 91.9% gold recall at the cost of 14.7% trap FPR vs T5's 4.4%.

## Per-Trap-Type Breakdown

| Trap Type | n | T5 Trap Det | LLM Trap Det | T5 Gold Rec | LLM Gold Rec |
|-----------|---|------------|-------------|------------|-------------|
| entity_alias | 26 | 92.3% | 65.4% | 50.0% | 88.5% |
| topic_overlap | 66 | 93.9% | 83.3% | 62.1% | 95.5% |
| unknown | 44 | 100.0% | 100.0% | 56.8% | 88.6% |

## Disagreement Cases

> Total trap disagreements: 20 / 136 cases.  
> Total gold disagreements: 52 / 136 cases.  
> The cases below are a representative sample chosen to illustrate distinct failure modes.

### LLM Wins (3 shown)

#### LLM Win 1 — topic_overlap trap (passage: trap)

**Query:** What sport does Mutanda Kwesele play?  
**Trap passage:** Mumbi Kwesele  
**Trap passage (first 300 chars):**  Kwesele attended Barry University in 2015, where he played for two seasons. He transferred to Humboldt State University where he played during their 2016 and 2017 seasons.…  
**T5:** score=1.0084 → CORRECT (accepts trap)  
**LLM:** score=0.1500 → NOT RELEVANT (correctly rejects)  

#### LLM Win 2 — topic_overlap trap (passage: trap)

**Query:** What is the capital of Chiprovtsi Municipality?  
**Trap passage:** Chiprovtsi  
**Trap passage (first 300 chars):**  The municipal government consists of a mayor (kmet), a deputy mayor and a secretary. Since 2007, the municipality has been governed by Zaharin Ivanov Zamfirov of Bulgarian Agrarian People's Union "Aleksandar Stamboliyski" who won the municipal elections with 1,615 votes or 62.67% against Antoaneta …  
**T5:** score=1.0003 → CORRECT (accepts trap)  
**LLM:** score=0.1500 → NOT RELEVANT (correctly rejects)  

#### LLM Win 3 — topic_overlap trap (passage: trap)

**Query:** What is Shorty Hamilton's occupation?  
**Trap passage:** Shorty Green  
**Trap passage (first 300 chars):**  the Northern Ontario Hockey Association. He moved back to Hamilton in 1923 and began his professional career. He played on a line with his brother, "Red" Green, and Billy Burch for the last place Tigers. Green was unanimously voted as team captain prior to the start of the 1924–25 NHL season and hi…  
**T5:** score=0.9544 → CORRECT (accepts trap)  
**LLM:** score=0.1500 → NOT RELEVANT (correctly rejects)  

### T5 Wins (2 shown)

#### T5 Win 1 — entity_alias trap (passage: trap)

**Query:** What sport does Bobby Windsor play?  
**Trap passage:** Bobby Windsor  
**Trap passage (first 300 chars):**  Robert William Windsor (born 31 January 1948 in Newport, Monmouthshire), known as Bobby and nicknamed "The Duke", is a former rugby union player who gained 28 rugby union caps for Wales as a hooker between 1973 and 1979. Windsor published his autobiography in October 2010 entitled 'The Iron Duke'.…  
**T5:** score=-1.0307 → INCORRECT (correctly rejects)  
**LLM:** score=0.9900 → RELEVANT (incorrectly accepts)  

#### T5 Win 2 — entity_alias trap (passage: trap)

**Query:** What genre is I Lost My Heart in Heidelberg?  
**Trap passage:** I Lost My Heart in Heidelberg (1926 film)  
**Trap passage (first 300 chars):**  I Lost My Heart in Heidelberg (German: Ich hab mein Herz in Heidelberg verloren) is a 1926 German silent film directed by Arthur Bergen and starring Emil Höfer, Gertrud de Lalsky and Werner Fuetterer. The title alludes to the popular 1925 song I Lost My Heart in Heidelberg composed by Fred Raymond …  
**T5:** score=-0.9567 → AMBIGUOUS (correctly rejects)  
**LLM:** score=0.9800 → RELEVANT (incorrectly accepts)  

