# Choosing the embedding model (group_text: average)

Run on 2026-11-06 on 3500 dev rows, with shortlists of k = 10. Seconds = embedding the 190 standard groups and every dev name, on the machine that ran it (model already loaded).

**Rule:** the highest recall@10 wins, because every mapping method chooses from this shortlist and can never beat it. Ties go to the higher top-1, then to the faster model.

| model | top-1 | top-3 | recall@10 | seconds |
|---|---|---|---|---|
| intfloat/multilingual-e5-base | 65.3% | 81.5% | 90.9% | 6.7 |
| BAAI/bge-m3 | 70.3% | 84.8% | 92.6% | 22.5 |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 42.9% | 60.7% | 74.9% | 5.3 |

**Chosen: `BAAI/bge-m3`** (recall@10 92.6%, top-1 70.3%).

## Recall@10 by language and by source

| model | en | fr | mixed | nl | llm | rules |
|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-base | 93.4% | 90.4% | 92.9% | 86.6% | 80.5% | 98.7% |
| BAAI/bge-m3 | 95.1% | 91.1% | 93.5% | 90.5% | 83.5% | 99.4% |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 87.4% | 72.7% | 74.7% | 64.2% | 61.6% | 84.9% |

Note: the LLM-written dev rows carry the generating LLM's own labels, which are not hand-checked. On the test split the hand check found 13% of those labels wrong, so the `llm` column understates every model by a similar margin. That affects all models alike, so the comparison holds.
