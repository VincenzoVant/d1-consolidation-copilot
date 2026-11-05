# Choosing the embedding model (group_text: average)

Run on 2026-11-03 on 3500 dev rows, with shortlists of k = 10. Seconds = embedding the 190 standard groups and every dev name, on the machine that ran it (model already loaded).

**Rule:** the highest recall@10 wins, because every mapping method chooses from this shortlist and can never beat it. Ties go to the higher top-1, then to the faster model.

| model | top-1 | top-3 | recall@10 | seconds |
|---|---|---|---|---|
| intfloat/multilingual-e5-base | 65.0% | 81.1% | 90.4% | 6.8 |
| BAAI/bge-m3 | 70.2% | 84.5% | 91.9% | 17.2 |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 42.7% | 60.3% | 74.7% | 5.4 |

**Chosen: `BAAI/bge-m3`** (recall@10 91.9%, top-1 70.2%).

## Recall@10 by language and by source

| model | en | fr | mixed | nl | llm | rules |
|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-base | 92.8% | 89.2% | 92.8% | 86.6% | 79.3% | 98.7% |
| BAAI/bge-m3 | 94.2% | 90.3% | 93.2% | 90.0% | 82.0% | 99.4% |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 87.0% | 72.7% | 74.2% | 64.2% | 61.1% | 84.9% |

Note: the LLM-written dev rows carry the generating LLM's own labels, which are not hand-checked. On the test split the hand check found 13% of those labels wrong, so the `llm` column understates every model by a similar margin. That affects all models alike, so the comparison holds.
