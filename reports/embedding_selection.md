# Choosing the embedding model

Run on 2026-11-06 on 3500 dev rows, with shortlists of k = 10. Seconds = embedding the 190 standard groups and every dev name, on the machine that ran it (model already loaded).

**Rule:** the highest recall@10 wins, because every mapping method chooses from this shortlist and can never beat it. Ties go to the higher top-1, then to the faster model.

| model | top-1 | top-3 | recall@10 | seconds |
|---|---|---|---|---|
| intfloat/multilingual-e5-base | 59.0% | 73.9% | 83.4% | 5.0 |
| BAAI/bge-m3 | 65.1% | 79.7% | 87.5% | 11.7 |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 39.6% | 57.9% | 71.3% | 4.2 |

**Chosen: `BAAI/bge-m3`** (recall@10 87.5%, top-1 65.1%).

## Recall@10 by language and by source

| model | en | fr | mixed | nl | llm | rules |
|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-base | 86.6% | 82.9% | 86.4% | 77.6% | 64.3% | 97.8% |
| BAAI/bge-m3 | 88.9% | 86.7% | 90.1% | 84.5% | 71.9% | 99.2% |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 83.8% | 68.7% | 71.2% | 61.1% | 54.3% | 84.0% |

Note: the LLM-written dev rows carry the generating LLM's own labels, which are not hand-checked. On the test split the hand check found 13% of those labels wrong, so the `llm` column understates every model by a similar margin. That affects all models alike, so the comparison holds.
