# Choosing the embedding model

Run on 2026-11-03 on 3500 dev rows, with shortlists of k = 10. Seconds = embedding the 190 standard groups and every dev name, on the machine that ran it (model already loaded).

**Rule:** the highest recall@10 wins, because every mapping method chooses from this shortlist and can never beat it. Ties go to the higher top-1, then to the faster model.

| model | top-1 | top-3 | recall@10 | seconds |
|---|---|---|---|---|
| intfloat/multilingual-e5-base | 58.4% | 73.4% | 82.8% | 5.0 |
| BAAI/bge-m3 | 64.7% | 79.2% | 87.0% | 12.1 |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 39.2% | 57.5% | 71.1% | 4.5 |

**Chosen: `BAAI/bge-m3`** (recall@10 87.0%, top-1 64.7%).

## Recall@10 by language and by source

| model | en | fr | mixed | nl | llm | rules |
|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-base | 85.8% | 81.8% | 86.0% | 77.5% | 62.9% | 97.8% |
| BAAI/bge-m3 | 88.7% | 85.9% | 89.4% | 84.1% | 70.7% | 99.2% |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 83.8% | 68.4% | 70.9% | 60.6% | 53.8% | 84.0% |

Note: the LLM-written dev rows carry the generating LLM's own labels, which are not hand-checked. On the test split the hand check found 13% of those labels wrong, so the `llm` column understates every model by a similar margin. That affects all models alike, so the comparison holds.
