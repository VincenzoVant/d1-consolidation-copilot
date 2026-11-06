# Choosing the embedding model (group_text: descriptions)

Run on 2026-11-06 on 3500 dev rows, with shortlists of k = 10. Seconds = embedding the 190 standard groups and every dev name, on the machine that ran it (model already loaded).

**Rule:** the highest recall@10 wins, because every mapping method chooses from this shortlist and can never beat it. Ties go to the higher top-1, then to the faster model.

| model | top-1 | top-3 | recall@10 | seconds |
|---|---|---|---|---|
| intfloat/multilingual-e5-base | 64.6% | 82.3% | 91.7% | 5.8 |
| BAAI/bge-m3 | 68.9% | 86.0% | 93.9% | 14.5 |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 41.5% | 60.5% | 75.3% | 4.6 |

**Chosen: `BAAI/bge-m3`** (recall@10 93.9%, top-1 68.9%).

## Recall@10 by language and by source

| model | en | fr | mixed | nl | llm | rules |
|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-base | 95.7% | 90.2% | 93.2% | 87.4% | 83.4% | 97.9% |
| BAAI/bge-m3 | 97.1% | 93.0% | 94.0% | 91.3% | 87.1% | 99.0% |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 90.1% | 72.6% | 73.8% | 64.0% | 64.4% | 83.5% |

Note: the LLM-written dev rows carry the generating LLM's own labels, which are not hand-checked. On the test split the hand check found 13% of those labels wrong, so the `llm` column understates every model by a similar margin. That affects all models alike, so the comparison holds.
