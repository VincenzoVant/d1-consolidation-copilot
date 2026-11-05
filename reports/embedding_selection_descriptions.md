# Choosing the embedding model (group_text: descriptions)

Run on 2026-11-03 on 3500 dev rows, with shortlists of k = 10. Seconds = embedding the 190 standard groups and every dev name, on the machine that ran it (model already loaded).

**Rule:** the highest recall@10 wins, because every mapping method chooses from this shortlist and can never beat it. Ties go to the higher top-1, then to the faster model.

| model | top-1 | top-3 | recall@10 | seconds |
|---|---|---|---|---|
| intfloat/multilingual-e5-base | 64.1% | 81.5% | 91.0% | 6.1 |
| BAAI/bge-m3 | 68.5% | 85.6% | 93.6% | 14.8 |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 40.9% | 60.0% | 75.1% | 5.2 |

**Chosen: `BAAI/bge-m3`** (recall@10 93.6%, top-1 68.5%).

## Recall@10 by language and by source

| model | en | fr | mixed | nl | llm | rules |
|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-base | 94.4% | 89.9% | 92.7% | 86.8% | 81.9% | 97.9% |
| BAAI/bge-m3 | 96.9% | 92.3% | 93.9% | 91.1% | 86.4% | 99.0% |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 90.2% | 72.2% | 73.6% | 63.5% | 63.9% | 83.5% |

Note: the LLM-written dev rows carry the generating LLM's own labels, which are not hand-checked. On the test split the hand check found 13% of those labels wrong, so the `llm` column understates every model by a similar margin. That affects all models alike, so the comparison holds.
