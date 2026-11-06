# embeddings on dev

3500 accounts from `dev.jsonl`; threshold 0.9281479086042382; target precision 98%.

| measure | value | 95% interval (companies resampled) |
|---|---|---|
| auto-mapped | 32.1% | 27.1% – 36.9% |
| precision of auto-mapped | 98.0% | 97.0% – 98.9% |
| sent to review | 67.9% | |
| top-1 | 70.3% | 64.9% – 75.6% |
| top-3 | 84.8% | |
| recall@10 (shortlist) | 92.6% | |

## By source

| source | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| rules | 2000 | 49.9% | 99.3% | 90.0% |
| llm | 1500 | 8.4% | 88.1% | 44.0% |

## By language

| language | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| en | 900 | 35.2% | 97.5% | 73.2% |
| fr | 900 | 28.2% | 97.2% | 68.1% |
| mixed | 850 | 36.5% | 98.7% | 74.5% |
| nl | 850 | 28.6% | 98.8% | 65.3% |

## By noise

| noise | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| legacy_code | 1700 | 32.4% | 97.6% | 71.9% |
| llm | 1500 | 8.4% | 88.1% | 44.0% |
| case | 534 | 40.3% | 98.6% | 86.3% |
| sub_account | 400 | 41.5% | 100.0% | 90.0% |
| no noise | 352 | 67.6% | 100.0% | 97.7% |
| abbreviation | 293 | 29.7% | 100.0% | 81.2% |
| typo | 286 | 32.9% | 98.9% | 85.0% |
| truncated | 161 | 21.1% | 79.4% | 58.4% |
| bilingual | 140 | 79.3% | 100.0% | 95.0% |
| accents_stripped | 112 | 42.0% | 100.0% | 89.3% |

## By industry

| industry | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| construction | 900 | 32.2% | 98.3% | 70.3% |
| software | 900 | 30.7% | 99.3% | 68.4% |
| retail | 850 | 31.4% | 97.8% | 70.9% |
| services | 850 | 34.2% | 96.9% | 71.5% |

## By overlaps_dev

| overlaps_dev | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| new name | 3500 | 32.1% | 98.0% | 70.3% |

## Cost

0 calls and $0.00 per 1,000 accounts; median latency 0 ms; cache hits 0%.
