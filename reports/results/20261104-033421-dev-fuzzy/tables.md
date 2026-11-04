# fuzzy on dev

3500 accounts from `dev.jsonl`; threshold 0.8541204566265104; target precision 98%.

| measure | value | 95% interval (companies resampled) |
|---|---|---|
| auto-mapped | 33.7% | 27.2% – 39.8% |
| precision of auto-mapped | 98.4% | 97.2% – 99.2% |
| sent to review | 66.3% | |
| top-1 | 57.7% | 49.1% – 65.9% |
| top-3 | 67.2% | |
| recall@10 (shortlist) | 87.0% | |

## By source

| source | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| rules | 2000 | 57.0% | 100.0% | 89.0% |
| llm | 1500 | 2.7% | 53.7% | 16.0% |

## By language

| language | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| en | 900 | 36.7% | 99.1% | 55.9% |
| fr | 900 | 35.9% | 97.8% | 56.8% |
| mixed | 850 | 28.0% | 97.9% | 60.4% |
| nl | 850 | 34.0% | 98.6% | 58.0% |

## By noise

| noise | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| legacy_code | 1700 | 34.1% | 98.1% | 58.5% |
| llm | 1500 | 2.7% | 53.7% | 16.0% |
| case | 534 | 55.8% | 100.0% | 87.1% |
| sub_account | 400 | 6.5% | 100.0% | 83.5% |
| no noise | 352 | 96.6% | 100.0% | 98.3% |
| abbreviation | 293 | 7.2% | 100.0% | 65.5% |
| typo | 286 | 46.2% | 100.0% | 86.7% |
| truncated | 161 | 3.7% | 100.0% | 55.9% |
| bilingual | 140 | 0.7% | 100.0% | 87.1% |
| accents_stripped | 112 | 52.7% | 100.0% | 87.5% |

## By industry

| industry | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| construction | 900 | 32.8% | 98.6% | 58.0% |
| software | 900 | 33.6% | 98.3% | 55.3% |
| retail | 850 | 33.3% | 98.9% | 58.9% |
| services | 850 | 35.3% | 97.7% | 58.7% |

## By overlaps_dev

| overlaps_dev | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| new name | 3500 | 33.7% | 98.4% | 57.7% |

## Cost

0 calls and $0.00 per 1,000 accounts; median latency 0 ms; cache hits 0%.
