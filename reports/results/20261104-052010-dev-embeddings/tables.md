# embeddings on dev

3500 accounts from `dev.jsonl`; threshold 0.9250848633139125; target precision 98%.

| measure | value | 95% interval (companies resampled) |
|---|---|---|
| auto-mapped | 32.6% | 27.4% – 37.6% |
| precision of auto-mapped | 98.1% | 97.0% – 98.9% |
| sent to review | 67.4% | |
| top-1 | 70.2% | 64.9% – 75.5% |
| top-3 | 84.5% | |
| recall@10 (shortlist) | 91.9% | |

## By source

| source | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| rules | 2000 | 51.1% | 99.2% | 90.0% |
| llm | 1500 | 7.8% | 88.0% | 43.7% |

## By language

| language | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| en | 900 | 35.6% | 97.8% | 72.8% |
| fr | 900 | 28.9% | 97.7% | 68.1% |
| mixed | 850 | 37.2% | 98.1% | 74.4% |
| nl | 850 | 28.7% | 98.8% | 65.4% |

## By noise

| noise | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| legacy_code | 1700 | 32.7% | 98.2% | 71.6% |
| llm | 1500 | 7.8% | 88.0% | 43.7% |
| case | 534 | 41.2% | 98.6% | 86.3% |
| sub_account | 400 | 42.5% | 100.0% | 90.0% |
| no noise | 352 | 68.8% | 100.0% | 97.7% |
| abbreviation | 293 | 31.1% | 100.0% | 81.2% |
| typo | 286 | 34.6% | 98.0% | 85.0% |
| truncated | 161 | 23.0% | 78.4% | 58.4% |
| bilingual | 140 | 80.0% | 100.0% | 95.0% |
| accents_stripped | 112 | 42.9% | 100.0% | 89.3% |

## By industry

| industry | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| construction | 900 | 32.6% | 98.3% | 70.1% |
| software | 900 | 30.4% | 99.6% | 68.8% |
| retail | 850 | 31.8% | 98.1% | 70.6% |
| services | 850 | 35.6% | 96.4% | 71.3% |

## By overlaps_dev

| overlaps_dev | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| new name | 3500 | 32.6% | 98.1% | 70.2% |

## Cost

0 calls and $0.00 per 1,000 accounts; median latency 0 ms; cache hits 0%.
