# fuzzy on dev

3500 accounts from `dev.jsonl`; threshold 0.875874049216863; target precision 98%.

| measure | value | 95% interval (companies resampled) |
|---|---|---|
| auto-mapped | 31.6% | 25.5% – 37.4% |
| precision of auto-mapped | 98.6% | 97.5% – 99.3% |
| sent to review | 68.4% | |
| top-1 | 57.9% | 49.5% – 66.0% |
| top-3 | 67.5% | |
| recall@10 (shortlist) | 92.6% | |

## By source

| source | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| rules | 2000 | 53.3% | 100.0% | 89.0% |
| llm | 1500 | 2.6% | 59.0% | 16.5% |

## By language

| language | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| en | 900 | 34.1% | 99.3% | 55.9% |
| fr | 900 | 34.0% | 97.7% | 57.1% |
| mixed | 850 | 26.1% | 98.6% | 60.5% |
| nl | 850 | 31.9% | 98.5% | 58.4% |

## By noise

| noise | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| legacy_code | 1700 | 32.0% | 98.3% | 58.8% |
| llm | 1500 | 2.6% | 59.0% | 16.5% |
| case | 534 | 50.9% | 100.0% | 87.1% |
| sub_account | 400 | 2.8% | 100.0% | 83.5% |
| no noise | 352 | 94.6% | 100.0% | 98.3% |
| abbreviation | 293 | 2.4% | 100.0% | 65.5% |
| typo | 286 | 38.8% | 100.0% | 86.7% |
| truncated | 161 | 1.9% | 100.0% | 55.9% |
| bilingual | 140 | 0.7% | 100.0% | 87.1% |
| accents_stripped | 112 | 50.9% | 100.0% | 87.5% |

## By industry

| industry | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| construction | 900 | 31.0% | 98.9% | 57.9% |
| software | 900 | 31.4% | 98.9% | 56.2% |
| retail | 850 | 30.5% | 98.5% | 58.9% |
| services | 850 | 33.5% | 97.9% | 58.7% |

## By overlaps_dev

| overlaps_dev | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| new name | 3500 | 31.6% | 98.6% | 57.9% |

## Cost

0 calls and $0.00 per 1,000 accounts; median latency 0 ms; cache hits 0%.
