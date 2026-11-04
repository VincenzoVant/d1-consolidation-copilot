# embeddings on dev

3500 accounts from `dev.jsonl`; threshold 0.9501254307739344; target precision 98%.

| measure | value | 95% interval (companies resampled) |
|---|---|---|
| auto-mapped | 23.0% | 18.9% – 27.1% |
| precision of auto-mapped | 98.3% | 97.0% – 99.3% |
| sent to review | 77.0% | |
| top-1 | 64.7% | 58.1% – 71.3% |
| top-3 | 79.2% | |
| recall@10 (shortlist) | 87.0% | |

## By source

| source | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| rules | 2000 | 37.5% | 99.6% | 88.9% |
| llm | 1500 | 3.7% | 80.0% | 32.5% |

## By language

| language | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| en | 900 | 25.7% | 97.8% | 65.6% |
| fr | 900 | 20.7% | 97.8% | 62.7% |
| mixed | 850 | 26.9% | 99.1% | 70.1% |
| nl | 850 | 18.8% | 98.1% | 60.7% |

## By noise

| noise | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| legacy_code | 1700 | 23.5% | 98.2% | 66.4% |
| llm | 1500 | 3.7% | 80.0% | 32.5% |
| case | 534 | 25.7% | 99.3% | 86.1% |
| sub_account | 400 | 30.8% | 100.0% | 89.2% |
| no noise | 352 | 50.0% | 100.0% | 96.0% |
| abbreviation | 293 | 18.4% | 100.0% | 78.2% |
| typo | 286 | 22.4% | 100.0% | 84.3% |
| truncated | 161 | 16.1% | 88.5% | 62.1% |
| bilingual | 140 | 68.6% | 100.0% | 97.1% |
| accents_stripped | 112 | 31.2% | 100.0% | 84.8% |

## By industry

| industry | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| construction | 900 | 23.4% | 99.1% | 64.1% |
| software | 900 | 21.3% | 99.5% | 64.4% |
| retail | 850 | 22.0% | 97.9% | 65.8% |
| services | 850 | 25.4% | 96.8% | 64.7% |

## By overlaps_dev

| overlaps_dev | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| new name | 3500 | 23.0% | 98.3% | 64.7% |

## Cost

0 calls and $0.00 per 1,000 accounts; median latency 0 ms; cache hits 0%.
