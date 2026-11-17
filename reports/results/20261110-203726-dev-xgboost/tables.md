# xgboost on dev

3500 accounts from `dev.jsonl`; threshold 0.8433460543648038; target precision 98%.

| measure | value | 95% interval (companies resampled) |
|---|---|---|
| auto-mapped | 78.8% | 72.5% – 84.9% |
| precision of auto-mapped | 98.0% | 97.0% – 98.8% |
| sent to review | 21.2% | |
| top-1 | 84.9% | 81.0% – 88.8% |
| top-3 | 90.1% | |
| recall@10 (shortlist) | 92.6% | |

## By source

| source | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| rules | 2000 | 94.8% | 99.0% | 96.3% |
| llm | 1500 | 57.5% | 95.8% | 69.8% |

## By code_scheme

| code_scheme | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| legacy | 1700 | 64.5% | 95.0% | 77.1% |
| pcmn | 1000 | 99.4% | 100.0% | 99.4% |
| llm | 800 | 83.6% | 100.0% | 83.6% |

## By language

| language | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| en | 900 | 81.0% | 97.4% | 86.7% |
| fr | 900 | 76.0% | 98.4% | 83.4% |
| mixed | 850 | 81.9% | 98.0% | 86.4% |
| nl | 850 | 76.5% | 98.3% | 83.3% |

## By noise

| noise | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| legacy_code | 1700 | 64.5% | 95.0% | 77.1% |
| llm | 1500 | 57.5% | 95.8% | 69.8% |
| case | 534 | 93.3% | 98.2% | 94.8% |
| sub_account | 400 | 93.2% | 98.9% | 96.2% |
| no noise | 352 | 100.0% | 100.0% | 100.0% |
| abbreviation | 293 | 90.4% | 97.7% | 91.1% |
| typo | 286 | 89.2% | 99.2% | 93.7% |
| truncated | 161 | 79.5% | 93.0% | 82.0% |
| bilingual | 140 | 95.0% | 99.2% | 98.6% |
| accents_stripped | 112 | 92.9% | 100.0% | 95.5% |

## By industry

| industry | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| construction | 900 | 77.1% | 97.6% | 83.7% |
| software | 900 | 78.1% | 98.3% | 84.7% |
| retail | 850 | 80.2% | 98.1% | 85.6% |
| services | 850 | 80.0% | 98.1% | 85.9% |

## By overlaps_dev

| overlaps_dev | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| new name | 3500 | 78.8% | 98.0% | 84.9% |

## Cost

0 calls and $0.00 per 1,000 accounts; median latency 0 ms; cache hits 0%.
