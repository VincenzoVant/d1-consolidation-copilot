# cascade on dev

3500 accounts from `dev.jsonl`; thresholds xgboost: code_match 0.15537693965589777, other 0.9330614215301211 (one per segment); then llm on the rest: code_match None, other None (one per segment); target precision 98%.

| measure | value | 95% interval (companies resampled) |
|---|---|---|
| auto-mapped | 74.3% | 67.1% – 81.1% |
| precision of auto-mapped | 98.5% | 97.9% – 99.0% |
| sent to review | 25.7% | |
| top-1 | 88.8% | 86.1% – 91.5% |
| top-3 | 91.9% | |
| recall@10 (shortlist) | 92.6% | |

## By segment

| segment | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| code_match | 1615 | 99.9% | 98.8% | 98.8% |
| other | 1885 | 52.3% | 98.1% | 80.3% |

## By source

| source | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| rules | 2000 | 92.0% | 99.5% | 97.4% |
| llm | 1500 | 50.5% | 96.0% | 77.4% |

## By code_scheme

| code_scheme | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| legacy | 1700 | 56.2% | 97.6% | 86.0% |
| pcmn | 1000 | 98.7% | 99.8% | 99.1% |
| llm | 800 | 82.0% | 97.9% | 82.0% |

## By language

| language | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| en | 900 | 74.8% | 98.5% | 91.8% |
| fr | 900 | 72.1% | 98.2% | 86.8% |
| mixed | 850 | 77.4% | 98.5% | 89.9% |
| nl | 850 | 72.8% | 98.9% | 86.8% |

## By noise

| noise | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| legacy_code | 1700 | 56.2% | 97.6% | 86.0% |
| llm | 1500 | 50.5% | 96.0% | 77.4% |
| case | 534 | 89.1% | 99.6% | 96.3% |
| sub_account | 400 | 88.2% | 99.4% | 97.5% |
| no noise | 338 | 100.0% | 100.0% | 100.0% |
| abbreviation | 293 | 84.6% | 99.6% | 94.2% |
| typo | 286 | 85.0% | 99.2% | 94.4% |
| truncated | 161 | 70.2% | 93.8% | 83.2% |
| bilingual | 140 | 93.6% | 100.0% | 99.3% |
| accents_stripped | 112 | 91.1% | 100.0% | 99.1% |
| misfiled_code | 69 | 68.1% | 70.2% | 75.4% |

## By industry

| industry | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| construction | 900 | 72.0% | 98.3% | 87.7% |
| software | 900 | 73.7% | 98.5% | 89.0% |
| retail | 850 | 76.4% | 99.1% | 88.8% |
| services | 850 | 75.2% | 98.1% | 89.9% |

## By overlaps_dev

| overlaps_dev | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| new name | 3500 | 74.3% | 98.5% | 88.8% |

## Cost

0 calls and $0.00 per 1,000 accounts; median latency 0 ms; cache hits 26%.
