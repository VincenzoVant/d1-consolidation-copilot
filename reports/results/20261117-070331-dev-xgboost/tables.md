# xgboost on dev

3500 accounts from `dev.jsonl`; thresholds code_match 0.15537693965589777, other 0.9330614215301211 (one per segment); target precision 98%.

| measure | value | 95% interval (companies resampled) |
|---|---|---|
| auto-mapped | 74.3% | 67.1% – 81.1% |
| precision of auto-mapped | 98.5% | 97.9% – 99.0% |
| sent to review | 25.7% | |
| top-1 | 84.5% | 80.5% – 88.3% |
| top-3 | 90.1% | |
| recall@10 (shortlist) | 92.6% | |

## By segment

| segment | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| code_match | 1615 | 99.9% | 98.8% | 98.8% |
| other | 1885 | 52.3% | 98.1% | 72.3% |

## By source

| source | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| rules | 2000 | 92.0% | 99.5% | 96.2% |
| llm | 1500 | 50.5% | 96.0% | 68.9% |

## By code_scheme

| code_scheme | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| legacy | 1700 | 56.2% | 97.6% | 77.4% |
| pcmn | 1000 | 98.7% | 99.8% | 99.1% |
| llm | 800 | 82.0% | 97.9% | 81.5% |

## By language

| language | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| en | 900 | 74.8% | 98.5% | 86.4% |
| fr | 900 | 72.1% | 98.2% | 82.6% |
| mixed | 850 | 77.4% | 98.5% | 86.1% |
| nl | 850 | 72.8% | 98.9% | 82.9% |

## By noise

| noise | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| legacy_code | 1700 | 56.2% | 97.6% | 77.4% |
| llm | 1500 | 50.5% | 96.0% | 68.9% |
| case | 534 | 89.1% | 99.6% | 94.8% |
| sub_account | 400 | 88.2% | 99.4% | 96.0% |
| no noise | 338 | 100.0% | 100.0% | 100.0% |
| abbreviation | 293 | 84.6% | 99.6% | 91.1% |
| typo | 286 | 85.0% | 99.2% | 93.0% |
| truncated | 161 | 70.2% | 93.8% | 80.7% |
| bilingual | 140 | 93.6% | 100.0% | 98.6% |
| accents_stripped | 112 | 91.1% | 100.0% | 95.5% |
| misfiled_code | 69 | 68.1% | 70.2% | 69.6% |

## By industry

| industry | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| construction | 900 | 72.0% | 98.3% | 83.2% |
| software | 900 | 73.7% | 98.5% | 84.3% |
| retail | 850 | 76.4% | 99.1% | 85.5% |
| services | 850 | 75.2% | 98.1% | 85.1% |

## By overlaps_dev

| overlaps_dev | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| new name | 3500 | 74.3% | 98.5% | 84.5% |

## Cost

0 calls and $0.00 per 1,000 accounts; median latency 0 ms; cache hits 0%.
