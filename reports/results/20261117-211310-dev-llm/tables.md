# llm on dev

3500 accounts from `dev.jsonl`; thresholds code_match 0.10968756696605012, other 0.97335482755094 (one per segment); target precision 98%.

| measure | value | 95% interval (companies resampled) |
|---|---|---|
| auto-mapped | 61.4% | 54.3% – 68.3% |
| precision of auto-mapped | 99.6% | 99.3% – 99.9% |
| sent to review | 38.6% | |
| top-1 | 88.5% | 85.7% – 91.1% |
| top-3 | 88.5% | |
| recall@10 (shortlist) | 92.6% | |

## By segment

| segment | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| code_match | 1554 | 99.9% | 99.8% | 99.7% |
| other | 1946 | 30.7% | 99.2% | 79.4% |

## By source

| source | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| rules | 2000 | 72.7% | 99.9% | 97.5% |
| llm | 1500 | 46.4% | 99.0% | 76.3% |

## By code_scheme

| code_scheme | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| legacy | 1700 | 34.5% | 99.1% | 87.4% |
| pcmn | 1000 | 95.4% | 99.9% | 98.7% |
| llm | 800 | 76.0% | 99.7% | 78.0% |

## By language

| language | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| en | 900 | 63.6% | 99.7% | 91.2% |
| fr | 900 | 57.8% | 99.6% | 86.8% |
| mixed | 850 | 63.3% | 99.4% | 88.5% |
| nl | 850 | 61.1% | 99.8% | 87.3% |

## By noise

| noise | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| legacy_code | 1700 | 34.5% | 99.1% | 87.4% |
| llm | 1500 | 46.4% | 99.0% | 76.3% |
| case | 534 | 70.2% | 100.0% | 96.4% |
| sub_account | 400 | 50.2% | 99.5% | 96.8% |
| no noise | 338 | 100.0% | 100.0% | 100.0% |
| abbreviation | 293 | 64.8% | 99.5% | 95.6% |
| typo | 286 | 66.4% | 100.0% | 96.2% |
| truncated | 161 | 55.9% | 98.9% | 82.6% |
| bilingual | 140 | 75.0% | 100.0% | 97.1% |
| accents_stripped | 112 | 63.4% | 100.0% | 97.3% |
| misfiled_code | 69 | 18.8% | 76.9% | 88.4% |

## By industry

| industry | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| construction | 900 | 61.8% | 99.5% | 88.4% |
| software | 900 | 60.4% | 99.8% | 87.6% |
| retail | 850 | 60.7% | 99.8% | 88.0% |
| services | 850 | 62.7% | 99.4% | 89.9% |

## By overlaps_dev

| overlaps_dev | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| new name | 3500 | 61.4% | 99.6% | 88.5% |

## Cost

0 calls and $0.00 per 1,000 accounts; median latency 0 ms; cache hits 100%.
