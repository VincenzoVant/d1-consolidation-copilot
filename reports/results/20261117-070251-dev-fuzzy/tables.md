# fuzzy on dev

3500 accounts from `dev.jsonl`; thresholds code_match 0.0010190848393807409, other 0.9420413730453838 (one per segment); target precision 98%.

| measure | value | 95% interval (companies resampled) |
|---|---|---|
| auto-mapped | 29.3% | 21.5% – 37.9% |
| precision of auto-mapped | 99.8% | 99.4% – 100.0% |
| sent to review | 70.7% | |
| top-1 | 57.9% | 49.5% – 66.0% |
| top-3 | 67.5% | |
| recall@10 (shortlist) | 92.6% | |

## By segment

| segment | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| code_match | 985 | 100.0% | 99.8% | 99.8% |
| other | 2515 | 1.7% | 100.0% | 41.5% |

## By source

| source | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| rules | 2000 | 45.0% | 99.9% | 89.0% |
| llm | 1500 | 8.5% | 99.2% | 16.5% |

## By code_scheme

| code_scheme | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| legacy | 1700 | 2.4% | 97.5% | 58.8% |
| pcmn | 1000 | 85.9% | 100.0% | 89.5% |
| llm | 800 | 16.0% | 99.2% | 16.5% |

## By language

| language | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| en | 900 | 28.8% | 100.0% | 55.9% |
| fr | 900 | 28.6% | 99.6% | 57.1% |
| mixed | 850 | 30.2% | 99.6% | 60.5% |
| nl | 850 | 29.9% | 100.0% | 58.4% |

## By noise

| noise | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| legacy_code | 1700 | 2.4% | 97.5% | 58.8% |
| llm | 1500 | 8.5% | 99.2% | 16.5% |
| case | 534 | 44.9% | 100.0% | 87.1% |
| sub_account | 400 | 40.2% | 99.4% | 83.5% |
| no noise | 338 | 98.2% | 100.0% | 98.2% |
| abbreviation | 293 | 34.1% | 100.0% | 65.5% |
| typo | 286 | 38.8% | 100.0% | 86.7% |
| truncated | 161 | 28.0% | 97.8% | 55.9% |
| bilingual | 140 | 42.9% | 100.0% | 87.1% |
| accents_stripped | 112 | 43.8% | 100.0% | 87.5% |
| misfiled_code | 69 | 5.8% | 75.0% | 63.8% |

## By industry

| industry | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| construction | 900 | 29.6% | 99.6% | 57.9% |
| software | 900 | 28.2% | 99.6% | 56.2% |
| retail | 850 | 29.1% | 100.0% | 58.9% |
| services | 850 | 30.6% | 100.0% | 58.7% |

## By overlaps_dev

| overlaps_dev | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| new name | 3500 | 29.3% | 99.8% | 57.9% |

## Cost

0 calls and $0.00 per 1,000 accounts; median latency 0 ms; cache hits 0%.
