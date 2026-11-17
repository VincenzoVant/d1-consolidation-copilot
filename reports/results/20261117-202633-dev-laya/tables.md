# laya on dev

3500 accounts from `dev.jsonl`; thresholds code_match 0.10464433595017186, other None (one per segment); target precision 98%.

| measure | value | 95% interval (companies resampled) |
|---|---|---|
| auto-mapped | 24.3% | 18.3% – 30.2% |
| precision of auto-mapped | 98.5% | 97.3% – 99.3% |
| sent to review | 75.7% | |
| top-1 | 39.3% | 35.0% – 43.7% |
| top-3 | 62.9% | |
| recall@10 (shortlist) | 92.6% | |

## By segment

| segment | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| code_match | 853 | 99.9% | 98.5% | 98.5% |
| other | 2647 | 0.0% | – | 20.2% |

## By source

| source | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| rules | 2000 | 27.6% | 98.4% | 45.6% |
| llm | 1500 | 20.1% | 98.7% | 31.0% |

## By code_scheme

| code_scheme | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| legacy | 1700 | 0.4% | 0.0% | 29.8% |
| pcmn | 1000 | 54.5% | 99.4% | 56.5% |
| llm | 800 | 37.5% | 99.0% | 38.1% |

## By language

| language | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| en | 900 | 31.9% | 99.7% | 58.6% |
| fr | 900 | 22.2% | 97.5% | 34.2% |
| mixed | 850 | 23.5% | 98.5% | 36.5% |
| nl | 850 | 19.4% | 97.6% | 27.2% |

## By noise

| noise | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| legacy_code | 1700 | 0.4% | 0.0% | 29.8% |
| llm | 1500 | 20.1% | 98.7% | 31.0% |
| case | 534 | 27.7% | 99.3% | 46.8% |
| sub_account | 400 | 28.0% | 97.3% | 47.8% |
| no noise | 338 | 58.3% | 100.0% | 58.3% |
| abbreviation | 293 | 25.6% | 98.7% | 45.7% |
| typo | 286 | 24.1% | 97.1% | 42.3% |
| truncated | 161 | 21.1% | 97.1% | 30.4% |
| bilingual | 140 | 22.9% | 96.9% | 31.4% |
| accents_stripped | 112 | 25.9% | 96.6% | 33.9% |
| misfiled_code | 69 | 8.7% | 0.0% | 43.5% |

## By industry

| industry | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| construction | 900 | 22.2% | 98.5% | 39.9% |
| software | 900 | 24.7% | 98.6% | 38.1% |
| retail | 850 | 27.2% | 98.7% | 39.4% |
| services | 850 | 23.4% | 98.0% | 39.9% |

## By overlaps_dev

| overlaps_dev | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| new name | 3500 | 24.3% | 98.5% | 39.3% |

## Cost

0 calls and $0.00 per 1,000 accounts; median latency 0 ms; cache hits 100%.
