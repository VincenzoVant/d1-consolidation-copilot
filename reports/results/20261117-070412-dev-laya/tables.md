# laya on dev

3500 accounts from `dev.jsonl`; thresholds code_match 0.1441747649185961, other None (one per segment); target precision 98%.

| measure | value | 95% interval (companies resampled) |
|---|---|---|
| auto-mapped | 23.3% | 17.5% – 29.0% |
| precision of auto-mapped | 98.4% | 97.2% – 99.2% |
| sent to review | 76.7% | |
| top-1 | 37.6% | 33.0% – 42.1% |
| top-3 | 62.9% | |
| recall@10 (shortlist) | 92.6% | |

## By segment

| segment | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| code_match | 815 | 100.0% | 98.4% | 98.4% |
| other | 2685 | 0.0% | – | 19.1% |

## By source

| source | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| rules | 2000 | 27.3% | 98.3% | 45.0% |
| llm | 1500 | 18.0% | 98.5% | 27.8% |

## By code_scheme

| code_scheme | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| legacy | 1700 | 0.4% | 0.0% | 28.5% |
| pcmn | 1000 | 53.9% | 99.4% | 55.9% |
| llm | 800 | 33.6% | 98.9% | 34.1% |

## By language

| language | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| en | 900 | 31.1% | 99.6% | 57.1% |
| fr | 900 | 21.2% | 97.4% | 32.7% |
| mixed | 850 | 22.8% | 98.5% | 35.3% |
| nl | 850 | 17.6% | 97.3% | 24.5% |

## By noise

| noise | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| legacy_code | 1700 | 0.4% | 0.0% | 28.5% |
| llm | 1500 | 18.0% | 98.5% | 27.8% |
| case | 534 | 27.5% | 99.3% | 46.1% |
| sub_account | 400 | 27.8% | 97.3% | 47.0% |
| no noise | 338 | 58.3% | 100.0% | 58.3% |
| abbreviation | 293 | 25.3% | 98.6% | 45.1% |
| typo | 286 | 22.7% | 96.9% | 39.9% |
| truncated | 161 | 20.5% | 97.0% | 29.2% |
| bilingual | 140 | 22.9% | 96.9% | 31.4% |
| accents_stripped | 112 | 25.0% | 96.4% | 33.0% |
| misfiled_code | 69 | 8.7% | 0.0% | 43.5% |

## By industry

| industry | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| construction | 900 | 21.3% | 98.4% | 38.3% |
| software | 900 | 23.7% | 98.6% | 36.4% |
| retail | 850 | 25.6% | 98.6% | 37.4% |
| services | 850 | 22.6% | 97.9% | 38.2% |

## By overlaps_dev

| overlaps_dev | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| new name | 3500 | 23.3% | 98.4% | 37.6% |

## Cost

0 calls and $0.00 per 1,000 accounts; median latency 0 ms; cache hits 100%.
