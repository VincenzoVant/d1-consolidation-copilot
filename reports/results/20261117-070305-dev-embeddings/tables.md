# embeddings on dev

3500 accounts from `dev.jsonl`; thresholds code_match 0.09074411968764654, other 0.9989019912301914 (one per segment); target precision 98%.

| measure | value | 95% interval (companies resampled) |
|---|---|---|
| auto-mapped | 35.9% | 28.0% – 44.1% |
| precision of auto-mapped | 99.7% | 99.2% – 100.0% |
| sent to review | 64.1% | |
| top-1 | 70.3% | 64.9% – 75.6% |
| top-3 | 84.8% | |
| recall@10 (shortlist) | 92.6% | |

## By segment

| segment | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| code_match | 1189 | 99.9% | 99.7% | 99.7% |
| other | 2311 | 3.0% | 98.6% | 55.1% |

## By source

| source | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| rules | 2000 | 46.8% | 99.9% | 90.0% |
| llm | 1500 | 21.5% | 99.1% | 44.0% |

## By code_scheme

| code_scheme | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| legacy | 1700 | 3.8% | 100.0% | 71.9% |
| pcmn | 1000 | 87.3% | 99.9% | 90.6% |
| llm | 800 | 40.0% | 99.1% | 41.4% |

## By language

| language | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| en | 900 | 37.6% | 100.0% | 73.2% |
| fr | 900 | 33.4% | 98.7% | 68.1% |
| mixed | 850 | 39.3% | 100.0% | 74.5% |
| nl | 850 | 33.5% | 100.0% | 65.3% |

## By noise

| noise | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| legacy_code | 1700 | 3.8% | 100.0% | 71.9% |
| llm | 1500 | 21.5% | 99.1% | 44.0% |
| case | 534 | 43.4% | 100.0% | 86.3% |
| sub_account | 400 | 44.2% | 100.0% | 90.0% |
| no noise | 338 | 97.6% | 100.0% | 97.6% |
| abbreviation | 293 | 43.3% | 100.0% | 81.2% |
| typo | 286 | 39.2% | 99.1% | 85.0% |
| truncated | 161 | 33.5% | 100.0% | 58.4% |
| bilingual | 140 | 53.6% | 100.0% | 95.0% |
| accents_stripped | 112 | 47.3% | 100.0% | 89.3% |
| misfiled_code | 69 | 10.1% | 57.1% | 73.9% |

## By industry

| industry | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| construction | 900 | 36.8% | 99.4% | 70.3% |
| software | 900 | 34.4% | 100.0% | 68.4% |
| retail | 850 | 35.5% | 100.0% | 70.9% |
| services | 850 | 37.1% | 99.4% | 71.5% |

## By overlaps_dev

| overlaps_dev | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| new name | 3500 | 35.9% | 99.7% | 70.3% |

## Cost

0 calls and $0.00 per 1,000 accounts; median latency 0 ms; cache hits 0%.
