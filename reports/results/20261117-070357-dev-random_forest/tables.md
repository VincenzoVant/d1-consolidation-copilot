# random_forest on dev

3500 accounts from `dev.jsonl`; thresholds code_match 0.21061441171832757, other 0.9254532783701099 (one per segment); target precision 98%.

| measure | value | 95% interval (companies resampled) |
|---|---|---|
| auto-mapped | 74.2% | 67.1% – 81.0% |
| precision of auto-mapped | 98.5% | 97.9% – 99.1% |
| sent to review | 25.8% | |
| top-1 | 83.6% | 79.4% – 87.6% |
| top-3 | 90.4% | |
| recall@10 (shortlist) | 92.6% | |

## By segment

| segment | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| code_match | 1613 | 100.0% | 98.8% | 98.8% |
| other | 1887 | 52.1% | 98.2% | 70.6% |

## By source

| source | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| rules | 2000 | 91.6% | 99.6% | 95.9% |
| llm | 1500 | 50.9% | 95.9% | 67.2% |

## By code_scheme

| code_scheme | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| legacy | 1700 | 56.5% | 97.7% | 75.5% |
| pcmn | 1000 | 98.1% | 99.8% | 99.2% |
| llm | 800 | 81.8% | 97.9% | 81.1% |

## By language

| language | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| en | 900 | 75.3% | 98.5% | 85.4% |
| fr | 900 | 71.2% | 98.3% | 81.7% |
| mixed | 850 | 77.5% | 98.6% | 85.6% |
| nl | 850 | 72.7% | 98.7% | 81.5% |

## By noise

| noise | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| legacy_code | 1700 | 56.5% | 97.7% | 75.5% |
| llm | 1500 | 50.9% | 95.9% | 67.2% |
| case | 534 | 88.4% | 100.0% | 94.0% |
| sub_account | 400 | 89.2% | 99.4% | 95.8% |
| no noise | 338 | 100.0% | 100.0% | 100.0% |
| abbreviation | 293 | 85.3% | 99.2% | 90.8% |
| typo | 286 | 84.3% | 99.2% | 92.0% |
| truncated | 161 | 69.6% | 94.6% | 80.1% |
| bilingual | 140 | 92.9% | 100.0% | 96.4% |
| accents_stripped | 112 | 90.2% | 100.0% | 95.5% |
| misfiled_code | 69 | 58.0% | 65.0% | 69.6% |

## By industry

| industry | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| construction | 900 | 72.2% | 98.5% | 82.6% |
| software | 900 | 73.9% | 98.6% | 83.4% |
| retail | 850 | 75.5% | 99.2% | 84.6% |
| services | 850 | 75.2% | 97.8% | 83.8% |

## By overlaps_dev

| overlaps_dev | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| new name | 3500 | 74.2% | 98.5% | 83.6% |

## Cost

0 calls and $0.00 per 1,000 accounts; median latency 0 ms; cache hits 0%.
