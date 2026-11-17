# random_forest on dev

3500 accounts from `dev.jsonl`; threshold 0.8152832773825895; target precision 98%.

| measure | value | 95% interval (companies resampled) |
|---|---|---|
| auto-mapped | 78.1% | 71.7% – 84.1% |
| precision of auto-mapped | 98.0% | 97.1% – 98.8% |
| sent to review | 21.9% | |
| top-1 | 84.1% | 80.0% – 88.2% |
| top-3 | 90.5% | |
| recall@10 (shortlist) | 92.6% | |

## By source

| source | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| rules | 2000 | 94.0% | 99.0% | 95.9% |
| llm | 1500 | 56.8% | 95.8% | 68.5% |

## By code_scheme

| code_scheme | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| legacy | 1700 | 63.4% | 95.0% | 75.4% |
| pcmn | 1000 | 99.1% | 100.0% | 99.4% |
| llm | 800 | 83.1% | 100.0% | 83.5% |

## By language

| language | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| en | 900 | 80.6% | 97.7% | 86.0% |
| fr | 900 | 75.6% | 97.9% | 82.2% |
| mixed | 850 | 80.9% | 98.0% | 86.1% |
| nl | 850 | 75.3% | 98.6% | 82.1% |

## By noise

| noise | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| legacy_code | 1700 | 63.4% | 95.0% | 75.4% |
| llm | 1500 | 56.8% | 95.8% | 68.5% |
| case | 534 | 91.4% | 99.0% | 94.2% |
| sub_account | 400 | 94.0% | 98.4% | 95.8% |
| no noise | 352 | 100.0% | 100.0% | 100.0% |
| abbreviation | 293 | 89.8% | 98.1% | 91.5% |
| typo | 286 | 89.9% | 98.1% | 92.0% |
| truncated | 161 | 75.2% | 90.1% | 80.1% |
| bilingual | 140 | 94.3% | 100.0% | 96.4% |
| accents_stripped | 112 | 94.6% | 98.1% | 95.5% |

## By industry

| industry | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| construction | 900 | 76.9% | 97.7% | 83.1% |
| software | 900 | 77.8% | 97.9% | 83.8% |
| retail | 850 | 79.3% | 98.1% | 85.2% |
| services | 850 | 78.5% | 98.5% | 84.5% |

## By overlaps_dev

| overlaps_dev | accounts | auto-mapped | precision of auto-mapped | top-1 |
|---|---|---|---|---|
| new name | 3500 | 78.1% | 98.0% | 84.1% |

## Cost

0 calls and $0.00 per 1,000 accounts; median latency 0 ms; cache hits 0%.
