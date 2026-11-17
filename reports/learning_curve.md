# Learning curve

Each method refitted on a growing share of the train companies (whole companies, the same mix of rule-based and LLM-written), calibrated on dev the same way every time. Auto share at the target precision, with a 95% interval over dev companies.

## xgboost

| share of train | companies | accounts | top-1 | auto-mapped | 95% interval | precision | auto (rules) | auto (LLM) |
|---|---|---|---|---|---|---|---|---|
| 25% | 60 | 3,000 | 83.0% | 70.1% | 62.3% – 77.4% | 98.5% | 87.5% | 46.9% |
| 50% | 120 | 5,999 | 83.6% | 71.9% | 64.3% – 79.3% | 98.8% | 90.3% | 47.3% |
| 75% | 180 | 8,999 | 83.9% | 73.9% | 66.6% – 81.0% | 98.5% | 91.6% | 50.3% |
| 100% | 240 | 11,999 | 84.5% | 74.3% | 67.1% – 81.1% | 98.5% | 92.0% | 50.5% |

## random_forest

| share of train | companies | accounts | top-1 | auto-mapped | 95% interval | precision | auto (rules) | auto (LLM) |
|---|---|---|---|---|---|---|---|---|
| 25% | 60 | 3,000 | 82.3% | 70.0% | 62.4% – 77.3% | 98.7% | 87.8% | 46.4% |
| 50% | 120 | 5,999 | 83.2% | 71.5% | 64.0% – 79.0% | 98.8% | 89.6% | 47.4% |
| 75% | 180 | 8,999 | 83.2% | 72.5% | 65.1% – 79.7% | 98.5% | 90.0% | 49.2% |
| 100% | 240 | 11,999 | 83.6% | 74.2% | 67.1% – 81.0% | 98.5% | 91.6% | 50.9% |

## Decision

- **xgboost:** 75% → 100% of the companies changes the auto share by +0.3 points; the interval's half-width is 7.0, so more data stopped helping.
- **random_forest:** 75% → 100% of the companies changes the auto share by +1.7 points; the interval's half-width is 7.0, so more data stopped helping.
