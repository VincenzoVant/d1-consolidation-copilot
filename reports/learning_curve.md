# Learning curve

Each method refitted on a growing share of the train companies (whole companies, the same mix of rule-based and LLM-written), calibrated on dev the same way every time. Auto share at the target precision, with a 95% interval over dev companies.

## xgboost

| share of train | companies | accounts | top-1 | auto-mapped | 95% interval | precision | auto (rules) | auto (LLM) |
|---|---|---|---|---|---|---|---|---|
| 25% | 60 | 3,000 | 83.6% | 75.3% | 68.4% – 81.8% | 98.0% | 92.7% | 52.1% |
| 50% | 120 | 5,999 | 84.2% | 76.6% | 69.7% – 83.1% | 98.1% | 93.8% | 53.7% |
| 75% | 180 | 8,999 | 84.4% | 78.2% | 71.6% – 84.4% | 98.0% | 94.8% | 56.2% |
| 100% | 240 | 11,999 | 84.9% | 78.8% | 72.5% – 84.9% | 98.0% | 94.8% | 57.5% |

## random_forest

| share of train | companies | accounts | top-1 | auto-mapped | 95% interval | precision | auto (rules) | auto (LLM) |
|---|---|---|---|---|---|---|---|---|
| 25% | 60 | 3,000 | 83.1% | 71.2% | 64.2% – 78.0% | 98.0% | 91.5% | 44.1% |
| 50% | 120 | 5,999 | 83.7% | 76.0% | 69.4% – 82.3% | 98.1% | 93.2% | 53.0% |
| 75% | 180 | 8,999 | 83.7% | 78.1% | 71.8% – 84.1% | 98.0% | 94.1% | 56.7% |
| 100% | 240 | 11,999 | 84.1% | 78.1% | 71.7% – 84.1% | 98.0% | 94.0% | 56.8% |

## Decision

- **xgboost:** 75% → 100% of the companies changes the auto share by +0.6 points; the interval's half-width is 6.2, so more data stopped helping.
- **random_forest:** 75% → 100% of the companies changes the auto share by +0.0 points; the interval's half-width is 6.2, so more data stopped helping.
