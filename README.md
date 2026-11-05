# D1 Consolidation Copilot

> **All data in this project is synthetic.** No real company ledgers are used, and no claim of production accuracy is made.

Maps the charts of accounts of several group companies (Dutch, French, English, mixed) onto the Belgian standard chart of accounts, and matches intercompany entries between them.

## Setup

Python 3.12 is required.

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
```

With plain pip:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run the tests with `pytest` (or `uv run pytest`).

## Build the data

```bash
python -m d1.datagen build
```

This writes `data/generated/{train,dev,test}.jsonl` (12,000 / 3,500 / 600 accounts) from the rule-based generator and the LLM-written company charts frozen in `data/frozen/llm_charts/`. It needs no network and no LLM access. Running it again leaves identical files untouched; if the result would differ from the saved files, it stops and keeps them, unless you add `--force` (the same holds for `apply-review`).

The frozen charts were written once by the generator model with:

```bash
python -m d1.datagen fetch-llm --split all   # 78 calls; only companies without a frozen file
```

By default this runs through the Claude CLI on your own login (`llm_backend` in `config/models.yaml`).

## Hand check of the test set

```bash
python -m d1.datagen export-review   # 250 test rows -> data/labels/test_review.csv
python -m d1.datagen apply-review    # verdicts -> data/generated/test_checked.jsonl
```

All 200 LLM-written test rows and a random 50 rule-based rows are checked by hand: `ok`, `fix` (with `corrected_target`) or `drop`.

## Notebooks: the pipeline, step by step

`notebooks/` has one notebook per step, in order. Each follows one real account or record through that step, then shows it on all the data with charts:

01 standard chart · 02 noise · 03 rule-based companies · 04 LLM-written companies · 05 splits · 06 hand check · 07 retrieval · 08 embedding selection · 09 calibration · 10 baselines and pipeline · 11 metrics and the test guard · 12 group descriptions

## Choosing the embedding model

```bash
python -m d1.evaluation.select_embedder   # compares the candidate models on dev -> reports/
```

The three multilingual candidates in `config/models.yaml` each build the top-10 shortlist for every dev account, and the model whose shortlist most often holds the right group wins. The result, with charts and the reasoning, is in [`notebooks/08_embedding_selection.ipynb`](notebooks/08_embedding_selection.ipynb): **BAAI/bge-m3**, with the right group on the shortlist for 87.0% of dev accounts (e5-base 82.8%, mpnet 71.1%). The first run downloads the models (about 4 GB).

## Describing the standard groups

```bash
python -m d1.mapper.descriptions                              # once: Sonnet describes the 190 groups -> data/frozen/group_descriptions/
python -m d1.evaluation.select_embedder --group-text average  # the comparison again, groups embedded with their descriptions
```

Official group names miss how bookkeepers write: "Gross wages shop staff" put group 620 *Remuneration and Direct Social Benefits* at rank 47. Each group now gets a one-line description and everyday ledger words in Dutch, French and English. They're written by a different model from the one that wrote the synthetic accounts, from the chart alone, and frozen.

Each group is scored by the mean of its names-only and its described similarity (`group_text: average`). On dev with bge-m3, that takes the shortlist from 87.0% to 91.9% (LLM-written accounts 70.7% → 82.0%). The embeddings baseline then auto-maps 32.6% instead of 23.0% at 98% precision. See [`notebooks/12_group_descriptions.ipynb`](notebooks/12_group_descriptions.ipynb).

## Training and calibrating a method

```bash
python -m d1.mapper.train --method fuzzy        # or embeddings, or all
```

This calibrates the method on dev and saves it in `artifacts/methods/<method>/`. On dev, the two baselines auto-map 33.7% (fuzzy) and 32.6% (embeddings, with the group descriptions) of accounts at 98% precision. But on the LLM-written accounts, fuzzy's auto-maps are right only 54% of the time (notebook 10).

## Evaluating a method

```bash
python -m d1.evaluation.run_mapping --split dev --method embeddings
```

This writes `reports/results/<run>/` with the metrics (95% intervals over companies), readable tables and one line per account. The test split runs only with `--reported`, reads the hand-checked labels, and every such run is logged in `reports/test_runs.jsonl`. The train split is refused.

`requirements.txt` holds the exact versions from `uv.lock`. After changing dependencies in `pyproject.toml`, update both:

```bash
uv lock
uv export --frozen --extra dev --no-hashes --no-emit-project -o requirements.txt
```
