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

This writes `data/generated/{train,dev,test}.jsonl` (12,000 / 3,500 / 600 accounts) from the rule-based generator and the LLM-written company charts frozen in `data/frozen/llm_charts/`. It needs no network and no LLM access.

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

## Choosing the embedding model

```bash
python -m d1.evaluation.select_embedder   # compares the candidate models on dev -> reports/
```

The three multilingual candidates in `config/models.yaml` each build the top-10 shortlist for every dev account, and the model whose shortlist most often holds the right group wins. The result, with charts and the reasoning, is in [`notebooks/embedding_selection.ipynb`](notebooks/embedding_selection.ipynb): **BAAI/bge-m3**, with the right group on the shortlist for 87.0% of dev accounts (e5-base 82.8%, mpnet 71.1%). The first run downloads the models (about 4 GB).

`requirements.txt` holds the exact versions from `uv.lock`. After changing dependencies in `pyproject.toml`, update both:

```bash
uv lock
uv export --frozen --extra dev --no-hashes --no-emit-project -o requirements.txt
```
