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

`requirements.txt` holds the exact versions from `uv.lock`. After changing dependencies in `pyproject.toml`, update both:

```bash
uv lock
uv export --frozen --extra dev --no-hashes --no-emit-project -o requirements.txt
```
