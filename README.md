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

`requirements.txt` holds the exact versions from `uv.lock`. After changing dependencies in `pyproject.toml`, update both:

```bash
uv lock
uv export --frozen --extra dev --no-hashes --no-emit-project -o requirements.txt
```
