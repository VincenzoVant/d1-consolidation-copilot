# D1 Consolidation Copilot

> **All data in this project is synthetic.** No real company ledgers are used, and no claim of production accuracy is made. The account names hold no real company, bank or brand names: where a ledger would name one, they use a generic word ("bank", "subsidiary").

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

01 standard chart · 02 noise · 03 rule-based companies · 04 LLM-written companies · 05 splits · 06 hand check · 07 retrieval · 08 embedding selection · 09 calibration · 10 baselines and pipeline · 11 metrics and the test guard · 12 group descriptions · 13 pair features · 14 rerankers · 15 Laya

## Choosing the embedding model

```bash
python -m d1.evaluation.select_embedder   # compares the candidate models on dev -> reports/
```

The three multilingual candidates in `config/models.yaml` each build the top-10 shortlist for every dev account, and the model whose shortlist most often holds the right group wins. The result, with charts and the reasoning, is in [`notebooks/08_embedding_selection.ipynb`](notebooks/08_embedding_selection.ipynb): **BAAI/bge-m3**, with the right group on the shortlist for 87.5% of dev accounts (e5-base 83.4%, mpnet 71.3%). The first run downloads the models (about 4 GB).

## Describing the standard groups

```bash
python -m d1.mapper.descriptions                              # once: Sonnet describes the 190 groups -> data/frozen/group_descriptions/
python -m d1.evaluation.select_embedder --group-text average  # the comparison again, groups embedded with their descriptions
```

Official group names miss how bookkeepers write: "Gross wages shop staff" put group 620 *Remuneration and Direct Social Benefits* at rank 47. Each group now gets a one-line description and everyday ledger words in Dutch, French and English. They're written by a different model from the one that wrote the synthetic accounts, from the chart alone, and frozen.

Each group is scored by the mean of its names-only and its described similarity (`group_text: average`). On dev with bge-m3, that takes the shortlist from 87.5% to 92.6% (LLM-written accounts 71.9% → 83.5%). The embeddings baseline then auto-maps 32.1% instead of 21.4% at 98% precision (with one pooled threshold, as when the choice was made). See [`notebooks/12_group_descriptions.ipynb`](notebooks/12_group_descriptions.ipynb).

## Training and calibrating a method

```bash
python -m d1.mapper.train --method fuzzy        # or embeddings, xgboost, random_forest, laya, or all
```

This calibrates the method on dev and saves it in `artifacts/methods/<method>/`.

**The 98% bar holds in each segment the system can see.** One threshold over all accounts is set mostly by the easy ones, and lets a hard group in below 98%. So there is one threshold where the account's own code starts with the picked group and one where it doesn't (`mapping.threshold_by: code_match`), and each must reach 98% on dev on its own. Whether an account was written by an LLM isn't known on real data; whether its code agrees with the pick is. On dev, the baselines auto-map 29.3% (fuzzy) and 35.9% (embeddings) of accounts, at 99.8% and 99.7%. Almost all of that is in the segment where the code agrees with their pick; on names alone they rarely reach 98%.

## XGBoost and random-forest rerankers

```bash
python -m d1.mapper.train --method xgboost                          # or random_forest
python -m d1.evaluation.learning_curve --method xgboost random_forest
```

Each candidate on an account's shortlist becomes 27 numbers (the shortlist score, letters, the account code, the meaning), and a tree model trained on the train split scores each one. On dev both auto-map about 74% of accounts at 98.5% precision (XGBoost 74.3%, interval 67.1–81.1%; top-1 84.5%), against 29–36% for the baselines. Much of that gain is the account code, but the names carry weight too: on accounts whose code doesn't agree with the pick, XGBoost still auto-maps 52% at 98.1%. LLM-written accounts remain the hard part (50.5% auto-mapped at 96.0%). More rule-based training data stopped helping: from 75% to 100% of the train companies, the auto share changes by +0.3 points (XGBoost) and +1.7 (random forest), well inside its ±7.0 interval. See [`notebooks/14_rerankers.ipynb`](notebooks/14_rerankers.ipynb). The random forest's model file (28 MB) isn't committed: `--method random_forest` rebuilds it in seconds.

**The account codes are realistically imperfect.** 4% of the PCMN-style and LLM codes are misfiled under a neighbouring group's number, as a bookkeeper who picked the wrong sub-account would (`misfiled_code`). Without them, a matching code was right 100% of the time in the data and the rerankers trusted it blindly. Those 69 dev accounts are where the rerankers' mistakes concentrate: XGBoost auto-maps 68% of them at 70% precision.

## Laya: a pretrained decision model, zero-shot

```bash
python -m d1.mapper.train --method laya   # downloads laya-multilingual once (pinned revision)
```

[Laya](https://huggingface.co/convaiinnovations/laya-multilingual) reads an account and its 10 candidates and answers one multiple-choice question, without any training on this data. Its top-1 on dev is 39.3% (embeddings baseline 70.3%), and it auto-maps 24.3% at 98.5%, all where the account's code agrees with its pick. Its "none of these" answer doesn't veto the pick: on dev, the right group was on the shortlist for 81% of those answers, so "none" is kept as a doubt signal that lowers the confidence instead. This is in line with its authors' own note that the base model is "a fast base to specialise, not a zero-shot decision engine". It runs locally on the Mac's GPU (about 0.2 s per account, no cost), asks each question under 6 rotations of the option order to cancel a position bias, and caches every answer, so a rerun is instant. See [`notebooks/15_laya.ipynb`](notebooks/15_laya.ipynb).

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
