"""LLM-written company charts: free-form names that no noise rule produced.

Each company is written once by the generator model, frozen as a JSON file in
data/frozen/llm_charts/, and from then on only read back. So the splits rebuild
identically on any machine, without a Claude login, and the hand check of the
test rows stays valid.

  select_llm_groups -> build_prompt -> fetch_llm_company (one LLM call, frozen)
                                    -> load_llm_company  (frozen file -> CompanyAccount rows)

Code schemes: LLMs number accounts after the target group (1680 for group 168),
which hands a mapper the answer. So, as for rule-based companies, the split plan
gives each company a scheme, applied when loading: "llm" keeps the LLM's own
codes (like a company that follows PCMN numbering), "legacy" keeps the LLM's
names but replaces the codes with sequential 4-digit codes that never start with
the target. The scheme does not change the prompt, so it is not part of the call.
"""

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from d1.config import REPO_ROOT
from d1.datagen.rule_charts import BASE_SHARE, legacy_codes
from d1.datagen.schema import CompanyAccount, Language, Split, StandardAccount
from d1.llm import LLMBackend, LLMError

PROMPT_PATH = REPO_ROOT / "prompts" / "generate_company_chart.md"
PROMPT_VERSION = "v1"  # bump when the prompt changes, so frozen files show which prompt made them
SYSTEM = "You write realistic synthetic charts of accounts. Return JSON only."
TIMEOUT_S = 300  # one company is a long answer from a large model

LLMCodeScheme = Literal["llm", "legacy"]

LANGUAGE_NAMES: dict[str, str] = {
    "nl": "Dutch",
    "fr": "French",
    "en": "English",
    "mixed": "a mix of Dutch, French and English, as in a bilingual Brussels company",
}

LLM_CHART_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "accounts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "target_code": {"type": "string"},
                    "code": {"type": "string"},
                    "name": {"type": "string"},
                },
                "required": ["target_code", "code", "name"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["accounts"],
    "additionalProperties": False,
}


class LLMOutputError(Exception):
    """An LLM-written chart could not be produced or is unusable."""


class FrozenOutputMissing(Exception):
    """A company's frozen LLM file does not exist yet."""


class LLMAccount(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_code: str
    code: str
    name: str


class _Chart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accounts: list[LLMAccount]


def _min_rows(n_accounts: int) -> int:
    """Fewest usable rows a company may have: one per group it was given."""
    return math.ceil(BASE_SHARE * n_accounts)


def select_llm_groups(chart: list[StandardAccount], n_accounts: int, rng: Random) -> list[StandardAccount]:
    """The groups one LLM company gets: 80% of n_accounts different groups, as for rule-based companies."""
    n_groups = round(BASE_SHARE * n_accounts)
    if n_groups > len(chart):
        raise ValueError(f"{n_accounts} accounts need {n_groups} different groups, but the chart has only {len(chart)} groups")
    return sorted(rng.sample(chart, n_groups), key=lambda a: a.code)


def build_prompt(industry: str, language: Language, groups: list[StandardAccount], n_accounts: int) -> str:
    """Fill the prompt template in prompts/ for one company."""
    rows = []
    for g in groups:
        local = f"{g.name_nl} / {g.name_fr}" if language == "mixed" else g.name(language)
        rows.append(f"| {g.code} | {g.name_en} | {local} |")
    return PROMPT_PATH.read_text(encoding="utf-8").format(
        industry=industry,
        language_name=LANGUAGE_NAMES[language],
        n_groups=len(groups),
        n_accounts=n_accounts,
        groups_table="\n".join(rows),
    )


def parse_chart(response: dict, allowed_codes: set[str]) -> list[LLMAccount]:
    """Check the answer's shape, then keep only usable rows.

    Dropped: rows whose target is not one of the groups the company was given,
    rows with an empty name, and repeated codes (the first one is kept).
    """
    try:
        chart = _Chart.model_validate(response)
    except ValidationError as e:
        raise LLMOutputError(f"answer has the wrong shape: {e}") from None
    rows, seen = [], set()
    for row in chart.accounts:
        code, name = row.code.strip(), row.name.strip()
        if row.target_code not in allowed_codes or not name or not code or code in seen:
            continue
        seen.add(code)
        rows.append(LLMAccount(target_code=row.target_code, code=code, name=name))
    return rows


def frozen_path(frozen_dir: Path, company_id: str) -> Path:
    return Path(frozen_dir) / f"{company_id}.json"


def fetch_llm_company(
    backend: LLMBackend,
    *,
    generator_model: str,
    mapping_model: str,
    company_id: str,
    split: Split,
    industry: str,
    language: Language,
    groups: list[StandardAccount],
    n_accounts: int,
    frozen_dir: Path,
) -> Path:
    """Have the generator model write one company and freeze the answer.

    Does nothing if the company's frozen file already exists, so an interrupted
    run can simply be restarted. Nothing is written unless the answer is usable.
    """
    path = frozen_path(frozen_dir, company_id)
    if path.exists():
        return path

    prompt = build_prompt(industry, language, groups, n_accounts)
    try:
        response = backend.complete(prompt, schema=LLM_CHART_SCHEMA, model=generator_model, system=SYSTEM, timeout_s=TIMEOUT_S)
    except LLMError as e:
        raise LLMOutputError(f"{company_id}: {e}") from None
    if response.model == mapping_model:
        raise LLMOutputError(f"{company_id}: answered by {response.model}, the mapping model; the generator must differ")

    try:
        usable = len(parse_chart(response.structured, {g.code for g in groups}))
    except LLMOutputError as e:
        raise LLMOutputError(f"{company_id}: {e}") from None
    if usable < _min_rows(n_accounts):
        raise LLMOutputError(f"{company_id}: only {usable} usable accounts, need at least {_min_rows(n_accounts)}")

    record = {
        "company_id": company_id,
        "split": split,
        "industry": industry,
        "language": language,
        "requested_model": generator_model,
        "served_model": response.model,
        "backend": type(backend).__name__,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest(),
        "group_codes": [g.code for g in groups],
        "n_accounts": n_accounts,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "usage": {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
            "latency_ms": response.latency_ms,
        },
        "response": response.structured,
    }
    # Write to a temporary file, then rename: an interrupted run never leaves a
    # half-written file that the next run would skip as "already fetched".
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def _cut_to(rows: list[LLMAccount], n_accounts: int) -> list[LLMAccount]:
    """At most n_accounts rows, dropping surplus second accounts before any group's first.

    The LLM writes in group order and often a few accounts too many; cutting the
    tail would always drop the highest groups (revenue). The LLM's order is kept.
    """
    seen: set[str] = set()
    firsts, seconds = [], []
    for i, row in enumerate(rows):
        (seconds if row.target_code in seen else firsts).append(i)
        seen.add(row.target_code)
    keep = set(firsts[:n_accounts] + seconds[: max(0, n_accounts - len(firsts))])
    return [row for i, row in enumerate(rows) if i in keep]


def load_llm_company(
    frozen_dir: Path, company_id: str, split: Split, n_accounts: int, code_scheme: LLMCodeScheme = "llm"
) -> list[CompanyAccount]:
    """Turn one frozen file into the company's n_accounts rows, with codes per code_scheme."""
    path = frozen_path(frozen_dir, company_id)
    if not path.exists():
        raise FrozenOutputMissing(f"{path} is missing: run `python -m d1.datagen fetch-llm --split {split}`")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["split"] != split:
        raise LLMOutputError(f"{path.name} belongs to split {record['split']}, not {split}")

    rows = _cut_to(parse_chart(record["response"], set(record["group_codes"])), n_accounts)
    if len(rows) < _min_rows(n_accounts):
        raise LLMOutputError(f"{path.name}: only {len(rows)} usable accounts, need at least {_min_rows(n_accounts)}")

    if code_scheme == "legacy":
        # Seeded by the company id, so every rebuild gives the same codes.
        codes = legacy_codes([row.target_code for row in rows], Random(company_id))
        noise = ["llm", "legacy_code"]
    else:
        codes = [row.code for row in rows]
        noise = ["llm"]
    return [
        CompanyAccount(
            account_id=f"{company_id}-{i:03d}",
            company_id=company_id,
            industry=record["industry"],
            split=split,
            source="llm",
            language=record["language"],
            code_scheme=code_scheme,
            code=code,
            name=row.name,
            noise=list(noise),
            target_code=row.target_code,
        )
        for i, (code, row) in enumerate(zip(codes, rows))
    ]
