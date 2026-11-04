"""What every mapping method gets, what it returns, and how it is found by name.

A method receives accounts and, for each, the embedding shortlist; it returns,
per account, a score for each candidate it considered and its pick (or none).
Calibration and the auto/review decision happen afterwards, the same for all
methods (pipeline.py), so methods differ only in how they score and choose.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel

from d1.config import Config, Method
from d1.datagen.schema import CompanyAccount
from d1.mapper.retrieve import Candidate


class AccountIn(BaseModel):
    """One account to map, as a company exports it."""

    account_id: str
    name: str
    code: str | None = None
    company_id: str | None = None
    neighbours: list[str] = []  # names of the accounts just before and after it in the company's chart


class MethodOutput(BaseModel):
    account_id: str
    scores: dict[str, float]  # candidate code -> raw score, higher = better, in the method's own order
    chosen_code: str | None  # None: the method said "none of these", or failed
    status: Literal["ok", "none", "invalid", "error", "capped"] = "ok"
    reason: str | None = None
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    cached: bool = False


class MappingMethod(Protocol):
    name: Method
    needs_training: bool

    def fit(self, train_rows: list[CompanyAccount], candidates: list[list[Candidate]]) -> None: ...

    def score(self, accounts: list[AccountIn], candidates: list[list[Candidate]]) -> list[MethodOutput]: ...

    def version(self) -> str: ...

    def save(self, folder: Path) -> None: ...

    def load(self, folder: Path) -> None: ...


# name -> a function that builds the method from the config. Each method's module adds itself.
METHOD_REGISTRY: dict[Method, Callable[[Config], MappingMethod]] = {}


def register(name: Method):
    def add(factory: Callable[[Config], MappingMethod]) -> Callable[[Config], MappingMethod]:
        METHOD_REGISTRY[name] = factory
        return factory

    return add


def accounts_from_rows(rows: list[CompanyAccount], n_neighbours: int = 2) -> list[AccountIn]:
    """Generated rows as method input, with the names of up to n_neighbours accounts on each side.

    Neighbours never cross into another company: in a real export they would not exist.
    """
    out = []
    for i, row in enumerate(rows):
        same = lambda j: 0 <= j < len(rows) and rows[j].company_id == row.company_id  # noqa: E731
        before = [rows[j].name for j in range(i - n_neighbours, i) if same(j)]
        after = [rows[j].name for j in range(i + 1, i + 1 + n_neighbours) if same(j)]
        out.append(AccountIn(account_id=row.account_id, name=row.name, code=row.code, company_id=row.company_id, neighbours=before + after))
    return out


def check_lengths(accounts: list[AccountIn], candidates: list[list[Candidate]]) -> None:
    if len(accounts) != len(candidates):
        raise ValueError(f"{len(accounts)} accounts but {len(candidates)} shortlists")


def save_method_info(folder: Path, name: str, version: str, **params) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "method.json").write_text(json.dumps({"method": name, "version": version, "params": params}, indent=2) + "\n", encoding="utf-8")


def read_method_info(folder: Path, expected: str) -> dict:
    info = json.loads((folder / "method.json").read_text(encoding="utf-8"))
    if info["method"] != expected:
        raise ValueError(f"{folder} holds a {info['method']} method, not {expected}")
    return info
