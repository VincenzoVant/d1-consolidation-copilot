"""The two baselines every real method must beat.

embeddings  take the shortlist's number 1, with its cosine score
fuzzy       ignore the embeddings; compare letters with all 190 group names
            (rapidfuzz WRatio, best of the en/nl/fr names), so a method that
            only matches spelling is measured too
"""

from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process

from d1.config import Config
from d1.datagen.schema import CompanyAccount, StandardAccount
from d1.datagen.splits import normalize_name
from d1.datagen.standard_chart import load_standard_chart
from d1.mapper.methods.base import (
    AccountIn, MethodOutput, check_lengths, read_method_info, register, save_method_info,
)  # fmt: skip
from d1.mapper.retrieve import Candidate

LANGUAGES = ("en", "nl", "fr")


class EmbeddingsMethod:
    name = "embeddings"
    needs_training = False

    def fit(self, train_rows: list[CompanyAccount], candidates: list[list[Candidate]]) -> None:
        pass

    def score(self, accounts: list[AccountIn], candidates: list[list[Candidate]]) -> list[MethodOutput]:
        check_lengths(accounts, candidates)
        out = []
        for account, shortlist in zip(accounts, candidates):
            if not shortlist:
                out.append(MethodOutput(account_id=account.account_id, scores={}, chosen_code=None, status="none", reason="empty shortlist"))
                continue
            out.append(MethodOutput(account_id=account.account_id, scores={c.code: c.score for c in shortlist}, chosen_code=shortlist[0].code))
        return out

    def version(self) -> str:
        return "embeddings-top1-v1"

    def save(self, folder: Path) -> None:
        save_method_info(Path(folder), self.name, self.version())

    def load(self, folder: Path) -> None:
        read_method_info(Path(folder), self.name)


class FuzzyMethod:
    name = "fuzzy"
    needs_training = False

    def __init__(self, chart: list[StandardAccount], k: int = 10):
        self.chart = chart
        self.k = k
        self._codes = [a.code for a in chart]
        # One row per group and language, normalised once: "Frais d'établissement" -> "frais d etablissement".
        self._choices = [normalize_name(a.name(lang)) for a in chart for lang in LANGUAGES]

    def fit(self, train_rows: list[CompanyAccount], candidates: list[list[Candidate]]) -> None:
        pass

    def score(self, accounts: list[AccountIn], candidates: list[list[Candidate]]) -> list[MethodOutput]:
        check_lengths(accounts, candidates)  # the shortlist is ignored, but must still match the accounts
        if not accounts:
            return []
        queries = [normalize_name(a.name) for a in accounts]
        # All accounts against all 570 names at once; per group, the best of its three languages.
        sim = process.cdist(queries, self._choices, scorer=fuzz.WRatio, workers=-1, dtype=np.float32)
        per_group = sim.reshape(len(accounts), len(self.chart), len(LANGUAGES)).max(axis=2) / 100.0
        order = np.argsort(-per_group, axis=1, kind="stable")[:, : self.k]
        out = []
        for account, row, idx in zip(accounts, per_group, order):
            scores = {self._codes[j]: float(row[j]) for j in idx}
            if row[idx[0]] <= 0:
                out.append(MethodOutput(account_id=account.account_id, scores=scores, chosen_code=None, status="none", reason="no letters in common"))
            else:
                out.append(MethodOutput(account_id=account.account_id, scores=scores, chosen_code=self._codes[idx[0]]))
        return out

    def version(self) -> str:
        return f"fuzzy-wratio-k{self.k}-v1"

    def save(self, folder: Path) -> None:
        save_method_info(Path(folder), self.name, self.version(), k=self.k)

    def load(self, folder: Path) -> None:
        self.k = int(read_method_info(Path(folder), self.name)["params"]["k"])


@register("embeddings")
def _embeddings(config: Config) -> EmbeddingsMethod:
    return EmbeddingsMethod()


@register("fuzzy")
def _fuzzy(config: Config) -> FuzzyMethod:
    chart = load_standard_chart(config.data.chart_path, expected=config.data.expected_groups)
    return FuzzyMethod(chart, k=config.models.mapping.candidates_k)
