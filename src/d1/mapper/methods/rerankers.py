"""XGBoost and random-forest rerankers: learn which shortlist candidate is the right one.

Every account's shortlist of 10 becomes 10 rows of 27 numbers (features.py). A
tree model, trained on the train split, gives each row the probability "this
is the right group", and the account's pick is the row with the highest one.
Both models share this code; only the model differs:

  xgboost        boosted trees: each new tree corrects the ones before it; early
                 stopping on held-out companies decides how many trees
  random_forest  500 independent trees, each on a bootstrap sample, averaged

The hold-out is 10% of the train *companies*, never dev: dev calibrates the
scores and chooses the method, so it must not shape the model as well. Whole
companies are held out, because accounts of one company look alike. On the
hold-out, XGBoost stops adding trees, and both models measure which features
they rely on.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from random import Random
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# The module, not its names: features.py imports methods.base, so when it is imported first,
# this file runs while features.py is still half-loaded. Its names are looked up when used.
from d1.mapper import features as pair_features
from d1.config import Config
from d1.datagen.schema import CompanyAccount
from d1.datagen.standard_chart import load_standard_chart
from d1.mapper.methods.base import (
    AccountIn, MethodOutput, check_lengths, read_method_info, register, save_method_info,
)  # fmt: skip
from d1.mapper.retrieve import Candidate, Retriever

HOLDOUT_SHARE = 0.10
MODEL_FILES = {"xgboost": "model.json", "random_forest": "model.joblib"}
IMPORTANCE_REPEATS = 3  # each feature is shuffled this many times; the mean drop is its importance


class RerankerError(Exception):
    """A reranker used in the wrong order, or with data it can't learn from."""


# --- the two models ---------------------------------------------------------------------------------


def xgboost_model(seed: int = 0, **overrides) -> XGBClassifier:
    """Boosted trees. n_estimators is only a ceiling: early stopping picks the number of trees."""
    params = dict(objective="binary:logistic", n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, early_stopping_rounds=50, eval_metric="logloss", random_state=seed)  # fmt: skip
    return XGBClassifier(**(params | overrides))


def random_forest_model(seed: int = 0, **overrides) -> RandomForestClassifier:
    """Independent deep trees, averaged. balanced_subsample: the 1 right row in 10 weighs as much as the 9 wrong ones."""
    params = dict(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    return RandomForestClassifier(**(params | overrides))


# --- the hold-out and the importance ------------------------------------------------------------------


def company_holdout(rows: list[CompanyAccount], share: float = HOLDOUT_SHARE, seed: int = 0) -> np.ndarray:
    """True for the rows of the held-out companies: about `share` of each source's companies.

    Per source (rule-based, LLM-written), so the hold-out has the same mix as train.
    At least one company is held out, and at least one stays for fitting.
    """
    by_source: dict[str, list[str]] = {}
    for r in rows:
        if r.company_id not in by_source.setdefault(r.source, []):
            by_source[r.source].append(r.company_id)
    rng = Random(seed)
    held: set[str] = set()
    for source in sorted(by_source):
        companies = sorted(by_source[source])
        if len(companies) >= 2:
            n = min(len(companies) - 1, max(1, round(share * len(companies))))
            held |= set(rng.sample(companies, n))
    if not held:
        raise RerankerError(f"the hold-out needs at least 2 train companies of one source, got {sum(map(len, by_source.values()))} companies")
    return np.array([r.company_id in held for r in rows], dtype=bool)


def top1_share(p: np.ndarray, y: np.ndarray, group: np.ndarray) -> float:
    """Share of accounts whose highest-scored candidate is the right one (an account without one counts as wrong)."""
    if len(group) == 0:
        return 0.0
    order = np.lexsort((-p, group))  # by account, and within an account the highest score first
    first = np.r_[True, group[order][1:] != group[order][:-1]]
    return float(y[order][first].sum() / first.sum())


def permutation_importance(
    predict: Callable[[np.ndarray], np.ndarray], X: np.ndarray, y: np.ndarray, group: np.ndarray, names: list[str],
    repeats: int = IMPORTANCE_REPEATS, seed: int = 0,
) -> list[tuple[str, float]]:  # fmt: skip
    """Per feature: how much top-1 drops when that column is shuffled, so it carries no information. Largest first.

    Measured on what the method is judged on (top-1 per account), and the same way
    for both models, so their importances can be compared.
    """
    rng = np.random.default_rng(seed)
    base = top1_share(predict(X), y, group)
    out = []
    for j, name in enumerate(names):
        drops = []
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[:, j] = rng.permutation(shuffled[:, j])
            drops.append(base - top1_share(predict(shuffled), y, group))
        out.append((name, float(np.mean(drops))))
    return sorted(out, key=lambda kv: -kv[1])


# --- the method ------------------------------------------------------------------------------------


class PairwiseReranker:
    """A MappingMethod that scores each (account, candidate) pair with a tree model and picks the best."""

    needs_training = True

    def __init__(self, name: str, featurizer: pair_features.PairFeaturizer, model_factory: Callable[[int], Any], seed: int = 0):
        if name not in MODEL_FILES:
            raise RerankerError(f"unknown reranker {name!r}: choose from {', '.join(MODEL_FILES)}")
        self.name = name
        self.featurizer = featurizer
        self.model_factory = model_factory
        self.seed = seed
        self.model: Any = None  # set by fit() or load()
        self.importance: list[tuple[str, float]] = []
        self.holdout_companies: list[str] = []
        self.fit_info: dict = {}

    def fit(self, train_rows: list[CompanyAccount], candidates: list[list[Candidate]]) -> None:
        """Fit the features' PCA and the model on 90% of the train companies; the other 10% is the hold-out."""
        if len(train_rows) != len(candidates):
            raise ValueError(f"{len(train_rows)} rows but {len(candidates)} shortlists")
        held = company_holdout(train_rows, seed=self.seed)
        fit_rows = [r for r, h in zip(train_rows, held) if not h]
        held_rows = [r for r, h in zip(train_rows, held) if h]
        self.featurizer.fit(fit_rows)  # the hold-out doesn't shape the features either
        X, y, _ = pair_features.training_pairs(fit_rows, [c for c, h in zip(candidates, held) if not h], self.featurizer)
        X_held, y_held, g_held = pair_features.training_pairs(held_rows, [c for c, h in zip(candidates, held) if h], self.featurizer)

        model = self.model_factory(self.seed)
        if getattr(model, "early_stopping_rounds", None):  # XGBoost: stop when the hold-out stops improving
            model.fit(X, y, eval_set=[(X_held, y_held)], verbose=False)
        else:
            model.fit(X, y)
        if isinstance(model, RandomForestClassifier):
            # Fitted in parallel, but predicted on one thread: in parallel, the 500 trees' votes are added up
            # in a varying order, and the last digit of a probability would change from run to run.
            model.set_params(n_jobs=1)
        self.model = model
        self.holdout_companies = sorted({r.company_id for r in held_rows})
        self.importance = permutation_importance(self._predict, X_held, y_held, g_held, self.featurizer.feature_names, seed=self.seed)
        best = getattr(model, "best_iteration", None)
        self.fit_info = {
            "train_accounts": len(fit_rows), "train_pairs": int(len(y)), "holdout_companies": len(self.holdout_companies),
            "holdout_accounts": len(held_rows), "holdout_top1": top1_share(self._predict(X_held), y_held, g_held),
            "trees": int(best) + 1 if best is not None else getattr(model, "n_estimators", None),
        }  # fmt: skip

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def score(self, accounts: list[AccountIn], candidates: list[list[Candidate]]) -> list[MethodOutput]:
        check_lengths(accounts, candidates)
        if self.model is None:
            raise RerankerError("call fit() on train rows, or load() a trained reranker, before score()")
        matrices = self.featurizer.transform(accounts, candidates)
        sizes = [len(m) for m in matrices]
        probs = self._predict(np.vstack(matrices)) if sum(sizes) else np.zeros(0)
        out = []
        for account, shortlist, p in zip(accounts, candidates, np.split(probs, np.cumsum(sizes)[:-1])):
            if not shortlist:
                out.append(MethodOutput(account_id=account.account_id, scores={}, chosen_code=None, status="none", reason="empty shortlist"))
                continue
            order = np.argsort(-p, kind="stable")  # ties keep the shortlist's order
            scores = {shortlist[i].code: float(p[i]) for i in order}
            out.append(MethodOutput(account_id=account.account_id, scores=scores, chosen_code=shortlist[order[0]].code))
        return out

    def feature_importance(self) -> list[tuple[str, float]]:
        """(feature, top-1 drop on the hold-out when it is shuffled), largest first."""
        return list(self.importance)

    def version(self) -> str:
        return f"{self.name}-{pair_features.VERSION}-pca{self.featurizer.n_pca}-v1"

    # --- saving ------------------------------------------------------------------------------------

    def save(self, folder: Path) -> None:
        """method.json (settings, hold-out, importance), featurizer files and the model file.

        XGBoost is saved in its own JSON format. The random forest has none, so it is
        a joblib file: a pickle, to be loaded only from artifacts made by this code.
        """
        if self.model is None:
            raise RerankerError("nothing to save: fit() the reranker first")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.featurizer.save(folder)
        path = folder / MODEL_FILES[self.name]
        if self.name == "xgboost":
            self.model.save_model(path)
        else:
            joblib.dump(self.model, path, compress=3)
        plain = {k: v for k, v in self.model.get_params().items() if isinstance(v, (str, int, bool)) or (isinstance(v, float) and np.isfinite(v))}
        save_method_info(folder, self.name, self.version(), seed=self.seed, model=plain, holdout_companies=self.holdout_companies,
                         fit_info=self.fit_info, importance=[[f, v] for f, v in self.importance])  # fmt: skip

    def load(self, folder: Path) -> None:
        folder = Path(folder)
        params = read_method_info(folder, self.name)["params"]
        configured = self.featurizer.n_pca
        self.featurizer.load(folder)
        if self.featurizer.n_pca != configured:
            raise RerankerError(f"{folder} was trained with {self.featurizer.n_pca} PCA features, but the config says {configured}: retrain it")
        path = folder / MODEL_FILES[self.name]
        if self.name == "xgboost":
            model = self.model_factory(self.seed)
            model.load_model(path)
        else:
            model = joblib.load(path)
        self.model = model
        self.seed = int(params["seed"])
        self.holdout_companies = list(params["holdout_companies"])
        self.fit_info = dict(params["fit_info"])
        self.importance = [(f, float(v)) for f, v in params["importance"]]


# --- building from the config ---------------------------------------------------------------------------


def _build(name: str, factory: Callable[[int], Any], config: Config, retriever: Retriever) -> PairwiseReranker:
    chart = load_standard_chart(config.data.chart_path, expected=config.data.expected_groups)
    featurizer = pair_features.PairFeaturizer(retriever, chart, n_pca=config.models.mapping.n_pca)
    return PairwiseReranker(name, featurizer, factory)


@register("xgboost")
def _xgboost(config: Config, retriever: Retriever) -> PairwiseReranker:
    return _build("xgboost", xgboost_model, config, retriever)


@register("random_forest")
def _random_forest(config: Config, retriever: Retriever) -> PairwiseReranker:
    return _build("random_forest", random_forest_model, config, retriever)
