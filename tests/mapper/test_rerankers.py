from functools import partial
from random import Random

import numpy as np
import pytest

from d1.config import load_config
from d1.datagen.schema import CompanyAccount
from d1.mapper.features import PairFeaturizer
from d1.mapper.methods import METHOD_REGISTRY
from d1.mapper.methods.base import accounts_from_rows
from d1.mapper.methods.baselines import EmbeddingsMethod
from d1.mapper.methods.rerankers import (
    PairwiseReranker, RerankerError, company_holdout, random_forest_model, top1_share, xgboost_model,
)  # fmt: skip
from d1.mapper.retrieve import Candidate, Retriever

# Small models, so every test runs in about a second.
FACTORIES = {
    "xgboost": partial(xgboost_model, n_estimators=80),
    "random_forest": partial(random_forest_model, n_estimators=40),
}


def _rows(chart, split: str, n_companies: int, per_company: int = 30, seed: int = 0) -> list[CompanyAccount]:
    """Accounts whose *name* says nothing ("Rekening 3 17") but whose code starts with the right group.

    Only the code gives the answer away, so a model that learns from the features
    must learn code_prefix_match.
    """
    rng = Random(seed)
    codes = [a.code for a in chart]
    rows = []
    for c in range(n_companies):
        for i in range(per_company):
            target = rng.choice(codes)
            rows.append(CompanyAccount(
                account_id=f"{split}-c{c:02d}-{i:03d}", company_id=f"{split}-c{c:02d}", industry="retail", split=split,
                source="rules", language="nl", code_scheme="pcmn", code=f"{target}{i:03d}", name=f"Rekening {c} {i}",
                noise=[], target_code=target,
            ))  # fmt: skip
    return rows


def _shortlists(rows, chart, k: int = 10, seed: int = 1) -> list[list[Candidate]]:
    """The right group at a random place among k, under made-up falling scores: the embeddings' number 1 is luck."""
    rng = Random(seed)
    codes = [a.code for a in chart]
    out = []
    for r in rows:
        others = rng.sample([c for c in codes if c != r.target_code], k - 1)
        place = rng.randrange(k)
        ordered = others[:place] + [r.target_code] + others[place:]
        scores = sorted((rng.uniform(0.3, 0.9) for _ in range(k)), reverse=True)
        out.append([Candidate(code=c, score=s) for c, s in zip(ordered, scores)])
    return out


@pytest.fixture
def retriever(chart, fake_embedder):
    return Retriever(fake_embedder, chart)


@pytest.fixture
def data(chart):
    train = _rows(chart, "train", n_companies=10, seed=0)
    dev = _rows(chart, "dev", n_companies=3, seed=1)
    return train, _shortlists(train, chart, seed=2), dev, _shortlists(dev, chart, seed=3)


def _reranker(name, retriever, chart, n_pca=4, factory=None) -> PairwiseReranker:
    return PairwiseReranker(name, PairFeaturizer(retriever, chart, n_pca=n_pca), factory or FACTORIES[name])


def _top1(outputs, rows) -> float:
    return float(np.mean([o.chosen_code == r.target_code for o, r in zip(outputs, rows)]))


# --- fitting and scoring ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["xgboost", "random_forest"])
def test_fit_then_score_shapes(name, retriever, chart, data):
    train, train_lists, dev, dev_lists = data
    method = _reranker(name, retriever, chart)
    assert method.name == name and method.needs_training
    method.fit(train, train_lists)
    outputs = method.score(accounts_from_rows(dev), dev_lists)
    assert [o.account_id for o in outputs] == [r.account_id for r in dev]
    for out, shortlist in zip(outputs, dev_lists):
        assert sorted(out.scores) == sorted(c.code for c in shortlist)  # every candidate gets a score
        assert all(0.0 <= s <= 1.0 for s in out.scores.values())  # a probability of "this is the right group"
        assert list(out.scores.values()) == sorted(out.scores.values(), reverse=True)  # best first
        assert out.chosen_code == next(iter(out.scores)) and out.status == "ok"


def test_empty_shortlist_says_none(retriever, chart, data):
    train, train_lists, dev, _ = data
    method = _reranker("xgboost", retriever, chart)
    method.fit(train, train_lists)
    (out,) = method.score(accounts_from_rows(dev[:1]), [[]])
    assert out.chosen_code is None and out.status == "none" and out.scores == {}


@pytest.mark.parametrize("name", ["xgboost", "random_forest"])
def test_learns_obvious_signal(name, retriever, chart, data):
    """When only the code gives the answer away, the reranker finds it and the embeddings' number 1 doesn't."""
    train, train_lists, dev, dev_lists = data
    method = _reranker(name, retriever, chart)
    method.fit(train, train_lists)
    accounts = accounts_from_rows(dev)
    reranked = _top1(method.score(accounts, dev_lists), dev)
    baseline = _top1(EmbeddingsMethod().score(accounts, dev_lists), dev)
    assert reranked > 0.9 and reranked > baseline + 0.5


def test_score_before_fit_raises(retriever, chart, data):
    _, _, dev, dev_lists = data
    with pytest.raises(RerankerError, match="fit"):
        _reranker("xgboost", retriever, chart).score(accounts_from_rows(dev), dev_lists)


@pytest.mark.parametrize("name", ["xgboost", "random_forest"])
def test_seeded_and_reproducible(name, retriever, chart, data):
    train, train_lists, dev, dev_lists = data
    runs = []
    for _ in range(2):
        method = _reranker(name, retriever, chart)
        method.fit(train, train_lists)
        runs.append([o.scores for o in method.score(accounts_from_rows(dev), dev_lists)])
    assert runs[0] == runs[1]


# --- the hold-out: whole train companies, never dev --------------------------------------------------


def test_company_holdout_keeps_companies_whole_and_is_seeded(chart):
    rows = _rows(chart, "train", n_companies=20, per_company=5)
    held = company_holdout(rows, share=0.1, seed=0)
    companies = {r.company_id for r, h in zip(rows, held) if h}
    assert len(companies) == 2  # 10% of 20 companies
    assert all(h == (r.company_id in companies) for r, h in zip(rows, held))  # all of a company, or none of it
    assert np.array_equal(held, company_holdout(rows, share=0.1, seed=0))
    assert not np.array_equal(held, company_holdout(rows, share=0.1, seed=1))
    assert company_holdout(rows[:10], share=0.1, seed=0).sum() == 5  # 2 companies: one is held out
    with pytest.raises(RerankerError, match="companies"):
        company_holdout(rows[:5], share=0.1, seed=0)  # one company: nothing to hold out


class SpyModel:
    """Records what it is fitted on; scores a pair by its cosine."""

    early_stopping_rounds = 5

    def fit(self, X, y, eval_set=None, verbose=None):
        self.n_fitted, self.eval_set = len(y), eval_set
        return self

    def predict_proba(self, X):
        p = np.clip(X[:, 0], 0, 1)
        return np.c_[1 - p, p]


def test_early_stopping_uses_train_holdout_only(retriever, chart, data):
    train, train_lists, _, _ = data
    spy = SpyModel()
    method = _reranker("xgboost", retriever, chart, factory=lambda seed: spy)
    method.fit(train, train_lists)
    held = company_holdout(train, seed=0)
    assert set(method.holdout_companies) == {r.company_id for r, h in zip(train, held) if h}
    assert set(method.holdout_companies) <= {r.company_id for r in train}  # train companies, not dev
    n_held_pairs = sum(len(s) for s, h in zip(train_lists, held) if h)
    (X_eval, y_eval), = spy.eval_set
    assert len(y_eval) == n_held_pairs and spy.n_fitted == sum(len(s) for s in train_lists) - n_held_pairs
    # The PCA is fitted on the same 90%: the held-out companies don't shape the features either.
    alone = PairFeaturizer(retriever, chart, n_pca=4).fit([r for r, h in zip(train, held) if not h])
    assert np.array_equal(method.featurizer.components, alone.components)


def test_model_without_early_stopping_gets_no_eval_set(retriever, chart, data):
    train, train_lists, _, _ = data
    spy = SpyModel()
    spy.early_stopping_rounds = None
    _reranker("random_forest", retriever, chart, factory=lambda seed: spy).fit(train, train_lists)
    assert spy.eval_set is None


# --- importance ------------------------------------------------------------------------------------


def test_top1_share_counts_accounts_not_pairs():
    p = np.array([0.9, 0.1, 0.2, 0.8, 0.5, 0.4])
    y = np.array([1, 0, 1, 0, 0, 0])
    group = np.array([0, 0, 1, 1, 2, 2])  # account 0 right, account 1 wrong, account 2 has no right candidate
    assert top1_share(p, y, group) == pytest.approx(1 / 3)


@pytest.mark.parametrize("name", ["xgboost", "random_forest"])
def test_importance_covers_all_features(name, retriever, chart, data):
    train, train_lists, _, _ = data
    method = _reranker(name, retriever, chart)
    method.fit(train, train_lists)
    importance = method.feature_importance()
    assert [f for f, _ in importance] != [] and sorted(f for f, _ in importance) == sorted(method.featurizer.feature_names)
    values = [v for _, v in importance]
    assert values == sorted(values, reverse=True)
    assert importance[0][0] == "code_prefix_match"  # the only feature that gives the answer away here


# --- saving -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("name, model_file", [("xgboost", "model.json"), ("random_forest", "model.joblib")])
def test_save_load_same_scores(name, model_file, retriever, chart, data, tmp_path):
    train, train_lists, dev, dev_lists = data
    method = _reranker(name, retriever, chart)
    method.fit(train, train_lists)
    method.save(tmp_path)
    assert {"method.json", "featurizer.json", "featurizer.npz", model_file} <= {p.name for p in tmp_path.iterdir()}
    loaded = _reranker(name, retriever, chart)
    loaded.load(tmp_path)
    accounts = accounts_from_rows(dev)
    assert [o.scores for o in loaded.score(accounts, dev_lists)] == [o.scores for o in method.score(accounts, dev_lists)]
    assert loaded.feature_importance() == method.feature_importance()
    assert loaded.version() == method.version()


def test_load_refuses_the_other_model(retriever, chart, data, tmp_path):
    train, train_lists, _, _ = data
    method = _reranker("xgboost", retriever, chart)
    method.fit(train, train_lists)
    method.save(tmp_path)
    with pytest.raises(ValueError, match="xgboost"):
        _reranker("random_forest", retriever, chart).load(tmp_path)


def test_load_refuses_another_pca_size(retriever, chart, data, tmp_path):
    train, train_lists, _, _ = data
    method = _reranker("xgboost", retriever, chart, n_pca=4)
    method.fit(train, train_lists)
    method.save(tmp_path)
    with pytest.raises(RerankerError, match="4 PCA features.*says 8"):
        _reranker("xgboost", retriever, chart, n_pca=8).load(tmp_path)


# --- building from the config ------------------------------------------------------------------------


def test_registry_builds_both_rerankers(retriever):
    config = load_config()
    for name in ("xgboost", "random_forest"):
        method = METHOD_REGISTRY[name](config, retriever)
        assert isinstance(method, PairwiseReranker) and method.name == name and method.needs_training
        assert method.featurizer.n_pca == config.models.mapping.n_pca


def test_version_names_the_model_and_the_features(retriever, chart):
    version = _reranker("random_forest", retriever, chart, n_pca=4).version()
    assert "random_forest" in version and "pair-features-v1" in version and "pca4" in version


def test_unknown_name_refused(retriever, chart):
    with pytest.raises(RerankerError, match="lightgbm"):
        PairwiseReranker("lightgbm", PairFeaturizer(retriever, chart, n_pca=4), FACTORIES["xgboost"])
