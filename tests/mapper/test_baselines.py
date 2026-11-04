from random import Random

import pytest

from d1.config import load_config
from d1.datagen.noise import strip_accents
from d1.datagen.rule_charts import generate_rule_company
from d1.mapper.methods import METHOD_REGISTRY
from d1.mapper.methods.base import AccountIn, accounts_from_rows
from d1.mapper.methods.baselines import EmbeddingsMethod, FuzzyMethod
from d1.mapper.retrieve import Candidate


def _cands(*pairs):
    return [Candidate(code=c, score=s) for c, s in pairs]


# --- the account input ----------------------------------------------------------------------


def test_accounts_from_rows_keep_order_and_add_neighbours(chart):
    rows = generate_rule_company("dev-c000", "dev", "retail", "nl", chart, 10, Random(0), code_scheme="legacy")
    accounts = accounts_from_rows(rows, n_neighbours=2)
    assert [a.account_id for a in accounts] == [r.account_id for r in rows]
    assert accounts[0].neighbours == [rows[1].name, rows[2].name]  # nothing before the first account
    assert accounts[5].neighbours == [rows[3].name, rows[4].name, rows[6].name, rows[7].name]
    assert accounts[5].code == rows[5].code and accounts[5].company_id == "dev-c000"


def test_neighbours_stay_within_one_company(chart):
    rows = []
    for c in range(2):
        rows += generate_rule_company(f"dev-c00{c}", "dev", "retail", "nl", chart, 5, Random(c), code_scheme="legacy")
    accounts = accounts_from_rows(rows, n_neighbours=2)
    assert all(n in [r.name for r in rows[:5]] for n in accounts[4].neighbours)  # last of company 0
    assert all(n in [r.name for r in rows[5:]] for n in accounts[5].neighbours)  # first of company 1


# --- embeddings baseline: take the shortlist's number 1 ---------------------------------------


def test_embeddings_method_picks_top1():
    method = EmbeddingsMethod()
    (out,) = method.score([AccountIn(account_id="a1", name="Leveranciers")], [_cands(("440", 0.8), ("400", 0.7), ("441", 0.6))])
    assert out.chosen_code == "440" and out.status == "ok"
    assert out.scores == {"440": 0.8, "400": 0.7, "441": 0.6}
    assert list(out.scores) == ["440", "400", "441"]  # in shortlist order


def test_embeddings_method_without_candidates_says_none():
    (out,) = EmbeddingsMethod().score([AccountIn(account_id="a1", name="x")], [[]])
    assert out.chosen_code is None and out.status == "none" and out.scores == {}


def test_embeddings_method_needs_one_shortlist_per_account():
    with pytest.raises(ValueError, match="2 accounts but 1 shortlists"):
        EmbeddingsMethod().score([AccountIn(account_id="a", name="x"), AccountIn(account_id="b", name="y")], [[]])


# --- fuzzy baseline: letters only, over all 190 groups --------------------------------------


def test_fuzzy_exact_name_first(chart):
    method = FuzzyMethod(chart, k=10)
    names = ["Leveranciers", "Fournisseurs", "Suppliers"]
    outputs = method.score([AccountIn(account_id=n, name=n) for n in names], [[] for _ in names])
    for out in outputs:
        assert out.chosen_code == "440" and out.scores["440"] == pytest.approx(1.0)
        assert len(out.scores) == 10
        assert list(out.scores.values()) == sorted(out.scores.values(), reverse=True)


def test_fuzzy_handles_accents_and_case(chart):
    a = next(a for a in chart if a.code == "200")  # "Frais d'établissement"
    messy = strip_accents(a.name_fr).upper()
    (out,) = FuzzyMethod(chart).score([AccountIn(account_id="x", name=messy)], [[]])
    assert out.chosen_code == "200" and out.scores["200"] == pytest.approx(1.0)


def test_fuzzy_ignores_the_embedding_shortlist(chart):
    method = FuzzyMethod(chart)
    account = AccountIn(account_id="x", name="Leveranciers")
    (with_list,) = method.score([account], [_cands(("100", 0.9))])
    (without,) = method.score([account], [[]])
    assert with_list.scores == without.scores and with_list.chosen_code == "440"


# --- shared behaviour ------------------------------------------------------------------------


def test_registry_builds_both_baselines():
    config = load_config()
    assert isinstance(METHOD_REGISTRY["embeddings"](config), EmbeddingsMethod)
    assert isinstance(METHOD_REGISTRY["fuzzy"](config), FuzzyMethod)
    assert not METHOD_REGISTRY["fuzzy"](config).needs_training


def test_baselines_save_and_load(chart, tmp_path):
    EmbeddingsMethod().save(tmp_path / "embeddings")
    FuzzyMethod(chart, k=7).save(tmp_path / "fuzzy")
    embeddings, fuzzy = EmbeddingsMethod(), FuzzyMethod(chart)
    embeddings.load(tmp_path / "embeddings")
    fuzzy.load(tmp_path / "fuzzy")
    assert fuzzy.k == 7 and fuzzy.version() == FuzzyMethod(chart, k=7).version()
    assert embeddings.version() == EmbeddingsMethod().version()


def test_loading_another_methods_artifacts_raises(chart, tmp_path):
    EmbeddingsMethod().save(tmp_path / "x")
    with pytest.raises(ValueError, match="embeddings"):
        FuzzyMethod(chart).load(tmp_path / "x")
