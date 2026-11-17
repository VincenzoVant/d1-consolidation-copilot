import numpy as np
import pytest

from d1.datagen.schema import CompanyAccount
from d1.mapper.features import (
    PAIR_FEATURES, FeaturizerError, PairFeaturizer, account_features, lexical_features, training_pairs,
)  # fmt: skip
from d1.mapper.methods.base import AccountIn
from d1.mapper.retrieve import Candidate, Retriever

COL = {name: i for i, name in enumerate(PAIR_FEATURES)}


def _row(i: int, name: str, target: str, code: str = "", split: str = "train", company: str = "c0") -> CompanyAccount:
    return CompanyAccount(
        account_id=f"{split}-{company}-{i:03d}", company_id=f"{split}-{company}", industry="retail", split=split,
        source="rules", language="nl", code_scheme="pcmn" if code else "legacy", code=code or f"{1000 + i}",
        name=name, noise=[], target_code=target,
    )  # fmt: skip


def _train_rows(chart, n: int = 60) -> list[CompanyAccount]:
    """Rows named after their own group (in turn en, nl, fr), so every target is easy to find."""
    rows = []
    for i, a in enumerate(chart[:n]):
        name = [a.name_en, a.name_nl, a.name_fr][i % 3]
        rows.append(_row(i, name, a.code, code=f"{a.code}000", company=f"c{i % 6}"))
    return rows


@pytest.fixture
def retriever(chart, fake_embedder):
    return Retriever(fake_embedder, chart)


@pytest.fixture
def fitted(retriever, chart):
    return PairFeaturizer(retriever, chart, n_pca=4).fit(_train_rows(chart))


def _one(featurizer, retriever, name: str, code: str | None = None, k: int = 10) -> tuple[np.ndarray, list[Candidate]]:
    (shortlist,) = retriever.top_k([name], k)
    (matrix,) = featurizer.transform([AccountIn(account_id="a", name=name, code=code)], [shortlist])
    return matrix, shortlist


# --- the feature list ---------------------------------------------------------------------------


def test_feature_names_are_fixed_and_complete():
    assert PAIR_FEATURES[:11] == ["cosine", "rank", "gap_to_top", "parent_cosine", "fuzzy_wratio", "token_jaccard",
                                  "code_prefix_match", "class_digit_match", "name_len", "n_digits", "is_bilingual"]  # fmt: skip
    assert PAIR_FEATURES[11:] == [f"emb_pca_{i}" for i in range(16)]
    assert len(set(PAIR_FEATURES)) == 27


def test_feature_matrix_shape_and_order(retriever, chart):
    featurizer = PairFeaturizer(retriever, chart, n_pca=4).fit(_train_rows(chart))
    assert featurizer.feature_names == PAIR_FEATURES[:11] + [f"emb_pca_{i}" for i in range(4)]
    names = ["Leveranciers", "Huur kantoor Gent", "Bank zichtrekening"]
    shortlists = retriever.top_k(names, 10)
    matrices = featurizer.transform([AccountIn(account_id=str(i), name=n) for i, n in enumerate(names)], shortlists)
    assert len(matrices) == 3
    for m, shortlist in zip(matrices, shortlists):
        assert m.shape == (10, 15) and m.dtype == np.float32
        assert np.isfinite(m).all()
        assert list(m[:, COL["cosine"]]) == pytest.approx([c.score for c in shortlist])  # rows follow the shortlist


def test_rank_and_gap_consistent_with_cosine(fitted, retriever):
    m, shortlist = _one(fitted, retriever, "Fournisseurs")
    assert list(m[:, COL["rank"]]) == list(range(1, 11))
    assert m[:, COL["gap_to_top"]] == pytest.approx(shortlist[0].score - m[:, COL["cosine"]], abs=1e-6)
    assert m[0, COL["gap_to_top"]] == 0 and (m[:, COL["gap_to_top"]] >= 0).all()


def test_parent_cosine_compares_the_account_with_the_parent_group(fitted, retriever, chart, fake_embedder):
    m, shortlist = _one(fitted, retriever, "Leveranciers")
    by_code = {a.code: a for a in chart}
    account = fake_embedder.encode(["Leveranciers"], "query")[0]
    for row, c in zip(m, shortlist):
        parent = fake_embedder.encode([fitted.parent_text(by_code[c.code])], "passage")[0]
        assert row[COL["parent_cosine"]] == pytest.approx(float(account @ parent), abs=1e-5)


# --- letters ------------------------------------------------------------------------------------


def test_lexical_features_reward_the_official_name(chart):
    suppliers = next(a for a in chart if a.code == "440")
    exact = lexical_features("Leveranciers", suppliers)
    assert exact == {"fuzzy_wratio": 1.0, "token_jaccard": 1.0}  # the Dutch official name, in any case or accent
    assert lexical_features("LEVERANCIERS", suppliers) == exact
    unrelated = lexical_features("Huur kantoor Gent", suppliers)
    assert unrelated["token_jaccard"] == 0.0 and unrelated["fuzzy_wratio"] < 0.5


def test_token_jaccard_takes_the_best_language(chart):
    suppliers = next(a for a in chart if a.code == "440")
    # "fournisseurs" is one of the two words; the French name is the single word "Fournisseurs"
    assert lexical_features("Fournisseurs divers", suppliers)["token_jaccard"] == pytest.approx(0.5)


# --- the account code -----------------------------------------------------------------------------


def test_code_prefix_match(fitted, retriever):
    m, shortlist = _one(fitted, retriever, "Leveranciers", code="440000")
    codes = [c.code for c in shortlist]
    assert "440" in codes
    for row, code in zip(m, codes):
        assert row[COL["code_prefix_match"]] == (1.0 if code == "440" else 0.0)
        assert row[COL["class_digit_match"]] == (1.0 if code[0] == "4" else 0.0)


def test_no_code_means_no_code_signal(fitted, retriever):
    m, _ = _one(fitted, retriever, "Leveranciers", code=None)
    assert (m[:, COL["code_prefix_match"]] == 0).all() and (m[:, COL["class_digit_match"]] == 0).all()
    assert (m[:, COL["n_digits"]] == 0).all()


def test_account_features():
    assert account_features("Terug te vorderen btw / TVA à récupérer", "411000") == {"name_len": 39, "n_digits": 6, "is_bilingual": 1.0}
    assert account_features("Huur", "3130") == {"name_len": 4, "n_digits": 4, "is_bilingual": 0.0}
    assert account_features("Staatsbon / OLO", None)["is_bilingual"] == 1.0  # two parts of 3+ letters
    assert account_features("BTW 6/12/21%", "A-12")["is_bilingual"] == 0.0  # a slash between numbers is not two languages
    assert account_features("BTW 6/12/21%", "A-12")["n_digits"] == 2


# --- the embedding features, fitted on train only --------------------------------------------------


def test_pca_fitted_on_train_only(retriever, chart):
    featurizer = PairFeaturizer(retriever, chart, n_pca=4)
    (shortlist,) = retriever.top_k(["Leveranciers"], 10)
    with pytest.raises(FeaturizerError, match="fit"):
        featurizer.transform([AccountIn(account_id="a", name="Leveranciers")], [shortlist])
    dev_rows = [r.model_copy(update={"split": "dev"}) for r in _train_rows(chart)]
    with pytest.raises(FeaturizerError, match="train"):
        featurizer.fit(dev_rows)


def test_pca_depends_only_on_the_rows_it_was_fitted_on(retriever, chart):
    rows = _train_rows(chart)
    a = PairFeaturizer(retriever, chart, n_pca=4).fit(rows)
    b = PairFeaturizer(retriever, chart, n_pca=4).fit(rows)
    c = PairFeaturizer(retriever, chart, n_pca=4).fit(rows[:40])
    assert np.array_equal(a.components, b.components)  # same rows, same result
    assert len(a.explained_variance) == 4 and 0 < sum(a.explained_variance) <= 1
    assert not np.allclose(a.components, c.components)


def test_pca_features_are_the_projection_of_the_product(fitted, retriever, fake_embedder):
    m, shortlist = _one(fitted, retriever, "Leveranciers")
    account = fake_embedder.encode(["Leveranciers"], "query")[0]
    index = {code: i for i, code in enumerate(retriever.codes)}
    for row, c in zip(m, shortlist):
        product = account * retriever.standard_embeddings[index[c.code]]
        expected = (product - fitted.mean) @ fitted.components.T
        assert row[COL["emb_pca_0"]:] == pytest.approx(expected, abs=1e-5)


def test_no_pca_features(retriever, chart, tmp_path):
    featurizer = PairFeaturizer(retriever, chart, n_pca=0).fit(_train_rows(chart))
    assert featurizer.feature_names == PAIR_FEATURES[:11] and featurizer.explained_variance == []
    m, _ = _one(featurizer, retriever, "Leveranciers", code="440000")
    assert m.shape == (10, 11)
    featurizer.save(tmp_path)
    b, _ = _one(PairFeaturizer(retriever, chart, n_pca=0).load(tmp_path), retriever, "Leveranciers", code="440000")
    assert np.array_equal(m, b)


def test_too_few_rows_for_the_pca(retriever, chart):
    with pytest.raises(FeaturizerError, match="at least"):
        PairFeaturizer(retriever, chart, n_pca=16).fit(_train_rows(chart, n=10))


def test_batches_give_the_same_features(fitted, retriever):
    names = [a for a in ["Leveranciers", "Huur", "Klanten", "Bank", "Lonen"] * 3]
    accounts = [AccountIn(account_id=str(i), name=n) for i, n in enumerate(names)]
    shortlists = retriever.top_k(names, 10)
    whole = fitted.transform(accounts, shortlists)
    fitted.batch_size = 4
    in_batches = fitted.transform(accounts, shortlists)
    assert all(np.array_equal(x, y) for x, y in zip(whole, in_batches))


def test_empty_shortlist_and_length_mismatch(fitted):
    (m,) = fitted.transform([AccountIn(account_id="a", name="x")], [[]])
    assert m.shape == (0, len(fitted.feature_names))
    with pytest.raises(ValueError, match="shortlists"):
        fitted.transform([AccountIn(account_id="a", name="x")], [])


def test_unknown_candidate_code_raises(fitted):
    with pytest.raises(FeaturizerError, match="999"):
        fitted.transform([AccountIn(account_id="a", name="x")], [[Candidate(code="999", score=0.5)]])


# --- saving ----------------------------------------------------------------------------------------


def test_save_load_gives_the_same_features(fitted, retriever, chart, tmp_path):
    fitted.save(tmp_path)
    loaded = PairFeaturizer(retriever, chart).load(tmp_path)
    assert loaded.feature_names == fitted.feature_names
    assert loaded.explained_variance == fitted.explained_variance
    a, _ = _one(fitted, retriever, "Leveranciers", code="440000")
    b, _ = _one(loaded, retriever, "Leveranciers", code="440000")
    assert np.array_equal(a, b)


def test_load_refuses_another_embedder_or_group_text(fitted, chart, tmp_path, FakeEmbedder):
    fitted.save(tmp_path)
    other = Retriever(FakeEmbedder("fake/other"), chart)
    with pytest.raises(FeaturizerError, match="fake/other"):
        PairFeaturizer(other, chart).load(tmp_path)
    described = Retriever(FakeEmbedder(), chart, descriptions={a.code: "x" for a in chart}, group_text="average")
    with pytest.raises(FeaturizerError, match="group_text"):
        PairFeaturizer(described, chart).load(tmp_path)


# --- training pairs --------------------------------------------------------------------------------


def test_training_pairs_one_positive_per_covered_account(fitted, retriever, chart):
    rows = _train_rows(chart, n=30) + [_row(99, "Iets heel anders zonder betekenis", "792")]
    shortlists = retriever.top_k([r.name for r in rows], 10)
    X, y, group = training_pairs(rows, shortlists, fitted)
    assert X.shape == (sum(len(s) for s in shortlists), len(fitted.feature_names))
    assert y.dtype == np.int8 and set(np.unique(y)) <= {0, 1}
    for i, (row, shortlist) in enumerate(zip(rows, shortlists)):
        covered = row.target_code in [c.code for c in shortlist]
        assert y[group == i].sum() == (1 if covered else 0)  # not covered: negatives only
    assert not any(c.code == "792" for c in shortlists[-1])  # the nonsense row is the uncovered one
    assert list(np.unique(group)) == list(range(len(rows)))
