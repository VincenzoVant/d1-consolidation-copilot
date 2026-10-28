from random import Random

import pytest

from d1.config import load_config
from d1.datagen.rule_charts import SUB_QUALIFIERS, generate_rule_company
from d1.datagen.standard_chart import load_standard_chart


@pytest.fixture
def mini_chart(fixtures_dir):
    return load_standard_chart(fixtures_dir / "mini_groups.csv")


@pytest.fixture(scope="module")
def real_chart():
    return load_standard_chart(load_config().data.chart_path, expected=190)


def _company(chart, seed=3, n=5, language="nl", **kw):
    return generate_rule_company("dev-c00", "dev", "retail", language, chart, n, Random(seed), **kw)


def test_exact_count_unique_codes_valid_targets(mini_chart):
    rows = _company(mini_chart)
    assert len(rows) == 5
    assert len({r.code for r in rows}) == 5
    assert {r.target_code for r in rows} <= {a.code for a in mini_chart}


def test_metadata_fields(mini_chart):
    rows = _company(mini_chart)
    assert [r.account_id for r in rows] == [f"dev-c00-{i:03d}" for i in range(5)]
    assert all(r.company_id == "dev-c00" and r.split == "dev" and r.industry == "retail" for r in rows)
    assert all(r.source == "rules" and r.language == "nl" for r in rows)


def test_same_seed_same_output_different_seed_differs(real_chart):
    assert _company(real_chart, seed=3, n=50) == _company(real_chart, seed=3, n=50)
    assert _company(real_chart, seed=3, n=50) != _company(real_chart, seed=4, n=50)


def test_80_percent_distinct_groups_rest_sub_accounts(real_chart):
    rows = _company(real_chart, n=50)
    base = [r for r in rows if "sub_account" not in r.noise]
    subs = [r for r in rows if "sub_account" in r.noise]
    assert len(base) == 40 and len({r.target_code for r in base}) == 40
    assert len(subs) == 10


def test_sub_accounts_reuse_their_base_name(real_chart):
    rows = _company(real_chart, n=50)
    base_name = {r.target_code: r.name for r in rows if "sub_account" not in r.noise}
    for sub in (r for r in rows if "sub_account" in r.noise):
        assert sub.target_code in base_name
        assert sub.name.startswith(base_name[sub.target_code] + " - ")


def test_sub_account_names_unique_within_group(real_chart):
    rows = _company(real_chart, n=50, seed=8)
    names = [r.name for r in rows if "sub_account" in r.noise]
    assert len(names) == len(set(names))


def test_pcmn_codes_extend_the_target_code(real_chart):
    rows = _company(real_chart, n=50, code_scheme="pcmn")
    assert all(r.code_scheme == "pcmn" and len(r.code) == 6 and r.code.startswith(r.target_code) for r in rows)
    assert all(r.code.endswith("000") for r in rows if "sub_account" not in r.noise)


def test_legacy_codes_never_reveal_the_target(real_chart):
    for seed in range(20):
        rows = _company(real_chart, n=50, seed=seed, code_scheme="legacy")
        for r in rows:
            assert r.code_scheme == "legacy" and "legacy_code" in r.noise
            assert len(r.code) == 4 and not r.code.startswith(r.target_code)


def test_rows_sorted_by_code(real_chart):
    for scheme in ("pcmn", "legacy"):
        codes = [r.code for r in _company(real_chart, n=50, code_scheme=scheme)]
        assert codes == sorted(codes)


def test_code_scheme_drawn_when_not_given(real_chart):
    schemes = {_company(real_chart, n=10, seed=s)[0].code_scheme for s in range(20)}
    assert schemes == {"pcmn", "legacy"}


def test_single_language_company_uses_that_language(mini_chart):
    rows = _company(mini_chart, language="fr", n=6, seed=1)
    chart = {a.code: a for a in mini_chart}
    # Before noise the name is the French one; noise can change it, so compare
    # only rows that came through without any noise.
    clean = [r for r in rows if not r.noise]
    assert all(r.name == chart[r.target_code].name_fr for r in clean)


def test_mixed_company_marks_rows_mixed_and_has_bilingual_names(real_chart):
    rows = _company(real_chart, language="mixed", n=50, seed=5)
    assert all(r.language == "mixed" for r in rows)
    assert any("bilingual" in r.noise and " / " in r.name for r in rows)


def test_no_bilingual_label_when_nl_and_fr_names_are_equal(real_chart):
    same = [a for a in real_chart if a.name_nl == a.name_fr]
    assert same  # e.g. 212 Goodwill
    for seed in range(30):
        for r in _company(same, language="mixed", n=min(len(same), 5), seed=seed):
            assert "bilingual" not in r.noise


def test_qualifiers_exist_for_every_language():
    assert set(SUB_QUALIFIERS) == {"nl", "fr", "en"}
    assert all(len(q) >= 4 for q in SUB_QUALIFIERS.values())


def test_too_many_accounts_raises(mini_chart):
    with pytest.raises(ValueError, match="only 6 groups"):
        _company(mini_chart, n=50)
