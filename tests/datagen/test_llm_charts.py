import json
from random import Random

import pytest

from d1.config import load_config
from d1.datagen.llm_charts import (
    LLM_CHART_SCHEMA,
    PROMPT_VERSION,
    FrozenOutputMissing,
    LLMOutputError,
    build_prompt,
    fetch_llm_company,
    load_llm_company,
    parse_chart,
    select_llm_groups,
)
from d1.datagen.standard_chart import load_standard_chart
from d1.llm import LLMError, LLMResponse


@pytest.fixture(scope="module")
def chart():
    return load_standard_chart(load_config().data.chart_path, expected=190)


@pytest.fixture
def groups(chart):
    return select_llm_groups(chart, 10, Random(1))  # 8 groups for 10 accounts


def _answer(groups, n):
    """A well-formed answer: one account per group, then second accounts until there are n."""
    rows = [{"target_code": g.code, "code": f"{g.code}0", "name": f"Rekening {g.name_nl}"} for g in groups]
    rows += [{"target_code": g.code, "code": f"{g.code}1", "name": f"Rekening {g.name_nl} - Gent"} for g in groups]
    return {"accounts": rows[:n]}


class FakeBackend:
    def __init__(self, structured=None, model="claude-opus-5-5", raises=None):
        self.structured, self.model, self.raises = structured, model, raises
        self.calls = []

    def complete(self, prompt, schema, model, system, timeout_s=120.0):
        self.calls.append({"prompt": prompt, "schema": schema, "model": model, "system": system, "timeout_s": timeout_s})
        if self.raises:
            raise self.raises
        return LLMResponse(structured=self.structured, model=self.model, input_tokens=900, output_tokens=700, cost_usd=0.07, latency_ms=1)


def _fetch(backend, groups, frozen_dir, company_id="test-c08", split="test", n=10, code_scheme="llm"):
    return fetch_llm_company(
        backend,
        generator_model="claude-opus-5-5",
        mapping_model="claude-haiku-5-5",
        company_id=company_id,
        split=split,
        industry="construction",
        language="nl",
        code_scheme=code_scheme,
        groups=groups,
        n_accounts=n,
        frozen_dir=frozen_dir,
    )


# --- choosing groups and writing the prompt ----------------------------------


def test_select_llm_groups_is_seeded_sorted_and_80_percent(chart):
    a = select_llm_groups(chart, 50, Random(4))
    assert a == select_llm_groups(chart, 50, Random(4)) != select_llm_groups(chart, 50, Random(5))
    assert len(a) == 40 and len({g.code for g in a}) == 40
    assert [g.code for g in a] == sorted(g.code for g in a)


def test_select_llm_groups_too_many_raises(chart):
    with pytest.raises(ValueError, match="only 190 groups"):
        select_llm_groups(chart, 300, Random(0))


def test_build_prompt_lists_every_group_and_language(groups):
    prompt = build_prompt("construction", "nl", groups, 10)
    for g in groups:
        assert g.code in prompt and g.name_en in prompt and g.name_nl in prompt
    assert "Dutch" in prompt and "construction" in prompt and "exactly 10 accounts" in prompt
    assert "{" not in prompt and "}" not in prompt  # every placeholder filled


def test_build_prompt_mixed_language(groups):
    prompt = build_prompt("retail", "mixed", groups, 10)
    assert "Dutch, French and English" in prompt and "Brussels" in prompt
    assert all(f"{g.name_nl} / {g.name_fr}" in prompt for g in groups)


def test_schema_forbids_extra_fields():
    item = LLM_CHART_SCHEMA["properties"]["accounts"]["items"]
    assert LLM_CHART_SCHEMA["additionalProperties"] is False and item["additionalProperties"] is False
    assert set(item["required"]) == {"target_code", "code", "name"}


# --- parsing the answer --------------------------------------------------------


def test_parse_drops_out_of_set_empty_and_duplicate_rows():
    response = {
        "accounts": [
            {"target_code": "604", "code": "6040", "name": "Aankopen bouwmateriaal"},
            {"target_code": "999", "code": "9990", "name": "Not a group we asked for"},
            {"target_code": "604", "code": "6041", "name": "   "},
            {"target_code": "440", "code": "6040", "name": "Duplicate code, dropped"},
            {"target_code": "440", "code": " 4400 ", "name": " Leveranciers "},
        ]
    }
    rows = parse_chart(response, {"604", "440"})
    assert [(r.target_code, r.code, r.name) for r in rows] == [
        ("604", "6040", "Aankopen bouwmateriaal"),
        ("440", "4400", "Leveranciers"),
    ]


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"accounts": "not a list"},
        {"accounts": [{"target_code": "604", "code": "6040"}]},  # no name
        {"accounts": [{"target_code": "604", "code": "6040", "name": "x", "extra": 1}]},
    ],
)
def test_parse_wrong_shape_raises(response):
    with pytest.raises(LLMOutputError, match="shape"):
        parse_chart(response, {"604"})


# --- fetching and freezing -----------------------------------------------------


def test_fetch_writes_frozen_file_then_never_calls_again(groups, tmp_path):
    fake = FakeBackend(_answer(groups, 10))
    path = _fetch(fake, groups, tmp_path)
    assert _fetch(fake, groups, tmp_path) == path
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["timeout_s"] == 300 and call["schema"] == LLM_CHART_SCHEMA

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert path == tmp_path / "test-c08.json"
    assert saved["company_id"] == "test-c08" and saved["split"] == "test"
    assert saved["industry"] == "construction" and saved["language"] == "nl" and saved["code_scheme"] == "llm"
    assert saved["requested_model"] == saved["served_model"] == "claude-opus-5-5"
    assert saved["backend"] == "FakeBackend" and saved["prompt_version"] == PROMPT_VERSION
    assert len(saved["prompt_sha256"]) == 64
    assert saved["group_codes"] == [g.code for g in groups] and saved["n_accounts"] == 10
    assert saved["usage"] == {"input_tokens": 900, "output_tokens": 700, "cost_usd": 0.07, "latency_ms": 1}
    assert saved["response"] == _answer(groups, 10)
    assert list(tmp_path.iterdir()) == [path]  # no temporary file left behind


def test_fetch_backend_error_raises_and_writes_nothing(groups, tmp_path):
    with pytest.raises(LLMOutputError, match="test-c08.*timed out"):
        _fetch(FakeBackend(raises=LLMError("claude CLI timed out after 300s")), groups, tmp_path)
    assert not any(tmp_path.iterdir())


def test_fetch_refuses_answer_from_the_mapping_model(groups, tmp_path):
    with pytest.raises(LLMOutputError, match="mapping model"):
        _fetch(FakeBackend(_answer(groups, 10), model="claude-haiku-5-5"), groups, tmp_path)
    assert not any(tmp_path.iterdir())


def test_fetch_too_few_rows_raises_and_writes_nothing(groups, tmp_path):
    with pytest.raises(LLMOutputError, match="only 5 usable accounts"):
        _fetch(FakeBackend(_answer(groups, 5)), groups, tmp_path)
    assert not any(tmp_path.iterdir())


# --- loading frozen files --------------------------------------------------------


def test_load_returns_llm_company_accounts(groups, tmp_path):
    _fetch(FakeBackend(_answer(groups, 12)), groups, tmp_path, n=12)
    rows = load_llm_company(tmp_path, "test-c08", "test", 10)  # file has 12, the split asks for 10
    assert len(rows) == 10
    assert [r.account_id for r in rows] == [f"test-c08-{i:03d}" for i in range(10)]
    assert all(r.source == "llm" and r.code_scheme == "llm" and r.noise == ["llm"] for r in rows)
    assert all(r.industry == "construction" and r.language == "nl" and r.split == "test" for r in rows)
    assert {r.target_code for r in rows} <= {g.code for g in groups}
    # The llm scheme keeps the codes the LLM wrote.
    assert [r.code for r in rows] == [a["code"] for a in _answer(groups, 12)["accounts"][:10]]


def test_load_cuts_surplus_second_accounts_not_whole_groups(groups, tmp_path):
    # The real LLM writes a few accounts too many, in group order. Cutting the tail
    # would drop the highest groups (revenue); every group must keep its account.
    pairs = [
        {"target_code": g.code, "code": f"{g.code}{k}", "name": f"Rekening {g.name_nl} {k}"}
        for g in groups
        for k in (0, 1)
    ]
    _fetch(FakeBackend({"accounts": pairs}), groups, tmp_path, n=16)
    rows = load_llm_company(tmp_path, "test-c08", "test", 10)
    assert len(rows) == 10
    assert {r.target_code for r in rows} == {g.code for g in groups}
    assert [r.code for r in rows] == sorted(r.code for r in rows)  # the LLM's order is kept


def test_load_legacy_scheme_replaces_codes_so_they_never_reveal_the_target(groups, tmp_path):
    # _answer's codes start with the target, just as the real trial's did.
    _fetch(FakeBackend(_answer(groups, 10)), groups, tmp_path, code_scheme="legacy")
    rows = load_llm_company(tmp_path, "test-c08", "test", 10)
    assert all(r.code_scheme == "legacy" and r.noise == ["llm", "legacy_code"] for r in rows)
    assert all(len(r.code) == 4 and not r.code.startswith(r.target_code) for r in rows)
    assert len({r.code for r in rows}) == 10
    # Only the codes change: names and targets are the LLM's, and reloading gives the same codes.
    answer = _answer(groups, 10)["accounts"]
    assert [(r.name, r.target_code) for r in rows] == [(a["name"], a["target_code"]) for a in answer]
    assert rows == load_llm_company(tmp_path, "test-c08", "test", 10)


def test_load_missing_file_raises_with_command_hint(tmp_path):
    with pytest.raises(FrozenOutputMissing, match=r"test-c03\.json.*fetch-llm --split test"):
        load_llm_company(tmp_path, "test-c03", "test", 25)


def test_load_too_few_rows_raises(groups, tmp_path):
    path = _fetch(FakeBackend(_answer(groups, 10)), groups, tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["response"]["accounts"] = saved["response"]["accounts"][:3]
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(LLMOutputError, match="only 3 usable accounts"):
        load_llm_company(tmp_path, "test-c08", "test", 10)


def test_load_wrong_split_raises(groups, tmp_path):
    _fetch(FakeBackend(_answer(groups, 10)), groups, tmp_path)
    with pytest.raises(LLMOutputError, match="split test, not dev"):
        load_llm_company(tmp_path, "test-c08", "dev", 10)
