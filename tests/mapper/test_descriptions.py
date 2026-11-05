import json

import pytest
import yaml

from d1.config import ConfigError, load_config
from d1.llm import LLMError, LLMResponse
from d1.mapper.descriptions import (
    DescriptionError, GroupDescription, batches, build_prompt, fetch_batch, load_descriptions, parse_descriptions,
)  # fmt: skip
from d1.mapper.retrieve import Retriever, standard_text

WRITER = "claude-sonnet-5-5"
GENERATOR = "claude-opus-5-5"


def _entry(code, **changes):
    entry = {
        "code": code,
        "description": f"What is booked in group {code}.",
        "words_nl": [f"nl-{code}"],
        "words_fr": [f"fr-{code}"],
        "words_en": [f"en-{code}"],
    }
    return {**entry, **changes}


class FakeBackend:
    """Answers every batch with one well-formed entry per group in the prompt, unless told otherwise."""

    def __init__(self, model=WRITER, raises=None, drop=None):
        self.model, self.raises, self.drop = model, raises, drop
        self.calls = []

    def complete(self, prompt, schema, model, system, timeout_s=120.0):
        self.calls.append({"prompt": prompt, "model": model})
        if self.raises:
            raise self.raises
        codes = [line.split("|")[1].strip() for line in prompt.splitlines() if line.startswith("| ") and line[2:5].isdigit()]
        entries = [_entry(c) for c in codes if c != self.drop]
        return LLMResponse(structured={"groups": entries}, model=self.model, input_tokens=10, output_tokens=20, cost_usd=0.01, latency_ms=1)


def _fetch_all(chart, folder, backend=None):
    backend = backend or FakeBackend()
    for number, groups in batches(chart):
        fetch_batch(backend, writer_model=WRITER, generator_model=GENERATOR, number=number, groups=groups, frozen_dir=folder)
    return backend


# --- the prompt sees the chart only -----------------------------------------------------------


def test_batches_cover_the_chart_once_in_code_order(chart):
    parts = batches(chart, size=19)
    assert [n for n, _ in parts] == list(range(1, 11))
    codes = [g.code for _, groups in parts for g in groups]
    assert codes == sorted(a.code for a in chart) and len(codes) == 190


def test_prompt_lists_every_group_with_its_names_and_parent(chart):
    groups = chart[:5]
    prompt = build_prompt(groups)
    for g in groups:
        assert f"| {g.code} | {g.name_en} | {g.name_nl} | {g.name_fr} | {g.parent_name_en} |" in prompt
    assert "{" not in prompt  # every placeholder filled in


# --- reading the answer ---------------------------------------------------------------------


def test_parse_keeps_one_entry_per_group(chart):
    groups = chart[:3]
    parsed = parse_descriptions({"groups": [_entry(g.code) for g in groups]}, [g.code for g in groups])
    assert [p.code for p in parsed] == [g.code for g in groups]


@pytest.mark.parametrize("problem", ["missing", "unknown", "duplicate", "empty description"])
def test_parse_rejects_a_bad_answer(chart, problem):
    codes = [g.code for g in chart[:3]]
    entries = [_entry(c) for c in codes]
    if problem == "missing":
        entries = entries[:2]
    elif problem == "unknown":
        entries.append(_entry("999"))
    elif problem == "duplicate":
        entries.append(_entry(codes[0]))
    else:
        entries[0]["description"] = "  "
    with pytest.raises(DescriptionError):
        parse_descriptions({"groups": entries}, codes)


def test_text_holds_description_and_ledger_words():
    d = GroupDescription(code="620", description="Gross pay of staff.", words_nl=["brutolonen", "wedden"],
                         words_fr=["salaires bruts"], words_en=["gross wages"])  # fmt: skip
    assert d.text() == "Gross pay of staff. brutolonen, wedden; salaires bruts; gross wages"


# --- fetching: once, frozen, and never by the model that wrote the accounts ------------------


def test_fetch_freezes_the_answer_with_its_provenance(chart, tmp_path):
    (number, groups), *_ = batches(chart)
    backend = FakeBackend()
    path = fetch_batch(backend, writer_model=WRITER, generator_model=GENERATOR, number=number, groups=groups, frozen_dir=tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "batch-01.json" and backend.calls[0]["model"] == WRITER
    assert record["served_model"] == WRITER and record["group_codes"] == [g.code for g in groups]
    assert len(record["prompt_sha256"]) == 64 and len(record["response"]["groups"]) == len(groups)


def test_fetch_skips_a_batch_already_frozen(chart, tmp_path):
    _fetch_all(chart, tmp_path)
    again = _fetch_all(chart, tmp_path)
    assert again.calls == []  # frozen answers are read, never asked for again


def test_fetch_refuses_the_generator_model(chart, tmp_path):
    (number, groups), *_ = batches(chart)
    # Opus wrote the LLM accounts: its own words would match its own accounts and inflate recall.
    with pytest.raises(DescriptionError, match="generator"):
        fetch_batch(FakeBackend(model=GENERATOR), writer_model=WRITER, generator_model=GENERATOR, number=number, groups=groups, frozen_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_fetch_writes_nothing_when_the_answer_is_incomplete(chart, tmp_path):
    (number, groups), *_ = batches(chart)
    with pytest.raises(DescriptionError):
        fetch_batch(FakeBackend(drop=groups[0].code), writer_model=WRITER, generator_model=GENERATOR, number=number, groups=groups, frozen_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_fetch_failure_names_the_batch(chart, tmp_path):
    (number, groups), *_ = batches(chart)
    with pytest.raises(DescriptionError, match="batch 1"):
        fetch_batch(FakeBackend(raises=LLMError("timed out")), writer_model=WRITER, generator_model=GENERATOR, number=number, groups=groups, frozen_dir=tmp_path)


# --- loading: all 190 groups or nothing -------------------------------------------------------


def test_descriptions_cover_every_group(chart, tmp_path):
    _fetch_all(chart, tmp_path)
    texts = load_descriptions(tmp_path, chart)
    assert sorted(texts) == sorted(a.code for a in chart)
    assert texts["620"] == "What is booked in group 620. nl-620; fr-620; en-620"


def test_missing_group_raises(chart, tmp_path):
    _fetch_all(chart, tmp_path)
    (tmp_path / "batch-10.json").unlink()
    with pytest.raises(DescriptionError, match="no description"):
        load_descriptions(tmp_path, chart)


def test_unknown_code_raises(chart, tmp_path):
    _fetch_all(chart[:-1], tmp_path)  # fetched for a chart with one group more than the one loaded
    with pytest.raises(DescriptionError, match="not in the chart"):
        load_descriptions(tmp_path, chart[:-2])


def test_no_folder_names_the_command(chart, tmp_path):
    with pytest.raises(DescriptionError, match=r"python -m d1\.mapper\.descriptions"):
        load_descriptions(tmp_path / "missing", chart)


# --- the descriptions in retrieval -------------------------------------------------------------


def test_standard_text_with_and_without_description(chart):
    a = next(x for x in chart if x.code == "620")
    plain = f"{a.name_en} | {a.name_nl} | {a.name_fr} ({a.parent_name_en})"
    assert standard_text(a) == plain
    assert standard_text(a, None) == plain
    assert standard_text(a, "Gross pay. brutolonen") == plain + " — Gross pay. brutolonen"


def test_retriever_embeds_the_descriptions(chart, FakeEmbedder):
    texts = {a.code: f"description of {a.code}" for a in chart}
    plain = Retriever(FakeEmbedder(), chart)
    described = Retriever(FakeEmbedder(), chart, descriptions=texts, group_text="descriptions")
    assert plain.group_text == "names" and described.group_text == "descriptions"
    assert described.texts == [standard_text(a, texts[a.code]) for a in chart]
    assert not (plain.standard_embeddings == described.standard_embeddings).all()


def test_average_scores_are_the_mean_of_both(chart, FakeEmbedder):
    texts = {a.code: f"description of {a.code}" for a in chart}
    names = ["Brutolonen winkelpersoneel", "Leveranciers"]
    plain = Retriever(FakeEmbedder(), chart).top_k(names, len(chart))
    described = Retriever(FakeEmbedder(), chart, descriptions=texts, group_text="descriptions").top_k(names, len(chart))
    average = Retriever(FakeEmbedder(), chart, descriptions=texts, group_text="average").top_k(names, len(chart))
    for p, d, a in zip(plain, described, average):
        p, d = {c.code: c.score for c in p}, {c.code: c.score for c in d}
        assert all(c.score == pytest.approx((p[c.code] + d[c.code]) / 2, abs=1e-6) for c in a)
        assert [c.score for c in a] == sorted((c.score for c in a), reverse=True)


def test_retriever_needs_descriptions_unless_names_only(chart, FakeEmbedder):
    with pytest.raises(ValueError, match="needs descriptions"):
        Retriever(FakeEmbedder(), chart, group_text="average")
    with pytest.raises(ValueError, match="group_text"):
        Retriever(FakeEmbedder(), chart, descriptions={a.code: "x" for a in chart})  # given but not used: say so
    with pytest.raises(ValueError, match="unknown group_text"):
        Retriever(FakeEmbedder(), chart, descriptions={a.code: "x" for a in chart}, group_text="both")


def test_retriever_needs_a_description_for_every_group(chart, FakeEmbedder):
    texts = {a.code: "x" for a in chart[1:]}
    with pytest.raises(ValueError, match=chart[0].code):
        Retriever(FakeEmbedder(), chart, descriptions=texts, group_text="descriptions")


def test_cache_differs_with_descriptions(chart, tmp_path, FakeEmbedder):
    Retriever(FakeEmbedder(), chart, cache_dir=tmp_path)
    again = FakeEmbedder()
    Retriever(again, chart, cache_dir=tmp_path, descriptions={a.code: "x" for a in chart}, group_text="descriptions")
    assert again.kinds() == ["passage"]  # described vectors are never mixed up with plain ones
    assert len(list(tmp_path.glob("*.npy"))) == 2
    third = FakeEmbedder()
    Retriever(third, chart, cache_dir=tmp_path, descriptions={a.code: "x" for a in chart}, group_text="average")
    assert third.kinds() == []  # the average reuses both cached sets


# --- config -----------------------------------------------------------------------------------


def _edit_models(root, **changes):
    path = root / "config" / "models.yaml"
    data = yaml.safe_load(path.read_text())
    for key, value in changes.items():
        if key == "group_text":
            data["mapping"][key] = value
        else:
            data[key] = value
    path.write_text(yaml.safe_dump(data))


def test_config_has_the_writer_and_the_switch():
    config = load_config()
    assert config.models.description_model not in (config.models.generator_model,)
    assert config.models.mapping.group_text == "average"  # chosen on dev in notebook 12
    assert config.data.frozen_descriptions_dir.name == "group_descriptions"


def test_description_model_must_differ_from_the_generator(config_root):
    _edit_models(config_root, description_model=GENERATOR)
    with pytest.raises(ConfigError, match="description_model"):
        load_config(config_root)


def test_group_text_must_be_a_known_choice(config_root):
    _edit_models(config_root, group_text="both")
    with pytest.raises(ConfigError, match="group_text"):
        load_config(config_root)
