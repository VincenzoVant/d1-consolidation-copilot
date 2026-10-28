import json
from collections import Counter

import pytest

from d1.config import load_config
from d1.datagen.llm_charts import FrozenOutputMissing, LLMOutputError
from d1.datagen.schema import CompanyAccount, read_jsonl
from d1.datagen.splits import (
    build_all,
    company_plan,
    fetch_missing_llm,
    mark_overlaps,
    normalize_name,
)
from d1.llm import LOGIN_HINT, LLMError


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _plan(cfg, split):
    return company_plan(split, cfg.data.splits[split], cfg.data.industries, cfg.data.languages)


# --- the company plan ------------------------------------------------------------


def test_test_plan_is_balanced(cfg):
    plan = _plan(cfg, "test")
    assert len(plan) == 16
    assert Counter(c.language for c in plan) == {"nl": 4, "fr": 4, "en": 4, "mixed": 4}
    assert all(Counter(c.source for c in plan if c.language == lang) == {"rules": 2, "llm": 2} for lang in cfg.data.languages)
    assert all({c.language for c in plan if c.industry == ind} == set(cfg.data.languages) for ind in cfg.data.industries)


def test_dev_plan_is_balanced(cfg):
    plan = _plan(cfg, "dev")
    assert len(plan) == 70
    assert Counter(c.source for c in plan) == {"rules": 40, "llm": 30}
    assert all({c.source for c in plan if c.industry == ind} == {"rules", "llm"} for ind in cfg.data.industries)
    assert set(Counter(c.language for c in plan).values()) <= {17, 18}  # 70 / 4 languages


def test_train_plan_is_balanced(cfg):
    plan = _plan(cfg, "train")
    assert len(plan) == 240
    assert Counter(c.language for c in plan) == {lang: 60 for lang in cfg.data.languages}
    assert all(Counter(c.source for c in plan if c.language == lang) == {"rules": 50, "llm": 10} for lang in cfg.data.languages)


def test_plan_ids_indices_and_sizes(cfg):
    plan = _plan(cfg, "test")
    assert [c.company_id for c in plan] == [f"test-c{i:03d}" for i in range(16)]
    assert [c.index for c in plan] == list(range(16))
    assert {c.n_accounts for c in plan if c.source == "rules"} == {50}
    assert {c.n_accounts for c in plan if c.source == "llm"} == {25}


# --- overlap with names seen in train/dev -----------------------------------------------


def test_normalize_name():
    assert normalize_name("LEVERANCIERS.") == normalize_name("Leveranciers") == "leveranciers"
    assert normalize_name("  Créances / Vord.-Gent ") == "creances vord gent"


def _row(name, split="test", i=0):
    return CompanyAccount(
        account_id=f"{split}-c000-{i:03d}", company_id=f"{split}-c000", industry="retail", split=split,
        source="rules", language="nl", code_scheme="pcmn", code="440000", name=name, noise=[], target_code="440",
    )  # fmt: skip


def test_overlap_flag():
    test_rows = [_row("LEVERANCIERS.", i=0), _row("Huur loods", i=1)]
    marked = mark_overlaps(test_rows, [_row("Leveranciers", split="dev")])
    assert [r.overlaps_dev for r in marked] == [True, False]
    assert not test_rows[0].overlaps_dev  # the input rows are left unchanged


# --- fetching the LLM companies -----------------------------------------------------------


def test_fetch_missing_llm_fetches_each_company_once(small_root, fake_writer):
    config = load_config(small_root)
    paths = fetch_missing_llm(config, "test", fake_writer)
    assert [p.name for p in paths] == ["test-c003.json", "test-c004.json"]
    assert fetch_missing_llm(config, "test", fake_writer) == []
    assert fake_writer.calls == 2
    for p in paths:
        saved = json.loads(p.read_text(encoding="utf-8"))
        assert saved["code_scheme"] in {"llm", "legacy"} and len(saved["group_codes"]) == 8


def test_fetch_all_covers_every_split_and_is_reproducible(small_root, fake_writer):
    config = load_config(small_root)
    paths = fetch_missing_llm(config, "all", fake_writer)
    assert len(paths) == 5
    first = {p.name: json.loads(p.read_text(encoding="utf-8"))["group_codes"] for p in paths}
    for p in paths:
        p.unlink()
    again = fetch_missing_llm(config, "all", fake_writer)
    # A company deleted and fetched again is asked about exactly the same groups.
    assert {p.name: json.loads(p.read_text(encoding="utf-8"))["group_codes"] for p in again} == first


class FailsOnce:
    """Fails the first call, then writes valid charts."""

    def __init__(self, error, inner):
        self.inner, self.error, self.calls = inner, error, 0

    def complete(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise self.error
        return self.inner.complete(*args, **kwargs)


def test_one_failure_keeps_the_others_and_rerun_retries_it(small_root, fake_writer):
    config = load_config(small_root)
    flaky = FailsOnce(LLMError("claude CLI timed out after 300s"), fake_writer)
    with pytest.raises(LLMOutputError, match="1 of 5 companies failed"):
        fetch_missing_llm(config, "all", flaky, max_workers=1)
    assert flaky.calls == 5 and len(list(config.data.frozen_llm_dir.iterdir())) == 4
    assert len(fetch_missing_llm(config, "all", fake_writer)) == 1
    assert fake_writer.calls == 5  # 4 through FailsOnce, then the one retried


def test_not_logged_in_stops_after_the_first_failure(full_root, fake_writer):
    config = load_config(full_root)  # 78 LLM companies
    writer = FailsOnce(LLMError(f"claude CLI returned an error: Not logged in - {LOGIN_HINT}"), fake_writer)
    with pytest.raises(LLMOutputError, match="stopped.*log in"):
        fetch_missing_llm(config, "all", writer, max_workers=2)
    # Calls already running finish; nothing new starts. No point asking 77 more times.
    assert writer.calls <= 4


def test_fetch_logs_progress(small_root, fake_writer):
    lines = []
    fetch_missing_llm(load_config(small_root), "test", fake_writer, log=lines.append)
    assert "2 to fetch" in lines[0]
    assert any("[2/2]" in line for line in lines)
    assert "$0.06" in lines[-1]  # total cost of the run


# --- building the splits ----------------------------------------------------------------


def test_split_row_counts(full_root, fake_writer):
    config = load_config(full_root)
    fetch_missing_llm(config, "all", fake_writer)
    assert fake_writer.calls == 78
    paths = build_all(config)
    assert {split: len(read_jsonl(p)) for split, p in paths.items()} == {"train": 12000, "dev": 3500, "test": 600}


def test_build_is_deterministic(small_root, fake_writer):
    config = load_config(small_root)
    fetch_missing_llm(config, "all", fake_writer)
    first = {split: p.read_bytes() for split, p in build_all(config).items()}
    assert {split: p.read_bytes() for split, p in build_all(config).items()} == first


def test_rows_follow_the_plan_and_ids_are_unique(small_root, fake_writer):
    config = load_config(small_root)
    fetch_missing_llm(config, "all", fake_writer)
    rows = {split: read_jsonl(p) for split, p in build_all(config).items()}
    for split, split_rows in rows.items():
        plan = {c.company_id: c for c in company_plan(split, config.data.splits[split], config.data.industries, config.data.languages)}
        assert {r.company_id for r in split_rows} == set(plan)
        for r in split_rows:
            spec = plan[r.company_id]
            assert (r.split, r.source, r.industry, r.language) == (split, spec.source, spec.industry, spec.language)
    ids = [r.account_id for split_rows in rows.values() for r in split_rows]
    assert len(ids) == len(set(ids))


def test_build_without_frozen_files_names_the_command_and_writes_nothing(small_root):
    config = load_config(small_root)
    with pytest.raises(FrozenOutputMissing, match=r"5 of 5 LLM companies.*fetch-llm --split all"):
        build_all(config)
    assert not config.data.generated_dir.exists()


def test_build_rejects_frozen_file_made_for_a_different_plan(small_root, fake_writer):
    config = load_config(small_root)
    (path, *_) = fetch_missing_llm(config, "all", fake_writer)
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["language"] = "en" if saved["language"] != "en" else "fr"
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(LLMOutputError, match="language"):
        build_all(config)
