import json
import random
import threading
import time

import pytest

from d1.config import load_config
from d1.llm import ClaudeCLIBackend, LLMError, LLMResponse
from d1.mapper.answer_cache import AnswerCache
from d1.mapper.methods import METHOD_REGISTRY
from d1.mapper.methods.base import AccountIn
from d1.mapper.methods.llm_method import NONE, LLMMethod, answer_schema, build_prompt, make_batches
from d1.mapper.retrieve import Candidate, Retriever

MODEL = "claude-haiku-5-5"


class FakeBackend:
    """Answers like the CLI backend, from a rule the test sets, and records every call.

    choose(label, prompt) -> (choice, confidence) per account label; the default picks
    the first candidate listed under that account, at 80. `structured` replaces the
    whole answer (for malformed answers); `fail` raises LLMError; `jitter` makes calls
    finish in random order, as threads do.
    """

    def __init__(self, choose=None, structured=None, fail=None, jitter=0.0, cost=0.003, effort=None):
        self.choose = choose
        self.structured = structured
        self.fail = fail
        self.jitter = jitter
        self.cost = cost
        self.effort = effort
        self.calls = []
        self._lock = threading.Lock()

    def complete(self, prompt, schema, model, system, timeout_s=120.0):
        with self._lock:
            self.calls.append({"prompt": prompt, "schema": schema, "model": model, "system": system, "timeout_s": timeout_s})
        if self.jitter:
            time.sleep(random.random() * self.jitter)
        if self.fail:
            raise LLMError(self.fail)
        labels = schema["properties"]["answers"]["items"]["properties"]["account"]["enum"]
        if self.structured is not None:
            structured = self.structured
        else:
            answers = []
            for label in labels:
                block = prompt.split(f"## Account {label}\n")[1].split("## Account")[0]
                first = block.split("Candidate groups:\n- ")[1].split(":")[0]
                choice, confidence = self.choose(label, block) if self.choose else (first, 80)
                answers.append({"account": label, "choice": choice, "confidence": confidence, "reason": f"because {label}"})
            structured = {"answers": answers}
        return LLMResponse(structured=structured, model=model, input_tokens=100, output_tokens=50, cost_usd=self.cost, latency_ms=1234.0)


def _cands(*codes):
    return [Candidate(code=c, score=0.9 - 0.01 * i) for i, c in enumerate(codes)]


SHORTLIST = _cands("440", "400", "604")
OTHER = _cands("620", "550", "604")


def _account(i, company="c1", name=None, code="440000", before=(), after=()):
    return AccountIn(account_id=f"a{i}", name=name or f"account {i}", code=code, company_id=company, before=list(before), after=list(after))


def _method(chart, backend, **kwargs) -> LLMMethod:
    return LLMMethod(backend, MODEL, chart, **kwargs)


# --- the prompt and the schema -------------------------------------------------------------------------


def test_prompt_contains_candidates_and_neighbours(chart):
    account = _account(1, name="Leveranciers binnenland", code="440100", before=["Ontvangen vooruitbetalingen"], after=["Te betalen wissels", "Lonen"])
    prompt = build_prompt([("A1", account, SHORTLIST)], {a.code: a for a in chart})
    block = prompt.split("## Account A1\n")[1]
    assert "Name: Leveranciers binnenland" in block and "Company code: 440100" in block
    assert 'Just above it: "Ontvangen vooruitbetalingen"' in block
    assert 'Just below it: "Te betalen wissels"; "Lonen"' in block
    group = next(a for a in chart if a.code == "440")
    assert f"- 440: {group.name_en} / {group.name_nl} / {group.name_fr}" in block  # all three languages
    assert block.index("- 440:") < block.index("- 400:") < block.index("- 604:")  # shortlist order
    assert '"none"' in prompt  # the way out is offered


def test_prompt_without_code_or_neighbours(chart):
    account = AccountIn(account_id="x", name="Divers")
    block = build_prompt([("A1", account, SHORTLIST)], {a.code: a for a in chart}).split("## Account A1\n")[1]
    assert "Company code: (none)" in block and "Just above" not in block and "Just below" not in block


def test_schema_limits_labels_and_choices():
    batch = [("A1", _account(1), SHORTLIST), ("A2", _account(2), OTHER)]
    item = answer_schema(batch)["properties"]["answers"]["items"]
    assert item["properties"]["account"]["enum"] == ["A1", "A2"]
    assert item["properties"]["choice"]["enum"] == ["440", "400", "604", "620", "550", NONE]  # every candidate once, then none
    assert item["required"] == ["account", "choice", "confidence", "reason"] and item["additionalProperties"] is False
    assert item["properties"]["confidence"] == {"type": "integer", "minimum": 0, "maximum": 100}


# --- batching ------------------------------------------------------------------------------------------


def test_batches_follow_companies_and_size():
    accounts = [_account(i, company="c1") for i in range(4)] + [_account(i, company="c2") for i in range(4, 6)]
    shortlists = [SHORTLIST] * 6
    assert make_batches(accounts, shortlists, 3) == [[0, 1, 2], [3], [4, 5]]  # never across two companies
    assert make_batches(accounts, shortlists, 1) == [[i] for i in range(6)]
    shortlists[1] = []  # nothing to choose from: no question
    assert make_batches(accounts, shortlists, 3) == [[0, 2, 3], [4, 5]]


def test_order_kept_with_concurrent_calls(chart):
    names = [f"n{i}" for i in range(12)]
    accounts = [_account(i, company=f"c{i % 3}", name=n) for i, n in enumerate(names)]
    backend = FakeBackend(jitter=0.02)
    outputs = _method(chart, backend, batch_size=2, max_workers=4).score(accounts, [SHORTLIST] * 12)
    assert [o.account_id for o in outputs] == [a.account_id for a in accounts]
    assert all(o.status == "ok" and o.chosen_code == "440" for o in outputs)


# --- reading the answer --------------------------------------------------------------------------------


def test_valid_choice_ok(chart):
    backend = FakeBackend(choose=lambda label, block: ("400", 73))
    (out,) = _method(chart, backend).score([_account(1)], [SHORTLIST])
    assert out.status == "ok" and out.chosen_code == "400"
    assert out.scores == {"400": 0.73}  # one score: the stated confidence; the other candidates get none
    assert out.reason == "because A1" and out.cost_usd == pytest.approx(0.003) and out.latency_ms == 1234.0 and not out.cached
    assert backend.calls[0]["model"] == MODEL


def test_none_choice(chart):
    backend = FakeBackend(choose=lambda label, block: (NONE, 90))
    (out,) = _method(chart, backend).score([_account(1)], [SHORTLIST])
    assert out.status == "none" and out.chosen_code is None and out.scores == {}
    assert "none of these groups" in out.reason and "because A1" in out.reason


def test_out_of_candidate_choice_is_invalid(chart):
    """The schema lists every candidate of the batch; a choice from another account's list is still refused."""
    backend = FakeBackend(choose=lambda label, block: ("620", 95) if label == "A1" else ("620", 95))
    first, second = _method(chart, backend, batch_size=2).score([_account(1), _account(2)], [SHORTLIST, OTHER])
    assert first.status == "invalid" and first.chosen_code is None and "620" in first.reason
    assert second.status == "ok" and second.chosen_code == "620"


@pytest.mark.parametrize("structured", [
    {"answers": [{"account": "A1", "choice": "440", "confidence": 80, "reason": "x"}]},  # A2 missing
    {"answers": [{"account": "A1", "choice": "440", "confidence": 80, "reason": "x"}] * 2 + [{"account": "A2", "choice": "440", "confidence": 80, "reason": "x"}]},
    {"answers": [{"account": "A1", "choice": "440", "reason": "x"}, {"account": "A2", "choice": "440", "confidence": 80, "reason": "x"}]},
    {"answers": [{"account": "A1", "choice": "440", "confidence": 180, "reason": "x"}, {"account": "A2", "choice": "440", "confidence": 80, "reason": "x"}]},
    {"result": "no answers key"},
])  # fmt: skip
def test_malformed_answer_is_error_for_the_whole_batch(chart, structured, tmp_path):
    cache = AnswerCache(tmp_path / "llm.sqlite")
    outputs = _method(chart, FakeBackend(structured=structured), batch_size=2, cache=cache).score([_account(1), _account(2)], [SHORTLIST, SHORTLIST])
    assert [o.status for o in outputs] == ["error", "error"] and all(o.chosen_code is None for o in outputs)
    assert len(cache) == 0  # a malformed answer is not kept: asked again next time


def test_backend_error_is_error_and_not_cached(chart, tmp_path):
    cache = AnswerCache(tmp_path / "llm.sqlite")
    (out,) = _method(chart, FakeBackend(fail="claude CLI timed out after 120s"), cache=cache).score([_account(1)], [SHORTLIST])
    assert out.status == "error" and out.chosen_code is None and "timed out" in out.reason
    assert len(cache) == 0
    backend = FakeBackend()
    (again,) = _method(chart, backend, cache=cache).score([_account(1)], [SHORTLIST])
    assert again.status == "ok" and len(backend.calls) == 1  # asked again


def test_empty_shortlist_makes_no_call(chart):
    backend = FakeBackend()
    empty, fine = _method(chart, backend).score([_account(1), _account(2)], [[], SHORTLIST])
    assert empty.status == "none" and empty.reason == "empty shortlist" and len(backend.calls) == 1
    assert fine.status == "ok"
    with pytest.raises(ValueError, match="shortlists"):
        _method(chart, backend).score([_account(1)], [])


def test_batch_cost_is_split_over_its_accounts(chart):
    outputs = _method(chart, FakeBackend(cost=0.003), batch_size=3).score([_account(i) for i in range(3)], [SHORTLIST] * 3)
    assert [o.cost_usd for o in outputs] == pytest.approx([0.001] * 3)
    assert all(o.latency_ms == 1234.0 for o in outputs)  # each waited for the whole call


# --- the cap and the cache -----------------------------------------------------------------------------


def test_cap_reached_returns_capped_without_call(chart):
    backend = FakeBackend()
    method = _method(chart, backend, max_calls=2)
    outputs = method.score([_account(i) for i in range(3)], [SHORTLIST] * 3)
    assert [o.status for o in outputs] == ["ok", "ok", "capped"] and len(backend.calls) == 2 and method.calls_made == 2
    assert "cap" in outputs[2].reason
    (later,) = method.score([_account(9)], [SHORTLIST])
    assert later.status == "capped" and len(backend.calls) == 2  # the cap holds over calls to score()


def test_cache_hit_skips_backend(chart, tmp_path):
    accounts = [_account(1), _account(2, name="other")]
    first = _method(chart, FakeBackend(), cache=AnswerCache(tmp_path / "llm.sqlite")).score(accounts, [SHORTLIST, SHORTLIST])
    backend = FakeBackend()
    method = _method(chart, backend, cache=AnswerCache(tmp_path / "llm.sqlite"), max_calls=0)  # a cache hit is not a call
    again = method.score(accounts, [SHORTLIST, SHORTLIST])
    assert backend.calls == [] and method.calls_made == 0
    assert [o.chosen_code for o in again] == [o.chosen_code for o in first] and [o.reason for o in again] == [o.reason for o in first]
    assert all(o.cached and o.cost_usd == 0 and o.latency_ms == 0 for o in again)  # free and instant this time


def test_last_usage_keeps_the_first_time_cost(chart, tmp_path):
    """A cached run is free, but the notebook and the dev summary need what the answers cost the first time."""
    accounts = [_account(i) for i in range(4)]
    method = _method(chart, FakeBackend(cost=0.002), batch_size=2, cache=AnswerCache(tmp_path / "llm.sqlite"))
    method.score(accounts, [SHORTLIST] * 4)
    assert method.last_usage["calls"] == 2 and method.last_usage["cached_calls"] == 0
    again = _method(chart, FakeBackend(), batch_size=2, cache=AnswerCache(tmp_path / "llm.sqlite"))
    again.score(accounts, [SHORTLIST] * 4)
    usage = again.last_usage
    assert usage["calls"] == 0 and usage["cached_calls"] == 2 and usage["failed_calls"] == 0 and usage["capped_calls"] == 0
    assert usage["first_time_cost_usd"] == pytest.approx(0.004) and usage["median_call_ms"] == 1234.0
    assert usage["input_tokens"] == 200 and usage["output_tokens"] == 100 and usage["wall_s"] >= 0


def test_same_question_asked_once(chart, tmp_path):
    backend = FakeBackend()
    twins = [AccountIn(account_id=i, name="Leveranciers", code="440000") for i in ("a", "b")]
    _method(chart, backend, cache=AnswerCache(tmp_path / "llm.sqlite")).score(twins, [SHORTLIST, SHORTLIST])
    assert len(backend.calls) == 1


def test_cache_key_follows_model_batch_and_prompt(chart, tmp_path):
    cache = AnswerCache(tmp_path / "llm.sqlite")
    accounts = [_account(1), _account(2)]
    backend = FakeBackend()
    _method(chart, backend, cache=cache).score(accounts, [SHORTLIST] * 2)  # 2 calls
    _method(chart, backend, cache=cache, batch_size=2).score(accounts, [SHORTLIST] * 2)  # 1 call: another batch size
    LLMMethod(backend, "claude-sonnet-5-5", chart, cache=cache).score(accounts, [SHORTLIST] * 2)  # 2 calls: another model
    _method(chart, backend, cache=cache).score(accounts, [SHORTLIST] * 2)  # 0: all cached
    assert len(backend.calls) == 5


# --- identity, saving, registry ------------------------------------------------------------------------


def test_version_names_model_prompt_batch_and_effort(chart):
    version = _method(chart, FakeBackend(effort="low"), batch_size=10).version()
    assert version.startswith(f"llm-{MODEL}-prompt-v1-") and version.endswith("-b10-effort-low")
    assert _method(chart, FakeBackend()).version().endswith("-b1-effort-default")


def test_save_load_refuses_other_settings(chart, tmp_path):
    _method(chart, FakeBackend(effort="low"), batch_size=10).save(tmp_path)
    _method(chart, FakeBackend(effort="low"), batch_size=10).load(tmp_path)
    with pytest.raises(ValueError, match="retrain"):
        _method(chart, FakeBackend(effort="low"), batch_size=1).load(tmp_path)
    with pytest.raises(ValueError, match="retrain"):
        LLMMethod(FakeBackend(effort="low"), "claude-sonnet-5-5", chart, batch_size=10).load(tmp_path)
    info = json.loads((tmp_path / "method.json").read_text())
    assert info["params"]["model"] == MODEL and info["params"]["batch_size"] == 10 and info["params"]["effort"] == "low"


def test_registry_builds_llm_from_the_config(chart, fake_embedder):
    config = load_config()
    method = METHOD_REGISTRY["llm"](config, Retriever(fake_embedder, chart))
    assert isinstance(method, LLMMethod) and not method.needs_training
    assert method.model == config.models.mapping_model
    assert method.batch_size == config.models.mapping.llm_batch_size
    assert isinstance(method.backend, ClaudeCLIBackend) and method.backend.effort == config.models.mapping.llm_effort
    assert method.cache is not None and method.cache.path == config.root / ".cache" / "llm" / "answers.sqlite"
