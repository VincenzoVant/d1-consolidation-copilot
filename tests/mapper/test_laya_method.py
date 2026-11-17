import os
import subprocess
import sys

import numpy as np
import pytest

from d1.config import load_config
from d1.datagen.schema import read_jsonl
from d1.mapper.methods import METHOD_REGISTRY
from d1.mapper.methods.base import AccountIn
from d1.mapper.answer_cache import AnswerCache
from d1.mapper.methods.laya_method import INSTRUCTIONS, LayaMethod, option_text
from d1.mapper.retrieve import Candidate, Retriever


class FakeAgent:
    """Answers like Laya, from rules the test sets, and records every call.

    prefer: account body -> the label it should favour. slot_bias: extra weight for
    whatever option sits in slot 0, the way the real checkpoint favours early slots.
    Probabilities come back keyed in the caller's label order, whatever order the
    options were shown in, as Laya documents.
    """

    def __init__(self, prefer=None, slot_bias=0.0, fail_on=None):
        self.prefer = prefer or {}
        self.slot_bias = slot_bias
        self.fail_on = fail_on
        self.calls = []

    def predict(self, state, questions, **kwargs):
        self.calls.append((state, questions, kwargs))
        if state["body"] == self.fail_on:
            raise RuntimeError("MPS backend out of memory")
        answers = {}
        for qid, q in questions.items():
            labels = list(q["criteria"])
            order = q.get("option_order", list(range(len(labels))))
            weight = {label: 1.0 for label in labels}
            if state["body"] in self.prefer:
                weight[self.prefer[state["body"]]] += 5.0
            weight[labels[order[0]]] += self.slot_bias  # whatever the model sees first
            total = sum(weight.values())
            probs = {label: weight[label] / total for label in labels}
            answers[qid] = {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs}
        return {"answers": answers}


def _cands(*codes):
    return [Candidate(code=c, score=0.9 - 0.01 * i) for i, c in enumerate(codes)]


SHORTLIST = _cands("440", "400", "604", "550", "620")


def _method(chart, agent, **kwargs) -> LayaMethod:
    return LayaMethod("convaiinnovations/laya-multilingual", chart, agent_loader=lambda model_id, revision: agent, **kwargs)


# --- the question Laya is asked -------------------------------------------------------------------


def test_question_has_k_candidates_plus_none(chart):
    agent = FakeAgent()
    _method(chart, agent, rotations=1).score([AccountIn(account_id="a", name="Leveranciers", code="440000")], [SHORTLIST])
    (state, questions, kwargs), = agent.calls
    (q,) = questions.values()
    assert q["type"] == "choice" and q["instructions"] == INSTRUCTIONS
    assert list(q["criteria"]) == ["440", "400", "604", "550", "620", "none"]  # the shortlist's order, then "none"
    assert q["criteria"]["440"] == "Suppliers / Leveranciers / Fournisseurs"
    assert q["criteria"]["none"] == "none of these groups"
    assert state == {"body": "Leveranciers (code 440000)"}
    assert kwargs["head_max_len"] == 640  # room for all three names of every option


def test_state_without_a_code(chart):
    agent = FakeAgent()
    _method(chart, agent, rotations=1).score([AccountIn(account_id="a", name="Huur kantoor")], [SHORTLIST])
    assert agent.calls[0][0] == {"body": "Huur kantoor"}


def test_option_text_names_all_three_languages(chart):
    group = next(a for a in chart if a.code == "604")
    assert option_text(group) == f"{group.name_en} / {group.name_nl} / {group.name_fr}"


# --- position bias: every option in every slot once ---------------------------------------------------


def test_all_rotations_put_every_option_in_every_slot_once(chart):
    agent = FakeAgent()
    k = len(SHORTLIST) + 1
    _method(chart, agent, rotations=k).score([AccountIn(account_id="a", name="x")], [SHORTLIST])
    (_, questions, _), = agent.calls  # all rotations in one call: one forward pass
    assert len(questions) == k
    orders = [q["option_order"] for q in questions.values()]
    for slot in range(k):
        assert sorted(order[slot] for order in orders) == list(range(k))


def test_fewer_rotations_are_spread_over_the_slots(chart):
    """The default: 6 rotations of 11 options, each option in 6 different slots, evenly spaced."""
    agent = FakeAgent()
    shortlist = _cands("440", "400", "604", "550", "620", "700", "708", "741", "750", "751")
    method = _method(chart, agent)
    assert method.rotations == 6
    method.score([AccountIn(account_id="a", name="x")], [shortlist])
    (_, questions, _), = agent.calls
    shifts = [q["option_order"][0] for q in questions.values()]  # the option in slot 0 = the shift
    assert shifts == [0, 1, 3, 5, 7, 9]
    for option in range(11):
        slots = [q["option_order"].index(option) for q in questions.values()]
        assert len(set(slots)) == 6
    assert _method(chart, FakeAgent(), rotations=40).shifts(11) == list(range(11))  # never more than k


def test_rotations_cancel_a_slot_bias(chart):
    """A model that favours slot 0 picks the first candidate when asked once, and nothing in particular after rotating."""
    account = AccountIn(account_id="a", name="x")
    (once,) = _method(chart, FakeAgent(slot_bias=3.0), rotations=1).score([account], [SHORTLIST])
    (rotated,) = _method(chart, FakeAgent(slot_bias=3.0), rotations=len(SHORTLIST) + 1).score([account], [SHORTLIST])
    assert once.chosen_code == "440" and once.scores["440"] > 2 * once.scores["400"]
    assert max(rotated.scores.values()) == pytest.approx(min(rotated.scores.values()))


# --- reading the answer -------------------------------------------------------------------------------


def test_scores_map_back_to_codes(chart):
    agent = FakeAgent(prefer={"Kantoorbenodigdheden": "620"})
    (out,) = _method(chart, agent).score([AccountIn(account_id="a", name="Kantoorbenodigdheden")], [SHORTLIST])
    assert set(out.scores) == {"440", "400", "604", "550", "620"}  # the candidates; "none" is not a group
    assert out.chosen_code == "620" and out.status == "ok"
    assert list(out.scores) == sorted(out.scores, key=out.scores.get, reverse=True)  # best first
    assert next(iter(out.scores)) == "620"
    assert out.latency_ms >= 0 and out.cost_usd == 0.0  # it runs locally: no money, only time


def test_none_answer_sets_status_none(chart):
    agent = FakeAgent(prefer={"Diverse": "none"})
    (out,) = _method(chart, agent).score([AccountIn(account_id="a", name="Diverse")], [SHORTLIST])
    assert out.chosen_code is None and out.status == "none"
    assert "none" in out.reason and "440" in out.scores  # the candidates' scores are kept for the reviewer


def test_batching_preserves_order(chart):
    names = ["Leveranciers", "Huur", "Lonen", "Bank", "Klanten"]
    agent = FakeAgent(prefer={n: code for n, code in zip(names, ["440", "400", "604", "550", "620"])})
    outputs = _method(chart, agent).score([AccountIn(account_id=n, name=n) for n in names], [SHORTLIST] * 5)
    assert [o.account_id for o in outputs] == names
    assert [o.chosen_code for o in outputs] == ["440", "400", "604", "550", "620"]


def test_empty_shortlist_and_a_failing_account(chart):
    agent = FakeAgent(fail_on="kapot")
    accounts = [AccountIn(account_id="a", name="leeg"), AccountIn(account_id="b", name="kapot"), AccountIn(account_id="c", name="ok")]
    empty, failed, fine = _method(chart, agent).score(accounts, [[], SHORTLIST, SHORTLIST])
    assert empty.status == "none" and empty.chosen_code is None and len(agent.calls) == 2  # no question for an empty list
    assert failed.status == "error" and failed.chosen_code is None and "out of memory" in failed.reason
    assert fine.status == "ok"  # one failure doesn't stop the batch
    with pytest.raises(ValueError, match="shortlists"):
        _method(chart, agent).score(accounts, [SHORTLIST])


# --- loading the model ------------------------------------------------------------------------------


def test_model_loaded_once_on_first_use_with_the_pinned_revision(chart):
    loads = []
    method = LayaMethod("convaiinnovations/laya-multilingual", chart, revision="abc123",
                        agent_loader=lambda model_id, revision: loads.append((model_id, revision)) or FakeAgent())  # fmt: skip
    assert loads == []  # building the method costs nothing
    method.score([AccountIn(account_id="a", name="x")], [SHORTLIST])
    method.score([AccountIn(account_id="b", name="y")], [SHORTLIST])
    assert loads == [("convaiinnovations/laya-multilingual", "abc123")]


def test_use_tf_disabled(chart, monkeypatch):
    monkeypatch.delenv("USE_TF", raising=False)
    _method(chart, FakeAgent()).score([AccountIn(account_id="a", name="x")], [SHORTLIST])
    assert os.environ["USE_TF"] == "0"  # upstream: with TensorFlow installed, loading can crash


def test_openmp_single_threaded_once_d1_is_imported():
    """XGBoost and PyTorch each bring an OpenMP library; two at once crash Laya or XGBoost on macOS."""
    env = {k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"}
    out = subprocess.run([sys.executable, "-c", "import d1, os; print(os.environ['OMP_NUM_THREADS'])"], env=env, capture_output=True, text=True)
    assert out.stdout.strip() == "1"
    env["OMP_NUM_THREADS"] = "4"  # a value set on purpose is kept
    out = subprocess.run([sys.executable, "-c", "import d1, os; print(os.environ['OMP_NUM_THREADS'])"], env=env, capture_output=True, text=True)
    assert out.stdout.strip() == "4"


# --- saving, version, registry -----------------------------------------------------------------------


def test_version_names_model_revision_and_settings(chart):
    version = LayaMethod("convaiinnovations/laya-multilingual", chart, revision="e4e9ddf21a7b", agent_loader=None).version()
    assert version == "laya-convaiinnovations/laya-multilingual@e4e9ddf-rot6-h640-v1"
    once = LayaMethod("convaiinnovations/laya-multilingual", chart, revision="e4e9ddf21a7b", rotations=1, agent_loader=None)
    assert "-rot1-" in once.version()


# --- the answer cache ---------------------------------------------------------------------------------


def test_cache_answers_each_question_once(chart, tmp_path):
    agent = FakeAgent(prefer={"Leveranciers": "440"})
    accounts = [AccountIn(account_id=i, name="Leveranciers") for i in ("a", "b")]  # the same question twice
    first = _method(chart, agent, cache=AnswerCache(tmp_path / "laya.sqlite")).score(accounts, [SHORTLIST, SHORTLIST])
    assert len(agent.calls) == 1  # asked once, within one run
    again = _method(chart, agent, cache=AnswerCache(tmp_path / "laya.sqlite")).score(accounts, [SHORTLIST, SHORTLIST])
    assert len(agent.calls) == 1  # and not at all in the next run: the file is reused
    assert [o.scores for o in again] == [o.scores for o in first] and [o.chosen_code for o in again] == ["440", "440"]
    assert not first[0].cached and all(o.cached for o in again) and again[0].latency_ms == 0


def test_cache_is_saved_as_it_goes(chart, tmp_path, monkeypatch):
    """A crash part-way keeps what was already asked."""
    import d1.mapper.methods.laya_method as laya_module

    monkeypatch.setattr(laya_module, "SAVE_EVERY", 2)
    cache = AnswerCache(tmp_path / "laya.sqlite")
    agent = FakeAgent(fail_on="name 4")
    original = agent.predict

    def predict(state, questions, **kwargs):
        if state["body"] == "name 4":
            raise KeyboardInterrupt  # the process is stopped at the fifth account
        return original(state, questions, **kwargs)

    agent.predict = predict
    accounts = [AccountIn(account_id=str(i), name=f"name {i}") for i in range(6)]
    with pytest.raises(KeyboardInterrupt):
        _method(chart, agent, cache=cache).score(accounts, [SHORTLIST] * 6)
    assert len(cache) == 4  # saved in pairs: the four answered before the stop


def test_cache_key_follows_every_setting(chart, tmp_path):
    cache = AnswerCache(tmp_path / "laya.sqlite")
    agent = FakeAgent()
    account = [AccountIn(account_id="a", name="x", code="440000")]
    _method(chart, agent, cache=cache).score(account, [SHORTLIST])
    _method(chart, agent, cache=cache, rotations=3).score(account, [SHORTLIST])  # other rotations
    _method(chart, agent, cache=cache).score(account, [SHORTLIST[:4]])  # another shortlist
    _method(chart, agent, cache=cache).score([account[0].model_copy(update={"code": None})], [SHORTLIST])  # no code
    assert len(agent.calls) == 4


def test_errors_are_not_cached(chart, tmp_path):
    cache = AnswerCache(tmp_path / "laya.sqlite")
    agent = FakeAgent(fail_on="kapot")
    account = [AccountIn(account_id="a", name="kapot")]
    assert _method(chart, agent, cache=cache).score(account, [SHORTLIST])[0].status == "error"
    agent.fail_on = None
    assert _method(chart, agent, cache=cache).score(account, [SHORTLIST])[0].status == "ok"  # asked again
    assert len(agent.calls) == 2


def test_save_load_refuses_another_model_or_revision(chart, tmp_path):
    method = LayaMethod("convaiinnovations/laya-multilingual", chart, revision="aaa", agent_loader=None)
    method.save(tmp_path)
    LayaMethod("convaiinnovations/laya-multilingual", chart, revision="aaa", agent_loader=None).load(tmp_path)
    with pytest.raises(ValueError, match="bbb"):
        LayaMethod("convaiinnovations/laya-multilingual", chart, revision="bbb", agent_loader=None).load(tmp_path)
    with pytest.raises(ValueError, match="convaiinnovations/laya"):
        LayaMethod("convaiinnovations/laya", chart, revision="aaa", agent_loader=None).load(tmp_path)


def test_registry_builds_laya_with_the_reviewed_revision(chart, fake_embedder):
    config = load_config()
    method = METHOD_REGISTRY["laya"](config, Retriever(fake_embedder, chart))
    assert isinstance(method, LayaMethod) and not method.needs_training
    assert method.model_id == config.models.mapping.laya_model
    assert method.revision == "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"  # laya's own list of reviewed commits
    assert method.cache is not None and method.cache.path == config.root / ".cache" / "laya" / "answers.sqlite"


# --- the real model (downloads it once; run with `uv run pytest -m network tests/mapper/test_laya_method.py`) ----


@pytest.mark.network
def test_real_laya_on_five_dev_accounts_next_to_xgboost(chart, fake_embedder):
    """The real checkpoint answers in the expected shape, in a process that has XGBoost loaded and trained."""
    from xgboost import XGBClassifier

    XGBClassifier(n_estimators=5).fit(np.random.rand(50, 3), np.arange(50) % 2)  # the crash came with XGBoost present
    config = load_config()
    dev = read_jsonl(config.data.generated_dir / "dev.jsonl")[:5]
    retriever = Retriever(fake_embedder, chart)
    method = METHOD_REGISTRY["laya"](config, retriever)
    outputs = method.score([AccountIn(account_id=r.account_id, name=r.name, code=r.code) for r in dev], retriever.top_k([r.name for r in dev], 10))
    assert [o.account_id for o in outputs] == [r.account_id for r in dev]
    for o in outputs:
        assert o.status in ("ok", "none") and len(o.scores) == 10
        assert sum(o.scores.values()) <= 1.0 + 1e-6  # probabilities, with "none" taking the rest
