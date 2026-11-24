"""The cascade: the first method auto-maps what it is sure of, the LLM looks at the rest."""

import json

import pytest
import yaml

from d1.config import ConfigError, load_config
from d1.evaluation.run_mapping import evaluate
from d1.mapper.calibrate import FEATURES, Calibrator
from d1.mapper.cascade import CascadeMapper, load_mapper, merge
from d1.mapper.methods import METHOD_REGISTRY
from d1.mapper.methods.base import AccountIn, MethodOutput
from d1.mapper.methods.llm_method import LLMMethod
from d1.mapper.pipeline import Mapper, MapperError, Thresholds
from d1.mapper.retrieve import Retriever
from d1.mapper.train import main, train_cascade, train_method

K = 10
CAL = Calibrator.from_json({"type": "logistic", "features": FEATURES, "coef": [10.0, 30.0], "intercept": -7.0, "n_fitted": 100})
CLEAR = {"scores": {"440": 0.9, "400": 0.1}, "chosen_code": "440"}  # P(right) close to 1 with CAL
UNSURE = {"scores": {"440": 0.50, "400": 0.49}, "chosen_code": "440"}  # P(right) about 0.15 with CAL


class StubMethod:
    """Answers what the test tells it to, per account id, and records which accounts it was asked about."""

    def __init__(self, name, answers):
        self.name, self.answers, self.asked = name, answers, []
        self.needs_training = False

    def score(self, accounts, candidates):
        self.asked += [a.account_id for a in accounts]
        return [MethodOutput(account_id=a.account_id, **self.answers[a.account_id]) for a in accounts]

    def version(self):
        return f"{self.name}-stub"


def _stage(fake_embedder, chart, name, answers):
    thresholds = Thresholds(method=name, t_auto=0.9, target_precision=0.98, fitted_on="dev", embedding_model=fake_embedder.model_id)
    return Mapper(Retriever(fake_embedder, chart), StubMethod(name, answers), CAL, thresholds, K)


def _cascade(fake_embedder, chart, first_answers, llm_answers):
    return CascadeMapper(_stage(fake_embedder, chart, "xgboost", first_answers), _stage(fake_embedder, chart, "llm", llm_answers))


def _accounts(*ids):
    return [AccountIn(account_id=i, name=f"account {i}") for i in ids]


# --- routing: auto, cascade_llm, review -------------------------------------------------------------


def test_pipeline_routes_auto_cascade_review(fake_embedder, chart):
    first = {"a": CLEAR, "b": UNSURE, "c": UNSURE}
    llm = {"b": {"scores": {"400": 0.9}, "chosen_code": "400"}, "c": {"scores": {"441": 0.3, "400": 0.29}, "chosen_code": "441", "reason": "not sure"}}
    cascade = _cascade(fake_embedder, chart, first, llm)
    a, b, c = cascade.map(_accounts("a", "b", "c"))
    assert (a.decision_path, a.method, a.chosen_code) == ("auto", "xgboost", "440")  # the first method was sure
    assert (b.decision_path, b.method, b.chosen_code) == ("cascade_llm", "llm", "400")  # the LLM was sure of the rest
    assert (c.decision_path, c.method, c.chosen_code, c.reason) == ("review", "llm", "441", "not sure")  # nobody was sure
    assert cascade.second.method.asked == ["b", "c"]  # the LLM is only asked about what the first method left


def test_llm_not_called_when_the_first_method_is_sure_of_everything(fake_embedder, chart):
    cascade = _cascade(fake_embedder, chart, {"a": CLEAR, "b": CLEAR}, {})
    assert [d.decision_path for d in cascade.map(_accounts("a", "b"))] == ["auto", "auto"]
    assert cascade.second.method.asked == []


@pytest.mark.parametrize("status", ["none", "invalid", "error", "capped"])
def test_llm_without_any_answer_falls_back_to_the_first_methods_list(fake_embedder, chart, status):
    llm = {"b": {"scores": {}, "chosen_code": None, "status": status, "reason": "llm: something", "cost_usd": 0.002}}
    (d,) = _cascade(fake_embedder, chart, {"b": UNSURE}, llm).map(_accounts("b"))
    assert d.decision_path == "review" and d.status == status and "llm: something" in d.reason
    # Claude gave no ranking at all, so the reviewer still gets options: the first method's.
    assert d.chosen_code == "440" and d.method == "xgboost" and [c.code for c in d.candidates] == ["440", "400"]
    assert d.cost_usd == 0.002  # the LLM call is still paid for


@pytest.mark.parametrize("status", ["none", "invalid"])
def test_llm_ranking_without_a_pick_is_still_what_the_reviewer_reads(fake_embedder, chart, status):
    """Claude said none fits (or put another account's group first), but ranked this account's candidates: its list is shown."""
    llm = {"b": {"scores": {"441": 0.2, "400": 0.1}, "chosen_code": None, "status": status, "reason": "llm: none of these groups (x)"}}
    (d,) = _cascade(fake_embedder, chart, {"b": UNSURE}, llm).map(_accounts("b"))
    assert d.decision_path == "review" and d.status == status and d.method == "llm"
    assert [c.code for c in d.candidates] == ["441", "400"] and d.chosen_code == "441"  # Claude's first is the suggestion
    assert d.candidates[0].reason == "llm: none of these groups (x)" and d.candidates[1].reason is None


def test_review_shows_only_claudes_ranking(fake_embedder, chart):
    llm = {"b": {"scores": {"400": 0.3, "441": 0.29}, "chosen_code": "400"}}
    (d,) = _cascade(fake_embedder, chart, {"b": UNSURE}, llm).map(_accounts("b"))
    assert d.decision_path == "review" and [c.code for c in d.candidates] == ["400", "441"]  # no fill from the first method


def test_decisions_keep_input_order_and_record_both_stages(fake_embedder, chart):
    first = {i: (CLEAR if int(i) % 2 else UNSURE) for i in map(str, range(6))}
    llm = {i: CLEAR for i in first}
    decisions = _cascade(fake_embedder, chart, first, llm).map(_accounts(*first))
    assert [d.account_id for d in decisions] == list(first)
    assert [d.decision_path for d in decisions] == ["cascade_llm", "auto"] * 3
    versions = decisions[0].model_versions
    assert versions["first_method"] == "xgboost-stub" and versions["second_method"] == "llm-stub"
    assert {"embedder", "first_calibrator", "second_calibrator"} <= set(versions)
    assert all(d.model_versions == versions for d in decisions)  # the same pair of stages for every account


def test_merge_keeps_the_llms_cost_on_an_auto_map(fake_embedder, chart):
    first = _stage(fake_embedder, chart, "xgboost", {"b": UNSURE}).map(_accounts("b"))[0]
    second = _stage(fake_embedder, chart, "llm", {"b": {**CLEAR, "cost_usd": 0.003, "cached": True}}).map(_accounts("b"))[0]
    d = merge(first, second)
    assert d.decision_path == "cascade_llm" and d.cost_usd == 0.003 and d.cached


def test_empty_input_gives_no_decisions(fake_embedder, chart):
    assert _cascade(fake_embedder, chart, {}, {}).map([]) == []


# --- the setting ---------------------------------------------------------------------------------------


def _set_mapping(root, **values):
    path = root / "config" / "models.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["mapping"].update(values)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_config_names_the_cascades_first_method(config_root):
    assert load_config(config_root).models.mapping.cascade_first == "xgboost"


def test_the_cascade_asks_claude_for_a_ranked_five_and_the_llm_method_for_one(mapper_root, fake_embedder, chart):
    from d1.mapper.cascade import second_method

    config = load_config(mapper_root)
    assert config.models.mapping.cascade_llm_choices == 5 and config.models.mapping.llm_choices == 1
    retriever = Retriever(fake_embedder, chart)
    assert second_method(config, retriever).n_choices == 5
    assert METHOD_REGISTRY["llm"](config, retriever).n_choices == 1  # notebook 16's method keeps its own question


def test_cascade_skipped_when_winner_is_llm(config_root):
    _set_mapping(config_root, cascade_first="llm")
    with pytest.raises(ConfigError, match="cascade_first"):  # the LLM is the second step: it can't also be the first
        load_config(config_root)


def test_cascade_first_must_be_a_compared_method(config_root):
    _set_mapping(config_root, cascade_first="xgboost", methods=["fuzzy", "llm"])
    with pytest.raises(ConfigError, match="cascade_first"):
        load_config(config_root)


# --- training on dev, loading, evaluating ---------------------------------------------------------------


@pytest.fixture
def fake_llm(monkeypatch, FirstPickBackend):
    """The LLM method on a fake Claude; backend.asked lists the account names it was asked about."""
    backend = FirstPickBackend()

    def build(config, retriever):
        from d1.datagen.standard_chart import load_standard_chart

        return LLMMethod(backend, "fake-model", load_standard_chart(config.data.chart_path, expected=config.data.expected_groups), batch_size=10)

    monkeypatch.setitem(METHOD_REGISTRY, "llm", build)
    return backend


def _factory(FakeEmbedder):
    return lambda model_name: FakeEmbedder()


@pytest.fixture
def cascade_root(mapper_root, FakeEmbedder, fake_llm):
    """mapper_root with fuzzy trained, as the cascade's first method."""
    _set_mapping(mapper_root, cascade_first="fuzzy")
    train_method(load_config(mapper_root), "fuzzy", _factory(FakeEmbedder))
    return mapper_root


def test_train_cascade_writes_its_artifacts(cascade_root, FakeEmbedder, fake_llm):
    summary = train_cascade(load_config(cascade_root), _factory(FakeEmbedder))
    folder = cascade_root / "artifacts" / "methods" / "cascade"
    assert {"cascade.json", "dev_summary.json", "second"} <= {p.name for p in folder.iterdir()}
    assert {"method.json", "calibrator.json", "thresholds.json"} <= {p.name for p in (folder / "second").iterdir()}
    info = json.loads((folder / "cascade.json").read_text())
    assert info["first"] == "fuzzy" and info["second"] == "llm" and len(info["first_fingerprint"]) == 64
    second = json.loads((folder / "second" / "thresholds.json").read_text())
    assert second["method"] == "llm" and second["threshold_by"] == "code_match" and second["fitted_on"] == "dev"
    assert summary["method"] == "cascade" and summary["first"] == "fuzzy" and summary["n_dev"] == 200


def test_the_llm_sees_only_what_the_first_method_did_not_auto_map(cascade_root, FakeEmbedder, fake_llm):
    summary = train_cascade(load_config(cascade_root), _factory(FakeEmbedder))
    stages = summary["stages"]
    assert stages["first_auto"] + stages["sent_to_llm"] == 200
    assert len(fake_llm.asked) == stages["sent_to_llm"]
    assert stages["first_auto"] + stages["llm_auto"] + stages["review"] == 200


def test_every_stage_and_segment_holds_the_target(cascade_root, FakeEmbedder, fake_llm):
    summary = train_cascade(load_config(cascade_root), _factory(FakeEmbedder))
    for cell in summary["by_stage_segment"].values():
        if cell["auto"] >= 50:  # thresholds need 50 auto-mapped dev accounts behind them
            assert cell["auto_precision"] >= 0.98
    assert summary["auto_share"] >= summary["first_alone"]["auto_share"]  # the cascade only adds auto-maps


def test_train_cascade_needs_the_first_method_trained(mapper_root, FakeEmbedder, fake_llm):
    _set_mapping(mapper_root, cascade_first="fuzzy")
    with pytest.raises(MapperError, match=r"python -m d1\.mapper\.train --method fuzzy"):
        train_cascade(load_config(mapper_root), _factory(FakeEmbedder))


def test_load_cascade_and_reproduce_the_dev_summary(cascade_root, FakeEmbedder, fake_llm):
    config = load_config(cascade_root)
    summary = train_cascade(config, _factory(FakeEmbedder))
    mapper = load_mapper(config, "cascade", _factory(FakeEmbedder))
    assert isinstance(mapper, CascadeMapper)
    metrics = evaluate(config, "dev", "cascade", _factory(FakeEmbedder))
    assert metrics["overall"]["auto_share"] == pytest.approx(summary["auto_share"])
    assert metrics["overall"]["top1"] == pytest.approx(summary["top1"])
    assert metrics["thresholds"]["first"]["method"] == "fuzzy" and metrics["thresholds"]["second"]["method"] == "llm"


def test_load_mapper_for_one_method_is_the_plain_mapper(cascade_root, FakeEmbedder):
    assert type(load_mapper(load_config(cascade_root), "fuzzy", _factory(FakeEmbedder))) is Mapper


def test_load_refuses_a_cascade_whose_first_method_was_retrained(cascade_root, FakeEmbedder, fake_llm):
    config = load_config(cascade_root)
    train_cascade(config, _factory(FakeEmbedder))
    calibrator = cascade_root / "artifacts" / "methods" / "fuzzy" / "calibrator.json"
    data = json.loads(calibrator.read_text())
    calibrator.write_text(json.dumps({**data, "intercept": data["intercept"] + 1}))
    with pytest.raises(MapperError, match=r"python -m d1\.mapper\.train --method cascade"):
        load_mapper(config, "cascade", _factory(FakeEmbedder))


def test_load_before_training_names_the_command(cascade_root, FakeEmbedder):
    with pytest.raises(MapperError, match=r"python -m d1\.mapper\.train --method cascade"):
        load_mapper(load_config(cascade_root), "cascade", _factory(FakeEmbedder))


def test_cli_trains_the_cascade(cascade_root, FakeEmbedder, fake_llm, capsys):
    assert main(["--method", "cascade"], root=cascade_root, embedder_factory=_factory(FakeEmbedder)) == 0
    out = capsys.readouterr().out
    assert out.startswith("cascade (fuzzy, then llm): ") and "sent to the LLM" in out


def test_train_cascade_can_try_another_first_method_without_saving(cascade_root, FakeEmbedder, fake_llm):
    config = load_config(cascade_root)
    train_method(config, "embeddings", _factory(FakeEmbedder))
    summary = train_cascade(config, _factory(FakeEmbedder), first="embeddings", save=False)
    assert summary["first"] == "embeddings"
    assert not (cascade_root / "artifacts" / "methods" / "cascade").exists()


def test_the_reviewer_reads_claudes_reason_on_its_choice(fake_embedder, chart):
    reason = "BOIP/EUIPO trademark registration is an intangible right, classed under concessions, patents, licences, brands and similar rights."
    llm = {"b": {"scores": {"211": 0.3, "742": 0.29}, "chosen_code": "211", "reason": reason}}
    (d,) = _cascade(fake_embedder, chart, {"b": UNSURE}, llm).map(_accounts("b"))
    options = d.review_options()
    assert d.decision_path == "review" and options[0].code == "211" and options[0].reason == reason
    assert [c.code for c in options] == ["211", "742"] and options[1].reason is None
