import json
import shutil
from random import Random

import pytest
import yaml

from d1.config import REPO_ROOT, load_config
from d1.datagen.rule_charts import generate_rule_company
from d1.datagen.schema import write_jsonl
from d1.mapper.calibrate import FEATURES, Calibrator
from d1.mapper.methods.base import AccountIn, MethodOutput
from d1.mapper.pipeline import Mapper, MapperError, Thresholds
from d1.mapper.retrieve import Retriever
from d1.mapper.train import main, train_method

K = 10
CAL = Calibrator.from_json({"type": "logistic", "features": FEATURES, "coef": [10.0, 30.0], "intercept": -7.0, "n_fitted": 100})


class StubMethod:
    """A method that answers what the test tells it to, per account id."""

    name = "embeddings"
    needs_training = False

    def __init__(self, answers):
        self.answers = answers

    def fit(self, train_rows, candidates):
        pass

    def score(self, accounts, candidates):
        return [MethodOutput(account_id=a.account_id, **self.answers[a.account_id]) for a in accounts]

    def version(self):
        return "stub-v1"


def _mapper(fake_embedder, chart, answers, t_auto=0.9, calibrator=CAL):
    thresholds = Thresholds(method="embeddings", t_auto=t_auto, target_precision=0.98, fitted_on="dev", embedding_model=fake_embedder.model_id)
    return Mapper(Retriever(fake_embedder, chart), StubMethod(answers), calibrator, thresholds, K)


CLEAR = {"scores": {"440": 0.9, "400": 0.1}, "chosen_code": "440"}  # P(right) close to 1 with CAL
UNSURE = {"scores": {"440": 0.50, "400": 0.49}, "chosen_code": "440"}  # P(right) about 0.15 with CAL


# --- auto-map or review ---------------------------------------------------------------------


def test_above_threshold_auto(fake_embedder, chart):
    (d,) = _mapper(fake_embedder, chart, {"a": CLEAR}).map([AccountIn(account_id="a", name="Leveranciers")])
    assert d.decision_path == "auto" and d.chosen_code == "440" and d.confidence > 0.99
    assert d.method == "embeddings" and d.status == "ok"


def test_below_threshold_review_keeps_candidates(fake_embedder, chart):
    (d,) = _mapper(fake_embedder, chart, {"a": UNSURE}).map([AccountIn(account_id="a", name="Leveranciers")])
    assert d.decision_path == "review" and d.confidence == pytest.approx(CAL.predict([[0.50, 0.01]])[0])
    assert [(c.code, c.score) for c in d.candidates] == [("440", 0.50), ("400", 0.49)]  # what the reviewer sees
    assert d.chosen_code == "440"  # the suggestion is kept for the reviewer


def test_none_choice_goes_to_review(fake_embedder, chart):
    answer = {"scores": {}, "chosen_code": None, "status": "none", "reason": "no group fits"}
    (d,) = _mapper(fake_embedder, chart, {"a": answer}).map([AccountIn(account_id="a", name="x")])
    assert d.decision_path == "review" and d.chosen_code is None and d.confidence == 0.0
    assert d.status == "none" and d.reason == "no group fits"


def test_none_threshold_sends_all_to_review(fake_embedder, chart):
    decisions = _mapper(fake_embedder, chart, {"a": CLEAR, "b": CLEAR}, t_auto=None).map(
        [AccountIn(account_id="a", name="x"), AccountIn(account_id="b", name="y")]
    )
    assert [d.decision_path for d in decisions] == ["review", "review"]


def test_decisions_keep_input_order(fake_embedder, chart):
    answers = {i: (CLEAR if int(i) % 2 else UNSURE) for i in map(str, range(6))}
    decisions = _mapper(fake_embedder, chart, answers).map([AccountIn(account_id=i, name=f"n{i}") for i in answers])
    assert [d.account_id for d in decisions] == list(answers)


def test_model_versions_recorded(fake_embedder, chart):
    (d,) = _mapper(fake_embedder, chart, {"a": CLEAR}).map([AccountIn(account_id="a", name="x")])
    assert d.model_versions["embedder"] == "fake/trigram-256" and d.model_versions["method"] == "stub-v1"
    assert len(d.model_versions["calibrator"]) == 12
    other = Calibrator.from_json({**CAL.to_json(), "intercept": -6.0})
    (d2,) = _mapper(fake_embedder, chart, {"a": CLEAR}, calibrator=other).map([AccountIn(account_id="a", name="x")])
    assert d2.model_versions["calibrator"] != d.model_versions["calibrator"]  # a refit calibrator is traceable


def test_empty_input_gives_no_decisions(fake_embedder, chart):
    assert _mapper(fake_embedder, chart, {}).map([]) == []


# --- training and loading the artifacts ----------------------------------------------------


@pytest.fixture
def mapper_root(config_root, chart):
    """A repo root with the real chart and small train/dev splits (no test split at all)."""
    shutil.copytree(REPO_ROOT / "data" / "source", config_root / "data" / "source")
    config = load_config(config_root)
    for split, n_companies in (("train", 2), ("dev", 4)):
        rows = []
        for c, language in enumerate(["nl", "fr", "en", "mixed"][:n_companies]):
            rows += generate_rule_company(f"{split}-c{c:03d}", split, "retail", language, chart, 50, Random(c), code_scheme="legacy")
        # Mislabel every fourth row, so even an exact-name method is sometimes wrong (calibration needs both).
        rows = [r.model_copy(update={"target_code": "604" if r.target_code != "604" else "600"}) if i % 4 == 0 else r for i, r in enumerate(rows)]
        write_jsonl(rows, config.data.generated_dir / f"{split}.jsonl")
    return config_root


def _fake_factory(FakeEmbedder):
    return lambda model_name: FakeEmbedder()


def test_train_writes_the_artifacts(mapper_root, FakeEmbedder):
    config = load_config(mapper_root)
    summary = train_method(config, "fuzzy", _fake_factory(FakeEmbedder))
    folder = mapper_root / "artifacts" / "methods" / "fuzzy"
    assert sorted(p.name for p in folder.iterdir()) == ["calibrator.json", "dev_summary.json", "method.json", "thresholds.json"]
    thresholds = json.loads((folder / "thresholds.json").read_text())
    assert thresholds["method"] == "fuzzy" and thresholds["fitted_on"] == "dev" and thresholds["target_precision"] == 0.98
    assert thresholds["embedding_model"] == "fake/trigram-256"
    assert summary["n_dev"] == 200 and 0 <= summary["auto_share"] <= 1
    # The mislabelled quarter can't be right, so 75% is the ceiling; noise costs fuzzy a few more.
    assert 0.5 < summary["top1"] <= 0.76


def test_trained_mapper_from_artifacts(mapper_root, FakeEmbedder):
    config = load_config(mapper_root)
    train_method(config, "fuzzy", _fake_factory(FakeEmbedder))
    mapper = Mapper.from_artifacts(config, "fuzzy", embedder_factory=_fake_factory(FakeEmbedder))
    decisions = mapper.map([AccountIn(account_id="x", name="Leveranciers"), AccountIn(account_id="y", name="zzz qqq")])
    assert decisions[0].chosen_code == "440" and decisions[0].method == "fuzzy"
    assert {d.decision_path for d in decisions} <= {"auto", "review"}


def test_from_artifacts_without_a_method_names_the_setting(mapper_root):
    with pytest.raises(MapperError, match="selected_method"):
        Mapper.from_artifacts(load_config(mapper_root))  # selected_method is null until Task 3.5


def test_from_artifacts_before_training_names_the_command(mapper_root, FakeEmbedder):
    with pytest.raises(MapperError, match=r"python -m d1\.mapper\.train --method fuzzy"):
        Mapper.from_artifacts(load_config(mapper_root), "fuzzy", embedder_factory=_fake_factory(FakeEmbedder))


def test_from_artifacts_refuses_another_embedding_model(mapper_root, FakeEmbedder):
    config = load_config(mapper_root)
    train_method(config, "fuzzy", _fake_factory(FakeEmbedder))
    # The thresholds were fitted with one embedder; loading with another would silently change every score.
    with pytest.raises(MapperError, match="fake/trigram-256"):
        Mapper.from_artifacts(config, "fuzzy", embedder_factory=lambda name: FakeEmbedder("fake/other"))


def _set_group_text(root, value: str) -> None:
    path = root / "config" / "models.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["mapping"]["group_text"] = value
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _freeze_descriptions(config, chart):
    folder = config.data.frozen_descriptions_dir
    shutil.rmtree(folder, ignore_errors=True)  # replace the real descriptions with tiny ones
    folder.mkdir(parents=True)
    groups = [{"code": a.code, "description": f"about {a.code}", "words_nl": [], "words_fr": [], "words_en": []} for a in chart]
    (folder / "batch-01.json").write_text(json.dumps({"response": {"groups": groups}}), encoding="utf-8")


def test_train_records_the_group_text(mapper_root, FakeEmbedder):
    _set_group_text(mapper_root, "names")
    train_method(load_config(mapper_root), "fuzzy", _fake_factory(FakeEmbedder))
    thresholds = json.loads((mapper_root / "artifacts" / "methods" / "fuzzy" / "thresholds.json").read_text())
    assert thresholds["group_text"] == "names"


def test_from_artifacts_refuses_another_group_text(mapper_root, chart, FakeEmbedder):
    _set_group_text(mapper_root, "names")
    train_method(load_config(mapper_root), "fuzzy", _fake_factory(FakeEmbedder))
    # Shortlists with descriptions are different shortlists: the calibrator was not fitted on them.
    _set_group_text(mapper_root, "average")
    _freeze_descriptions(load_config(mapper_root), chart)
    with pytest.raises(MapperError, match="group_text"):
        Mapper.from_artifacts(load_config(mapper_root), "fuzzy", embedder_factory=_fake_factory(FakeEmbedder))


def test_build_retriever_follows_the_setting(mapper_root, chart, FakeEmbedder):
    from d1.mapper.pipeline import build_retriever

    _set_group_text(mapper_root, "names")
    assert build_retriever(load_config(mapper_root), FakeEmbedder(), chart).descriptions is None

    _set_group_text(mapper_root, "average")
    config = load_config(mapper_root)
    shutil.rmtree(config.data.frozen_descriptions_dir)
    with pytest.raises(MapperError, match=r"python -m d1\.mapper\.descriptions"):
        build_retriever(config, FakeEmbedder(), chart)  # nothing frozen yet
    _freeze_descriptions(config, chart)
    retriever = build_retriever(config, FakeEmbedder(), chart)
    assert retriever.group_text == "average" and retriever.descriptions["620"] == "about 620"


# --- command line ---------------------------------------------------------------------------


def test_cli_trains_one_method(mapper_root, FakeEmbedder, capsys):
    assert main(["--method", "embeddings"], root=mapper_root, embedder_factory=_fake_factory(FakeEmbedder)) == 0
    out = capsys.readouterr().out
    assert out.startswith("embeddings: top-1 ")  # with the fake embedder no threshold reaches 98%, which it says
    assert (mapper_root / "artifacts" / "methods" / "embeddings" / "thresholds.json").is_file()


def test_cli_all_trains_the_built_methods_and_skips_the_rest(mapper_root, FakeEmbedder, capsys):
    assert main(["--method", "all"], root=mapper_root, embedder_factory=_fake_factory(FakeEmbedder)) == 0
    out = capsys.readouterr().out
    assert "fuzzy:" in out and "embeddings:" in out and "xgboost: not built yet" in out


def test_cli_method_not_built_yet(mapper_root, capsys):
    assert main(["--method", "laya"], root=mapper_root) == 1
    assert "not built yet" in capsys.readouterr().err


def test_cli_without_dev_split_names_build(config_root, capsys):
    shutil.copytree(REPO_ROOT / "data" / "source", config_root / "data" / "source")
    assert main(["--method", "fuzzy"], root=config_root) == 1
    assert "python -m d1.datagen build" in capsys.readouterr().err


def test_map_can_reuse_shortlists_computed_before(fake_embedder, chart):
    mapper = _mapper(fake_embedder, chart, {"a": CLEAR})
    shortlists = mapper.retriever.top_k(["Leveranciers"], K)
    calls = len(fake_embedder.calls)
    (d,) = mapper.map([AccountIn(account_id="a", name="Leveranciers")], shortlists=shortlists)
    assert len(fake_embedder.calls) == calls and d.decision_path == "auto"  # no second retrieval


def test_decisions_carry_cost_and_latency(fake_embedder, chart):
    answer = {**CLEAR, "cost_usd": 0.0012, "latency_ms": 800.0, "cached": True}
    (d,) = _mapper(fake_embedder, chart, {"a": answer}).map([AccountIn(account_id="a", name="x")])
    assert (d.cost_usd, d.latency_ms, d.cached) == (0.0012, 800.0, True)


def test_threshold_holds_out_of_fold_and_with_the_saved_calibrator(mapper_root, FakeEmbedder):
    config = load_config(mapper_root)
    summary = train_method(config, "fuzzy", _fake_factory(FakeEmbedder))
    if summary["t_auto"] is not None:
        assert summary["auto_precision"] >= 0.98 and summary["out_of_fold"]["auto_precision"] >= 0.98
    assert set(summary["out_of_fold"]) == {"t_auto", "auto_share", "auto_precision"}
