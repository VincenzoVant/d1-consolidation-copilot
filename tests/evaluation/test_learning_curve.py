import json
import shutil
from random import Random

import pytest

from d1.config import REPO_ROOT, load_config
from d1.datagen.rule_charts import generate_rule_company
from d1.datagen.schema import write_jsonl
from d1.evaluation.learning_curve import learning_curve, main, more_data_helps, subsample_companies, write_report
from d1.mapper.methods.baselines import EmbeddingsMethod
from d1.mapper.retrieve import Retriever


def _companies(chart, split: str, n_rules: int, n_llm: int, per_company: int = 10):
    rows = []
    for c in range(n_rules + n_llm):
        company = generate_rule_company(f"{split}-c{c:03d}", split, "retail", "nl", chart, per_company, Random(c), code_scheme="legacy")
        if c >= n_rules:  # stands in for an LLM-written company
            company = [r.model_copy(update={"source": "llm"}) for r in company]
        rows += company
    return rows


class RecordingMethod:
    """Remembers the rows each fit() got; scores like the embeddings baseline."""

    name = "embeddings"
    needs_training = True

    def __init__(self, log: list):
        self.log = log

    def fit(self, train_rows, candidates):
        assert len(train_rows) == len(candidates)
        self.log.append(train_rows)

    def score(self, accounts, candidates):
        return EmbeddingsMethod().score(accounts, candidates)

    def version(self):
        return "recording"


# --- subsampling by company ------------------------------------------------------------------------


def test_subsample_keeps_whole_companies_nested_and_the_source_mix(chart):
    rows = _companies(chart, "train", n_rules=16, n_llm=4)
    sizes = {}
    previous: set[str] = set()
    for fraction in (0.25, 0.5, 0.75, 1.0):
        subset = subsample_companies(rows, fraction, seed=0)
        companies = {r.company_id for r in subset}
        assert len(subset) == sum(1 for r in rows if r.company_id in companies)  # every account of a chosen company
        assert previous <= companies  # 25% is inside 50%, inside 75%: the curve compares like with like
        previous = companies
        by_source = {s: len({r.company_id for r in subset if r.source == s}) for s in ("rules", "llm")}
        sizes[fraction] = by_source
    assert sizes[0.25] == {"rules": 4, "llm": 1} and sizes[1.0] == {"rules": 16, "llm": 4}
    assert subsample_companies(rows, 1.0, seed=0) == rows  # all of it, in the original order
    assert subsample_companies(rows, 0.5, seed=0) != subsample_companies(rows, 0.5, seed=1)


def test_subsample_refuses_a_fraction_outside_zero_one(chart):
    rows = _companies(chart, "train", n_rules=4, n_llm=0)
    for fraction in (0.0, 1.5):
        with pytest.raises(ValueError, match="fraction"):
            subsample_companies(rows, fraction, seed=0)


# --- the curve -------------------------------------------------------------------------------------


def test_learning_curve_subsamples_whole_companies(chart, fake_embedder):
    train = _companies(chart, "train", n_rules=8, n_llm=4)
    dev = _companies(chart, "dev", n_rules=3, n_llm=2, per_company=20)
    log: list = []
    points = learning_curve(lambda: RecordingMethod(log), train, dev, Retriever(fake_embedder, chart), k=10,
                            fractions=(0.25, 0.5, 1.0), seed=0, min_support=5)  # fmt: skip
    assert [p["fraction"] for p in points] == [0.25, 0.5, 1.0]
    assert [len(rows) for rows in log] == [p["n_train_accounts"] for p in points] == [30, 60, 120]
    assert [p["n_train_companies"] for p in points] == [3, 6, 12]
    for rows in log:
        companies = {r.company_id for r in rows}
        assert len(rows) == 10 * len(companies)  # whole companies of 10 accounts
    for p in points:
        assert 0 <= p["top1"] <= 1 and 0 <= p["auto_share"] <= 1
        low, high = p["auto_share_ci"]
        assert low <= p["auto_share"] <= high
        assert set(p["by_source"]) == {"rules", "llm"} and p["by_source"]["llm"]["n"] == 40
    # The recording method ignores what it was trained on, so every point is the same.
    assert len({p["top1"] for p in points}) == 1


# --- the decision rule and the report -----------------------------------------------------------


def _point(fraction, auto_share, ci):
    return {"fraction": fraction, "n_train_accounts": int(12000 * fraction), "n_train_companies": int(240 * fraction),
            "top1": 0.8, "auto_share": auto_share, "auto_share_ci": list(ci), "auto_precision": 0.98, "t_auto": 0.9,
            "by_source": {"rules": {"n": 2000, "top1": 0.9, "auto_share": 0.5}, "llm": {"n": 1500, "top1": 0.6, "auto_share": 0.1}}}  # fmt: skip


def test_more_data_helps_only_beyond_the_interval():
    # From 75% to 100% of the companies: 6 points gained, the 100% point's interval is +-3 points.
    assert more_data_helps([_point(0.75, 0.30, (0.27, 0.33)), _point(1.0, 0.36, (0.33, 0.39))])
    assert not more_data_helps([_point(0.75, 0.34, (0.31, 0.37)), _point(1.0, 0.36, (0.33, 0.39))])
    with pytest.raises(ValueError, match="0.75"):
        more_data_helps([_point(0.5, 0.30, (0.27, 0.33)), _point(1.0, 0.36, (0.33, 0.39))])


def test_write_report(tmp_path):
    curves = {"xgboost": [_point(f, 0.30 + 0.02 * i, (0.27, 0.37)) for i, f in enumerate((0.25, 0.5, 0.75, 1.0))],
              "random_forest": [_point(f, 0.28, (0.25, 0.31)) for f in (0.25, 0.5, 0.75, 1.0)]}  # fmt: skip
    written = write_report(curves, tmp_path)
    assert sorted(p.name for p in written) == ["learning_curve.json", "learning_curve.md", "learning_curve.png"]
    assert json.loads((tmp_path / "learning_curve.json").read_text())["curves"] == curves
    text = (tmp_path / "learning_curve.md").read_text()
    assert "xgboost" in text and "random_forest" in text and "12,000" in text
    assert "more data stopped helping" in text  # neither curve gains more than its interval at the end


# --- the command line --------------------------------------------------------------------------------


@pytest.fixture
def curve_root(config_root, chart):
    shutil.copytree(REPO_ROOT / "data" / "source", config_root / "data" / "source")
    out = load_config(config_root).data.generated_dir
    write_jsonl(_companies(chart, "train", n_rules=8, n_llm=4), out / "train.jsonl")
    write_jsonl(_companies(chart, "dev", n_rules=3, n_llm=2, per_company=20), out / "dev.jsonl")
    return config_root


def test_cli_writes_the_report(curve_root, FakeEmbedder, capsys):
    assert main(["--method", "xgboost"], root=curve_root, embedder_factory=lambda name: FakeEmbedder()) == 0
    data = json.loads((curve_root / "reports" / "learning_curve.json").read_text())
    assert [p["n_train_accounts"] for p in data["curves"]["xgboost"]] == [30, 60, 90, 120]
    assert "xgboost:" in capsys.readouterr().out


def test_cli_refuses_a_method_that_learns_nothing(curve_root, FakeEmbedder, capsys):
    assert main(["--method", "fuzzy"], root=curve_root, embedder_factory=lambda name: FakeEmbedder()) == 1
    assert "learns nothing" in capsys.readouterr().err
