import csv
import json
import shutil

import numpy as np
import pytest

from d1.config import REPO_ROOT, load_config
from d1.datagen.schema import CompanyAccount, write_jsonl
from d1.evaluation.select_embedder import (
    ROWS_COLUMNS, choose_best, compare_embedders, main, render_report, shortlist_scores,
)
from d1.mapper.retrieve import Candidate

K = 10


def _rows(chart, split="dev"):
    """One dev row per group, named with its own English name: an easy split for the fake."""
    return [
        CompanyAccount(
            account_id=f"{split}-c000-{i:03d}", company_id=f"{split}-c000", industry="retail", split=split,
            source="rules" if i % 2 else "llm", language=["nl", "fr", "en", "mixed"][i % 4],
            code_scheme="legacy", code=f"{1000 + i}", name=a.name_en, noise=[], target_code=a.code,
        )  # fmt: skip
        for i, a in enumerate(chart)
    ]


class RandomEmbedder:
    """A model that understands nothing: random vectors."""

    def __init__(self, model_id):
        self.model_id = model_id
        self.rng = np.random.default_rng(0)

    def encode(self, texts, kind):
        v = self.rng.normal(size=(len(texts), 32)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def _factory(FakeEmbedder):
    return lambda name: RandomEmbedder(name) if name == "random" else FakeEmbedder(name)


# --- scoring one shortlist -----------------------------------------------------------------


def test_shortlist_scores_on_toy_lists():
    def c(*codes):
        return [Candidate(code=code, score=1.0 - i / 10) for i, code in enumerate(codes)]

    shortlists = [c("A", "B", "C", "D"), c("B", "A", "C", "D"), c("B", "C", "A", "D"), c("B", "C", "D", "E")]
    scores = shortlist_scores(shortlists, ["A", "A", "A", "A"], k=4)
    assert scores == {"top1": 0.25, "top3": 0.75, "recall_at_k": 0.75}


def test_shortlist_scores_need_one_gold_per_shortlist():
    with pytest.raises(ValueError, match="2 shortlists but 1 gold"):
        shortlist_scores([[], []], ["A"], k=3)


# --- comparing models --------------------------------------------------------------------


def test_compare_embedders_scores_every_model(chart, FakeEmbedder):
    ticks = iter([0.0, 2.5, 10.0, 11.0])  # start/stop for each model
    results = compare_embedders(["fake/a", "random"], _rows(chart), chart, K, _factory(FakeEmbedder), clock=lambda: next(ticks))
    assert [r["model"] for r in results] == ["fake/a", "random"]
    good, bad = results
    assert good["recall_at_k"] == 1.0 and good["top1"] > 0.9  # exact standard names
    assert bad["recall_at_k"] < 0.5  # random vectors: about 10 in 190
    for r in results:
        assert 0 <= r["top1"] <= r["top3"] <= r["recall_at_k"] <= 1
    assert good["seconds"] == 2.5 and bad["seconds"] == 1.0


def test_compare_embedders_breaks_recall_down(chart, FakeEmbedder):
    (result,) = compare_embedders(["fake/a"], _rows(chart), chart, K, _factory(FakeEmbedder))
    assert set(result["recall_by_language"]) == {"nl", "fr", "en", "mixed"}
    assert set(result["recall_by_source"]) == {"rules", "llm"}
    assert result["n_rows"] == len(chart)


def test_compare_embedders_keeps_the_rank_of_the_right_group_per_row(chart, FakeEmbedder):
    rows = _rows(chart)
    good, bad = compare_embedders(["fake/a", "random"], rows, chart, K, _factory(FakeEmbedder))
    for r in (good, bad):
        details = r["per_row"]
        assert [d["account_id"] for d in details] == [row.account_id for row in rows]
        ranks = [d["gold_rank"] for d in details]
        assert all(1 <= rank <= len(chart) for rank in ranks)  # the full ranking, not just the top k
        assert r["recall_at_k"] == sum(rank <= K for rank in ranks) / len(ranks)
        assert r["top1"] == sum(rank == 1 for rank in ranks) / len(ranks)
        assert all(d["top1_score"] >= d["top2_score"] for d in details)
        assert all(len(d["top3_codes"]) == 3 for d in details)
        assert r["dim"] == 256 if r is good else r["dim"] == 32


def test_model_load_is_not_timed(chart, FakeEmbedder):
    seen = []

    class Recording(FakeEmbedder):
        def encode(self, texts, kind):
            seen.append((len(texts), kind, clock_calls[0]))
            return super().encode(texts, kind)

    clock_calls = [0]

    def clock():
        clock_calls[0] += 1
        return float(clock_calls[0])

    compare_embedders(["fake/a"], _rows(chart), chart, K, lambda name: Recording(name), clock=clock)
    assert seen[0] == (1, "query", 0)  # a warm-up call before the clock starts loads the model


@pytest.mark.parametrize("split", ["train", "test"])
def test_only_dev_rows_are_accepted(chart, FakeEmbedder, split):
    # The model is a choice, so it is made on dev; test must stay unseen.
    with pytest.raises(ValueError, match="dev rows only"):
        compare_embedders(["fake/a"], _rows(chart, split), chart, K, _factory(FakeEmbedder))


# --- the rule for the winner -----------------------------------------------------------------


def _result(model, recall, top1, seconds):
    return {"model": model, "recall_at_k": recall, "top1": top1, "top3": top1, "seconds": seconds}


def test_best_recall_wins():
    results = [_result("a", 0.90, 0.80, 1), _result("b", 0.95, 0.50, 99)]
    assert choose_best(results)["model"] == "b"  # recall caps every method, top-1 does not


def test_tie_on_recall_goes_to_top1_then_speed():
    assert choose_best([_result("a", 0.9, 0.6, 1), _result("b", 0.9, 0.7, 50)])["model"] == "b"
    assert choose_best([_result("a", 0.9, 0.7, 30), _result("b", 0.9, 0.7, 20)])["model"] == "b"


# --- report ---------------------------------------------------------------------------------


def test_report_has_the_table_the_rule_and_the_winner(chart, FakeEmbedder):
    results = compare_embedders(["fake/a", "random"], _rows(chart), chart, K, _factory(FakeEmbedder))
    text = render_report(results, choose_best(results), K)
    assert text.startswith("# Choosing the embedding model")
    assert "| fake/a |" in text and "| random |" in text
    assert "recall@10" in text and "**Chosen: `fake/a`**" in text
    assert "190 dev rows" in text


# --- command line ---------------------------------------------------------------------------


@pytest.fixture
def cli_root(config_root):
    shutil.copytree(REPO_ROOT / "data" / "source", config_root / "data" / "source")
    return config_root


def test_cli_writes_report(cli_root, chart, FakeEmbedder, capsys):
    config = load_config(cli_root)
    write_jsonl(_rows(chart), config.data.generated_dir / "dev.jsonl")
    assert main([], root=cli_root, embedder_factory=_factory(FakeEmbedder)) == 0
    report = cli_root / "reports" / "embedding_selection.md"
    assert report.is_file()
    for model in config.models.embedding_candidates:
        assert f"| {model} |" in report.read_text(encoding="utf-8")
    assert "embedding_model" in capsys.readouterr().out  # tells you what to set in models.yaml

    # One line per (model, dev row), for the notebook: it draws every chart from this file.
    with open(cli_root / "reports" / "embedding_selection_rows.csv", encoding="utf-8", newline="") as f:
        lines = list(csv.DictReader(f))
    assert len(lines) == len(config.models.embedding_candidates) * len(chart)
    assert set(ROWS_COLUMNS) == set(lines[0])
    first = lines[0]
    assert first["noise"] == "" and first["top3_codes"].count(";") == 2 and int(first["gold_rank"]) >= 1

    # And one summary per model (speed, vector size), without the per-row details.
    summary = json.loads((cli_root / "reports" / "embedding_selection.json").read_text(encoding="utf-8"))
    assert [s["model"] for s in summary["models"]] == config.models.embedding_candidates
    assert summary["k"] == K and summary["chosen"] in config.models.embedding_candidates
    assert all("per_row" not in s and s["dim"] == 256 and s["seconds"] >= 0 for s in summary["models"])


def test_cli_without_dev_split_gives_hint(cli_root, capsys):
    assert main([], root=cli_root) == 1
    assert "python -m d1.datagen build" in capsys.readouterr().err


def test_cli_model_that_cannot_load_gives_one_line(cli_root, chart, FakeEmbedder, capsys):
    write_jsonl(_rows(chart), load_config(cli_root).data.generated_dir / "dev.jsonl")

    def factory(name):
        if "mpnet" in name:
            raise OSError("Can't load the model (403 Forbidden)")
        return FakeEmbedder(name)

    assert main([], root=cli_root, embedder_factory=factory) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: could not load sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    assert "403 Forbidden" in err and "run it again" in err and err.count("\n") == 1
    assert not (cli_root / "reports" / "embedding_selection.md").exists()  # no report with a model missing
