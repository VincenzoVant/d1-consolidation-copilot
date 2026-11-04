import numpy as np
import pytest

from d1.datagen.schema import CompanyAccount
from d1.evaluation.metrics import (
    auto_precision, auto_share, bootstrap_ci, breakdown, cost_report, paired_bootstrap_diff, recall_at_k, review_share,
    topk_accuracy,
)  # fmt: skip
from d1.mapper.pipeline import MappingDecision
from d1.mapper.retrieve import Candidate


def _d(account_id, chosen, path="auto", cands=("440", "400", "441"), cost=0.0, latency=0.0, cached=False):
    return MappingDecision(
        account_id=account_id, method="embeddings", candidates=[Candidate(code=c, score=1 - i / 10) for i, c in enumerate(cands)],
        confidence=0.9, decision_path=path, chosen_code=chosen, status="ok", model_versions={},
        cost_usd=cost, latency_ms=latency, cached=cached,
    )  # fmt: skip


def _row(account_id, source="rules", language="nl", noise=(), company="c0", target="440"):
    return CompanyAccount(
        account_id=account_id, company_id=company, industry="retail", split="dev", source=source, language=language,
        code_scheme="legacy", code="1000", name="x", noise=list(noise), target_code=target,
    )  # fmt: skip


# --- the headline numbers ------------------------------------------------------------------


def test_auto_and_review_share():
    ds = [_d("a", "440"), _d("b", "440", "review"), _d("c", None, "review"), _d("d", "400")]
    assert auto_share(ds) == 0.5 and review_share(ds) == 0.5


def test_auto_precision_simple():
    gold = {"a": "440", "b": "440", "c": "440", "d": "440"}
    ds = [_d("a", "440"), _d("b", "400"), _d("c", "440"), _d("d", "400", "review")]
    assert auto_precision(ds, gold) == pytest.approx(2 / 3)  # review rows don't count
    assert auto_precision([_d("x", "440", "review")], {"x": "440"}) is None  # nothing auto-mapped: no precision


def test_topk():
    gold = {"a": "440", "b": "400", "c": "441", "d": "999"}
    ds = [_d(i, "440") for i in gold]
    assert topk_accuracy(ds, gold, 1) == 0.25 and topk_accuracy(ds, gold, 2) == 0.5 and topk_accuracy(ds, gold, 3) == 0.75


def test_recall_at_k():
    shortlists = {"a": [Candidate(code="440", score=0.9), Candidate(code="400", score=0.8)], "b": [Candidate(code="100", score=0.9)]}
    gold = {"a": "400", "b": "440"}
    assert recall_at_k(shortlists, gold, 1) == 0.0 and recall_at_k(shortlists, gold, 2) == 0.5


def test_metrics_refuse_decisions_without_gold():
    with pytest.raises(KeyError, match="zz"):
        auto_precision([_d("zz", "440")], {"a": "440"})


# --- uncertainty ----------------------------------------------------------------------------


def test_bootstrap_ci_contains_point_and_is_seeded():
    values = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1] * 10, dtype=float)
    low, high = bootstrap_ci(lambda idx: values[idx].mean(), len(values), n=500, seed=0)
    assert low < values.mean() < high
    assert (low, high) == bootstrap_ci(lambda idx: values[idx].mean(), len(values), n=500, seed=0)


def test_bootstrap_ci_resamples_whole_groups():
    # Group g0 is all 1, g1 all 0: resampling groups reaches 0 and 1, resampling items would not.
    values = np.array([1.0] * 20 + [0.0] * 20)
    groups = np.array(["g0"] * 20 + ["g1"] * 20)
    low, high = bootstrap_ci(lambda idx: values[idx].mean(), len(values), n=400, seed=1, groups=groups)
    assert low == 0.0 and high == 1.0


def test_paired_diff_zero_for_identical():
    a = np.array([1, 0, 1, 1, 0], dtype=float)
    assert paired_bootstrap_diff(a, a) == (0.0, 0.0, 0.0)


def test_paired_diff_brackets_the_difference():
    a = np.array([1.0] * 60 + [0.0] * 40)
    b = np.array([1.0] * 40 + [0.0] * 60)
    diff, low, high = paired_bootstrap_diff(a, b, n=500, seed=0, groups=np.repeat(np.arange(20), 5))
    assert diff == pytest.approx(0.2) and low <= 0.2 <= high and low > 0


# --- breakdowns -------------------------------------------------------------------------------


def test_breakdown_counts_multi_tag_rows_per_tag():
    rows = [_row("a", noise=["typo", "case"]), _row("b", noise=["typo"]), _row("c"), _row("d", noise=["case"])]
    ds = [_d("a", "440"), _d("b", "400"), _d("c", "440", "review"), _d("d", "440")]
    table = {r["value"]: r for r in breakdown(rows, ds, "noise")}
    assert table["typo"]["n"] == 2 and table["case"]["n"] == 2 and table["no noise"]["n"] == 1
    assert table["typo"]["auto_precision"] == 0.5 and table["case"]["auto_precision"] == 1.0
    assert table["no noise"]["auto_share"] == 0.0 and table["no noise"]["auto_precision"] is None


def test_breakdown_by_source():
    rows = [_row("a"), _row("b", source="llm"), _row("c", source="llm")]
    ds = [_d("a", "440"), _d("b", "400"), _d("c", "440", "review")]
    table = {r["value"]: r for r in breakdown(rows, ds, "source")}
    assert table["rules"] == {"value": "rules", "n": 1, "auto_share": 1.0, "auto_precision": 1.0, "top1": 1.0}
    assert table["llm"]["auto_precision"] == 0.0 and table["llm"]["top1"] == 0.5


# --- cost ---------------------------------------------------------------------------------


def test_cost_report_math():
    ds = [_d("a", "440", cost=0.002, latency=1000), _d("b", "440", cost=0.004, latency=3000), _d("c", "440", cached=True, latency=5)]
    report = cost_report(ds)
    assert report["calls_per_1000"] == pytest.approx(2 / 3 * 1000)  # cached answers are not calls
    assert report["cost_per_1000_usd"] == pytest.approx(0.006 / 3 * 1000)
    assert report["median_latency_ms"] == 2000  # cached ones excluded
    assert report["cache_hit_rate"] == pytest.approx(1 / 3)


def test_cost_report_free_methods():
    report = cost_report([_d("a", "440"), _d("b", "440")])
    assert report["cost_per_1000_usd"] == 0.0 and report["calls_per_1000"] == 0.0
