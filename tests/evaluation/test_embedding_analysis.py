import pandas as pd
import pytest

from d1.evaluation.embedding_analysis import (
    company_bootstrap_ci, head_to_head, load_rows, paired_difference, recall_by, recall_curve,
)  # fmt: skip
from d1.evaluation.select_embedder import ROWS_COLUMNS


def _df(ranks_a, ranks_b, companies=None, noise=None, languages=None):
    """Two models, a and b, over the same accounts; ranks are the right group's rank."""
    n = len(ranks_a)
    companies = companies or [f"c{i % 2}" for i in range(n)]
    noise = noise or [[] for _ in range(n)]
    languages = languages or ["nl"] * n
    rows = []
    for model, ranks in (("a", ranks_a), ("b", ranks_b)):
        for i, rank in enumerate(ranks):
            rows.append({
                "model": model, "account_id": f"dev-{i:03d}", "company_id": companies[i], "language": languages[i],
                "source": "rules", "code_scheme": "legacy", "noise_tags": noise[i], "name": f"name {i}",
                "target_code": "440", "gold_rank": rank, "top1_score": 0.9, "top2_score": 0.8,
            })  # fmt: skip
    return pd.DataFrame(rows)


def test_load_rows_reads_the_csv_written_by_the_comparison(tmp_path):
    path = tmp_path / "rows.csv"
    header = ",".join(ROWS_COLUMNS)
    line_1 = "m,dev-c000-000,dev-c000,nl,rules,legacy,typo;case,NA,040,12,0.81,0.80,040;100;101"
    line_2 = "m,dev-c000-001,dev-c000,fr,llm,llm,,Fournisseurs,440,1,0.9,0.7,440;400;439"
    path.write_text("\n".join([header, line_1, line_2]) + "\n", encoding="utf-8")
    df = load_rows(path)
    assert df["target_code"].tolist() == ["040", "440"]  # codes stay text: "040" is not 40
    assert df["name"].tolist() == ["NA", "Fournisseurs"]  # "NA" is an account name, not a missing value
    assert df["noise_tags"].tolist() == [["typo", "case"], []]
    assert df["top3_codes"].tolist() == [["040", "100", "101"], ["440", "400", "439"]]
    assert df["gold_rank"].dtype.kind == "i" and df["top1_score"].dtype.kind == "f"


def test_recall_curve_counts_ranks_up_to_k():
    df = _df([1, 2, 5, 50], [3, 3, 3, 3])
    curve = recall_curve(df, [1, 3, 10])
    assert curve.loc[1, "a"] == 0.25 and curve.loc[3, "a"] == 0.5 and curve.loc[10, "a"] == 0.75
    assert curve.loc[1, "b"] == 0.0 and curve.loc[3, "b"] == 1.0
    assert list(curve.columns) == ["a", "b"]  # models keep the order they appear in


def test_recall_by_language():
    df = _df([1, 20, 1, 1], [1, 1, 20, 20], languages=["nl", "nl", "fr", "fr"])
    table = recall_by(df, "language", k=10)
    assert table.loc["nl", "a"] == 0.5 and table.loc["fr", "a"] == 1.0
    assert table.loc["nl", "b"] == 1.0 and table.loc["fr", "b"] == 0.0
    assert table.loc["nl", "n"] == 2


def test_recall_by_noise_counts_a_row_once_per_tag():
    noise = [["typo", "case"], ["typo"], [], ["case"]]
    df = _df([1, 20, 1, 20], [1, 1, 1, 1], noise=noise)
    table = recall_by(df, "noise", k=10)
    assert table.loc["typo", "n"] == 2 and table.loc["typo", "a"] == 0.5
    assert table.loc["case", "n"] == 2 and table.loc["case", "a"] == 0.5
    assert table.loc["no noise", "n"] == 1 and table.loc["no noise", "a"] == 1.0


def test_bootstrap_ci_brackets_the_point_and_is_seeded():
    ranks = [1, 1, 20, 1, 20, 1, 1, 1, 20, 1] * 5
    df = _df(ranks, [1] * 50, companies=[f"c{i // 5}" for i in range(50)])
    ci = company_bootstrap_ci(df, k=10, n_boot=500, seed=0)
    a = ci.set_index("model").loc["a"]
    assert a["recall"] == pytest.approx(0.7) and a["low"] <= 0.7 <= a["high"] and a["low"] < a["high"]
    b = ci.set_index("model").loc["b"]
    assert b["low"] == b["high"] == 1.0  # every row a hit: no uncertainty left
    assert ci.equals(company_bootstrap_ci(df, k=10, n_boot=500, seed=0))


def test_bootstrap_resamples_whole_companies():
    # Company c0 has every hit, c1 every miss. Resampling companies gives recall 0, 0.5 or 1;
    # resampling single rows would almost never reach 0 or 1 with 40 rows.
    df = _df([1] * 20 + [50] * 20, [1] * 40, companies=["c0"] * 20 + ["c1"] * 20)
    ci = company_bootstrap_ci(df, k=10, n_boot=400, seed=1).set_index("model").loc["a"]
    assert ci["low"] == 0.0 and ci["high"] == 1.0


def test_paired_difference():
    df = _df([1] * 10, [1] * 5 + [50] * 5, companies=[f"c{i}" for i in range(10)])
    diff, low, high = paired_difference(df, "a", "b", k=10, n_boot=500, seed=0)
    assert diff == pytest.approx(0.5) and 0 < low <= 0.5 <= high
    assert paired_difference(df, "a", "a", k=10) == (0.0, 0.0, 0.0)


def test_head_to_head():
    df = _df([1, 1, 20, 20], [1, 20, 1, 20])
    assert head_to_head(df, "a", "b", k=10) == {"both": 1, "only_a": 1, "only_b": 1, "neither": 1}
