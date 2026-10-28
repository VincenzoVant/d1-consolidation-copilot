import csv
from random import Random

import pytest

from d1.config import load_config
from d1.datagen.handcheck import REVIEW_COLUMNS, ReviewError, apply_review, export_review, review_selection
from d1.datagen.rule_charts import generate_rule_company
from d1.datagen.schema import CompanyAccount
from d1.datagen.standard_chart import load_standard_chart

RULE_SAMPLE, SEED = 5, 2002


@pytest.fixture(scope="module")
def chart():
    return load_standard_chart(load_config().data.chart_path, expected=190)


def _llm_rows(chart):
    """10 LLM-written rows over two companies, one French, one mixed."""
    rows = []
    for c, language in enumerate(["fr", "mixed"]):
        for i, account in enumerate(chart[40 + 5 * c : 45 + 5 * c]):
            rows.append(
                CompanyAccount(
                    account_id=f"test-c1{c}-{i:03d}", company_id=f"test-c1{c}", industry="retail", split="test",
                    source="llm", language=language, code_scheme="llm", code=f"{account.code}0",
                    name=f"Compte {account.name_fr} é", noise=["llm"], target_code=account.code,
                )  # fmt: skip
            )
    return rows


@pytest.fixture
def rows(chart):
    """40 rule-based rows (two companies) + 10 LLM-written rows, like a small test split."""
    rules = []
    for c, language in enumerate(["nl", "en"]):
        rules += generate_rule_company(f"test-c0{c}", "test", "software", language, chart, 20, Random(c))
    return rules + _llm_rows(chart)


def _read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def _write(path, lines, delimiter=";", encoding="utf-8-sig"):
    with open(path, "w", encoding=encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(lines)


def _fill(path, verdict="ok", changes=None, **write_options):
    """Give every line `verdict`, then apply per-account changes {account_id: {column: value}}."""
    lines = _read(path)
    for line in lines:
        line["verdict"] = verdict
        line.update((changes or {}).get(line["account_id"], {}))
    _write(path, lines, **write_options)
    return lines


def _export(rows, chart, path):
    return export_review(rows, chart, path, RULE_SAMPLE, SEED)


def _apply(rows, chart, path):
    return apply_review(rows, chart, path, RULE_SAMPLE, SEED)


# --- choosing what to check ------------------------------------------------------------


def test_selection_is_all_llm_rows_plus_rule_sample(rows):
    selected = review_selection(rows, RULE_SAMPLE, SEED)
    assert len(selected) == 15 and len({r.account_id for r in selected}) == 15
    assert sum(r.source == "llm" for r in selected) == 10
    assert sum(r.source == "rules" for r in selected) == RULE_SAMPLE
    assert [r.account_id for r in selected] == sorted(r.account_id for r in selected)
    assert selected == review_selection(list(reversed(rows)), RULE_SAMPLE, SEED)  # input order does not matter
    assert selected != review_selection(rows, RULE_SAMPLE, SEED + 1)


def test_selection_larger_than_available_raises(rows):
    with pytest.raises(ValueError, match="only 40 rule-based rows"):
        review_selection(rows, 41, SEED)


# --- export ---------------------------------------------------------------------------------


def test_export_writes_an_excel_ready_csv(rows, chart, tmp_path):
    path = tmp_path / "review.csv"
    assert _export(rows, chart, path) == 15
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")  # BOM, so Excel shows accents correctly
    lines = _read(path)
    assert list(lines[0]) == REVIEW_COLUMNS
    assert all(line["verdict"] == line["corrected_target"] == line["note"] == "" for line in lines)
    by_code = {a.code: a for a in chart}
    for line in lines:
        account = by_code[line["target_code"]]
        assert line["target_name_en"] == account.name_en
        expected = {
            "nl": account.name_nl, "fr": account.name_fr, "en": account.name_en,
            "mixed": f"{account.name_nl} / {account.name_fr}",
        }[line["language"]]  # fmt: skip
        assert line["target_name_local"] == expected


def test_export_refuses_to_overwrite_filled_review(rows, chart, tmp_path):
    path = tmp_path / "review.csv"
    _export(rows, chart, path)
    _export(rows, chart, path)  # no verdicts yet: overwriting is fine
    lines = _read(path)
    lines[3]["verdict"] = "ok"
    _write(path, lines)
    with pytest.raises(ReviewError, match="already has verdicts"):
        _export(rows, chart, path)
    assert _read(path)[3]["verdict"] == "ok"  # the hand work is untouched


# --- apply ----------------------------------------------------------------------------------


def test_export_then_apply_all_ok(rows, chart, tmp_path):
    path = tmp_path / "review.csv"
    _export(rows, chart, path)
    _fill(path, "ok")
    result = _apply(rows, chart, path)
    selected = {r.account_id for r in review_selection(rows, RULE_SAMPLE, SEED)}
    assert [r.account_id for r in result.rows] == [r.account_id for r in rows]
    assert [r.target_code for r in result.rows] == [r.target_code for r in rows]
    assert all(r.label_status == "checked" for r in result.rows if r.account_id in selected)
    assert result.counts == {"llm": {"ok": 10, "fix": 0, "drop": 0}, "rules": {"ok": 5, "fix": 0, "drop": 0}}
    assert result.rule_fix_rate == 0.0


def test_unsampled_rule_rows_become_constructed(rows, chart, tmp_path):
    path = tmp_path / "review.csv"
    _export(rows, chart, path)
    _fill(path, "ok")
    result = _apply(rows, chart, path)
    rules = [r for r in result.rows if r.source == "rules"]
    assert sum(r.label_status == "constructed" for r in rules) == 35
    assert sum(r.label_status == "checked" for r in rules) == 5


def test_fix_and_drop(rows, chart, tmp_path):
    path = tmp_path / "review.csv"
    _export(rows, chart, path)
    llm = [r for r in rows if r.source == "llm"]
    new_target = next(a.code for a in chart if a.code != llm[0].target_code)
    _fill(path, "ok", {
        llm[0].account_id: {"verdict": "fix", "corrected_target": new_target, "note": "belongs elsewhere"},
        llm[1].account_id: {"verdict": "Drop"},
    })  # fmt: skip
    result = _apply(rows, chart, path)
    by_id = {r.account_id: r for r in result.rows}
    assert by_id[llm[0].account_id].target_code == new_target
    assert by_id[llm[0].account_id].label_status == "corrected"
    assert llm[1].account_id not in by_id and len(result.rows) == len(rows) - 1
    assert result.counts["llm"] == {"ok": 8, "fix": 1, "drop": 1}


def test_rule_sample_fix_rate(rows, chart, tmp_path):
    path = tmp_path / "review.csv"
    _export(rows, chart, path)
    sampled_rule = next(line for line in _read(path) if line["account_id"].startswith("test-c0"))
    new_target = next(a.code for a in chart if a.code != sampled_rule["target_code"])
    _fill(path, "ok", {sampled_rule["account_id"]: {"verdict": "fix", "corrected_target": new_target}})
    assert _apply(rows, chart, path).rule_fix_rate == pytest.approx(1 / 5)


def test_apply_accepts_excel_belgian_csv(rows, chart, tmp_path):
    path = tmp_path / "review.csv"
    _export(rows, chart, path)
    _fill(path, " OK ", delimiter=";", encoding="utf-8-sig")
    result = _apply(rows, chart, path)
    assert sum(r.label_status == "checked" for r in result.rows) == 15


@pytest.mark.parametrize("delimiter,encoding", [(",", "utf-8"), ("\t", "utf-8"), (";", "cp1252")])
def test_apply_accepts_other_delimiters_and_old_excel_encoding(rows, chart, tmp_path, delimiter, encoding):
    # Excel's plain "CSV" format on a Belgian machine saves as cp1252, not UTF-8.
    path = tmp_path / "review.csv"
    _export(rows, chart, path)
    _fill(path, "ok", delimiter=delimiter, encoding=encoding)
    assert sum(r.label_status == "checked" for r in _apply(rows, chart, path).rows) == 15


def test_apply_reports_every_stale_row(rows, chart, tmp_path):
    path = tmp_path / "review.csv"
    _export(rows, chart, path)
    _fill(path, "ok")
    llm_ids = [r.account_id for r in rows if r.source == "llm"][:3]
    rows2 = [r.model_copy(update={"name": r.name + "x"}) if r.account_id in llm_ids else r for r in rows]
    with pytest.raises(ReviewError) as e:
        _apply(rows2, chart, path)
    assert len([m for m in e.value.messages if "name changed" in m]) == 3


def test_apply_collects_every_problem_at_once(rows, chart, tmp_path):
    path = tmp_path / "review.csv"
    _export(rows, chart, path)
    ids = [line["account_id"] for line in _read(path)]
    lines = _fill(path, "ok", {
        ids[0]: {"verdict": ""},
        ids[1]: {"verdict": "maybe"},
        ids[2]: {"verdict": "fix", "corrected_target": "999"},
        ids[3]: {"verdict": "fix", "corrected_target": ""},
        ids[4]: {"verdict": "ok", "corrected_target": "604"},
    })  # fmt: skip
    lines.append({**lines[5]})  # a duplicated line
    lines.append({**lines[6], "account_id": "test-c99-000"})  # an id that does not exist
    _write(path, [line for line in lines if line["account_id"] != ids[7]])  # a selected row deleted
    with pytest.raises(ReviewError) as e:
        _apply(rows, chart, path)
    text = "\n".join(e.value.messages)
    for expected in [
        f"{ids[0]}): missing verdict",
        f"{ids[1]}): unknown verdict 'maybe'",
        f"{ids[2]}): corrected_target '999' is not a group",
        f"{ids[3]}): corrected_target '' is not a group",
        f"{ids[4]}): corrected_target is filled but the verdict is ok",
        f"{ids[5]}): appears twice",
        "(test-c99-000): unknown account_id",
        f"{ids[7]}: selected for review but missing",
    ]:
        assert expected in text, expected
    assert len(e.value.messages) == 8


def test_apply_without_review_file_names_the_command(rows, chart, tmp_path):
    with pytest.raises(ReviewError, match="export-review"):
        _apply(rows, chart, tmp_path / "missing.csv")
