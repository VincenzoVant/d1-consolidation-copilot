import csv
from pathlib import Path

import pytest
from pydantic import ValidationError

from d1.config import load_config
from d1.datagen.schema import CompanyAccount, read_jsonl, write_jsonl
from d1.datagen.standard_chart import ChartError, load_standard_chart

REAL_CHART = load_config().data.chart_path


def _copy_fixture(fixtures_dir: Path, dest: Path, edit) -> Path:
    """Copy the mini chart to dest, letting edit(rows) change the rows first."""
    with open(fixtures_dir / "mini_groups.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    edit(rows)
    with open(dest, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return dest


# --- the real Odoo file ---------------------------------------------------


def test_real_chart_has_190_groups():
    chart = load_standard_chart(REAL_CHART, expected=190)
    codes = [a.code for a in chart]
    assert len(chart) == 190
    assert all(len(c) == 3 and c.isdigit() for c in codes)
    assert codes == sorted(codes)
    assert len(set(codes)) == 190


def test_every_real_group_has_en_fr_nl_names():
    for a in load_standard_chart(REAL_CHART):
        assert a.name("en") and a.name("fr") and a.name("nl"), a.code


def test_group_132_falls_back_to_class_name():
    # 132 has no 2-digit parent row in the source file.
    g132 = next(a for a in load_standard_chart(REAL_CHART) if a.code == "132")
    assert g132.parent_code is None
    assert g132.parent_name_en == g132.class_name_en
    assert g132.class_code == "1"


def test_range_groups_keep_code_end():
    chart = {a.code: a for a in load_standard_chart(REAL_CHART)}
    assert chart["164"].code_end == "165"
    assert chart["700"].code_end == "707"
    assert chart["120"].code_end is None
    assert sum(1 for a in chart.values() if a.code_end) == 9


# --- the mini fixture -----------------------------------------------------


def test_mini_chart_keeps_only_three_digit_groups(fixtures_dir):
    chart = load_standard_chart(fixtures_dir / "mini_groups.csv")
    assert [a.code for a in chart] == ["600", "601", "604", "617", "618", "657"]


def test_mini_chart_parents_and_languages(fixtures_dir):
    a604 = next(a for a in load_standard_chart(fixtures_dir / "mini_groups.csv") if a.code == "604")
    assert a604.name_en == "Purchases of Goods for Resale"
    assert a604.parent_code == "60"
    assert a604.parent_name_en == "Goods for Resale, Raw Materials and Consumables"
    assert a604.class_code == "6"
    assert a604.class_name_en == "Expenses"
    assert a604.name("nl") == a604.name_nl and a604.name("fr") == a604.name_fr


def test_mini_657_has_no_parent_and_a_range(fixtures_dir):
    # The fixture leaves out the "65" row on purpose.
    a657 = next(a for a in load_standard_chart(fixtures_dir / "mini_groups.csv") if a.code == "657")
    assert a657.parent_code is None
    assert a657.parent_name_en == "Expenses"
    assert a657.code_end == "658"


# --- errors ---------------------------------------------------------------


def test_count_mismatch_raises(fixtures_dir):
    with pytest.raises(ChartError, match="expected 5"):
        load_standard_chart(fixtures_dir / "mini_groups.csv", expected=5)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ChartError, match="not found"):
        load_standard_chart(tmp_path / "nope.csv")


def test_duplicate_code_raises(fixtures_dir, tmp_path):
    path = _copy_fixture(fixtures_dir, tmp_path / "dup.csv", lambda rows: rows.append(dict(rows[-1])))
    with pytest.raises(ChartError, match="duplicate"):
        load_standard_chart(path)


def test_missing_name_raises(fixtures_dir, tmp_path):
    def blank_604_dutch(rows):
        next(r for r in rows if r["code_prefix_start"] == "604")["name@nl"] = ""

    path = _copy_fixture(fixtures_dir, tmp_path / "blank.csv", blank_604_dutch)
    with pytest.raises(ChartError, match="604"):
        load_standard_chart(path)


# --- company accounts (schema.py) -----------------------------------------


def _company_account(**overrides) -> CompanyAccount:
    fields = dict(
        account_id="dev-c00-000",
        company_id="dev-c00",
        industry="retail",
        split="dev",
        source="rules",
        language="nl",
        code_scheme="pcmn",
        code="604000",
        name="Aankopen handelsgoederen",
        noise=["abbreviation"],
        target_code="604",
    )
    return CompanyAccount(**(fields | overrides))


def test_company_account_defaults():
    acc = _company_account()
    assert acc.overlaps_dev is False
    assert acc.label_status == "unchecked"


def test_company_account_rejects_bad_values():
    with pytest.raises(ValidationError):
        _company_account(name="")
    with pytest.raises(ValidationError):
        _company_account(language="de")
    with pytest.raises(ValidationError):
        _company_account(split="validation")


def test_jsonl_roundtrip(tmp_path):
    rows = [
        _company_account(),
        _company_account(account_id="dev-c00-001", name="Créances / Vorderingen", language="mixed", noise=[]),
    ]
    path = tmp_path / "out" / "rows.jsonl"
    write_jsonl(rows, path)
    assert read_jsonl(path) == rows
