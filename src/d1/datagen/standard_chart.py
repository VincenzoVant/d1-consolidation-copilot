"""Load the Belgian standard chart (Odoo l10n_be account groups) as mapping targets.

The source file mixes three levels of rows:
  1 digit  - a class, e.g. "6" Expenses
  2 digits - a parent group, e.g. "60" Goods for Resale, Raw Materials and Consumables
  3 digits - a group, e.g. "604" Purchases of Goods for Resale  <- the mapping targets
Only 3-digit groups are returned; the other rows supply parent and class names.

Quirks of the pinned file (see data/source/odoo_l10n_be/SOURCE.md):
  - some groups cover a range ("164" to "165"), given in code_prefix_end;
  - group "132" has no 2-digit parent row, so its parent name falls back to the class name.
"""

import csv
from pathlib import Path

from pydantic import ValidationError

from d1.datagen.schema import StandardAccount

# Column in the Odoo file -> language. English is the file's base "name" column.
NAME_COLUMNS = {"en": "name", "fr": "name@fr", "nl": "name@nl"}
REQUIRED_COLUMNS = {"code_prefix_start", "code_prefix_end", *NAME_COLUMNS.values()}


class ChartError(Exception):
    """The standard chart file is missing, malformed or not what we expect."""


def _read_rows(path: Path) -> list[dict[str, str]]:
    try:
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise ChartError(f"{path.name}: missing columns {', '.join(sorted(missing))}")
            return [{k: (v or "").strip() for k, v in row.items()} for row in reader]
    except FileNotFoundError:
        raise ChartError(f"standard chart not found at {path}") from None


def _index_by_code(rows: list[dict[str, str]], path: Path) -> dict[str, dict[str, str]]:
    by_code: dict[str, dict[str, str]] = {}
    for row in rows:
        code = row["code_prefix_start"]
        if not (code.isdigit() and 1 <= len(code) <= 3):
            raise ChartError(f"{path.name}: unexpected group code {code!r}")
        if code in by_code:
            raise ChartError(f"{path.name}: duplicate group code {code}")
        by_code[code] = row
    return by_code


def _names(row: dict[str, str], code: str, path: Path) -> dict[str, str]:
    names = {lang: row[column] for lang, column in NAME_COLUMNS.items()}
    empty = [lang for lang, name in names.items() if not name]
    if empty:
        raise ChartError(f"{path.name}: group {code} has no {'/'.join(empty)} name")
    return names


def load_standard_chart(path: Path, expected: int | None = None) -> list[StandardAccount]:
    """Return the 3-digit groups of the chart at path, sorted by code.

    If expected is given, the number of groups must match it exactly, so a
    changed source file fails loudly instead of silently changing the targets.
    """
    path = Path(path)
    by_code = _index_by_code(_read_rows(path), path)

    accounts = []
    for code in sorted(c for c in by_code if len(c) == 3):
        row = by_code[code]
        names = _names(row, code, path)

        class_row = by_code.get(code[0])
        if class_row is None:
            raise ChartError(f"{path.name}: group {code} has no class row {code[0]}")
        class_name_en = _names(class_row, code[0], path)["en"]

        parent_row = by_code.get(code[:2])
        parent_code = code[:2] if parent_row is not None else None
        parent_name_en = _names(parent_row, code[:2], path)["en"] if parent_row else class_name_en

        try:
            accounts.append(
                StandardAccount(
                    code=code,
                    code_end=row["code_prefix_end"] or None,
                    parent_code=parent_code,
                    class_code=code[0],
                    name_en=names["en"],
                    name_fr=names["fr"],
                    name_nl=names["nl"],
                    parent_name_en=parent_name_en,
                    class_name_en=class_name_en,
                )
            )
        except ValidationError as e:
            raise ChartError(f"{path.name}: group {code} is invalid: {e}") from None

    if expected is not None and len(accounts) != expected:
        raise ChartError(f"{path.name}: expected {expected} 3-digit groups, found {len(accounts)}")
    return accounts
