"""Hand check of the test split: export rows to a CSV, read the verdicts back.

Rule-based rows have labels correct by construction; an LLM-written row only has
the group the LLM said it meant. So a person checks every LLM-written test row,
plus a random sample of rule-based rows (noise can make a name ambiguous):

  export_review -> data/labels/test_review.csv  (open in Excel, fill in verdict)
  apply_review  -> the test rows with label_status checked / corrected / constructed

Verdicts: ok (the target is right), fix (put the right group in corrected_target),
drop (no group fits; the row leaves the test set).
"""

import csv
import io
from collections import Counter
from pathlib import Path
from random import Random

from pydantic import BaseModel

from d1.datagen.schema import CompanyAccount, StandardAccount

REVIEW_COLUMNS = [
    "account_id", "company_id", "industry", "language", "source", "code", "name",
    "target_code", "target_name_en", "target_name_local", "verdict", "corrected_target", "note",
]  # fmt: skip
VERDICTS = ("ok", "fix", "drop")
REQUIRED_COLUMNS = {"account_id", "name", "verdict", "corrected_target"}
DELIMITER = ";"  # Excel in a Belgian locale splits columns on ";", not ","


class ReviewError(Exception):
    """The review file is missing or has problems; messages lists every one of them."""

    def __init__(self, messages: list[str]):
        self.messages = messages
        if len(messages) == 1:
            text = messages[0]
        else:
            shown = "\n  - ".join(messages[:20])
            more = f"\n  ... and {len(messages) - 20} more" if len(messages) > 20 else ""
            text = f"{len(messages)} problems in the review:\n  - {shown}{more}"
        super().__init__(text)


class ReviewResult(BaseModel):
    rows: list[CompanyAccount]  # the whole test split, dropped rows removed
    counts: dict[str, dict[str, int]]  # source -> verdict -> number of rows
    rule_fix_rate: float | None  # share of sampled rule-based rows that needed fix or drop


def review_selection(rows: list[CompanyAccount], rule_sample: int, seed: int) -> list[CompanyAccount]:
    """Every LLM-written row plus a seeded random sample of rule-based rows, in account-id order."""
    llm = [r for r in rows if r.source == "llm"]
    rules = sorted((r for r in rows if r.source == "rules"), key=lambda r: r.account_id)
    if rule_sample > len(rules):
        raise ValueError(f"cannot sample {rule_sample}: only {len(rules)} rule-based rows")
    return sorted(llm + Random(seed).sample(rules, rule_sample), key=lambda r: r.account_id)


def _local_name(account: StandardAccount, language: str) -> str:
    if language == "mixed":
        return f"{account.name_nl} / {account.name_fr}"
    return account.name(language)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a review file however Excel saved it: BOM or not, UTF-8 or cp1252, ; , or tab."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")  # Excel's plain "CSV" format on a Western-European machine
    header = text.split("\n", 1)[0]
    delimiter = max((";", ",", "\t"), key=header.count)
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter)
    fields = {(name or "").strip() for name in reader.fieldnames or []}
    missing = REQUIRED_COLUMNS - fields
    if missing:
        raise ReviewError([f"{path.name}: missing columns {', '.join(sorted(missing))}"])
    return [{(k or "").strip(): (v or "").strip() for k, v in line.items() if k} for line in reader]


def export_review(rows: list[CompanyAccount], chart: list[StandardAccount], path: Path, rule_sample: int, seed: int) -> int:
    """Write the rows to check to path; returns how many.

    Refuses to overwrite a file that already has a verdict: that would throw
    away hand work. A file without any verdict is overwritten.
    """
    path = Path(path)
    if path.exists() and any(line.get("verdict") for line in _read_csv(path)):
        raise ReviewError([f"{path.name} already has verdicts; move it away first if you really want a fresh export"])
    by_code = {a.code: a for a in chart}
    selected = review_selection(rows, rule_sample, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig adds a BOM, which tells Excel the file is UTF-8 (otherwise "é" shows as "Ã©").
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS, delimiter=DELIMITER)
        writer.writeheader()
        for r in selected:
            target = by_code[r.target_code]
            writer.writerow({
                "account_id": r.account_id, "company_id": r.company_id, "industry": r.industry,
                "language": r.language, "source": r.source, "code": r.code, "name": r.name,
                "target_code": r.target_code, "target_name_en": target.name_en,
                "target_name_local": _local_name(target, r.language),
                "verdict": "", "corrected_target": "", "note": "",
            })  # fmt: skip
    return len(selected)


def apply_review(rows: list[CompanyAccount], chart: list[StandardAccount], path: Path, rule_sample: int, seed: int) -> ReviewResult:
    """Apply the verdicts in path to the test rows.

    Every problem in the file is collected first and reported together, so one
    run lists everything to fix. Nothing is applied while any problem remains.
    """
    path = Path(path)
    if not path.exists():
        raise ReviewError([f"{path} not found: run `python -m d1.datagen export-review` first"])
    by_id = {r.account_id: r for r in rows}
    selected = {r.account_id for r in review_selection(rows, rule_sample, seed)}
    codes = {a.code for a in chart}

    problems: list[str] = []
    seen: set[str] = set()
    decisions: dict[str, tuple[str, str]] = {}
    for line_no, line in enumerate(_read_csv(path), start=2):  # line 1 is the header
        account_id = line.get("account_id", "")
        where = f"line {line_no} ({account_id})"
        if account_id not in by_id:
            problems.append(f"{where}: unknown account_id")
            continue
        if account_id in seen:
            problems.append(f"{where}: appears twice")
            continue
        seen.add(account_id)
        if account_id not in selected:
            problems.append(f"{where}: not part of the review selection")
            continue
        row = by_id[account_id]
        if line["name"] != row.name.strip():
            # The generator changed after the export: the verdict was about a different row.
            problems.append(f"{where}: name changed since the export ({line['name']!r} -> {row.name!r})")
            continue
        verdict, corrected = line["verdict"].lower(), line["corrected_target"]
        if not verdict:
            problems.append(f"{where}: missing verdict")
        elif verdict not in VERDICTS:
            problems.append(f"{where}: unknown verdict {line['verdict']!r} (use ok, fix or drop)")
        elif verdict == "fix" and corrected not in codes:
            problems.append(f"{where}: corrected_target {corrected!r} is not a group in the standard chart")
        elif verdict == "fix" and corrected == row.target_code:
            problems.append(f"{where}: corrected_target equals the current target; use ok")
        elif verdict != "fix" and corrected:
            problems.append(f"{where}: corrected_target is filled but the verdict is {verdict}")
        else:
            decisions[account_id] = (verdict, corrected)
    problems += [f"{a}: selected for review but missing from {path.name}" for a in sorted(selected - seen)]
    if problems:
        raise ReviewError(problems)

    counts = {"llm": Counter(), "rules": Counter()}
    out: list[CompanyAccount] = []
    for row in rows:
        if row.account_id not in decisions:
            # Only rule-based rows can be outside the selection: all LLM rows are checked.
            out.append(row.model_copy(update={"label_status": "constructed"}))
            continue
        verdict, corrected = decisions[row.account_id]
        counts[row.source][verdict] += 1
        if verdict == "ok":
            out.append(row.model_copy(update={"label_status": "checked"}))
        elif verdict == "fix":
            out.append(row.model_copy(update={"label_status": "corrected", "target_code": corrected}))
    rule_changed = counts["rules"]["fix"] + counts["rules"]["drop"]
    return ReviewResult(
        rows=out,
        counts={source: {v: c[v] for v in VERDICTS} for source, c in counts.items()},
        rule_fix_rate=rule_changed / rule_sample if rule_sample else None,
    )
