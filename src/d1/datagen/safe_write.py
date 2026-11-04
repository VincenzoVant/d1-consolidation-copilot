"""Write generated data without ever silently replacing what later steps were built on.

The results, the notebooks and the hand check all read data/generated/. If a
rebuild would produce different content (say the generator code changed), the
existing file is kept and the command stops, unless it is run with --force.
A rebuild that produces the same content leaves the file untouched.
"""

import os
from pathlib import Path
from typing import Literal

Status = Literal["new", "unchanged", "replaced"]


class DataChangedError(Exception):
    """A rebuild would change data that already exists."""


def _differing_lines(old: str, new: str) -> tuple[int, int]:
    a, b = old.splitlines(), new.splitlines()
    return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b)), max(len(a), len(b))


def safe_write(outputs: dict[Path, str], force: bool = False, rerun_hint: str = "run it again with --force") -> dict[Path, Status]:
    """Write each text to its path: new files always, changed files only with force.

    All or nothing: every path is checked before any is written, so a refused
    file never leaves the others half updated.
    """
    status: dict[Path, Status] = {}
    refused = []
    for path, text in outputs.items():
        path = Path(path)
        if not path.exists():
            status[path] = "new"
            continue
        old = path.read_text(encoding="utf-8")
        if old == text:
            status[path] = "unchanged"
        else:
            status[path] = "replaced"
            changed, total = _differing_lines(old, text)
            refused.append(f"{path.name} would change ({changed} of {total} lines differ)")
    if refused and not force:
        raise DataChangedError(
            "; ".join(refused) + ". Results, notebooks and the hand check were made on the current file, "
            f"so it was kept and nothing was written. If the change is intended, {rerun_hint}."
        )
    for path, text in outputs.items():
        path = Path(path)
        if status[path] == "unchanged":
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)  # a crash mid-write never leaves half a file under the real name
    return status
