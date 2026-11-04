import os

import pytest

from d1.datagen.safe_write import DataChangedError, safe_write


def _old_mtime(path):
    os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    return path.stat().st_mtime_ns


def test_new_file_is_written(tmp_path):
    path = tmp_path / "sub" / "dev.jsonl"
    assert safe_write({path: "a\nb\n"}) == {path: "new"}
    assert path.read_text(encoding="utf-8") == "a\nb\n"


def test_identical_file_is_left_untouched(tmp_path):
    path = tmp_path / "dev.jsonl"
    path.write_text("a\nb\n", encoding="utf-8")
    before = _old_mtime(path)
    assert safe_write({path: "a\nb\n"}) == {path: "unchanged"}
    assert path.stat().st_mtime_ns == before  # not even rewritten with the same content


def test_changed_file_is_refused_and_nothing_is_written(tmp_path):
    changed, new = tmp_path / "dev.jsonl", tmp_path / "test.jsonl"
    changed.write_text("a\nb\nc\n", encoding="utf-8")
    with pytest.raises(DataChangedError) as e:
        safe_write({new: "x\n", changed: "a\nB\nc\nd\n"})
    message = str(e.value)
    assert "dev.jsonl" in message and "2 of 4 lines differ" in message and "--force" in message
    assert changed.read_text(encoding="utf-8") == "a\nb\nc\n"  # the existing data is kept
    assert not new.exists()  # all or nothing: no file is written when one is refused


def test_force_replaces(tmp_path):
    path = tmp_path / "dev.jsonl"
    path.write_text("old\n", encoding="utf-8")
    assert safe_write({path: "new\n"}, force=True) == {path: "replaced"}
    assert path.read_text(encoding="utf-8") == "new\n"
    assert [p.name for p in tmp_path.iterdir()] == ["dev.jsonl"]  # no temporary file left behind
