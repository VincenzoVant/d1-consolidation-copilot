from unittest.mock import Mock

import pytest

from d1.config import load_config
from d1.datagen.__main__ import main
from d1.llm import LLMError


def test_build_offline_without_frozen_files_gives_hint(small_root, capsys):
    factory = Mock()
    assert main(["build"], backend_factory=factory, root=small_root) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "fetch-llm" in err and err.count("\n") == 1
    factory.assert_not_called()  # build never needs an LLM


def test_fetch_then_build(small_root, fake_writer, capsys):
    assert main(["fetch-llm", "--split", "all"], backend_factory=lambda config: fake_writer, root=small_root) == 0
    assert "5 new frozen files" in capsys.readouterr().out

    factory = Mock()
    assert main(["build"], backend_factory=factory, root=small_root) == 0
    out = capsys.readouterr().out
    assert "train: 80 accounts" in out and "dev: 50 accounts" in out and "test: 80 accounts" in out
    assert (load_config(small_root).data.generated_dir / "test.jsonl").is_file()
    factory.assert_not_called()


def test_fetch_failure_returns_1_with_one_error_line(small_root, capsys):
    class Broken:
        def complete(self, *args, **kwargs):
            raise LLMError("claude CLI timed out after 300s")

    assert main(["fetch-llm", "--split", "test"], backend_factory=lambda config: Broken(), root=small_root) == 1
    captured = capsys.readouterr()
    assert "timed out" in captured.out  # each failed company is listed in the progress output
    assert captured.err.startswith("error: 2 of 2 companies failed") and captured.err.count("\n") == 1


def test_backend_that_cannot_start_returns_1(small_root, capsys):
    def factory(config):
        raise LLMError("llm_backend is anthropic_api but ANTHROPIC_API_KEY is not set")

    assert main(["fetch-llm", "--split", "test"], backend_factory=factory, root=small_root) == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_bad_config_returns_1(tmp_path, capsys):
    assert main(["build"], root=tmp_path) == 1  # no config/ folder at all
    assert "data.yaml" in capsys.readouterr().err


def test_split_and_workers_are_validated():
    with pytest.raises(SystemExit):
        main(["fetch-llm", "--split", "prod"])
    with pytest.raises(SystemExit):
        main(["fetch-llm", "--split", "test", "--workers", "0"])


# --- export-review / apply-review --------------------------------------------------------


def _built(small_root, fake_writer):
    main(["fetch-llm", "--split", "all"], backend_factory=lambda config: fake_writer, root=small_root)
    main(["build"], root=small_root)
    return load_config(small_root)


def test_export_review_needs_a_built_test_split(small_root, capsys):
    assert main(["export-review"], root=small_root) == 1
    assert "build" in capsys.readouterr().err


def test_export_fill_and_apply_review(small_root, fake_writer, capsys):
    config = _built(small_root, fake_writer)
    capsys.readouterr()
    assert main(["export-review"], root=small_root) == 0
    assert "30 rows" in capsys.readouterr().out  # 20 LLM rows + review_rule_sample 10

    path = config.data.review_path
    text = path.read_text(encoding="utf-8-sig").splitlines()
    header = text[0].split(";")
    filled = [text[0]] + [";".join("ok" if col == "verdict" else cell for col, cell in zip(header, line.split(";"))) for line in text[1:]]
    path.write_text("\n".join(filled) + "\n", encoding="utf-8-sig")

    assert main(["apply-review"], root=small_root) == 0
    out = capsys.readouterr().out
    assert "llm: ok 20, fix 0, drop 0" in out and "rules: ok 10, fix 0, drop 0" in out
    assert "fix rate: 0.0% (0 of 10)" in out
    checked = config.data.generated_dir / "test_checked.jsonl"
    assert checked.is_file() and len(checked.read_text(encoding="utf-8").splitlines()) == 80


def test_apply_review_lists_the_problems(small_root, fake_writer, capsys):
    _built(small_root, fake_writer)
    main(["export-review"], root=small_root)
    capsys.readouterr()
    assert main(["apply-review"], root=small_root) == 1  # nothing filled in yet
    err = capsys.readouterr().err
    assert err.startswith("error: 30 problems in the review:") and "missing verdict" in err
