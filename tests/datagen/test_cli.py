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
