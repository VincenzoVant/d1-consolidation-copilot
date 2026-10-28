import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx
import pytest
import yaml

from d1.config import load_config
from d1.llm import AnthropicAPIBackend, ClaudeCLIBackend, LLMError, LLMResponse, make_backend

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


@pytest.fixture
def cli_result(fixtures_dir) -> dict:
    """The JSON the real `claude -p` printed for "Answer ok=true." (captured once)."""
    return json.loads((fixtures_dir / "claude_cli_result.json").read_text())


class FakeRunner:
    """Stands in for subprocess.run: records the call, returns or raises what it is given."""

    def __init__(self, stdout="", returncode=0, stderr="", raises=None):
        self.stdout, self.returncode, self.stderr, self.raises = stdout, returncode, stderr, raises
        self.calls = []

    def __call__(self, argv, **kwargs):
        # The working directory only exists during the call, so inspect it now.
        cwd = Path(kwargs["cwd"])
        self.calls.append({"argv": argv, "cwd_was_empty": not any(cwd.iterdir()), **kwargs})
        if self.raises:
            raise self.raises
        return subprocess.CompletedProcess(argv, self.returncode, self.stdout, self.stderr)


def _cli(runner, **kw):
    return ClaudeCLIBackend(runner=runner).complete(
        "Answer ok=true.", SCHEMA, "claude-haiku-5-5", "Return JSON only.", **kw
    )


def _value_after(argv, flag):
    return argv[argv.index(flag) + 1]


# --- Claude CLI ------------------------------------------------------------


def test_cli_command_has_isolation_flags(cli_result):
    runner = FakeRunner(stdout=json.dumps(cli_result))
    _cli(runner, timeout_s=60)
    call = runner.calls[0]
    argv = call["argv"]
    assert argv[:2] == ["claude", "-p"]
    assert _value_after(argv, "--output-format") == "json"
    assert json.loads(_value_after(argv, "--json-schema")) == SCHEMA
    assert _value_after(argv, "--model") == "claude-haiku-5-5"
    assert _value_after(argv, "--system-prompt") == "Return JSON only."
    assert _value_after(argv, "--tools") == ""
    assert _value_after(argv, "--setting-sources") == ""
    for flag in ("--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
        assert flag in argv
    assert "--bare" not in argv  # bare mode ignores the Claude login
    # The prompt goes in on stdin, never as an argument that could look like a flag.
    assert call["input"] == "Answer ok=true." and "Answer ok=true." not in argv
    assert call["cwd_was_empty"] and call["timeout"] == 60


def test_cli_uses_the_login_not_an_api_key(cli_result, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    runner = FakeRunner(stdout=json.dumps(cli_result))
    _cli(runner)
    assert "ANTHROPIC_API_KEY" not in runner.calls[0]["env"]
    assert runner.calls[0]["env"]["PATH"] == os.environ["PATH"]


def test_cli_parses_captured_fixture(cli_result):
    response = _cli(FakeRunner(stdout=json.dumps(cli_result)))
    assert isinstance(response, LLMResponse)
    assert response.structured == {"ok": True}
    assert response.model == "claude-haiku-5-5"
    assert response.input_tokens == 2 + 1145  # new input + input written to the prompt cache
    assert response.output_tokens == 52
    assert response.cost_usd == pytest.approx(0.0002552)
    assert response.latency_ms >= 0


def test_cli_nonzero_exit_raises_llm_error():
    with pytest.raises(LLMError, match="exit code 1.*boom"):
        _cli(FakeRunner(returncode=1, stderr="boom"))


def test_cli_not_installed_raises_with_install_hint():
    with pytest.raises(LLMError, match="install Claude Code"):
        _cli(FakeRunner(raises=FileNotFoundError("claude")))


def test_cli_timeout_raises():
    with pytest.raises(LLMError, match="timed out after 60s"):
        _cli(FakeRunner(raises=subprocess.TimeoutExpired("claude", 60)), timeout_s=60)


def test_cli_is_error_result_raises(cli_result):
    cli_result.update(is_error=True, subtype="error_during_execution", result="overloaded")
    with pytest.raises(LLMError, match="error_during_execution.*overloaded"):
        _cli(FakeRunner(stdout=json.dumps(cli_result)))


def test_cli_not_logged_in_gives_login_hint(cli_result):
    cli_result.update(is_error=True, result="Not logged in · Please run /login")
    with pytest.raises(LLMError, match="run `claude` once and log in"):
        _cli(FakeRunner(stdout=json.dumps(cli_result), returncode=1))


def test_cli_missing_structured_answer_raises(cli_result):
    del cli_result["structured_output"]
    with pytest.raises(LLMError, match="no structured answer"):
        _cli(FakeRunner(stdout=json.dumps(cli_result)))


def test_cli_output_not_json_raises():
    with pytest.raises(LLMError, match="not JSON"):
        _cli(FakeRunner(stdout="Something went wrong"))


# --- Anthropic API -----------------------------------------------------------


class FakeMessages:
    def __init__(self, response=None, raises=None):
        self.response, self.raises, self.kwargs = response, raises, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.raises:
            raise self.raises
        return self.response


def _api_response(stop_reason="end_turn", text='{"ok": true}'):
    return SimpleNamespace(
        stop_reason=stop_reason,
        model="claude-haiku-5-5",
        content=[SimpleNamespace(type="text", text=text)] if text else [],
        usage=SimpleNamespace(input_tokens=20, output_tokens=5),
    )


def _api(messages, **kw):
    client = SimpleNamespace(messages=messages)
    return AnthropicAPIBackend(client).complete("Answer ok=true.", SCHEMA, "claude-haiku-5-5", "Return JSON only.", **kw)


def test_api_backend_parses_answer_and_sends_schema():
    messages = FakeMessages(_api_response())
    response = _api(messages, timeout_s=30)
    assert response.structured == {"ok": True}
    assert (response.model, response.input_tokens, response.output_tokens) == ("claude-haiku-5-5", 20, 5)
    assert response.cost_usd is None  # the API does not report cost; the report computes it from tokens
    assert messages.kwargs["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}}
    assert messages.kwargs["system"] == "Return JSON only."
    assert messages.kwargs["messages"] == [{"role": "user", "content": "Answer ok=true."}]
    assert messages.kwargs["timeout"] == 30


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_api_backend_refusal_raises(stop_reason):
    with pytest.raises(LLMError, match=stop_reason):
        _api(FakeMessages(_api_response(stop_reason=stop_reason, text=None)))


def test_api_backend_api_error_raises():
    error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
    with pytest.raises(LLMError, match="Anthropic API"):
        _api(FakeMessages(raises=error))


# --- make_backend ------------------------------------------------------------


def _set_backend(root: Path, value: str) -> None:
    path = root / "config" / "models.yaml"
    data = yaml.safe_load(path.read_text())
    data["llm_backend"] = value
    path.write_text(yaml.safe_dump(data))


def test_make_backend_follows_config(config_root, monkeypatch):
    assert isinstance(make_backend(load_config(config_root)), ClaudeCLIBackend)
    _set_backend(config_root, "anthropic_api")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert isinstance(make_backend(load_config(config_root)), AnthropicAPIBackend)


def test_make_backend_api_without_key_raises(config_root, monkeypatch):
    _set_backend(config_root, "anthropic_api")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        make_backend(load_config(config_root))
