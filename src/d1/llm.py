"""One way to ask an LLM for a JSON answer, whichever route the call takes.

Every LLM call in D1 (writing company charts, the LLM mapping method, the
cascade) goes through LLMBackend.complete(). Two backends implement it:

  ClaudeCLIBackend     runs the `claude` command on your own Claude login (default)
  AnthropicAPIBackend  calls the Anthropic API with ANTHROPIC_API_KEY (for the deployed service)

Both return the same LLMResponse and turn every failure into an LLMError, so
callers handle one exception type and can route the account to review.
"""

import json
import os
import subprocess
import tempfile
import time
from collections.abc import Callable
from typing import Protocol

import anthropic
from pydantic import BaseModel, ConfigDict

from d1.config import Config

LOGIN_HINT = "run `claude` once and log in"


class LLMError(Exception):
    """An LLM call failed; the message says what went wrong and, where possible, what to do."""


class LLMResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    structured: dict  # the answer, already parsed and shaped by the JSON schema
    model: str  # the model that actually answered
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None  # reported by the CLI; None from the API
    latency_ms: float


class LLMBackend(Protocol):
    def complete(self, prompt: str, schema: dict, model: str, system: str, timeout_s: float = 120.0) -> LLMResponse: ...


class ClaudeCLIBackend:
    """Runs `claude -p` as a separate program, once per call.

    runner defaults to subprocess.run; tests pass a fake so no real call is made.
    """

    def __init__(self, executable: str = "claude", runner: Callable[..., subprocess.CompletedProcess] = subprocess.run):
        self.executable = executable
        self.runner = runner

    def command(self, schema: dict, model: str, system: str) -> list[str]:
        return [
            self.executable,
            "-p",  # print mode: answer once and exit; the prompt arrives on stdin
            "--output-format", "json",
            "--json-schema", json.dumps(schema),
            "--model", model,
            "--system-prompt", system,  # replaces Claude Code's own system prompt
            # Keep each call clean: no tools, no MCP servers, none of your settings,
            # plugins or hooks, no slash commands, nothing saved to session history.
            # Not --bare: bare mode only accepts ANTHROPIC_API_KEY and ignores your login.
            "--tools", "",
            "--strict-mcp-config",
            "--setting-sources", "",
            "--disable-slash-commands",
            "--no-session-persistence",
        ]  # fmt: skip

    def complete(self, prompt: str, schema: dict, model: str, system: str, timeout_s: float = 120.0) -> LLMResponse:
        # Without an API key in its environment the CLI always uses your login.
        env = {key: value for key, value in os.environ.items() if key != "ANTHROPIC_API_KEY"}
        start = time.perf_counter()
        # An empty working directory, so no project CLAUDE.md is read into the call.
        with tempfile.TemporaryDirectory(prefix="d1-claude-") as cwd:
            try:
                proc = self.runner(
                    self.command(schema, model, system),
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    cwd=cwd,
                    env=env,
                )
            except FileNotFoundError:
                raise LLMError(
                    f"'{self.executable}' was not found: install Claude Code, or set llm_backend: anthropic_api"
                ) from None
            except subprocess.TimeoutExpired:
                raise LLMError(f"claude CLI timed out after {timeout_s:g}s") from None
        latency_ms = (time.perf_counter() - start) * 1000
        return _parse_cli_output(proc, model, latency_ms)


def _parse_cli_output(proc: subprocess.CompletedProcess, requested_model: str, latency_ms: float) -> LLMResponse:
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        result = None
    if not isinstance(result, dict):
        if proc.returncode != 0:
            raise _with_login_hint(f"claude CLI failed with exit code {proc.returncode}: {proc.stderr.strip()}")
        raise LLMError(f"claude CLI output is not JSON: {proc.stdout[:200]!r}")
    if result.get("is_error") or proc.returncode != 0:
        raise _with_login_hint(f"claude CLI returned an error ({result.get('subtype', 'unknown')}): {result.get('result', '')}")

    structured = result.get("structured_output")
    if not isinstance(structured, dict):
        raise LLMError(f"claude CLI returned no structured answer (subtype {result.get('subtype', 'unknown')})")

    usage = result.get("usage") or {}
    # Input sent fresh, written to the prompt cache, or read from it: all of it was input.
    input_tokens = sum(usage.get(key, 0) for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    # modelUsage is keyed by the model that answered; if there are several, the one that wrote the most.
    models = result.get("modelUsage") or {}
    served = max(models, key=lambda m: models[m].get("outputTokens", 0)) if models else requested_model
    return LLMResponse(
        structured=structured,
        model=served,
        input_tokens=input_tokens,
        output_tokens=usage.get("output_tokens", 0),
        cost_usd=result.get("total_cost_usd"),
        latency_ms=latency_ms,
    )


def _with_login_hint(message: str) -> LLMError:
    lowered = message.lower()
    if "not logged in" in lowered or "/login" in lowered:
        message += f" - {LOGIN_HINT}"
    return LLMError(message)


class AnthropicAPIBackend:
    """Calls the Messages API with a JSON-schema output format."""

    def __init__(self, client: anthropic.Anthropic):
        self.client = client

    def complete(self, prompt: str, schema: dict, model: str, system: str, timeout_s: float = 120.0) -> LLMResponse:
        start = time.perf_counter()
        try:
            message = self.client.messages.create(
                model=model,
                max_tokens=16000,
                system=system,
                output_config={"format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout_s,
            )
        except anthropic.APIError as e:
            raise LLMError(f"Anthropic API call failed: {e}") from None
        latency_ms = (time.perf_counter() - start) * 1000

        if message.stop_reason in ("refusal", "max_tokens"):
            raise LLMError(f"Anthropic API stopped with {message.stop_reason}; no usable answer")
        text = next((block.text for block in message.content if block.type == "text"), None)
        if text is None:
            raise LLMError("Anthropic API returned no structured answer")
        try:
            structured = json.loads(text)
        except json.JSONDecodeError:
            raise LLMError(f"Anthropic API answer is not JSON: {text[:200]!r}") from None
        return LLMResponse(
            structured=structured,
            model=message.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            latency_ms=latency_ms,
        )


def make_backend(config: Config) -> LLMBackend:
    """The backend named by llm_backend in config/models.yaml."""
    if config.models.llm_backend == "claude_cli":
        return ClaudeCLIBackend()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise LLMError("llm_backend is anthropic_api but ANTHROPIC_API_KEY is not set: set it, or use llm_backend: claude_cli")
    return AnthropicAPIBackend(anthropic.Anthropic())
