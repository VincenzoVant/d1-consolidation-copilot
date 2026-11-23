"""The LLM method: Claude reads the account, its neighbours and the candidates, and picks one.

Like Laya it learns nothing from our data (zero-shot). What it gets that no other
method gets: the names of the accounts just above and below in the company's chart,
and an LLM's knowledge of accounting. The prompt is prompts/choose_group.md.

One call can carry several accounts (batch_size): consecutive accounts of one
company, each with its own candidate list. The answer is JSON in a fixed shape
(answer_schema), one entry per account: a choice ("none" allowed), a confidence
from 0 to 100 and a one-line reason. Everything that can go wrong ends as a status
on that account, never as a crash:

  ok        the choice is one of the account's own candidates
  none      the LLM said none of them fits
  invalid   the choice is not one of this account's candidates
  error     the call failed (CLI error, timeout, not logged in) or the answer was malformed
  capped    the call cap (max_calls) was reached, so no call was made

Every answer is kept in an AnswerCache, so a question is asked once: training pays,
the evaluation run and the notebook don't. Errors are never cached.
"""

import hashlib
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from d1.config import REPO_ROOT, Config
from d1.datagen.schema import CompanyAccount, StandardAccount
from d1.datagen.standard_chart import load_standard_chart
from d1.llm import LLMBackend, LLMError, make_backend
from d1.mapper.answer_cache import AnswerCache, cache_key
from d1.mapper.methods.base import (
    AccountIn, MethodOutput, check_lengths, read_method_info, register, save_method_info,
)  # fmt: skip
from d1.mapper.retrieve import Candidate, Retriever

PROMPT_PATH = REPO_ROOT / "prompts" / "choose_group.md"
PROMPT_VERSION = "v1"
SYSTEM = "You map company accounts to groups of the Belgian standard chart of accounts. Return JSON only."
NONE = "none"
MAX_CALLS = 5000  # a runaway loop can't make more calls than this per method
MAX_WORKERS = 8  # calls at the same time: each CLI call is its own process
TIMEOUT_S = 300.0
SAVE_EVERY = 20  # answers (calls) written to the cache per batch

Item = tuple[str, AccountIn, list[Candidate]]  # (label in the prompt, the account, its shortlist)


# --- the question ----------------------------------------------------------------------------------------


def candidate_line(group: StandardAccount) -> str:
    part_of = f" (part of: {group.parent_name_en})" if group.parent_name_en else ""
    return f"- {group.code}: {group.name_en} / {group.name_nl} / {group.name_fr}{part_of}"


def _quoted(names: list[str]) -> str:
    return "; ".join(f'"{n}"' for n in names)


def account_block(label: str, account: AccountIn, shortlist: list[Candidate], groups: dict[str, StandardAccount]) -> str:
    lines = [f"## Account {label}", f"Name: {account.name}", f"Company code: {account.code or '(none)'}"]
    if account.before:
        lines.append(f"Just above it: {_quoted(account.before)}")
    if account.after:
        lines.append(f"Just below it: {_quoted(account.after)}")
    lines.append("Candidate groups:")
    lines += [candidate_line(groups[c.code]) for c in shortlist]
    return "\n".join(lines)


def build_prompt(items: list[Item], groups: dict[str, StandardAccount]) -> str:
    blocks = "\n\n".join(account_block(label, account, shortlist, groups) for label, account, shortlist in items)
    return PROMPT_PATH.read_text(encoding="utf-8").format(n_accounts=len(items), accounts=blocks)


def answer_schema(items: list[Item]) -> dict:
    """One answer per account label; the choice is any candidate of the batch, or none (checked per account later)."""
    codes = list(dict.fromkeys(c.code for _, _, shortlist in items for c in shortlist))
    answer = {
        "type": "object",
        "properties": {
            "account": {"type": "string", "enum": [label for label, _, _ in items]},
            "choice": {"type": "string", "enum": codes + [NONE]},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            "reason": {"type": "string"},
        },
        "required": ["account", "choice", "confidence", "reason"],
        "additionalProperties": False,
    }
    return {"type": "object", "properties": {"answers": {"type": "array", "items": answer}}, "required": ["answers"], "additionalProperties": False}


def parse_answers(structured: dict, labels: list[str]) -> dict[str, dict]:
    """label -> {choice, confidence, reason}; every label exactly once, or ValueError."""
    answers = structured.get("answers") if isinstance(structured, dict) else None
    if not isinstance(answers, list):
        raise ValueError("the answer has no list of answers")
    out: dict[str, dict] = {}
    for a in answers:
        if not isinstance(a, dict) or a.get("account") not in labels:
            raise ValueError(f"an answer for an account that wasn't asked: {a!r:.80}")
        label = a["account"]
        if label in out:
            raise ValueError(f"account {label} answered twice")
        confidence = a.get("confidence")
        if not isinstance(a.get("choice"), str) or not isinstance(a.get("reason"), str):
            raise ValueError(f"account {label}: choice or reason missing")
        if isinstance(confidence, bool) or not isinstance(confidence, int) or not 0 <= confidence <= 100:
            raise ValueError(f"account {label}: confidence {confidence!r} is not a whole number from 0 to 100")
        out[label] = {"choice": a["choice"], "confidence": confidence, "reason": a["reason"]}
    missing = [label for label in labels if label not in out]
    if missing:
        raise ValueError(f"no answer for account(s) {', '.join(missing)}")
    return out


def make_batches(accounts: list[AccountIn], candidates: list[list[Candidate]], size: int) -> list[list[int]]:
    """Indices of accounts with a shortlist, cut into runs of at most size from one company, in input order."""
    batches: list[list[int]] = []
    for i, (account, shortlist) in enumerate(zip(accounts, candidates)):
        if not shortlist:
            continue
        last = batches[-1] if batches else None
        if last and len(last) < size and accounts[last[-1]].company_id == account.company_id and account.company_id is not None:
            last.append(i)
        else:
            batches.append([i])
    return batches


# --- the method --------------------------------------------------------------------------------------------


class LLMMethod:
    name = "llm"
    needs_training = False

    def __init__(
        self,
        backend: LLMBackend,
        model: str,
        chart: list[StandardAccount],
        cache: AnswerCache | None = None,
        batch_size: int = 1,
        max_calls: int = MAX_CALLS,
        timeout_s: float = TIMEOUT_S,
        max_workers: int = MAX_WORKERS,
        progress: bool = False,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self.backend = backend
        self.model = model
        self.cache = cache
        self.batch_size = batch_size
        self.max_calls = max_calls
        self.timeout_s = timeout_s
        self.max_workers = max_workers
        self.progress = progress  # a line every 20 calls: a dev run takes many minutes
        self.calls_made = 0  # over the method's life, so the cap holds across calls to score()
        self.last_usage: dict = {}  # what the last score() asked, and what its answers cost the first time
        self._groups = {a.code: a for a in chart}

    @property
    def effort(self) -> str | None:
        return getattr(self.backend, "effort", None)

    def fit(self, train_rows: list[CompanyAccount], candidates: list[list[Candidate]]) -> None:
        pass  # zero-shot: nothing is learned from our data

    def _ask(self, prompt: str, schema: dict, labels: list[str]) -> dict:
        """One call; the record that is cached. Raises LLMError or ValueError (malformed answer)."""
        response = self.backend.complete(prompt, schema=schema, model=self.model, system=SYSTEM, timeout_s=self.timeout_s)
        return {"answers": parse_answers(response.structured, labels), "model": response.model, "cost_usd": response.cost_usd or 0.0,
                "latency_ms": response.latency_ms, "input_tokens": response.input_tokens, "output_tokens": response.output_tokens}  # fmt: skip

    def score(self, accounts: list[AccountIn], candidates: list[list[Candidate]]) -> list[MethodOutput]:
        check_lengths(accounts, candidates)
        start = time.perf_counter()
        version = self.answer_version()
        jobs = []  # (indices, labels, prompt, schema, key) per batch
        for batch in make_batches(accounts, candidates, self.batch_size):
            items = [(f"A{j + 1}", accounts[i], candidates[i]) for j, i in enumerate(batch)]
            prompt, schema = build_prompt(items, self._groups), answer_schema(items)
            jobs.append((batch, [label for label, _, _ in items], prompt, schema, cache_key(version, SYSTEM, prompt, schema)))
        cached = self.cache.get_many([job[4] for job in jobs]) if self.cache is not None else {}

        todo = list(dict.fromkeys(job[4] for job in jobs if job[4] not in cached))  # a repeated question is asked once
        allowed = max(0, self.max_calls - self.calls_made)
        to_call, capped = set(todo[:allowed]), set(todo[allowed:])
        self.calls_made += len(to_call)
        fresh: dict[str, dict] = {}
        failed: dict[str, str] = {}
        if to_call:
            first = {}  # key -> the first job asking it
            for job in jobs:
                first.setdefault(job[4], job)
            unsaved: dict[str, dict] = {}
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {pool.submit(self._ask, first[k][2], first[k][3], first[k][1]): k for k in todo[:allowed]}
                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        fresh[key] = unsaved[key] = future.result()
                    except (LLMError, ValueError) as e:  # this batch goes to review; the others go on
                        failed[key] = f"llm: {e}"
                    done = len(fresh) + len(failed)
                    if self.cache is not None and (len(unsaved) >= SAVE_EVERY or done == len(to_call)):
                        self.cache.put_many(unsaved)  # saved as it goes: a crash keeps what was answered
                        unsaved = {}
                    if self.progress and (done % SAVE_EVERY == 0 or done == len(to_call)):
                        print(f"llm: {done:,} of {len(to_call):,} calls answered ({len(cached):,} were cached, {len(failed)} failed)", flush=True)

        keys = list(dict.fromkeys(job[4] for job in jobs))
        records = [cached[k] if k in cached else fresh[k] for k in keys if k in cached or k in fresh]
        self.last_usage = {
            "calls": len(to_call), "cached_calls": sum(k in cached for k in keys), "failed_calls": len(failed), "capped_calls": len(capped),
            "wall_s": time.perf_counter() - start,
            # A cached answer is free now, but it was paid for once: this is what the answers cost when first asked.
            "first_time_cost_usd": sum(r["cost_usd"] for r in records),
            "median_call_ms": statistics.median(r["latency_ms"] for r in records) if records else 0.0,
            "input_tokens": sum(r["input_tokens"] for r in records), "output_tokens": sum(r["output_tokens"] for r in records),
        }  # fmt: skip

        out: list[MethodOutput | None] = [None] * len(accounts)
        for batch, labels, _, _, key in jobs:
            for i, label in zip(batch, labels):
                out[i] = self._output(accounts[i], candidates[i], label, key, len(batch), cached, fresh, failed, capped)
        for i, account in enumerate(accounts):
            if out[i] is None:  # no shortlist: nothing to choose from
                out[i] = MethodOutput(account_id=account.account_id, scores={}, chosen_code=None, status="none", reason="empty shortlist")
        return out

    def _output(self, account, shortlist, label, key, batch_len, cached, fresh, failed, capped) -> MethodOutput:
        aid = account.account_id
        if key in capped:
            return MethodOutput(account_id=aid, scores={}, chosen_code=None, status="capped", reason=f"llm: the cap of {self.max_calls:,} calls was reached")
        if key in failed:
            return MethodOutput(account_id=aid, scores={}, chosen_code=None, status="error", reason=failed[key])
        hit = key in cached
        record = cached[key] if hit else fresh[key]
        # A fresh call's cost is shared by the accounts in it; each of them waited for the whole call.
        cost, ms = (0.0, 0.0) if hit else (record["cost_usd"] / batch_len, record["latency_ms"])
        answer = record["answers"][label]
        choice, reason = answer["choice"], answer["reason"]
        if choice == NONE:
            return MethodOutput(account_id=aid, scores={}, chosen_code=None, status="none", reason=f"llm: none of these groups ({reason})",
                                cost_usd=cost, latency_ms=ms, cached=hit)  # fmt: skip
        if choice not in {c.code for c in shortlist}:
            return MethodOutput(account_id=aid, scores={}, chosen_code=None, status="invalid",
                                reason=f"llm: chose {choice}, which is not one of this account's candidates", cost_usd=cost, latency_ms=ms, cached=hit)  # fmt: skip
        return MethodOutput(account_id=aid, scores={choice: answer["confidence"] / 100}, chosen_code=choice, reason=reason,
                            cost_usd=cost, latency_ms=ms, cached=hit)  # fmt: skip

    # --- identity and saving --------------------------------------------------------------------------------

    def answer_version(self) -> str:
        """What an answer depends on besides the prompt text itself: model, prompt, batch size, effort."""
        prompt_sha = hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest()[:7]
        return f"llm-{self.model}-prompt-{PROMPT_VERSION}-{prompt_sha}-b{self.batch_size}-effort-{self.effort or 'default'}"

    def version(self) -> str:
        return self.answer_version()

    def save(self, folder: Path) -> None:
        """Nothing learned to save: only the settings the calibration was fitted on."""
        save_method_info(Path(folder), self.name, self.version(), model=self.model, prompt_version=PROMPT_VERSION,
                         prompt_sha256=hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest(), batch_size=self.batch_size,
                         effort=self.effort)  # fmt: skip

    def load(self, folder: Path) -> None:
        """Refuses a folder calibrated with another model, prompt, batch size or effort: its threshold wouldn't fit."""
        info = read_method_info(Path(folder), self.name)
        if info["version"] != self.version():
            raise ValueError(f"{folder} was calibrated as {info['version']}, but this is {self.version()}: retrain it")


@register("llm")
def _llm(config: Config, retriever: Retriever) -> LLMMethod:
    chart = load_standard_chart(config.data.chart_path, expected=config.data.expected_groups)
    mapping = config.models.mapping
    backend = make_backend(config, effort=mapping.llm_effort)
    cache = AnswerCache(config.root / ".cache" / "llm" / "answers.sqlite")
    return LLMMethod(backend, config.models.mapping_model, chart, cache=cache, batch_size=mapping.llm_batch_size, progress=True)

