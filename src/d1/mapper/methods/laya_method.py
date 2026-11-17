"""Laya: a pretrained decision model that picks among the candidates, zero-shot.

Laya (convaiinnovations/laya-multilingual) answers typed questions about a text in
one forward pass, without generating text. For each account it gets one `choice`
question: which of the 10 shortlisted groups (each shown with its English, Dutch
and French name), or "none of these". It is not trained on our data at all.

Two things its authors document, and this method handles:

  position bias   where an option sits changes the answer. So the question is asked
                  under 6 rotations, each option in 6 different slots, all in one
                  forward pass, and the probabilities are averaged. (All 11 rotations
                  took 373 ms per account against 225 ms, with top-1 within noise on
                  300 dev accounts: 45.0% against 43.7%.)
  over-confidence the multilingual checkpoint ships without fitted temperatures, so
                  its raw probabilities are far too sure. The calibration on dev
                  (calibrate.py), the same as for every method, corrects that.

The model runs locally (on the Mac's GPU): no money per call, only time. Answers are
kept in an AnswerCache, so a question asked once (in training) is instant afterwards
(in the evaluation run and the notebook).
"""

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from laya.revisions import PINNED_REVISIONS

from d1.config import Config
from d1.datagen.schema import CompanyAccount, StandardAccount
from d1.datagen.standard_chart import load_standard_chart
from d1.mapper.answer_cache import AnswerCache, cache_key
from d1.mapper.methods.base import (
    AccountIn, MethodOutput, check_lengths, read_method_info, register, save_method_info,
)  # fmt: skip
from d1.mapper.retrieve import Candidate, Retriever

INSTRUCTIONS = "Which Belgian standard account group does this company account belong to?"
NONE = "none"
NONE_TEXT = "none of these groups"
# Laya splits its question budget evenly over the options. With the default 256 tokens and
# 11 options, each gets 21 tokens, which cuts the French and often the Dutch name off
# (median 32 tokens for all three). 640 gives 56 per option: all three names for 90% of groups.
HEAD_MAX_LEN = 640
ROTATIONS = 6
SAVE_EVERY = 200  # answers written to the cache per batch


def option_text(group: StandardAccount) -> str:
    return f"{group.name_en} / {group.name_nl} / {group.name_fr}"


def _load_agent(model_id: str, revision: str | None) -> Any:
    import laya  # imported here: it pulls in torch only when the model is really needed

    return laya.load(model_id, revision=revision)


class LayaMethod:
    name = "laya"
    needs_training = False

    def __init__(
        self,
        model_id: str,
        chart: list[StandardAccount],
        revision: str | None = None,
        agent_loader: Callable[[str, str | None], Any] | None = _load_agent,
        rotations: int = ROTATIONS,
        head_max_len: int = HEAD_MAX_LEN,
        cache: AnswerCache | None = None,
        progress: bool = False,
    ):
        if rotations < 1:
            raise ValueError(f"rotations must be at least 1, got {rotations}")
        self.model_id = model_id
        self.revision = revision
        self.rotations = rotations
        self.head_max_len = head_max_len
        self.cache = cache
        self.progress = progress  # print a line every 200 questions (the command-line runs take minutes)
        self._groups = {a.code: a for a in chart}
        self._loader = agent_loader
        self._agent = None  # loaded on first use: building the method costs nothing

    # --- the question ------------------------------------------------------------------------------

    def question(self, shortlist: list[Candidate]) -> dict:
        """One choice question: the candidates in shortlist order, each with its three names, then "none"."""
        criteria = {c.code: option_text(self._groups[c.code]) for c in shortlist} | {NONE: NONE_TEXT}
        return {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}

    def shifts(self, k: int) -> list[int]:
        """Which rotations to ask for k options: `rotations` shifts spread evenly (6 of 11: 0, 1, 3, 5, 7, 9)."""
        n = min(self.rotations, k)
        return [(j * k) // n for j in range(n)]

    def questions(self, shortlist: list[Candidate]) -> dict[str, dict]:
        """The question under each shift r: slot i shows option (i + r) mod k. One shift (0) is the listed order."""
        base = self.question(shortlist)
        k = len(base["criteria"])
        if self.rotations == 1:
            return {"q0": base}
        return {f"q{r}": base | {"option_order": [(i + r) % k for i in range(k)]} for r in self.shifts(k)}

    @staticmethod
    def state(account: AccountIn) -> dict:
        body = account.name if not account.code else f"{account.name} (code {account.code})"
        return {"body": body}

    # --- scoring --------------------------------------------------------------------------------------

    def _get_agent(self) -> Any:
        if self._agent is None:
            os.environ["USE_TF"] = "0"  # upstream: with TensorFlow installed, loading can crash the process
            self._agent = self._loader(self.model_id, self.revision)
        return self._agent

    def fit(self, train_rows: list[CompanyAccount], candidates: list[list[Candidate]]) -> None:
        pass  # zero-shot: nothing is learned from our data

    def _ask(self, state: dict, questions: dict) -> dict[str, float]:
        """Laya's probabilities per label, averaged over the rotations (they come back keyed by label)."""
        answers = self._get_agent().predict(state, questions, head_max_len=self.head_max_len)["answers"]
        labels = list(next(iter(questions.values()))["criteria"])
        return {label: sum(a["probabilities"][label] for a in answers.values()) / len(answers) for label in labels}

    def score(self, accounts: list[AccountIn], candidates: list[list[Candidate]]) -> list[MethodOutput]:
        check_lengths(accounts, candidates)
        version = self.version()
        jobs = []  # (account, shortlist, state, questions, key) for every account with candidates
        for account, shortlist in zip(accounts, candidates):
            if shortlist:
                state, questions = self.state(account), self.questions(shortlist)
                jobs.append((account, shortlist, state, questions, cache_key(version, state, questions)))
        cached = self.cache.get_many([job[4] for job in jobs]) if self.cache is not None else {}

        fresh: dict[str, dict[str, float]] = {}  # answers computed in this call, by key: a repeated question is asked once
        failed: dict[str, str] = {}
        latency: dict[str, float] = {}
        unsaved: dict[str, dict[str, float]] = {}
        todo = len({job[4] for job in jobs} - cached.keys())
        for _, _, state, questions, key in jobs:
            if key in cached or key in fresh or key in failed:
                continue
            start = time.perf_counter()
            try:
                fresh[key] = unsaved[key] = self._ask(state, questions)
            except (RuntimeError, ValueError) as e:  # e.g. the GPU running out of memory: this account goes to review
                failed[key] = f"laya: {e}"
            latency[key] = (time.perf_counter() - start) * 1000
            done = len(fresh) + len(failed)
            # Saved as it goes, so a crash at account 3,000 keeps the first 3,000 (errors are never stored).
            if self.cache is not None and (len(unsaved) >= SAVE_EVERY or done == todo):
                self.cache.put_many(unsaved)
                unsaved = {}
            if self.progress and (done % SAVE_EVERY == 0 or done == todo):
                print(f"laya: {done:,} of {todo:,} questions asked ({len(cached):,} were cached)", flush=True)
        if self.cache is not None and unsaved:
            self.cache.put_many(unsaved)

        by_account = {job[0].account_id: job for job in jobs}
        out = []
        for account, shortlist in zip(accounts, candidates):
            if not shortlist:
                out.append(MethodOutput(account_id=account.account_id, scores={}, chosen_code=None, status="none", reason="empty shortlist"))
                continue
            key = by_account[account.account_id][4]
            if key in failed:
                out.append(MethodOutput(account_id=account.account_id, scores={}, chosen_code=None, status="error", reason=failed[key]))
                continue
            probs = cached.get(key) or fresh[key]
            hit = key in cached
            ms = 0.0 if hit else latency[key]
            scores = dict(sorted(((c.code, probs[c.code]) for c in shortlist), key=lambda kv: -kv[1]))
            labels = [c.code for c in shortlist] + [NONE]
            best = max(labels, key=lambda label: probs[label])  # ties: the earlier label, as max() keeps the first
            if best == NONE:
                out.append(MethodOutput(account_id=account.account_id, scores=scores, chosen_code=None, status="none",
                                        reason=f"laya: none of these groups (p = {probs[NONE]:.2f})", latency_ms=ms, cached=hit))  # fmt: skip
            else:
                out.append(MethodOutput(account_id=account.account_id, scores=scores, chosen_code=best, latency_ms=ms, cached=hit))
        return out

    # --- identity and saving ------------------------------------------------------------------------------

    def version(self) -> str:
        revision = (self.revision or "unpinned")[:7]
        return f"laya-{self.model_id}@{revision}-rot{self.rotations}-h{self.head_max_len}-v1"

    def save(self, folder: Path) -> None:
        """No weights to save (the model is downloaded, pinned by revision): only the settings."""
        save_method_info(Path(folder), self.name, self.version(), model_id=self.model_id, revision=self.revision,
                         rotations=self.rotations, head_max_len=self.head_max_len)  # fmt: skip

    def load(self, folder: Path) -> None:
        """Refuses a folder calibrated on another model or revision: its threshold would not fit these scores."""
        params = read_method_info(Path(folder), self.name)["params"]
        if params["model_id"] != self.model_id:
            raise ValueError(f"{folder} was calibrated with {params['model_id']}, but the config says {self.model_id}: retrain it")
        if params["revision"] != self.revision:
            raise ValueError(f"{folder} was calibrated with revision {params['revision']}, but this is {self.revision}: retrain it")
        self.rotations = int(params["rotations"])
        self.head_max_len = int(params["head_max_len"])


@register("laya")
def _laya(config: Config, retriever: Retriever) -> LayaMethod:
    chart = load_standard_chart(config.data.chart_path, expected=config.data.expected_groups)
    model_id = config.models.mapping.laya_model
    cache = AnswerCache(config.root / ".cache" / "laya" / "answers.sqlite")
    return LayaMethod(model_id, chart, revision=PINNED_REVISIONS.get(model_id), cache=cache, progress=True)
