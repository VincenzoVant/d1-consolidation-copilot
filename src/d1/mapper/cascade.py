"""The cascade: the first method maps what it is sure of, the LLM looks at the rest.

  account -> first method (XGBoost) -> sure enough?  yes -> auto (the first method's pick)
                                        no  -> the LLM, on the same shortlist -> sure enough? yes -> cascade_llm
                                                                                             no  -> review

Why: the LLM picks the right group more often than XGBoost (88.5% against 84.5% on dev),
but auto-maps fewer accounts. XGBoost is right on its sure accounts, and those cost nothing;
the LLM is then only asked about the quarter it leaves, where its knowledge helps most.

Claude's question in the cascade is the ranked one (mapping.cascade_llm_choices: 5): its five
most likely groups with probabilities, and a one-line reason for the first. An account that
isn't auto-mapped goes to the reviewer with exactly Claude's ranking, the reason on top.

The LLM's second-stage calibrator and thresholds are fitted on those leftover dev accounts
only (train.train_cascade), not taken from the LLM method alone: the leftovers are the hard
accounts, and the same stated confidence means less there. Each stage, in each segment,
holds the 98% on its own, so the cascade's auto-mapped accounts hold it together too.

Artifacts: artifacts/methods/cascade/
  cascade.json      which first method, its fingerprint (a retrained first method invalidates the cascade)
  second/           the LLM as fitted on the leftovers: method.json, calibrator.json, thresholds.json
  dev_summary.json  what the cascade does on dev
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from d1.config import Config, Method
from d1.mapper.embed import Embedder, SentenceTransformerEmbedder
from d1.mapper.methods import METHOD_REGISTRY, AccountIn, MappingMethod
from d1.mapper.pipeline import Mapper, MapperError, MappingDecision, Thresholds, artifact_dir
from d1.mapper.retrieve import Candidate, Retriever

CASCADE = "cascade"
SECOND = "llm"
FINGERPRINT_FILES = ("method.json", "calibrator.json", "thresholds.json")


class CascadeThresholds(BaseModel):
    """Both stages' thresholds, as the evaluation reports them."""

    method: str = CASCADE
    first: Thresholds
    second: Thresholds
    target_precision: float
    threshold_by: str


def fingerprint(folder: Path) -> str:
    """A hash of a trained method's saved files: changes whenever the method is trained again."""
    h = hashlib.sha256()
    for name in FINGERPRINT_FILES:
        h.update(name.encode())
        h.update((folder / name).read_bytes())
    return h.hexdigest()


def second_method(config: Config, retriever: Retriever) -> MappingMethod:
    """Claude as the cascade asks it: the LLM method with cascade_llm_choices choices (when set)."""
    method = METHOD_REGISTRY[SECOND](config, retriever)
    n = config.models.mapping.cascade_llm_choices
    if n is not None:
        method.n_choices = n
    return method


def merge(first: MappingDecision, second: MappingDecision) -> MappingDecision:
    """One account the first method wasn't sure of, after Claude looked at it.

    - Claude is sure enough: cascade_llm, Claude's pick;
    - Claude picked, but not surely: review;
    - Claude said none fits, or put another account's group first, but still ranked this account's
      candidates: review, its first-ranked group as the suggestion, its status and reason kept;
    - Claude gave no ranking at all (the call failed or was capped): review with the first method's
      list, so the reviewer still has options, and Claude's status and reason, so they see why.
    The reviewer reads Claude's ranking only, never filled up with the first method's candidates,
    and Claude's reason sits on the top option.
    """
    if not second.candidates:
        reason = f"llm {second.status}" + (f": {second.reason}" if second.reason else "")
        return first.model_copy(update={"status": second.status, "reason": reason, "cost_usd": second.cost_usd,
                                        "latency_ms": second.latency_ms, "cached": second.cached})  # fmt: skip
    if second.chosen_code is None:
        top, *others = second.candidates
        candidates = [top.model_copy(update={"reason": second.reason}), *others]
        return second.model_copy(update={"chosen_code": top.code, "candidates": candidates})
    path = "cascade_llm" if second.decision_path == "auto" else "review"
    return second.model_copy(update={"decision_path": path})


def combine(first: list[MappingDecision], rest: list[int], second: list[MappingDecision], versions: dict[str, str]) -> list[MappingDecision]:
    """The first stage's decisions, with the accounts at positions rest replaced by their merged second-stage decision."""
    out = list(first)
    for i, d in zip(rest, second, strict=True):
        out[i] = merge(first[i], d)
    return [d.model_copy(update={"model_versions": versions}) for d in out]


class CascadeMapper:
    """Two Mappers one after the other; maps like a Mapper (same input, same decisions)."""

    def __init__(self, first: Mapper, second: Mapper):
        self.first = first
        self.second = second
        self.retriever = first.retriever
        self.k = first.k

    @property
    def thresholds(self) -> CascadeThresholds:
        t = self.first.thresholds
        return CascadeThresholds(first=t, second=self.second.thresholds, target_precision=t.target_precision, threshold_by=t.threshold_by)

    def versions(self) -> dict[str, str]:
        first, second = self.first.versions(), self.second.versions()
        return {"embedder": first["embedder"], "first_method": first["method"], "first_calibrator": first["calibrator"],
                "second_method": second["method"], "second_calibrator": second["calibrator"]}  # fmt: skip

    def map(self, accounts: list[AccountIn], shortlists: list[list[Candidate]] | None = None) -> list[MappingDecision]:
        if not accounts:
            return []
        if shortlists is None:
            shortlists = self.retriever.top_k([a.name for a in accounts], self.k)
        first = self.first.map(accounts, shortlists)
        rest = [i for i, d in enumerate(first) if d.decision_path != "auto"]
        second = self.second.map([accounts[i] for i in rest], [shortlists[i] for i in rest])
        return combine(first, rest, second, self.versions())

    @classmethod
    def from_artifacts(cls, config: Config, embedder_factory: Callable[[str], Embedder] = SentenceTransformerEmbedder) -> "CascadeMapper":
        folder = artifact_dir(config, CASCADE)
        retrain = "run `python -m d1.mapper.train --method cascade`"
        if not (folder / "cascade.json").exists():
            raise MapperError(f"{folder} has no trained cascade: {retrain}")
        info = json.loads((folder / "cascade.json").read_text(encoding="utf-8"))
        first = Mapper.from_artifacts(config, info["first"], embedder_factory=embedder_factory)
        if fingerprint(artifact_dir(config, info["first"])) != info["first_fingerprint"]:
            raise MapperError(f"{info['first']} was trained again after the cascade was fitted on it: {retrain} again")
        second = Mapper.from_folder(config, info["second"], folder / "second", first.retriever, trained_as=CASCADE,
                                    method=second_method(config, first.retriever))
        return cls(first, second)


def load_mapper(
    config: Config, name: Method | str | None = None, embedder_factory: Callable[[str], Embedder] = SentenceTransformerEmbedder
) -> Mapper | CascadeMapper:
    """The cascade for name "cascade", otherwise the Mapper of one method (default: selected_method)."""
    if name == CASCADE:
        return CascadeMapper.from_artifacts(config, embedder_factory)
    return Mapper.from_artifacts(config, name, embedder_factory=embedder_factory)
