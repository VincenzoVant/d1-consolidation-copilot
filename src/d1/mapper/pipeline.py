"""From an account to a decision: auto-map it, or send it to a person.

  account name -> shortlist (retrieve.py) -> a method scores and picks (methods/)
               -> calibrated confidence (calibrate.py) -> auto if confidence >= t_auto, else review

The threshold t_auto and the calibrator are fitted per method on dev by
train.py and saved under artifacts/methods/<method>/; Mapper.from_artifacts
loads them. Every decision records which embedder, method and calibrator made it.
The cascade (two methods one after the other) is in cascade.py.
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

from d1.config import Config, Method
from d1.datagen.schema import StandardAccount
from d1.datagen.standard_chart import load_standard_chart
from d1.mapper.calibrate import SEGMENTS, Calibrator, features, segment_of
from d1.mapper.descriptions import DescriptionError, load_descriptions
from d1.mapper.embed import Embedder, SentenceTransformerEmbedder
from d1.mapper.methods import METHOD_REGISTRY, AccountIn, MappingMethod, MethodOutput
from d1.mapper.retrieve import Candidate, Retriever

ARTIFACTS_DIR = Path("artifacts") / "methods"
REVIEW_OPTIONS = 5  # how many groups a reviewer is shown for an account that isn't auto-mapped


class MapperError(Exception):
    """A problem the user can fix: a setting, a missing file, a mismatch."""


class MappingDecision(BaseModel):
    account_id: str
    method: Method
    candidates: list[Candidate]  # what the method considered, best first: what a reviewer sees
    confidence: float  # calibrated P(chosen_code is right); 0 when there is no pick
    decision_path: Literal["auto", "cascade_llm", "review"]
    chosen_code: str | None  # kept on review too, as the suggestion
    reason: str | None = None
    status: str
    model_versions: dict[str, str]
    segment: str | None = None  # code_match / other: which threshold applied (calibrate.segment_of)
    cost_usd: float = 0.0  # what the method's call cost (the LLM), for the cost report
    latency_ms: float = 0.0
    cached: bool = False

    def review_options(self, n: int = REVIEW_OPTIONS) -> list[Candidate]:
        """What a reviewer chooses from: the first n candidates, the pick first, with its reason on it."""
        return self.candidates[:n]


class Thresholds(BaseModel):
    method: Method
    t_auto: float | None  # None: no threshold reached the target precision on dev, so everything is reviewed
    target_precision: float
    fitted_on: Literal["dev"]
    embedding_model: str  # the shortlists these were fitted on; loading with another model is refused
    group_text: str = "names"  # what the groups were embedded as for those shortlists; a mismatch is refused too
    threshold_by: Literal["pooled", "code_match"] = "pooled"  # one threshold for all, or one per segment
    t_auto_by_segment: dict[str, float | None] | None = None  # with code_match: the threshold of each segment

    @model_validator(mode="after")
    def _segments_complete(self) -> "Thresholds":
        if self.threshold_by == "code_match":
            missing = [s for s in SEGMENTS if s not in (self.t_auto_by_segment or {})]
            if missing:
                raise ValueError(f"threshold_by code_match needs a threshold (or None) for every segment; missing: {', '.join(missing)}")
        return self

    def threshold_for(self, segment: str) -> float | None:
        if self.threshold_by == "code_match":
            return self.t_auto_by_segment[segment]
        return self.t_auto


def artifact_dir(config: Config, method: str) -> Path:
    return config.root / ARTIFACTS_DIR / method


def build_retriever(config: Config, embedder: Embedder, chart: list[StandardAccount]) -> Retriever:
    """The shortlist builder as config/models.yaml sets it (mapping.group_text)."""
    group_text = config.models.mapping.group_text
    descriptions = None
    if group_text != "names":
        try:
            descriptions = load_descriptions(config.data.frozen_descriptions_dir, chart)
        except DescriptionError as e:
            raise MapperError(f"group_text is {group_text}, but {e}") from None
    cache = config.root / ".cache" / "embeddings"
    return Retriever(embedder, chart, cache_dir=cache, descriptions=descriptions, group_text=group_text)


def calibrator_hash(calibrator: Calibrator) -> str:
    return hashlib.sha256(json.dumps(calibrator.to_json(), sort_keys=True).encode()).hexdigest()[:12]


class Mapper:
    def __init__(
        self,
        retriever: Retriever,
        method: MappingMethod,
        calibrator: Calibrator,
        thresholds: Thresholds,
        k: int,
    ):
        self.retriever = retriever
        self.method = method
        self.calibrator = calibrator
        self.thresholds = thresholds
        self.k = k

    def map(self, accounts: list[AccountIn], shortlists: list[list[Candidate]] | None = None) -> list[MappingDecision]:
        """One decision per account, in input order.

        shortlists can be passed in when they were already computed (the evaluation
        also measures recall on them), so retrieval isn't run twice.
        """
        if not accounts:
            return []
        if shortlists is None:
            shortlists = self.retriever.top_k([a.name for a in accounts], self.k)
        return self.decide(accounts, self.method.score(accounts, shortlists))

    def versions(self) -> dict[str, str]:
        return {
            "embedder": self.retriever.embedder.model_id,
            "method": self.method.version(),
            "calibrator": calibrator_hash(self.calibrator),
        }

    def decide(self, accounts: list[AccountIn], outputs: list[MethodOutput]) -> list[MappingDecision]:
        """The method's outputs, calibrated and held against the thresholds: one decision per account."""
        versions = self.versions()
        decisions = []
        for account, out in zip(accounts, outputs):
            # Best first; equal scores keep the method's own order (sorted() is stable).
            ranked = sorted(out.scores.items(), key=lambda kv: -kv[1])
            if out.chosen_code is None:
                confidence = 0.0  # "none" or a failure: nothing to trust
            else:
                confidence = float(self.calibrator.predict(features(list(out.scores.values()))[None, :])[0])
            segment = segment_of(account.code, out.chosen_code)
            t_auto = self.thresholds.threshold_for(segment)
            auto = out.chosen_code is not None and t_auto is not None and confidence >= t_auto
            decisions.append(
                MappingDecision(
                    account_id=account.account_id, method=self.method.name,
                    candidates=[Candidate(code=c, score=s, reason=out.reason if c == out.chosen_code else None) for c, s in ranked],
                    confidence=confidence,
                    decision_path="auto" if auto else "review", chosen_code=out.chosen_code,
                    reason=out.reason, status=out.status, model_versions=versions, segment=segment,
                    cost_usd=out.cost_usd, latency_ms=out.latency_ms, cached=out.cached,
                )  # fmt: skip
            )
        return decisions

    @classmethod
    def from_artifacts(
        cls,
        config: Config,
        method: Method | None = None,
        embedder_factory: Callable[[str], Embedder] = SentenceTransformerEmbedder,
    ) -> "Mapper":
        """The Mapper for one trained method (default: selected_method in models.yaml)."""
        name = method or config.models.mapping.selected_method
        if name is None:
            raise MapperError("no method given and selected_method in config/models.yaml is not set yet (it is chosen in the method comparison)")
        if name not in METHOD_REGISTRY:
            raise MapperError(f"{name}: not built yet")
        folder = artifact_dir(config, name)
        if not (folder / "thresholds.json").exists():
            raise MapperError(f"{folder} has no trained {name} method: run `python -m d1.mapper.train --method {name}`")
        embedder = embedder_factory(config.models.embedding_model)
        chart = load_standard_chart(config.data.chart_path, expected=config.data.expected_groups)
        return cls.from_folder(config, name, folder, build_retriever(config, embedder, chart))

    @classmethod
    def from_folder(
        cls, config: Config, name: Method, folder: Path, retriever: Retriever, trained_as: str | None = None, method: MappingMethod | None = None
    ) -> "Mapper":
        """The Mapper for the method saved in folder, on a retriever built already (the cascade's two stages share one).

        trained_as names the train command's --method in the messages (cascade for its second stage);
        method is the method object to load into, when it isn't built from the config as is (the cascade's Claude).
        """
        retrain = f"run `python -m d1.mapper.train --method {trained_as or name}` again"
        thresholds = Thresholds.model_validate_json((folder / "thresholds.json").read_text(encoding="utf-8"))
        calibrator = Calibrator.from_json(json.loads((folder / "calibrator.json").read_text(encoding="utf-8")))
        if retriever.embedder.model_id != thresholds.embedding_model:
            raise MapperError(
                f"{name} was calibrated on shortlists from {thresholds.embedding_model}, but the embedder is {retriever.embedder.model_id}: {retrain}"
            )
        if thresholds.group_text != config.models.mapping.group_text:
            raise MapperError(f"{name} was calibrated with group_text {thresholds.group_text}, but the config says {config.models.mapping.group_text}: {retrain}")
        if thresholds.threshold_by != config.models.mapping.threshold_by:
            raise MapperError(
                f"{name} was calibrated with threshold_by {thresholds.threshold_by}, but the config says {config.models.mapping.threshold_by}: {retrain}"
            )
        mapping_method = method or METHOD_REGISTRY[name](config, retriever)
        mapping_method.load(folder)
        return cls(retriever, mapping_method, calibrator, thresholds, config.models.mapping.candidates_k)
