"""The shortlist: for each account name, the k standard groups closest in meaning.

Every mapping method chooses among these same k candidates, so the methods
differ only in how well they choose. If the right group is not on the
shortlist, no method can find it: recall@k caps them all.

The 190 standard-group vectors are computed once per model and cached on disk.
"""

import hashlib
import os
import re
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from d1.datagen.schema import StandardAccount
from d1.mapper.embed import Embedder


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    score: float  # cosine similarity between the account name and the group


def standard_text(a: StandardAccount, description: str | None = None) -> str:
    """What a standard group is embedded as: its three names, plus its parent.

    The parent tells apart groups with the same name, such as pension
    provisions on the balance sheet (160) and as a cost (635). With a
    description (descriptions.py), its everyday wording is added after a dash.
    """
    text = f"{a.name_en} | {a.name_nl} | {a.name_fr} ({a.parent_name_en})"
    return f"{text} — {description}" if description else text


def _slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")


def _read_cache(path: Path, n_rows: int) -> np.ndarray | None:
    """The cached vectors, or None if the file is missing, damaged or the wrong shape."""
    try:
        vectors = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError):
        return None
    if vectors.ndim != 2 or vectors.shape[0] != n_rows or vectors.dtype != np.float32:
        return None
    return vectors


GROUP_TEXTS = ("names", "descriptions", "average")


class Retriever:
    """Builds shortlists by cosine similarity between account names and the 190 groups.

    group_text says what a group is compared with:
      names         its official names (standard_text without description)
      descriptions  its names plus its description (descriptions.py)
      average       the mean of both similarities: the description helps loosely named
                    accounts, the plain names keep exact-name matches clearly ahead
    """

    def __init__(
        self,
        embedder: Embedder,
        chart: list[StandardAccount],
        cache_dir: Path | None = None,
        descriptions: dict[str, str] | None = None,
        group_text: str = "names",
    ):
        if group_text not in GROUP_TEXTS:
            raise ValueError(f"unknown group_text {group_text!r}: choose from {', '.join(GROUP_TEXTS)}")
        if group_text == "names" and descriptions is not None:
            raise ValueError("descriptions were given but group_text is 'names': set group_text to use them")
        if group_text != "names":
            if descriptions is None:
                raise ValueError(f"group_text {group_text!r} needs descriptions")
            missing = [a.code for a in chart if a.code not in descriptions]
            if missing:
                raise ValueError(f"no description for groups {', '.join(missing[:5])}")
        self.embedder = embedder
        self.codes = [a.code for a in chart]
        self.group_text = group_text
        self.descriptions = descriptions
        plain_texts = [standard_text(a) for a in chart]
        if group_text == "names":
            self.texts = plain_texts
            self.standard_embeddings = self._standard_embeddings(plain_texts, cache_dir)
            return
        self.texts = [standard_text(a, descriptions[a.code]) for a in chart]
        described = self._standard_embeddings(self.texts, cache_dir)
        if group_text == "descriptions":
            self.standard_embeddings = described
        else:
            # A dot product is linear, so scoring against the mean vector gives the mean of both
            # cosines. The mean vector is not unit length on purpose: its scores stay cosines' means.
            plain = self._standard_embeddings(plain_texts, cache_dir)
            self.standard_embeddings = ((plain + described) / 2).astype(np.float32)

    def _standard_embeddings(self, texts: list[str], cache_dir: Path | None) -> np.ndarray:
        if cache_dir is None:
            return self.embedder.encode(texts, "passage")
        # The file name holds a hash of the texts, so a changed chart (or a changed
        # standard_text) gets new vectors instead of silently reusing stale ones.
        digest = hashlib.sha256("\n".join([self.embedder.model_id, *texts]).encode()).hexdigest()[:12]
        path = Path(cache_dir) / f"standard_embeddings-{_slug(self.embedder.model_id)}-{digest}.npy"
        cached = _read_cache(path, len(texts))
        if cached is not None:
            return cached
        vectors = self.embedder.encode(texts, "passage")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, vectors)
        os.replace(tmp, path)  # a crash mid-write never leaves a half file under the real name
        return vectors

    def top_k(self, names: list[str], k: int) -> list[list[Candidate]]:
        """For each name, the k closest groups, highest score first."""
        if k < 1:
            raise ValueError("k must be at least 1")
        if not names:
            return []
        # The same name often appears in many companies ("Bank", "Huur"): encode it once.
        unique = list(dict.fromkeys(names))
        position = {name: i for i, name in enumerate(unique)}
        scores = self.embedder.encode(unique, "query") @ self.standard_embeddings.T
        k = min(k, len(self.codes))
        # Stable sort: equal scores keep chart order, so results never vary run to run.
        order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        shortlists = [
            [Candidate(code=self.codes[j], score=float(row[j])) for j in idx] for idx, row in zip(order, scores)
        ]
        return [shortlists[position[name]] for name in names]
