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


def standard_text(a: StandardAccount) -> str:
    """What a standard group is embedded as: its three names, plus its parent.

    The parent tells apart groups with the same name, such as pension
    provisions on the balance sheet (160) and as a cost (635).
    """
    return f"{a.name_en} | {a.name_nl} | {a.name_fr} ({a.parent_name_en})"


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


class Retriever:
    def __init__(self, embedder: Embedder, chart: list[StandardAccount], cache_dir: Path | None = None):
        self.embedder = embedder
        self.codes = [a.code for a in chart]
        self.standard_embeddings = self._standard_embeddings([standard_text(a) for a in chart], cache_dir)

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
