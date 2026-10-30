"""Turn text into vectors with a multilingual sentence-embedding model.

Texts with the same meaning get vectors pointing the same way, across
languages: "Leveranciers", "Fournisseurs" and "Suppliers" land close together.
Every vector has length 1, so a dot product is the cosine similarity.
"""

from collections.abc import Callable
from typing import Any, Literal, Protocol

import numpy as np

Kind = Literal["query", "passage"]  # query: a company's account name; passage: a standard group


class Embedder(Protocol):
    model_id: str

    def encode(self, texts: list[str], kind: Kind) -> np.ndarray:
        """One row per text, float32, each row of length 1."""
        ...


def _load_sentence_transformer(model_name: str) -> Any:
    # Imported here: it pulls in torch, which takes seconds and is not needed by tests.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


class SentenceTransformerEmbedder:
    """A Hugging Face sentence-embedding model, downloaded on first use and loaded once.

    The e5 models were trained with "query: " and "passage: " in front of the
    text and work worse without them; the other candidates were trained without.
    """

    def __init__(self, model_name: str, loader: Callable[[str], Any] = _load_sentence_transformer):
        self.model_id = model_name
        self._loader = loader
        self._model = None
        self._prefixes = model_name.startswith("intfloat/multilingual-e5-")

    def encode(self, texts: list[str], kind: Kind) -> np.ndarray:
        if self._model is None:
            self._model = self._loader(self.model_id)
        if self._prefixes:
            texts = [f"{kind}: {t}" for t in texts]
        # Normalized here rather than by the library, so every model is treated the same.
        return normalize_rows(self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False))
