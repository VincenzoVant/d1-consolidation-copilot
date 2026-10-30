import hashlib

import numpy as np
import pytest

from d1.config import load_config
from d1.datagen.standard_chart import load_standard_chart

DIM = 256


class FakeEmbedder:
    """Stands in for a real embedding model, with no download and in milliseconds.

    A text becomes a bag of its 3-letter pieces, each hashed to one of 256
    slots, so texts that share letters end up close. That is enough to check
    the retrieval code; it says nothing about how good a real model is.
    calls records (kind, number of texts) for every encode().
    """

    model_id = "fake/trigram-256"

    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        self.calls.append((kind, len(texts)))
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            padded = f"  {text.casefold()} "
            for i in range(len(padded) - 2):
                slot = int.from_bytes(hashlib.md5(padded[i : i + 3].encode()).digest()[:4], "little") % DIM
                out[row, slot] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.calls]


@pytest.fixture(scope="session")
def chart():
    return load_standard_chart(load_config().data.chart_path, expected=190)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture(name="FakeEmbedder")
def fake_embedder_class():
    """The class itself, for tests that need several fresh embedders."""
    return FakeEmbedder
