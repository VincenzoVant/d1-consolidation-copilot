from collections import Counter

import numpy as np
import pytest

from d1.config import load_config
from d1.mapper.embed import SentenceTransformerEmbedder
from d1.mapper.retrieve import Retriever, standard_text

NAMES = ["Leveranciers", "Fournisseurs à payer", "Bank KBC zichtrekening", "Huur kantoor Gent"]


# --- the text of a standard group ---------------------------------------------------------


def test_standard_text_has_all_languages_and_parent(chart):
    a = next(a for a in chart if a.code == "440")
    assert standard_text(a) == f"{a.name_en} | {a.name_nl} | {a.name_fr} ({a.parent_name_en})"


# --- top_k --------------------------------------------------------------------------------


def test_top_k_sorted_and_length(fake_embedder, chart):
    result = Retriever(fake_embedder, chart).top_k(NAMES, 10)
    codes = {a.code for a in chart}
    assert len(result) == len(NAMES)
    for candidates in result:
        assert len(candidates) == 10
        scores = [c.score for c in candidates]
        assert scores == sorted(scores, reverse=True)
        assert len({c.code for c in candidates}) == 10 and {c.code for c in candidates} <= codes
        assert all(-1.0 - 1e-6 <= s <= 1.0 + 1e-6 for s in scores)  # cosine similarity


def test_exact_standard_name_ranks_first(fake_embedder, chart):
    # Spread over the chart, so a row/code mix-up anywhere would show. Names that
    # occur twice in the chart (e.g. pension provisions as debt and as cost) are skipped.
    counts = Counter(a.name_en for a in chart)
    picked = [a for a in chart[::19] if counts[a.name_en] == 1]
    assert len(picked) >= 8
    result = Retriever(fake_embedder, chart).top_k([a.name_en for a in picked], 3)
    assert [r[0].code for r in result] == [a.code for a in picked]


def test_k_larger_than_chart_returns_every_group(fake_embedder, chart):
    (candidates,) = Retriever(fake_embedder, chart).top_k(["Leveranciers"], 500)
    assert len(candidates) == len(chart)


def test_k_below_one_raises(fake_embedder, chart):
    with pytest.raises(ValueError, match="k must be at least 1"):
        Retriever(fake_embedder, chart).top_k(NAMES, 0)


def test_empty_input_returns_empty(fake_embedder, chart):
    retriever = Retriever(fake_embedder, chart)
    assert retriever.top_k([], 10) == []
    assert "query" not in fake_embedder.kinds()  # no model call for nothing


def test_queries_are_encoded_as_queries_and_groups_as_passages(fake_embedder, chart):
    Retriever(fake_embedder, chart).top_k(NAMES, 5)
    assert fake_embedder.calls == [("passage", len(chart)), ("query", len(NAMES))]


def test_standard_embeddings_follow_chart_order(fake_embedder, chart):
    retriever = Retriever(fake_embedder, chart)
    assert retriever.standard_embeddings.shape == (len(chart), 256)
    assert retriever.standard_embeddings.dtype == np.float32
    expected = fake_embedder.encode([standard_text(a) for a in chart], "passage")
    assert np.allclose(retriever.standard_embeddings, expected)


# --- cache of the standard-group vectors --------------------------------------------------


def test_cache_written_and_reused(chart, tmp_path, FakeEmbedder):
    first = FakeEmbedder()
    r1 = Retriever(first, chart, cache_dir=tmp_path)
    assert first.kinds() == ["passage"]
    assert len(list(tmp_path.glob("standard_embeddings-fake-trigram-256-*.npy"))) == 1

    second = FakeEmbedder()
    r2 = Retriever(second, chart, cache_dir=tmp_path)
    assert second.kinds() == []  # read from disk, not encoded again
    assert np.array_equal(r1.standard_embeddings, r2.standard_embeddings)


def test_cache_not_reused_when_chart_changes(chart, tmp_path, FakeEmbedder):
    Retriever(FakeEmbedder(), chart, cache_dir=tmp_path)
    other = FakeEmbedder()
    Retriever(other, chart[:100], cache_dir=tmp_path)
    assert other.kinds() == ["passage"]  # a different chart must not get the old vectors
    assert len(list(tmp_path.glob("*.npy"))) == 2


def test_damaged_cache_is_rebuilt(chart, tmp_path, FakeEmbedder):
    Retriever(FakeEmbedder(), chart, cache_dir=tmp_path)
    (path,) = tmp_path.glob("*.npy")
    path.write_bytes(b"not a numpy file")
    again = FakeEmbedder()
    r = Retriever(again, chart, cache_dir=tmp_path)
    assert again.kinds() == ["passage"] and r.standard_embeddings.shape == (len(chart), 256)


def test_without_cache_dir_nothing_is_written(fake_embedder, chart, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Retriever(fake_embedder, chart)
    assert list(tmp_path.iterdir()) == []


# --- the real-model wrapper, with the model swapped out ------------------------------------


class RecordingModel:
    """Pretends to be a SentenceTransformer: remembers the texts, returns unnormalized vectors."""

    def __init__(self):
        self.texts: list[str] = []

    def encode(self, texts, **kwargs):
        self.texts += list(texts)
        return np.arange(1, 1 + 3 * len(texts), dtype=np.float64).reshape(len(texts), 3)


@pytest.mark.parametrize(
    "model_name,query,passage",
    [
        ("intfloat/multilingual-e5-base", "query: Leveranciers", "passage: Suppliers"),
        ("intfloat/multilingual-e5-large", "query: Leveranciers", "passage: Suppliers"),
        ("BAAI/bge-m3", "Leveranciers", "Suppliers"),
        ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "Leveranciers", "Suppliers"),
    ],
)
def test_prefixes_only_for_e5_models(model_name, query, passage):
    model = RecordingModel()
    embedder = SentenceTransformerEmbedder(model_name, loader=lambda name: model)
    embedder.encode(["Leveranciers"], "query")
    embedder.encode(["Suppliers"], "passage")
    assert model.texts == [query, passage]


def test_wrapper_returns_unit_length_float32():
    embedder = SentenceTransformerEmbedder("BAAI/bge-m3", loader=lambda name: RecordingModel())
    out = embedder.encode(["a", "b"], "query")
    assert out.dtype == np.float32 and out.shape == (2, 3)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_wrapper_loads_the_model_only_when_first_needed():
    loaded = []
    embedder = SentenceTransformerEmbedder("BAAI/bge-m3", loader=lambda name: loaded.append(name) or RecordingModel())
    assert embedder.model_id == "BAAI/bge-m3" and loaded == []
    embedder.encode(["a"], "query")
    embedder.encode(["b"], "query")
    assert loaded == ["BAAI/bge-m3"]


# --- real model (downloads it once; run with `uv run pytest -m network tests/mapper -v`) ----


@pytest.mark.network
def test_real_model_smoke():
    embedder = SentenceTransformerEmbedder(load_config().models.embedding_model)
    out = embedder.encode(["Leveranciers", "Fournisseurs", "Bankkosten"], "query")
    assert out.shape[0] == 3 and out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    # Dutch and French for "suppliers" are closer to each other than to "bank charges".
    assert out[0] @ out[1] > out[0] @ out[2]
