"""Pair features: the numbers a tree model sees for one (account, candidate group) pair.

XGBoost and random forests can't read text. So every account is turned into
one row per candidate on its shortlist, and each row holds numbers that say how
well the account fits that candidate:

  from the shortlist   cosine, rank, gap_to_top, parent_cosine
  from the letters     fuzzy_wratio, token_jaccard
  from the code        code_prefix_match, class_digit_match
  from the account     name_len, n_digits, is_bilingual
  from the meaning     emb_pca_0 ... emb_pca_15

The model (Task 3.2) learns which patterns mean "this is the right group".

The emb_pca features compress the element-wise product of the account vector
and the group vector (1,024 numbers, which sum to the cosine) into 16. The
cosine says *how much* the two agree; the product says *on what*. The
compression (a PCA) is fitted on train rows only, on each account paired with
its right group, and then applied unchanged to dev, test and new accounts.
"""

import json
import re
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process
from sklearn.decomposition import PCA

from d1.datagen.schema import CompanyAccount, StandardAccount
from d1.datagen.splits import normalize_name
from d1.mapper.methods.base import AccountIn, check_lengths
from d1.mapper.retrieve import Candidate, Retriever

BASE_FEATURES = [
    "cosine", "rank", "gap_to_top", "parent_cosine",          # from the shortlist
    "fuzzy_wratio", "token_jaccard",                          # from the letters
    "code_prefix_match", "class_digit_match",                 # from the account code
    "name_len", "n_digits", "is_bilingual",                   # from the account alone
]  # fmt: skip
N_PCA = 16
PAIR_FEATURES = BASE_FEATURES + [f"emb_pca_{i}" for i in range(N_PCA)]
LANGUAGES = ("en", "nl", "fr")
BATCH_SIZE = 512  # accounts per batch: 5,120 pairs x 1,024 numbers stay well under 100 MB
VERSION = "pair-features-v1"

# Two parts around " / ", each with a run of 3+ letters: "Terug te vorderen btw / TVA à récupérer".
_BILINGUAL = re.compile(r"[^\W\d_]{3,}.*\s/\s.*[^\W\d_]{3,}")


class FeaturizerError(Exception):
    """Used before fitting, fitted on the wrong rows, or loaded with another retriever."""


def account_features(name: str, code: str | None) -> dict[str, float]:
    """The features that depend on the account alone (the same on all its candidate rows).

    name_len      characters in the name: a truncated name stops at a fixed length
    n_digits      digits in the account code: 6 is a PCMN-style code (whose prefix is
                  the group), 4 an old company numbering (whose prefix means nothing)
    is_bilingual  the name holds two language versions, "nl / fr"
    """
    return {
        "name_len": len(name),
        "n_digits": sum(ch.isdigit() for ch in code) if code else 0,
        "is_bilingual": 1.0 if _BILINGUAL.search(name) else 0.0,
    }


def _tokens(text: str) -> set[str]:
    return set(normalize_name(text).split())


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def lexical_features(name: str, group: StandardAccount) -> dict[str, float]:
    """How close the letters and words are, in the group's best-matching language.

    fuzzy_wratio   rapidfuzz WRatio (0 to 1), forgiving for typos, word order and cut-off names
    token_jaccard  shared words / all words, after removing case, accents and punctuation
    """
    query = normalize_name(name)
    return {
        "fuzzy_wratio": max(fuzz.WRatio(query, normalize_name(group.name(lang))) for lang in LANGUAGES) / 100.0,
        "token_jaccard": max(_jaccard(_tokens(name), _tokens(group.name(lang))) for lang in LANGUAGES),
    }


class PairFeaturizer:
    """Turns (account, shortlist) into a k x n_features matrix, one row per candidate.

    It uses the same retriever as the shortlist, so the group vectors (names,
    descriptions or their average) are the ones the cosines came from.
    """

    def __init__(self, retriever: Retriever, chart: list[StandardAccount], n_pca: int = N_PCA, batch_size: int = BATCH_SIZE):
        if sorted(a.code for a in chart) != sorted(retriever.codes):
            raise FeaturizerError("the chart and the retriever hold different groups")
        self.retriever = retriever
        self.n_pca = n_pca
        self.batch_size = batch_size
        self.mean: np.ndarray | None = None  # set by fit() or load()
        self.components: np.ndarray | None = None
        self.explained_variance: list[float] = []  # share of the products' variation each PCA feature keeps
        self._index = {code: i for i, code in enumerate(retriever.codes)}
        groups = {a.code: a for a in chart}
        ordered = [groups[code] for code in retriever.codes]  # everything below follows the retriever's order
        self._class_digit = [a.class_code for a in ordered]
        # Parents are shared ("Trade Debts" for 440 and 441), so each parent text is embedded once.
        parent_texts = [self.parent_text(a) for a in ordered]
        unique = list(dict.fromkeys(parent_texts))
        vectors = retriever.passage_embeddings(unique)
        self._parent_vectors = vectors[[unique.index(t) for t in parent_texts]]
        self._choices = [normalize_name(a.name(lang)) for a in ordered for lang in LANGUAGES]
        self._group_tokens = [[_tokens(a.name(lang)) for lang in LANGUAGES] for a in ordered]

    @staticmethod
    def parent_text(group: StandardAccount) -> str:
        """The parent group's name ("Trade Debts" for 440 Suppliers), or the class name at the top."""
        return group.parent_name_en or group.class_name_en

    @property
    def feature_names(self) -> list[str]:
        return BASE_FEATURES + [f"emb_pca_{i}" for i in range(self.n_pca)]

    # --- fitting ---------------------------------------------------------------------------------

    def fit(self, rows: list[CompanyAccount]) -> "PairFeaturizer":
        """Fit the PCA on train rows: each account's vector times its right group's vector."""
        other = sorted({r.split for r in rows} - {"train"})
        if other:
            raise FeaturizerError(f"the PCA is fitted on train rows only, but got {', '.join(other)} rows")
        if len(rows) < self.n_pca:
            raise FeaturizerError(f"need at least {self.n_pca} train rows for {self.n_pca} PCA features, got {len(rows)}")
        unknown = sorted({r.target_code for r in rows} - self._index.keys())
        if unknown:
            raise FeaturizerError(f"target groups not in the chart: {', '.join(unknown)}")
        if self.n_pca == 0:  # no meaning features: only the 11 base features (to measure what the PCA adds)
            dim = self.retriever.standard_embeddings.shape[1]
            self.mean, self.components, self.explained_variance = np.zeros(dim, np.float32), np.zeros((0, dim), np.float32), []
            return self
        accounts = self._encode([r.name for r in rows])
        targets = self.retriever.standard_embeddings[[self._index[r.target_code] for r in rows]]
        pca = PCA(n_components=self.n_pca, svd_solver="full").fit(accounts * targets)  # "full": exact, no randomness
        self.mean = pca.mean_.astype(np.float32)
        self.components = pca.components_.astype(np.float32)
        self.explained_variance = [round(float(v), 6) for v in pca.explained_variance_ratio_]
        return self

    # --- transforming ----------------------------------------------------------------------------

    def transform(self, accounts: list[AccountIn], candidates: list[list[Candidate]]) -> list[np.ndarray]:
        """One float32 matrix per account: a row per candidate, in shortlist order."""
        check_lengths(accounts, candidates)
        if self.components is None:
            raise FeaturizerError("call fit() on train rows, or load() a fitted featurizer, before transform()")
        unknown = sorted({c.code for shortlist in candidates for c in shortlist} - self._index.keys())
        if unknown:
            raise FeaturizerError(f"candidate groups not in the chart: {', '.join(unknown)}")
        out: list[np.ndarray] = []
        for start in range(0, len(accounts), self.batch_size):
            end = start + self.batch_size
            out += self._transform_batch(accounts[start:end], candidates[start:end])
        return out

    def _encode(self, names: list[str]) -> np.ndarray:
        """Account vectors, each distinct name encoded once."""
        unique = list(dict.fromkeys(names))
        vectors = self.retriever.embedder.encode(unique, "query")
        position = {name: i for i, name in enumerate(unique)}
        return vectors[[position[n] for n in names]]

    def _transform_batch(self, accounts: list[AccountIn], candidates: list[list[Candidate]]) -> list[np.ndarray]:
        names = list(dict.fromkeys(a.name for a in accounts))
        name_pos = {n: i for i, n in enumerate(names)}
        vectors = self.retriever.embedder.encode(names, "query") if names else np.zeros((0, 1), np.float32)
        # Letters: every distinct name against all 570 names (190 groups x 3 languages) in one call.
        wratio = process.cdist([normalize_name(n) for n in names], self._choices, scorer=fuzz.WRatio, workers=-1, dtype=np.float32)
        wratio = wratio.reshape(len(names), -1, len(LANGUAGES)).max(axis=2) / 100.0

        # One entry per (account, candidate) pair, in order.
        pair_name = np.array([name_pos[a.name] for a, s in zip(accounts, candidates) for _ in s], dtype=np.int64)
        pair_group = np.array([self._index[c.code] for s in candidates for c in s], dtype=np.int64)
        n_pairs = len(pair_group)
        X = np.zeros((n_pairs, len(self.feature_names)), dtype=np.float32)
        col = {f: i for i, f in enumerate(self.feature_names)}

        row = 0
        for account, shortlist in zip(accounts, candidates):
            if not shortlist:
                continue
            top = shortlist[0].score
            own = account_features(account.name, account.code)
            tokens = _tokens(account.name)
            for rank, c in enumerate(shortlist, start=1):
                g = self._index[c.code]
                X[row, col["cosine"]] = c.score
                X[row, col["rank"]] = rank
                X[row, col["gap_to_top"]] = top - c.score
                X[row, col["token_jaccard"]] = max(_jaccard(tokens, t) for t in self._group_tokens[g])
                X[row, col["code_prefix_match"]] = 1.0 if account.code and account.code.startswith(c.code) else 0.0
                X[row, col["class_digit_match"]] = 1.0 if account.code and account.code[0] == self._class_digit[g] else 0.0
                for f in ("name_len", "n_digits", "is_bilingual"):
                    X[row, col[f]] = own[f]
                row += 1

        if n_pairs:
            pair_vectors = vectors[pair_name]
            X[:, col["parent_cosine"]] = np.einsum("pd,pd->p", pair_vectors, self._parent_vectors[pair_group])
            X[:, col["fuzzy_wratio"]] = wratio[pair_name, pair_group]
            if self.n_pca:
                product = pair_vectors * self.retriever.standard_embeddings[pair_group]
                X[:, len(BASE_FEATURES) :] = (product - self.mean) @ self.components.T

        sizes = np.cumsum([len(s) for s in candidates])[:-1]
        return np.split(X, sizes)

    # --- saving ----------------------------------------------------------------------------------

    def save(self, folder: Path) -> None:
        if self.components is None:
            raise FeaturizerError("nothing to save: the featurizer is not fitted")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / "featurizer.npz", mean=self.mean, components=self.components)
        info = {"version": VERSION, "n_pca": self.n_pca, "feature_names": self.feature_names, "explained_variance": self.explained_variance,
                "embedding_model": self.retriever.embedder.model_id, "group_text": self.retriever.group_text}  # fmt: skip
        (folder / "featurizer.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    def load(self, folder: Path) -> "PairFeaturizer":
        """A fitted featurizer saved by save(); refused if the retriever would give other vectors."""
        folder = Path(folder)
        info = json.loads((folder / "featurizer.json").read_text(encoding="utf-8"))
        model = self.retriever.embedder.model_id
        if info["embedding_model"] != model:
            raise FeaturizerError(f"{folder} was fitted with {info['embedding_model']}, but the retriever uses {model}")
        if info["group_text"] != self.retriever.group_text:
            raise FeaturizerError(f"{folder} was fitted with group_text {info['group_text']}, but the retriever uses {self.retriever.group_text}")
        arrays = np.load(folder / "featurizer.npz", allow_pickle=False)
        self.n_pca = int(info["n_pca"])
        self.mean, self.components = arrays["mean"], arrays["components"]
        self.explained_variance = list(info["explained_variance"])
        if info["feature_names"] != self.feature_names:
            raise FeaturizerError(f"{folder} lists other features than this code makes: retrain the method")
        return self


def training_pairs(
    rows: list[CompanyAccount], candidates: list[list[Candidate]], featurizer: PairFeaturizer
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All pairs of the rows stacked: X (features), y (1 = right group), group (the row each pair belongs to).

    An account whose right group is not on its shortlist gives only negatives:
    the model must also learn what "none of these fits" looks like.
    """
    accounts = [AccountIn(account_id=r.account_id, name=r.name, code=r.code, company_id=r.company_id) for r in rows]
    matrices = featurizer.transform(accounts, candidates)
    n_features = len(featurizer.feature_names)
    X = np.vstack(matrices) if matrices else np.zeros((0, n_features), dtype=np.float32)
    y = np.array([1 if c.code == r.target_code else 0 for r, s in zip(rows, candidates) for c in s], dtype=np.int8)
    group = np.array([i for i, s in enumerate(candidates) for _ in s], dtype=np.int32)
    return X, y, group
