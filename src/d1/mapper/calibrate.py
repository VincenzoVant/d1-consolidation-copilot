"""From a method's raw scores to "how likely is this right?", and the auto-map threshold.

Every method scores its candidates on its own scale (a cosine, a tree-model
probability, an LLM's 0-100 confidence). None of those is the chance of being
right. The calibrator learns that chance on dev from two numbers per account:

  s1           the top candidate's score
  s1 - s2      its lead over the runner-up (a clear winner is more trustworthy)

Then the threshold: auto-map only accounts whose calibrated probability is high
enough that, on dev, at least 98% of the auto-mapped accounts are right.
Every method is calibrated this same way, so their results compare fairly.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold

FEATURES = ["s1", "s1_minus_s2"]


def features(scores: list[float]) -> np.ndarray:
    """[s1, s1 - s2] from one account's candidate scores; s2 = 0 without a runner-up."""
    ordered = sorted(scores, reverse=True)
    s1 = ordered[0] if ordered else 0.0
    s2 = ordered[1] if len(ordered) > 1 else 0.0
    return np.array([s1, s1 - s2], dtype=float)


def feature_matrix(score_lists: list[list[float]]) -> np.ndarray:
    return np.array([features(s) for s in score_lists], dtype=float).reshape(-1, len(FEATURES))


class Calibrator:
    """A logistic regression on the two features: P(the top candidate is right)."""

    def __init__(self):
        self.coef: np.ndarray | None = None
        self.intercept: float = 0.0
        self.n_fitted = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Calibrator":
        y = np.asarray(y, dtype=bool)
        if y.all() or not y.any():
            raise ValueError("calibration needs both right and wrong examples")
        # Barely regularised (C=100): two inputs and thousands of rows cannot overfit,
        # while the default penalty would flatten the probabilities, because the inputs are small numbers.
        model = LogisticRegression(C=100.0).fit(X, y)
        self.coef = model.coef_[0].astype(float)
        self.intercept = float(model.intercept_[0])
        self.n_fitted = len(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("call fit() or from_json() before predict()")
        return 1.0 / (1.0 + np.exp(-(np.asarray(X, dtype=float) @ self.coef + self.intercept)))

    def to_json(self) -> dict:
        """Plain numbers only: safe to commit and to load, unlike a pickled model."""
        return {"type": "logistic", "features": FEATURES, "coef": self.coef.tolist(), "intercept": self.intercept, "n_fitted": self.n_fitted}

    @classmethod
    def from_json(cls, data: dict) -> "Calibrator":
        if data.get("features") != FEATURES:
            raise ValueError(f"calibrator was fitted on {data.get('features')}, expected {FEATURES}")
        cal = cls()
        cal.coef = np.array(data["coef"], dtype=float)
        cal.intercept = float(data["intercept"])
        cal.n_fitted = int(data["n_fitted"])
        return cal


def cross_fit_probs(X: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0, groups: np.ndarray | None = None) -> np.ndarray:
    """Out-of-fold probabilities: each row is predicted by a calibrator fitted without it.

    The threshold is chosen on these, so it is not tuned on the very rows it is
    judged on. With groups (company ids), a whole company is always in the same
    fold: accounts of one company look alike, so splitting a company over the
    fitting and the predicted part would leak.
    """
    y = np.asarray(y, dtype=bool)
    if groups is None:
        splits = KFold(n_splits=folds, shuffle=True, random_state=seed).split(X)
    else:
        splits = GroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(X, groups=groups)
    p = np.empty(len(y), dtype=float)
    for fit_idx, predict_idx in splits:
        p[predict_idx] = Calibrator().fit(X[fit_idx], y[fit_idx]).predict(X[predict_idx])
    return p


def precision_curve(p: np.ndarray, correct: np.ndarray) -> dict[str, np.ndarray]:
    """For every observed probability t (highest first): what auto-mapping all accounts with p >= t gives.

    support = how many accounts are auto-mapped, precision = the share of them that is right,
    auto_share = support as a share of all accounts.
    """
    p = np.asarray(p, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    order = np.argsort(-p, kind="stable")
    hits = np.cumsum(correct[order])
    thresholds = np.unique(p)[::-1]
    support = len(p) - np.searchsorted(np.sort(p), thresholds, side="left")
    precision = hits[support - 1] / support
    return {"threshold": thresholds, "support": support, "precision": precision, "auto_share": support / len(p)}


def choose_threshold(p: np.ndarray, correct: np.ndarray, target: float = 0.98, min_support: int = 50) -> float | None:
    """The lowest t (the most accounts auto-mapped) whose precision is at least target.

    min_support stops a threshold being trusted on a handful of accounts: 5 out
    of 5 right says little. None if no threshold reaches the target: then
    everything goes to review.
    """
    curve = precision_curve(p, correct)
    ok = (curve["precision"] >= target - 1e-12) & (curve["support"] >= min_support)
    if not ok.any():
        return None
    return float(curve["threshold"][ok].min())


def stricter_threshold(*thresholds: float | None) -> float | None:
    """The highest of the thresholds; None if any is None (unreachable one way means review).

    train.py chooses one threshold on the out-of-fold probabilities (honest) and one
    on the saved calibrator's probabilities (what the Mapper applies), and keeps the
    stricter, so the target holds both ways. They differ when many accounts share
    one score: fuzzy gives 29% of dev accounts a top score of exactly 1.0, and the
    two calibrators then place that whole block on different sides of the line.
    """
    if any(t is None for t in thresholds):
        return None
    return max(thresholds)


def reliability_bins(p: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Accounts grouped by predicted probability: if calibrated, mean_p matches accuracy in every bin."""
    p = np.asarray(p, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    index = np.minimum((p * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = index == b
        if mask.any():
            out.append({"low": b / n_bins, "high": (b + 1) / n_bins, "mean_p": float(p[mask].mean()),
                        "accuracy": float(correct[mask].mean()), "n": int(mask.sum())})  # fmt: skip
    return out
