import json

import numpy as np
import pytest

from d1.mapper.calibrate import (
    Calibrator, choose_threshold, cross_fit_probs, feature_matrix, features, precision_curve, reliability_bins,
)  # fmt: skip


def _scored(n=600, seed=0):
    """Fake method output: a higher top score and a bigger lead mean more often right."""
    rng = np.random.default_rng(seed)
    s1 = rng.uniform(0.3, 0.95, n)
    margin = rng.uniform(0, 0.3, n) * s1
    p_true = 1 / (1 + np.exp(-(12 * (s1 - 0.6) + 10 * margin)))
    y = rng.random(n) < p_true
    return np.column_stack([s1, margin]), y


# --- the two inputs ---------------------------------------------------------------------------


def test_features_top_score_and_lead():
    assert features([0.2, 0.9, 0.5]).tolist() == pytest.approx([0.9, 0.4])  # sorted first: s1 = 0.9, s2 = 0.5


def test_features_with_one_or_no_score():
    assert features([0.7]).tolist() == pytest.approx([0.7, 0.7])  # no runner-up: s2 = 0
    assert features([]).tolist() == [0.0, 0.0]  # the method gave nothing


def test_feature_matrix_one_row_per_account():
    X = feature_matrix([[0.9, 0.5], [0.6], [0.8, 0.8, 0.1]])
    assert X.shape == (3, 2) and X[2].tolist() == pytest.approx([0.8, 0.0])


# --- the calibrator -----------------------------------------------------------------------------


def test_calibrator_gives_probabilities_in_the_right_direction():
    X, y = _scored()
    cal = Calibrator().fit(X, y)
    p = cal.predict(np.array([[0.4, 0.0], [0.9, 0.25]]))
    assert 0 < p[0] < p[1] < 1  # a higher, clearer top score is more likely right


def test_calibrator_json_roundtrip_same_predictions():
    X, y = _scored()
    cal = Calibrator().fit(X, y)
    data = json.loads(json.dumps(cal.to_json()))  # plain numbers only, no pickle
    assert np.allclose(Calibrator.from_json(data).predict(X), cal.predict(X))
    assert data["features"] == ["s1", "s1_minus_s2"] and data["n_fitted"] == len(y)


def test_calibrator_needs_both_outcomes():
    X, _ = _scored(50)
    with pytest.raises(ValueError, match="right and wrong"):
        Calibrator().fit(X, np.ones(50, bool))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        Calibrator().predict(np.zeros((1, 2)))


# --- out-of-fold probabilities ---------------------------------------------------------------


def test_cross_fit_has_no_in_fold_leak():
    # A row's out-of-fold probability comes from a calibrator that never saw that row,
    # so flipping the row's own label cannot change it.
    X, y = _scored()
    p = cross_fit_probs(X, y, folds=5, seed=0)
    flipped = y.copy()
    flipped[7] = ~flipped[7]
    assert cross_fit_probs(X, flipped, folds=5, seed=0)[7] == pytest.approx(p[7], abs=0)
    assert p.shape == y.shape and ((0 < p) & (p < 1)).all()


def test_cross_fit_keeps_companies_together():
    # With groups, a company is never split over the fitting and the predicted part:
    # changing every label of one company leaves that company's probabilities untouched.
    X, y = _scored()
    groups = np.repeat([f"c{i}" for i in range(20)], 30)
    p = cross_fit_probs(X, y, folds=5, seed=0, groups=groups)
    changed = y.copy()
    changed[groups == "c3"] = ~changed[groups == "c3"]
    p2 = cross_fit_probs(X, changed, folds=5, seed=0, groups=groups)
    assert np.array_equal(p2[groups == "c3"], p[groups == "c3"])
    assert not np.array_equal(p2, p)  # other companies did see the change


def test_cross_fit_is_seeded():
    X, y = _scored()
    assert np.array_equal(cross_fit_probs(X, y, seed=3), cross_fit_probs(X, y, seed=3))


# --- choosing the threshold --------------------------------------------------------------------


def test_threshold_on_toy_set():
    p = np.array([0.99] * 60 + [0.90] * 40 + [0.50] * 100)
    correct = np.array([1] * 60 + [1] * 39 + [0] + [1] * 50 + [0] * 50, bool)
    # At 0.90: 99 of 100 right = 0.99 >= 0.98. Adding the 0.50 bucket drops it to 149/200.
    assert choose_threshold(p, correct, 0.98, 50) == 0.90


def test_threshold_unreachable_returns_none():
    p = np.linspace(0.1, 0.99, 200)
    correct = np.arange(200) % 2 == 0  # half wrong at every level
    assert choose_threshold(p, correct, 0.98, 50) is None


def test_min_support_respected():
    p = np.array([0.99] * 30 + [0.80] * 70)
    correct = np.array([1] * 30 + [1] * 60 + [0] * 10, bool)
    assert choose_threshold(p, correct, 0.98, min_support=20) == 0.99  # 30 rows, all right
    assert choose_threshold(p, correct, 0.98, min_support=50) is None  # 0.99 has too few rows, 0.80 is 90%


def test_threshold_takes_the_largest_auto_share_that_qualifies():
    p = np.array([0.99] * 50 + [0.95] * 50 + [0.90] * 50)
    correct = np.ones(150, bool)
    assert choose_threshold(p, correct, 0.98, 50) == 0.90  # every level qualifies: auto-map them all


def test_precision_curve():
    p = np.array([0.9, 0.9, 0.6, 0.3])
    correct = np.array([True, True, False, True])
    curve = precision_curve(p, correct)
    assert curve["threshold"].tolist() == [0.9, 0.6, 0.3]
    assert curve["support"].tolist() == [2, 3, 4]
    assert curve["precision"].tolist() == pytest.approx([1.0, 2 / 3, 0.75])
    assert curve["auto_share"].tolist() == pytest.approx([0.5, 0.75, 1.0])


# --- reliability -----------------------------------------------------------------------------


def test_reliability_bins():
    p = np.array([0.05, 0.15, 0.15, 0.95])
    correct = np.array([False, True, False, True])
    bins = reliability_bins(p, correct, n_bins=10)
    assert [b["n"] for b in bins] == [1, 2, 1]  # empty bins are left out
    assert bins[1]["mean_p"] == pytest.approx(0.15) and bins[1]["accuracy"] == 0.5
    assert bins[2]["low"] == pytest.approx(0.9) and bins[2]["high"] == pytest.approx(1.0)


# --- a threshold that holds for both the honest and the saved probabilities ---------------


def test_stricter_threshold_takes_the_higher():
    from d1.mapper.calibrate import stricter_threshold

    assert stricter_threshold(0.8528, 0.8541) == 0.8541
    assert stricter_threshold(0.95, 0.9477) == 0.95
    assert stricter_threshold(None, 0.9) is None  # unreachable either way: everything to review
    assert stricter_threshold(0.9, None) is None
