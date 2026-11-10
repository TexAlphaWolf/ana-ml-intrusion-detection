from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idsbench.models import DESCRIPTIONS, MODEL_NAMES, make_detector, signed_log1p
from idsbench.splits import official_split


def test_every_model_has_a_description():
    assert set(DESCRIPTIONS) == set(MODEL_NAMES)


def test_signed_log1p_keeps_sign_and_order():
    x = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    out = signed_log1p(x)
    assert (np.diff(out) > 0).all() and out[2] == 0 and out[0] == -out[4]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_each_detector_fits_and_scores_every_row(unsw, name):
    s = official_split(unsw, 1)
    cols = list(s.train.X.columns)
    det = make_detector(name, cols, s.train.cat_cols, seed=1)
    det.fit(s.train.X.iloc[:800], s.train.y[:800])
    scores = det.score(s.test.X.iloc[:300])
    assert scores.shape == (300,) and np.isfinite(scores).all()
    if name != "dummy":
        assert scores[s.test.y[:300]].mean() > scores[~s.test.y[:300]].mean()  # attacks score higher than normal


def test_unknown_model_is_refused():
    with pytest.raises(ValueError, match="unknown model"):
        make_detector("svm", ["a"], [], 1)


def test_unseen_category_value_does_not_break_prediction(unsw):
    s = official_split(unsw, 1)
    det = make_detector("logreg", list(s.train.X.columns), s.train.cat_cols, seed=1)
    det.fit(s.train.X.iloc[:600], s.train.y[:600])
    odd = s.test.X.iloc[:5].copy()
    odd["proto"] = "never-seen-protocol"
    assert np.isfinite(det.score(odd)).all()


def test_isolation_forest_never_looks_at_attack_rows(unsw):
    s = official_split(unsw, 1)
    X, y = s.train.X.iloc[:800].copy(), s.train.y[:800]
    cols = list(X.columns)
    a = make_detector("isolation", cols, s.train.cat_cols, seed=1)
    a.fit(X, y)
    X2 = X.copy()
    num = [c for c in cols if c not in s.train.cat_cols]
    X2.loc[y, num] = 123456.0  # destroy the attack rows
    b = make_detector("isolation", cols, s.train.cat_cols, seed=1)
    b.fit(X2, y)
    probe = s.test.X.iloc[:200]
    assert np.allclose(a.score(probe), b.score(probe))


def test_a_training_set_with_one_class_scores_zero(unsw):
    s = official_split(unsw, 1)
    benign = s.train.X[~s.train.y].iloc[:200]
    det = make_detector("dummy", list(benign.columns), s.train.cat_cols, seed=1)
    det.fit(benign, np.zeros(len(benign), bool))
    assert (det.score(s.test.X.iloc[:10]) == 0).all()


def test_logistic_regression_reports_when_it_did_not_converge():
    X = pd.DataFrame({"a": np.arange(40.0), "b": np.arange(40.0)[::-1]})
    y = np.arange(40) % 2 == 0
    det = make_detector("logreg", ["a", "b"], [], seed=1)
    det.pipeline[-1].set_params(max_iter=1)
    det.fit(X, y)
    assert det.notes == "did not converge"
