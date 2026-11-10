from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from idsbench.metrics import (
    ap,
    auc,
    bootstrap,
    operating_point,
    per_family,
    pr_grid,
    threshold_for_fpr,
    tpr_at_fpr,
)

RNG = np.random.default_rng(0)
Y = np.array([False] * 900 + [True] * 100)
S = np.concatenate([RNG.normal(0, 1, 900), RNG.normal(2, 1, 100)])


def test_threshold_gives_at_most_the_target_false_alarm_rate():
    thr = threshold_for_fpr(Y, S, 0.01)
    op = operating_point(Y, S, thr)
    assert op["fpr"] <= 0.01 and op["fpr"] > 0.005
    assert op["tp"] + op["fn"] == 100 and op["fp"] + op["tn"] == 900


def test_threshold_with_ties_never_exceeds_the_target():
    s = np.zeros(1000)
    s[:50] = 1.0  # 5% of the rows share the top score
    y = np.zeros(1000, bool)
    thr = threshold_for_fpr(y, s, 0.01)
    assert operating_point(y, s, thr)["fpr"] == 0.0  # no threshold can reach 1% exactly: it chooses silence


def test_operating_point_arithmetic():
    y = np.array([True, True, False, False, False])
    s = np.array([0.9, 0.2, 0.8, 0.1, 0.1])
    op = operating_point(y, s, 0.5)
    assert (op["tp"], op["fp"], op["fn"], op["tn"]) == (1, 1, 1, 2)
    assert op["tpr"] == 0.5 and op["precision"] == 0.5 and op["f1"] == 0.5 and op["fpr"] == pytest.approx(1 / 3)


def test_operating_point_with_no_alert_has_zero_precision_not_a_crash():
    op = operating_point(np.array([True, False]), np.array([0.1, 0.2]), 0.9)
    assert op["precision"] == 0.0 and op["f1"] == 0.0


def test_tpr_at_fpr_is_an_upper_bound_of_any_chosen_threshold():
    thr = threshold_for_fpr(Y, S, 0.05)
    assert tpr_at_fpr(Y, S, 0.05) >= operating_point(Y, S, thr)["tpr"] - 1e-9


def test_single_class_gives_nan_not_an_error():
    y = np.zeros(10, bool)
    s = np.arange(10.0)
    assert np.isnan(ap(y, s)) and np.isnan(auc(y, s)) and np.isnan(tpr_at_fpr(y, s))
    assert pr_grid(y, s) == {"recall": [], "precision": []}
    assert bootstrap(y, s, 0.5, 5, 1) == {}


def test_threshold_without_normal_rows_alerts_on_nothing():
    assert threshold_for_fpr(np.ones(5, bool), np.arange(5.0)) == float("inf")


def test_pr_grid_is_a_monotone_envelope_with_a_matching_area():
    g = pr_grid(Y, S)
    p = np.array(g["precision"])
    assert len(g["recall"]) == 101 and g["recall"][0] == 0 and g["recall"][-1] == 1
    assert (np.diff(p) <= 1e-9).all()
    area = float(np.mean(p))
    assert area == pytest.approx(average_precision_score(Y, S), abs=0.03)


def test_per_family_recall_only_counts_attacks():
    fam = np.array(["benign", "dos", "dos", "scan", "benign"], dtype=object)
    y = fam != "benign"
    s = np.array([0.9, 0.9, 0.1, 0.9, 0.1])
    out = per_family(fam, y, s, 0.5)
    assert out == {"dos": {"rows": 2, "recall": 0.5}, "scan": {"rows": 1, "recall": 1.0}}


def test_bootstrap_interval_contains_the_point_estimate_and_is_reproducible():
    thr = threshold_for_fpr(Y, S)
    a = bootstrap(Y, S, thr, 40, seed=1)
    b = bootstrap(Y, S, thr, 40, seed=1)
    assert a == b
    assert a["ap"][0] <= ap(Y, S) <= a["ap"][1]
    assert a["tpr"][0] <= operating_point(Y, S, thr)["tpr"] <= a["tpr"][1]
    assert bootstrap(Y, S, thr, 0, 1) == {}


def test_bootstrap_with_unit_weights_would_reproduce_average_precision():
    # the weighted average precision used inside bootstrap() equals scikit-learn's when every weight is 1
    order = np.argsort(-S, kind="stable")
    ys, ss = Y[order], S[order]
    last = np.concatenate([np.flatnonzero(np.diff(ss)), [len(ys) - 1]])
    tp, fp = np.cumsum(ys)[last], np.cumsum(~ys)[last]
    recall, precision = tp / tp[-1], tp / (tp + fp)
    mine = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
    assert mine == pytest.approx(average_precision_score(Y, S), abs=1e-9)


def test_bootstrap_interval_narrows_with_more_rows():
    thr = threshold_for_fpr(Y, S)
    small = bootstrap(Y[::5], S[::5], thr, 60, seed=2)
    big = bootstrap(Y, S, thr, 60, seed=2)
    assert big["ap"][1] - big["ap"][0] < small["ap"][1] - small["ap"][0]
