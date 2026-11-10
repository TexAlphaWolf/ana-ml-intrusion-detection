"""Numbers that describe a detector, with the choices written down.

Threshold-free: average precision (the area under the precision-recall curve, which depends on the share of attacks)
and ROC AUC. At an operating point: an alert threshold is chosen on the validation rows so that at most 1% of normal
flows raise an alert, and then used unchanged on the test rows. The gap between the false-alarm rate promised on
validation and the one seen on test is part of the result.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

TARGET_FPR = 0.01


def ap(y: np.ndarray, s: np.ndarray) -> float:
    if y.min() == y.max():
        return float("nan")
    return float(average_precision_score(y, s))


def auc(y: np.ndarray, s: np.ndarray) -> float:
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def threshold_for_fpr(y: np.ndarray, s: np.ndarray, fpr: float = TARGET_FPR) -> float:
    """Alert when score > threshold. At most `fpr` of the normal rows in (y, s) are above it."""
    neg = s[~y]
    if len(neg) == 0:
        return float("inf")
    return float(np.quantile(neg, 1 - fpr, method="higher"))


def operating_point(y: np.ndarray, s: np.ndarray, thr: float) -> dict[str, float | int]:
    pred = s > thr
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * tpr / (prec + tpr) if prec + tpr else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "tpr": tpr, "fpr": fpr, "precision": prec, "f1": f1}


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, fpr: float = TARGET_FPR) -> float:
    """The best recall any threshold could give at this false-alarm rate on these very rows (an upper bound)."""
    if y.min() == y.max():
        return float("nan")
    f, t, _ = roc_curve(y, s)
    return float(t[f <= fpr].max())


def pr_grid(y: np.ndarray, s: np.ndarray, points: int = 101) -> dict[str, list[float]]:
    """Interpolated precision at evenly spaced recall values: a small, plottable form of the precision-recall curve."""
    if y.min() == y.max():
        return {"recall": [], "precision": []}
    p, r, _ = precision_recall_curve(y, s)
    p, r = p[::-1], r[::-1]  # recall ascending
    env = np.maximum.accumulate(p[::-1])[::-1]
    grid = np.linspace(0, 1, points)
    idx = np.searchsorted(r, grid, side="left")
    prec = np.where(idx < len(r), env[np.minimum(idx, len(r) - 1)], 0.0)
    return {"recall": [round(float(g), 4) for g in grid], "precision": [round(float(v), 4) for v in prec]}


def per_family(family: np.ndarray, y: np.ndarray, s: np.ndarray, thr: float) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for fam in np.unique(family[y]):
        m = family == fam
        out[str(fam)] = {"rows": int(m.sum()), "recall": float((s[m] > thr).mean())}
    return out


def bootstrap(y: np.ndarray, s: np.ndarray, thr: float, resamples: int, seed: int) -> dict[str, list[float]]:
    """95% intervals for average precision, recall and false-alarm rate (Poisson bootstrap over the test rows).

    Each resample gives every row a random weight from a Poisson(1) distribution, which is the usual large-sample
    equivalent of drawing the rows with replacement. The rows are sorted by score once, so a resample costs one pass.
    """
    if resamples <= 0 or y.min() == y.max():
        return {}
    rng = np.random.default_rng(seed)
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    n = len(ys)
    last = np.concatenate([np.flatnonzero(np.diff(ss)), [n - 1]])  # the last row of each distinct score
    alert = ss > thr
    aps, tprs, fprs = [], [], []
    for _ in range(resamples):
        w = rng.poisson(1.0, n)
        pos, neg = w * ys, w * ~ys
        tp, fp = np.cumsum(pos)[last], np.cumsum(neg)[last]
        total_pos, total_neg = tp[-1], fp[-1]
        if total_pos == 0 or total_neg == 0:
            continue
        recall, precision = tp / total_pos, tp / np.maximum(tp + fp, 1)
        aps.append(float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision)))
        tprs.append(float(pos[alert].sum() / total_pos))
        fprs.append(float(neg[alert].sum() / total_neg))

    def ci(v: list[float]) -> list[float]:
        return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]

    return {"ap": ci(aps), "tpr": ci(tprs), "fpr": ci(fprs)} if aps else {}
