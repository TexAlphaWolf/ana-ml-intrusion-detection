"""Data hygiene: the steps that decide whether a benchmark number means anything, with a count for each.

Both public datasets have problems that inflate scores if they are left alone:
exact duplicate rows (a row can land in the training and the test set), rows with the same features and different
labels (noise nobody can learn), infinite or missing values, and columns that never change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def row_hash(frame: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(frame, index=False).to_numpy()


def clean(
    X: pd.DataFrame, labels: np.ndarray, *, keep_cols: tuple[str, ...] = ()
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """Return (rows to keep, constant columns to drop, counts).

    The order matters and is the same every time: non-finite values, then conflicting labels, then exact duplicates.
    """
    n = len(X)
    counts: dict[str, int] = {"rows_in": n}
    num = X.select_dtypes(include=[np.number])
    bad = np.zeros(n, dtype=bool)
    for col in num.columns:  # one column at a time: the whole table as float64 would need gigabytes for CIC-IDS2017
        bad |= ~np.isfinite(num[col].to_numpy())
    counts["non_finite"] = int(bad.sum())
    keep = ~bad

    h = row_hash(X)
    # the same features with different labels: keep neither (the label is not learnable from these features)
    frame = pd.DataFrame({"h": h, "lab": labels, "keep": keep})
    nlab = frame[frame["keep"]].groupby("h")["lab"].nunique()
    conflicting = frame["h"].isin(nlab[nlab > 1].index) & keep
    counts["conflicting_labels"] = int(conflicting.sum())
    keep &= ~conflicting.to_numpy()

    dup = pd.Series(h).duplicated(keep="first").to_numpy() & keep
    counts["duplicates"] = int(dup.sum())
    keep &= ~dup
    counts["rows_out"] = int(keep.sum())

    sub = num[keep] if keep.any() else num
    constant = [c for c in num.columns if sub[c].nunique(dropna=False) <= 1 and c not in keep_cols]
    counts["constant_columns"] = len(constant)
    return np.flatnonzero(keep), constant, counts
