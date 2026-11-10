from __future__ import annotations

import numpy as np
import pandas as pd

from idsbench.clean import clean
from idsbench.dataset import Dataset, norm_label, norm_name


def test_names_are_normalised():
    assert norm_name(" Flow Bytes/s") == "flow_bytes_s"
    assert norm_name("Fwd Header Length.1") == "fwd_header_length_1"
    assert norm_name("Init_Win_bytes_forward") == "init_win_bytes_forward"
    assert norm_label("Web Attack \x96 Brute Force") == "web attack brute force"
    assert norm_label("Web Attack � XSS") == "web attack xss"
    assert norm_label("BENIGN") == "benign"


def test_clean_counts_each_problem_once_and_in_order():
    X = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.inf, 4.0, 4.0, 5.0, 5.0, 7.0],
            "b": [1.0, 1.0, 1.0, 9.0, 9.0, 3.0, 3.0, 8.0],
            "const": [1.0] * 8,
            "proto": ["tcp"] * 8,
        }
    )
    y = np.array([0, 0, 0, 1, 1, 0, 1, 1])  # rows 5 and 6 have the same features and different labels
    keep, constant, counts = clean(X, y)
    assert counts == {
        "rows_in": 8,
        "non_finite": 1,
        "conflicting_labels": 2,
        "duplicates": 1,
        "rows_out": 4,
        "constant_columns": 1,
    }
    assert list(keep) == [0, 1, 3, 7]
    assert constant == ["const"]


def test_clean_keeps_the_first_of_a_duplicate_pair():
    X = pd.DataFrame({"a": [1.0, 1.0, 2.0]})
    keep, _, counts = clean(X, np.array([1, 1, 0]))
    assert list(keep) == [0, 2] and counts["duplicates"] == 1


def test_clean_on_empty_numeric_part():
    keep, constant, counts = clean(pd.DataFrame({"s": ["x", "y"]}), np.array([0, 1]))
    assert list(keep) == [0, 1] and constant == [] and counts["non_finite"] == 0


def test_dataset_take_and_counts():
    ds = Dataset(
        "x",
        pd.DataFrame({"a": [1.0, 2.0, 3.0]}),
        np.array([False, True, True]),
        np.array(["benign", "dos", "dos"], dtype=object),
        np.array(["train"] * 3, dtype=object),
    )
    sub = ds.take(np.array([1, 2]))
    assert len(ds) == 3 and len(sub) == 2 and sub.family_counts() == {"dos": 2}
    assert list(sub.X["a"]) == [2.0, 3.0]
