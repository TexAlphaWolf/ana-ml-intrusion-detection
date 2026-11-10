from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from idsbench.dataset import DatasetError
from idsbench.loaders import detect, load, load_cic, load_unsw
from idsbench.synth import CIC_DAY_FILES, make_unsw, write_cic, write_unsw


def test_detects_each_dataset(folders: dict[str, Path], tmp_path: Path):
    assert detect(folders["unsw"]) == "unsw"
    assert detect(folders["cic"]) == "cic"
    (tmp_path / "other.csv").write_text("a,b\n1,2\n")
    assert detect(tmp_path) is None
    with pytest.raises(DatasetError, match="cannot tell"):
        load(tmp_path)


def test_unsw_shape_and_cleaning(unsw):
    assert unsw.name == "unsw" and set(unsw.group) == {"train", "test"}
    assert unsw.cat_cols == ["proto", "service", "state"]
    assert "id" not in unsw.X.columns and "label" not in unsw.X.columns and "attack_cat" not in unsw.X.columns
    assert unsw.hygiene["duplicates"] > 0
    assert unsw.hygiene["rows_out"] == len(unsw)
    assert set(unsw.family) >= {"benign", "generic", "exploits"} and not unsw.y[unsw.family == "benign"].any()
    assert unsw.y[unsw.family != "benign"].all()
    assert unsw.X.select_dtypes("number").dtypes.eq("float32").all()


def test_cic_shape_and_defects_of_the_real_files_are_handled(cic):
    assert cic.name == "cic" and set(cic.group) == {"monday", "tuesday", "wednesday", "thursday", "friday"}
    cols = set(cic.X.columns)
    assert "destination_port" in cols and "flow_bytes_s" in cols
    assert not any(c.endswith("_1") for c in cols)  # the repeated 'Fwd Header Length' column is gone
    assert cic.hygiene["non_finite"] > 0  # Infinity and empty cells
    assert cic.hygiene["duplicates"] > 0
    assert "web attack brute force" in set(cic.family)  # the broken dash does not survive
    assert not any("\x96" in f or "�" in f for f in set(cic.family))
    assert set(cic.family[cic.group == "monday"]) == {"benign"}


def test_cic_identifier_columns_are_dropped(tmp_path: Path):
    write_cic(tmp_path, rows_per_day=300, seed=2)
    for files in CIC_DAY_FILES.values():  # the "GeneratedLabelledFlows" variant has these columns in every file
        path = tmp_path / files[0]
        d = pd.read_csv(path, encoding="latin-1")
        for pos, (name, val) in enumerate((("Flow ID", "x"), (" Source IP", "y"), (" Timestamp", "z"))):
            d.insert(pos, name, val)
        path.write_bytes(d.to_csv(index=False).replace(",inf", ",Infinity").encode("latin-1"))
    ds = load_cic(tmp_path)
    assert not {"flow_id", "source_ip", "timestamp"} & set(ds.X.columns)
    assert "destination_port" in ds.X.columns


def test_unsw_test_rows_that_repeat_training_rows_are_counted_and_removed(tmp_path: Path):
    train = make_unsw(300, 1)
    test = make_unsw(200, 2)
    test.iloc[:20, 1:] = train.iloc[:20, 1:].to_numpy()  # twenty test rows copied from the training file
    train.to_csv(tmp_path / "UNSW_NB15_training-set.csv", index=False)
    test.to_csv(tmp_path / "UNSW_NB15_testing-set.csv", index=False)
    ds = load_unsw(tmp_path)
    assert ds.hygiene["test_rows_identical_to_a_training_row"] >= 20
    assert (ds.group == "test").sum() <= 200 - 20


def test_unsw_label_and_category_disagreement_is_reported(tmp_path: Path):
    train = make_unsw(200, 1)
    train.loc[train.index[:5], "label"] = 1 - train.loc[train.index[:5], "label"]
    train.to_csv(tmp_path / "UNSW_NB15_training-set.csv", index=False)
    make_unsw(100, 2).to_csv(tmp_path / "UNSW_NB15_testing-set.csv", index=False)
    assert load_unsw(tmp_path).hygiene["label_disagreements"] == 5


def test_sample_reads_a_fraction(folders: dict[str, Path], unsw):
    part = load_unsw(folders["unsw"], sample=0.25, seed=1)
    assert 0.15 * len(unsw) < len(part) < 0.4 * len(unsw)


def test_errors_say_what_is_wrong(tmp_path: Path):
    with pytest.raises(DatasetError, match="not a folder"):
        load(tmp_path / "missing")
    with pytest.raises(DatasetError, match=r"no \.csv"):
        load(tmp_path)
    write_unsw(tmp_path, rows=100)
    pd.DataFrame({"x": [1]}).to_csv(tmp_path / "UNSW_NB15_training-set.csv", index=False)
    with pytest.raises(DatasetError, match="missing columns"):
        load_unsw(tmp_path)
    (tmp_path / "UNSW_NB15_training-set.csv").unlink()
    with pytest.raises(DatasetError, match="expected one training-set"):
        load_unsw(tmp_path)


def test_cic_files_with_different_columns_are_refused(tmp_path: Path):
    write_cic(tmp_path, rows_per_day=100, seed=1)
    path = tmp_path / CIC_DAY_FILES["friday"][0]
    d = pd.read_csv(path, encoding="latin-1")
    path.write_bytes(d.drop(columns=[" Idle Min"]).to_csv(index=False).encode("latin-1"))
    with pytest.raises(DatasetError, match="same columns"):
        load_cic(tmp_path)


def test_cic_rows_without_a_label_are_dropped_and_counted(tmp_path: Path):
    write_cic(tmp_path, rows_per_day=200, seed=4)
    path = tmp_path / CIC_DAY_FILES["monday"][0]
    d = pd.read_csv(path, encoding="latin-1")
    empty = pd.DataFrame([[None] * len(d.columns)] * 3, columns=d.columns)
    path.write_bytes(pd.concat([d, empty]).to_csv(index=False).replace(",inf", ",Infinity").encode("latin-1"))
    ds = load_cic(tmp_path)
    assert ds.hygiene["rows_without_label"] == 3 and "nan" not in set(ds.family)
