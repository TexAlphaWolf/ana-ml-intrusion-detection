"""Read the two public datasets from the files as published, and bring them to the shape in dataset.py.

UNSW-NB15: the two partition files `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`.
CIC-IDS2017: the eight `MachineLearningCSV` files, one per working period (Monday to Friday).
Neither dataset is stored in this repository; see data/README.md for where to get them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from idsbench.clean import clean, row_hash
from idsbench.dataset import BENIGN, Dataset, DatasetError, norm_label, norm_name

UNSW_CATEGORICAL = ["proto", "service", "state"]
UNSW_REQUIRED = ["dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "attack_cat", "label"]
CIC_REQUIRED = ["destination_port", "flow_duration", "total_fwd_packets", "total_backward_packets", "label"]
# columns of the "GeneratedLabelledFlows" variant that identify a flow instead of describing it: a model must not see them
CIC_IDENTIFIERS = {"flow_id", "source_ip", "destination_ip", "source_port", "timestamp", "simulation_time"}
DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday")


def _csv_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise DatasetError(f"{folder} is not a folder")
    found = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".csv")
    if not found:
        raise DatasetError(f"no .csv file in {folder}")
    return found


def detect(folder: Path) -> str | None:
    """'unsw', 'cic' or None, from the file names."""
    names = [p.name.lower() for p in _csv_files(folder)]
    if any("training-set" in n for n in names) and any("testing-set" in n for n in names):
        return "unsw"
    if any(n.startswith(DAYS) for n in names):
        return "cic"
    return None


def _read(path: Path, **kw: object) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kw)  # type: ignore[call-overload,no-any-return]
    except (pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError) as e:
        raise DatasetError(f"cannot read {path.name}: {e}") from e


def _require(frame: pd.DataFrame, needed: list[str], where: str) -> None:
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise DatasetError(f"{where}: missing columns {missing}; found {list(frame.columns)[:12]} ...")


def _numeric(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    # a value that is not a number becomes NaN and the row is dropped by clean(), with a count
    out: pd.DataFrame = frame[cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    return out


def load_unsw(folder: Path, *, sample: float | None = None, seed: int = 0) -> Dataset:
    files = _csv_files(folder)
    train = [p for p in files if "training-set" in p.name.lower()]
    test = [p for p in files if "testing-set" in p.name.lower()]
    if len(train) != 1 or len(test) != 1:
        raise DatasetError(
            f"expected one training-set and one testing-set csv in {folder}, found {[p.name for p in files]}"
        )
    parts = []
    for group, path in (("train", train[0]), ("test", test[0])):
        df = _read(path)
        df.columns = [norm_name(c) for c in df.columns]
        _require(df, UNSW_REQUIRED, path.name)
        df = df.drop(columns=[c for c in ("id",) if c in df.columns])
        if sample:
            df = df.sample(frac=sample, random_state=seed)
        df["group"] = group
        parts.append(df)
    raw = pd.concat(parts, ignore_index=True)

    fam = raw["attack_cat"].astype(str).map(norm_label).replace({"normal": BENIGN, "": BENIGN, "nan": BENIGN})
    lab = pd.to_numeric(raw["label"], errors="coerce")
    disagree = int(((lab == 0) != (fam == BENIGN)).sum())
    cat_cols = list(UNSW_CATEGORICAL)
    numeric_cols = [c for c in raw.columns if c not in (*cat_cols, "attack_cat", "label", "group")]
    X = _numeric(raw, numeric_cols)
    for c in cat_cols:
        X[c] = raw[c].astype(str).str.strip().str.lower().to_numpy()
    y = (fam != BENIGN).to_numpy()
    return _finish("unsw", X, y, fam.to_numpy(dtype=object), raw["group"].to_numpy(dtype=object), cat_cols, disagree)


def load_cic(folder: Path, *, sample: float | None = None, seed: int = 0) -> Dataset:
    xs: list[pd.DataFrame] = []
    fams: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    unlabelled = 0
    for path in _csv_files(folder):
        df = _read(path, encoding="latin-1", low_memory=False)
        df.columns = [norm_name(c) for c in df.columns]
        _require(df, CIC_REQUIRED, path.name)
        # the original files have the column 'Fwd Header Length' twice; pandas calls the second one '..._1'
        for c in [c for c in df.columns if c.endswith("_1") and c[:-2] in df.columns]:
            if df[c].equals(df[c[:-2]]):
                df = df.drop(columns=c)
        df = df.drop(columns=[c for c in df.columns if c in CIC_IDENTIFIERS])
        if sample:
            df = df.sample(frac=sample, random_state=seed)
        # some files end with empty rows: no label, nothing to learn
        has_label = df["label"].notna().to_numpy()
        unlabelled += int((~has_label).sum())
        df = df[has_label]
        name = path.name.lower()
        fams.append(df["label"].astype(str).map(norm_label).replace({"": BENIGN}).to_numpy(dtype=object))
        groups.append(np.full(len(df), next((d for d in DAYS if name.startswith(d)), "unknown"), dtype=object))
        xs.append(_numeric(df, [c for c in df.columns if c not in ("label", "group")]))  # float32 at once: memory
    cols = list(xs[0].columns)
    for x in xs[1:]:
        if list(x.columns) != cols:
            raise DatasetError("the csv files do not have the same columns")
    X = pd.concat(xs, ignore_index=True)
    family = np.concatenate(fams)
    return _finish("cic", X, family != BENIGN, family, np.concatenate(groups), [], 0, unlabelled)


def _finish(
    name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    family: np.ndarray,
    group: np.ndarray,
    cat_cols: list[str],
    label_disagreements: int,
    unlabelled: int = 0,
) -> Dataset:
    cross = 0
    if name == "unsw":
        tr, te = group == "train", group == "test"
        cross = int(np.isin(row_hash(X[te]), row_hash(X[tr])).sum())
    keep, constant, counts = clean(X, y.astype(np.int8))
    counts["label_disagreements"] = label_disagreements
    counts["rows_without_label"] = unlabelled
    if name == "unsw":
        counts["test_rows_identical_to_a_training_row"] = cross
    X = X.drop(columns=constant).iloc[keep].reset_index(drop=True)
    cat_cols = [c for c in cat_cols if c in X.columns]
    hygiene: dict[str, object] = {**counts, "dropped_columns": constant, "features": X.shape[1]}
    return Dataset(name, X, y[keep], family[keep], group[keep], cat_cols, hygiene)


def load(folder: Path, kind: str | None = None, **kw: object) -> Dataset:
    kind = kind or detect(folder)
    if kind == "unsw":
        return load_unsw(folder, **kw)  # type: ignore[arg-type]
    if kind == "cic":
        return load_cic(folder, **kw)  # type: ignore[arg-type]
    raise DatasetError(
        f"{folder}: cannot tell which dataset this is (expected UNSW_NB15_*-set.csv or Monday-*.csv ...)"
    )
