"""The common shape of a dataset after loading: features, a yes/no label, an attack family and a group."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BENIGN = "benign"


def norm_name(name: str) -> str:
    """'  Flow Bytes/s' -> 'flow_bytes_s': the column names of the two datasets are written in different styles."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def norm_label(text: str) -> str:
    """'Web Attack \x96 Brute Force' -> 'web attack brute force' (the dash is a broken character in the original)."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


class DatasetError(Exception):
    """The files are not what the loader expects; the message says what was found."""


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray  # True = attack
    family: np.ndarray  # 'benign' or the attack family, lower case
    group: np.ndarray  # where a row comes from: 'train'/'test' (UNSW) or the weekday (CIC)
    cat_cols: list[str] = field(default_factory=list)
    hygiene: dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)

    def take(self, idx: np.ndarray) -> Dataset:
        return Dataset(
            self.name,
            self.X.iloc[idx].reset_index(drop=True),
            self.y[idx],
            self.family[idx],
            self.group[idx],
            self.cat_cols,
            self.hygiene,
        )

    def family_counts(self) -> dict[str, int]:
        names, counts = np.unique(self.family, return_counts=True)
        return {str(n): int(c) for n, c in sorted(zip(names, counts, strict=True), key=lambda t: -t[1])}
