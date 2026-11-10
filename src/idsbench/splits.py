"""How the data is divided into training, validation and test rows, and what each division is meant to show.

random   rows of the same traffic on both sides (all families, the same days). Optimistic: it is what most
         published scores use, and it is shown here to measure how optimistic it is.
official the dataset's own partition (UNSW-NB15: separate training and testing files).
by-day   CIC-IDS2017: Monday to Wednesday to train, Thursday and Friday to test. Some attacks only appear in the
         test days: the model has never seen them, as it happens when a new attack arrives.
transfer train on one dataset, test on the other, with the few features both have.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from idsbench.dataset import Dataset

TRAIN_DAYS = ("monday", "tuesday", "wednesday")
TEST_DAYS = ("thursday", "friday")


@dataclass
class Split:
    protocol: str
    description: str
    train: Dataset
    val: Dataset
    test: Dataset

    @property
    def cat_cols(self) -> list[str]:
        return self.train.cat_cols


def stratified_parts(family: np.ndarray, fractions: tuple[float, ...], seed: int) -> list[np.ndarray]:
    """Index arrays with the given fractions, family by family, so that rare families are present on every side.

    A family with fewer than three rows goes entirely to the first part (it cannot be shared fairly).
    """
    rng = np.random.default_rng(seed)
    parts: list[list[np.ndarray]] = [[] for _ in fractions]
    for fam in np.unique(family):
        idx = rng.permutation(np.flatnonzero(family == fam))
        n = len(idx)
        if n < 3:
            parts[0].append(idx)
            continue
        cut = np.cumsum([max(1, round(f * n)) for f in fractions[:-1]])
        cut = np.minimum(cut, n - 1)
        for i, piece in enumerate(np.split(idx, cut)):
            parts[i].append(piece)
    return [np.sort(np.concatenate(p)) if p else np.array([], dtype=int) for p in parts]


def _three(ds: Dataset, pool: np.ndarray, test: np.ndarray, seed: int) -> tuple[Dataset, Dataset, Dataset]:
    tr, va = stratified_parts(ds.family[pool], (0.8, 0.2), seed)
    return ds.take(pool[tr]), ds.take(pool[va]), ds.take(test)


def random_split(ds: Dataset, seed: int) -> Split:
    a, b, c = stratified_parts(ds.family, (0.6, 0.2, 0.2), seed)
    return Split(
        "random",
        "random 60/20/20 split of all rows, family by family (optimistic: train and test come from the same traffic)",
        ds.take(a),
        ds.take(b),
        ds.take(c),
    )


def official_split(ds: Dataset, seed: int) -> Split:
    pool = np.flatnonzero(ds.group == "train")
    test = np.flatnonzero(ds.group == "test")
    tr, va, te = _three(ds, pool, test, seed)
    return Split(
        "official",
        "the dataset's own partition: training file for fitting (80%) and for choosing the alert threshold (20%), "
        "testing file for the result",
        tr,
        va,
        te,
    )


def by_day_split(ds: Dataset, seed: int) -> Split:
    pool = np.flatnonzero(np.isin(ds.group, TRAIN_DAYS))
    test = np.flatnonzero(np.isin(ds.group, TEST_DAYS))
    tr, va, te = _three(ds, pool, test, seed)
    return Split(
        "by-day",
        "Monday to Wednesday for fitting (80%) and for choosing the alert threshold (20%); Thursday and Friday for the "
        "result: several attack families appear only there",
        tr,
        va,
        te,
    )


def realistic_split(ds: Dataset, seed: int) -> Split:
    return {"unsw": official_split, "cic": by_day_split}[ds.name](ds, seed)


COMMON = [
    "duration",
    "fwd_packets",
    "bwd_packets",
    "fwd_bytes",
    "bwd_bytes",
    "packets_per_s",
    "bytes_per_s",
    "mean_packet",
]


def common_view(ds: Dataset) -> Dataset:
    """Only what both datasets measure: duration, packets and bytes in each direction, and three ratios."""
    X = ds.X
    if ds.name == "unsw":
        parts = {
            "duration": X["dur"],
            "fwd_packets": X["spkts"],
            "bwd_packets": X["dpkts"],
            "fwd_bytes": X["sbytes"],
            "bwd_bytes": X["dbytes"],
        }
    else:
        parts = {
            "duration": X["flow_duration"] / 1e6,  # CIC counts microseconds, UNSW seconds
            "fwd_packets": X["total_fwd_packets"],
            "bwd_packets": X["total_backward_packets"],
            "fwd_bytes": X["total_length_of_fwd_packets"],
            "bwd_bytes": X["total_length_of_bwd_packets"],
        }
    out = pd.DataFrame(parts).astype(np.float64)
    pk = out["fwd_packets"] + out["bwd_packets"]
    by = out["fwd_bytes"] + out["bwd_bytes"]
    dur = out["duration"].clip(lower=1e-6)
    out["packets_per_s"] = pk / dur
    out["bytes_per_s"] = by / dur
    out["mean_packet"] = by / pk.clip(lower=1)
    return Dataset(ds.name, out[COMMON].astype(np.float32), ds.y, ds.family, ds.group, [], ds.hygiene)


def transfer_split(source: Dataset, target: Dataset, seed: int) -> Split:
    src, dst = common_view(source), common_view(target)
    tr, va = stratified_parts(src.family, (0.8, 0.2), seed)
    return Split(
        f"transfer {source.name}->{target.name}",
        f"fit and choose the threshold on {source.name}, test on all of {target.name}, using only the "
        f"{len(COMMON)} features both have (the attack families are different: only 'attack or not' can be compared)",
        src.take(tr),
        src.take(va),
        dst,
    )
