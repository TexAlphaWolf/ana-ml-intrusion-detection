from __future__ import annotations

from pathlib import Path

import pytest

from idsbench.dataset import Dataset
from idsbench.experiment import Config, run_all
from idsbench.loaders import load
from idsbench.synth import write_cic, write_unsw

FAST = Config(models=("dummy", "logreg", "tree", "boosting", "isolation"), seed=5, resamples=3, importance_rows=400)


@pytest.fixture(scope="session")
def folders(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("synthetic")
    write_unsw(root / "unsw", rows=2400, seed=3)
    write_cic(root / "cic", rows_per_day=900, seed=3)
    return {"unsw": root / "unsw", "cic": root / "cic"}


@pytest.fixture(scope="session")
def unsw(folders: dict[str, Path]) -> Dataset:
    return load(folders["unsw"], "unsw")


@pytest.fixture(scope="session")
def cic(folders: dict[str, Path]) -> Dataset:
    return load(folders["cic"], "cic")


@pytest.fixture(scope="session")
def report(unsw: Dataset, cic: Dataset) -> dict:
    return run_all({"unsw": unsw, "cic": cic}, FAST, quick=True, synthetic=True)
