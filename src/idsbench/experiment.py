"""Run the detectors on a split and collect everything the report needs, as plain JSON-able data."""

from __future__ import annotations

import os
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance

from idsbench import __version__
from idsbench.dataset import Dataset
from idsbench.metrics import (
    TARGET_FPR,
    ap,
    auc,
    bootstrap,
    operating_point,
    per_family,
    pr_grid,
    threshold_for_fpr,
    tpr_at_fpr,
)
from idsbench.models import DESCRIPTIONS, MODEL_NAMES, make_detector
from idsbench.splits import Split, random_split, realistic_split, transfer_split

Log = Callable[[str], None]


@dataclass
class Config:
    models: tuple[str, ...] = MODEL_NAMES
    seed: int = 1000
    max_train: int | None = 500_000
    resamples: int = 50
    importance_rows: int = 20_000
    n_jobs: int = 1
    sample: float | None = None


def _share(d: Dataset) -> float:
    return round(float(d.y.mean()), 4) if len(d) else 0.0


def _subsample(ds: Dataset, limit: int | None, seed: int) -> Dataset:
    if limit is None or len(ds) <= limit:
        return ds
    idx = np.sort(np.random.default_rng(seed).choice(len(ds), limit, replace=False))
    return ds.take(idx)


def _drop(ds: Dataset, cols: tuple[str, ...]) -> Dataset:
    present = [c for c in cols if c in ds.X.columns]
    if not present:
        return ds
    return Dataset(ds.name, ds.X.drop(columns=present), ds.y, ds.family, ds.group, ds.cat_cols, ds.hygiene)


def _false_positive_view(test: Dataset, scores: np.ndarray, thr: float) -> dict[str, Any]:
    for col in ("service", "destination_port"):
        if col in test.X.columns:
            benign = ~test.y
            fp = benign & (scores > thr)
            vc_b = test.X.loc[benign, col].astype(str).value_counts()
            vc_f = test.X.loc[fp, col].astype(str).value_counts().head(5)
            return {"column": col, "top": {str(k): [int(v), int(vc_b.get(k, 0))] for k, v in vc_f.items()}}
    return {}


def run_split(
    split: Split, cfg: Config, *, exclude: tuple[str, ...] = (), label: str = "", log: Log = lambda s: None
) -> dict[str, Any]:
    train, val, test = (_drop(d, exclude) for d in (split.train, split.val, split.test))
    full_train = len(train)
    train = _subsample(train, cfg.max_train, cfg.seed)
    columns = list(train.X.columns)
    seen = set(train.family.tolist())

    run: dict[str, Any] = {
        "id": label or split.protocol,
        "dataset": train.name if train.name == test.name else f"{train.name}->{test.name}",
        "protocol": split.protocol,
        "description": split.description,
        "excluded_features": [c for c in exclude if c not in train.X.columns],
        "features": len(columns),
        "rows": {"train": len(train), "train_available": full_train, "val": len(val), "test": len(test)},
        "attack_share": {"train": _share(train), "val": _share(val), "test": _share(test)},
        "families": {
            "test": {
                str(f): {"rows": int((test.family == f).sum()), "seen_in_training": bool(f in seen)}
                for f in sorted(set(test.family[test.y].tolist()))
            }
        },
        "target_fpr": TARGET_FPR,
        "models": {},
    }
    fitted: dict[str, Any] = {}
    val_ap: dict[str, float] = {}
    for name in cfg.models:
        det = make_detector(name, columns, [c for c in train.cat_cols if c in columns], cfg.seed, cfg.n_jobs)
        t0 = time.perf_counter()
        det.fit(train.X, train.y)
        fit_s = time.perf_counter() - t0
        s_val = det.score(val.X)
        t0 = time.perf_counter()
        s_test = det.score(test.X)
        pred_s = time.perf_counter() - t0
        thr = threshold_for_fpr(val.y, s_val)
        op_val = operating_point(val.y, s_val, thr)
        op_test = operating_point(test.y, s_test, thr)
        val_ap[name] = ap(val.y, s_val)
        entry = {
            "description": DESCRIPTIONS[name],
            "fit_seconds": round(fit_s, 2),
            "predict_rows_per_second": round(len(test) / max(pred_s, 1e-9)),
            "val_ap": round(val_ap[name], 4),
            "test_ap": round(ap(test.y, s_test), 4),
            "test_roc_auc": round(auc(test.y, s_test), 4),
            "threshold": thr,
            "val_operating_point": op_val,
            "test_operating_point": op_test,
            "test_best_possible_tpr_at_target_fpr": round(tpr_at_fpr(test.y, s_test), 4),
            "intervals": bootstrap(test.y, s_test, thr, cfg.resamples, cfg.seed),
            "per_family_recall": per_family(test.family, test.y, s_test, thr),
            "curve": pr_grid(test.y, s_test),
            "notes": det.notes,
        }
        run["models"][name] = entry
        fitted[name] = (det, s_test, thr)
        log(
            f"  {run['id']}: {name:10s} AP {entry['test_ap']:.3f}  recall {op_test['tpr']:.3f}  false alarms {op_test['fpr']:.4f}"
        )

    supervised = [m for m in cfg.models if m not in ("dummy", "isolation")]
    if supervised:
        best = max(supervised, key=lambda m: val_ap[m])  # chosen on validation rows, never on the test rows
        det, s_test, thr = fitted[best]
        run["analysis_model"] = best
        run["false_positives"] = _false_positive_view(test, s_test, thr)
        if cfg.importance_rows > 0 and len(test) > 0:
            rng = np.random.default_rng(cfg.seed)
            take = np.sort(rng.choice(len(test), min(cfg.importance_rows, len(test)), replace=False))
            Xs, ys = test.X.iloc[take], test.y[take]
            if ys.min() != ys.max():
                imp = permutation_importance(
                    det.pipeline, Xs, ys, scoring="average_precision", n_repeats=3, random_state=cfg.seed, n_jobs=1
                )
                order = np.argsort(-imp.importances_mean)[:10]
                run["importance"] = {
                    "model": best,
                    "metric": "drop in average precision when one column is shuffled (test rows, 3 repeats)",
                    "top": [
                        [columns[i], round(float(imp.importances_mean[i]), 4), round(float(imp.importances_std[i]), 4)]
                        for i in order
                    ],
                }
    return run


def environment(cfg: Config, quick: bool, synthetic: bool, seconds: float) -> dict[str, Any]:
    return {
        "idsbench": __version__,
        "system": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "cpus": os.cpu_count(),
        "seed": cfg.seed,
        "quick": quick,
        "synthetic": synthetic,
        "sample": cfg.sample,
        "max_train": cfg.max_train,
        "resamples": cfg.resamples,
        "seconds": round(seconds, 1),
        "command": " ".join(sys.argv[1:]),
    }


def run_all(
    datasets: dict[str, Dataset],
    cfg: Config,
    *,
    quick: bool = False,
    synthetic: bool = False,
    log: Log = lambda s: None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    runs = []
    for name, ds in datasets.items():
        log(f"{name}: {len(ds)} rows after cleaning")
        runs.append(run_split(random_split(ds, cfg.seed), cfg, label=f"{name}-random", log=log))
        real = realistic_split(ds, cfg.seed)
        runs.append(run_split(real, cfg, label=f"{name}-{real.protocol}", log=log))
        if name == "unsw":
            runs.append(
                run_split(
                    real,
                    cfg,
                    exclude=("sttl", "dttl", "ct_state_ttl"),
                    label="unsw-official-no-ttl",
                    log=log,
                )
            )
        if name == "cic":
            runs.append(
                run_split(
                    realistic_split(ds, cfg.seed),
                    cfg,
                    exclude=("destination_port",),
                    label="cic-by-day-no-port",
                    log=log,
                )
            )
    if len(datasets) == 2:
        a, b = datasets["unsw"], datasets["cic"]
        runs.append(run_split(transfer_split(a, b, cfg.seed), cfg, label="transfer-unsw-to-cic", log=log))
        runs.append(run_split(transfer_split(b, a, cfg.seed), cfg, label="transfer-cic-to-unsw", log=log))
    return {
        "environment": environment(cfg, quick, synthetic, time.perf_counter() - t0),
        "data": {n: {"rows": len(d), "families": d.family_counts(), "hygiene": d.hygiene} for n, d in datasets.items()},
        "runs": runs,
    }
