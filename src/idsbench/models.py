"""The detectors compared. Every one is a scikit-learn pipeline with the same preprocessing, so that the difference
between the numbers comes from the model and not from the way the features were prepared."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

MODEL_NAMES = ("dummy", "logreg", "tree", "forest", "boosting", "isolation")
DESCRIPTIONS = {
    "dummy": "always the class prior (the floor: what knowing only the share of attacks gives)",
    "logreg": "logistic regression on log-scaled, standardised features",
    "tree": "one decision tree (depth 14, 10 rows per leaf)",
    "forest": "random forest, 150 trees",
    "boosting": "histogram gradient boosting, 200 iterations",
    "isolation": "isolation forest fitted on normal traffic only (no attack label is used to fit it)",
}


def signed_log1p(x: Any) -> Any:
    return np.sign(x) * np.log1p(np.abs(x))


def _preprocessor(columns: list[str], cat_cols: list[str], log_scale: bool) -> ColumnTransformer:
    num_cols = [c for c in columns if c not in cat_cols]
    num: Any = (
        Pipeline([("log", FunctionTransformer(signed_log1p)), ("scale", StandardScaler())])
        if log_scale
        else "passthrough"
    )
    parts: list[tuple[str, Any, list[str]]] = [("num", num, num_cols)]
    if cat_cols:
        enc = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.001, sparse_output=False)
        parts.append(("cat", enc, cat_cols))
    return ColumnTransformer(parts)


@dataclass
class Detector:
    name: str
    pipeline: Pipeline
    unsupervised: bool = False
    notes: str = ""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            if self.unsupervised:
                self.pipeline.fit(X[~y])
            else:
                self.pipeline.fit(X, y)
        last = self.pipeline[-1]
        if isinstance(last, LogisticRegression) and int(np.max(last.n_iter_)) >= last.max_iter:
            self.notes = "did not converge"

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Higher = more likely an attack."""
        if self.unsupervised:
            return np.asarray(-self.pipeline.score_samples(X), dtype=np.float64)
        proba = self.pipeline.predict_proba(X)
        if proba.shape[1] == 1:  # a training set with one class
            return np.zeros(len(X))
        return np.asarray(proba[:, 1], dtype=np.float64)


def make_detector(name: str, columns: list[str], cat_cols: list[str], seed: int, n_jobs: int = 1) -> Detector:
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}; choose from {', '.join(MODEL_NAMES)}")
    linear = name == "logreg"
    pre = _preprocessor(columns, cat_cols, log_scale=linear or name == "isolation")
    model: Any
    if name == "dummy":
        model = DummyClassifier(strategy="prior")
    elif name == "logreg":
        model = LogisticRegression(max_iter=2000, random_state=seed)
    elif name == "tree":
        model = DecisionTreeClassifier(max_depth=14, min_samples_leaf=10, random_state=seed)
    elif name == "forest":
        model = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, n_jobs=n_jobs, random_state=seed)
    elif name == "boosting":
        model = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    else:
        model = IsolationForest(n_estimators=150, random_state=seed, n_jobs=n_jobs)
    return Detector(name, Pipeline([("pre", pre), ("model", model)]), unsupervised=name == "isolation")
