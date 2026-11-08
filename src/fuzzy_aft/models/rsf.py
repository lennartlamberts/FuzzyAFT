from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _to_structured_y(t: np.ndarray, event: np.ndarray) -> np.ndarray:
    y = np.zeros(t.shape[0], dtype=[("event", "?"), ("time", "<f8")])
    y["event"] = np.asarray(event).astype(bool)
    y["time"] = np.asarray(t).astype(float)
    return y


def _predict_survival_matrix(model: Any, x: np.ndarray, times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float).reshape(-1)
    try:
        surv = model.predict_survival_function(x, return_array=True)
        surv = np.asarray(surv, dtype=float)
        if surv.ndim == 2 and surv.shape[1] == times.shape[0]:
            return np.clip(surv, 1e-12, 1.0)
    except TypeError:
        pass
    fns = model.predict_survival_function(x)
    out = np.zeros((x.shape[0], times.shape[0]), dtype=float)
    for i, fn in enumerate(fns):
        out[i, :] = fn(times)
    return np.clip(out, 1e-12, 1.0)


@dataclass(frozen=True)
class RSFFitInfo:
    n_estimators: int
    max_depth: int | None
    max_features: float | str | None
    min_samples_split: int
    min_samples_leaf: int
    n_jobs: int | None


class RandomSurvivalForestModel:
    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_depth: int | None = None,
        max_features: float | str | None = "sqrt",
        min_samples_split: int = 10,
        min_samples_leaf: int = 15,
        n_jobs: int | None = -1,
        random_state: int | None = None,
    ) -> None:
        from sksurv.ensemble import RandomSurvivalForest

        self.model = RandomSurvivalForest(
            n_estimators=int(n_estimators),
            max_depth=max_depth if max_depth is None else int(max_depth),
            max_features=max_features,
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            n_jobs=n_jobs if n_jobs is None else int(n_jobs),
            random_state=random_state if random_state is None else int(random_state),
        )
        self.fit_info_: RSFFitInfo | None = None

    def fit(self, x_train: np.ndarray, t_train: np.ndarray, event_train: np.ndarray) -> None:
        x_train = np.asarray(x_train, dtype=float)
        t_train = np.asarray(t_train, dtype=float).reshape(-1)
        event_train = np.asarray(event_train, dtype=int).reshape(-1)
        y_train = _to_structured_y(t_train, event_train)
        self.model.fit(x_train, y_train)
        self.fit_info_ = RSFFitInfo(
            n_estimators=int(getattr(self.model, "n_estimators", 0)),
            max_depth=getattr(self.model, "max_depth", None),
            max_features=getattr(self.model, "max_features", None),
            min_samples_split=int(getattr(self.model, "min_samples_split", 0)),
            min_samples_leaf=int(getattr(self.model, "min_samples_leaf", 0)),
            n_jobs=getattr(self.model, "n_jobs", None),
        )

    def predict_risk(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        risk = self.model.predict(x)
        return np.asarray(risk, dtype=float).reshape(-1)

    def predict_survival(self, x: np.ndarray, times: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return _predict_survival_matrix(self.model, x, times)

