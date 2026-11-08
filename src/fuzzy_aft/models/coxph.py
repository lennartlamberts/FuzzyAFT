from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


class CoxPHModel:
    """
    Classical Cox proportional hazards baseline wrapper.
    """

    def __init__(self) -> None:
        self.model = CoxPHFitter()
        self.feature_names: list[str] | None = None

    def fit(self, x_train: np.ndarray, t_train: np.ndarray, event_train: np.ndarray) -> None:
        self.feature_names = [f"x{i}" for i in range(x_train.shape[1])]
        df_train = pd.DataFrame(x_train, columns=self.feature_names)
        df_train["time"] = t_train
        df_train["event"] = event_train.astype(int)
        self.model.fit(df_train, duration_col="time", event_col="event")

    def predict_risk(self, x: np.ndarray) -> np.ndarray:
        if self.feature_names is None:
            raise RuntimeError("Model must be fit before prediction.")
        df_x = pd.DataFrame(x, columns=self.feature_names)
        return self.model.predict_partial_hazard(df_x).to_numpy().reshape(-1)

    def predict_survival(self, x: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.feature_names is None:
            raise RuntimeError("Model must be fit before prediction.")
        df_x = pd.DataFrame(x, columns=self.feature_names)
        surv_df = self.model.predict_survival_function(df_x, times=times)
        return surv_df.to_numpy().T

    @property
    def log_likelihood_(self) -> float:
        return float(self.model.log_likelihood_)
