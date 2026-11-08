from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torchtuples as tt
from pycox.models import CoxPH

from .mlp_backbone import MLPBackbone


class _DeepSurvBackboneNet(nn.Module):
    """MLP backbone + scalar risk head for pycox.CoxPH."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...] = (128, 128, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = MLPBackbone(input_dim=input_dim, hidden_sizes=hidden_sizes, dropout=dropout)
        self.head = nn.Linear(self.backbone.output_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"x must have shape (batch, features), got {tuple(x.shape)}.")
        z = self.backbone(x)
        return self.head(z)


@dataclass(frozen=True)
class DeepSurvFitInfo:
    best_epoch: int
    best_val_loss: float


class DeepSurvModel:
    """DeepSurv baseline implemented with pycox.CoxPH."""

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (128, 128, 128),
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 200,
        patience: int = 20,
        min_delta: float = 0.0,
        device: str = "cpu",
        random_state: int = 0,
    ) -> None:
        self.hidden_sizes = tuple(int(v) for v in hidden_sizes)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.device = str(device)
        self.random_state = int(random_state)

        self.model_: CoxPH | None = None
        self.fit_info_: DeepSurvFitInfo | None = None

    def _check_ready(self) -> CoxPH:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted.")
        return self.model_

    def fit(
        self,
        x_train: np.ndarray,
        t_train: np.ndarray,
        event_train: np.ndarray,
        x_val: np.ndarray | None = None,
        t_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> None:
        torch.manual_seed(int(self.random_state))
        np.random.seed(int(self.random_state))

        xtr = np.asarray(x_train, dtype=np.float32)
        ttr = np.asarray(t_train, dtype=np.float32).reshape(-1)
        etr = np.asarray(event_train, dtype=np.float32).reshape(-1)
        if xtr.ndim != 2:
            raise ValueError("x_train must be 2D.")
        if not (xtr.shape[0] == ttr.shape[0] == etr.shape[0]):
            raise ValueError("Train arrays have inconsistent lengths.")

        net = _DeepSurvBackboneNet(
            input_dim=int(xtr.shape[1]),
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
        )
        model = CoxPH(
            net=net,
            optimizer=tt.optim.Adam(lr=self.learning_rate),
            device=self.device,
        )
        callbacks = [tt.callbacks.EarlyStopping(patience=self.patience, min_delta=self.min_delta)]

        val_data = None
        use_val = x_val is not None and t_val is not None and event_val is not None
        if use_val:
            xva = np.asarray(x_val, dtype=np.float32)
            tva = np.asarray(t_val, dtype=np.float32).reshape(-1)
            eva = np.asarray(event_val, dtype=np.float32).reshape(-1)
            val_data = (xva, (tva, eva))

        log = model.fit(
            xtr,
            (ttr, etr),
            batch_size=self.batch_size,
            epochs=self.max_epochs,
            callbacks=callbacks,
            verbose=False,
            val_data=val_data,
        )
        model.compute_baseline_hazards()
        self.model_ = model

        best_epoch = self.max_epochs
        best_val = float("nan")
        try:
            hist = log.to_pandas()
            if "val_loss" in hist.columns and hist["val_loss"].notna().any():
                best_row = int(hist["val_loss"].astype(float).idxmin())
                best_epoch = best_row + 1
                best_val = float(hist.loc[best_row, "val_loss"])
            elif "train_loss" in hist.columns and hist["train_loss"].notna().any():
                best_row = int(hist["train_loss"].astype(float).idxmin())
                best_epoch = best_row + 1
                best_val = float(hist.loc[best_row, "train_loss"])
        except Exception:
            pass
        self.fit_info_ = DeepSurvFitInfo(best_epoch=int(best_epoch), best_val_loss=float(best_val))

    def predict_risk(self, x: np.ndarray) -> np.ndarray:
        model = self._check_ready()
        x = np.asarray(x, dtype=np.float32)
        pred = model.predict(x, batch_size=self.batch_size, numpy=True, eval_=True)
        return np.asarray(pred, dtype=float).reshape(-1)

    def predict_survival_df(self, x: np.ndarray):
        model = self._check_ready()
        x = np.asarray(x, dtype=np.float32)
        return model.predict_surv_df(x, batch_size=self.batch_size, eval_=True)

    def predict_survival(self, x: np.ndarray, times: np.ndarray) -> np.ndarray:
        surv_df = self.predict_survival_df(x)
        t_src = surv_df.index.to_numpy(dtype=float)
        s_src = surv_df.to_numpy(dtype=float).T  # [n, T_src]
        t_tgt = np.asarray(times, dtype=float).reshape(-1)

        s_out = np.empty((s_src.shape[0], t_tgt.size), dtype=float)
        for i in range(s_src.shape[0]):
            s_out[i] = np.interp(t_tgt, t_src, s_src[i], left=s_src[i, 0], right=s_src[i, -1])
        return np.clip(s_out, 1e-12, 1.0)
