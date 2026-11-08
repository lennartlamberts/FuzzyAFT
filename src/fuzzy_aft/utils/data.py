# src/utils/data.py

from __future__ import annotations

from typing import Any, Dict, Tuple
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _make_loader(ds: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=False,
        num_workers=0,
        generator=generator,
    )


# =========================================================
# 1) Raw dataset loaders
# =========================================================

def load_metabric_pycox() -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Returns:
        X      : (n, d) float
        t      : (n,) positive durations (original time scale)
        delta  : (n,) {0,1} event indicator (1=event, 0=censored)
        meta   : dict with feature_names
    """
    from pycox.datasets import metabric

    df = metabric.read_df()  # columns include 'duration', 'event'
    t = df["duration"].to_numpy(dtype=float)
    delta = df["event"].to_numpy(dtype=int)

    X_df = df.drop(columns=["duration", "event"])
    X = X_df.to_numpy(dtype=float)

    meta = {"feature_names": list(X_df.columns)}
    return X, t, delta, meta


def load_whas500_sksurv() -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Returns:
        X      : (n, d) float (one-hot encoded if needed)
        t      : (n,) positive durations
        delta  : (n,) {0,1} event indicator (1=event, 0=censored)
        meta   : dict with feature_names
    """
    from sksurv.datasets import load_whas500

    X_df, y = load_whas500()
    t = y["lenfol"].astype(float)
    delta = y["fstat"].astype(int)

    # Deterministic one-hot encoding for any categorical columns
    if isinstance(X_df, pd.DataFrame):
        X_df = pd.get_dummies(X_df, drop_first=True)
        X = X_df.to_numpy(dtype=float)
        meta = {"feature_names": list(X_df.columns)}
    else:
        X = np.asarray(X_df, dtype=float)
        meta = {"feature_names": None}

    return X, t, delta, meta


# =========================================================
# 2) Torch dataset
# =========================================================

class SurvivalDataset(Dataset):
    """
    PyTorch Dataset for survival AFT on log-time.

    Each item is:
        x      : float tensor (d,)
        s      : float tensor ()  log duration
        delta  : float tensor ()  1=event, 0=censored
    """

    def __init__(self, X: np.ndarray, s: np.ndarray, delta: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.s = torch.tensor(s, dtype=torch.float32)
        self.delta = torch.tensor(delta, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.s[idx], self.delta[idx]


# =========================================================
# 3) Splits + preprocessing
# =========================================================

def make_splits(
    X: np.ndarray,
    t: np.ndarray,
    delta: np.ndarray,
    seed: int,
    split=(0.70, 0.15, 0.15),
    stratify_by_censoring: bool = True,
):
    tr, va, te = split
    if not np.isclose(tr + va + te, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    strat = delta if stratify_by_censoring else None

    X_tr, X_tmp, t_tr, t_tmp, d_tr, d_tmp = train_test_split(
        X, t, delta,
        test_size=(1.0 - tr),
        random_state=seed,
        stratify=strat
    )

    rel_test = te / (va + te)
    strat_tmp = d_tmp if stratify_by_censoring else None

    X_va, X_te, t_va, t_te, d_va, d_te = train_test_split(
        X_tmp, t_tmp, d_tmp,
        test_size=rel_test,
        random_state=seed,
        stratify=strat_tmp
    )

    return (X_tr, t_tr, d_tr), (X_va, t_va, d_va), (X_te, t_te, d_te)


def preprocess_aft_logtime(
    train,
    val,
    test,
    standardize: bool = True,
    log_time: bool = True,
    log_time_eps: float = 1e-8,
):
    (X_tr, t_tr, d_tr) = train
    (X_va, t_va, d_va) = val
    (X_te, t_te, d_te) = test

    # Impute non-finite features (NaN/inf) using train means to keep downstream
    # models/StandardScaler numerically well-defined.
    X_tr = np.asarray(X_tr, dtype=float)
    X_va = np.asarray(X_va, dtype=float)
    X_te = np.asarray(X_te, dtype=float)
    bad_tr = ~np.isfinite(X_tr)
    bad_va = ~np.isfinite(X_va)
    bad_te = ~np.isfinite(X_te)
    n_bad = int(bad_tr.sum() + bad_va.sum() + bad_te.sum())
    if n_bad > 0:
        train_vals = np.where(np.isfinite(X_tr), X_tr, np.nan)
        col_means = np.nanmean(train_vals, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        X_tr = np.where(np.isfinite(X_tr), X_tr, col_means)
        X_va = np.where(np.isfinite(X_va), X_va, col_means)
        X_te = np.where(np.isfinite(X_te), X_te, col_means)
        warnings.warn(
            "Found non-finite feature values; imputing with train column means. "
            f"Counts: train={int(bad_tr.sum())}, val={int(bad_va.sum())}, test={int(bad_te.sum())}.",
            RuntimeWarning,
            stacklevel=2,
        )

    # log-time transform
    if log_time:
        if log_time_eps <= 0:
            raise ValueError("log_time_eps must be > 0.")

        n_bad_tr = int((t_tr <= 0).sum())
        n_bad_va = int((t_va <= 0).sum())
        n_bad_te = int((t_te <= 0).sum())
        n_bad_total = n_bad_tr + n_bad_va + n_bad_te

        if n_bad_total > 0:
            warnings.warn(
                "Found non-positive durations for log transform; "
                f"clipping to {log_time_eps:g}. "
                f"Counts: train={n_bad_tr}, val={n_bad_va}, test={n_bad_te}.",
                RuntimeWarning,
                stacklevel=2,
            )

        t_tr_safe = np.clip(t_tr, a_min=log_time_eps, a_max=None)
        t_va_safe = np.clip(t_va, a_min=log_time_eps, a_max=None)
        t_te_safe = np.clip(t_te, a_min=log_time_eps, a_max=None)

        s_tr = np.log(t_tr_safe)
        s_va = np.log(t_va_safe)
        s_te = np.log(t_te_safe)
    else:
        s_tr, s_va, s_te = t_tr, t_va, t_te

    # standardize features (fit on train only)
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        X_te = scaler.transform(X_te)

    return (X_tr, s_tr, d_tr), (X_va, s_va, d_va), (X_te, s_te, d_te), scaler


# =========================================================
# 4) Public API: get_dataloaders(cfg)
# =========================================================

def get_dataloaders(cfg: Dict[str, Any]):
    """
    Main entry point used by experiment scripts.

    Expects cfg keys (as in your YAML):
      cfg["dataset"]["name"] in {"metabric","whas500"}
      cfg["dataset"]["split"] with train/val/test
      cfg["dataset"]["standardize"], cfg["dataset"]["log_time"], cfg["dataset"]["stratify_by_censoring"]
      cfg["training"]["batch_size"]
      cfg["experiment"]["seeds"] (list) OR cfg["experiment"]["seed"] (int)

    Returns:
      train_loader, val_loader, test_loader, meta
    """

    ds_name = cfg["dataset"]["name"].lower()

    # pick a seed (support either "seed" or "seeds")
    if "seed" in cfg.get("experiment", {}):
        seed = int(cfg["experiment"]["seed"])
    else:
        seed = int(cfg["experiment"]["seeds"][0])

    # load raw data
    if ds_name == "metabric":
        X, t, delta, meta = load_metabric_pycox()
    elif ds_name == "whas500":
        X, t, delta, meta = load_whas500_sksurv()
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    # split
    split_cfg = cfg["dataset"]["split"]
    train, val, test = make_splits(
        X, t, delta,
        seed=seed,
        split=(split_cfg["train"], split_cfg["val"], split_cfg["test"]),
        stratify_by_censoring=bool(cfg["dataset"].get("stratify_by_censoring", True)),
    )

    # preprocess
    train_p, val_p, test_p, scaler = preprocess_aft_logtime(
        train, val, test,
        standardize=bool(cfg["dataset"].get("standardize", True)),
        log_time=bool(cfg["dataset"].get("log_time", True)),
        log_time_eps=float(cfg["dataset"].get("log_time_eps", 1e-8)),
    )

    # torch datasets + loaders
    bs = int(cfg["training"]["batch_size"])
    train_ds = SurvivalDataset(*train_p)
    val_ds = SurvivalDataset(*val_p)
    test_ds = SurvivalDataset(*test_p)

    train_loader = _make_loader(train_ds, batch_size=bs, shuffle=True, seed=seed)
    val_loader = _make_loader(val_ds, batch_size=bs, shuffle=False, seed=seed + 1)
    test_loader = _make_loader(test_ds, batch_size=bs, shuffle=False, seed=seed + 2)

    meta_out = {
        **meta,
        "seed": seed,
        "scaler": scaler,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
    }

    return train_loader, val_loader, test_loader, meta_out
