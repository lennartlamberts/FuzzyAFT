from __future__ import annotations

import math
from typing import Any

import numpy as np
from lifelines.utils import concordance_index
from scipy.special import ndtr
from sksurv.metrics import integrated_brier_score


def _to_structured_y(t: np.ndarray, delta: np.ndarray) -> np.ndarray:
    y = np.zeros(t.shape[0], dtype=[("event", "?"), ("time", "<f8")])
    y["event"] = delta.astype(bool)
    y["time"] = t.astype(float)
    return y


def cindex_from_mu(s: np.ndarray, delta: np.ndarray, mu: np.ndarray) -> float:
    # In AFT on log-time, larger mu implies longer survival.
    s = np.asarray(s, dtype=float).reshape(-1)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    delta = np.asarray(delta).reshape(-1)
    ok = np.isfinite(s) & np.isfinite(mu) & np.isfinite(delta.astype(float))
    if ok.sum() < 2:
        return float("nan")
    return float(concordance_index(event_times=s[ok], predicted_scores=mu[ok], event_observed=delta[ok]))


def ibs_from_gaussian_surrogate(
    s_train: np.ndarray,
    delta_train: np.ndarray,
    s_test: np.ndarray,
    delta_test: np.ndarray,
    mu_test: np.ndarray,
    tau_test: np.ndarray,
    *,
    eta: float = 1.0,
    n_time_grid: int = 100,
    eps: float = 1e-12,
) -> float:
    if eta <= 0.0:
        raise ValueError("eta must be > 0.")

    t_train = np.exp(s_train)
    t_test = np.exp(s_test)

    observed_train = t_train[delta_train.astype(bool)]
    if observed_train.size < 5:
        return float("nan")

    t_min = float(max(np.quantile(observed_train, 0.05), eps))
    t_max = float(np.quantile(observed_train, 0.95))
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        return float("nan")

    times = np.linspace(t_min, t_max, n_time_grid)
    sigma = np.clip(tau_test / math.sqrt(eta), eps, None)

    # S(t|x) = 1 - Phi((log t - mu) / sigma)
    log_times = np.log(times)[None, :]
    z = (log_times - mu_test[:, None]) / sigma[:, None]
    surv = np.clip(1.0 - ndtr(z), eps, 1.0)

    y_train = _to_structured_y(t=t_train, delta=delta_train)
    y_test = _to_structured_y(t=t_test, delta=delta_test)
    try:
        return float(integrated_brier_score(y_train, y_test, surv, times))
    except Exception:
        return float("nan")


def test_cnll_from_arrays(
    s: np.ndarray,
    delta: np.ndarray,
    mu: np.ndarray,
    tau: np.ndarray,
    *,
    eta: float = 1.0,
    eps: float = 1e-12,
) -> float:
    if eta <= 0.0:
        raise ValueError("eta must be > 0.")

    sigma = np.clip(tau / math.sqrt(eta), eps, None)
    z = (s - mu) / sigma
    log_pdf = -0.5 * (math.log(2.0 * math.pi) + 2.0 * np.log(sigma) + z * z)
    cdf = np.clip(ndtr(z), eps, 1.0 - eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_surv = np.log1p(-cdf)
    log_surv = np.where(np.isfinite(log_surv), log_surv, math.log(eps))
    ll = np.where(delta > 0.5, log_pdf, log_surv)
    return float(-np.mean(ll))


def survival_metrics(
    *,
    s_train: np.ndarray,
    delta_train: np.ndarray,
    s_test: np.ndarray,
    delta_test: np.ndarray,
    mu_test: np.ndarray,
    tau_test: np.ndarray,
    eta: float,
) -> dict[str, Any]:
    return {
        "cindex": cindex_from_mu(s=s_test, delta=delta_test, mu=mu_test),
        "ibs": ibs_from_gaussian_surrogate(
            s_train=s_train,
            delta_train=delta_train,
            s_test=s_test,
            delta_test=delta_test,
            mu_test=mu_test,
            tau_test=tau_test,
            eta=eta,
        ),
        "test_cNLL": test_cnll_from_arrays(
            s=s_test,
            delta=delta_test,
            mu=mu_test,
            tau=tau_test,
            eta=eta,
        ),
    }
