from __future__ import annotations

from typing import Any

import numpy as np
import torch

from fuzzy_aft.losses import censored_interval_score, censored_interval_score_from_bounds, fuzzy_gamma_cut_bounds, gaussian_interval_bounds


def _bounds_np(
    mu: np.ndarray,
    tau: np.ndarray,
    alpha: float,
    eta: float,
    interval_variant: str,
    gamma: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mu_t = torch.from_numpy(mu.astype(np.float64))
    tau_t = torch.from_numpy(tau.astype(np.float64))
    variant = str(interval_variant).strip().lower()
    if variant == "gaussian_surrogate":
        lower, upper = gaussian_interval_bounds(mu=mu_t, tau=tau_t, alpha=alpha, eta=eta)
    elif variant in {"fuzzy_gammacut", "gammacut", "fuzzy_alphacut", "alphacut"}:
        lower, upper = fuzzy_gamma_cut_bounds(mu=mu_t, tau=tau_t, gamma=float(alpha if gamma is None else gamma))
    else:
        raise ValueError(f"Unknown interval variant '{interval_variant}'.")
    return lower.numpy(), upper.numpy()


def coverage_uncensored(s: np.ndarray, delta: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    mask = delta.astype(bool)
    if mask.sum() == 0:
        return float("nan")
    inside = (s[mask] >= lower[mask]) & (s[mask] <= upper[mask])
    return float(np.mean(inside))


def mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(upper - lower))


def lower_bound_violation_rate(s: np.ndarray, delta: np.ndarray, lower: np.ndarray) -> float:
    # For right-censored samples, lower bound above censoring time is a violation.
    mask = ~delta.astype(bool)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(lower[mask] > s[mask]))


def censored_interval_score_metric(
    s: np.ndarray,
    delta: np.ndarray,
    mu: np.ndarray,
    tau: np.ndarray,
    *,
    alpha: float,
    eta: float,
) -> float:
    s_t = torch.from_numpy(s.astype(np.float64))
    d_t = torch.from_numpy(delta.astype(np.float64))
    mu_t = torch.from_numpy(mu.astype(np.float64))
    tau_t = torch.from_numpy(tau.astype(np.float64))
    val = censored_interval_score(s=s_t, delta=d_t, mu=mu_t, tau=tau_t, alpha=alpha, eta=eta)
    return float(val.item())


def interval_metrics(
    *,
    s_test: np.ndarray,
    delta_test: np.ndarray,
    mu_test: np.ndarray,
    tau_test: np.ndarray,
    alpha: float,
    eta: float,
    interval_variant: str = "gaussian_surrogate",
    gamma: float | None = None,
) -> dict[str, Any]:
    lower, upper = _bounds_np(
        mu=mu_test,
        tau=tau_test,
        alpha=alpha,
        eta=eta,
        interval_variant=interval_variant,
        gamma=gamma,
    )
    return interval_metrics_from_bounds(
        s_test=s_test,
        delta_test=delta_test,
        lower_test=lower,
        upper_test=upper,
        alpha=alpha,
    )


def interval_metrics_from_bounds(
    *,
    s_test: np.ndarray,
    delta_test: np.ndarray,
    lower_test: np.ndarray,
    upper_test: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    s_t = torch.from_numpy(s_test.astype(np.float64))
    d_t = torch.from_numpy(delta_test.astype(np.float64))
    l_t = torch.from_numpy(lower_test.astype(np.float64))
    u_t = torch.from_numpy(upper_test.astype(np.float64))
    cis = float(censored_interval_score_from_bounds(s=s_t, delta=d_t, lower=l_t, upper=u_t, alpha=alpha).item())
    return {
        "coverage_uncensored": coverage_uncensored(
            s=s_test,
            delta=delta_test,
            lower=lower_test,
            upper=upper_test,
        ),
        "mean_interval_width": mean_interval_width(lower=lower_test, upper=upper_test),
        "censored_interval_score": cis,
        "lower_bound_violation_rate": lower_bound_violation_rate(
            s=s_test,
            delta=delta_test,
            lower=lower_test,
        ),
    }
