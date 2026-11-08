from __future__ import annotations

import math

import torch
import torch.nn as nn


def gaussian_interval_bounds(
    mu: torch.Tensor,
    tau: torch.Tensor,
    alpha: float = 0.1,
    eta: float = 1.0,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Central prediction interval from Gaussian surrogate induced by power normalization.
    sigma = tau / sqrt(eta), interval = [mu - z*sigma, mu + z*sigma],
    with z = Phi^{-1}(1 - alpha/2).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    if eta <= 0.0:
        raise ValueError("eta must be > 0.")

    sigma = torch.clamp(tau / math.sqrt(eta), min=eps)
    q = torch.tensor(1.0 - alpha / 2.0, device=mu.device, dtype=mu.dtype)
    z = torch.distributions.Normal(0.0, 1.0).icdf(q)

    lower = mu - z * sigma
    upper = mu + z * sigma
    return lower, upper


def fuzzy_gamma_cut_bounds(
    mu: torch.Tensor,
    tau: torch.Tensor,
    gamma: float = 0.1,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    True gamma-cut interval for Gaussian-shaped possibility distribution:
      pi(s|x) = exp(-(s-mu)^2 / (2*tau^2))
      A_gamma = [mu - tau*sqrt(2*log(1/gamma)), mu + tau*sqrt(2*log(1/gamma))]
    """
    if not (0.0 < gamma < 1.0):
        raise ValueError("gamma must be in (0, 1).")
    gamma_safe = float(max(min(gamma, 1.0 - eps), eps))
    tau_safe = torch.clamp(tau, min=eps)
    radius = tau_safe * math.sqrt(2.0 * math.log(1.0 / gamma_safe))
    lower = mu - radius
    upper = mu + radius
    return lower, upper


def censored_interval_score(
    s: torch.Tensor,
    delta: torch.Tensor,
    mu: torch.Tensor,
    tau: torch.Tensor,
    alpha: float = 0.1,
    eta: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Interval score for right-censored survival targets on log-time.

    Event (delta=1): standard interval score
      (u-l) + 2/alpha * (l-s)_+ + 2/alpha * (s-u)_+

    Censored (delta=0): width + lower-bound violation penalty
      (u-l) + 2/alpha * (l-s)_+
    """
    lower, upper = gaussian_interval_bounds(mu=mu, tau=tau, alpha=alpha, eta=eta, eps=eps)
    return censored_interval_score_from_bounds(s=s, delta=delta, lower=lower, upper=upper, alpha=alpha)


def censored_interval_score_alphacut(
    s: torch.Tensor,
    delta: torch.Tensor,
    mu: torch.Tensor,
    tau: torch.Tensor,
    alpha: float = 0.1,
    gamma: float = 0.1,
    eps: float = 1e-12,
) -> torch.Tensor:
    lower, upper = fuzzy_gamma_cut_bounds(mu=mu, tau=tau, gamma=gamma, eps=eps)
    return censored_interval_score_from_bounds(s=s, delta=delta, lower=lower, upper=upper, alpha=alpha)


def censored_interval_score_from_bounds(
    s: torch.Tensor,
    delta: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    alpha: float = 0.1,
) -> torch.Tensor:
    """
    Interval score for right-censored targets given explicit interval bounds [lower, upper].
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    width = upper - lower

    event_penalty = (2.0 / alpha) * (torch.relu(lower - s) + torch.relu(s - upper))
    cens_penalty = (2.0 / alpha) * torch.relu(lower - s)

    score = delta * (width + event_penalty) + (1.0 - delta) * (width + cens_penalty)
    return score.mean()


class IntervalScoreLoss(nn.Module):
    def __init__(self, alpha: float = 0.1, eta: float = 1.0, eps: float = 1e-12) -> None:
        super().__init__()
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        if eta <= 0.0:
            raise ValueError("eta must be > 0.")
        self.alpha = alpha
        self.eta = eta
        self.eps = eps

    def forward(self, mu: torch.Tensor, tau: torch.Tensor, s: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return censored_interval_score(
            s=s,
            delta=delta,
            mu=mu,
            tau=tau,
            alpha=self.alpha,
            eta=self.eta,
            eps=self.eps,
        )


class FuzzyCutIntervalScoreLoss(nn.Module):
    def __init__(self, alpha: float = 0.1, gamma: float = 0.1, eps: float = 1e-12) -> None:
        super().__init__()
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        if not (0.0 < gamma < 1.0):
            raise ValueError("gamma must be in (0, 1).")
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, mu: torch.Tensor, tau: torch.Tensor, s: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return censored_interval_score_alphacut(
            s=s,
            delta=delta,
            mu=mu,
            tau=tau,
            alpha=self.alpha,
            gamma=self.gamma,
            eps=self.eps,
        )
