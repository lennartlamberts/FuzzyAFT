from __future__ import annotations

import math

import torch
import torch.nn as nn


def _gaussian_log_pdf(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    z = (x - mu) / sigma
    return -0.5 * (math.log(2.0 * math.pi) + 2.0 * torch.log(sigma) + z * z)


def _gaussian_log_survival(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, eps: float) -> torch.Tensor:
    z = (x - mu) / sigma
    cdf = torch.special.ndtr(z)
    surv = torch.clamp(1.0 - cdf, min=eps)
    return torch.log(surv)


def censored_nll(
    s: torch.Tensor,
    delta: torch.Tensor,
    mu: torch.Tensor,
    tau: torch.Tensor,
    eta: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Right-censored NLL on log-time using power-normalized Gaussian surrogate.

    Possibility:
      pi(s|x) = exp(-(s-mu)^2 / (2 * tau^2))
    Power-normalized surrogate:
      p_eta(s|x) ~ Normal(mu, sigma^2), sigma = tau / sqrt(eta)
    """
    if eta <= 0.0:
        raise ValueError("eta must be > 0.")

    sigma = torch.clamp(tau / math.sqrt(eta), min=eps)
    log_pdf = _gaussian_log_pdf(s, mu, sigma)
    log_surv = _gaussian_log_survival(s, mu, sigma, eps=eps)

    ll = delta * log_pdf + (1.0 - delta) * log_surv
    return -ll.mean()


class CensoredNLLLoss(nn.Module):
    def __init__(self, eta: float = 1.0, eps: float = 1e-12) -> None:
        super().__init__()
        if eta <= 0.0:
            raise ValueError("eta must be > 0.")
        self.eta = eta
        self.eps = eps

    def forward(self, mu: torch.Tensor, tau: torch.Tensor, s: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return censored_nll(s=s, delta=delta, mu=mu, tau=tau, eta=self.eta, eps=self.eps)
