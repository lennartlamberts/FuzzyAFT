from __future__ import annotations

import math

import torch
import torch.nn as nn


def gaussian_censored_nll(
    s: torch.Tensor,
    delta: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    sigma = torch.clamp(sigma, min=eps)
    z = (s - mu) / sigma
    log_pdf = -0.5 * (math.log(2.0 * math.pi) + 2.0 * torch.log(sigma) + z * z)
    cdf = torch.special.ndtr(z)
    log_surv = torch.log(torch.clamp(1.0 - cdf, min=eps))
    ll = delta * log_pdf + (1.0 - delta) * log_surv
    return -ll.mean()


class GaussianCensoredNLLLoss(nn.Module):
    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor, s: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return gaussian_censored_nll(s=s, delta=delta, mu=mu, sigma=sigma, eps=self.eps)
