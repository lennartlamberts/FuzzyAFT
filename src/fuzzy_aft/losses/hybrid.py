from __future__ import annotations

import torch
import torch.nn as nn

from .cnll import CensoredNLLLoss
from .interval_score import FuzzyCutIntervalScoreLoss, IntervalScoreLoss


class HybridLoss(nn.Module):
    """
    L = L_cNLL + lambda_interval * L_IS
    """

    def __init__(
        self,
        lambda_interval: float = 1.0,
        eta: float = 1.0,
        alpha: float = 0.1,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if lambda_interval < 0.0:
            raise ValueError("lambda_interval must be >= 0.")
        self.lambda_interval = lambda_interval
        self.cnll = CensoredNLLLoss(eta=eta, eps=eps)
        self.interval = IntervalScoreLoss(alpha=alpha, eta=eta, eps=eps)

    def forward(
        self,
        mu: torch.Tensor,
        tau: torch.Tensor,
        s: torch.Tensor,
        delta: torch.Tensor,
        return_components: bool = False,
    ):
        loss_cnll = self.cnll(mu=mu, tau=tau, s=s, delta=delta)
        loss_is = self.interval(mu=mu, tau=tau, s=s, delta=delta)
        total = loss_cnll + self.lambda_interval * loss_is

        if return_components:
            return {
                "loss": total,
                "cNLL": loss_cnll,
                "interval_score": loss_is,
            }
        return total


class HybridFuzzyCutLoss(nn.Module):
    """
    L = L_cNLL + lambda_interval * L_IS(gamma-cut)
    """

    def __init__(
        self,
        lambda_interval: float = 1.0,
        eta: float = 1.0,
        alpha: float = 0.1,
        gamma: float = 0.1,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if lambda_interval < 0.0:
            raise ValueError("lambda_interval must be >= 0.")
        self.lambda_interval = lambda_interval
        self.cnll = CensoredNLLLoss(eta=eta, eps=eps)
        self.interval = FuzzyCutIntervalScoreLoss(alpha=alpha, gamma=gamma, eps=eps)

    def forward(
        self,
        mu: torch.Tensor,
        tau: torch.Tensor,
        s: torch.Tensor,
        delta: torch.Tensor,
        return_components: bool = False,
    ):
        loss_cnll = self.cnll(mu=mu, tau=tau, s=s, delta=delta)
        loss_is = self.interval(mu=mu, tau=tau, s=s, delta=delta)
        total = loss_cnll + self.lambda_interval * loss_is

        if return_components:
            return {
                "loss": total,
                "cNLL": loss_cnll,
                "interval_score": loss_is,
            }
        return total
