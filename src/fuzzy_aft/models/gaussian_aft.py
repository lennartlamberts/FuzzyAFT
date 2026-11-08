from __future__ import annotations

import torch
import torch.nn as nn

from .mlp_backbone import MLPBackbone


class GaussianAFTMLP(nn.Module):
    """
    Gaussian AFT baseline on log-time.
    Outputs:
      - mu(x)
      - log_sigma(x)
      - sigma(x) = exp(log_sigma(x))
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...] = (128, 128, 128),
        dropout: float = 0.1,
        sigma_param: str = "log",
        sigma_min: float = 1e-6,
    ) -> None:
        super().__init__()
        if sigma_param not in {"log", "softplus"}:
            raise ValueError("sigma_param must be 'log' or 'softplus'.")
        if sigma_min <= 0.0:
            raise ValueError("sigma_min must be > 0.")

        self.encoder = MLPBackbone(input_dim=input_dim, hidden_sizes=hidden_sizes, dropout=dropout)
        self.mu_head = nn.Linear(self.encoder.output_dim, 1)
        self.sigma_head = nn.Linear(self.encoder.output_dim, 1)
        self.sigma_param = sigma_param
        self.sigma_min = float(sigma_min)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 2:
            raise ValueError(f"x must have shape (batch, features), got {tuple(x.shape)}.")

        h = self.encoder(x)
        mu = self.mu_head(h).squeeze(-1)
        raw_sigma = self.sigma_head(h).squeeze(-1)
        if self.sigma_param == "log":
            log_sigma = raw_sigma
            sigma = torch.exp(log_sigma).clamp_min(self.sigma_min)
        else:
            sigma = torch.nn.functional.softplus(raw_sigma).clamp_min(self.sigma_min)
            log_sigma = torch.log(sigma)
        return {"mu": mu, "log_sigma": log_sigma, "sigma": sigma}
