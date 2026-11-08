from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp_backbone import MLPBackbone


class FuzzyAFTMLP(nn.Module):
    """
    Fuzzy AFT model trained with the gamma-cut interval objective.

    Output contract:
      - mu  : predictive location, shape (batch,)
      - tau : predictive imprecision, strictly positive, shape (batch,)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...] = (128, 128, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = MLPBackbone(input_dim=input_dim, hidden_sizes=hidden_sizes, dropout=dropout)
        self.mu_head = nn.Linear(self.encoder.output_dim, 1)
        self.g_head = nn.Linear(self.encoder.output_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 2:
            raise ValueError(f"x must have shape (batch, features), got {tuple(x.shape)}.")

        h = self.encoder(x)
        mu = self.mu_head(h).squeeze(-1)
        g = self.g_head(h).squeeze(-1)
        tau = F.softplus(g)

        return {"mu": mu, "tau": tau}
