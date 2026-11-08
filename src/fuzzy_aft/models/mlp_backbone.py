from __future__ import annotations

import torch.nn as nn


class MLPBackbone(nn.Module):
    """Reusable MLP encoder for tabular survival models."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...] = (128, 128, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0.")
        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must contain at least one layer.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")

        layers: list[nn.Module] = []
        prev_dim = int(input_dim)
        for h in hidden_sizes:
            if int(h) <= 0:
                raise ValueError("All hidden layer sizes must be > 0.")
            layers.append(nn.Linear(prev_dim, int(h)))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev_dim = int(h)

        self.net = nn.Sequential(*layers)
        self.output_dim = int(prev_dim)

    def forward(self, x):
        return self.net(x)
