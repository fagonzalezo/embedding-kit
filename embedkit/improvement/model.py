"""EmbeddingRefiner: MLP projector head."""

from __future__ import annotations

import torch
import torch.nn as nn


class EmbeddingRefiner(nn.Module):
    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        normalize_output: bool = True,
    ):
        super().__init__()
        self.normalize_output = normalize_output

        layers: list[nn.Module] = []
        in_d = input_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_d, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, target_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        if self.normalize_output:
            z = nn.functional.normalize(z, dim=1)
        return z
