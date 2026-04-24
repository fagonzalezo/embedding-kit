"""Abstract base loss."""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn


class BaseLoss(nn.Module):
    @abstractmethod
    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...
