"""Alignment + Uniformity loss (Wang & Isola, 2020)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from embedkit.improvement.losses.base import BaseLoss


class AlignUniformLoss(BaseLoss):
    def __init__(self, alpha: float = 2.0, t: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.t = t

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # alignment: mean ||z_i - z_j||^alpha for positive pairs
        l_align = (z_i - z_j).norm(dim=1).pow(self.alpha).mean()

        # uniformity: log-average pairwise Gaussian potential on combined set
        z = torch.cat([z_i, z_j], dim=0)
        sq_diffs = torch.pdist(z, p=2).pow(2)
        l_uniform = sq_diffs.mul(-self.t).exp().mean().log()

        return l_align + l_uniform
