"""CombinedLoss: weighted sum of multiple losses."""

from __future__ import annotations

import torch

from embedkit.improvement.losses.base import BaseLoss


class CombinedLoss(BaseLoss):
    def __init__(self, losses: list[tuple[BaseLoss, float]]):
        super().__init__()
        self.loss_fns = torch.nn.ModuleList([l for l, _ in losses])
        self.weights = [w for _, w in losses]

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        total = torch.tensor(0.0, device=z_i.device)
        for loss_fn, w in zip(self.loss_fns, self.weights):
            try:
                total = total + w * loss_fn(z_i, z_j, labels)
            except (TypeError, ValueError):
                total = total + w * loss_fn(z_i, z_j)
        return total
