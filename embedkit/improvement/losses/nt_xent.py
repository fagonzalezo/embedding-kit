"""NT-Xent (symmetric InfoNCE) contrastive loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from embedkit.improvement.losses.base import BaseLoss


class NTXentLoss(BaseLoss):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)  # (2N, d)
        z = F.normalize(z, dim=1)

        sim = torch.mm(z, z.T) / self.temperature  # (2N, 2N)

        # mask out self-similarity
        mask = torch.eye(2 * n, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float("-inf"))

        # positives are pairs (i, i+N) and (i+N, i)
        pos_idx = torch.arange(n, device=z.device)
        targets_i = pos_idx + n
        targets_j = pos_idx

        loss_i = F.cross_entropy(sim[:n], targets_i)
        loss_j = F.cross_entropy(sim[n:], targets_j)
        return (loss_i + loss_j) / 2
