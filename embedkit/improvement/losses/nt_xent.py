"""NT-Xent (symmetric InfoNCE) contrastive loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from embedkit.improvement.losses.base import BaseLoss


class NTXentLoss(BaseLoss):
    def __init__(self, temperature: float = 0.07, chunk_size: int | None = None):
        super().__init__()
        self.temperature = temperature
        self.chunk_size = chunk_size

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = z_i.shape[0]
        z = F.normalize(torch.cat([z_i, z_j], dim=0), dim=1)  # (2N, d)

        targets_i = torch.arange(n, device=z.device) + n
        targets_j = torch.arange(n, device=z.device)

        if self.chunk_size is None or self.chunk_size >= 2 * n:
            return self._loss_full(z, n, targets_i, targets_j)
        return self._loss_chunked(z, n, targets_i, targets_j)

    def _loss_full(self, z, n, targets_i, targets_j):
        sim = torch.mm(z, z.T) / self.temperature  # (2N, 2N)
        mask = torch.eye(2 * n, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float("-inf"))
        loss_i = F.cross_entropy(sim[:n], targets_i)
        loss_j = F.cross_entropy(sim[n:], targets_j)
        return (loss_i + loss_j) / 2

    def _loss_chunked(self, z, n, targets_i, targets_j):
        """Memory-efficient row-block computation of NT-Xent."""
        two_n = 2 * n
        cs = self.chunk_size
        losses = []
        for start in range(0, two_n, cs):
            end = min(start + cs, two_n)
            rows = z[start:end]  # (cs, d)
            logits = torch.mm(rows, z.T) / self.temperature  # (cs, 2N)
            for i in range(end - start):
                logits[i, start + i] = float("-inf")
            targets = torch.where(
                torch.arange(end - start, device=z.device) + start < n,
                targets_i[start:end] if start < n else targets_j[start - n:end - n],
                targets_j[max(0, start - n):max(0, end - n)] if start >= n else targets_i[start:end],
            )
            # Simpler: just build targets for this slice
            global_idx = torch.arange(start, end, device=z.device)
            tgt = torch.where(global_idx < n, global_idx + n, global_idx - n)
            losses.append(F.cross_entropy(logits, tgt))
        return torch.stack(losses).mean()
