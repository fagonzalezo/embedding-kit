"""Supervised Contrastive Loss (Khosla et al., 2020)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from embedkit.improvement.losses.base import BaseLoss


class SupConLoss(BaseLoss):
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

        if labels is not None:
            lbl = torch.cat([labels, labels], dim=0)
            pos_mask = (lbl.unsqueeze(0) == lbl.unsqueeze(1)).float()
            pos_mask.fill_diagonal_(0.0)
        else:
            pos_mask = torch.zeros(2 * n, 2 * n, device=z.device)
            idx = torch.arange(n, device=z.device)
            pos_mask[idx, idx + n] = 1.0
            pos_mask[idx + n, idx] = 1.0

        if self.chunk_size is None or self.chunk_size >= 2 * n:
            return self._loss_full(z, n, pos_mask)
        return self._loss_chunked(z, n, pos_mask)

    def _loss_full(self, z, n, pos_mask):
        two_n = 2 * n
        sim = torch.mm(z, z.T) / self.temperature
        eye_mask = torch.eye(two_n, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(eye_mask, float("-inf"))
        log_prob = F.log_softmax(sim, dim=1)
        n_pos = pos_mask.sum(dim=1).clamp(min=1)
        loss = -(pos_mask * log_prob).sum(dim=1) / n_pos
        return loss.mean()

    def _loss_chunked(self, z, n, pos_mask):
        """Row-block SupCon to avoid materializing full (2N)×(2N) sim."""
        two_n = 2 * n
        cs = self.chunk_size
        losses = []
        for start in range(0, two_n, cs):
            end = min(start + cs, two_n)
            rows = z[start:end]
            logits = torch.mm(rows, z.T) / self.temperature  # (cs, 2N)
            for i in range(end - start):
                logits[i, start + i] = float("-inf")
            log_prob = F.log_softmax(logits, dim=1)  # (cs, 2N)
            pm_chunk = pos_mask[start:end]  # (cs, 2N)
            n_pos = pm_chunk.sum(dim=1).clamp(min=1)
            losses.append((-(pm_chunk * log_prob).sum(dim=1) / n_pos).mean())
        return torch.stack(losses).mean()
