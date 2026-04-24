"""Supervised Contrastive Loss (Khosla et al., 2020)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from embedkit.improvement.losses.base import BaseLoss


class SupConLoss(BaseLoss):
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
        z = F.normalize(torch.cat([z_i, z_j], dim=0), dim=1)  # (2N, d)
        sim = torch.mm(z, z.T) / self.temperature  # (2N, 2N)

        # mask: self-similarity
        eye_mask = torch.eye(2 * n, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(eye_mask, float("-inf"))

        if labels is not None:
            lbl = torch.cat([labels, labels], dim=0)
            pos_mask = (lbl.unsqueeze(0) == lbl.unsqueeze(1)).float()
            pos_mask.fill_diagonal_(0.0)
        else:
            # fall back to NT-Xent positives
            pos_mask = torch.zeros(2 * n, 2 * n, device=z.device)
            idx = torch.arange(n, device=z.device)
            pos_mask[idx, idx + n] = 1.0
            pos_mask[idx + n, idx] = 1.0

        log_prob = F.log_softmax(sim, dim=1)
        # mean over positives per anchor
        n_pos = pos_mask.sum(dim=1).clamp(min=1)
        loss = -(pos_mask * log_prob).sum(dim=1) / n_pos
        return loss.mean()
