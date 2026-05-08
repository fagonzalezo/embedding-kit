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

        l_align = (z_i - z_j).norm(dim=1).pow(self.alpha).mean()

        z = torch.cat([z_i, z_j], dim=0)
        l_uniform = _chunked_uniformity(z, self.t)

        return l_align + l_uniform


def _chunked_uniformity(z: torch.Tensor, t: float, chunk: int = 1024) -> torch.Tensor:
    """Compute log-average pairwise Gaussian potential without torch.pdist.

    Processes rows in chunks to bound peak memory to O(chunk × 2N × d).
    Avoids CUDA's torch.pdist size limit and the N×N×D broadcast.
    """
    n = z.shape[0]
    total_kernel = torch.tensor(0.0, device=z.device)
    count = 0
    for i in range(0, n, chunk):
        zi = z[i: i + chunk]  # (ci, d)
        # only upper-triangle pairs to avoid double-counting
        for j in range(i, n, chunk):
            zj = z[j: j + chunk]  # (cj, d)
            sq = torch.cdist(zi, zj, p=2).pow(2)  # (ci, cj)
            if i == j:
                # keep strict upper triangle within the diagonal block
                mask = torch.triu(torch.ones_like(sq, dtype=torch.bool), diagonal=1)
                sq = sq[mask]
            else:
                sq = sq.reshape(-1)
            total_kernel = total_kernel + sq.mul(-t).exp().sum()
            count += sq.numel()
    return (total_kernel / max(count, 1)).log()
