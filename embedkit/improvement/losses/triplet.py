"""Triplet loss with hard/semi-hard/easy mining."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from embedkit.improvement.losses.base import BaseLoss


class TripletLoss(BaseLoss):
    def __init__(
        self,
        margin: float = 0.5,
        mining: Literal["easy", "hard", "semi-hard"] = "hard",
        distance: Literal["euclidean", "cosine"] = "euclidean",
    ):
        super().__init__()
        self.margin = margin
        self.mining = mining
        self.distance = distance

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if labels is None:
            raise ValueError("TripletLoss requires labels")
        z = torch.cat([z_i, z_j], dim=0)
        lbl = torch.cat([labels, labels], dim=0)
        if self.distance == "cosine":
            z = F.normalize(z, dim=1)

        dist = _pairwise_dist(z, self.distance)
        return _triplet_loss(dist, lbl, self.margin, self.mining)


def _pairwise_dist(z: torch.Tensor, distance: str) -> torch.Tensor:
    if distance == "cosine":
        sim = torch.mm(z, z.T)
        return 1.0 - sim
    # euclidean squared
    sq = (z ** 2).sum(dim=1, keepdim=True)
    d2 = sq + sq.T - 2 * torch.mm(z, z.T)
    return d2.clamp(min=0).sqrt()


def _triplet_loss(dist: torch.Tensor, labels: torch.Tensor, margin: float, mining: str) -> torch.Tensor:
    n = dist.shape[0]
    same = labels.unsqueeze(0) == labels.unsqueeze(1)  # (n, n)
    diff = ~same
    eye = torch.eye(n, device=dist.device, dtype=torch.bool)
    same_no_diag = same & ~eye

    losses = []
    for i in range(n):
        pos_dists = dist[i][same_no_diag[i]]
        neg_dists = dist[i][diff[i]]
        if pos_dists.numel() == 0 or neg_dists.numel() == 0:
            continue
        d_pos = pos_dists.max() if mining == "hard" else pos_dists.mean()
        if mining == "hard":
            d_neg = neg_dists.min()
        elif mining == "semi-hard":
            semi = neg_dists[neg_dists > d_pos]
            d_neg = semi.min() if semi.numel() > 0 else neg_dists.min()
        else:
            d_neg = neg_dists.mean()
        losses.append(F.relu(d_pos - d_neg + margin))

    if not losses:
        return dist.sum() * 0.0
    return torch.stack(losses).mean()
