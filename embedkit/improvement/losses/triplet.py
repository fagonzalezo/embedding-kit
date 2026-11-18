"""Triplet loss with hard/semi-hard/easy mining."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from embedkit.improvement.losses.base import BaseLoss


class TripletLoss(BaseLoss):
    uses_labels = True

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


_EPS = 1e-12


def _pairwise_dist(z: torch.Tensor, distance: str) -> torch.Tensor:
    if distance == "cosine":
        sim = torch.mm(z, z.T)
        return 1.0 - sim
    # Euclidean: sqrt(d2 + eps) has a finite gradient at d2 = 0 (duplicate embeddings), unlike
    # sqrt(d2) and torch.cdist. The diagonal is zeroed before the square root; it is never
    # selected by the masks below.
    sq = (z ** 2).sum(dim=1, keepdim=True)
    d2 = (sq + sq.T - 2 * torch.mm(z, z.T)).clamp(min=0)
    eye = torch.eye(z.shape[0], device=z.device, dtype=torch.bool)
    d2 = d2.masked_fill(eye, 0.0)
    return (d2 + _EPS).sqrt()


def _triplet_loss(
    dist: torch.Tensor, labels: torch.Tensor, margin: float, mining: str
) -> torch.Tensor:
    """Batch-masked triplet loss; see the class docstring for the mining modes."""
    n = dist.shape[0]
    inf = float("inf")
    same = labels.unsqueeze(0) == labels.unsqueeze(1)  # (n, n)
    eye = torch.eye(n, device=dist.device, dtype=torch.bool)
    pos_mask = same & ~eye
    neg_mask = ~same
    # Anchors with no positive or no negative are skipped.
    valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)

    if mining == "hard":
        d_pos = dist.masked_fill(~pos_mask, -inf).max(dim=1).values
        d_neg = dist.masked_fill(~neg_mask, inf).min(dim=1).values
    elif mining == "semi-hard":
        d_pos = (dist * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
        neg_min = dist.masked_fill(~neg_mask, inf).min(dim=1).values
        harder = neg_mask & (dist > d_pos.detach().unsqueeze(1))
        semi_min = dist.masked_fill(~harder, inf).min(dim=1).values
        d_neg = torch.where(torch.isinf(semi_min), neg_min, semi_min)
    elif mining == "easy":
        d_pos = (dist * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
        d_neg = (dist * neg_mask).sum(dim=1) / neg_mask.sum(dim=1).clamp(min=1)
    else:
        raise ValueError(f"Unknown mining mode {mining!r}; use 'hard', 'semi-hard' or 'easy'")

    # Replace the infinities of skipped anchors before they enter arithmetic.
    zero = torch.zeros_like(d_pos)
    d_pos = torch.where(valid, d_pos, zero)
    d_neg = torch.where(valid, d_neg, zero)
    per_anchor = F.relu(d_pos - d_neg + margin)

    n_valid = valid.sum()
    if n_valid == 0:
        return dist.sum() * 0.0
    return (per_anchor * valid).sum() / n_valid
